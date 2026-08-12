"""Strict read-only Phase-2B V2 recognizer I/O structural verifier.

This module verifies the public structure shared by one V2 recognizer-input
archive and one V2 recognizer-prediction archive.  It does not run a
recognizer, reconstruct source custody, validate archive membership, score a
prediction, or establish runtime/capacity/effect/C1 evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import os
import stat
import sys
from typing import Final, Sequence

from .hashing import stable_hash
from .phase2b_recognizer_input_archive_v2 import (
    ARCHIVE_HEADER_BYTES_V2,
    DecodedRecognizerInputArchiveV2,
    MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
    TrustedRecognizerInputRowV2,
    decode_public_recognizer_input_archive_v2,
)
from .phase2b_recognizer_prediction_archive_v2 import (
    DecodedRecognizerPredictionArchiveV2,
    MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
    PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION,
    PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION,
    PublicPredictionRunContextV2,
    PublicRecognizerPredictionRecordV2,
    RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
    RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
    decode_public_recognizer_prediction_archive_v2,
)
from .phase2b_trusted_wire_batch_v2 import TRUSTED_WIRE_BATCH_V2_POLICY_ID
from .phase2b_trusted_wire_v1 import NON_AUTHORITATIVE_CLAIM_LEVEL


STRICT_RECOGNIZER_CLI_V2_COMMAND: Final = "phase2b-verify-v2-structure"
STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-strict-recognizer-structural-receipt/2"
)
STRICT_RECOGNIZER_CLI_V2_GENERIC_REJECTION_REASON: Final = (
    "strict_v2_structural_verification_failed"
)

_INPUT_ROWS_DOMAIN_V2: Final = b"HEGEL/PHASE2B/PREDICTION_INPUT_ROWS/V2\x00"
_INPUT_ROWS_ROOT_PREFIX_V2: Final = "phase2b_prediction_input_rows_v2_"
_RECEIPT_ID_PREFIX_V2: Final = "phase2b_strict_recognizer_receipt_v2_"
_RECEIPT_ISSUE_TOKEN_V2: Final = object()
_READ_CHUNK_BYTES_V2: Final = 1 << 20
_MAXIMUM_PATH_BYTES_V2: Final = 4096
_EXACT_CASE_COUNT_V2: Final = 960

_TRUE_RECEIPT_CLAIMS_V2: Final = (
    "structural_input_archive_verified",
    "structural_prediction_archive_verified",
    "cross_archive_context_binding_verified",
    "ordered_row_identity_verified",
    "seven_input_root_columns_positionally_verified",
)
_FALSE_RECEIPT_CLAIMS_V2: Final = (
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
_RECEIPT_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "schema_version",
    "policy_id",
    "claim_level",
    "receipt_id",
    "input_archive_id",
    "input_archive_sha256",
    "input_archive_version",
    "input_archive_policy_id",
    "prediction_archive_id",
    "prediction_archive_sha256",
    "prediction_archive_version",
    "prediction_archive_policy_id",
    "batch_id",
    "batch_policy_id",
    "run_context_id",
    "execution_freeze_manifest_id",
    "protocol_id",
    "case_count",
    *_TRUE_RECEIPT_CLAIMS_V2,
    "metric_results",
    "scored_rows",
    *_FALSE_RECEIPT_CLAIMS_V2,
)
_REJECTION_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "schema_version",
    "policy_id",
    "claim_level",
    "receipt",
    "metric_results",
    "scored_rows",
    "partial_output_published",
    *_FALSE_RECEIPT_CLAIMS_V2,
)
_RECEIPT_PREIMAGE_FIELDS_V2: Final = tuple(
    name for name in _RECEIPT_FIELDS_V2 if name != "receipt_id"
)


def _assert_field_manifests_v2() -> None:
    receipt_manifest = tuple(item.name for item in fields(StrictRecognizerStructuralReceiptV2))
    rejection_manifest = tuple(item.name for item in fields(StrictRecognizerStructuralRejectionV2))
    if receipt_manifest != _RECEIPT_FIELDS_V2:
        raise RuntimeError("strict V2 receipt field manifest drift")
    if rejection_manifest != _REJECTION_FIELDS_V2:
        raise RuntimeError("strict V2 rejection field manifest drift")


def _require_exact_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must use bounded exact ASCII text")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise TypeError(f"{name} must use bounded exact ASCII text") from exc
    if len(encoded) > 512:
        raise TypeError(f"{name} must use bounded exact ASCII text")
    return value


def _require_hex64(value: object, *, name: str) -> str:
    text = _require_exact_text(value, name=name)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} digest drift")
    return text


def _require_digest(value: object, *, prefix: str, name: str) -> str:
    text = _require_exact_text(value, name=name)
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    _require_hex64(text[len(prefix) :], name=name)
    return text


def _validate_dependency_identity_v2() -> None:
    identities = (
        (RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2, "phase2b_recognizer_input_archive_policy_v2_"),
        (RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2, "phase2b_recognizer_prediction_archive_policy_v2_"),
        (TRUSTED_WIRE_BATCH_V2_POLICY_ID, "phase2b_trusted_wire_batch_v2_policy_"),
    )
    for value, prefix in identities:
        _require_digest(value, prefix=prefix, name="strict V2 dependency")
    for value in (
        TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
        PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION,
        NON_AUTHORITATIVE_CLAIM_LEVEL,
    ):
        _require_exact_text(value, name="strict V2 dependency")


_validate_dependency_identity_v2()

STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID: Final = stable_hash(
    {
        "version": STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION,
        "receipt_fields": _RECEIPT_FIELDS_V2,
        "rejection_fields": _REJECTION_FIELDS_V2,
        "true_claims": _TRUE_RECEIPT_CLAIMS_V2,
        "false_claims": _FALSE_RECEIPT_CLAIMS_V2,
        "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
        "success_disposition": "COMPLETE",
        "success_reason": "strict_v2_structural_input_output_binding_complete",
        "rejection_disposition": "ABSTAIN",
        "rejection_reason": STRICT_RECOGNIZER_CLI_V2_GENERIC_REJECTION_REASON,
    },
    prefix="phase2b_strict_recognizer_cli_schema_v2_",
)

STRICT_RECOGNIZER_CLI_V2_POLICY_ID: Final = stable_hash(
    {
        "version": STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION,
        "schema_id": STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID,
        "input_archive": {
            "version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
            "policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
            "header_bytes": ARCHIVE_HEADER_BYTES_V2,
            "maximum_bytes": MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
        },
        "prediction_archive": {
            "version": RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
            "policy_id": RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
            "record_schema_version": PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION,
            "header_bytes": PREDICTION_ARCHIVE_HEADER_BYTES_V2,
            "maximum_bytes": MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
        },
        "batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "case_count": _EXACT_CASE_COUNT_V2,
        "input_row_root_formula": (
            "sha256(V2_domain||u32_count||repeated(u16_ascii_length||row_id))"
        ),
        "cross_archive_fields": (
            "input_archive_id",
            "input_archive_sha256",
            "batch_id",
            "batch_policy_id",
            "ordered_input_row_ids",
            "seven_input_roots_per_position",
        ),
        "decoder_calls": (
            "one_public_input_archive_v2_decode",
            "one_public_prediction_archive_v2_decode",
            "no_private_or_v1_decode",
        ),
        "file_contract": (
            "command:phase2b-verify-v2-structure",
            "argv:exactly_input_archive_and_prediction_archive_absolute_paths",
            "canonical_absolute_path",
            "componentwise_openat_nofollow_nonblocking_fd",
            "single_link_regular_file",
            "bounded_read_before_any_decode",
            "pre_post_fstat_stability",
            "read_only_no_output_artifact",
        ),
        "output_contract": (
            "success_exit_0_stdout_one_compact_sorted_ascii_json_line_stderr_empty",
            "failure_exit_2_stderr_one_generic_compact_sorted_ascii_json_line_stdout_empty",
            "no_path_exception_or_usage_leak",
            "no_partial_receipt",
        ),
        "scope": (
            "non_authoritative_structural_mechanics_only",
            "no_partition_labels",
            "no_recognizer_runtime_scoring_effect_capacity_or_c1_claim",
        ),
    },
    prefix="phase2b_strict_recognizer_cli_policy_v2_",
)


class StrictRecognizerCliDispositionV2(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


def _receipt_mapping_without_id_v2(
    value: "StrictRecognizerStructuralReceiptV2",
) -> dict[str, object]:
    return {
        name: _json_value_v2(getattr(value, name))
        for name in _RECEIPT_PREIMAGE_FIELDS_V2
    }


def _json_value_v2(value: object) -> object:
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is tuple:
        return [_json_value_v2(item) for item in value]
    if type(value) is StrictRecognizerCliDispositionV2:
        return value.value
    raise TypeError("strict V2 JSON value type drift")


def _validate_claims_v2(value: object, *, complete: bool) -> None:
    claims = tuple(
        object.__getattribute__(value, name)
        for name in _FALSE_RECEIPT_CLAIMS_V2
    )
    if any(type(item) is not bool or item for item in claims):
        raise ValueError("strict V2 non-evidence claim drift")
    if complete:
        true_claims = tuple(
            object.__getattribute__(value, name)
            for name in _TRUE_RECEIPT_CLAIMS_V2
        )
        if any(type(item) is not bool or not item for item in true_claims):
            raise ValueError("strict V2 structural claim drift")


@dataclass(frozen=True, slots=True, init=False)
class StrictRecognizerStructuralReceiptV2:
    disposition: StrictRecognizerCliDispositionV2
    reason: str
    schema_version: str
    policy_id: str
    claim_level: str
    receipt_id: str
    input_archive_id: str
    input_archive_sha256: str
    input_archive_version: str
    input_archive_policy_id: str
    prediction_archive_id: str
    prediction_archive_sha256: str
    prediction_archive_version: str
    prediction_archive_policy_id: str
    batch_id: str
    batch_policy_id: str
    run_context_id: str
    execution_freeze_manifest_id: str
    protocol_id: str
    case_count: int
    structural_input_archive_verified: bool
    structural_prediction_archive_verified: bool
    cross_archive_context_binding_verified: bool
    ordered_row_identity_verified: bool
    seven_input_root_columns_positionally_verified: bool
    metric_results: tuple[()]
    scored_rows: tuple[()]
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

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("strict V2 receipts are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        input_archive_id: str,
        input_archive_sha256: str,
        input_archive_version: str,
        input_archive_policy_id: str,
        prediction_archive_id: str,
        prediction_archive_sha256: str,
        prediction_archive_version: str,
        prediction_archive_policy_id: str,
        batch_id: str,
        batch_policy_id: str,
        run_context_id: str,
        execution_freeze_manifest_id: str,
        protocol_id: str,
    ) -> "StrictRecognizerStructuralReceiptV2":
        if token is not _RECEIPT_ISSUE_TOKEN_V2:
            raise TypeError("strict V2 receipt issuer is private")
        _require_digest(
            input_archive_id,
            prefix="phase2b_recognizer_input_archive_v2_",
            name="strict V2 input archive ID",
        )
        _require_hex64(input_archive_sha256, name="strict V2 input SHA")
        if (
            type(input_archive_version) is not str
            or input_archive_version != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
            or type(input_archive_policy_id) is not str
            or input_archive_policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        ):
            raise ValueError("strict V2 input archive identity drift")
        _require_digest(
            prediction_archive_id,
            prefix="phase2b_recognizer_prediction_archive_v2_",
            name="strict V2 prediction archive ID",
        )
        _require_hex64(prediction_archive_sha256, name="strict V2 prediction SHA")
        if (
            type(prediction_archive_version) is not str
            or prediction_archive_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
            or type(prediction_archive_policy_id) is not str
            or prediction_archive_policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        ):
            raise ValueError("strict V2 prediction archive identity drift")
        _require_digest(
            batch_id,
            prefix="phase2b_trusted_wire_batch_v2_",
            name="strict V2 batch ID",
        )
        if type(batch_policy_id) is not str or batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID:
            raise ValueError("strict V2 batch policy drift")
        _require_digest(
            run_context_id,
            prefix="phase2b_public_prediction_run_context_v2_",
            name="strict V2 context ID",
        )
        _require_digest(
            execution_freeze_manifest_id,
            prefix="phase2b_execution_freeze_",
            name="strict V2 freeze ID",
        )
        _require_digest(
            protocol_id,
            prefix="phase2b_protocol_",
            name="strict V2 protocol ID",
        )
        value = object.__new__(cls)
        frozen = (
            ("disposition", StrictRecognizerCliDispositionV2.COMPLETE),
            ("reason", "strict_v2_structural_input_output_binding_complete"),
            ("schema_version", STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION),
            ("policy_id", STRICT_RECOGNIZER_CLI_V2_POLICY_ID),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("input_archive_id", input_archive_id),
            ("input_archive_sha256", input_archive_sha256),
            ("input_archive_version", input_archive_version),
            ("input_archive_policy_id", input_archive_policy_id),
            ("prediction_archive_id", prediction_archive_id),
            ("prediction_archive_sha256", prediction_archive_sha256),
            ("prediction_archive_version", prediction_archive_version),
            ("prediction_archive_policy_id", prediction_archive_policy_id),
            ("batch_id", batch_id),
            ("batch_policy_id", batch_policy_id),
            ("run_context_id", run_context_id),
            ("execution_freeze_manifest_id", execution_freeze_manifest_id),
            ("protocol_id", protocol_id),
            ("case_count", _EXACT_CASE_COUNT_V2),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        for name in _TRUE_RECEIPT_CLAIMS_V2:
            object.__setattr__(value, name, True)
        object.__setattr__(value, "metric_results", ())
        object.__setattr__(value, "scored_rows", ())
        for name in _FALSE_RECEIPT_CLAIMS_V2:
            object.__setattr__(value, name, False)
        object.__setattr__(
            value,
            "receipt_id",
            stable_hash(
                _receipt_mapping_without_id_v2(value),
                prefix=_RECEIPT_ID_PREFIX_V2,
            ),
        )
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not StrictRecognizerStructuralReceiptV2:
            raise TypeError("strict V2 receipt exact type drift")
        if self.disposition is not StrictRecognizerCliDispositionV2.COMPLETE:
            raise ValueError("strict V2 receipt disposition drift")
        if (
            type(self.reason) is not str
            or self.reason != "strict_v2_structural_input_output_binding_complete"
            or type(self.schema_version) is not str
            or self.schema_version != STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION
            or type(self.policy_id) is not str
            or self.policy_id != STRICT_RECOGNIZER_CLI_V2_POLICY_ID
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
            or type(self.case_count) is not int
            or self.case_count != _EXACT_CASE_COUNT_V2
        ):
            raise ValueError("strict V2 receipt fixed field drift")
        _require_digest(
            self.receipt_id,
            prefix=_RECEIPT_ID_PREFIX_V2,
            name="strict V2 receipt ID",
        )
        _require_digest(
            self.input_archive_id,
            prefix="phase2b_recognizer_input_archive_v2_",
            name="strict V2 input archive ID",
        )
        _require_hex64(self.input_archive_sha256, name="strict V2 input SHA")
        if (
            type(self.input_archive_version) is not str
            or self.input_archive_version != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
            or type(self.input_archive_policy_id) is not str
            or self.input_archive_policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        ):
            raise ValueError("strict V2 receipt input identity drift")
        _require_digest(
            self.prediction_archive_id,
            prefix="phase2b_recognizer_prediction_archive_v2_",
            name="strict V2 prediction archive ID",
        )
        _require_hex64(
            self.prediction_archive_sha256,
            name="strict V2 prediction SHA",
        )
        if (
            type(self.prediction_archive_version) is not str
            or self.prediction_archive_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
            or type(self.prediction_archive_policy_id) is not str
            or self.prediction_archive_policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        ):
            raise ValueError("strict V2 receipt prediction identity drift")
        _require_digest(
            self.batch_id,
            prefix="phase2b_trusted_wire_batch_v2_",
            name="strict V2 batch ID",
        )
        if type(self.batch_policy_id) is not str or self.batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID:
            raise ValueError("strict V2 receipt batch policy drift")
        _require_digest(
            self.run_context_id,
            prefix="phase2b_public_prediction_run_context_v2_",
            name="strict V2 context ID",
        )
        _require_digest(
            self.execution_freeze_manifest_id,
            prefix="phase2b_execution_freeze_",
            name="strict V2 freeze ID",
        )
        _require_digest(
            self.protocol_id,
            prefix="phase2b_protocol_",
            name="strict V2 protocol ID",
        )
        if (
            type(self.metric_results) is not tuple
            or self.metric_results != ()
            or type(self.scored_rows) is not tuple
            or self.scored_rows != ()
        ):
            raise ValueError("strict V2 receipt leaked scoring output")
        _validate_claims_v2(self, complete=True)
        expected = stable_hash(
            _receipt_mapping_without_id_v2(self),
            prefix=_RECEIPT_ID_PREFIX_V2,
        )
        if self.receipt_id != expected:
            raise ValueError("strict V2 receipt root drift")

    def to_mapping(self) -> dict[str, object]:
        self._validate()
        return {
            name: _json_value_v2(getattr(self, name))
            for name in _RECEIPT_FIELDS_V2
        }


@dataclass(frozen=True, slots=True)
class StrictRecognizerStructuralRejectionV2:
    disposition: StrictRecognizerCliDispositionV2 = StrictRecognizerCliDispositionV2.ABSTAIN
    reason: str = STRICT_RECOGNIZER_CLI_V2_GENERIC_REJECTION_REASON
    schema_version: str = STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION
    policy_id: str = STRICT_RECOGNIZER_CLI_V2_POLICY_ID
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    receipt: None = None
    metric_results: tuple[()] = ()
    scored_rows: tuple[()] = ()
    partial_output_published: bool = False
    input_archive_membership_verified: bool = False
    batch_policy_membership_verified: bool = False
    source_registry_projection_verified: bool = False
    source_public_disjoint_verified: bool = False
    single_live_allocation_verified: bool = False
    secret_custodian_replay_verified: bool = False
    execution_manifest_authority_verified: bool = False
    partition_manifest_authority_verified: bool = False
    derived_mapping_verified: bool = False
    recognizer_executed: bool = False
    runtime_executed: bool = False
    actual_960_case_run_verified: bool = False
    recognizer_capacity_evidence: bool = False
    origin_authenticated: bool = False
    formal_uuid_audit: bool = False
    formal_covert_audit: bool = False
    sealed_holdout_eligible: bool = False
    scoring_performed: bool = False
    prediction_scored: bool = False
    effect_evidence: bool = False
    c1_exit_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self) is not StrictRecognizerStructuralRejectionV2:
            raise TypeError("strict V2 rejection exact type drift")
        if (
            self.disposition is not StrictRecognizerCliDispositionV2.ABSTAIN
            or type(self.reason) is not str
            or self.reason != STRICT_RECOGNIZER_CLI_V2_GENERIC_REJECTION_REASON
            or type(self.schema_version) is not str
            or self.schema_version != STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION
            or type(self.policy_id) is not str
            or self.policy_id != STRICT_RECOGNIZER_CLI_V2_POLICY_ID
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
            or self.receipt is not None
            or type(self.metric_results) is not tuple
            or self.metric_results != ()
            or type(self.scored_rows) is not tuple
            or self.scored_rows != ()
            or type(self.partial_output_published) is not bool
            or self.partial_output_published
        ):
            raise ValueError("strict V2 rejection boundary drift")
        _validate_claims_v2(self, complete=False)

    def to_mapping(self) -> dict[str, object]:
        self.__post_init__()
        return {
            name: _json_value_v2(getattr(self, name))
            for name in _REJECTION_FIELDS_V2
        }


_assert_field_manifests_v2()


def _input_row_ids_root_v2(values: tuple[str, ...]) -> str:
    if type(values) is not tuple or len(values) != _EXACT_CASE_COUNT_V2:
        raise TypeError("strict V2 ordered row IDs require exact frozen count")
    digest = hashlib.sha256()
    digest.update(_INPUT_ROWS_DOMAIN_V2)
    digest.update(_EXACT_CASE_COUNT_V2.to_bytes(4, "big"))
    for value in values:
        text = _require_digest(
            value,
            prefix="phase2b_recognizer_input_row_v2_",
            name="strict V2 ordered row ID",
        )
        encoded = text.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return _INPUT_ROWS_ROOT_PREFIX_V2 + digest.hexdigest()


def _validate_input_wrapper_v2(
    decoded: object,
    archive: bytes,
) -> DecodedRecognizerInputArchiveV2:
    if type(decoded) is not DecodedRecognizerInputArchiveV2:
        raise TypeError("strict V2 input decoder exact result drift")
    if type(decoded.archive) is not bytes or decoded.archive != archive:
        raise ValueError("strict V2 input decoder byte parity drift")
    if (
        type(decoded.archive_version) is not str
        or decoded.archive_version != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or type(decoded.policy_id) is not str
        or decoded.policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or type(decoded.batch_policy_id) is not str
        or decoded.batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
        or type(decoded.claim_level) is not str
        or decoded.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("strict V2 input decoder identity drift")
    _require_digest(
        decoded.archive_id,
        prefix="phase2b_recognizer_input_archive_v2_",
        name="strict V2 input archive ID",
    )
    _require_digest(
        decoded.batch_id,
        prefix="phase2b_trusted_wire_batch_v2_",
        name="strict V2 input batch ID",
    )
    if (
        type(decoded.rows) is not tuple
        or len(decoded.rows) != _EXACT_CASE_COUNT_V2
        or any(type(item) is not TrustedRecognizerInputRowV2 for item in decoded.rows)
        or type(decoded.row_ids) is not tuple
        or len(decoded.row_ids) != _EXACT_CASE_COUNT_V2
    ):
        raise TypeError("strict V2 input row shape drift")
    for row in decoded.rows:
        values = (
            row.row_id,
            row.authority_content_id,
            row.envelope_id,
            row.namespace_audit_id,
            row.padding_sha256,
            row.payload_sha256,
            row.public_registry_id,
            row.transform_result_id,
        )
        _validate_row_root_values_v2(values, name="strict V2 input row")
    for value in decoded.row_ids:
        _require_digest(
            value,
            prefix="phase2b_recognizer_input_row_v2_",
            name="strict V2 stored input row ID",
        )
    for name in (
        "structural_archive_verified",
        "row_bijection_verified",
        "registry_schema_verified",
        "registry_authority_exact_scope_verified",
        "compact_typed_replay_verified",
        "direct_payload_transform_replay_verified",
        "cross_row_public_uuid_disjoint_verified",
    ):
        if type(getattr(decoded, name)) is not bool or not getattr(decoded, name):
            raise ValueError("strict V2 input structural claim drift")
    for name in (
        "batch_policy_membership_verified",
        "source_registry_projection_verified",
        "source_public_disjoint_verified",
        "single_live_allocation_verified",
        "secret_custodian_replay_verified",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "recognizer_executed",
        "prediction_archive_evaluated",
        "recognizer_capacity_evidence",
        "c1_exit_evidence",
    ):
        if type(getattr(decoded, name)) is not bool or getattr(decoded, name):
            raise ValueError("strict V2 input non-evidence claim drift")
    return decoded


def _validate_prediction_wrapper_v2(
    decoded: object,
    archive: bytes,
) -> DecodedRecognizerPredictionArchiveV2:
    if type(decoded) is not DecodedRecognizerPredictionArchiveV2:
        raise TypeError("strict V2 prediction decoder exact result drift")
    if type(decoded.archive) is not bytes or decoded.archive != archive:
        raise ValueError("strict V2 prediction decoder byte parity drift")
    if (
        type(decoded.schema_version) is not str
        or decoded.schema_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
        or type(decoded.policy_id) is not str
        or decoded.policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        or type(decoded.claim_level) is not str
        or decoded.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("strict V2 prediction decoder identity drift")
    _require_digest(
        decoded.archive_id,
        prefix="phase2b_recognizer_prediction_archive_v2_",
        name="strict V2 prediction archive ID",
    )
    if type(decoded.context) is not PublicPredictionRunContextV2:
        raise TypeError("strict V2 prediction context type drift")
    context_string_fields = (
        "context_id",
        "batch_id",
        "batch_policy_id",
        "input_archive_id",
        "input_archive_policy_id",
        "input_archive_sha256",
        "input_archive_version",
        "input_row_ids_root",
        "execution_freeze_manifest_id",
        "protocol_id",
        "schema_version",
        "claim_level",
    )
    if any(
        type(getattr(decoded.context, name)) is not str
        for name in context_string_fields
    ):
        raise TypeError("strict V2 prediction context scalar type drift")
    if (
        type(decoded.context.expected_prediction_count) is not int
        or decoded.context.expected_prediction_count != _EXACT_CASE_COUNT_V2
    ):
        raise ValueError("strict V2 prediction context count drift")
    if (
        type(decoded.records) is not tuple
        or len(decoded.records) != _EXACT_CASE_COUNT_V2
        or any(
            type(item) is not PublicRecognizerPredictionRecordV2
            for item in decoded.records
        )
        or type(decoded.input_row_ids) is not tuple
        or len(decoded.input_row_ids) != _EXACT_CASE_COUNT_V2
    ):
        raise TypeError("strict V2 prediction record shape drift")
    for value in decoded.input_row_ids:
        _require_digest(
            value,
            prefix="phase2b_recognizer_input_row_v2_",
            name="strict V2 stored prediction input row ID",
        )
    for record in decoded.records:
        values = (
            record.input_row_id,
            record.input_authority_content_id,
            record.input_envelope_id,
            record.input_namespace_audit_id,
            record.input_padding_sha256,
            record.input_payload_sha256,
            record.input_public_registry_id,
            record.input_transform_result_id,
        )
        _validate_row_root_values_v2(values, name="strict V2 prediction record")
        if (
            type(record.schema_version) is not str
            or record.schema_version
            != PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION
            or type(record.run_context_id) is not str
        ):
            raise ValueError("strict V2 prediction record identity drift")
    for name in (
        "structural_archive_verified",
        "canonical_record_framing_verified",
        "record_schema_verified",
        "row_root_coverage_verified",
    ):
        if type(getattr(decoded, name)) is not bool or not getattr(decoded, name):
            raise ValueError("strict V2 prediction structural claim drift")
    for name in (
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
    ):
        if type(getattr(decoded, name)) is not bool or getattr(decoded, name):
            raise ValueError("strict V2 prediction non-evidence claim drift")
    return decoded


def _validate_row_root_values_v2(
    values: tuple[object, ...],
    *,
    name: str,
) -> None:
    if type(values) is not tuple or len(values) != 8 or any(
        type(item) is not str for item in values
    ):
        raise TypeError(f"{name} scalar type drift")
    _require_digest(values[0], prefix="phase2b_recognizer_input_row_v2_", name=f"{name} row ID")
    _require_digest(values[1], prefix="phase2b_public_transform_evidence_", name=f"{name} authority root")
    _require_digest(values[2], prefix="phase2b_trusted_envelope_v2_", name=f"{name} envelope root")
    _require_digest(values[3], prefix="phase2b_namespace_audit_v2_", name=f"{name} namespace root")
    _require_hex64(values[4], name=f"{name} padding SHA")
    _require_hex64(values[5], name=f"{name} payload SHA")
    _require_digest(values[6], prefix="phase2b_public_recognizer_registry_v2_", name=f"{name} registry root")
    _require_digest(values[7], prefix="phase2b_exact_transform_result_", name=f"{name} transform root")


def _preflight_cross_archive_binding_v2(
    input_decoded: DecodedRecognizerInputArchiveV2,
    prediction_decoded: DecodedRecognizerPredictionArchiveV2,
) -> None:
    context = prediction_decoded.context
    string_fields = (
        "context_id",
        "batch_id",
        "batch_policy_id",
        "input_archive_id",
        "input_archive_policy_id",
        "input_archive_sha256",
        "input_archive_version",
        "input_row_ids_root",
        "execution_freeze_manifest_id",
        "protocol_id",
        "schema_version",
        "claim_level",
    )
    if any(type(getattr(context, name)) is not str for name in string_fields):
        raise TypeError("strict V2 prediction context scalar type drift")
    _require_digest(
        context.context_id,
        prefix="phase2b_public_prediction_run_context_v2_",
        name="strict V2 context ID",
    )
    _require_digest(
        context.batch_id,
        prefix="phase2b_trusted_wire_batch_v2_",
        name="strict V2 context batch ID",
    )
    _require_digest(
        context.batch_policy_id,
        prefix="phase2b_trusted_wire_batch_v2_policy_",
        name="strict V2 context batch policy",
    )
    _require_digest(
        context.input_archive_id,
        prefix="phase2b_recognizer_input_archive_v2_",
        name="strict V2 context input archive ID",
    )
    _require_digest(
        context.input_archive_policy_id,
        prefix="phase2b_recognizer_input_archive_policy_v2_",
        name="strict V2 context input policy",
    )
    _require_hex64(context.input_archive_sha256, name="strict V2 context input SHA")
    _require_digest(
        context.input_row_ids_root,
        prefix="phase2b_prediction_input_rows_v2_",
        name="strict V2 context row root",
    )
    _require_digest(
        context.execution_freeze_manifest_id,
        prefix="phase2b_execution_freeze_",
        name="strict V2 context freeze ID",
    )
    _require_digest(
        context.protocol_id,
        prefix="phase2b_protocol_",
        name="strict V2 context protocol ID",
    )
    if (
        type(context.expected_prediction_count) is not int
        or context.expected_prediction_count != _EXACT_CASE_COUNT_V2
        or context.input_archive_id != input_decoded.archive_id
        or context.batch_id != input_decoded.batch_id
        or context.batch_policy_id != input_decoded.batch_policy_id
        or context.batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
        or context.input_archive_policy_id != input_decoded.policy_id
        or context.input_archive_policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or context.input_archive_version != input_decoded.archive_version
        or context.input_archive_version != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or context.schema_version != PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION
        or context.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("strict V2 prediction context input binding drift")
    input_row_ids = tuple(row.row_id for row in input_decoded.rows)
    if (
        input_decoded.row_ids != input_row_ids
        or prediction_decoded.input_row_ids != input_row_ids
        or context.input_row_ids_root != _input_row_ids_root_v2(input_row_ids)
    ):
        raise ValueError("strict V2 ordered row identity drift")
    for input_row, record in zip(input_decoded.rows, prediction_decoded.records):
        row_values = (
            input_row.row_id,
            input_row.authority_content_id,
            input_row.envelope_id,
            input_row.namespace_audit_id,
            input_row.padding_sha256,
            input_row.payload_sha256,
            input_row.public_registry_id,
            input_row.transform_result_id,
        )
        record_values = (
            record.input_row_id,
            record.input_authority_content_id,
            record.input_envelope_id,
            record.input_namespace_audit_id,
            record.input_padding_sha256,
            record.input_payload_sha256,
            record.input_public_registry_id,
            record.input_transform_result_id,
        )
        if row_values != record_values:
            raise ValueError("strict V2 seven-root positional binding drift")
        if (
            type(record.schema_version) is not str
            or record.schema_version
            != PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION
            or type(record.run_context_id) is not str
            or record.run_context_id != context.context_id
        ):
            raise ValueError("strict V2 record context identity drift")


def verify_strict_recognizer_io_structure_v2(
    *,
    input_archive: bytes,
    prediction_archive: bytes,
) -> StrictRecognizerStructuralReceiptV2:
    """Verify exact public V2 archive bytes as structural mechanics only."""

    if type(input_archive) is not bytes or type(prediction_archive) is not bytes:
        raise TypeError("strict V2 verifier requires exact archive bytes")
    if not (
        ARCHIVE_HEADER_BYTES_V2
        <= len(input_archive)
        <= MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2
    ):
        raise ValueError("strict V2 input archive byte cap drift")
    if not (
        PREDICTION_ARCHIVE_HEADER_BYTES_V2
        <= len(prediction_archive)
        <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
    ):
        raise ValueError("strict V2 prediction archive byte cap drift")

    input_decoded = _validate_input_wrapper_v2(
        decode_public_recognizer_input_archive_v2(input_archive),
        input_archive,
    )
    prediction_decoded = _validate_prediction_wrapper_v2(
        decode_public_recognizer_prediction_archive_v2(prediction_archive),
        prediction_archive,
    )
    _preflight_cross_archive_binding_v2(
        input_decoded,
        prediction_decoded,
    )
    input_sha256 = hashlib.sha256(input_archive).hexdigest()
    prediction_sha256 = hashlib.sha256(prediction_archive).hexdigest()
    if prediction_decoded.context.input_archive_sha256 != input_sha256:
        raise ValueError("strict V2 prediction context input SHA binding drift")
    return StrictRecognizerStructuralReceiptV2._issue(
        _RECEIPT_ISSUE_TOKEN_V2,
        input_archive_id=input_decoded.archive_id,
        input_archive_sha256=input_sha256,
        input_archive_version=input_decoded.archive_version,
        input_archive_policy_id=input_decoded.policy_id,
        prediction_archive_id=prediction_decoded.archive_id,
        prediction_archive_sha256=prediction_sha256,
        prediction_archive_version=prediction_decoded.schema_version,
        prediction_archive_policy_id=prediction_decoded.policy_id,
        batch_id=input_decoded.batch_id,
        batch_policy_id=input_decoded.batch_policy_id,
        run_context_id=prediction_decoded.context.context_id,
        execution_freeze_manifest_id=(
            prediction_decoded.context.execution_freeze_manifest_id
        ),
        protocol_id=prediction_decoded.context.protocol_id,
    )


def _canonical_json_line_v2(value: object) -> str:
    if type(value) is StrictRecognizerStructuralReceiptV2:
        mapping = value.to_mapping()
    elif type(value) is StrictRecognizerStructuralRejectionV2:
        mapping = value.to_mapping()
    else:
        raise TypeError("strict V2 CLI output exact type drift")
    return json.dumps(
        mapping,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _parse_cli_arguments_v2(argv: Sequence[str] | None) -> tuple[str, str]:
    supplied: object = sys.argv[1:] if argv is None else argv
    if type(supplied) not in (list, tuple) or len(supplied) != 4:
        raise ValueError("strict V2 CLI argument shape drift")
    raw: object = tuple(supplied)
    if type(raw) is not tuple or len(raw) != 4 or any(type(item) is not str for item in raw):
        raise ValueError("strict V2 CLI argument shape drift")
    values: dict[str, str] = {}
    for index in (0, 2):
        flag = raw[index]
        value = raw[index + 1]
        if flag not in ("--input-archive", "--prediction-archive") or flag in values:
            raise ValueError("strict V2 CLI argument name drift")
        values[flag] = value
    if set(values) != {"--input-archive", "--prediction-archive"}:
        raise ValueError("strict V2 CLI required argument drift")
    return values["--input-archive"], values["--prediction-archive"]


def _validate_canonical_absolute_path_v2(value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError("strict V2 path shape drift")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("strict V2 path encoding drift") from exc
    if len(encoded) > _MAXIMUM_PATH_BYTES_V2 or not os.path.isabs(value):
        raise ValueError("strict V2 path bound drift")
    if os.path.normpath(value) != value or os.path.abspath(value) != value:
        raise ValueError("strict V2 path canonicality drift")
    return value


def _stat_fingerprint_v2(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_bounded_regular_file_v2(
    path: object,
    *,
    minimum_bytes: int,
    maximum_bytes: int,
) -> bytes:
    canonical = _validate_canonical_absolute_path_v2(path)
    if (
        type(minimum_bytes) is not int
        or type(maximum_bytes) is not int
        or not 1 <= minimum_bytes <= maximum_bytes
    ):
        raise ValueError("strict V2 file bound contract drift")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if not nofollow or not directory or not nonblock:
        raise OSError("strict V2 secure path flags unavailable")
    common = getattr(os, "O_CLOEXEC", 0) | nofollow | nonblock
    components = canonical.split(os.sep)[1:]
    if not components or any(not item or item in (".", "..") for item in components):
        raise ValueError("strict V2 path component drift")
    parent_fd = os.open(os.sep, os.O_RDONLY | common | directory)
    try:
        for component in components[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | common | directory,
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = next_fd
        fd = os.open(
            components[-1],
            os.O_RDONLY | common,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("strict V2 input is not a regular file")
        if not minimum_bytes <= before.st_size <= maximum_bytes:
            raise ValueError("strict V2 file byte cap drift")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(_READ_CHUNK_BYTES_V2, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("strict V2 file grew beyond byte cap")
            chunks.append(chunk)
        after = os.fstat(fd)
        if _stat_fingerprint_v2(before) != _stat_fingerprint_v2(after):
            raise ValueError("strict V2 file changed during read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size or not minimum_bytes <= len(payload) <= maximum_bytes:
            raise ValueError("strict V2 file read length drift")
        return payload
    finally:
        os.close(fd)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict structural command with a fixed fail-closed surface."""

    try:
        input_path, prediction_path = _parse_cli_arguments_v2(argv)
        input_archive = _read_bounded_regular_file_v2(
            input_path,
            minimum_bytes=ARCHIVE_HEADER_BYTES_V2,
            maximum_bytes=MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
        )
        prediction_archive = _read_bounded_regular_file_v2(
            prediction_path,
            minimum_bytes=PREDICTION_ARCHIVE_HEADER_BYTES_V2,
            maximum_bytes=MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
        )
        receipt = verify_strict_recognizer_io_structure_v2(
            input_archive=input_archive,
            prediction_archive=prediction_archive,
        )
        line = _canonical_json_line_v2(receipt)
    except Exception:
        rejection = StrictRecognizerStructuralRejectionV2()
        sys.stderr.write(_canonical_json_line_v2(rejection))
        return 2
    sys.stdout.write(line)
    return 0


__all__ = [
    "STRICT_RECOGNIZER_CLI_V2_COMMAND",
    "STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION",
    "STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID",
    "STRICT_RECOGNIZER_CLI_V2_POLICY_ID",
    "STRICT_RECOGNIZER_CLI_V2_GENERIC_REJECTION_REASON",
    "StrictRecognizerCliDispositionV2",
    "StrictRecognizerStructuralReceiptV2",
    "StrictRecognizerStructuralRejectionV2",
    "verify_strict_recognizer_io_structure_v2",
    "main",
]

"""Pure decision core for the bounded Q0.5b shadow retry campaign.

This module performs no filesystem, process, Docker, network, artifact, retry,
or execution action. It validates one digest-bound ``SlotObservation`` and
returns one decision primitive. A caller may use ``RETRY_EXACT_GAP`` to build
a new independently authorized slot, but this core neither creates nor invokes
that slot. The campaign has exactly three possible slots and no slot four.

Raw supervisor stdout is classified by :func:`classify_supervisor_stdout_v2`.
That classifier freezes the existing supervisor success-result and error/2
registries; it does not invent a normalized wire. The wrapper records the
classifier's raw-length, raw-digest, canonicality, kind, status, and artifact
facts verbatim in the SlotObservation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Final, Mapping, NoReturn


__all__ = (
    "ShadowRetryCampaignValidationErrorV2",
    "SLOT_OBSERVATION_SCHEMA_V2",
    "SLOT_OBSERVATION_FIELDS_V2",
    "FAILURE_ROW_FIELDS_V2",
    "CLEANUP_EVENT_ROW_FIELDS_V2",
    "PARTITION_CERTIFICATE_SCHEMA_V2",
    "PARTITION_CERTIFICATE_FIELDS_V2",
    "SUCCESS_RESULT_FIELDS_V2",
    "ERROR_WIRE_SCHEMA_V2",
    "ERROR_WIRE_FIELDS_V2",
    "STDOUT_WIRE_SUCCESS_RESULT_V2",
    "STDOUT_WIRE_ERROR_V2",
    "STDOUT_WIRE_INVALID_V2",
    "MAX_CAMPAIGN_SLOTS_V2",
    "CAPTURE_MAX_BYTES_V2",
    "TARGET_WAIT_NS_V2",
    "MAXIMUM_GAP_NS_V2",
    "GAP_ERROR_CODE_V2",
    "SUCCESS_V2",
    "RETRY_EXACT_GAP_V2",
    "EXHAUSTED_RETRYABLE_GAP_V2",
    "TERMINAL_NONRETRYABLE_V2",
    "FAIL_TOP_LEVEL_V2",
    "FAIL_CANONICAL_V2",
    "canonical_campaign_json_bytes_v2",
    "strict_campaign_json_value_v2",
    "campaign_record_digest_v2",
    "slot_observation_record_digest_v2",
    "failure_rows_digest_v2",
    "cleanup_rows_digest_v2",
    "partition_certificate_digest_v2",
    "classify_supervisor_stdout_v2",
    "evaluate_slot_observation_v2",
)


SLOT_OBSERVATION_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-campaign-slot-observation/2"
)
ERROR_WIRE_SCHEMA_V2: Final = "hegel-phase3a-q05b-dual-qualification-error/2"
PARTITION_CERTIFICATE_SCHEMA_V2: Final = (
    "hegel-phase3a-q05b-shadow-retry-post-origin-partition-certificate/2"
)
STDOUT_WIRE_SUCCESS_RESULT_V2: Final = "SUCCESS_RESULT"
STDOUT_WIRE_ERROR_V2: Final = "ERROR_WIRE"
STDOUT_WIRE_INVALID_V2: Final = "INVALID"

MAX_CAMPAIGN_SLOTS_V2: Final = 3
CAPTURE_MAX_BYTES_V2: Final = 64 * 1024 * 1024
TARGET_WAIT_NS_V2: Final = 125_000_000
MAXIMUM_GAP_NS_V2: Final = 250_000_000
GAP_ERROR_CODE_V2: Final = "HELD_ACTOR_GAP_EXCEEDED"

SUCCESS_V2: Final = "SUCCESS"
RETRY_EXACT_GAP_V2: Final = "RETRY_EXACT_GAP"
EXHAUSTED_RETRYABLE_GAP_V2: Final = "EXHAUSTED_RETRYABLE_GAP"
TERMINAL_NONRETRYABLE_V2: Final = "TERMINAL_NONRETRYABLE"

FAIL_TOP_LEVEL_V2: Final = "FAIL_Q05B_RETRY_CAMPAIGN_V2_TOP_LEVEL"
FAIL_CANONICAL_V2: Final = "FAIL_Q05B_RETRY_CAMPAIGN_V2_CANONICAL"

_SLOT_DIGEST_DOMAIN: Final = (
    "HEGEL/Q05B/SHADOW-RETRY-CAMPAIGN/SLOT-OBSERVATION/V2\x00"
)
_FAILURE_ROWS_DIGEST_DOMAIN: Final = (
    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R13/TRAJECTORY_FAILURE_ROWS/V2\x00"
)
_CLEANUP_ROWS_DIGEST_DOMAIN: Final = (
    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R13/TRAJECTORY_CLEANUP_ROWS/V2\x00"
)
_PARTITION_CERTIFICATE_DIGEST_DOMAIN: Final = (
    "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R13/"
    "POST_ORIGIN_PARTITION_CERTIFICATE/V2\x00"
)
_ACTUAL_FINAL_DELIVERY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/FINAL_DELIVERY/V1\x00"
)
_ACTUAL_FINAL_DELIVERY_IDENTITY_SCHEMA: Final = (
    "hegel-phase3a-q05b-final-delivery-identity/2"
)
_ACTUAL_FINAL_DELIVERY_HANDOFF: Final = (
    "FINAL_ANCHORED_REPLAY_AFTER_POSTPUBLICATION_LIVE_"
    "BEFORE_HANDLE_CLOSE_AND_RETURN"
)
_TAGGED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,255}\Z")
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_MAX_COUNTER: Final = (1 << 53) - 1
_MAX_UINT64: Final = (1 << 64) - 1
_MAX_REGISTERED_CLEANUP_PAIRS_PER_ACTOR: Final = 2
_MAX_RETAINED_FAILURE_ROWS: Final = 7
_MAX_RETAINED_CLEANUP_ROWS: Final = 12
_MAX_PARTITION_CERTIFICATE_BYTES: Final = 256 * 1024

_STAGE_REGISTRY: Final = (
    (1, "FREEZE_SOURCE_IMAGES_AND_COMMAND_REGISTRY"),
    (2, "MATERIALIZE_AND_SEAL_THREE_SOURCE_SNAPSHOTS_AND_CARGO"),
    (3, "OFFLINE_RUST_TEST_AND_RELEASE_BUILD_AND_SEAL_BINARY"),
    (4, "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL"),
    (5, "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES"),
    (6, "RUN_HELD_TRUSTED_HOST_SEMANTIC_REPLAY"),
    (7, "CLOSE_HOST_RESOURCE_ISOLATION_NEGATIVE_VECTORS_AND_TOCTOU"),
    (8, "BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE"),
    (9, "ADD_PREDICATE20_AND_BUILD_CANONICAL_ARTIFACT_IN_MEMORY"),
    (10, "STRICTLY_REPLAY_COMPLETE_ARTIFACT_BEFORE_PUBLICATION"),
)

_SAMPLE_OPERATIONS: Final = frozenset(
    {"LIVE_RESOURCE_SAMPLE", "HELD_FINAL_RESOURCE_SAMPLE"}
)
_ROLE_ACTOR: Final = {
    "1": "PYTHON_ENDPOINT",
    "2": "RUST_ENDPOINT",
    "3": "TRUSTED_HOST_REPLAY",
}
_SAMPLE_FIELDS: Final = (
    "role_id_or_null",
    "sample_ordinal_or_null",
    "previous_finish_ns_or_null",
    "scheduled_start_ns_or_null",
    "actual_start_ns_or_null",
    "finish_ns_or_null",
    "collection_duration_ns_or_null",
    "scheduler_lateness_ns_or_null",
    "target_wait_ns_or_null",
    "gap_ns_or_null",
    "maximum_gap_ns_or_null",
    "gap_exceeded_or_null",
)

_TOP_FIELDS: Final = frozenset(
    {
        "schema_version",
        "campaign_id",
        "slot_number",
        "expected_trace_id",
        "expected_producer_id",
        "invocation_count",
        "supervisor_returned",
        "supervisor_reaped",
        "process_group_absent",
        "exit_code_or_null",
        "stdout_capture_complete",
        "stdout_capture_bounded",
        "stdout_raw_byte_count",
        "stdout_raw_sha256",
        "stdout_canonical_single_lf",
        "stdout_wire_kind",
        "stdout_wire_value_or_null",
        "stdout_status_or_null",
        "stdout_artifact_written_or_null",
        "stderr_capture_complete",
        "stderr_capture_bounded",
        "stderr_raw_byte_count",
        "stderr_raw_sha256",
        "stderr_empty",
        "trajectory_status",
        "trajectory_trace_id",
        "trajectory_producer_id",
        "trajectory_torn_tail_or_null",
        "trajectory_open_span_ids",
        "trajectory_terminal_error_code_or_null",
        "trajectory_terminal_error_detail_or_null",
        "trajectory_terminal_event_type_or_null",
        "trajectory_event_count",
        "trajectory_first_event_record_digest",
        "trajectory_final_event_record_digest",
        "trajectory_canonical_event_payloads_digest",
        "trajectory_partition_certificate_or_null",
        "trajectory_failure_rows",
        "trajectory_cleanup_rows",
        "artifact_state",
        "artifact_validation_status",
        "artifact_replay_status",
        "cleanup_certain",
        "docker_absence_certain",
        "clone_clean_certain",
        "cargo_isolation_certain",
        "source_binding_certain",
        "external_binding_certain",
        "replay_certain",
        "record_digest",
    }
)

_FAILURE_ROW_FIELDS: Final = frozenset(
    {
        "sequence",
        "timestamp_ns",
        "span_id",
        "operation",
        "stage_or_null",
        "substep_or_null",
        "actor_or_null",
        "detail_or_null",
        "error_code",
        "error_detail",
        "record_digest",
        *_SAMPLE_FIELDS,
    }
)

_CLEANUP_EVENT_ROW_FIELDS: Final = frozenset(
    {
        "sequence",
        "timestamp_ns",
        "event_type",
        "span_id",
        "operation",
        "stage_or_null",
        "substep_or_null",
        "actor_or_null",
        "detail_or_null",
        "error_code_or_null",
        "error_detail_or_null",
        "record_digest",
    }
)

_PARTITION_CERTIFICATE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "trace_id",
        "producer_id",
        "trajectory_status",
        "event_count",
        "first_event_record_digest",
        "final_event_record_digest",
        "canonical_event_payloads_digest",
        "origin_sequence",
        "origin_record_digest",
        "origin_actor_or_null",
        "suffix_event_count",
        "suffix_event_root",
        "failure_event_count",
        "failure_rows_digest",
        "failure_rows_retained_count",
        "failure_rows_truncated",
        "cleanup_nonfailure_event_count",
        "cleanup_rows_digest",
        "cleanup_rows_retained_count",
        "cleanup_rows_truncated",
        "other_nonfailure_event_count",
        "other_nonfailure_event_root",
        "partitioned_event_count",
        "unpartitioned_event_count",
        "certificate_digest",
    }
)

_SUCCESS_RESULT_FIELDS: Final = frozenset(
    {
        "artifact_path",
        "artifact_sha256",
        "artifact_set_root",
        "candidate_receipt_root",
        "final_receipt_root",
        "qualification_count",
        "qualification_mask",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_formal_output_roots",
        "actual_admission_prepublication_live_replay_root",
        "actual_admission_pre_artifact_live_replay_root",
        "actual_admission_stage10_live_replay_root",
        "actual_admission_postpublication_live_replay_root",
        "actual_final_delivery_identity",
        "actual_final_delivery_root",
        "publisher_handle_close_status",
        "stage_roots",
    }
)

_ERROR_WIRE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "status",
        "detail",
        "qualification_predicate_count",
        "qualification_predicate_mask",
        "qualification_predicate_total",
        "qualification_candidate_receipt",
        "qualification_final_receipt",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_formal_output_roots",
        "q1_receipt",
        "q2_state",
        "m3_formal_roots",
        "receipt_created",
        "artifact_written",
    }
)

_FINAL_DELIVERY_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "handoff_point",
        "source_commit",
        "artifact_path",
        "ordered_stage_root_rows",
        "actual_admission_evidence_root",
        "pre_artifact_live_marker_replay_root",
        "stage_10_live_marker_replay_root",
        "prepublication_live_marker_replay_root",
        "postpublication_live_marker_replay_root",
        "artifact_set_root",
        "candidate_receipt_root",
        "final_receipt_root",
        "publisher_parent_device",
        "publisher_parent_inode",
        "publisher_file_device",
        "publisher_file_inode",
        "publisher_file_mode",
        "publisher_file_nlink",
        "payload_length",
        "payload_sha256",
        "first_anchored_replay_sha256",
        "final_path_matches_publisher_owned_inode",
        "same_uid_mutation_after_handoff_outside_claim",
    }
)

SLOT_OBSERVATION_FIELDS_V2: Final = tuple(sorted(_TOP_FIELDS))
FAILURE_ROW_FIELDS_V2: Final = tuple(sorted(_FAILURE_ROW_FIELDS))
CLEANUP_EVENT_ROW_FIELDS_V2: Final = tuple(
    sorted(_CLEANUP_EVENT_ROW_FIELDS)
)
PARTITION_CERTIFICATE_FIELDS_V2: Final = tuple(
    sorted(_PARTITION_CERTIFICATE_FIELDS)
)
SUCCESS_RESULT_FIELDS_V2: Final = tuple(sorted(_SUCCESS_RESULT_FIELDS))
ERROR_WIRE_FIELDS_V2: Final = tuple(sorted(_ERROR_WIRE_FIELDS))


class ShadowRetryCampaignValidationErrorV2(ValueError):
    """Typed error for malformed top levels and strict codec misuse."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class _ObservationFactInvalidV2(Exception):
    """Internal marker: shaped observation contains a noncanonical fact."""


def _fail(code: str, detail: str) -> NoReturn:
    raise ShadowRetryCampaignValidationErrorV2(code, detail)


def _validate_json_tree(value: object, label: str = "value") -> None:
    value_type = type(value)
    if value is None or value_type in (bool, int):
        return
    if value_type is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ShadowRetryCampaignValidationErrorV2(
                FAIL_CANONICAL_V2, f"{label} contains an unpaired surrogate"
            ) from error
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{label}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(FAIL_CANONICAL_V2, f"{label} contains a non-string key")
            try:
                key.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ShadowRetryCampaignValidationErrorV2(
                    FAIL_CANONICAL_V2,
                    f"{label} contains a key with an unpaired surrogate",
                ) from error
            _validate_json_tree(item, f"{label}.{key}")
        return
    _fail(FAIL_CANONICAL_V2, f"{label} contains a non-exact JSON type")


def canonical_campaign_json_bytes_v2(value: object) -> bytes:
    """Return deterministic compact ASCII JSON followed by exactly one LF."""

    try:
        _validate_json_tree(value)
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ShadowRetryCampaignValidationErrorV2(
            FAIL_CANONICAL_V2, "canonical JSON encoding failed"
        ) from error
    return (encoded + "\n").encode("ascii")


def strict_campaign_json_value_v2(payload: bytes) -> object:
    """Decode one duplicate-free exact canonical ASCII JSON value plus one LF."""

    if type(payload) is not bytes:
        _fail(FAIL_CANONICAL_V2, "strict JSON payload is not exact bytes")
    if not payload or not payload.endswith(b"\n"):
        _fail(FAIL_CANONICAL_V2, "strict JSON payload lacks its single final LF")
    duplicate = False

    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                duplicate = True
            result[key] = item
        return result

    def reject_number(_value: str) -> NoReturn:
        raise ValueError("non-integer or nonfinite number")

    try:
        decoded = json.loads(
            payload[:-1].decode("ascii"),
            object_pairs_hook=exact_object,
            parse_constant=reject_number,
            parse_float=reject_number,
        )
    except (UnicodeDecodeError, TypeError, ValueError, RecursionError) as error:
        raise ShadowRetryCampaignValidationErrorV2(
            FAIL_CANONICAL_V2, "strict JSON decoding failed"
        ) from error
    if duplicate:
        _fail(FAIL_CANONICAL_V2, "strict JSON contains a duplicate key")
    if canonical_campaign_json_bytes_v2(decoded) != payload:
        _fail(FAIL_CANONICAL_V2, "strict JSON bytes are not canonical single-LF")
    return decoded


def campaign_record_digest_v2(domain: str, value: object) -> str:
    """Return a tagged digest over an explicit NUL-terminated ASCII domain."""

    if type(domain) is not str or not domain.endswith("\x00"):
        _fail(FAIL_CANONICAL_V2, "digest domain must be NUL terminated text")
    try:
        domain_bytes = domain.encode("ascii")
    except UnicodeEncodeError as error:
        raise ShadowRetryCampaignValidationErrorV2(
            FAIL_CANONICAL_V2, "digest domain must be ASCII"
        ) from error
    return "sha256:" + hashlib.sha256(
        domain_bytes + canonical_campaign_json_bytes_v2(value)[:-1]
    ).hexdigest()


def slot_observation_record_digest_v2(value: Mapping[str, object]) -> str:
    """Derive the frozen SlotObservation record digest."""

    if type(value) is not dict or "record_digest" not in value:
        _fail(FAIL_TOP_LEVEL_V2, "slot observation digest field is absent")
    body = {key: item for key, item in value.items() if key != "record_digest"}
    return campaign_record_digest_v2(_SLOT_DIGEST_DOMAIN, body)


def failure_rows_digest_v2(value: object) -> str:
    """Bind the exact sequence-ordered retained failure-row projection."""

    return campaign_record_digest_v2(_FAILURE_ROWS_DIGEST_DOMAIN, value)


def cleanup_rows_digest_v2(value: object) -> str:
    """Bind the exact sequence-ordered retained cleanup-event projection."""

    return campaign_record_digest_v2(_CLEANUP_ROWS_DIGEST_DOMAIN, value)


def partition_certificate_digest_v2(value: Mapping[str, object]) -> str:
    """Derive the bounded post-origin partition-certificate digest."""

    if type(value) is not dict or "certificate_digest" not in value:
        _fail(FAIL_TOP_LEVEL_V2, "partition certificate digest field is absent")
    body = {
        key: item for key, item in value.items() if key != "certificate_digest"
    }
    return campaign_record_digest_v2(
        _PARTITION_CERTIFICATE_DIGEST_DOMAIN,
        body,
    )


def _integer(
    value: object,
    *,
    minimum: int = 0,
    maximum: int = _MAX_COUNTER,
) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _identifier(value: object) -> bool:
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _optional_text(value: object) -> bool:
    return value is None or (
        type(value) is str
        and 1 <= len(value.encode("utf-8")) <= 4096
        and "\x00" not in value
    )


def _raw_root(value: object) -> bool:
    return type(value) is str and _RAW_SHA256.fullmatch(value) is not None


def _absolute_path_text(value: object) -> bool:
    return (
        type(value) is str
        and value.startswith("/")
        and 1 < len(value.encode("utf-8")) <= 4096
        and "\x00" not in value
    )


def _error_wire(value: object) -> bool:
    if type(value) is not dict or frozenset(value) != _ERROR_WIRE_FIELDS:
        return False
    detail = value["detail"]
    return (
        value["schema_version"] == ERROR_WIRE_SCHEMA_V2
        and _identifier(value["status"])
        and type(detail) is str
        and 1 <= len(detail.encode("utf-8")) <= 4096
        and value["qualification_predicate_count"] == 0
        and type(value["qualification_predicate_count"]) is int
        and value["qualification_predicate_mask"] == 0
        and type(value["qualification_predicate_mask"]) is int
        and value["qualification_predicate_total"] == 20
        and type(value["qualification_predicate_total"]) is int
        and value["qualification_candidate_receipt"] is None
        and value["qualification_final_receipt"] is None
        and value["q1_state"] == "NOT_RUN"
        and value["q1_gate_count"] == 0
        and type(value["q1_gate_count"]) is int
        and value["q1_gate_mask"] == 0
        and type(value["q1_gate_mask"]) is int
        and value["q1_formal_output_roots"] == [None] * 8
        and type(value["q1_formal_output_roots"]) is list
        and value["q1_receipt"] is None
        and value["q2_state"] == "NOT_RUN"
        and value["m3_formal_roots"] is None
        and value["receipt_created"] is False
        and type(value["artifact_written"]) is bool
    )


def _final_delivery_identity(
    value: object,
    success: Mapping[str, object],
) -> bool:
    if type(value) is not dict or frozenset(value) != _FINAL_DELIVERY_IDENTITY_FIELDS:
        return False
    rows = value["ordered_stage_root_rows"]
    if type(rows) is not list or len(rows) != len(_STAGE_REGISTRY):
        return False
    stage_roots: list[str] = []
    for expected, row in zip(_STAGE_REGISTRY, rows, strict=True):
        if (
            type(row) is not list
            or len(row) != 3
            or type(row[0]) is not int
            or (row[0], row[1]) != expected
            or not _raw_root(row[2])
        ):
            return False
        stage_roots.append(row[2])
    numeric_fields = (
        "publisher_parent_device",
        "publisher_parent_inode",
        "publisher_file_device",
        "publisher_file_inode",
    )
    if not all(_integer(value[field], maximum=_MAX_UINT64) for field in numeric_fields):
        return False
    if not (
        value["schema_version"] == _ACTUAL_FINAL_DELIVERY_IDENTITY_SCHEMA
        and value["handoff_point"] == _ACTUAL_FINAL_DELIVERY_HANDOFF
        and type(value["source_commit"]) is str
        and _SOURCE_COMMIT.fullmatch(value["source_commit"]) is not None
        and value["artifact_path"] == success["artifact_path"]
        and _absolute_path_text(value["artifact_path"])
        and _raw_root(value["actual_admission_evidence_root"])
        and value["pre_artifact_live_marker_replay_root"]
        == success["actual_admission_pre_artifact_live_replay_root"]
        and value["stage_10_live_marker_replay_root"]
        == success["actual_admission_stage10_live_replay_root"]
        and value["prepublication_live_marker_replay_root"]
        == success["actual_admission_prepublication_live_replay_root"]
        and value["postpublication_live_marker_replay_root"]
        == success["actual_admission_postpublication_live_replay_root"]
        and value["artifact_set_root"] == success["artifact_set_root"]
        and value["candidate_receipt_root"] == success["candidate_receipt_root"]
        and value["final_receipt_root"] == success["final_receipt_root"]
        and value["publisher_file_mode"] == 0o444
        and type(value["publisher_file_mode"]) is int
        and value["publisher_file_nlink"] == 1
        and type(value["publisher_file_nlink"]) is int
        and _integer(value["payload_length"], minimum=1, maximum=_MAX_UINT64)
        and value["payload_sha256"] == success["artifact_sha256"]
        and _raw_root(value["payload_sha256"])
        and value["first_anchored_replay_sha256"] == value["payload_sha256"]
        and value["final_path_matches_publisher_owned_inode"] is True
        and value["same_uid_mutation_after_handoff_outside_claim"] is True
        and stage_roots == success["stage_roots"]
    ):
        return False
    expected_root = hashlib.sha256(
        _ACTUAL_FINAL_DELIVERY_ROOT_DOMAIN
        + canonical_campaign_json_bytes_v2(value)
    ).hexdigest()
    return success["actual_final_delivery_root"] == expected_root


def _success_result(value: object) -> bool:
    if type(value) is not dict or frozenset(value) != _SUCCESS_RESULT_FIELDS:
        return False
    root_fields = (
        "artifact_sha256",
        "artifact_set_root",
        "candidate_receipt_root",
        "final_receipt_root",
        "actual_admission_prepublication_live_replay_root",
        "actual_admission_pre_artifact_live_replay_root",
        "actual_admission_stage10_live_replay_root",
        "actual_admission_postpublication_live_replay_root",
        "actual_final_delivery_root",
    )
    if not all(_raw_root(value[field]) for field in root_fields):
        return False
    if not (
        _absolute_path_text(value["artifact_path"])
        and value["qualification_count"] == 20
        and type(value["qualification_count"]) is int
        and value["qualification_mask"] == 0xFFFFF
        and type(value["qualification_mask"]) is int
        and value["q1_state"] == "NOT_RUN"
        and value["q1_gate_count"] == 0
        and type(value["q1_gate_count"]) is int
        and value["q1_gate_mask"] == 0
        and type(value["q1_gate_mask"]) is int
        and value["q1_formal_output_roots"] == [None] * 8
        and type(value["q1_formal_output_roots"]) is list
        and value["publisher_handle_close_status"] == "CLOSED"
        and type(value["publisher_handle_close_status"]) is str
        and type(value["stage_roots"]) is list
        and len(value["stage_roots"]) == len(_STAGE_REGISTRY)
        and all(_raw_root(root) for root in value["stage_roots"])
    ):
        return False
    return _final_delivery_identity(value["actual_final_delivery_identity"], value)


def classify_supervisor_stdout_v2(payload: bytes) -> dict[str, object]:
    """Classify raw stdout against the two exact existing supervisor registries.

    Invalid or noncanonical bytes produce an ``INVALID`` fact row rather than
    an exception. API misuse (a non-``bytes`` argument) raises the typed codec
    error. The returned mapping has exactly the stdout fact fields consumed by
    ``SlotObservation``.
    """

    if type(payload) is not bytes:
        _fail(FAIL_CANONICAL_V2, "supervisor stdout is not exact bytes")
    facts: dict[str, object] = {
        "stdout_raw_byte_count": len(payload),
        "stdout_raw_sha256": hashlib.sha256(payload).hexdigest(),
        "stdout_canonical_single_lf": False,
        "stdout_wire_kind": STDOUT_WIRE_INVALID_V2,
        "stdout_wire_value_or_null": None,
        "stdout_status_or_null": None,
        "stdout_artifact_written_or_null": None,
    }
    if not 1 <= len(payload) <= CAPTURE_MAX_BYTES_V2:
        return facts
    try:
        value = strict_campaign_json_value_v2(payload)
    except ShadowRetryCampaignValidationErrorV2:
        return facts
    facts["stdout_canonical_single_lf"] = True
    facts["stdout_wire_value_or_null"] = value
    if _success_result(value):
        facts["stdout_wire_kind"] = STDOUT_WIRE_SUCCESS_RESULT_V2
    elif _error_wire(value):
        assert type(value) is dict
        facts["stdout_wire_kind"] = STDOUT_WIRE_ERROR_V2
        facts["stdout_status_or_null"] = value["status"]
        facts["stdout_artifact_written_or_null"] = value["artifact_written"]
    return facts


def _stdout_facts_replay_exact(value: Mapping[str, object]) -> bool:
    wire_value = value["stdout_wire_value_or_null"]
    if type(wire_value) is not dict:
        return False
    try:
        replayed = classify_supervisor_stdout_v2(
            canonical_campaign_json_bytes_v2(wire_value)
        )
    except ShadowRetryCampaignValidationErrorV2:
        return False
    return all(value[field] == observed for field, observed in replayed.items())


def _require_top_level(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _fail(FAIL_TOP_LEVEL_V2, "slot observation is not an exact mapping")
    if frozenset(value) != _TOP_FIELDS:
        _fail(FAIL_TOP_LEVEL_V2, "slot observation top-level fields differ")
    if value["schema_version"] != SLOT_OBSERVATION_SCHEMA_V2:
        _fail(FAIL_TOP_LEVEL_V2, "slot observation schema differs")
    digest = value["record_digest"]
    if type(digest) is not str or _TAGGED_SHA256.fullmatch(digest) is None:
        _fail(FAIL_TOP_LEVEL_V2, "slot observation digest is malformed")
    try:
        expected = slot_observation_record_digest_v2(value)
    except ShadowRetryCampaignValidationErrorV2 as error:
        raise _ObservationFactInvalidV2 from error
    if digest != expected:
        _fail(FAIL_TOP_LEVEL_V2, "slot observation digest differs")
    return value


def _common_exact(value: Mapping[str, object]) -> bool:
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    return (
        _identifier(value["campaign_id"])
        and _integer(
            value["slot_number"], minimum=1, maximum=MAX_CAMPAIGN_SLOTS_V2
        )
        and _identifier(value["expected_trace_id"])
        and _identifier(value["expected_producer_id"])
        and value["invocation_count"] == 1
        and type(value["invocation_count"]) is int
        and value["supervisor_returned"] is True
        and value["supervisor_reaped"] is True
        and value["process_group_absent"] is True
        and type(value["exit_code_or_null"]) is int
        and value["stdout_capture_complete"] is True
        and value["stdout_capture_bounded"] is True
        and _integer(
            value["stdout_raw_byte_count"],
            minimum=1,
            maximum=CAPTURE_MAX_BYTES_V2,
        )
        and _raw_root(value["stdout_raw_sha256"])
        and value["stdout_canonical_single_lf"] is True
        and _stdout_facts_replay_exact(value)
        and type(value["stdout_wire_kind"]) is str
        and value["stdout_wire_kind"]
        in {STDOUT_WIRE_SUCCESS_RESULT_V2, STDOUT_WIRE_ERROR_V2}
        and (
            value["stdout_status_or_null"] is None
            or _identifier(value["stdout_status_or_null"])
        )
        and (
            value["stdout_artifact_written_or_null"] is None
            or type(value["stdout_artifact_written_or_null"]) is bool
        )
        and value["stderr_capture_complete"] is True
        and value["stderr_capture_bounded"] is True
        and value["stderr_raw_byte_count"] == 0
        and type(value["stderr_raw_byte_count"]) is int
        and value["stderr_raw_sha256"] == empty_sha256
        and value["stderr_empty"] is True
        and type(value["trajectory_status"]) is str
        and value["trajectory_trace_id"] == value["expected_trace_id"]
        and type(value["trajectory_trace_id"]) is str
        and value["trajectory_producer_id"] == value["expected_producer_id"]
        and type(value["trajectory_producer_id"]) is str
        and value["trajectory_torn_tail_or_null"] is None
        and type(value["trajectory_open_span_ids"]) is list
        and value["trajectory_open_span_ids"] == []
        and (
            value["trajectory_terminal_error_code_or_null"] is None
            or _identifier(value["trajectory_terminal_error_code_or_null"])
        )
        and _integer(value["trajectory_event_count"], minimum=1)
        and type(value["trajectory_terminal_event_type_or_null"]) is str
        and value["trajectory_terminal_event_type_or_null"] in {"OK", "FAIL"}
        and type(value["trajectory_first_event_record_digest"]) is str
        and _TAGGED_SHA256.fullmatch(
            value["trajectory_first_event_record_digest"]
        )
        is not None
        and type(value["trajectory_final_event_record_digest"]) is str
        and _TAGGED_SHA256.fullmatch(
            value["trajectory_final_event_record_digest"]
        )
        is not None
        and type(value["trajectory_canonical_event_payloads_digest"]) is str
        and _TAGGED_SHA256.fullmatch(
            value["trajectory_canonical_event_payloads_digest"]
        )
        is not None
        and type(value["trajectory_failure_rows"]) is list
        and type(value["trajectory_cleanup_rows"]) is list
        and _trajectory_partition_surface_exact(value)
        and _optional_text(value["trajectory_terminal_error_detail_or_null"])
        and type(value["artifact_state"]) is str
        and type(value["artifact_validation_status"]) is str
        and type(value["artifact_replay_status"]) is str
        and value["cleanup_certain"] is True
        and value["docker_absence_certain"] is True
        and value["clone_clean_certain"] is True
        and value["cargo_isolation_certain"] is True
        and value["source_binding_certain"] is True
        and value["external_binding_certain"] is True
        and value["replay_certain"] is True
    )


def _is_success(value: Mapping[str, object]) -> bool:
    if not _common_exact(value):
        return False
    return (
        value["exit_code_or_null"] == 0
        and value["stdout_wire_kind"] == STDOUT_WIRE_SUCCESS_RESULT_V2
        and value["stdout_status_or_null"] is None
        and value["stdout_artifact_written_or_null"] is None
        and value["trajectory_status"] == "COMPLETE_OK"
        and value["trajectory_terminal_error_code_or_null"] is None
        and value["trajectory_terminal_error_detail_or_null"] is None
        and value["trajectory_terminal_event_type_or_null"] == "OK"
        and value["trajectory_partition_certificate_or_null"] is None
        and value["trajectory_failure_rows"] == []
        and value["trajectory_cleanup_rows"] == []
        and value["artifact_state"] == "VALID"
        and value["artifact_validation_status"] == "VALID"
        and value["artifact_replay_status"] == "VALID"
    )


def _failure_row_shape(row: object) -> dict[str, object] | None:
    if type(row) is not dict or frozenset(row) != _FAILURE_ROW_FIELDS:
        return None
    if not (
        _integer(row["sequence"], minimum=1)
        and _integer(row["timestamp_ns"], maximum=_MAX_UINT64)
        and _identifier(row["span_id"])
        and _identifier(row["operation"])
        and _optional_text(row["stage_or_null"])
        and _optional_text(row["substep_or_null"])
        and _optional_text(row["actor_or_null"])
        and _optional_text(row["detail_or_null"])
        and type(row["error_code"]) is str
        and _ERROR_CODE.fullmatch(row["error_code"]) is not None
        and type(row["error_detail"]) is str
        and 1 <= len(row["error_detail"].encode("utf-8")) <= 4096
        and "\x00" not in row["error_detail"]
        and type(row["record_digest"]) is str
        and _TAGGED_SHA256.fullmatch(row["record_digest"]) is not None
    ):
        return None
    return row


def _cleanup_event_row_shape(row: object) -> dict[str, object] | None:
    if type(row) is not dict or frozenset(row) != _CLEANUP_EVENT_ROW_FIELDS:
        return None
    event_type = row["event_type"]
    if not (
        _integer(row["sequence"], minimum=1)
        and _integer(row["timestamp_ns"], maximum=_MAX_UINT64)
        and type(event_type) is str
        and event_type in {"BEGIN", "OK"}
        and _identifier(row["span_id"])
        and _identifier(row["operation"])
        and _optional_text(row["stage_or_null"])
        and _optional_text(row["substep_or_null"])
        and _optional_text(row["actor_or_null"])
        and _optional_text(row["detail_or_null"])
        and type(row["record_digest"]) is str
        and _TAGGED_SHA256.fullmatch(row["record_digest"]) is not None
    ):
        return None
    if (
        row["error_code_or_null"] is not None
        or row["error_detail_or_null"] is not None
    ):
        return None
    return row


def _canonical_uint_text(value: str, *, maximum: int) -> bool:
    return (
        re.fullmatch(r"0|[1-9][0-9]*", value) is not None
        and int(value) <= maximum
    )


def _origin_gap_detail_exact(row: Mapping[str, object]) -> bool:
    """Bind the current supervisor's typed detail grammar to origin metrics."""

    actor = str(row["actor_or_null"])
    role = str(row["role_id_or_null"])
    ordinal = int(row["sample_ordinal_or_null"])
    previous = int(row["previous_finish_ns_or_null"])
    scheduled = int(row["scheduled_start_ns_or_null"])
    actual = int(row["actual_start_ns_or_null"])
    gap = int(row["gap_ns_or_null"])
    maximum = int(row["maximum_gap_ns_or_null"])
    target = int(row["target_wait_ns_or_null"])
    if row["operation"] == "LIVE_RESOURCE_SAMPLE":
        prefix = (
            "held actor sampling gap exceeded bound: "
            f"actor_id={actor}; role_id={role}; sample_ordinal={ordinal}; "
            f"previous_finish_ns={previous}; scheduled_start_ns={scheduled}; "
            f"actual_start_ns={actual}; gap_ns={gap}; "
            f"maximum_gap_ns={maximum}; target_wait_ns={target}; "
            "previous_duration_ns="
        )
    else:
        prefix = (
            "held final sampling gap exceeded bound: "
            f"actor_id={actor}; role_id={role}; sample_ordinal={ordinal}; "
            f"previous_finish_ns={previous}; actual_start_ns={actual}; "
            f"gap_ns={gap}; maximum_gap_ns={maximum}; "
            f"target_wait_ns={target}; previous_duration_ns="
        )
    detail = row["error_detail"]
    if type(detail) is not str or not detail.startswith(prefix):
        return False
    previous_duration_text = detail[len(prefix) :]
    return _canonical_uint_text(previous_duration_text, maximum=_MAX_UINT64)


def _origin_gap_row(row: Mapping[str, object]) -> bool:
    if row["operation"] not in _SAMPLE_OPERATIONS:
        return False
    role = row["role_id_or_null"]
    if (
        type(role) is not str
        or role not in _ROLE_ACTOR
        or row["actor_or_null"] != _ROLE_ACTOR[role]
    ):
        return False
    if not (
        _integer(row["sample_ordinal_or_null"], minimum=1)
        and _integer(row["previous_finish_ns_or_null"], maximum=_MAX_UINT64)
        and _integer(row["scheduled_start_ns_or_null"], maximum=_MAX_UINT64)
        and _integer(row["actual_start_ns_or_null"], maximum=_MAX_UINT64)
        and _integer(row["finish_ns_or_null"], maximum=_MAX_UINT64)
        and _integer(row["collection_duration_ns_or_null"], maximum=_MAX_UINT64)
        and _integer(row["scheduler_lateness_ns_or_null"], maximum=_MAX_UINT64)
        and _integer(
            row["target_wait_ns_or_null"], minimum=1, maximum=_MAX_UINT64
        )
        and _integer(row["gap_ns_or_null"], maximum=_MAX_UINT64)
        and _integer(
            row["maximum_gap_ns_or_null"], minimum=1, maximum=_MAX_UINT64
        )
        and row["gap_exceeded_or_null"] is True
    ):
        return False
    previous = int(row["previous_finish_ns_or_null"])
    scheduled = int(row["scheduled_start_ns_or_null"])
    actual = int(row["actual_start_ns_or_null"])
    finish = int(row["finish_ns_or_null"])
    duration = int(row["collection_duration_ns_or_null"])
    lateness = int(row["scheduler_lateness_ns_or_null"])
    target = int(row["target_wait_ns_or_null"])
    gap = int(row["gap_ns_or_null"])
    maximum = int(row["maximum_gap_ns_or_null"])
    return (
        target == TARGET_WAIT_NS_V2
        and maximum == MAXIMUM_GAP_NS_V2
        and previous <= _MAX_UINT64 - target
        and scheduled == previous + target
        and actual >= scheduled
        and lateness == actual - scheduled
        and gap == actual - previous
        and gap > maximum
        and finish >= actual
        and duration == finish - actual
        and finish == actual
        and duration == 0
        and int(row["timestamp_ns"]) >= finish
        and row["stage_or_null"] == "ACTOR_LIFECYCLE"
        and row["substep_or_null"]
        == (
            "COLLECT_BOUND_LIVE_RESOURCES"
            if row["operation"] == "LIVE_RESOURCE_SAMPLE"
            else "COLLECT_BOUND_HELD_FINAL_RESOURCES"
        )
        and row["detail_or_null"] == "MAXIMUM_UNSAMPLED_GAP_EXCEEDED"
        and row["span_id"]
        == (
            "sample-"
            if row["operation"] == "LIVE_RESOURCE_SAMPLE"
            else "held-final-sample-"
        )
        + str(row["actor_or_null"])
        + f"-{int(row['sample_ordinal_or_null']):06d}"
        and _origin_gap_detail_exact(row)
    )


def _propagation_contexts(
    origin: Mapping[str, object],
) -> tuple[tuple[str, str, str | None, str | None, str | None, str], ...]:
    actor = str(origin["actor_or_null"])
    contexts: list[tuple[str, str, str | None, str | None, str | None, str]] = []
    if origin["operation"] == "LIVE_RESOURCE_SAMPLE":
        contexts.append(
            (
                "LIVE_RESOURCE_SAMPLER",
                f"sampler-{actor}",
                "ACTOR_LIFECYCLE",
                "CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
                actor,
                "MAXIMUM_UNSAMPLED_GAP_EXCEEDED; sample_span="
                + str(origin["span_id"]),
            )
        )
    contexts.append(
        (
            "HELD_ACTOR_CLOSE",
            f"actor-close-{actor}",
            "ACTOR_LIFECYCLE",
            "DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            actor,
            "ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
        )
    )
    stage_id = 6 if actor == "TRUSTED_HOST_REPLAY" else 5
    contexts.extend(
        (
            (
                "SUPERVISOR_STAGE",
                f"supervisor-stage-{stage_id:02d}",
                f"STAGE_{stage_id:02d}",
                _STAGE_REGISTRY[stage_id - 1][1],
                None,
                "FAILED",
            ),
            (
                "SUPERVISOR_RUN",
                "supervisor-run",
                "RUN",
                None,
                None,
                "supervisor attempt failed",
            ),
        )
    )
    return tuple(contexts)


def _propagation_context_alternatives(
    origin: Mapping[str, object],
) -> tuple[
    tuple[tuple[str, str, str | None, str | None, str | None, str], ...],
    ...,
]:
    """Return only the supervisor propagation paths this origin can produce."""

    ordinary = _propagation_contexts(origin)
    if (
        origin["operation"] != "LIVE_RESOURCE_SAMPLE"
        or origin["actor_or_null"]
        not in {"PYTHON_ENDPOINT", "RUST_ENDPOINT"}
    ):
        return (ordinary,)
    sampler = ordinary[0]
    supervisor_run = ordinary[-1]
    stage_4_early_exit = (
        sampler,
        (
            "SUPERVISOR_STAGE",
            "supervisor-stage-04",
            "STAGE_04",
            _STAGE_REGISTRY[3][1],
            None,
            "FAILED",
        ),
        supervisor_run,
    )
    return (ordinary, stage_4_early_exit)


def _terminal_gap_detail(origin: Mapping[str, object]) -> str:
    raw_detail = str(origin["error_detail"])
    if origin["operation"] == "LIVE_RESOURCE_SAMPLE":
        return "held actor sampler failed: " + raw_detail
    return raw_detail


def _propagation_row_exact(
    row: Mapping[str, object],
    context: tuple[str, str, str | None, str | None, str | None, str],
    origin: Mapping[str, object],
) -> bool:
    operation, span_id, stage, substep, actor, detail = context
    expected_error_detail = (
        origin["error_detail"]
        if operation == "LIVE_RESOURCE_SAMPLER"
        else _terminal_gap_detail(origin)
    )
    return (
        row["operation"] == operation
        and row["span_id"] == span_id
        and row["stage_or_null"] == stage
        and row["substep_or_null"] == substep
        and row["actor_or_null"] == actor
        and row["detail_or_null"] == detail
        and row["error_detail"] == expected_error_detail
        and all(row[field] is None for field in _SAMPLE_FIELDS)
    )


def _canonical_padded_counter_text(value: object) -> int | None:
    if type(value) is not str or re.fullmatch(r"[0-9]{6,}", value) is None:
        return None
    counter = int(value)
    if (
        not 1 <= counter <= _MAX_COUNTER
        or value != f"{counter:06d}"
    ):
        return None
    return counter


def _registered_cleanup_begin(
    event: Mapping[str, object],
) -> tuple[str, int] | None:
    actor = event["actor_or_null"]
    span_id = event["span_id"]
    if type(actor) is not str or actor not in _ROLE_ACTOR.values():
        return None
    prefix = f"actor-cleanup-{actor}-"
    if type(span_id) is not str or not span_id.startswith(prefix):
        return None
    suffix = _canonical_padded_counter_text(span_id[len(prefix) :])
    if suffix is None:
        return None
    if not (
        event["event_type"] == "BEGIN"
        and event["operation"] == "HELD_ACTOR_ABORT_CLEANUP"
        and event["stage_or_null"] == "ACTOR_LIFECYCLE"
        and event["substep_or_null"]
        == "STOP_TERMINATE_DRAIN_REMOVE_ABSENCE"
        and event["detail_or_null"] == "ENTER"
        and event["error_code_or_null"] is None
        and event["error_detail_or_null"] is None
    ):
        return None
    return actor, suffix


def _registered_cleanup_ok(
    begin: Mapping[str, object],
    end: Mapping[str, object],
) -> bool:
    return (
        end["event_type"] == "OK"
        and end["span_id"] == begin["span_id"]
        and end["operation"] == begin["operation"]
        and end["stage_or_null"] == begin["stage_or_null"]
        and end["substep_or_null"] == begin["substep_or_null"]
        and end["actor_or_null"] == begin["actor_or_null"]
        and end["detail_or_null"] == "cleanup_error_count=0"
        and end["error_code_or_null"] is None
        and end["error_detail_or_null"] is None
    )


def _partition_certificate_shape(
    value: object,
) -> dict[str, object] | None:
    if (
        type(value) is not dict
        or frozenset(value) != _PARTITION_CERTIFICATE_FIELDS
        or value.get("schema_version") != PARTITION_CERTIFICATE_SCHEMA_V2
    ):
        return None
    if not (
        _identifier(value["trace_id"])
        and _identifier(value["producer_id"])
        and value["trajectory_status"] == "COMPLETE_FAIL"
        and _integer(value["event_count"], minimum=1)
        and type(value["first_event_record_digest"]) is str
        and _TAGGED_SHA256.fullmatch(value["first_event_record_digest"])
        is not None
        and type(value["final_event_record_digest"]) is str
        and _TAGGED_SHA256.fullmatch(value["final_event_record_digest"])
        is not None
        and type(value["canonical_event_payloads_digest"]) is str
        and _TAGGED_SHA256.fullmatch(value["canonical_event_payloads_digest"])
        is not None
        and _integer(
            value["origin_sequence"],
            minimum=1,
            maximum=int(value["event_count"]),
        )
        and type(value["origin_record_digest"]) is str
        and _TAGGED_SHA256.fullmatch(value["origin_record_digest"])
        is not None
        and _optional_text(value["origin_actor_or_null"])
        and _integer(value["suffix_event_count"], minimum=1)
        and type(value["suffix_event_root"]) is str
        and _TAGGED_SHA256.fullmatch(value["suffix_event_root"]) is not None
        and _integer(value["failure_event_count"], minimum=1)
        and type(value["failure_rows_digest"]) is str
        and _TAGGED_SHA256.fullmatch(value["failure_rows_digest"]) is not None
        and _integer(
            value["failure_rows_retained_count"],
            maximum=_MAX_RETAINED_FAILURE_ROWS,
        )
        and type(value["failure_rows_truncated"]) is bool
        and _integer(value["cleanup_nonfailure_event_count"])
        and type(value["cleanup_rows_digest"]) is str
        and _TAGGED_SHA256.fullmatch(value["cleanup_rows_digest"]) is not None
        and _integer(
            value["cleanup_rows_retained_count"],
            maximum=_MAX_RETAINED_CLEANUP_ROWS,
        )
        and type(value["cleanup_rows_truncated"]) is bool
        and _integer(value["other_nonfailure_event_count"])
        and type(value["other_nonfailure_event_root"]) is str
        and _TAGGED_SHA256.fullmatch(value["other_nonfailure_event_root"])
        is not None
        and _integer(value["partitioned_event_count"], minimum=1)
        and _integer(value["unpartitioned_event_count"])
        and type(value["certificate_digest"]) is str
        and _TAGGED_SHA256.fullmatch(value["certificate_digest"]) is not None
    ):
        return None
    try:
        if (
            len(canonical_campaign_json_bytes_v2(value))
            > _MAX_PARTITION_CERTIFICATE_BYTES
            or value["certificate_digest"]
            != partition_certificate_digest_v2(value)
        ):
            return None
    except ShadowRetryCampaignValidationErrorV2:
        return None
    event_count = int(value["event_count"])
    origin_sequence = int(value["origin_sequence"])
    failure_count = int(value["failure_event_count"])
    cleanup_count = int(value["cleanup_nonfailure_event_count"])
    other_count = int(value["other_nonfailure_event_count"])
    partitioned = int(value["partitioned_event_count"])
    unpartitioned = int(value["unpartitioned_event_count"])
    return value if (
        int(value["suffix_event_count"])
        == event_count - origin_sequence + 1
        and partitioned == failure_count + cleanup_count + other_count
        and partitioned + unpartitioned == int(value["suffix_event_count"])
        and int(value["failure_rows_retained_count"])
        <= failure_count
        and int(value["cleanup_rows_retained_count"])
        <= cleanup_count
        and (
            value["failure_rows_truncated"]
            is (int(value["failure_rows_retained_count"]) < failure_count)
        )
        and (
            value["cleanup_rows_truncated"]
            is (int(value["cleanup_rows_retained_count"]) < cleanup_count)
        )
    ) else None


def _trajectory_partition_surface_exact(value: Mapping[str, object]) -> bool:
    status = value["trajectory_status"]
    terminal_type = value["trajectory_terminal_event_type_or_null"]
    failures = value["trajectory_failure_rows"]
    cleanups = value["trajectory_cleanup_rows"]
    certificate_value = value["trajectory_partition_certificate_or_null"]
    if status == "COMPLETE_OK":
        return (
            terminal_type == "OK"
            and certificate_value is None
            and failures == []
            and cleanups == []
        )
    if status != "COMPLETE_FAIL" or terminal_type != "FAIL":
        return False
    certificate = _partition_certificate_shape(certificate_value)
    if certificate is None:
        return False
    if not (
        certificate["trace_id"] == value["trajectory_trace_id"]
        and certificate["producer_id"] == value["trajectory_producer_id"]
        and certificate["event_count"] == value["trajectory_event_count"]
        and certificate["first_event_record_digest"]
        == value["trajectory_first_event_record_digest"]
        and certificate["final_event_record_digest"]
        == value["trajectory_final_event_record_digest"]
        and certificate["canonical_event_payloads_digest"]
        == value["trajectory_canonical_event_payloads_digest"]
        and certificate["failure_rows_retained_count"] == len(failures)
        and certificate["cleanup_rows_retained_count"] == len(cleanups)
        and len(failures) <= _MAX_RETAINED_FAILURE_ROWS
        and len(cleanups) <= _MAX_RETAINED_CLEANUP_ROWS
        and all(_failure_row_shape(row) is not None for row in failures)
        and all(_cleanup_event_row_shape(row) is not None for row in cleanups)
    ):
        return False
    if certificate["failure_rows_truncated"] is False:
        try:
            if certificate["failure_rows_digest"] != failure_rows_digest_v2(
                failures
            ):
                return False
        except ShadowRetryCampaignValidationErrorV2:
            return False
    if certificate["cleanup_rows_truncated"] is False:
        try:
            if certificate["cleanup_rows_digest"] != cleanup_rows_digest_v2(
                cleanups
            ):
                return False
        except ShadowRetryCampaignValidationErrorV2:
            return False
    return True


def _exact_cleanup_grammar(
    value: object,
    *,
    origin_sequence: int,
    required_origin_actors: frozenset[str],
    allowed_cleanup_actors: frozenset[str],
) -> bool:
    if (
        type(value) is not list
        or len(value) > _MAX_RETAINED_CLEANUP_ROWS
        or not required_origin_actors
        or not required_origin_actors <= allowed_cleanup_actors
        or not allowed_cleanup_actors <= frozenset(_ROLE_ACTOR.values())
    ):
        return False
    prior_sequence = origin_sequence
    prior_timestamp = 0
    last_suffix = 0
    record_digests: set[str] = set()
    active: dict[str, Mapping[str, object]] = {}
    begin_counts = {actor: 0 for actor in _ROLE_ACTOR.values()}
    success_counts = {actor: 0 for actor in _ROLE_ACTOR.values()}
    for raw_event in value:
        event = _cleanup_event_row_shape(raw_event)
        if event is None or event["operation"] != "HELD_ACTOR_ABORT_CLEANUP":
            return False
        sequence = int(event["sequence"])
        timestamp = int(event["timestamp_ns"])
        digest = str(event["record_digest"])
        if (
            sequence <= prior_sequence
            or timestamp < prior_timestamp
            or digest in record_digests
        ):
            return False
        prior_sequence = sequence
        prior_timestamp = timestamp
        record_digests.add(digest)
        span_id = str(event["span_id"])
        if event["event_type"] == "BEGIN":
            registered = _registered_cleanup_begin(event)
            if registered is None:
                return False
            actor, suffix = registered
            if (
                actor not in allowed_cleanup_actors
                or suffix <= last_suffix
                or begin_counts[actor]
                >= _MAX_REGISTERED_CLEANUP_PAIRS_PER_ACTOR
                or span_id in active
            ):
                return False
            last_suffix = suffix
            begin_counts[actor] += 1
            active[span_id] = event
            continue
        begin = active.get(span_id)
        if begin is None or not _registered_cleanup_ok(begin, event):
            return False
        actor = str(begin["actor_or_null"])
        success_counts[actor] += 1
        active.pop(span_id)
    return (
        not active
        and all(success_counts[actor] >= 1 for actor in required_origin_actors)
    )


def _partition_certificate_retry_exact(
    value: Mapping[str, object],
    *,
    required_origin_actors: frozenset[str],
    allowed_cleanup_actors: frozenset[str],
) -> bool:
    certificate = _partition_certificate_shape(
        value["trajectory_partition_certificate_or_null"]
    )
    failures = value["trajectory_failure_rows"]
    cleanups = value["trajectory_cleanup_rows"]
    if certificate is None or type(failures) is not list or type(cleanups) is not list:
        return False
    if not (
        certificate["failure_rows_truncated"] is False
        and certificate["cleanup_rows_truncated"] is False
        and certificate["unpartitioned_event_count"] == 0
        and 1 <= certificate["failure_event_count"] <= _MAX_RETAINED_FAILURE_ROWS
        and certificate["failure_event_count"] == len(failures)
        and certificate["cleanup_nonfailure_event_count"] == len(cleanups)
        and certificate["failure_rows_digest"] == failure_rows_digest_v2(failures)
        and certificate["cleanup_rows_digest"] == cleanup_rows_digest_v2(cleanups)
        and failures
    ):
        return False
    origin = failures[0]
    final = failures[-1]
    if type(origin) is not dict or type(final) is not dict:
        return False
    return (
        certificate["origin_sequence"] == origin.get("sequence")
        and certificate["origin_record_digest"] == origin.get("record_digest")
        and certificate["origin_actor_or_null"] == origin.get("actor_or_null")
        and certificate["final_event_record_digest"]
        == final.get("record_digest")
        and final.get("sequence") == value["trajectory_event_count"]
        and _exact_cleanup_grammar(
            cleanups,
            origin_sequence=int(certificate["origin_sequence"]),
            required_origin_actors=required_origin_actors,
            allowed_cleanup_actors=allowed_cleanup_actors,
        )
    )


def _exact_gap_failure_grammar(
    value: object,
) -> tuple[dict[str, object], frozenset[str], frozenset[str]] | None:
    if (
        type(value) is not list
        or not value
        or len(value) > _MAX_RETAINED_FAILURE_ROWS
    ):
        return None
    prior_sequence = 0
    prior_timestamp = 0
    record_digests: set[str] = set()
    rows: list[dict[str, object]] = []
    for raw_row in value:
        row = _failure_row_shape(raw_row)
        if row is None:
            return None
        rows.append(row)
        sequence = int(row["sequence"])
        if sequence <= prior_sequence:
            return None
        prior_sequence = sequence
        timestamp = int(row["timestamp_ns"])
        if timestamp < prior_timestamp or row["record_digest"] in record_digests:
            return None
        prior_timestamp = timestamp
        record_digests.add(str(row["record_digest"]))
        if row["error_code"] != GAP_ERROR_CODE_V2:
            return None

    origins = [row for row in rows if _origin_gap_row(row)]
    if not origins or rows[0] is not origins[0]:
        return None

    if len(origins) == 1:
        origin = origins[0]
        for contexts in _propagation_context_alternatives(origin):
            if len(rows) != 1 + len(contexts):
                continue
            if all(
                _propagation_row_exact(row, context, origin)
                for row, context in zip(rows[1:], contexts, strict=True)
            ):
                actor = str(origin["actor_or_null"])
                allowed_cleanup_actors = (
                    frozenset({"TRUSTED_HOST_REPLAY"})
                    if actor == "TRUSTED_HOST_REPLAY"
                    else frozenset({"PYTHON_ENDPOINT", "RUST_ENDPOINT"})
                )
                return origin, frozenset({actor}), allowed_cleanup_actors
        return None

    # The frozen producer has only one multi-origin failure surface.  During
    # Stage 5 it closes PYTHON_ENDPOINT first while RUST_ENDPOINT's independent
    # live sampler remains running.  The peer sampler can therefore fail while
    # Python is propagating either its live-sampler gap or its held-final gap.
    # No trusted-host, dual-held-final, reverse-held-final, Stage-4, or second
    # actor-close combination is reachable from that fixed control flow.
    if len(origins) != 2:
        return None
    origins_by_actor = {str(origin["actor_or_null"]): origin for origin in origins}
    if (
        len(origins_by_actor) != 2
        or frozenset(origins_by_actor)
        != frozenset({"PYTHON_ENDPOINT", "RUST_ENDPOINT"})
    ):
        return None
    python_origin = origins_by_actor["PYTHON_ENDPOINT"]
    rust_origin = origins_by_actor["RUST_ENDPOINT"]
    if (
        python_origin["operation"]
        not in {"LIVE_RESOURCE_SAMPLE", "HELD_FINAL_RESOURCE_SAMPLE"}
        or rust_origin["operation"] != "LIVE_RESOURCE_SAMPLE"
    ):
        return None

    python_contexts = _propagation_contexts(python_origin)
    rust_contexts = _propagation_contexts(rust_origin)
    expected: dict[str, tuple[object, ...]] = {
        "PYTHON_CLOSE": (python_contexts[-3], python_origin),
        "RUST_SAMPLER": (rust_contexts[0], rust_origin),
        "STAGE_5": (python_contexts[-2], python_origin),
        "RUN": (python_contexts[-1], python_origin),
    }
    if python_origin["operation"] == "LIVE_RESOURCE_SAMPLE":
        expected["PYTHON_SAMPLER"] = (python_contexts[0], python_origin)

    origin_identities = {id(origin) for origin in origins}
    nonorigins = [row for row in rows if id(row) not in origin_identities]
    if len(nonorigins) != len(expected):
        return None
    positions: dict[str, int] = {}
    for label, (raw_context, raw_origin) in expected.items():
        context = raw_context
        origin = raw_origin
        assert type(context) is tuple and type(origin) is dict
        matches = [
            index
            for index, row in enumerate(rows)
            if _propagation_row_exact(row, context, origin)
        ]
        if len(matches) != 1:
            return None
        positions[label] = matches[0]
    if len(set(positions.values())) != len(positions):
        return None

    origin_positions = {
        actor: rows.index(origin) for actor, origin in origins_by_actor.items()
    }
    if not (
        origin_positions["RUST_ENDPOINT"] < positions["RUST_SAMPLER"]
        < positions["STAGE_5"]
        < positions["RUN"]
        and origin_positions["PYTHON_ENDPOINT"] < positions["PYTHON_CLOSE"]
        < positions["STAGE_5"]
    ):
        return None
    if (
        python_origin["operation"] == "LIVE_RESOURCE_SAMPLE"
        and not (
            origin_positions["PYTHON_ENDPOINT"] < positions["PYTHON_SAMPLER"]
            < positions["PYTHON_CLOSE"]
        )
    ):
        return None
    return (
        python_origin,
        frozenset({"PYTHON_ENDPOINT", "RUST_ENDPOINT"}),
        frozenset({"PYTHON_ENDPOINT", "RUST_ENDPOINT"}),
    )


def _is_exact_retryable_gap(value: Mapping[str, object]) -> bool:
    if not _common_exact(value):
        return False
    wire = value["stdout_wire_value_or_null"]
    rows = value["trajectory_failure_rows"]
    if (
        type(wire) is not dict
        or type(rows) is not list
        or not rows
    ):
        return False
    wire_detail = wire.get("detail")
    grammar = _exact_gap_failure_grammar(rows)
    if grammar is None:
        return False
    terminal_origin, required_origin_actors, allowed_cleanup_actors = grammar
    terminal_detail = _terminal_gap_detail(terminal_origin)
    return (
        value["exit_code_or_null"] == 1
        and value["stdout_wire_kind"] == STDOUT_WIRE_ERROR_V2
        and value["stdout_status_or_null"] == GAP_ERROR_CODE_V2
        and value["stdout_artifact_written_or_null"] is False
        and value["trajectory_status"] == "COMPLETE_FAIL"
        and value["trajectory_terminal_error_code_or_null"] == GAP_ERROR_CODE_V2
        and value["trajectory_terminal_event_type_or_null"] == "FAIL"
        and type(wire_detail) is str
        and wire_detail == terminal_detail
        and value["trajectory_terminal_error_detail_or_null"] == wire_detail
        and _partition_certificate_retry_exact(
            value,
            required_origin_actors=required_origin_actors,
            allowed_cleanup_actors=allowed_cleanup_actors,
        )
        and value["artifact_state"] == "ABSENT"
        and value["artifact_validation_status"] == "ABSENCE_VERIFIED"
        and value["artifact_replay_status"] == "ABSENCE_REPLAY_VERIFIED"
    )


def evaluate_slot_observation_v2(value: object) -> str:
    """Classify one exact digest-bound observation without taking any action."""

    try:
        observation = _require_top_level(value)
    except _ObservationFactInvalidV2:
        return TERMINAL_NONRETRYABLE_V2
    if _is_success(observation):
        return SUCCESS_V2
    if _is_exact_retryable_gap(observation):
        slot = int(observation["slot_number"])
        if slot < MAX_CAMPAIGN_SLOTS_V2:
            return RETRY_EXACT_GAP_V2
        return EXHAUSTED_RETRYABLE_GAP_V2
    return TERMINAL_NONRETRYABLE_V2

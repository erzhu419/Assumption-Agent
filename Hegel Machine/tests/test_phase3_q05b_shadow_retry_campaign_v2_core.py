from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from hegel_machine import phase3_q05b_shadow_retry_campaign_v2 as campaign


def _root(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _wire(value: object) -> bytes:
    return campaign.canonical_campaign_json_bytes_v2(value)


def _continuous_gap_detail() -> str:
    return (
        "held actor sampling gap exceeded bound: "
        "actor_id=RUST_ENDPOINT; role_id=2; sample_ordinal=1; "
        "previous_finish_ns=1000000000; scheduled_start_ns=1125000000; "
        "actual_start_ns=1250000001; gap_ns=250000001; "
        "maximum_gap_ns=250000000; target_wait_ns=125000000; "
        "previous_duration_ns=1000000"
    )


def _error_wire(
    *,
    status: str = campaign.GAP_ERROR_CODE_V2,
    artifact_written: bool = False,
    detail: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": campaign.ERROR_WIRE_SCHEMA_V2,
        "status": status,
        "detail": _continuous_gap_detail() if detail is None else detail,
        "qualification_predicate_count": 0,
        "qualification_predicate_mask": 0,
        "qualification_predicate_total": 20,
        "qualification_candidate_receipt": None,
        "qualification_final_receipt": None,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "m3_formal_roots": None,
        "receipt_created": False,
        "artifact_written": artifact_written,
    }


_STAGES = (
    "FREEZE_SOURCE_IMAGES_AND_COMMAND_REGISTRY",
    "MATERIALIZE_AND_SEAL_THREE_SOURCE_SNAPSHOTS_AND_CARGO",
    "OFFLINE_RUST_TEST_AND_RELEASE_BUILD_AND_SEAL_BINARY",
    "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL",
    "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
    "RUN_HELD_TRUSTED_HOST_SEMANTIC_REPLAY",
    "CLOSE_HOST_RESOURCE_ISOLATION_NEGATIVE_VECTORS_AND_TOCTOU",
    "BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE",
    "ADD_PREDICATE20_AND_BUILD_CANONICAL_ARTIFACT_IN_MEMORY",
    "STRICTLY_REPLAY_COMPLETE_ARTIFACT_BEFORE_PUBLICATION",
)


def _success_wire() -> dict[str, object]:
    stage_roots = [_root(f"stage-{index}") for index in range(1, 11)]
    result: dict[str, object] = {
        "artifact_path": "/fixed/slot/artifacts/phase3_q05b_dual_qualification_v2.json",
        "artifact_sha256": _root("artifact"),
        "artifact_set_root": _root("artifact-set"),
        "candidate_receipt_root": _root("candidate-receipt"),
        "final_receipt_root": _root("final-receipt"),
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "actual_admission_prepublication_live_replay_root": _root(
            "prepublication-live"
        ),
        "actual_admission_pre_artifact_live_replay_root": _root(
            "pre-artifact-live"
        ),
        "actual_admission_stage10_live_replay_root": _root("stage10-live"),
        "actual_admission_postpublication_live_replay_root": _root(
            "postpublication-live"
        ),
        "actual_final_delivery_identity": None,
        "actual_final_delivery_root": "0" * 64,
        "publisher_handle_close_status": "CLOSED",
        "stage_roots": stage_roots,
    }
    identity = {
        "schema_version": "hegel-phase3a-q05b-final-delivery-identity/2",
        "handoff_point": (
            "FINAL_ANCHORED_REPLAY_AFTER_POSTPUBLICATION_LIVE_"
            "BEFORE_HANDLE_CLOSE_AND_RETURN"
        ),
        "source_commit": "12" * 20,
        "artifact_path": result["artifact_path"],
        "ordered_stage_root_rows": [
            [index, name, stage_roots[index - 1]]
            for index, name in enumerate(_STAGES, start=1)
        ],
        "actual_admission_evidence_root": _root("admission"),
        "pre_artifact_live_marker_replay_root": result[
            "actual_admission_pre_artifact_live_replay_root"
        ],
        "stage_10_live_marker_replay_root": result[
            "actual_admission_stage10_live_replay_root"
        ],
        "prepublication_live_marker_replay_root": result[
            "actual_admission_prepublication_live_replay_root"
        ],
        "postpublication_live_marker_replay_root": result[
            "actual_admission_postpublication_live_replay_root"
        ],
        "artifact_set_root": result["artifact_set_root"],
        "candidate_receipt_root": result["candidate_receipt_root"],
        "final_receipt_root": result["final_receipt_root"],
        "publisher_parent_device": 2049,
        "publisher_parent_inode": 100,
        "publisher_file_device": 2049,
        "publisher_file_inode": 101,
        "publisher_file_mode": 0o444,
        "publisher_file_nlink": 1,
        "payload_length": 4096,
        "payload_sha256": result["artifact_sha256"],
        "first_anchored_replay_sha256": result["artifact_sha256"],
        "final_path_matches_publisher_owned_inode": True,
        "same_uid_mutation_after_handoff_outside_claim": True,
    }
    result["actual_final_delivery_identity"] = identity
    result["actual_final_delivery_root"] = hashlib.sha256(
        b"HEGEL/Q05B/ACTUAL/FINAL_DELIVERY/V1\x00" + _wire(identity)
    ).hexdigest()
    return result


def _failure_row(*, origin: bool, sequence: int = 6) -> dict[str, object]:
    row: dict[str, object] = {
        "sequence": sequence,
        "timestamp_ns": 2_000_000_000 + sequence,
        "span_id": (
            "sample-RUST_ENDPOINT-000001"
            if origin
            else f"propagated-{sequence}"
        ),
        "operation": "LIVE_RESOURCE_SAMPLE" if origin else "LIVE_RESOURCE_SAMPLER",
        "stage_or_null": "ACTOR_LIFECYCLE",
        "substep_or_null": (
            "COLLECT_BOUND_LIVE_RESOURCES" if origin else "continuous-sampling"
        ),
        "actor_or_null": "RUST_ENDPOINT",
        "detail_or_null": (
            "MAXIMUM_UNSAMPLED_GAP_EXCEEDED"
            if origin
            else "propagated gap failure"
        ),
        "error_code": campaign.GAP_ERROR_CODE_V2,
        "error_detail": _continuous_gap_detail(),
        "record_digest": "sha256:" + _root(f"event-{sequence}"),
        "role_id_or_null": None,
        "sample_ordinal_or_null": None,
        "previous_finish_ns_or_null": None,
        "scheduled_start_ns_or_null": None,
        "actual_start_ns_or_null": None,
        "finish_ns_or_null": None,
        "collection_duration_ns_or_null": None,
        "scheduler_lateness_ns_or_null": None,
        "target_wait_ns_or_null": None,
        "gap_ns_or_null": None,
        "maximum_gap_ns_or_null": None,
        "gap_exceeded_or_null": None,
    }
    if origin:
        previous = 1_000_000_000
        scheduled = previous + campaign.TARGET_WAIT_NS_V2
        actual = previous + campaign.MAXIMUM_GAP_NS_V2 + 1
        finish = actual
        row.update(
            {
                "role_id_or_null": "2",
                "sample_ordinal_or_null": 1,
                "previous_finish_ns_or_null": previous,
                "scheduled_start_ns_or_null": scheduled,
                "actual_start_ns_or_null": actual,
                "finish_ns_or_null": finish,
                "collection_duration_ns_or_null": 0,
                "scheduler_lateness_ns_or_null": actual - scheduled,
                "target_wait_ns_or_null": campaign.TARGET_WAIT_NS_V2,
                "gap_ns_or_null": actual - previous,
                "maximum_gap_ns_or_null": campaign.MAXIMUM_GAP_NS_V2,
                "gap_exceeded_or_null": True,
            }
        )
    return row


def _failure_rows() -> list[dict[str, object]]:
    origin = _failure_row(origin=True, sequence=6)
    detail = origin["error_detail"]
    outer_detail = "held actor sampler failed: " + detail

    def propagated(
        sequence: int,
        operation: str,
        span_id: str,
        stage: str,
        substep: str | None,
        actor: str | None,
        event_detail: str,
        error_detail: str = detail,
    ) -> dict[str, object]:
        row = _failure_row(origin=False, sequence=sequence)
        row.update(
            {
                "operation": operation,
                "span_id": span_id,
                "stage_or_null": stage,
                "substep_or_null": substep,
                "actor_or_null": actor,
                "detail_or_null": event_detail,
                "error_detail": error_detail,
            }
        )
        return row

    return [
        origin,
        propagated(
            7,
            "LIVE_RESOURCE_SAMPLER",
            "sampler-RUST_ENDPOINT",
            "ACTOR_LIFECYCLE",
            "CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            "RUST_ENDPOINT",
            "MAXIMUM_UNSAMPLED_GAP_EXCEEDED; sample_span=sample-RUST_ENDPOINT-000001",
        ),
        propagated(
            8,
            "HELD_ACTOR_CLOSE",
            "actor-close-RUST_ENDPOINT",
            "ACTOR_LIFECYCLE",
            "DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            "RUST_ENDPOINT",
            "ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
            outer_detail,
        ),
        propagated(
            11,
            "SUPERVISOR_STAGE",
            "supervisor-stage-05",
            "STAGE_05",
            "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
            None,
            "FAILED",
            outer_detail,
        ),
        propagated(
            12,
            "SUPERVISOR_RUN",
            "supervisor-run",
            "RUN",
            None,
            None,
            "supervisor attempt failed",
            outer_detail,
        ),
    ]


def _early_stage4_failure_rows() -> list[dict[str, object]]:
    origin, sampler, _actor_close, stage, supervisor_run = deepcopy(
        _failure_rows()
    )
    raw_detail = str(origin["error_detail"]).replace(
        "actor_id=RUST_ENDPOINT; role_id=2",
        "actor_id=PYTHON_ENDPOINT; role_id=1",
    )
    outer_detail = "held actor sampler failed: " + raw_detail
    origin.update(
        {
            "sequence": 5,
            "timestamp_ns": 2_000_000_005,
            "span_id": "sample-PYTHON_ENDPOINT-000001",
            "actor_or_null": "PYTHON_ENDPOINT",
            "error_detail": raw_detail,
            "role_id_or_null": "1",
            "record_digest": "sha256:" + _root("stage4-event-5"),
        }
    )
    sampler.update(
        {
            "sequence": 6,
            "timestamp_ns": 2_000_000_006,
            "span_id": "sampler-PYTHON_ENDPOINT",
            "actor_or_null": "PYTHON_ENDPOINT",
            "detail_or_null": (
                "MAXIMUM_UNSAMPLED_GAP_EXCEEDED; "
                "sample_span=sample-PYTHON_ENDPOINT-000001"
            ),
            "error_detail": raw_detail,
            "record_digest": "sha256:" + _root("stage4-event-6"),
        }
    )
    stage.update(
        {
            "sequence": 9,
            "timestamp_ns": 2_000_000_009,
            "span_id": "supervisor-stage-04",
            "stage_or_null": "STAGE_04",
            "substep_or_null": "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL",
            "error_detail": outer_detail,
            "record_digest": "sha256:" + _root("stage4-event-9"),
        }
    )
    supervisor_run.update(
        {
            "sequence": 10,
            "timestamp_ns": 2_000_000_010,
            "error_detail": outer_detail,
            "record_digest": "sha256:" + _root("stage4-event-10"),
        }
    )
    return [origin, sampler, stage, supervisor_run]


def _held_final_failure_rows() -> list[dict[str, object]]:
    origin = _failure_row(origin=True, sequence=5)
    detail = (
        "held final sampling gap exceeded bound: "
        "actor_id=RUST_ENDPOINT; role_id=2; sample_ordinal=1; "
        "previous_finish_ns=1000000000; actual_start_ns=1250000001; "
        "gap_ns=250000001; maximum_gap_ns=250000000; "
        "target_wait_ns=125000000; previous_duration_ns=1000000"
    )
    origin.update(
        {
            "operation": "HELD_FINAL_RESOURCE_SAMPLE",
            "span_id": "held-final-sample-RUST_ENDPOINT-000001",
            "substep_or_null": "COLLECT_BOUND_HELD_FINAL_RESOURCES",
            "error_detail": detail,
        }
    )
    propagated = deepcopy(_failure_rows()[2:])
    for sequence, row in zip((6, 9, 10), propagated, strict=True):
        row["sequence"] = sequence
        row["timestamp_ns"] = 2_000_000_000 + sequence
        row["record_digest"] = "sha256:" + _root(f"held-event-{sequence}")
        row["error_detail"] = detail
    return [origin, *propagated]


def _endpoint_gap_origin(
    actor: str,
    operation: str,
    *,
    sequence: int,
    label: str,
) -> dict[str, object]:
    role = "1" if actor == "PYTHON_ENDPOINT" else "2"
    row = _failure_row(origin=True, sequence=sequence)
    ordinal = 2 if actor == "PYTHON_ENDPOINT" else 3
    previous = int(row["previous_finish_ns_or_null"])
    scheduled = int(row["scheduled_start_ns_or_null"])
    actual = int(row["actual_start_ns_or_null"])
    gap = int(row["gap_ns_or_null"])
    maximum = int(row["maximum_gap_ns_or_null"])
    target = int(row["target_wait_ns_or_null"])
    if operation == "LIVE_RESOURCE_SAMPLE":
        span = f"sample-{actor}-{ordinal:06d}"
        substep = "COLLECT_BOUND_LIVE_RESOURCES"
        detail = (
            "held actor sampling gap exceeded bound: "
            f"actor_id={actor}; role_id={role}; sample_ordinal={ordinal}; "
            f"previous_finish_ns={previous}; scheduled_start_ns={scheduled}; "
            f"actual_start_ns={actual}; gap_ns={gap}; maximum_gap_ns={maximum}; "
            f"target_wait_ns={target}; previous_duration_ns=0"
        )
    else:
        assert operation == "HELD_FINAL_RESOURCE_SAMPLE"
        span = f"held-final-sample-{actor}-{ordinal:06d}"
        substep = "COLLECT_BOUND_HELD_FINAL_RESOURCES"
        detail = (
            "held final sampling gap exceeded bound: "
            f"actor_id={actor}; role_id={role}; sample_ordinal={ordinal}; "
            f"previous_finish_ns={previous}; actual_start_ns={actual}; "
            f"gap_ns={gap}; maximum_gap_ns={maximum}; target_wait_ns={target}; "
            "previous_duration_ns=0"
        )
    row.update(
        {
            "sequence": sequence,
            "timestamp_ns": 2_000_000_000 + sequence,
            "span_id": span,
            "operation": operation,
            "substep_or_null": substep,
            "actor_or_null": actor,
            "error_detail": detail,
            "role_id_or_null": role,
            "sample_ordinal_or_null": ordinal,
            "record_digest": "sha256:" + _root(label),
        }
    )
    return row


def _propagation_row(
    origin: dict[str, object],
    operation: str,
    *,
    sequence: int,
    label: str,
) -> dict[str, object]:
    actor = str(origin["actor_or_null"])
    terminal_detail = (
        "held actor sampler failed: " + str(origin["error_detail"])
        if origin["operation"] == "LIVE_RESOURCE_SAMPLE"
        else str(origin["error_detail"])
    )
    contexts = {
        "LIVE_RESOURCE_SAMPLER": (
            f"sampler-{actor}",
            "ACTOR_LIFECYCLE",
            "CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            actor,
            "MAXIMUM_UNSAMPLED_GAP_EXCEEDED; sample_span="
            + str(origin["span_id"]),
            str(origin["error_detail"]),
        ),
        "HELD_ACTOR_CLOSE": (
            f"actor-close-{actor}",
            "ACTOR_LIFECYCLE",
            "DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            actor,
            "ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
            terminal_detail,
        ),
        "SUPERVISOR_STAGE": (
            "supervisor-stage-05",
            "STAGE_05",
            "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
            None,
            "FAILED",
            terminal_detail,
        ),
        "SUPERVISOR_RUN": (
            "supervisor-run",
            "RUN",
            None,
            None,
            "supervisor attempt failed",
            terminal_detail,
        ),
    }
    span, stage, substep, row_actor, detail, error_detail = contexts[operation]
    row = _failure_row(origin=False, sequence=sequence)
    row.update(
        {
            "sequence": sequence,
            "timestamp_ns": 2_000_000_000 + sequence,
            "span_id": span,
            "operation": operation,
            "stage_or_null": stage,
            "substep_or_null": substep,
            "actor_or_null": row_actor,
            "detail_or_null": detail,
            "error_detail": error_detail,
            "record_digest": "sha256:" + _root(label),
        }
    )
    return row


def _event_row(
    *,
    sequence: int,
    event_type: str,
    span_id: str,
    operation: str,
    stage: str | None,
    substep: str | None,
    actor: str | None,
    detail: str | None,
    error_code: str | None = None,
    error_detail: str | None = None,
    record_digest: str | None = None,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "timestamp_ns": 2_000_000_000 + sequence,
        "event_type": event_type,
        "span_id": span_id,
        "operation": operation,
        "stage_or_null": stage,
        "substep_or_null": substep,
        "actor_or_null": actor,
        "detail_or_null": detail,
        "error_code_or_null": error_code,
        "error_detail_or_null": error_detail,
        "record_digest": record_digest
        or "sha256:" + _root(f"event-{sequence}-{event_type}"),
    }


def _events_for_failures(
    failure_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    assert failure_rows
    origin = failure_rows[0]
    assert origin["sequence"] == len(failure_rows) + 1
    events: list[dict[str, object]] = []
    for sequence, failure in enumerate(reversed(failure_rows), start=1):
        events.append(
            _event_row(
                sequence=sequence,
                event_type="BEGIN",
                span_id=str(failure["span_id"]),
                operation=str(failure["operation"]),
                stage=failure["stage_or_null"],
                substep=failure["substep_or_null"],
                actor=failure["actor_or_null"],
                detail="ENTER",
            )
        )
    failures_by_sequence = {
        int(row["sequence"]): row for row in failure_rows
    }
    cleanup_sequence = int(origin["sequence"]) + 1 + len(
        [
            row
            for row in failure_rows[1:]
            if row["operation"]
            in {"LIVE_RESOURCE_SAMPLER", "HELD_ACTOR_CLOSE"}
        ]
    )
    cleanup_span = f"actor-cleanup-{origin['actor_or_null']}-000001"
    for sequence in range(int(origin["sequence"]), int(failure_rows[-1]["sequence"]) + 1):
        failure = failures_by_sequence.get(sequence)
        if failure is not None:
            event = _event_row(
                sequence=sequence,
                event_type="FAIL",
                span_id=str(failure["span_id"]),
                operation=str(failure["operation"]),
                stage=failure["stage_or_null"],
                substep=failure["substep_or_null"],
                actor=failure["actor_or_null"],
                detail=failure["detail_or_null"],
                error_code=str(failure["error_code"]),
                error_detail=str(failure["error_detail"]),
                record_digest=str(failure["record_digest"]),
            )
            event["timestamp_ns"] = failure["timestamp_ns"]
            events.append(event)
        elif sequence == cleanup_sequence:
            events.append(
                _event_row(
                    sequence=sequence,
                    event_type="BEGIN",
                    span_id=cleanup_span,
                    operation="HELD_ACTOR_ABORT_CLEANUP",
                    stage="ACTOR_LIFECYCLE",
                    substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                    actor=str(origin["actor_or_null"]),
                    detail="ENTER",
                )
            )
        elif sequence == cleanup_sequence + 1:
            events.append(
                _event_row(
                    sequence=sequence,
                    event_type="OK",
                    span_id=cleanup_span,
                    operation="HELD_ACTOR_ABORT_CLEANUP",
                    stage="ACTOR_LIFECYCLE",
                    substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                    actor=str(origin["actor_or_null"]),
                    detail="cleanup_error_count=0",
                )
            )
        else:  # pragma: no cover - fixture grammar is exact.
            raise AssertionError(f"unfilled event sequence {sequence}")
    return events


def _success_events() -> list[dict[str, object]]:
    begin = _event_row(
        sequence=1,
        event_type="BEGIN",
        span_id="supervisor-run",
        operation="SUPERVISOR_RUN",
        stage="RUN",
        substep=None,
        actor=None,
        detail="ENTER",
    )
    end = _event_row(
        sequence=2,
        event_type="OK",
        span_id="supervisor-run",
        operation="SUPERVISOR_RUN",
        stage="RUN",
        substep=None,
        actor=None,
        detail="COMPLETE",
    )
    return [begin, end]


def _ordered_event_root(label: str, events: list[dict[str, object]]) -> str:
    return campaign.campaign_record_digest_v2(
        f"HEGEL/TEST/Q05B/PARTITION/{label}/V2\x00",
        [
            {
                "sequence": event["sequence"],
                "record_digest": event["record_digest"],
            }
            for event in events
        ],
    )


def _canonical_payloads_root(events: list[dict[str, object]]) -> str:
    return campaign.campaign_record_digest_v2(
        "HEGEL/TEST/Q05B/PARTITION/ALL_EVENTS/V2\x00",
        [event["record_digest"] for event in events],
    )


def _install_partition(
    value: dict[str, object],
    events: list[dict[str, object]],
) -> None:
    assert events
    failures = value["trajectory_failure_rows"]
    assert type(failures) is list
    value["trajectory_event_count"] = len(events)
    value["trajectory_first_event_record_digest"] = events[0]["record_digest"]
    value["trajectory_final_event_record_digest"] = events[-1]["record_digest"]
    value["trajectory_terminal_event_type_or_null"] = events[-1]["event_type"]
    value["trajectory_canonical_event_payloads_digest"] = (
        _canonical_payloads_root(events)
    )
    if not failures:
        value["trajectory_partition_certificate_or_null"] = None
        value["trajectory_cleanup_rows"] = []
        return

    origin = failures[0]
    assert type(origin) is dict
    origin_sequence = int(origin["sequence"])
    suffix = [
        event for event in events if int(event["sequence"]) >= origin_sequence
    ]
    failure_events = [
        event for event in suffix if event["event_type"] == "FAIL"
    ]
    cleanups = [
        deepcopy(event)
        for event in suffix
        if event["event_type"] != "FAIL"
        and event["operation"] == "HELD_ACTOR_ABORT_CLEANUP"
    ]
    other_nonfailures = [
        event
        for event in suffix
        if event["event_type"] != "FAIL"
        and event["operation"] != "HELD_ACTOR_ABORT_CLEANUP"
    ]
    assert [event["record_digest"] for event in failure_events] == [
        row["record_digest"] for row in failures
    ]
    value["trajectory_cleanup_rows"] = cleanups
    certificate: dict[str, object] = {
        "schema_version": campaign.PARTITION_CERTIFICATE_SCHEMA_V2,
        "trace_id": value["trajectory_trace_id"],
        "producer_id": value["trajectory_producer_id"],
        "trajectory_status": "COMPLETE_FAIL",
        "event_count": len(events),
        "first_event_record_digest": events[0]["record_digest"],
        "final_event_record_digest": events[-1]["record_digest"],
        "canonical_event_payloads_digest": value[
            "trajectory_canonical_event_payloads_digest"
        ],
        "origin_sequence": origin_sequence,
        "origin_record_digest": origin["record_digest"],
        "origin_actor_or_null": origin["actor_or_null"],
        "suffix_event_count": len(suffix),
        "suffix_event_root": _ordered_event_root("SUFFIX", suffix),
        "failure_event_count": len(failure_events),
        "failure_rows_digest": campaign.failure_rows_digest_v2(failures),
        "failure_rows_retained_count": len(failures),
        "failure_rows_truncated": False,
        "cleanup_nonfailure_event_count": len(cleanups),
        "cleanup_rows_digest": campaign.cleanup_rows_digest_v2(cleanups),
        "cleanup_rows_retained_count": len(cleanups),
        "cleanup_rows_truncated": False,
        "other_nonfailure_event_count": len(other_nonfailures),
        "other_nonfailure_event_root": _ordered_event_root(
            "OTHER_NONFAILURE",
            other_nonfailures,
        ),
        "partitioned_event_count": len(failure_events)
        + len(cleanups)
        + len(other_nonfailures),
        "unpartitioned_event_count": 0,
        "certificate_digest": None,
    }
    certificate["certificate_digest"] = (
        campaign.partition_certificate_digest_v2(certificate)
    )
    value["trajectory_partition_certificate_or_null"] = certificate


def _refresh_certificate(value: dict[str, object]) -> None:
    certificate = value["trajectory_partition_certificate_or_null"]
    assert type(certificate) is dict
    certificate["certificate_digest"] = None
    certificate["certificate_digest"] = (
        campaign.partition_certificate_digest_v2(certificate)
    )


def _dual_origin_value(
    python_operation: str = "LIVE_RESOURCE_SAMPLE",
    *,
    token_order: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    value = _base(1)
    python_origin = _endpoint_gap_origin(
        "PYTHON_ENDPOINT",
        python_operation,
        sequence=1,
        label="dual-python-origin",
    )
    rust_origin = _endpoint_gap_origin(
        "RUST_ENDPOINT",
        "LIVE_RESOURCE_SAMPLE",
        sequence=1,
        label="dual-rust-origin",
    )
    rows_by_token = {
        "P_ORIGIN": python_origin,
        "R_ORIGIN": rust_origin,
        "P_CLOSE": _propagation_row(
            python_origin,
            "HELD_ACTOR_CLOSE",
            sequence=1,
            label="dual-python-close",
        ),
        "R_SAMPLER": _propagation_row(
            rust_origin,
            "LIVE_RESOURCE_SAMPLER",
            sequence=1,
            label="dual-rust-sampler",
        ),
        "STAGE": _propagation_row(
            python_origin,
            "SUPERVISOR_STAGE",
            sequence=1,
            label="dual-stage",
        ),
        "RUN": _propagation_row(
            python_origin,
            "SUPERVISOR_RUN",
            sequence=1,
            label="dual-run",
        ),
    }
    if python_operation == "LIVE_RESOURCE_SAMPLE":
        rows_by_token["P_SAMPLER"] = _propagation_row(
            python_origin,
            "LIVE_RESOURCE_SAMPLER",
            sequence=1,
            label="dual-python-sampler",
        )
        default_order = (
            "R_ORIGIN",
            "R_SAMPLER",
            "P_ORIGIN",
            "P_SAMPLER",
            "P_CLOSE",
            "STAGE",
            "RUN",
        )
    else:
        default_order = (
            "R_ORIGIN",
            "R_SAMPLER",
            "P_ORIGIN",
            "P_CLOSE",
            "STAGE",
            "RUN",
        )
    order = default_order if token_order is None else token_order
    assert frozenset(order) == frozenset(rows_by_token)

    events = [
        _event_row(
            sequence=sequence,
            event_type="BEGIN",
            span_id=f"dual-prelude-{sequence}",
            operation="DUAL_PRELUDE",
            stage="STAGE_04",
            substep=None,
            actor=None,
            detail="ENTER",
        )
        for sequence in range(1, 7)
    ]
    failures: list[dict[str, object]] = []
    cleanup_inserted = False
    for token in order:
        if token == "STAGE" and not cleanup_inserted:
            for suffix, actor in enumerate(
                ("PYTHON_ENDPOINT", "RUST_ENDPOINT"),
                start=1,
            ):
                span = f"actor-cleanup-{actor}-{suffix:06d}"
                for event_type, detail in (
                    ("BEGIN", "ENTER"),
                    ("OK", "cleanup_error_count=0"),
                ):
                    events.append(
                        _event_row(
                            sequence=len(events) + 1,
                            event_type=event_type,
                            span_id=span,
                            operation="HELD_ACTOR_ABORT_CLEANUP",
                            stage="ACTOR_LIFECYCLE",
                            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                            actor=actor,
                            detail=detail,
                        )
                    )
            cleanup_inserted = True
        row = rows_by_token[token]
        sequence = len(events) + 1
        row["sequence"] = sequence
        row["timestamp_ns"] = 2_000_000_000 + sequence
        row["record_digest"] = "sha256:" + _root(f"dual-{token}-{sequence}")
        failures.append(row)
        events.append(
            _event_row(
                sequence=sequence,
                event_type="FAIL",
                span_id=str(row["span_id"]),
                operation=str(row["operation"]),
                stage=row["stage_or_null"],
                substep=row["substep_or_null"],
                actor=row["actor_or_null"],
                detail=row["detail_or_null"],
                error_code=str(row["error_code"]),
                error_detail=str(row["error_detail"]),
                record_digest=str(row["record_digest"]),
            )
        )

    terminal_detail = str(rows_by_token["RUN"]["error_detail"])
    value["trajectory_failure_rows"] = failures
    value["trajectory_terminal_error_detail_or_null"] = terminal_detail
    value.update(
        campaign.classify_supervisor_stdout_v2(
            _wire(_error_wire(detail=terminal_detail))
        )
    )
    _install_partition(value, events)
    return value, events


def _rehash_failure_semantics(value: dict[str, object], label: str) -> None:
    failures = value["trajectory_failure_rows"]
    certificate = value["trajectory_partition_certificate_or_null"]
    assert type(failures) is list and failures
    assert type(certificate) is dict
    for index, row in enumerate(failures):
        assert type(row) is dict
        row["record_digest"] = "sha256:" + _root(
            f"{label}-failure-{index}"
        )
    certificate["origin_sequence"] = failures[0]["sequence"]
    certificate["origin_record_digest"] = failures[0]["record_digest"]
    certificate["origin_actor_or_null"] = failures[0]["actor_or_null"]
    certificate["final_event_record_digest"] = failures[-1]["record_digest"]
    certificate["failure_event_count"] = len(failures)
    certificate["failure_rows_retained_count"] = len(failures)
    certificate["failure_rows_truncated"] = False
    certificate["failure_rows_digest"] = campaign.failure_rows_digest_v2(failures)
    value["trajectory_final_event_record_digest"] = failures[-1]["record_digest"]
    value["trajectory_canonical_event_payloads_digest"] = (
        campaign.campaign_record_digest_v2(
            "HEGEL/TEST/Q05B/REHASHED/ALL_EVENTS/V2\x00",
            failures,
        )
    )
    certificate["canonical_event_payloads_digest"] = value[
        "trajectory_canonical_event_payloads_digest"
    ]
    _refresh_certificate(value)


def _base(slot: int = 1) -> dict[str, object]:
    failure_rows = _failure_rows()
    events = _events_for_failures(failure_rows)
    stdout_facts = campaign.classify_supervisor_stdout_v2(
        _wire(_error_wire(detail=failure_rows[-1]["error_detail"]))
    )
    value = {
        "schema_version": campaign.SLOT_OBSERVATION_SCHEMA_V2,
        "campaign_id": "campaign-1",
        "slot_number": slot,
        "expected_trace_id": f"campaign-1-slot-{slot}",
        "expected_producer_id": "Q05B_SUPERVISOR_V4",
        "invocation_count": 1,
        "supervisor_returned": True,
        "supervisor_reaped": True,
        "process_group_absent": True,
        "exit_code_or_null": 1,
        "stdout_capture_complete": True,
        "stdout_capture_bounded": True,
        **stdout_facts,
        "stderr_capture_complete": True,
        "stderr_capture_bounded": True,
        "stderr_raw_byte_count": 0,
        "stderr_raw_sha256": hashlib.sha256(b"").hexdigest(),
        "stderr_empty": True,
        "trajectory_status": "COMPLETE_FAIL",
        "trajectory_trace_id": f"campaign-1-slot-{slot}",
        "trajectory_producer_id": "Q05B_SUPERVISOR_V4",
        "trajectory_torn_tail_or_null": None,
        "trajectory_open_span_ids": [],
        "trajectory_terminal_error_code_or_null": campaign.GAP_ERROR_CODE_V2,
        "trajectory_terminal_error_detail_or_null": failure_rows[-1][
            "error_detail"
        ],
        "trajectory_terminal_event_type_or_null": None,
        "trajectory_event_count": 0,
        "trajectory_first_event_record_digest": "sha256:" + "0" * 64,
        "trajectory_final_event_record_digest": "sha256:" + "0" * 64,
        "trajectory_canonical_event_payloads_digest": "sha256:" + "0" * 64,
        "trajectory_partition_certificate_or_null": None,
        "trajectory_failure_rows": failure_rows,
        "trajectory_cleanup_rows": [],
        "artifact_state": "ABSENT",
        "artifact_validation_status": "ABSENCE_VERIFIED",
        "artifact_replay_status": "ABSENCE_REPLAY_VERIFIED",
        "cleanup_certain": True,
        "docker_absence_certain": True,
        "clone_clean_certain": True,
        "cargo_isolation_certain": True,
        "source_binding_certain": True,
        "external_binding_certain": True,
        "replay_certain": True,
        "record_digest": None,
    }
    _install_partition(value, events)
    return value


def _seal(value: dict[str, object]) -> dict[str, object]:
    sealed = deepcopy(value)
    sealed["record_digest"] = None
    sealed["record_digest"] = campaign.slot_observation_record_digest_v2(sealed)
    return sealed


def _gap(slot: int = 1) -> dict[str, object]:
    return _seal(_base(slot))


def _early_stage4_value(
    slot: int = 1,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    value = _base(slot)
    rows = _early_stage4_failure_rows()
    events = _events_for_failures(rows)
    value["trajectory_failure_rows"] = rows
    value["trajectory_terminal_error_detail_or_null"] = rows[-1][
        "error_detail"
    ]
    value.update(
        campaign.classify_supervisor_stdout_v2(
            _wire(_error_wire(detail=rows[-1]["error_detail"]))
        )
    )
    _install_partition(value, events)
    return value, events


def _early_stage4_gap(slot: int = 1) -> dict[str, object]:
    value, _events = _early_stage4_value(slot)
    return _seal(value)


def _success(slot: int = 1) -> dict[str, object]:
    value = _base(slot)
    events = _success_events()
    value.update(campaign.classify_supervisor_stdout_v2(_wire(_success_wire())))
    value.update(
        {
            "exit_code_or_null": 0,
            "trajectory_status": "COMPLETE_OK",
            "trajectory_terminal_error_code_or_null": None,
            "trajectory_terminal_error_detail_or_null": None,
            "trajectory_failure_rows": [],
            "trajectory_cleanup_rows": [],
            "artifact_state": "VALID",
            "artifact_validation_status": "VALID",
            "artifact_replay_status": "VALID",
        }
    )
    _install_partition(value, events)
    return _seal(value)


def test_canonical_strict_codec_and_domain_separated_digest_known_answer() -> None:
    canonical = b'{"a":[true,null,7],"z":"\\u96ea"}\n'
    assert campaign.canonical_campaign_json_bytes_v2(
        {"z": "雪", "a": [True, None, 7]}
    ) == canonical
    assert campaign.strict_campaign_json_value_v2(canonical) == {
        "a": [True, None, 7],
        "z": "雪",
    }
    first = campaign.campaign_record_digest_v2("TEST/A\x00", {"x": 1})
    second = campaign.campaign_record_digest_v2("TEST/B\x00", {"x": 1})
    assert first == "sha256:" + hashlib.sha256(b'TEST/A\x00{"x":1}').hexdigest()
    assert first != second


def test_v2r13_partition_digest_domains_are_new_instance_authority() -> None:
    rows: list[dict[str, object]] = []
    failure_domain = (
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R13/"
        "TRAJECTORY_FAILURE_ROWS/V2\x00"
    )
    cleanup_domain = (
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R13/"
        "TRAJECTORY_CLEANUP_ROWS/V2\x00"
    )
    certificate_domain = (
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R13/"
        "POST_ORIGIN_PARTITION_CERTIFICATE/V2\x00"
    )
    certificate: dict[str, object] = {"certificate_digest": None}

    failure_digest = campaign.failure_rows_digest_v2(rows)
    cleanup_digest = campaign.cleanup_rows_digest_v2(rows)
    certificate_digest = campaign.partition_certificate_digest_v2(certificate)

    assert failure_digest == campaign.campaign_record_digest_v2(
        failure_domain, rows
    )
    assert cleanup_digest == campaign.campaign_record_digest_v2(
        cleanup_domain, rows
    )
    assert certificate_digest == campaign.campaign_record_digest_v2(
        certificate_domain, {}
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R2"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R3"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R4"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R5"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R6"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R7"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R8"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R9"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R10"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R11"), rows
    )
    assert failure_digest != campaign.campaign_record_digest_v2(
        failure_domain.replace("V2R13", "V2R12"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R2"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R3"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R4"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R5"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R6"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R7"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R8"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R9"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R10"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R11"), rows
    )
    assert cleanup_digest != campaign.campaign_record_digest_v2(
        cleanup_domain.replace("V2R13", "V2R12"), rows
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R2"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R3"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R4"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R5"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R6"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R7"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R8"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R9"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R10"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R11"), {}
    )
    assert certificate_digest != campaign.campaign_record_digest_v2(
        certificate_domain.replace("V2R13", "V2R12"), {}
    )


def test_version_neutral_slot_observation_digest_domain_is_retained() -> None:
    value: dict[str, object] = {"record_digest": None}
    assert campaign.slot_observation_record_digest_v2(
        value
    ) == campaign.campaign_record_digest_v2(
        "HEGEL/Q05B/SHADOW-RETRY-CAMPAIGN/SLOT-OBSERVATION/V2\x00",
        {},
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"x":1}',
        b'{"x":1}\n\n',
        b'{"x": 1}\n',
        b'{"x":1,"x":1}\n',
        b'{"x":1.0}\n',
        b'{"x":NaN}\n',
        b'{"x":"\\ud800"}\n',
        b'{"x":"\xff"}\n',
    ],
)
def test_strict_codec_rejects_every_noncanonical_surface(payload: bytes) -> None:
    with pytest.raises(campaign.ShadowRetryCampaignValidationErrorV2) as caught:
        campaign.strict_campaign_json_value_v2(payload)
    assert caught.value.code == campaign.FAIL_CANONICAL_V2


def test_classifier_freezes_exact_existing_error_wire_registry() -> None:
    wire = _wire(_error_wire())
    assert len(campaign.ERROR_WIRE_FIELDS_V2) == 17
    assert campaign.classify_supervisor_stdout_v2(wire) == {
        "stdout_raw_byte_count": len(wire),
        "stdout_raw_sha256": hashlib.sha256(wire).hexdigest(),
        "stdout_canonical_single_lf": True,
        "stdout_wire_kind": campaign.STDOUT_WIRE_ERROR_V2,
        "stdout_wire_value_or_null": _error_wire(),
        "stdout_status_or_null": campaign.GAP_ERROR_CODE_V2,
        "stdout_artifact_written_or_null": False,
    }


def test_classifier_freezes_exact_existing_success_result_registry() -> None:
    wire = _wire(_success_wire())
    facts = campaign.classify_supervisor_stdout_v2(wire)
    assert len(campaign.SUCCESS_RESULT_FIELDS_V2) == 19
    assert facts["stdout_raw_byte_count"] == len(wire)
    assert facts["stdout_raw_sha256"] == hashlib.sha256(wire).hexdigest()
    assert facts["stdout_canonical_single_lf"] is True
    assert facts["stdout_wire_kind"] == campaign.STDOUT_WIRE_SUCCESS_RESULT_V2
    assert facts["stdout_status_or_null"] is None
    assert facts["stdout_artifact_written_or_null"] is None


@pytest.mark.parametrize("slot", [1, 2, 3])
def test_success_is_terminal_success_in_every_existing_slot(slot: int) -> None:
    assert campaign.evaluate_slot_observation_v2(_success(slot)) == campaign.SUCCESS_V2


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        (1, campaign.RETRY_EXACT_GAP_V2),
        (2, campaign.RETRY_EXACT_GAP_V2),
        (3, campaign.EXHAUSTED_RETRYABLE_GAP_V2),
    ],
)
def test_exact_gap_primitive_depends_only_on_existing_slot_bound(
    slot: int, expected: str
) -> None:
    assert campaign.evaluate_slot_observation_v2(_gap(slot)) == expected


def test_live_gap_integration_kat_preserves_exact_two_tier_detail_flow() -> None:
    value = _gap(1)
    rows = value["trajectory_failure_rows"]
    raw_detail = rows[0]["error_detail"]
    outer_detail = "held actor sampler failed: " + raw_detail
    assert [row["operation"] for row in rows] == [
        "LIVE_RESOURCE_SAMPLE",
        "LIVE_RESOURCE_SAMPLER",
        "HELD_ACTOR_CLOSE",
        "SUPERVISOR_STAGE",
        "SUPERVISOR_RUN",
    ]
    assert [row["error_detail"] for row in rows] == [
        raw_detail,
        raw_detail,
        outer_detail,
        outer_detail,
        outer_detail,
    ]
    assert value["stdout_wire_value_or_null"]["detail"] == outer_detail
    assert value["trajectory_terminal_error_detail_or_null"] == outer_detail
    assert campaign.evaluate_slot_observation_v2(value) == (
        campaign.RETRY_EXACT_GAP_V2
    )


@pytest.mark.parametrize(
    ("python_operation", "token_order"),
    (
        (
            "LIVE_RESOURCE_SAMPLE",
            (
                "R_ORIGIN",
                "R_SAMPLER",
                "P_ORIGIN",
                "P_SAMPLER",
                "P_CLOSE",
                "STAGE",
                "RUN",
            ),
        ),
        (
            "LIVE_RESOURCE_SAMPLE",
            (
                "P_ORIGIN",
                "P_SAMPLER",
                "R_ORIGIN",
                "P_CLOSE",
                "R_SAMPLER",
                "STAGE",
                "RUN",
            ),
        ),
        (
            "HELD_FINAL_RESOURCE_SAMPLE",
            (
                "R_ORIGIN",
                "P_ORIGIN",
                "P_CLOSE",
                "R_SAMPLER",
                "STAGE",
                "RUN",
            ),
        ),
        (
            "HELD_FINAL_RESOURCE_SAMPLE",
            (
                "P_ORIGIN",
                "P_CLOSE",
                "R_ORIGIN",
                "R_SAMPLER",
                "STAGE",
                "RUN",
            ),
        ),
    ),
)
@pytest.mark.parametrize(
    ("slot", "expected"),
    (
        (1, campaign.RETRY_EXACT_GAP_V2),
        (2, campaign.RETRY_EXACT_GAP_V2),
        (3, campaign.EXHAUSTED_RETRYABLE_GAP_V2),
    ),
)
def test_exact_dual_origin_stage5_grammar_accepts_only_reachable_partial_orders(
    python_operation: str,
    token_order: tuple[str, ...],
    slot: int,
    expected: str,
) -> None:
    value, _events = _dual_origin_value(
        python_operation,
        token_order=token_order,
    )
    value["slot_number"] = slot
    rows = value["trajectory_failure_rows"]
    certificate = value["trajectory_partition_certificate_or_null"]
    assert type(rows) is list and type(certificate) is dict
    assert len(rows) == (7 if python_operation == "LIVE_RESOURCE_SAMPLE" else 6)
    assert certificate["origin_actor_or_null"] == rows[0]["actor_or_null"]
    assert value["stdout_wire_value_or_null"]["detail"] == rows[-1][
        "error_detail"
    ]
    assert rows[-1]["error_detail"].startswith(
        "held actor sampler failed: "
    ) is (python_operation == "LIVE_RESOURCE_SAMPLE")
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == expected


@pytest.mark.parametrize(
    "mutation",
    (
        "other-error",
        "duplicate-actor",
        "third-origin",
        "rust-held-final",
        "crossed-sampler",
        "missing-propagation",
        "rust-close",
        "wrong-terminal-cause",
        "stage4-dual",
    ),
)
def test_dual_origin_rehashed_semantic_forgeries_remain_nonretryable(
    mutation: str,
) -> None:
    value, _events = _dual_origin_value()
    rows = value["trajectory_failure_rows"]
    assert type(rows) is list
    python_origin = next(
        row
        for row in rows
        if row["actor_or_null"] == "PYTHON_ENDPOINT"
        and row["operation"] == "LIVE_RESOURCE_SAMPLE"
    )
    rust_origin = next(
        row
        for row in rows
        if row["actor_or_null"] == "RUST_ENDPOINT"
        and row["operation"] == "LIVE_RESOURCE_SAMPLE"
    )
    python_sampler = next(
        row
        for row in rows
        if row["operation"] == "LIVE_RESOURCE_SAMPLER"
        and row["actor_or_null"] == "PYTHON_ENDPOINT"
    )
    rust_sampler = next(
        row
        for row in rows
        if row["operation"] == "LIVE_RESOURCE_SAMPLER"
        and row["actor_or_null"] == "RUST_ENDPOINT"
    )
    python_close = next(row for row in rows if row["operation"] == "HELD_ACTOR_CLOSE")
    stage = next(row for row in rows if row["operation"] == "SUPERVISOR_STAGE")
    run = rows[-1]
    if mutation == "other-error":
        rust_sampler["error_code"] = "FAIL_POLICY"
    elif mutation == "duplicate-actor":
        replacement = deepcopy(python_origin)
        replacement["sequence"] = rust_origin["sequence"]
        replacement["timestamp_ns"] = rust_origin["timestamp_ns"]
        rows[rows.index(rust_origin)] = replacement
    elif mutation == "third-origin":
        replacement = deepcopy(rust_origin)
        replacement.update(
            {
                "actor_or_null": "TRUSTED_HOST_REPLAY",
                "role_id_or_null": "3",
                "span_id": "sample-TRUSTED_HOST_REPLAY-000003",
                "error_detail": str(replacement["error_detail"])
                .replace("actor_id=RUST_ENDPOINT", "actor_id=TRUSTED_HOST_REPLAY")
                .replace("role_id=2", "role_id=3"),
            }
        )
        replacement["sequence"] = rust_sampler["sequence"]
        replacement["timestamp_ns"] = rust_sampler["timestamp_ns"]
        rows[rows.index(rust_sampler)] = replacement
    elif mutation == "rust-held-final":
        replacement = _endpoint_gap_origin(
            "RUST_ENDPOINT",
            "HELD_FINAL_RESOURCE_SAMPLE",
            sequence=int(rust_origin["sequence"]),
            label="forged-rust-held-final",
        )
        rows[rows.index(rust_origin)] = replacement
    elif mutation == "crossed-sampler":
        python_sampler["error_detail"] = rust_origin["error_detail"]
    elif mutation == "missing-propagation":
        rust_sampler["operation"] = "INDEPENDENT_FAILURE"
    elif mutation == "rust-close":
        python_close["span_id"] = "actor-close-RUST_ENDPOINT"
        python_close["actor_or_null"] = "RUST_ENDPOINT"
    elif mutation == "wrong-terminal-cause":
        rust_terminal = "held actor sampler failed: " + str(
            rust_origin["error_detail"]
        )
        stage["error_detail"] = rust_terminal
        run["error_detail"] = rust_terminal
        value["trajectory_terminal_error_detail_or_null"] = rust_terminal
        value.update(
            campaign.classify_supervisor_stdout_v2(
                _wire(_error_wire(detail=rust_terminal))
            )
        )
    elif mutation == "stage4-dual":
        stage.update(
            {
                "span_id": "supervisor-stage-04",
                "stage_or_null": "STAGE_04",
                "substep_or_null": "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL",
            }
        )
    else:  # pragma: no cover - parametrization is exact.
        raise AssertionError(mutation)
    _rehash_failure_semantics(value, mutation)
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


@pytest.mark.parametrize("missing_actor", ("PYTHON_ENDPOINT", "RUST_ENDPOINT"))
def test_dual_origin_requires_complete_cleanup_for_both_origin_actors(
    missing_actor: str,
) -> None:
    value, events = _dual_origin_value()
    events[:] = [
        event
        for event in events
        if not (
            event["operation"] == "HELD_ACTOR_ABORT_CLEANUP"
            and event["actor_or_null"] == missing_actor
        )
    ]
    _renumber_events_and_failure_projection(value, events)
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


def _renumber_events_and_failure_projection(
    value: dict[str, object],
    events: list[dict[str, object]],
) -> None:
    failures = value["trajectory_failure_rows"]
    assert type(failures) is list
    failure_by_digest = {
        row["record_digest"]: row for row in failures if type(row) is dict
    }
    for sequence, event in enumerate(events, start=1):
        assert type(event) is dict
        event["sequence"] = sequence
        event["timestamp_ns"] = 2_000_000_000 + sequence
        failure = failure_by_digest.get(event["record_digest"])
        if failure is not None:
            failure["sequence"] = sequence
            failure["timestamp_ns"] = event["timestamp_ns"]
    _install_partition(value, events)


def _add_interleaved_peer_drain_and_repeated_cleanup(
    value: dict[str, object],
    events: list[dict[str, object]],
) -> None:
    failures = value["trajectory_failure_rows"]
    assert type(failures) is list
    origin_digest = failures[0]["record_digest"]
    origin_index = next(
        index
        for index, event in enumerate(events)
        if event["record_digest"] == origin_digest
    )
    peer_sampler_span = "sampler-RUST_ENDPOINT"
    peer_sample_span = "sample-RUST_ENDPOINT-000001"
    events[origin_index:origin_index] = [
        _event_row(
            sequence=1,
            event_type="BEGIN",
            span_id=peer_sampler_span,
            operation="LIVE_RESOURCE_SAMPLER",
            stage="ACTOR_LIFECYCLE",
            substep="CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            actor="RUST_ENDPOINT",
            detail="ENTER",
            record_digest="sha256:" + _root("peer-sampler-begin"),
        ),
        _event_row(
            sequence=1,
            event_type="BEGIN",
            span_id=peer_sample_span,
            operation="LIVE_RESOURCE_SAMPLE",
            stage="ACTOR_LIFECYCLE",
            substep="COLLECT_BOUND_LIVE_RESOURCES",
            actor="RUST_ENDPOINT",
            detail="sample_ordinal=1",
            record_digest="sha256:" + _root("peer-sample-begin"),
        ),
    ]
    origin_cleanup_ok = next(
        index
        for index, event in enumerate(events)
        if event["operation"] == "HELD_ACTOR_ABORT_CLEANUP"
        and event["event_type"] == "OK"
    )
    peer_cleanup_span = "actor-cleanup-RUST_ENDPOINT-000002"
    repeat_origin_cleanup_span = (
        "actor-cleanup-PYTHON_ENDPOINT-000003"
    )
    events[origin_cleanup_ok + 1 : origin_cleanup_ok + 1] = [
        _event_row(
            sequence=1,
            event_type="BEGIN",
            span_id=peer_cleanup_span,
            operation="HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="RUST_ENDPOINT",
            detail="ENTER",
            record_digest="sha256:" + _root("peer-cleanup-begin"),
        ),
        _event_row(
            sequence=1,
            event_type="OK",
            span_id=peer_sample_span,
            operation="LIVE_RESOURCE_SAMPLE",
            stage="ACTOR_LIFECYCLE",
            substep="COLLECT_BOUND_LIVE_RESOURCES",
            actor="RUST_ENDPOINT",
            detail="LIVE_RESOURCE_COLLECTION_COMPLETE",
            record_digest="sha256:" + _root("peer-sample-ok"),
        ),
        _event_row(
            sequence=1,
            event_type="OK",
            span_id=peer_sampler_span,
            operation="LIVE_RESOURCE_SAMPLER",
            stage="ACTOR_LIFECYCLE",
            substep="CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            actor="RUST_ENDPOINT",
            detail="sample_count=1",
            record_digest="sha256:" + _root("peer-sampler-ok"),
        ),
        _event_row(
            sequence=1,
            event_type="OK",
            span_id=peer_cleanup_span,
            operation="HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="RUST_ENDPOINT",
            detail="cleanup_error_count=0",
            record_digest="sha256:" + _root("peer-cleanup-ok"),
        ),
        _event_row(
            sequence=1,
            event_type="BEGIN",
            span_id=repeat_origin_cleanup_span,
            operation="HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
            record_digest="sha256:" + _root("repeat-cleanup-begin"),
        ),
        _event_row(
            sequence=1,
            event_type="OK",
            span_id=repeat_origin_cleanup_span,
            operation="HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="PYTHON_ENDPOINT",
            detail="cleanup_error_count=0",
            record_digest="sha256:" + _root("repeat-cleanup-ok"),
        ),
    ]
    _renumber_events_and_failure_projection(value, events)


def test_bounded_partition_accepts_interleaved_nonfailure_and_repeated_cleanup() -> None:
    value, events = _early_stage4_value(1)
    _add_interleaved_peer_drain_and_repeated_cleanup(value, events)
    assert [
        event["actor_or_null"]
        for event in value["trajectory_cleanup_rows"]
        if event["event_type"] == "OK"
    ] == ["PYTHON_ENDPOINT", "RUST_ENDPOINT", "PYTHON_ENDPOINT"]
    certificate = value["trajectory_partition_certificate_or_null"]
    assert type(certificate) is dict
    assert certificate["origin_sequence"] == 7
    assert certificate["suffix_event_count"] == 12
    assert certificate["failure_event_count"] == 4
    assert certificate["cleanup_nonfailure_event_count"] == 6
    assert certificate["other_nonfailure_event_count"] == 2
    assert certificate["partitioned_event_count"] == 12
    assert certificate["unpartitioned_event_count"] == 0
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == (
        campaign.RETRY_EXACT_GAP_V2
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "new-sample-begin",
        "unrelated-ok",
        "wrong-drain-actor",
        "wrong-drain-detail",
    ),
)
def test_benign_nonfailure_surface_is_aggregated_without_a_semantic_whitelist(
    mutation: str,
) -> None:
    value, events = _early_stage4_value(1)
    _add_interleaved_peer_drain_and_repeated_cleanup(value, events)
    failures = value["trajectory_failure_rows"]
    assert type(failures) is list
    origin_sequence = failures[0]["sequence"]
    if mutation == "new-sample-begin":
        peer_sampler_ok = next(
            index
            for index, event in enumerate(events)
            if event["operation"] == "LIVE_RESOURCE_SAMPLER"
            and event["event_type"] == "OK"
        )
        begin = _event_row(
            sequence=1,
            event_type="BEGIN",
            span_id="sample-RUST_ENDPOINT-000002",
            operation="LIVE_RESOURCE_SAMPLE",
            stage="ACTOR_LIFECYCLE",
            substep="COLLECT_BOUND_LIVE_RESOURCES",
            actor="RUST_ENDPOINT",
            detail="sample_ordinal=2",
            record_digest="sha256:" + _root("new-sample-begin"),
        )
        end = _event_row(
            sequence=1,
            event_type="OK",
            span_id="sample-RUST_ENDPOINT-000002",
            operation="LIVE_RESOURCE_SAMPLE",
            stage="ACTOR_LIFECYCLE",
            substep="COLLECT_BOUND_LIVE_RESOURCES",
            actor="RUST_ENDPOINT",
            detail="LIVE_RESOURCE_COLLECTION_COMPLETE",
            record_digest="sha256:" + _root("new-sample-ok"),
        )
        events[peer_sampler_ok:peer_sampler_ok] = [begin, end]
    elif mutation == "unrelated-ok":
        begin = _event_row(
            sequence=1,
            event_type="BEGIN",
            span_id="pre-origin-progress",
            operation="PRE_ORIGIN_PROGRESS",
            stage="RUN",
            substep=None,
            actor=None,
            detail="ENTER",
            record_digest="sha256:" + _root("pre-origin-progress-begin"),
        )
        end = _event_row(
            sequence=1,
            event_type="OK",
            span_id="pre-origin-progress",
            operation="PRE_ORIGIN_PROGRESS",
            stage="RUN",
            substep=None,
            actor=None,
            detail="COMPLETE",
            record_digest="sha256:" + _root("pre-origin-progress-ok"),
        )
        origin_index = next(
            index
            for index, event in enumerate(events)
            if event["sequence"] == origin_sequence
        )
        events.insert(origin_index, begin)
        cleanup_index = next(
            index
            for index, event in enumerate(events)
            if event["operation"] == "HELD_ACTOR_ABORT_CLEANUP"
        )
        events.insert(cleanup_index, end)
    elif mutation == "wrong-drain-actor":
        for event in events:
            if event["span_id"] == "sample-RUST_ENDPOINT-000001":
                event["actor_or_null"] = "TRUSTED_HOST_REPLAY"
    elif mutation == "wrong-drain-detail":
        drain = next(
            event
            for event in events
            if event["operation"] == "LIVE_RESOURCE_SAMPLE"
            and event["event_type"] == "OK"
        )
        drain["detail_or_null"] = "LIVE_RESOURCE_COLLECTION_FINISHED"
    else:  # pragma: no cover - parametrization is exact.
        raise AssertionError(mutation)
    _renumber_events_and_failure_projection(value, events)
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == (
        campaign.RETRY_EXACT_GAP_V2
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-origin-cleanup",
        "incomplete-cleanup",
        "third-origin-cleanup",
        "duplicate-cleanup-suffix",
        "noncanonical-cleanup-suffix",
        "independent-fail",
        "failure-truncated",
        "cleanup-truncated",
        "unpartitioned-event",
        "partition-count-mismatch",
        "failure-digest-mismatch",
        "cleanup-digest-mismatch",
        "certificate-digest-mismatch",
        "origin-digest-mismatch",
    ),
)
def test_cleanup_failure_and_certificate_mutations_are_nonretryable(
    mutation: str,
) -> None:
    value, events = _early_stage4_value(1)
    _add_interleaved_peer_drain_and_repeated_cleanup(value, events)
    failures = value["trajectory_failure_rows"]
    assert type(failures) is list
    if mutation == "missing-origin-cleanup":
        events[:] = [
            event
            for event in events
            if not (
                event["operation"] == "HELD_ACTOR_ABORT_CLEANUP"
                and event["actor_or_null"] == "PYTHON_ENDPOINT"
            )
        ]
        _renumber_events_and_failure_projection(value, events)
    elif mutation == "incomplete-cleanup":
        events[:] = [
            event
            for event in events
            if not (
                event["span_id"] == "actor-cleanup-PYTHON_ENDPOINT-000003"
                and event["event_type"] == "OK"
            )
        ]
        _renumber_events_and_failure_projection(value, events)
    elif mutation == "third-origin-cleanup":
        final_failure_index = next(
            index
            for index, event in enumerate(events)
            if event["operation"] == "SUPERVISOR_STAGE"
            and event["event_type"] == "FAIL"
        )
        span = "actor-cleanup-PYTHON_ENDPOINT-000004"
        events[final_failure_index:final_failure_index] = [
            _event_row(
                sequence=1,
                event_type="BEGIN",
                span_id=span,
                operation="HELD_ACTOR_ABORT_CLEANUP",
                stage="ACTOR_LIFECYCLE",
                substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                actor="PYTHON_ENDPOINT",
                detail="ENTER",
                record_digest="sha256:" + _root("third-cleanup-begin"),
            ),
            _event_row(
                sequence=1,
                event_type="OK",
                span_id=span,
                operation="HELD_ACTOR_ABORT_CLEANUP",
                stage="ACTOR_LIFECYCLE",
                substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                actor="PYTHON_ENDPOINT",
                detail="cleanup_error_count=0",
                record_digest="sha256:" + _root("third-cleanup-ok"),
            ),
        ]
        _renumber_events_and_failure_projection(value, events)
    elif mutation == "duplicate-cleanup-suffix":
        for event in events:
            if event["span_id"] == "actor-cleanup-RUST_ENDPOINT-000002":
                event["span_id"] = "actor-cleanup-RUST_ENDPOINT-000001"
        _install_partition(value, events)
    elif mutation == "noncanonical-cleanup-suffix":
        for event in events:
            if event["span_id"] == "actor-cleanup-PYTHON_ENDPOINT-000003":
                event["span_id"] = "actor-cleanup-PYTHON_ENDPOINT-0000003"
        _install_partition(value, events)
    elif mutation == "independent-fail":
        final_index = next(
            index
            for index, event in enumerate(events)
            if event["span_id"] == "supervisor-run"
            and event["event_type"] == "FAIL"
        )
        extra_digest = "sha256:" + _root("independent-fail")
        extra = deepcopy(failures[-1])
        extra.update(
            {
                "span_id": "independent-failure",
                "operation": "INDEPENDENT_FAILURE",
                "stage_or_null": "RUN",
                "substep_or_null": None,
                "detail_or_null": "FAILED",
                "record_digest": extra_digest,
            }
        )
        failures.insert(-1, extra)
        events.insert(
            final_index,
            _event_row(
                sequence=1,
                event_type="FAIL",
                span_id="independent-failure",
                operation="INDEPENDENT_FAILURE",
                stage="RUN",
                substep=None,
                actor=None,
                detail="FAILED",
                error_code=campaign.GAP_ERROR_CODE_V2,
                error_detail=str(extra["error_detail"]),
                record_digest=extra_digest,
            ),
        )
        _renumber_events_and_failure_projection(value, events)
    else:
        certificate = value["trajectory_partition_certificate_or_null"]
        assert type(certificate) is dict
        if mutation == "failure-truncated":
            certificate["failure_event_count"] += 1
            certificate["other_nonfailure_event_count"] -= 1
            certificate["failure_rows_truncated"] = True
        elif mutation == "cleanup-truncated":
            certificate["cleanup_nonfailure_event_count"] += 1
            certificate["other_nonfailure_event_count"] -= 1
            certificate["cleanup_rows_truncated"] = True
        elif mutation == "unpartitioned-event":
            certificate["other_nonfailure_event_count"] -= 1
            certificate["partitioned_event_count"] -= 1
            certificate["unpartitioned_event_count"] = 1
        elif mutation == "partition-count-mismatch":
            certificate["partitioned_event_count"] += 1
        elif mutation == "failure-digest-mismatch":
            certificate["failure_rows_digest"] = "sha256:" + _root(
                "wrong-failure-digest"
            )
        elif mutation == "cleanup-digest-mismatch":
            certificate["cleanup_rows_digest"] = "sha256:" + _root(
                "wrong-cleanup-digest"
            )
        elif mutation == "certificate-digest-mismatch":
            certificate["certificate_digest"] = "sha256:" + _root(
                "wrong-certificate-digest"
            )
        elif mutation == "origin-digest-mismatch":
            certificate["origin_record_digest"] = "sha256:" + _root(
                "wrong-origin-digest"
            )
        else:  # pragma: no cover - parametrization is exact.
            raise AssertionError(mutation)
        if mutation != "certificate-digest-mismatch":
            _refresh_certificate(value)
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        (1, campaign.RETRY_EXACT_GAP_V2),
        (2, campaign.RETRY_EXACT_GAP_V2),
        (3, campaign.EXHAUSTED_RETRYABLE_GAP_V2),
    ],
)
def test_early_stage4_live_gap_is_the_exact_retryable_alternative(
    slot: int, expected: str
) -> None:
    value = _early_stage4_gap(slot)
    rows = value["trajectory_failure_rows"]
    raw_detail = rows[0]["error_detail"]
    outer_detail = "held actor sampler failed: " + raw_detail
    assert [row["operation"] for row in rows] == [
        "LIVE_RESOURCE_SAMPLE",
        "LIVE_RESOURCE_SAMPLER",
        "SUPERVISOR_STAGE",
        "SUPERVISOR_RUN",
    ]
    assert [row["error_detail"] for row in rows] == [
        raw_detail,
        raw_detail,
        outer_detail,
        outer_detail,
    ]
    assert rows[2]["span_id"] == "supervisor-stage-04"
    assert rows[2]["stage_or_null"] == "STAGE_04"
    assert rows[2]["substep_or_null"] == (
        "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL"
    )
    assert value["stdout_wire_value_or_null"]["detail"] == outer_detail
    assert value["trajectory_terminal_error_detail_or_null"] == outer_detail
    assert campaign.evaluate_slot_observation_v2(value) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-sampler",
        "hybrid-actor-close",
        "stage-05",
        "wrong-stage-substep",
        "stage-actor",
        "stage-span",
        "stage-raw-detail",
        "sampler-prefixed-detail",
        "extra-prefix",
    ],
)
def test_early_stage4_live_gap_rejects_every_nearby_trace(
    mutation: str,
) -> None:
    value = _early_stage4_gap(1)
    rows = value["trajectory_failure_rows"]
    raw_detail = rows[0]["error_detail"]
    outer_detail = "held actor sampler failed: " + raw_detail
    if mutation == "missing-sampler":
        value["trajectory_failure_rows"] = [rows[0], *rows[2:]]
    elif mutation == "hybrid-actor-close":
        actor_close = deepcopy(_failure_rows()[2])
        actor_close["span_id"] = "actor-close-PYTHON_ENDPOINT"
        actor_close["actor_or_null"] = "PYTHON_ENDPOINT"
        actor_close["error_detail"] = outer_detail
        actor_close["sequence"] = 5
        actor_close["timestamp_ns"] = 2_000_000_005
        actor_close["record_digest"] = "sha256:" + _root(
            "stage4-hybrid-close"
        )
        for sequence, row in enumerate(rows[2:], start=6):
            row["sequence"] = sequence
            row["timestamp_ns"] = 2_000_000_000 + sequence
        value["trajectory_failure_rows"] = [
            *rows[:2],
            actor_close,
            *rows[2:],
        ]
    elif mutation == "stage-05":
        rows[2]["stage_or_null"] = "STAGE_05"
    elif mutation == "wrong-stage-substep":
        rows[2]["substep_or_null"] = (
            "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES"
        )
    elif mutation == "stage-actor":
        rows[2]["actor_or_null"] = "RUST_ENDPOINT"
    elif mutation == "stage-span":
        rows[2]["span_id"] = "supervisor-stage-4"
    elif mutation == "stage-raw-detail":
        rows[2]["error_detail"] = raw_detail
    elif mutation == "sampler-prefixed-detail":
        rows[1]["error_detail"] = outer_detail
    elif mutation == "extra-prefix":
        twice_prefixed = "held actor sampler failed: " + outer_detail
        rows[2]["error_detail"] = twice_prefixed
        rows[3]["error_detail"] = twice_prefixed
        value["trajectory_terminal_error_detail_or_null"] = twice_prefixed
        value.update(
            campaign.classify_supervisor_stdout_v2(
                _wire(_error_wire(detail=twice_prefixed))
            )
        )
    else:  # pragma: no cover - parametrization is exact.
        raise AssertionError(mutation)
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        (1, campaign.RETRY_EXACT_GAP_V2),
        (3, campaign.EXHAUSTED_RETRYABLE_GAP_V2),
    ],
)
def test_held_final_exact_gap_uses_the_same_bounded_slot_primitive(
    slot: int, expected: str
) -> None:
    value = _base(slot)
    rows = _held_final_failure_rows()
    events = _events_for_failures(rows)
    value["trajectory_failure_rows"] = rows
    value["trajectory_terminal_error_detail_or_null"] = rows[0]["error_detail"]
    value.update(
        campaign.classify_supervisor_stdout_v2(
            _wire(_error_wire(detail=rows[0]["error_detail"]))
        )
    )
    _install_partition(value, events)
    assert {row["error_detail"] for row in rows} == {rows[0]["error_detail"]}
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == expected


def test_campaign_has_exactly_three_slots_and_never_slot_four() -> None:
    assert campaign.MAX_CAMPAIGN_SLOTS_V2 == 3
    assert campaign.CAPTURE_MAX_BYTES_V2 == 67_108_864
    assert campaign.evaluate_slot_observation_v2(_gap(4)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )
    assert campaign.evaluate_slot_observation_v2(_success(4)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


@pytest.mark.parametrize(
    "malformation", ["not-map", "missing", "extra", "schema", "digest"]
)
def test_malformed_top_level_raises_typed_validation_error(malformation: str) -> None:
    value: object = _gap()
    if malformation == "not-map":
        value = []
    elif malformation == "missing":
        value.pop("campaign_id")
    elif malformation == "extra":
        value["unexpected"] = True
    elif malformation == "schema":
        value["schema_version"] = "wrong"
        value = _seal(value)
    else:
        value["record_digest"] = "sha256:" + ("0" * 64)
    with pytest.raises(campaign.ShadowRetryCampaignValidationErrorV2) as caught:
        campaign.evaluate_slot_observation_v2(value)
    assert caught.value.code == campaign.FAIL_TOP_LEVEL_V2


@pytest.mark.parametrize("location", ["top-float", "nested-float"])
def test_noncanonical_fact_types_are_terminal_not_top_level_exceptions(
    location: str,
) -> None:
    value = _gap()
    if location == "top-float":
        value["slot_number"] = 1.0
    else:
        value["trajectory_failure_rows"][0]["gap_ns_or_null"] = 250_000_001.0
    assert campaign.evaluate_slot_observation_v2(value) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


def _mutate_retry(name: str) -> dict[str, object]:
    value = _base()
    origin = value["trajectory_failure_rows"][0]
    propagated = value["trajectory_failure_rows"][1]
    if name == "slot-type":
        value["slot_number"] = "1"
    elif name == "invocation":
        value["invocation_count"] = 2
    elif name == "returned":
        value["supervisor_returned"] = False
    elif name == "reaped":
        value["supervisor_reaped"] = False
    elif name == "process-group":
        value["process_group_absent"] = False
    elif name == "exit":
        value["exit_code_or_null"] = 0
    elif name == "stdout-incomplete":
        value["stdout_capture_complete"] = False
    elif name == "stdout-unbounded":
        value["stdout_capture_bounded"] = False
    elif name == "stdout-count":
        value["stdout_raw_byte_count"] = 0
    elif name == "stdout-sha":
        value["stdout_raw_sha256"] = "0" * 64
    elif name == "stdout-noncanonical":
        value["stdout_canonical_single_lf"] = False
    elif name == "stdout-kind":
        value["stdout_wire_kind"] = campaign.STDOUT_WIRE_INVALID_V2
    elif name == "stdout-wire-value":
        value["stdout_wire_value_or_null"]["detail"] = "different detail"
    elif name == "wire-detail-binding":
        value.update(
            campaign.classify_supervisor_stdout_v2(
                _wire(_error_wire(detail="unrelated nonempty text"))
            )
        )
    elif name == "detail-metric-binding":
        bad_detail = origin["error_detail"].replace(
            "gap_ns=250000001", "gap_ns=250000002"
        )
        outer_bad_detail = "held actor sampler failed: " + bad_detail
        for index, row in enumerate(value["trajectory_failure_rows"]):
            row["error_detail"] = (
                bad_detail if index < 2 else outer_bad_detail
            )
        value["trajectory_terminal_error_detail_or_null"] = outer_bad_detail
        value.update(
            campaign.classify_supervisor_stdout_v2(
                _wire(_error_wire(detail=outer_bad_detail))
            )
        )
    elif name == "live-outer-detail-unprefixed":
        raw_detail = origin["error_detail"]
        for row in value["trajectory_failure_rows"][2:]:
            row["error_detail"] = raw_detail
        value["trajectory_terminal_error_detail_or_null"] = raw_detail
        value.update(
            campaign.classify_supervisor_stdout_v2(
                _wire(_error_wire(detail=raw_detail))
            )
        )
    elif name == "live-sampler-detail-prefixed":
        propagated["error_detail"] = "held actor sampler failed: " + origin[
            "error_detail"
        ]
    elif name == "stdout-status":
        value["stdout_status_or_null"] = "OTHER_ERROR"
    elif name == "stdout-artifact":
        value["stdout_artifact_written_or_null"] = True
    elif name == "stderr-incomplete":
        value["stderr_capture_complete"] = False
    elif name == "stderr-unbounded":
        value["stderr_capture_bounded"] = False
    elif name == "stderr-count":
        value["stderr_raw_byte_count"] = 1
    elif name == "stderr-sha":
        value["stderr_raw_sha256"] = "0" * 64
    elif name == "stderr-empty":
        value["stderr_empty"] = False
    elif name == "trace-status":
        value["trajectory_status"] = "INCOMPLETE_PREFIX"
    elif name == "trace-id":
        value["trajectory_trace_id"] = "wrong-trace"
    elif name == "producer":
        value["trajectory_producer_id"] = "wrong-producer"
    elif name == "torn":
        value["trajectory_torn_tail_or_null"] = ".event.staging-v2"
    elif name == "open":
        value["trajectory_open_span_ids"] = ["run"]
    elif name == "terminal-code":
        value["trajectory_terminal_error_code_or_null"] = "OTHER_ERROR"
    elif name == "terminal-detail":
        value["trajectory_terminal_error_detail_or_null"] = "unrelated"
    elif name == "artifact-state":
        value["artifact_state"] = "VALID"
    elif name == "artifact-validation":
        value["artifact_validation_status"] = "UNKNOWN"
    elif name == "artifact-replay":
        value["artifact_replay_status"] = "UNKNOWN"
    elif name in {
        "cleanup_certain",
        "docker_absence_certain",
        "clone_clean_certain",
        "cargo_isolation_certain",
        "source_binding_certain",
        "external_binding_certain",
        "replay_certain",
    }:
        value[name] = False
    elif name == "no-failures":
        value["trajectory_failure_rows"] = []
    elif name == "missing-propagation":
        value["trajectory_failure_rows"] = [origin]
    elif name == "row-fields":
        origin.pop("detail_or_null")
    elif name == "row-empty-optional":
        propagated["substep_or_null"] = ""
    elif name == "row-sequence":
        propagated["sequence"] = origin["sequence"]
    elif name == "row-timestamp":
        origin["timestamp_ns"] = -1
    elif name == "row-digest":
        origin["record_digest"] = "0" * 64
    elif name == "row-time-order":
        propagated["timestamp_ns"] = origin["timestamp_ns"] - 1
    elif name == "row-digest-duplicate":
        propagated["record_digest"] = origin["record_digest"]
    elif name == "origin-not-first":
        value["trajectory_failure_rows"] = [propagated, origin]
        propagated["sequence"], origin["sequence"] = 3, 4
    elif name == "two-origins":
        value["trajectory_failure_rows"] = [origin, deepcopy(origin)]
        value["trajectory_failure_rows"][1]["sequence"] = 4
        value["trajectory_failure_rows"][1]["span_id"] = "rust-sample-2"
    elif name == "origin-operation":
        origin["operation"] = "LIVE_RESOURCE_SAMPLER"
    elif name == "origin-code":
        origin["error_code"] = "OTHER_ERROR"
    elif name == "origin-actor":
        origin["actor_or_null"] = None
    elif name == "origin-role-actor":
        origin["role_id_or_null"] = "1"
    elif name == "origin-ordinal":
        origin["sample_ordinal_or_null"] = 0
    elif name == "origin-partial":
        origin["finish_ns_or_null"] = None
    elif name == "target":
        origin["target_wait_ns_or_null"] = 49_999_999
    elif name == "maximum":
        origin["maximum_gap_ns_or_null"] = 250_000_001
    elif name == "not-exceeded":
        origin["gap_exceeded_or_null"] = False
    elif name == "gap-boundary":
        origin["actual_start_ns_or_null"] = (
            origin["previous_finish_ns_or_null"] + campaign.MAXIMUM_GAP_NS_V2
        )
        origin["gap_ns_or_null"] = campaign.MAXIMUM_GAP_NS_V2
        origin["scheduler_lateness_ns_or_null"] = 200_000_000
        origin["finish_ns_or_null"] = origin["actual_start_ns_or_null"] + 1_000_000
    elif name == "gap-arithmetic":
        origin["gap_ns_or_null"] += 1
    elif name == "scheduled":
        origin["scheduled_start_ns_or_null"] += 1
    elif name == "lateness":
        origin["scheduler_lateness_ns_or_null"] += 1
    elif name == "finish-order":
        origin["finish_ns_or_null"] = origin["actual_start_ns_or_null"] - 1
    elif name == "duration":
        origin["collection_duration_ns_or_null"] += 1
    elif name == "origin-event-before-finish":
        origin["timestamp_ns"] = origin["finish_ns_or_null"] - 1
    elif name == "propagated-code":
        propagated["error_code"] = "OTHER_ERROR"
    elif name == "propagated-operation":
        propagated["operation"] = "UNRELATED_FAILURE"
    elif name == "propagated-detail":
        propagated["error_detail"] = "unrelated nonempty text"
    elif name == "propagated-detail-nul":
        propagated["error_detail"] = "x\x00y"
    elif name == "propagated-metric":
        propagated["gap_ns_or_null"] = 250_000_001
    else:  # pragma: no cover - parametrization is exact.
        raise AssertionError(name)
    return _seal(value)


@pytest.mark.parametrize(
    "mutation",
    [
        "slot-type",
        "invocation",
        "returned",
        "reaped",
        "process-group",
        "exit",
        "stdout-incomplete",
        "stdout-unbounded",
        "stdout-count",
        "stdout-sha",
        "stdout-noncanonical",
        "stdout-kind",
        "stdout-wire-value",
        "wire-detail-binding",
        "detail-metric-binding",
        "live-outer-detail-unprefixed",
        "live-sampler-detail-prefixed",
        "stdout-status",
        "stdout-artifact",
        "stderr-incomplete",
        "stderr-unbounded",
        "stderr-count",
        "stderr-sha",
        "stderr-empty",
        "trace-status",
        "trace-id",
        "producer",
        "torn",
        "open",
        "terminal-code",
        "terminal-detail",
        "artifact-state",
        "artifact-validation",
        "artifact-replay",
        "cleanup_certain",
        "docker_absence_certain",
        "clone_clean_certain",
        "cargo_isolation_certain",
        "source_binding_certain",
        "external_binding_certain",
        "replay_certain",
        "no-failures",
        "missing-propagation",
        "row-fields",
        "row-empty-optional",
        "row-sequence",
        "row-timestamp",
        "row-digest",
        "row-time-order",
        "row-digest-duplicate",
        "origin-not-first",
        "two-origins",
        "origin-operation",
        "origin-code",
        "origin-actor",
        "origin-role-actor",
        "origin-ordinal",
        "origin-partial",
        "target",
        "maximum",
        "not-exceeded",
        "gap-boundary",
        "gap-arithmetic",
        "scheduled",
        "lateness",
        "finish-order",
        "duration",
        "origin-event-before-finish",
        "propagated-code",
        "propagated-operation",
        "propagated-detail",
        "propagated-detail-nul",
        "propagated-metric",
    ],
)
def test_every_retry_conjunct_mutation_is_terminal_nonretryable(mutation: str) -> None:
    assert campaign.evaluate_slot_observation_v2(_mutate_retry(mutation)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


def _mutated_error_wire(name: str) -> dict[str, object]:
    value = _error_wire()
    replacements: dict[str, object] = {
        "schema_version": "wrong",
        "detail": "",
        "qualification_predicate_count": 1,
        "qualification_predicate_mask": 1,
        "qualification_predicate_total": 19,
        "qualification_candidate_receipt": "receipt",
        "qualification_final_receipt": "receipt",
        "q1_state": "RUN",
        "q1_gate_count": 1,
        "q1_gate_mask": 1,
        "q1_formal_output_roots": [None] * 7,
        "q1_receipt": "receipt",
        "q2_state": "RUN",
        "m3_formal_roots": [],
        "receipt_created": True,
    }
    if name == "missing":
        value.pop("detail")
    elif name == "extra":
        value["extra"] = None
    elif name == "status":
        value["status"] = "OTHER_ERROR"
    elif name == "artifact_written":
        value["artifact_written"] = True
    else:
        value[name] = deepcopy(replacements[name])
    return value


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "schema_version",
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
        "status",
        "artifact_written",
    ],
)
def test_every_error_wire_registry_or_retry_fact_mutation_is_nonretryable(
    mutation: str,
) -> None:
    facts = campaign.classify_supervisor_stdout_v2(
        _wire(_mutated_error_wire(mutation))
    )
    value = _base()
    value.update(facts)
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "artifact_path",
        "artifact_sha256",
        "qualification_count",
        "qualification_mask",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_formal_output_roots",
        "publisher_handle_close_status",
        "publisher_handle_close_error",
        "stage_roots",
        "identity_missing",
        "identity_cross_root",
        "identity_mode",
        "delivery_root",
    ],
)
def test_success_classifier_rejects_every_mutated_exact_registry_surface(
    mutation: str,
) -> None:
    value = _success_wire()
    if mutation == "missing":
        value.pop("artifact_set_root")
    elif mutation == "extra":
        value["status"] = "SUCCESS_RESULT"
    elif mutation == "artifact_path":
        value["artifact_path"] = "relative/artifact.json"
    elif mutation == "artifact_sha256":
        value["artifact_sha256"] = "0" * 64
    elif mutation == "qualification_count":
        value["qualification_count"] = 19
    elif mutation == "qualification_mask":
        value["qualification_mask"] = 0x7FFFF
    elif mutation == "q1_state":
        value["q1_state"] = "RUN"
    elif mutation == "q1_gate_count":
        value["q1_gate_count"] = 1
    elif mutation == "q1_gate_mask":
        value["q1_gate_mask"] = 1
    elif mutation == "q1_formal_output_roots":
        value["q1_formal_output_roots"] = [None] * 7
    elif mutation == "publisher_handle_close_status":
        value["publisher_handle_close_status"] = "NOT_ATTEMPTED"
    elif mutation == "publisher_handle_close_error":
        value["publisher_handle_close_status"] = "ERROR_AFTER_FINAL_HANDOFF"
    elif mutation == "stage_roots":
        value["stage_roots"] = list(reversed(value["stage_roots"]))
    elif mutation == "identity_missing":
        value["actual_final_delivery_identity"].pop("payload_length")
    elif mutation == "identity_cross_root":
        value["actual_final_delivery_identity"]["artifact_set_root"] = "0" * 64
    elif mutation == "identity_mode":
        value["actual_final_delivery_identity"]["publisher_file_mode"] = 0o644
    else:
        value["actual_final_delivery_root"] = "0" * 64
    facts = campaign.classify_supervisor_stdout_v2(_wire(value))
    assert facts["stdout_canonical_single_lf"] is True
    assert facts["stdout_wire_kind"] == campaign.STDOUT_WIRE_INVALID_V2


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("exit_code_or_null", 1),
        ("stdout_wire_kind", campaign.STDOUT_WIRE_ERROR_V2),
        ("stdout_status_or_null", campaign.GAP_ERROR_CODE_V2),
        ("stdout_artifact_written_or_null", False),
        ("trajectory_status", "COMPLETE_FAIL"),
        ("trajectory_terminal_error_code_or_null", campaign.GAP_ERROR_CODE_V2),
        ("trajectory_terminal_error_detail_or_null", "gap detail"),
        ("trajectory_failure_rows", [_failure_row(origin=True)]),
        ("artifact_state", "ABSENT"),
        ("artifact_validation_status", "ABSENCE_VERIFIED"),
        ("artifact_replay_status", "ABSENCE_REPLAY_VERIFIED"),
        ("cleanup_certain", False),
        ("docker_absence_certain", False),
        ("clone_clean_certain", False),
        ("cargo_isolation_certain", False),
        ("source_binding_certain", False),
        ("external_binding_certain", False),
        ("replay_certain", False),
    ],
)
def test_success_requires_every_success_specific_conjunct(
    field: str, replacement: object
) -> None:
    value = _success()
    value[field] = deepcopy(replacement)
    assert campaign.evaluate_slot_observation_v2(_seal(value)) == (
        campaign.TERMINAL_NONRETRYABLE_V2
    )


def test_noncanonical_or_duplicate_error_bytes_classify_invalid() -> None:
    canonical = _wire(_error_wire())
    noncanonical = canonical.replace(b":", b": ", 1)
    duplicate = canonical.replace(
        b'{"artifact_written":false,',
        b'{"artifact_written":false,"artifact_written":false,',
        1,
    )
    for payload in (noncanonical, duplicate, canonical[:-1], canonical + b"\n"):
        facts = campaign.classify_supervisor_stdout_v2(payload)
        assert facts["stdout_canonical_single_lf"] is False
        assert facts["stdout_wire_kind"] == campaign.STDOUT_WIRE_INVALID_V2


def test_classifier_api_misuse_raises_typed_validation_error() -> None:
    with pytest.raises(campaign.ShadowRetryCampaignValidationErrorV2) as caught:
        campaign.classify_supervisor_stdout_v2("not-bytes")
    assert caught.value.code == campaign.FAIL_CANONICAL_V2


def test_canonical_codec_wraps_recursive_tree_failure_in_typed_error() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(campaign.ShadowRetryCampaignValidationErrorV2) as caught:
        campaign.canonical_campaign_json_bytes_v2(cyclic)
    assert caught.value.code == campaign.FAIL_CANONICAL_V2


def test_v2r13_preconsumption_pending_is_not_a_slot_decision() -> None:
    decisions = {
        campaign.SUCCESS_V2,
        campaign.RETRY_EXACT_GAP_V2,
        campaign.EXHAUSTED_RETRYABLE_GAP_V2,
        campaign.TERMINAL_NONRETRYABLE_V2,
    }
    assert len(decisions) == 4
    assert all("PENDING" not in decision for decision in decisions)
    assert all("PENDING" not in name for name in campaign.__all__)

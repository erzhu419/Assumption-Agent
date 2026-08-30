"""Trusted-host replay for the full-node6 dual capacity diagnostic.

The host consumes files only after both quotient endpoints have exited.  It
does not import or call either quotient engine and therefore cannot create a
second measurement.  Replay is limited to canonical wire validation,
sidecar/manifest reconstruction, cross-endpoint semantic agreement and the
already-closed Q1 authority boundary.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Final, Mapping, NoReturn, Sequence


HOST_REPLAY_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-host-replay/1"
)
SIDECAR_MANIFEST_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-endpoint-sidecar-manifest/1"
)
ACTION_ID: Final = "full-node6-capacity-preflight-v1"
SUCCESS_STATUS: Final = "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY"
GUARD_STATUS: Final = "PREFLIGHT_CAPACITY_GUARD_HIT"
SIDECAR_PATHS: Final = (
    "diagnostic-summary-v1.json",
    "projection-summary-v1.json",
    "coverage-summary-v1.json",
    "fixed-point-summary-v1.json",
)
MANIFEST_PATH: Final = "endpoint-manifest-v1.json"
STREAM_KINDS: Final = ("PROGRAM", "COHORT", "CLASS", "COVERAGE")
FULL_NODE6_LIMITS: Final = {
    "maximum_ast_depth": 3,
    "maximum_ast_node_count": 6,
    "maximum_behavior_classes": 16_777_216,
    "maximum_continuation_bank_points": 67_108_864,
    "maximum_continuation_bank_points_per_class": 65_536,
    "maximum_raw_operator_applications": 4_294_967_295,
    "maximum_saturation_rounds": 16,
    "maximum_visible_frontier_points": 33_554_432,
    "maximum_visible_frontier_points_per_class": 65_536,
    "maximum_wall_time_seconds": 172_800,
    "maximum_work_queue_points": 67_108_864,
}
PROJECTION_STREAM_DOMAIN: Final = (
    b"HEGEL/Q1/CAPACITY/PREFLIGHT/COUNTING_DISCARD/V1\x00"
)
PROJECTION_STATE_DOMAIN: Final = (
    b"HEGEL/Q1/CAPACITY/PREFLIGHT/STATE_PROJECTION/V1\x00"
)
HOST_AGREEMENT_DOMAIN: Final = (
    b"HEGEL/Q1/FULL_NODE6/HOST/DUAL_SEMANTIC_AGREEMENT/V1\x00"
)
MAXIMUM_STDOUT_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_SIDECAR_BYTES: Final = 16 * 1024 * 1024
PROJECTED_HOST_REPLAY_OUTPUT_BYTES: Final = 1_048_576
PROJECTED_PEAK_HOST_REPLAY_SCRATCH_BYTES: Final = 0

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class FullNode6HostReplayError(RuntimeError):
    """Stable rejection raised by target-blind host replay."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise FullNode6HostReplayError(code, detail)


def canonical_json_line_v1(value: object) -> bytes:
    """Return one deterministic UTF-8 JSON line."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        _fail("FAIL_FULL_NODE6_HOST_WIRE", f"cannot encode canonical JSON: {error}")


def _canonical_object(payload: bytes, label: str, maximum: int) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > maximum:
        _fail("FAIL_FULL_NODE6_HOST_WIRE", f"{label} size/type differs")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_FULL_NODE6_HOST_WIRE", f"{label} JSON differs: {error}")
    if type(value) is not dict or canonical_json_line_v1(value) != payload:
        _fail("FAIL_FULL_NODE6_HOST_WIRE", f"{label} is not one canonical JSON line")
    return value


def _read_regular_file(root: Path, relative: str) -> bytes:
    path = root / relative
    try:
        status = path.lstat()
    except OSError as error:
        _fail("FAIL_FULL_NODE6_HOST_TREE", f"{relative} unavailable: {error}")
    if not stat.S_ISREG(status.st_mode) or path.is_symlink():
        _fail("FAIL_FULL_NODE6_HOST_TREE", f"{relative} is not a regular file")
    if status.st_size < 1 or status.st_size > MAXIMUM_SIDECAR_BYTES:
        _fail("FAIL_FULL_NODE6_HOST_TREE", f"{relative} size differs")
    try:
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        _fail("FAIL_FULL_NODE6_HOST_TREE", f"{relative} read failed: {error}")
    if (
        len(payload) != status.st_size
        or after.st_dev != status.st_dev
        or after.st_ino != status.st_ino
        or after.st_size != status.st_size
        or after.st_mtime_ns != status.st_mtime_ns
    ):
        _fail("FAIL_FULL_NODE6_HOST_TREE", f"{relative} changed during replay")
    return payload


def _uint(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", f"{label} must be uint")
    return value


def _digest(value: object, label: str, *, prefixed: bool = True) -> str:
    pattern = _DIGEST if prefixed else _HEX_DIGEST
    if type(value) is not str or pattern.fullmatch(value) is None:
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", f"{label} digest differs")
    return value


def _validate_manifest(
    actor_id: str,
    diagnostic: Mapping[str, object],
    payloads: Mapping[str, bytes],
    manifest_payload: bytes,
) -> dict[str, object]:
    manifest = _canonical_object(
        manifest_payload,
        f"{actor_id} manifest",
        MAXIMUM_SIDECAR_BYTES,
    )
    expected_rows = [
        {
            "byte_count": len(payloads[name]),
            "filename": name,
            "sha256": sha256(payloads[name]).hexdigest(),
        }
        for name in sorted(SIDECAR_PATHS)
    ]
    if manifest != {
        "actor_id": actor_id,
        "capsule_digest": diagnostic.get("capsule_digest"),
        "diagnostic_only": True,
        "implementation_id": diagnostic.get("implementation_id"),
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_state": "NOT_RUN",
        "rows": expected_rows,
        "schema_version": SIDECAR_MANIFEST_SCHEMA_VERSION,
        "source_commit": diagnostic.get("source_commit"),
    }:
        _fail("FAIL_FULL_NODE6_HOST_MANIFEST", f"{actor_id} manifest replay differs")
    return manifest


def _projection_summaries(diagnostic: Mapping[str, object]) -> list[object]:
    partitions = diagnostic.get("partitions")
    if type(partitions) is not list:
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", "partitions must be a list")
    return [
        {
            "diagnostic_state_projection": row.get("diagnostic_state_projection"),
            "formal_wire_projection": row.get("formal_wire_projection"),
        }
        for row in partitions
        if type(row) is dict
    ]


def _coverage_summaries(diagnostic: Mapping[str, object]) -> list[object]:
    partitions = diagnostic.get("partitions")
    if type(partitions) is not list:
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", "partitions must be a list")
    return [row.get("coverage") for row in partitions if type(row) is dict]


def _fixed_point_summaries(diagnostic: Mapping[str, object]) -> list[object]:
    partitions = diagnostic.get("partitions")
    if type(partitions) is not list:
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", "partitions must be a list")
    return [row.get("fixed_point") for row in partitions if type(row) is dict]


def _expected_sidecar(
    diagnostic: Mapping[str, object],
    kind: str,
) -> dict[str, object]:
    partitions = diagnostic.get("partitions")
    if type(partitions) is not list:
        _fail("FAIL_FULL_NODE6_HOST_SIDECAR", "partition set differs")
    field = {
        "coverage": "coverage",
        "fixed-point": "fixed_point",
    }.get(kind)
    rows = []
    for row in partitions:
        if type(row) is not dict:
            _fail("FAIL_FULL_NODE6_HOST_SIDECAR", "partition row differs")
        body = {
            "input_signature_id": row.get("input_signature_id"),
            "input_signature_name": row.get("input_signature_name"),
            "universe_root": row.get("universe_root"),
        }
        if kind == "projection":
            body.update(
                {
                    "diagnostic_state_projection": row.get(
                        "diagnostic_state_projection"
                    ),
                    "formal_wire_projection": row.get("formal_wire_projection"),
                }
            )
        else:
            assert field is not None
            body[field] = row.get(field)
            if kind == "fixed-point":
                body["same_run_record_set_root"] = row.get(
                    "same_run_record_set_root"
                )
        rows.append(body)
    return {
        "actor_id": diagnostic.get("actor_id"),
        "capsule_digest": diagnostic.get("capsule_digest"),
        "diagnostic_only": True,
        "implementation_id": diagnostic.get("implementation_id"),
        "partitions": rows,
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_state": "NOT_RUN",
        "schema_version": f"hegel-phase3a-q1-full-node6-{kind}-summary/1",
        "source_commit": diagnostic.get("source_commit"),
    }


@dataclass(frozen=True, slots=True)
class EndpointReplayV1:
    actor_id: str
    diagnostic: dict[str, object]
    normalized_semantics: dict[str, object]
    stdout_sha256: str
    sidecar_set_sha256: str
    actual_diagnostic_output_bytes: int


def _normalize_semantics(value: Mapping[str, object]) -> dict[str, object]:
    output = deepcopy(dict(value))
    for key in (
        "actor_id",
        "endpoint_schema_version",
        "implementation_id",
        "actual_preflight_peak_scratch_bytes",
        "actual_preflight_terminal_live_scratch_bytes",
        "actual_preflight_diagnostic_output_bytes",
        "actual_preflight_wall_time_ns",
        "actual_preflight_peak_rss_bytes",
    ):
        output.pop(key, None)
    partitions = output.get("partitions")
    if type(partitions) is list:
        for row in partitions:
            if type(row) is dict:
                row.pop("elapsed_monotonic_ns", None)
                # Frozen legal endpoint schedules may have different queue and
                # round high-waters.  They are aggregated explicitly by the
                # trusted host after endpoint-local validation.
                row.pop("peak_work_queue_points", None)
                row.pop("peak_saturation_round_count", None)
                barriers = row.get("depth_barriers")
                if type(barriers) is list:
                    for barrier in barriers:
                        if type(barrier) is dict:
                            # The freeze classifies intermediate bank-mutation
                            # history as an endpoint-local diagnostic.  Each
                            # endpoint is validated below, but legal frozen
                            # schedules need not report the same history.
                            barrier.pop("continuation_bank_mutation_count", None)
    return output


def _validate_diagnostic_projection(projection: object, label: str) -> None:
    if type(projection) is not dict:
        _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label} projection missing")
    if (
        projection.get("diagnostic_only") is not True
        or projection.get("formal_archive_materialized") is not False
        or projection.get("projection_kind")
        != "DIAGNOSTIC_STATE_AGREEMENT_KAT_NOT_FORMAL_ARCHIVE"
    ):
        _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label} authority differs")
    streams = projection.get("streams")
    if type(streams) is not list or len(streams) != 4:
        _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label} stream set differs")
    if tuple(row.get("stream_kind") for row in streams if type(row) is dict) != STREAM_KINDS:
        _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label} stream order differs")
    combined = sha256()
    combined.update(PROJECTION_STATE_DOMAIN)
    total_framed = 0
    for expected_kind, stream in zip(STREAM_KINDS, streams, strict=True):
        if type(stream) is not dict:
            _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label} stream type differs")
        count = _uint(stream.get("record_count"), f"{label}.{expected_kind}.count")
        payload_bytes = _uint(
            stream.get("payload_bytes"), f"{label}.{expected_kind}.payload"
        )
        framed_bytes = _uint(
            stream.get("framed_bytes"), f"{label}.{expected_kind}.framed"
        )
        digest = _digest(
            stream.get("counting_discard_sha256"),
            f"{label}.{expected_kind}.projection",
            prefixed=False,
        )
        if count < 1 or framed_bytes != payload_bytes + 8 * count:
            _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label}.{expected_kind} framing differs")
        total_framed += framed_bytes
        combined.update(expected_kind.encode("ascii"))
        combined.update(count.to_bytes(8, "big"))
        combined.update(framed_bytes.to_bytes(8, "big"))
        combined.update(digest.encode("ascii"))
    if projection.get("projected_diagnostic_state_bytes") != total_framed:
        _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label} diagnostic bytes differ")
    if projection.get("projected_diagnostic_scratch_bytes") != 0:
        _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label} diagnostic scratch differs")
    if projection.get("combined_state_projection_sha256") != combined.hexdigest():
        _fail("FAIL_FULL_NODE6_HOST_PROJECTION", f"{label} combined projection differs")


def _validate_formal_wire_projection(
    projection: object,
    label: str,
) -> tuple[int, int]:
    """Validate exact frozen-wire descriptors without materializing blobs."""

    if type(projection) is not dict:
        _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", f"{label} formal projection missing")
    if (
        projection.get("schema_version")
        != "hegel-phase3a-q1-full-node6-formal-wire-projection/1"
        or projection.get("projection_kind")
        != "FROZEN_FORMAL_WIRE_COUNTING_DISCARD_NOT_MATERIALIZED"
        or projection.get("diagnostic_only") is not True
        or projection.get("formal_archive_materialized") is not False
    ):
        _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", f"{label} formal boundary differs")
    streams = projection.get("streams")
    if type(streams) is not list or len(streams) != 4:
        _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", f"{label} formal stream set differs")
    total = 0
    for stream_id, (kind, stream) in enumerate(
        zip(STREAM_KINDS, streams, strict=True), 1
    ):
        if type(stream) is not dict:
            _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", f"{label} formal stream differs")
        if stream.get("stream_kind") != kind or stream.get("stream_kind_id") != stream_id:
            _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", f"{label} formal order differs")
        count = _uint(stream.get("record_count"), f"{label}.{kind}.formal count")
        payload = _uint(
            stream.get("canonical_record_payload_bytes"),
            f"{label}.{kind}.formal payload",
        )
        framed = _uint(stream.get("framed_bytes"), f"{label}.{kind}.formal framed")
        _uint(stream.get("chunk_count"), f"{label}.{kind}.chunk count")
        if (
            count < 1
            or framed != payload + 4 * count
            or stream.get("retained_framed_blob_count") != 0
            or stream.get("retained_framed_blob_bytes") != 0
        ):
            _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", f"{label}.{kind} framing differs")
        for name in (
            "archive_root",
            "chunk_manifest_subtree_root",
            "counting_discard_commitment",
            "external_sort_projection_root",
        ):
            _digest(stream.get(name), f"{label}.{kind}.{name}")
        total += framed
    projected_record_stream_bytes = _uint(
        projection.get("projected_record_stream_bytes"),
        f"{label} record stream",
    )
    chunk_manifest_bytes = _uint(
        projection.get("projected_chunk_manifest_stream_bytes"),
        f"{label} chunk manifest stream",
    )
    fixed_point_bytes = _uint(
        projection.get("projected_fixed_point_frame_bytes"),
        f"{label} fixed-point frame",
    )
    signature_manifest_bytes = _uint(
        projection.get("projected_signature_manifest_frame_bytes"),
        f"{label} signature manifest frame",
    )
    partition_payload_bytes = _uint(
        projection.get("projected_partition_payload_bytes"),
        f"{label} partition payload",
    )
    if (
        projected_record_stream_bytes != total
        or min(chunk_manifest_bytes, fixed_point_bytes, signature_manifest_bytes) < 1
        or partition_payload_bytes
        != total + chunk_manifest_bytes + fixed_point_bytes + signature_manifest_bytes
    ):
        _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", f"{label} formal payload differs")
    scratch_bytes = _uint(
        projection.get("projected_peak_formal_scratch_bytes"),
        f"{label} formal scratch",
    )
    return partition_payload_bytes, scratch_bytes


def _validate_partition(
    row: object,
    ordinal: int,
    *,
    projection_material_allowed: bool,
) -> tuple[int, int] | None:
    if type(row) is not dict:
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", f"partition {ordinal} type differs")
    if row.get("input_signature_id") != ordinal:
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", "partition order differs")
    expected_rows = 480 if ordinal == 1 else 85
    if row.get("universe_row_count") != expected_rows:
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", "universe row count differs")
    if row.get("terminal_status") not in (SUCCESS_STATUS, GUARD_STATUS):
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", "partition status differs")
    for key in (
        "raw_operator_application_count",
        "strict_admitted_application_count",
        "rewrite_collapse_count",
        "behavior_class_count",
        "signature_cohort_count",
        "continuation_bank_point_count",
        "visible_frontier_point_count",
        "maximum_bank_points_per_class",
        "maximum_frontier_points_per_class",
        "peak_raw_operator_application_count",
        "peak_behavior_class_count",
        "peak_visible_frontier_point_count",
        "peak_visible_frontier_points_per_class",
        "peak_continuation_bank_point_count",
        "peak_continuation_bank_points_per_class",
        "peak_work_queue_points",
        "peak_saturation_round_count",
        "vector_cache_entry_count",
        "vector_cache_hit_count",
        "vector_cache_miss_count",
    ):
        _uint(row.get(key), f"partition {ordinal}.{key}")
    if (
        row["peak_work_queue_points"]
        > FULL_NODE6_LIMITS["maximum_work_queue_points"]
        or row["peak_saturation_round_count"]
        > FULL_NODE6_LIMITS["maximum_saturation_rounds"]
    ):
        _fail(
            "FAIL_FULL_NODE6_HOST_FIXED_POINT",
            f"partition {ordinal} schedule high-water exceeds frozen limits",
        )
    if row.get("terminal_status") == SUCCESS_STATUS and row.get("traversal_closed") is not True:
        _fail("FAIL_FULL_NODE6_HOST_FIXED_POINT", "saturated partition is not closed")
    barriers = row.get("depth_barriers")
    if type(barriers) is not list or not barriers:
        _fail("FAIL_FULL_NODE6_HOST_FIXED_POINT", "depth barriers missing")
    limits = row.get("limits")
    if limits != FULL_NODE6_LIMITS or row.get("full_v16_structural_limits_applied") is not True:
        _fail(
            "FAIL_FULL_NODE6_HOST_FIXED_POINT",
            f"partition {ordinal} frozen full-node6 limits differ",
        )
    maximum_depth = 3
    barrier_uint_fields = (
        "depth",
        "eligible_raw_application_count",
        "strict_admitted_application_count",
        "rewrite_collapse_count",
        "new_behavior_class_count",
        "new_signature_cohort_count",
        "continuation_bank_mutation_count",
        "behavior_class_count_after_barrier",
        "signature_cohort_count_after_barrier",
        "continuation_bank_point_count_after_barrier",
        "visible_frontier_point_count_after_barrier",
    )
    expected_prefix_kinds = (
        "LEAF_SEED",
        *("CONSTRUCTION_DEPTH" for _ in range(maximum_depth)),
        "STRUCTURAL_BOUNDARY",
    )
    if (
        len(barriers) > len(expected_prefix_kinds)
        or tuple(barrier.get("depth") for barrier in barriers if type(barrier) is dict)
        != tuple(range(len(barriers)))
        or tuple(
            barrier.get("barrier_kind")
            for barrier in barriers
            if type(barrier) is dict
        )
        != expected_prefix_kinds[: len(barriers)]
    ):
        _fail(
            "FAIL_FULL_NODE6_HOST_FIXED_POINT",
            f"partition {ordinal} depth barrier sequence differs",
        )
    previous_classes = previous_cohorts = previous_bank = 0
    for barrier_index, barrier in enumerate(barriers):
        if type(barrier) is not dict:
            _fail(
                "FAIL_FULL_NODE6_HOST_FIXED_POINT",
                f"partition {ordinal} barrier {barrier_index} type differs",
            )
        if barrier.get("barrier_kind") not in (
            "LEAF_SEED",
            "CONSTRUCTION_DEPTH",
            "STRUCTURAL_BOUNDARY",
        ):
            _fail(
                "FAIL_FULL_NODE6_HOST_FIXED_POINT",
                f"partition {ordinal} barrier {barrier_index} kind differs",
            )
        for key in barrier_uint_fields:
            _uint(
                barrier.get(key),
                f"partition {ordinal}.depth_barriers[{barrier_index}].{key}",
            )
        if (
            barrier["strict_admitted_application_count"]
            != barrier["eligible_raw_application_count"]
            or barrier["rewrite_collapse_count"]
            > barrier["strict_admitted_application_count"]
            or barrier["new_behavior_class_count"]
            != barrier["behavior_class_count_after_barrier"] - previous_classes
            or barrier["new_signature_cohort_count"]
            != barrier["signature_cohort_count_after_barrier"] - previous_cohorts
            or barrier["continuation_bank_point_count_after_barrier"] < previous_bank
            or barrier["continuation_bank_mutation_count"]
            < barrier["continuation_bank_point_count_after_barrier"] - previous_bank
            or barrier["visible_frontier_point_count_after_barrier"]
            > barrier["continuation_bank_point_count_after_barrier"]
        ):
            _fail(
                "FAIL_FULL_NODE6_HOST_FIXED_POINT",
                f"partition {ordinal} barrier {barrier_index} transition differs",
            )
        previous_classes = barrier["behavior_class_count_after_barrier"]
        previous_cohorts = barrier["signature_cohort_count_after_barrier"]
        previous_bank = barrier["continuation_bank_point_count_after_barrier"]
    if (
        row.get("frozen_leaf_count") != 810
        or barriers[0]["eligible_raw_application_count"] != 810
        or barriers[0]["rewrite_collapse_count"] != 0
    ):
        _fail(
            "FAIL_FULL_NODE6_HOST_FIXED_POINT",
            f"partition {ordinal} leaf barrier differs",
        )
    row_is_saturated = row.get("terminal_status") == SUCCESS_STATUS
    if row_is_saturated:
        terminal_barrier = barriers[-1]
        prior_barrier = barriers[-2] if len(barriers) >= 2 else None
        completed_counted_barriers = sum(
            barrier["barrier_kind"] != "CONSTRUCTION_DEPTH"
            or barrier["eligible_raw_application_count"] > 0
            for barrier in barriers
        )
        if (
            len(barriers) != maximum_depth + 2
            or terminal_barrier["barrier_kind"] != "STRUCTURAL_BOUNDARY"
            or any(
                terminal_barrier[key] != 0
                for key in (
                    "eligible_raw_application_count",
                    "strict_admitted_application_count",
                    "rewrite_collapse_count",
                    "new_behavior_class_count",
                    "new_signature_cohort_count",
                    "continuation_bank_mutation_count",
                )
            )
            or terminal_barrier["behavior_class_count_after_barrier"]
            != row.get("behavior_class_count")
            or terminal_barrier["signature_cohort_count_after_barrier"]
            != row.get("signature_cohort_count")
            or terminal_barrier["continuation_bank_point_count_after_barrier"]
            != row.get("continuation_bank_point_count")
            or terminal_barrier["visible_frontier_point_count_after_barrier"]
            != row.get("visible_frontier_point_count")
            or type(prior_barrier) is not dict
            or any(
                terminal_barrier[key] != prior_barrier[key]
                for key in (
                    "behavior_class_count_after_barrier",
                    "signature_cohort_count_after_barrier",
                    "continuation_bank_point_count_after_barrier",
                    "visible_frontier_point_count_after_barrier",
                )
            )
            or sum(barrier["eligible_raw_application_count"] for barrier in barriers)
            != row.get("raw_operator_application_count")
            or sum(
                barrier["strict_admitted_application_count"] for barrier in barriers
            )
            != row.get("strict_admitted_application_count")
            or sum(barrier["rewrite_collapse_count"] for barrier in barriers)
            != row.get("rewrite_collapse_count")
            or row.get("peak_raw_operator_application_count")
            != row.get("raw_operator_application_count")
            or row.get("peak_behavior_class_count")
            != row.get("behavior_class_count")
            or row.get("peak_visible_frontier_point_count")
            < row.get("visible_frontier_point_count")
            or row.get("peak_visible_frontier_points_per_class")
            < row.get("maximum_frontier_points_per_class")
            or row.get("peak_continuation_bank_point_count")
            != row.get("continuation_bank_point_count")
            or row.get("peak_continuation_bank_points_per_class")
            != row.get("maximum_bank_points_per_class")
            or row.get("peak_work_queue_points") < row.get("vector_cache_entry_count")
            or row.get("peak_work_queue_points")
            < max(barrier["eligible_raw_application_count"] for barrier in barriers)
            or row.get("peak_saturation_round_count") < completed_counted_barriers
            or row.get("vector_cache_entry_count")
            != row.get("vector_cache_miss_count")
        ):
            _fail(
                "FAIL_FULL_NODE6_HOST_FIXED_POINT",
                f"partition {ordinal} saturated barrier replay differs",
            )
    elif (
        len(barriers) >= maximum_depth + 2
        or barriers[-1]["barrier_kind"] == "STRUCTURAL_BOUNDARY"
    ):
        _fail(
            "FAIL_FULL_NODE6_HOST_FIXED_POINT",
            f"partition {ordinal} guard carried a complete structural boundary",
        )
    fixed_point = row.get("fixed_point")
    record_set_root = row.get("same_run_record_set_root")
    if projection_material_allowed:
        if type(fixed_point) is not dict:
            _fail("FAIL_FULL_NODE6_HOST_FIXED_POINT", "fixed-point summary missing")
        if (
            fixed_point.get("work_queue_empty") is not True
            or fixed_point.get("zero_delta_structural_boundary") is not True
            or fixed_point.get(
                "all_eligible_operator_and_bank_tuple_expansions_covered"
            )
            is not True
            or fixed_point.get("no_resource_guard_hit") is not True
            or fixed_point.get("saturated_diagnostic_only") is not True
        ):
            _fail("FAIL_FULL_NODE6_HOST_FIXED_POINT", "fixed-point closure differs")
        _digest(record_set_root, f"partition {ordinal}.same-run record set")
    elif fixed_point is not None or record_set_root is not None:
        _fail(
            "FAIL_FULL_NODE6_HOST_GUARD_PREFIX",
            f"partition {ordinal} guard carried post-guard closure material",
        )
    coverage = row.get("coverage")
    if projection_material_allowed:
        if type(coverage) is not dict:
            _fail("FAIL_FULL_NODE6_HOST_COVERAGE", "coverage summary missing")
        if coverage.get("coverage_record_count") != 846:
            _fail("FAIL_FULL_NODE6_HOST_COVERAGE", "coverage registry count differs")
        if coverage.get("eligible_application_count") != row.get("raw_operator_application_count"):
            _fail("FAIL_FULL_NODE6_HOST_COVERAGE", "coverage eligible total differs")
        if coverage.get("processed_application_count") != row.get("raw_operator_application_count"):
            _fail("FAIL_FULL_NODE6_HOST_COVERAGE", "coverage processed total differs")
        if coverage.get("strict_admitted_application_count") != row.get("strict_admitted_application_count"):
            _fail("FAIL_FULL_NODE6_HOST_COVERAGE", "coverage strict total differs")
        if coverage.get("rewrite_collapse_count") != row.get("rewrite_collapse_count"):
            _fail("FAIL_FULL_NODE6_HOST_COVERAGE", "coverage rewrite total differs")
        _validate_diagnostic_projection(
            row.get("diagnostic_state_projection"), f"partition {ordinal}"
        )
        formal_projection = _validate_formal_wire_projection(
            row.get("formal_wire_projection"), f"partition {ordinal}"
        )
    elif (
        coverage is not None
        or row.get("diagnostic_state_projection") is not None
        or row.get("formal_wire_projection") is not None
    ):
        _fail(
            "FAIL_FULL_NODE6_HOST_GUARD_PREFIX",
            f"partition {ordinal} guard carried post-guard replay material",
        )
    return formal_projection if projection_material_allowed else None


def replay_endpoint_v1(
    actor_id: str,
    stdout_payload: bytes,
    output_root: Path,
    *,
    expected_capsule_digest: str,
    expected_source_commit: str,
) -> EndpointReplayV1:
    """Replay one exited endpoint's stdout and small sidecar set."""

    if actor_id not in ("PYTHON_ENDPOINT", "RUST_ENDPOINT"):
        raise ValueError("actor_id differs")
    try:
        root_status = output_root.lstat()
    except OSError as error:
        _fail("FAIL_FULL_NODE6_HOST_TREE", f"{actor_id} output root unavailable: {error}")
    if output_root.is_symlink() or not stat.S_ISDIR(root_status.st_mode):
        _fail("FAIL_FULL_NODE6_HOST_TREE", f"{actor_id} output root differs")
    diagnostic = _canonical_object(
        stdout_payload,
        f"{actor_id} stdout",
        MAXIMUM_STDOUT_BYTES,
    )
    if (
        diagnostic.get("actor_id") != actor_id
        or diagnostic.get("action_id") != ACTION_ID
        or diagnostic.get("capsule_digest") != expected_capsule_digest
        or diagnostic.get("source_commit") != expected_source_commit
        or _DIGEST.fullmatch(expected_capsule_digest) is None
        or _COMMIT.fullmatch(expected_source_commit) is None
    ):
        _fail("FAIL_FULL_NODE6_HOST_BINDING", f"{actor_id} execution binding differs")
    if (
        diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("q1_state") != "NOT_RUN"
        or diagnostic.get("q1_gate_count") != 0
        or diagnostic.get("q1_gate_mask") != 0
        or diagnostic.get("q1_formal_roots") is not None
        or diagnostic.get("q1_receipt") is not None
        or diagnostic.get("q1_output_slots") != [None] * 8
        or diagnostic.get("q1_execution_started") is not False
        or diagnostic.get("q2_state") != "NOT_RUN"
        or diagnostic.get("target_truth_accessed") is not False
        or diagnostic.get("split_accessed") is not False
        or diagnostic.get("role_evaluation_performed") is not False
        or diagnostic.get("active_transition_allowed") is not False
    ):
        _fail("FAIL_FULL_NODE6_HOST_AUTHORITY", f"{actor_id} Q1 boundary differs")
    partitions = diagnostic.get("partitions")
    if type(partitions) is not list or len(partitions) not in (1, 2):
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", f"{actor_id} partition set differs")
    statuses = [
        row.get("terminal_status") if type(row) is dict else None
        for row in partitions
    ]
    aggregate = diagnostic.get("terminal_status")
    saturated_shape = (
        aggregate == SUCCESS_STATUS
        and len(partitions) == 2
        and statuses == [SUCCESS_STATUS, SUCCESS_STATUS]
    )
    guard_shape = (
        aggregate == GUARD_STATUS
        and statuses[-1] == GUARD_STATUS
        and all(status == SUCCESS_STATUS for status in statuses[:-1])
    )
    if not saturated_shape and not guard_shape:
        _fail(
            "FAIL_FULL_NODE6_HOST_SEMANTICS",
            f"{actor_id} terminal prefix differs",
        )
    peak_scratch = diagnostic.get("actual_preflight_peak_scratch_bytes")
    terminal_live_scratch = diagnostic.get(
        "actual_preflight_terminal_live_scratch_bytes"
    )
    if (
        type(peak_scratch) is not int
        or peak_scratch < 0
        or type(terminal_live_scratch) is not int
        or terminal_live_scratch != 0
        or (saturated_shape and peak_scratch == 0)
        or (guard_shape and peak_scratch != 0)
    ):
        _fail(
            "FAIL_FULL_NODE6_HOST_RESOURCE",
            f"{actor_id} actual scratch accounting differs",
        )
    formal_partition_rows: list[tuple[int, int]] = []
    for ordinal, row in enumerate(partitions, 1):
        formal_row = _validate_partition(
            row,
            ordinal,
            projection_material_allowed=saturated_shape,
        )
        if formal_row is not None:
            formal_partition_rows.append(formal_row)
    aggregate_formal_fields = (
        "projected_closure_bundle_frame_bytes",
        "projected_formal_archive_payload_bytes",
        "projected_formal_total_output_bytes",
        "projected_peak_formal_scratch_bytes",
    )
    if saturated_shape:
        if len(formal_partition_rows) != 2:
            _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", "partition projection set differs")
        closure_frame = _uint(
            diagnostic.get("projected_closure_bundle_frame_bytes"),
            "aggregate closure bundle frame",
        )
        archive_payload = _uint(
            diagnostic.get("projected_formal_archive_payload_bytes"),
            "aggregate formal archive payload",
        )
        formal_total = _uint(
            diagnostic.get("projected_formal_total_output_bytes"),
            "aggregate formal total output",
        )
        peak_formal_scratch = _uint(
            diagnostic.get("projected_peak_formal_scratch_bytes"),
            "aggregate formal scratch",
        )
        if (
            closure_frame < 1
            or archive_payload
            != sum(row[0] for row in formal_partition_rows) + closure_frame
            or formal_total != archive_payload + PROJECTED_HOST_REPLAY_OUTPUT_BYTES
            or peak_formal_scratch != max(row[1] for row in formal_partition_rows)
        ):
            _fail("FAIL_FULL_NODE6_HOST_FORMAL_WIRE", "aggregate projection differs")
    elif any(diagnostic.get(field) is not None for field in aggregate_formal_fields):
        _fail("FAIL_FULL_NODE6_HOST_GUARD_PREFIX", "guard carried aggregate projection")

    payloads = {name: _read_regular_file(output_root, name) for name in SIDECAR_PATHS}
    manifest_payload = _read_regular_file(output_root, MANIFEST_PATH)
    _validate_manifest(actor_id, diagnostic, payloads, manifest_payload)
    decoded = {
        name: _canonical_object(payload, f"{actor_id} {name}", MAXIMUM_SIDECAR_BYTES)
        for name, payload in payloads.items()
    }
    if decoded["diagnostic-summary-v1.json"] != diagnostic:
        _fail("FAIL_FULL_NODE6_HOST_SIDECAR", "diagnostic sidecar differs")
    for kind, name in (
        ("projection", "projection-summary-v1.json"),
        ("coverage", "coverage-summary-v1.json"),
        ("fixed-point", "fixed-point-summary-v1.json"),
    ):
        if decoded[name] != _expected_sidecar(diagnostic, kind):
            _fail("FAIL_FULL_NODE6_HOST_SIDECAR", f"{kind} sidecar differs")
    sidecar_set = b"".join(payloads[name] for name in SIDECAR_PATHS) + manifest_payload
    return EndpointReplayV1(
        actor_id=actor_id,
        diagnostic=diagnostic,
        normalized_semantics=_normalize_semantics(diagnostic),
        stdout_sha256=sha256(stdout_payload).hexdigest(),
        sidecar_set_sha256=sha256(sidecar_set).hexdigest(),
        actual_diagnostic_output_bytes=(
            len(stdout_payload)
            + sum(len(payloads[name]) for name in SIDECAR_PATHS)
            + len(manifest_payload)
        ),
    )


def replay_dual_endpoints_v1(
    python_stdout: bytes,
    python_output_root: Path,
    rust_stdout: bytes,
    rust_output_root: Path,
    *,
    expected_capsule_digest: str,
    expected_source_commit: str,
) -> dict[str, object]:
    """Require exact semantic agreement without rerunning either endpoint."""

    try:
        if python_output_root.resolve(strict=True) == rust_output_root.resolve(strict=True):
            _fail("FAIL_FULL_NODE6_HOST_ISOLATION", "endpoint output roots are not distinct")
    except OSError as error:
        _fail("FAIL_FULL_NODE6_HOST_TREE", f"cannot resolve endpoint roots: {error}")
    python = replay_endpoint_v1(
        "PYTHON_ENDPOINT",
        python_stdout,
        python_output_root,
        expected_capsule_digest=expected_capsule_digest,
        expected_source_commit=expected_source_commit,
    )
    rust = replay_endpoint_v1(
        "RUST_ENDPOINT",
        rust_stdout,
        rust_output_root,
        expected_capsule_digest=expected_capsule_digest,
        expected_source_commit=expected_source_commit,
    )
    if python.normalized_semantics != rust.normalized_semantics:
        _fail(
            "FAIL_FULL_NODE6_HOST_DUAL_DISAGREEMENT",
            "Python and Rust semantic diagnostics differ",
        )
    python_partitions = python.diagnostic["partitions"]
    rust_partitions = rust.diagnostic["partitions"]
    work_queue_high_water_by_input_signature = [
        max(
            python_row["peak_work_queue_points"],
            rust_row["peak_work_queue_points"],
        )
        for python_row, rust_row in zip(
            python_partitions,
            rust_partitions,
            strict=True,
        )
    ]
    saturation_round_high_water = max(
        row["peak_saturation_round_count"]
        for row in (*python_partitions, *rust_partitions)
    )
    normalized = canonical_json_line_v1(python.normalized_semantics)
    agreement_sha256 = sha256(HOST_AGREEMENT_DOMAIN + normalized).hexdigest()
    terminal_status = python.normalized_semantics.get("terminal_status")
    if terminal_status not in (SUCCESS_STATUS, GUARD_STATUS):
        _fail("FAIL_FULL_NODE6_HOST_SEMANTICS", "aggregate terminal status differs")
    return {
        "action_id": ACTION_ID,
        "capsule_digest": expected_capsule_digest,
        "diagnostic_only": True,
        "dual_semantic_agreement_sha256": agreement_sha256,
        "endpoint_rows": [
            {
                "actor_id": python.actor_id,
                "actual_preflight_peak_scratch_bytes": python.diagnostic[
                    "actual_preflight_peak_scratch_bytes"
                ],
                "peak_saturation_round_count_by_input_signature": [
                    row["peak_saturation_round_count"] for row in python_partitions
                ],
                "peak_work_queue_points_by_input_signature": [
                    row["peak_work_queue_points"] for row in python_partitions
                ],
                "sidecar_set_sha256": python.sidecar_set_sha256,
                "stdout_sha256": python.stdout_sha256,
                "actual_diagnostic_output_bytes": (
                    python.actual_diagnostic_output_bytes
                ),
            },
            {
                "actor_id": rust.actor_id,
                "actual_preflight_peak_scratch_bytes": rust.diagnostic[
                    "actual_preflight_peak_scratch_bytes"
                ],
                "peak_saturation_round_count_by_input_signature": [
                    row["peak_saturation_round_count"] for row in rust_partitions
                ],
                "peak_work_queue_points_by_input_signature": [
                    row["peak_work_queue_points"] for row in rust_partitions
                ],
                "sidecar_set_sha256": rust.sidecar_set_sha256,
                "stdout_sha256": rust.stdout_sha256,
                "actual_diagnostic_output_bytes": rust.actual_diagnostic_output_bytes,
            },
        ],
        "formal_resource_projection": {
            "projected_formal_total_output_bytes": (
                python.normalized_semantics.get("projected_formal_total_output_bytes")
            ),
            "projected_peak_formal_scratch_bytes": (
                python.normalized_semantics.get("projected_peak_formal_scratch_bytes")
            ),
        },
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q1_state": "NOT_RUN",
        "registered_resource_guard_id": (
            python.normalized_semantics["partitions"][-1].get("resource_guard_id")
            if terminal_status == GUARD_STATUS
            else None
        ),
        "schedule_high_water_aggregation": {
            "peak_saturation_round_count_across_endpoints_and_signatures": (
                saturation_round_high_water
            ),
            "peak_work_queue_points_by_input_signature": (
                work_queue_high_water_by_input_signature
            ),
        },
        "projected_host_replay_output_bytes": PROJECTED_HOST_REPLAY_OUTPUT_BYTES,
        "projected_peak_host_replay_scratch_bytes": (
            PROJECTED_PEAK_HOST_REPLAY_SCRATCH_BYTES
        ),
        "schema_version": HOST_REPLAY_SCHEMA_VERSION,
        "source_commit": expected_source_commit,
        "terminal_status": terminal_status,
    }


def _require_empty_scratch_v1(path: Path) -> None:
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        _fail("FAIL_FULL_NODE6_HOST_SCRATCH", str(error))
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not stat.S_ISDIR(status.st_mode)
        or resolved != path
    ):
        _fail("FAIL_FULL_NODE6_HOST_SCRATCH", "scratch root identity differs")
    try:
        if any(path.iterdir()):
            _fail("FAIL_FULL_NODE6_HOST_SCRATCH", "scratch root is not empty")
        probe = path / ".host-replay-write-probe-v1"
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, b"probe\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        probe.unlink()
        directory_descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if any(path.iterdir()):
            _fail("FAIL_FULL_NODE6_HOST_SCRATCH", "scratch cleanup differs")
    except OSError as error:
        _fail("FAIL_FULL_NODE6_HOST_SCRATCH", str(error))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--python-output", type=Path, required=True)
    parser.add_argument("--rust-output", type=Path, required=True)
    parser.add_argument("--capsule-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.run:
        _fail("REJECT_FULL_NODE6_HOST_MODE", "--run is required")
    _require_empty_scratch_v1(arguments.scratch_dir)
    python_stdout = _read_regular_file(
        arguments.python_output,
        "diagnostic-summary-v1.json",
    )
    rust_stdout = _read_regular_file(
        arguments.rust_output,
        "diagnostic-summary-v1.json",
    )
    result = replay_dual_endpoints_v1(
        python_stdout,
        arguments.python_output,
        rust_stdout,
        arguments.rust_output,
        expected_capsule_digest=arguments.capsule_digest,
        expected_source_commit=arguments.source_commit,
    )
    _require_empty_scratch_v1(arguments.scratch_dir)
    sys.stdout.buffer.write(canonical_json_line_v1(result))
    return 0


__all__ = [
    "ACTION_ID",
    "EndpointReplayV1",
    "FullNode6HostReplayError",
    "HOST_REPLAY_SCHEMA_VERSION",
    "MANIFEST_PATH",
    "PROJECTED_HOST_REPLAY_OUTPUT_BYTES",
    "PROJECTED_PEAK_HOST_REPLAY_SCRATCH_BYTES",
    "SIDECAR_MANIFEST_SCHEMA_VERSION",
    "SIDECAR_PATHS",
    "STREAM_KINDS",
    "canonical_json_line_v1",
    "replay_dual_endpoints_v1",
    "replay_endpoint_v1",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FullNode6HostReplayError as error:
        sys.stdout.buffer.write(
            canonical_json_line_v1(
                {
                    "error_code": error.code,
                    "error_detail": error.detail,
                    "q1_gate_count": 0,
                    "q1_gate_mask": 0,
                    "q1_output_slots": [None] * 8,
                    "q1_state": "NOT_RUN",
                    "schema_version": (
                        "hegel-phase3a-q1-full-node6-host-replay-error/1"
                    ),
                    "terminal_status": (
                        "FULL_NODE6_HOST_REPLAY_FAILED_TERMINAL_NO_RESUME"
                    ),
                }
            )
        )
        sys.stderr.write(str(error) + "\n")
        raise SystemExit(1)

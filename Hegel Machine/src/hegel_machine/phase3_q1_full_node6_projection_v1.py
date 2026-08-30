"""Same-run counting/discard projection for the full node-6 diagnostic.

The admitted capacity runner measures each input-signature quotient exactly
once and returns its immutable final snapshot.  This module consumes those
same objects.  It never calls the public snapshot validator (which deliberately
starts a fresh traversal), never builds another quotient state, and never
evaluates a program on the production universe.  Coverage is a projection-only
replay of application identities over the final continuation bank.

All four formal-shaped streams remain diagnostic: framed record blobs are
counted and discarded, all Q1 output slots stay null, and no Q1 gate or receipt
is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import heapq
import json
from pathlib import Path
import tempfile
from typing import Callable, Final, Iterator, NoReturn

from . import phase3_q1_archive_projection_v1 as _archive
from . import phase3_q1_capacity_preflight_v1 as _capacity
from . import phase3_q1_external_sort_profile_v1 as _external
from . import phase3_q1_formal_archive_contract_v1 as _formal
from . import phase3_q1_full_node6_capacity_preflight_v1 as _full
from . import phase3_q1_partition_snapshot_v1 as _snapshot
from .phase3_q1_quotient_contract_v1 import OutputSortId
from .strict_ast_shrink6_v1 import (
    canonicalize_shrink6_source_ast,
    decode_shrink6_canonical_ast,
)
from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-projection/1"
PROJECTION_ID: Final = "hegel-phase3a-q1-full-node6-counting-discard-v1"
RECORD_SET_ROOT_DOMAIN: Final = "HEGEL/Q1/FULL_NODE6/SAME_RUN_RECORD_SET/V1"
RESULT_ROOT_DOMAIN: Final = "HEGEL/Q1/FULL_NODE6/PROJECTION_RESULT/V1"

STREAM_REGISTRY: Final = (
    ("program", _formal.ArchiveStreamKindId.PROGRAM),
    ("cohort", _formal.ArchiveStreamKindId.COHORT),
    ("class", _formal.ArchiveStreamKindId.CLASS),
    ("coverage", _formal.ArchiveStreamKindId.COVERAGE),
)
SIDECAR_PATHS: Final = tuple(
    (
        "diagnostic-summary-v1.json",
        "projection-summary-v1.json",
        "coverage-summary-v1.json",
        "fixed-point-summary-v1.json",
        "endpoint-manifest-v1.json",
    )
)
SCRATCH_ROOT: Final = Path("/scratch")
CAPACITY_STREAM_DIGEST_DOMAIN: Final = (
    b"HEGEL/Q1/CAPACITY/PREFLIGHT/COUNTING_DISCARD/V1\x00"
)
CAPACITY_STATE_DIGEST_DOMAIN: Final = (
    b"HEGEL/Q1/CAPACITY/PREFLIGHT/STATE_PROJECTION/V1\x00"
)
CAPACITY_PROJECTION_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-capacity-counting-discard-projection/1"
)
COVERAGE_SORT_BUFFER_BYTES: Final = 8 * 1024 * 1024
COVERAGE_SORT_MERGE_FAN_IN: Final = 16
FROZEN_SEMANTIC_BINDING_ROOT: Final = bytes.fromhex(
    "e3b3df3e81b7632c7c713ef5ec84913f990ad8232a25b851f20c46ac7416bfcb"
)
FROZEN_PROJECTION_PROFILE_ROOT: Final = bytes.fromhex(
    "aa441cdc49ab60324483b9aa44e9fdfc324a6ad49a6bff50af6daa775209816d"
)
FROZEN_PROJECTION_PROFILE_ROOT_ID: Final = (
    "sha256:" + FROZEN_PROJECTION_PROFILE_ROOT.hex()
)

_RetainedChunkConsumerV1 = Callable[
    [_formal.Q1ArchiveChunkManifestV1, bytes],
    None,
]


class FullNode6ProjectionError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise FullNode6ProjectionError(code, detail)


def canonical_json_bytes_v1(value: object) -> bytes:
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
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail("FAIL_FULL_NODE6_PROJECTION_JSON", str(error))


@dataclass(frozen=True, slots=True)
class CapacityCountingDiscardStreamV1:
    stream_kind: str
    record_count: int
    payload_bytes: int
    framed_bytes: int
    counting_discard_sha256: str

    def __post_init__(self) -> None:
        if self.stream_kind not in ("PROGRAM", "COHORT", "CLASS", "COVERAGE"):
            _fail("REJECT_FULL_NODE6_CAPACITY_PROJECTION", "stream kind differs")
        for name in ("record_count", "payload_bytes", "framed_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail("REJECT_FULL_NODE6_CAPACITY_PROJECTION", f"{name} differs")
        if self.framed_bytes != self.payload_bytes + 8 * self.record_count:
            _fail("REJECT_FULL_NODE6_CAPACITY_PROJECTION", "u64 framing count differs")
        if (
            type(self.counting_discard_sha256) is not str
            or len(self.counting_discard_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.counting_discard_sha256)
        ):
            _fail("REJECT_FULL_NODE6_CAPACITY_PROJECTION", "stream digest differs")

    def json_object(self) -> dict[str, object]:
        return {
            "counting_discard_sha256": self.counting_discard_sha256,
            "framed_bytes": self.framed_bytes,
            "payload_bytes": self.payload_bytes,
            "record_count": self.record_count,
            "stream_kind": self.stream_kind,
        }


@dataclass(frozen=True, slots=True)
class CapacityStateProjectionV1:
    streams: tuple[CapacityCountingDiscardStreamV1, ...]
    combined_state_projection_sha256: str
    schema_version: str = CAPACITY_PROJECTION_SCHEMA_VERSION
    projection_kind: str = "DIAGNOSTIC_STATE_AGREEMENT_KAT_NOT_FORMAL_ARCHIVE"
    diagnostic_only: bool = True
    formal_archive_materialized: bool = False
    projected_diagnostic_scratch_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            tuple(row.stream_kind for row in self.streams)
            != ("PROGRAM", "COHORT", "CLASS", "COVERAGE")
            or self.schema_version != CAPACITY_PROJECTION_SCHEMA_VERSION
            or self.projection_kind
            != "DIAGNOSTIC_STATE_AGREEMENT_KAT_NOT_FORMAL_ARCHIVE"
            or self.diagnostic_only is not True
            or self.formal_archive_materialized is not False
            or self.projected_diagnostic_scratch_bytes != 0
            or type(self.combined_state_projection_sha256) is not str
            or len(self.combined_state_projection_sha256) != 64
        ):
            _fail("REJECT_FULL_NODE6_CAPACITY_PROJECTION", "projection boundary differs")

    @property
    def projected_diagnostic_state_bytes(self) -> int:
        return sum(row.framed_bytes for row in self.streams)

    def json_object(self) -> dict[str, object]:
        return {
            "combined_state_projection_sha256": self.combined_state_projection_sha256,
            "diagnostic_only": True,
            "formal_archive_materialized": False,
            "projected_diagnostic_state_bytes": (
                self.projected_diagnostic_state_bytes
            ),
            "projected_diagnostic_scratch_bytes": 0,
            "projection_kind": self.projection_kind,
            "schema_version": self.schema_version,
            "streams": [row.json_object() for row in self.streams],
        }


@dataclass(frozen=True, slots=True)
class CoverageDiagnosticV1:
    coverage_record_count: int
    eligible_application_count: int
    processed_application_count: int
    strict_admitted_application_count: int
    rewrite_collapse_count: int
    zero_eligible_record_count: int
    all_eligible_covered: bool

    def __post_init__(self) -> None:
        for name in (
            "coverage_record_count",
            "eligible_application_count",
            "processed_application_count",
            "strict_admitted_application_count",
            "rewrite_collapse_count",
            "zero_eligible_record_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                _fail("REJECT_FULL_NODE6_COVERAGE", f"{name} differs")
        if (
            self.coverage_record_count != _formal.EXPECTED_COVERAGE_RECORD_COUNT
            or self.eligible_application_count != self.processed_application_count
            or self.strict_admitted_application_count
            > self.processed_application_count
            or self.rewrite_collapse_count > self.strict_admitted_application_count
            or self.zero_eligible_record_count > self.coverage_record_count
            or type(self.all_eligible_covered) is not bool
        ):
            _fail("REJECT_FULL_NODE6_COVERAGE", "coverage relation differs")

    def json_object(self) -> dict[str, object]:
        return {
            "all_eligible_covered": self.all_eligible_covered,
            "coverage_record_count": self.coverage_record_count,
            "eligible_application_count": self.eligible_application_count,
            "processed_application_count": self.processed_application_count,
            "rewrite_collapse_count": self.rewrite_collapse_count,
            "strict_admitted_application_count": self.strict_admitted_application_count,
            "zero_eligible_record_count": self.zero_eligible_record_count,
        }


@dataclass(frozen=True, slots=True)
class FullNode6PartitionProjectionV1:
    input_signature_id: int
    universe_root: bytes
    same_run_record_set_root: bytes
    streams: tuple[_archive.Q1CountingDiscardStreamV1, ...]
    capacity_state_projection: CapacityStateProjectionV1
    coverage: CoverageDiagnosticV1
    fixed_point_record: _formal.Q1FixedPointRecordV1
    saturation_state_root: bytes
    signature_archive_manifest: _formal.Q1SignatureArchiveManifestV1
    projection_partition_row: _formal.Q1ProjectionPartitionRowV1
    q1_state: str = "NOT_RUN"
    q1_gate_count: int = 0
    q1_gate_mask: int = 0
    q1_output_slots: tuple[None, ...] = (None,) * 8
    target_truth_accessed: bool = False
    split_accessed: bool = False
    role_evaluation_performed: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.streams) is not tuple
            or tuple(row.stream_kind_id for row in self.streams)
            != tuple(kind for _name, kind in STREAM_REGISTRY)
            or any(row.input_signature_id != self.input_signature_id for row in self.streams)
            or any(row.universe_root != self.universe_root for row in self.streams)
            or type(self.capacity_state_projection) is not CapacityStateProjectionV1
            or type(self.coverage) is not CoverageDiagnosticV1
            or type(self.fixed_point_record) is not _formal.Q1FixedPointRecordV1
            or type(self.saturation_state_root) is not bytes
            or len(self.saturation_state_root) != 32
            or type(self.signature_archive_manifest)
            is not _formal.Q1SignatureArchiveManifestV1
            or type(self.projection_partition_row)
            is not _formal.Q1ProjectionPartitionRowV1
            or self.q1_state != "NOT_RUN"
            or self.q1_gate_count != 0
            or self.q1_gate_mask != 0
            or self.q1_output_slots != (None,) * 8
            or self.target_truth_accessed
            or self.split_accessed
            or self.role_evaluation_performed
        ):
            _fail("REJECT_FULL_NODE6_PROJECTION", "partition authority boundary differs")
        if type(self.same_run_record_set_root) is not bytes or len(self.same_run_record_set_root) != 32:
            _fail("REJECT_FULL_NODE6_PROJECTION", "record-set root differs")
        _validate_formal_partition_assembly_v1(self)

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            SCHEMA_VERSION.encode("ascii"),
            self.input_signature_id,
            self.universe_root,
            self.same_run_record_set_root,
            tuple(row.canonical_object() for row in self.streams),
            self.capacity_state_projection.combined_state_projection_sha256.encode("ascii"),
            canonical_cbor_encode(tuple(self.coverage.json_object().items())),
            self.fixed_point_record.canonical_object(),
            self.saturation_state_root,
            self.signature_archive_manifest.canonical_object(),
            self.projection_partition_row.canonical_object(),
            self.q1_state.encode("ascii"),
            self.q1_gate_count,
            self.q1_gate_mask,
            self.q1_output_slots,
            self.target_truth_accessed,
            self.split_accessed,
            self.role_evaluation_performed,
        )


@dataclass(frozen=True, slots=True)
class FullNode6ProjectionResultV1:
    capsule_digest: str
    source_commit: str
    partitions: tuple[FullNode6PartitionProjectionV1, ...]
    closure_bundle: _formal.Q1ClosureBundleV1
    archive_projection_result: _formal.Q1ArchiveProjectionResultV1
    actual_preflight_peak_scratch_bytes: int
    actual_preflight_scratch_live_bytes_at_terminal: int
    diagnostic_only: bool = True
    quotient_measurements_per_signature: int = 1
    projection_remeasures_quotient: bool = False
    q1_state: str = "NOT_RUN"
    q1_gate_count: int = 0
    q1_gate_mask: int = 0
    q1_formal_roots: None = None
    q1_output_slots: tuple[None, ...] = (None,) * 8
    target_truth_accessed: bool = False
    split_accessed: bool = False
    role_evaluation_performed: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.partitions) is not tuple
            or tuple(row.input_signature_id for row in self.partitions) != (1, 2)
            or type(self.closure_bundle) is not _formal.Q1ClosureBundleV1
            or type(self.archive_projection_result)
            is not _formal.Q1ArchiveProjectionResultV1
            or type(self.actual_preflight_peak_scratch_bytes) is not int
            or self.actual_preflight_peak_scratch_bytes <= 0
            or self.actual_preflight_scratch_live_bytes_at_terminal != 0
            or self.diagnostic_only is not True
            or self.quotient_measurements_per_signature != 1
            or self.projection_remeasures_quotient is not False
            or self.q1_state != "NOT_RUN"
            or self.q1_gate_count != 0
            or self.q1_gate_mask != 0
            or self.q1_formal_roots is not None
            or self.q1_output_slots != (None,) * 8
            or self.target_truth_accessed
            or self.split_accessed
            or self.role_evaluation_performed
        ):
            _fail("REJECT_FULL_NODE6_PROJECTION", "aggregate authority boundary differs")
        if not self.capsule_digest.startswith("sha256:") or len(self.capsule_digest) != 71:
            _fail("REJECT_FULL_NODE6_PROJECTION", "capsule digest differs")
        if len(self.source_commit) != 40:
            _fail("REJECT_FULL_NODE6_PROJECTION", "source commit differs")
        _validate_formal_aggregate_assembly_v1(self)

    @property
    def diagnostic_root(self) -> bytes:
        return content_hash(
            RESULT_ROOT_DOMAIN,
            (
                1,
                SCHEMA_VERSION.encode("ascii"),
                self.capsule_digest.encode("ascii"),
                self.source_commit.encode("ascii"),
                tuple(row.canonical_object() for row in self.partitions),
                self.closure_bundle.canonical_object(),
                self.archive_projection_result.canonical_object(),
                self.actual_preflight_peak_scratch_bytes,
                self.actual_preflight_scratch_live_bytes_at_terminal,
                self.diagnostic_only,
                self.quotient_measurements_per_signature,
                self.projection_remeasures_quotient,
                self.q1_state.encode("ascii"),
                self.q1_gate_count,
                self.q1_gate_mask,
                self.q1_formal_roots,
                self.q1_output_slots,
                self.target_truth_accessed,
                self.split_accessed,
                self.role_evaluation_performed,
            ),
        )


def _ordered_chunk_manifest_objects_v1(
    streams: tuple[_archive.Q1CountingDiscardStreamV1, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        manifest.canonical_object()
        for stream in streams
        for manifest in stream.chunk_manifests
    )


def _framed_object_bytes_v1(value: object) -> int:
    return len(_formal.frame_canonical_record_v1(value))


def _require_frozen_projection_profile_v1() -> None:
    expected = _formal.projection_profile_root_v1(
        semantic_binding_root=FROZEN_SEMANTIC_BINDING_ROOT,
        coverage_registry_root=rfc6962_root(
            _formal.expected_coverage_registry_v1()
        ),
        resource_guard_registry=_formal.Q1_RESOURCE_GUARD_REGISTRY,
    )
    if expected != FROZEN_PROJECTION_PROFILE_ROOT:
        _fail(
            "FAIL_FULL_NODE6_FORMAL_ASSEMBLY",
            "frozen semantic/profile binding differs",
        )


def _assemble_formal_partition_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    streams: tuple[_archive.Q1CountingDiscardStreamV1, ...],
    coverage: CoverageDiagnosticV1,
) -> tuple[
    _formal.Q1FixedPointRecordV1,
    bytes,
    _formal.Q1SignatureArchiveManifestV1,
    _formal.Q1ProjectionPartitionRowV1,
]:
    """Strictly assemble the frozen formal projection DAG from same-run data."""

    _require_frozen_projection_profile_v1()
    full_node6_shape = (
        snapshot.terminal_status == _full.TERMINAL_SATURATED
        and snapshot.limits.maximum_ast_depth == 3
        and snapshot.limits.maximum_ast_node_count == 6
    )
    bounded_node3_kat_shape = (
        snapshot.terminal_status
        == _capacity.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED
        and snapshot.limits.maximum_ast_depth == 3
        and snapshot.limits.maximum_ast_node_count == 3
    )
    if (
        not (full_node6_shape or bounded_node3_kat_shape)
        or len(snapshot.depth_barriers) != 5
    ):
        _fail(
            "REJECT_FULL_NODE6_FORMAL_ASSEMBLY",
            "snapshot is neither saturated depth3/node6 nor bounded node3 KAT",
        )
    boundary = snapshot.depth_barriers[-1]
    prior = snapshot.depth_barriers[-2]
    if (
        boundary.depth != 4
        or boundary.barrier_kind != "STRUCTURAL_BOUNDARY"
        or any(
            getattr(boundary, name) != 0
            for name in (
                "eligible_raw_application_count",
                "strict_admitted_application_count",
                "rewrite_collapse_count",
                "new_behavior_class_count",
                "new_signature_cohort_count",
                "continuation_bank_mutation_count",
            )
        )
        or boundary.behavior_class_count_after_barrier
        != prior.behavior_class_count_after_barrier
        or boundary.signature_cohort_count_after_barrier
        != prior.signature_cohort_count_after_barrier
        or boundary.continuation_bank_point_count_after_barrier
        != prior.continuation_bank_point_count_after_barrier
        or boundary.visible_frontier_point_count_after_barrier
        != prior.visible_frontier_point_count_after_barrier
        or coverage.all_eligible_covered is not True
    ):
        _fail(
            "REJECT_FULL_NODE6_FORMAL_ASSEMBLY",
            "fixed-point boundary evidence differs",
        )
    descriptors = tuple(stream.descriptor for stream in streams)
    roots = tuple(descriptor.archive_root for descriptor in descriptors)
    fixed_point = _formal.Q1FixedPointRecordV1(
        input_signature_id=snapshot.input_signature_id,
        universe_root=snapshot.universe_root,
        projection_profile_root=FROZEN_PROJECTION_PROFILE_ROOT,
        raw_application_count=snapshot.raw_operator_application_count,
        strict_admitted_count=snapshot.strict_admitted_application_count,
        rewrite_collapse_count=snapshot.rewrite_collapse_count,
        maximum_depth=3,
        maximum_nodes=6,
        structural_boundary_depth=4,
        work_queue_empty=True,
        zero_delta_full_boundary=True,
        all_eligible_covered=True,
        final_class_delta=boundary.new_behavior_class_count,
        final_cohort_delta=boundary.new_signature_cohort_count,
        final_frontier_delta=(
            boundary.visible_frontier_point_count_after_barrier
            - prior.visible_frontier_point_count_after_barrier
        ),
        final_bank_delta=boundary.continuation_bank_mutation_count,
        program_count=snapshot.continuation_bank_point_count,
        class_count=snapshot.behavior_class_count,
        cohort_count=snapshot.signature_cohort_count,
        bank_point_count=snapshot.continuation_bank_point_count,
        frontier_point_count=snapshot.visible_frontier_point_count,
        maximum_bank_points_per_class=snapshot.maximum_bank_points_per_class,
        maximum_frontier_points_per_class=(
            snapshot.maximum_frontier_points_per_class
        ),
        program_archive_root=roots[0],
        bank_archive_root=roots[1],
        class_archive_root=roots[2],
        coverage_archive_root=roots[3],
    )
    saturation_state_root = _formal.signature_saturation_state_root_v1(
        input_signature_id=snapshot.input_signature_id,
        universe_root=snapshot.universe_root,
        semantic_binding_root=FROZEN_SEMANTIC_BINDING_ROOT,
        projection_profile_root=FROZEN_PROJECTION_PROFILE_ROOT,
        program_archive_root=roots[0],
        cohort_archive_root=roots[1],
        class_archive_root=roots[2],
        coverage_archive_root=roots[3],
        fixed_point_record_root=fixed_point.record_root,
    )
    chunk_objects = _ordered_chunk_manifest_objects_v1(streams)
    signature_manifest = _formal.Q1SignatureArchiveManifestV1(
        input_signature_id=snapshot.input_signature_id,
        universe_root=snapshot.universe_root,
        universe_row_count=snapshot.universe_row_count,
        semantic_binding_root=FROZEN_SEMANTIC_BINDING_ROOT,
        projection_profile_root=FROZEN_PROJECTION_PROFILE_ROOT,
        stream_descriptors=descriptors,
        fixed_point_record_root=fixed_point.record_root,
        saturation_state_root=saturation_state_root,
        chunk_manifest_count=len(chunk_objects),
        chunk_manifest_archive_root=rfc6962_root(chunk_objects),
    )
    record_stream_bytes = sum(
        descriptor.framed_stream_bytes for descriptor in descriptors
    )
    chunk_manifest_stream_bytes = sum(
        _framed_object_bytes_v1(value) for value in chunk_objects
    )
    fixed_point_frame_bytes = _framed_object_bytes_v1(
        fixed_point.canonical_object()
    )
    signature_manifest_frame_bytes = _framed_object_bytes_v1(
        signature_manifest.canonical_object()
    )
    stream_commitments = tuple(stream.diagnostic_commitment for stream in streams)
    external_roots = tuple(
        stream.external_sort_projection.diagnostic_root for stream in streams
    )
    peak_scratch = max(
        stream.external_sort_projection.charged_scratch_high_water_bytes
        for stream in streams
    )
    partition_stream_commitment = _formal.partition_stream_commitment_v1(
        input_signature_id=snapshot.input_signature_id,
        universe_root=snapshot.universe_root,
        raw_application_count=snapshot.raw_operator_application_count,
        behavior_class_count=snapshot.behavior_class_count,
        cohort_count=snapshot.signature_cohort_count,
        bank_point_count=snapshot.continuation_bank_point_count,
        frontier_point_count=snapshot.visible_frontier_point_count,
        maximum_bank_points_per_class=snapshot.maximum_bank_points_per_class,
        maximum_frontier_points_per_class=(
            snapshot.maximum_frontier_points_per_class
        ),
        program_record_count=descriptors[0].record_count,
        coverage_record_count=descriptors[3].record_count,
        projected_record_stream_bytes=record_stream_bytes,
        projected_chunk_manifest_stream_bytes=chunk_manifest_stream_bytes,
        stream_diagnostic_commitments=stream_commitments,
    )
    external_sort_root = _formal.partition_external_sort_root_v1(
        input_signature_id=snapshot.input_signature_id,
        universe_root=snapshot.universe_root,
        external_sort_stream_roots=external_roots,
        projected_peak_scratch_bytes=peak_scratch,
    )
    partition_payload_bytes = (
        record_stream_bytes
        + chunk_manifest_stream_bytes
        + fixed_point_frame_bytes
        + signature_manifest_frame_bytes
    )
    partition_row = _formal.Q1ProjectionPartitionRowV1(
        input_signature_id=snapshot.input_signature_id,
        universe_root=snapshot.universe_root,
        raw_application_count=snapshot.raw_operator_application_count,
        behavior_class_count=snapshot.behavior_class_count,
        cohort_count=snapshot.signature_cohort_count,
        bank_point_count=snapshot.continuation_bank_point_count,
        frontier_point_count=snapshot.visible_frontier_point_count,
        maximum_bank_points_per_class=snapshot.maximum_bank_points_per_class,
        maximum_frontier_points_per_class=(
            snapshot.maximum_frontier_points_per_class
        ),
        peak_work_queue_points=snapshot.peak_work_queue_points,
        program_record_count=descriptors[0].record_count,
        coverage_record_count=descriptors[3].record_count,
        projected_record_stream_bytes=record_stream_bytes,
        projected_chunk_manifest_stream_bytes=chunk_manifest_stream_bytes,
        projected_fixed_point_frame_bytes=fixed_point_frame_bytes,
        projected_signature_manifest_frame_bytes=signature_manifest_frame_bytes,
        projected_partition_payload_bytes=partition_payload_bytes,
        projected_peak_scratch_bytes=peak_scratch,
        stream_diagnostic_commitments=stream_commitments,
        diagnostic_stream_commitment=partition_stream_commitment,
        external_sort_stream_roots=external_roots,
        external_sort_projection_root=external_sort_root,
    )
    return (
        fixed_point,
        saturation_state_root,
        signature_manifest,
        partition_row,
    )


def _validate_formal_partition_assembly_v1(
    partition: FullNode6PartitionProjectionV1,
) -> None:
    streams = partition.streams
    descriptors = tuple(stream.descriptor for stream in streams)
    roots = tuple(descriptor.archive_root for descriptor in descriptors)
    fixed = partition.fixed_point_record
    manifest = partition.signature_archive_manifest
    row = partition.projection_partition_row
    chunk_objects = _ordered_chunk_manifest_objects_v1(streams)
    expected_saturation = _formal.signature_saturation_state_root_v1(
        input_signature_id=partition.input_signature_id,
        universe_root=partition.universe_root,
        semantic_binding_root=FROZEN_SEMANTIC_BINDING_ROOT,
        projection_profile_root=FROZEN_PROJECTION_PROFILE_ROOT,
        program_archive_root=roots[0],
        cohort_archive_root=roots[1],
        class_archive_root=roots[2],
        coverage_archive_root=roots[3],
        fixed_point_record_root=fixed.record_root,
    )
    if (
        fixed.input_signature_id != partition.input_signature_id
        or fixed.universe_root != partition.universe_root
        or fixed.projection_profile_root != FROZEN_PROJECTION_PROFILE_ROOT
        or fixed.program_archive_root != roots[0]
        or fixed.bank_archive_root != roots[1]
        or fixed.class_archive_root != roots[2]
        or fixed.coverage_archive_root != roots[3]
        or partition.saturation_state_root != expected_saturation
        or manifest.input_signature_id != partition.input_signature_id
        or manifest.universe_root != partition.universe_root
        or manifest.semantic_binding_root != FROZEN_SEMANTIC_BINDING_ROOT
        or manifest.projection_profile_root != FROZEN_PROJECTION_PROFILE_ROOT
        or manifest.stream_descriptors != descriptors
        or manifest.fixed_point_record_root != fixed.record_root
        or manifest.saturation_state_root != expected_saturation
        or manifest.chunk_manifest_count != len(chunk_objects)
        or manifest.chunk_manifest_archive_root != rfc6962_root(chunk_objects)
        or row.input_signature_id != partition.input_signature_id
        or row.universe_root != partition.universe_root
        or row.raw_application_count != fixed.raw_application_count
        or row.behavior_class_count != fixed.class_count
        or row.cohort_count != fixed.cohort_count
        or row.bank_point_count != fixed.bank_point_count
        or row.frontier_point_count != fixed.frontier_point_count
        or row.maximum_bank_points_per_class
        != fixed.maximum_bank_points_per_class
        or row.maximum_frontier_points_per_class
        != fixed.maximum_frontier_points_per_class
        or row.program_record_count != descriptors[0].record_count
        or row.coverage_record_count != descriptors[3].record_count
        or row.projected_record_stream_bytes
        != sum(descriptor.framed_stream_bytes for descriptor in descriptors)
        or row.projected_chunk_manifest_stream_bytes
        != sum(_framed_object_bytes_v1(value) for value in chunk_objects)
        or row.projected_fixed_point_frame_bytes
        != _framed_object_bytes_v1(fixed.canonical_object())
        or row.projected_signature_manifest_frame_bytes
        != _framed_object_bytes_v1(manifest.canonical_object())
        or row.stream_diagnostic_commitments
        != tuple(stream.diagnostic_commitment for stream in streams)
        or row.external_sort_stream_roots
        != tuple(stream.external_sort_projection.diagnostic_root for stream in streams)
    ):
        _fail(
            "FAIL_FULL_NODE6_FORMAL_ASSEMBLY",
            "partition formal DAG replay differs",
        )


def _validate_formal_aggregate_assembly_v1(
    result: FullNode6ProjectionResultV1,
) -> None:
    signature_rows = tuple(
        (
            partition.input_signature_id,
            partition.universe_root,
            partition.signature_archive_manifest.manifest_root,
            partition.saturation_state_root,
        )
        for partition in result.partitions
    )
    projection = result.archive_projection_result
    if (
        result.closure_bundle.semantic_binding_root
        != FROZEN_SEMANTIC_BINDING_ROOT
        or result.closure_bundle.projection_profile_root
        != FROZEN_PROJECTION_PROFILE_ROOT
        or result.closure_bundle.signature_rows != signature_rows
        or projection.semantic_binding_root != FROZEN_SEMANTIC_BINDING_ROOT
        or projection.projection_profile_root != FROZEN_PROJECTION_PROFILE_ROOT
        or projection.partition_rows
        != tuple(partition.projection_partition_row for partition in result.partitions)
        or projection.projected_closure_bundle_frame_bytes
        != _framed_object_bytes_v1(result.closure_bundle.canonical_object())
    ):
        _fail(
            "FAIL_FULL_NODE6_FORMAL_ASSEMBLY",
            "aggregate formal DAG replay differs",
        )


def formal_wire_projection_json_v1(
    partition: FullNode6PartitionProjectionV1,
) -> dict[str, object]:
    projection_row = partition.projection_partition_row
    rows: list[dict[str, object]] = []
    for (name, kind), stream in zip(STREAM_REGISTRY, partition.streams, strict=True):
        if stream.stream_kind_id is not kind:
            raise AssertionError("formal stream registry drift")
        descriptor = stream.descriptor
        sort = stream.external_sort_projection
        rows.append(
            {
                "archive_root": "sha256:" + descriptor.archive_root.hex(),
                "canonical_record_payload_bytes": (
                    stream.canonical_record_payload_bytes
                ),
                "chunk_count": descriptor.chunk_count,
                "chunk_manifest_subtree_root": (
                    "sha256:" + descriptor.chunk_manifest_subtree_root.hex()
                ),
                "counting_discard_commitment": (
                    "sha256:" + stream.diagnostic_commitment.hex()
                ),
                "external_sort_projection_root": (
                    "sha256:" + sort.diagnostic_root.hex()
                ),
                "framed_bytes": descriptor.framed_stream_bytes,
                "record_count": descriptor.record_count,
                "retained_framed_blob_bytes": 0,
                "retained_framed_blob_count": 0,
                "stream_kind": name.upper(),
                "stream_kind_id": int(kind),
            }
        )
    return {
        "diagnostic_only": True,
        "formal_archive_materialized": False,
        "projected_chunk_manifest_stream_bytes": (
            projection_row.projected_chunk_manifest_stream_bytes
        ),
        "projected_fixed_point_frame_bytes": (
            projection_row.projected_fixed_point_frame_bytes
        ),
        "projected_partition_payload_bytes": (
            projection_row.projected_partition_payload_bytes
        ),
        "projected_peak_formal_scratch_bytes": (
            projection_row.projected_peak_scratch_bytes
        ),
        "projected_record_stream_bytes": (
            projection_row.projected_record_stream_bytes
        ),
        "projected_signature_manifest_frame_bytes": (
            projection_row.projected_signature_manifest_frame_bytes
        ),
        "projection_kind": "FROZEN_FORMAL_WIRE_COUNTING_DISCARD_NOT_MATERIALIZED",
        "schema_version": (
            "hegel-phase3a-q1-full-node6-formal-wire-projection/1"
        ),
        "streams": rows,
    }


def _require_same_run_snapshot_v1(
    diagnostic: _full.FullNode6PartitionDiagnosticV1,
    snapshot: _snapshot.Q1PartitionSnapshotV1,
) -> None:
    """Check the returned pair structurally, without replaying behavior."""

    if type(snapshot) is not _snapshot.Q1PartitionSnapshotV1:
        _fail("REJECT_FULL_NODE6_SAME_RUN", "snapshot exact type differs")
    pairs = (
        ("input_signature_id", snapshot.input_signature_id),
        ("universe_row_count", snapshot.universe_row_count),
        ("terminal_status", snapshot.terminal_status),
        ("raw_operator_application_count", snapshot.raw_operator_application_count),
        ("strict_admitted_application_count", snapshot.strict_admitted_application_count),
        ("rewrite_collapse_count", snapshot.rewrite_collapse_count),
        ("behavior_class_count", snapshot.behavior_class_count),
        ("signature_cohort_count", snapshot.signature_cohort_count),
        ("continuation_bank_point_count", snapshot.continuation_bank_point_count),
        ("visible_frontier_point_count", snapshot.visible_frontier_point_count),
        ("maximum_bank_points_per_class", snapshot.maximum_bank_points_per_class),
        ("maximum_frontier_points_per_class", snapshot.maximum_frontier_points_per_class),
        ("peak_work_queue_points", snapshot.peak_work_queue_points),
        ("peak_saturation_round_count", snapshot.peak_saturation_round_count),
    )
    if any(getattr(diagnostic, name) != value for name, value in pairs):
        _fail("REJECT_FULL_NODE6_SAME_RUN", "diagnostic/snapshot counters differ")
    if diagnostic.universe_root != "sha256:" + snapshot.universe_root.hex():
        _fail("REJECT_FULL_NODE6_SAME_RUN", "diagnostic/snapshot universe differs")
    if (
        snapshot.q1_state != "NOT_RUN"
        or snapshot.q1_gate_count != 0
        or snapshot.q1_gate_mask != 0
        or snapshot.q1_formal_roots is not None
        or snapshot.q1_receipt is not None
        or snapshot.target_truth_accessed
        or snapshot.split_accessed
        or snapshot.role_evaluation_performed
    ):
        _fail("REJECT_FULL_NODE6_SAME_RUN", "snapshot authority boundary differs")


class _Rfc6962Frontier:
    """Append canonical record bytes without retaining the record sequence."""

    def __init__(self) -> None:
        self._levels: list[bytes | None] = []
        self.count = 0

    def append_canonical_record(self, payload: bytes) -> None:
        node = sha256(b"\x00" + payload).digest()
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            left = self._levels[level]
            assert left is not None
            node = sha256(b"\x01" + left + node).digest()
            self._levels[level] = None
            level += 1
        if level == len(self._levels):
            self._levels.append(node)
        else:
            self._levels[level] = node
        self.count += 1

    def root(self) -> bytes:
        if self.count == 0:
            return sha256(b"").digest()
        output: bytes | None = None
        for node in self._levels:
            if node is None:
                continue
            output = node if output is None else sha256(b"\x01" + node + output).digest()
        assert output is not None
        return output


class _RunReader:
    def __init__(self, path: Path) -> None:
        self._source = path.open("rb")

    def close(self) -> None:
        self._source.close()

    def next_row(self) -> tuple[bytes, bytes, bytes] | None:
        group = self._source.read(3)
        if not group:
            return None
        if len(group) != 3:
            _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "truncated group key")
        header = self._source.read(8)
        if len(header) != 8:
            _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "truncated row header")
        key_length = int.from_bytes(header[:4], "big")
        payload_length = int.from_bytes(header[4:], "big")
        if key_length < 1 or payload_length < 1:
            _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "empty sort row")
        key = self._source.read(key_length)
        payload = self._source.read(payload_length)
        if len(key) != key_length or len(payload) != payload_length:
            _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "truncated sort row")
        return group, key, payload


class _ScratchTrackerV1:
    """Track actual logical bytes of every live regular scratch file."""

    def __init__(self) -> None:
        self.live: dict[Path, int] = {}
        self.peak_logical_bytes = 0

    def create(self, path: Path) -> None:
        if path in self.live:
            _fail("FAIL_FULL_NODE6_SCRATCH_TRACKER", "scratch file created twice")
        try:
            status = path.stat(follow_symlinks=False)
        except OSError as error:
            _fail("INCONCLUSIVE_FULL_NODE6_SCRATCH_TRACKER", str(error))
        if not status.st_mode or not path.is_file() or status.st_size != 0:
            _fail(
                "FAIL_FULL_NODE6_SCRATCH_TRACKER",
                "new scratch path is not an empty regular file",
            )
        self.live[path] = 0

    def reconcile(self) -> None:
        for path, prior in tuple(self.live.items()):
            try:
                status = path.stat(follow_symlinks=False)
            except OSError as error:
                _fail("INCONCLUSIVE_FULL_NODE6_SCRATCH_TRACKER", str(error))
            if not path.is_file() or status.st_size < prior:
                _fail(
                    "FAIL_FULL_NODE6_SCRATCH_TRACKER",
                    "live scratch file disappeared, changed type, or shrank",
                )
            self.live[path] = status.st_size
        self.peak_logical_bytes = max(
            self.peak_logical_bytes,
            sum(self.live.values()),
        )

    def release_after_unlink(self, path: Path) -> None:
        if path not in self.live or path.exists():
            _fail(
                "FAIL_FULL_NODE6_SCRATCH_TRACKER",
                "scratch release does not follow exact unlink",
            )
        del self.live[path]

    def terminal_live_bytes(self) -> int:
        self.reconcile()
        return sum(self.live.values())

    def require_terminal_empty(self) -> None:
        if self.terminal_live_bytes() != 0 or self.live:
            _fail(
                "FAIL_FULL_NODE6_SCRATCH_TRACKER",
                "bounded projection left live scratch files",
            )

    def release_after_transient_tree_cleanup(self) -> None:
        """Prove cleanup removed every registered file after an exception."""

        for path in tuple(self.live):
            if path.exists():
                _fail(
                    "FAIL_FULL_NODE6_SCRATCH_TRACKER",
                    "transient cleanup left a registered scratch file",
                )
            del self.live[path]
        self.require_terminal_empty()


class _SpillSorter:
    """Deterministic bounded-memory run generation and 16-way merge."""

    def __init__(
        self,
        root: Path,
        channel: str,
        tracker: _ScratchTrackerV1 | None = None,
    ) -> None:
        self.root = root
        self.channel = channel
        self.tracker = tracker
        self.buffer: list[tuple[bytes, bytes, bytes]] = []
        self.buffer_bytes = 0
        self.runs: list[Path] = []
        self.sequence = 0

    @staticmethod
    def _write_row(
        target: object,
        row: tuple[bytes, bytes, bytes],
    ) -> None:
        group, key, payload = row
        if len(group) != 3 or not key or not payload:
            _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "sort row shape differs")
        if len(key) >= 1 << 32 or len(payload) >= 1 << 32:
            _fail("INCONCLUSIVE_FULL_NODE6_COVERAGE_SORT", "sort row exceeds u32")
        target.write(group)  # type: ignore[attr-defined]
        target.write(len(key).to_bytes(4, "big"))  # type: ignore[attr-defined]
        target.write(len(payload).to_bytes(4, "big"))  # type: ignore[attr-defined]
        target.write(key)  # type: ignore[attr-defined]
        target.write(payload)  # type: ignore[attr-defined]

    def add(self, group: bytes, key: bytes, payload: bytes) -> None:
        row_bytes = 11 + len(key) + len(payload)
        if self.buffer and self.buffer_bytes + row_bytes > COVERAGE_SORT_BUFFER_BYTES:
            self._flush()
        self.buffer.append((group, key, payload))
        self.buffer_bytes += row_bytes

    def _new_path(self, level: int) -> Path:
        path = self.root / f"{self.channel}-{level:04d}-{self.sequence:08d}.run"
        self.sequence += 1
        return path

    def _flush(self) -> None:
        if not self.buffer:
            return
        self.buffer.sort()
        path = self._new_path(0)
        try:
            with path.open("xb") as target:
                if self.tracker is not None:
                    self.tracker.create(path)
                for row in self.buffer:
                    self._write_row(target, row)
        except OSError as error:
            _fail("INCONCLUSIVE_FULL_NODE6_COVERAGE_SORT", str(error))
        self.runs.append(path)
        self.buffer.clear()
        self.buffer_bytes = 0

    def _merge_group(self, paths: tuple[Path, ...], level: int) -> Path:
        output = self._new_path(level)
        readers = [_RunReader(path) for path in paths]
        heap: list[tuple[bytes, bytes, bytes, int]] = []
        try:
            for index, reader in enumerate(readers):
                row = reader.next_row()
                if row is not None:
                    heapq.heappush(heap, (*row, index))
            with output.open("xb") as target:
                if self.tracker is not None:
                    self.tracker.create(output)
                while heap:
                    group, key, payload, index = heapq.heappop(heap)
                    self._write_row(target, (group, key, payload))
                    row = readers[index].next_row()
                    if row is not None:
                        heapq.heappush(heap, (*row, index))
        except OSError as error:
            _fail("INCONCLUSIVE_FULL_NODE6_COVERAGE_SORT", str(error))
        finally:
            for reader in readers:
                reader.close()
        for path in paths:
            _unlink_run_v1(path, self.tracker)
        return output

    def finish(self) -> Path | None:
        self._flush()
        current = tuple(self.runs)
        level = 1
        while len(current) > 1:
            output: list[Path] = []
            for start in range(0, len(current), COVERAGE_SORT_MERGE_FAN_IN):
                group = current[start : start + COVERAGE_SORT_MERGE_FAN_IN]
                output.append(group[0] if len(group) == 1 else self._merge_group(group, level))
            current = tuple(output)
            level += 1
        return current[0] if current else None


def _iter_run_rows_v1(path: Path | None) -> Iterator[tuple[bytes, bytes, bytes]]:
    if path is None:
        return
    reader = _RunReader(path)
    try:
        while True:
            row = reader.next_row()
            if row is None:
                return
            yield row
    finally:
        reader.close()


def _unlink_run_v1(
    path: Path | None,
    tracker: _ScratchTrackerV1 | None = None,
) -> None:
    if path is None:
        return
    if tracker is not None:
        tracker.reconcile()
    try:
        path.unlink()
    except OSError as error:
        _fail("INCONCLUSIVE_FULL_NODE6_COVERAGE_SORT", str(error))
    if tracker is not None:
        tracker.release_after_unlink(path)


def _lex_tuple_key_v1(values: tuple[bytes, ...]) -> bytes:
    """Encode byte-string tuple order without length-prefix reordering."""

    output = bytearray()
    for value in values:
        if type(value) is not bytes:
            _fail("FAIL_FULL_NODE6_SORT_KEY", "tuple key component differs")
        for byte in value:
            output.extend((1, byte))
        output.append(0)
    return bytes(output)


def _cbor_head_v1(major: int, value: int) -> bytes:
    if type(value) is not int or value < 0 or value >= 1 << 64:
        _fail("INCONCLUSIVE_FULL_NODE6_CBOR", "CBOR length is outside uint64")
    prefix = major << 5
    if value <= 23:
        return bytes((prefix | value,))
    if value <= 0xFF:
        return bytes((prefix | 24, value))
    if value <= 0xFFFF:
        return bytes((prefix | 25,)) + value.to_bytes(2, "big")
    if value <= 0xFFFFFFFF:
        return bytes((prefix | 26,)) + value.to_bytes(4, "big")
    return bytes((prefix | 27,)) + value.to_bytes(8, "big")


@dataclass(frozen=True, slots=True)
class _ExternalRunRangeV1:
    level: int
    index: int
    offset: int
    payload_bytes: int
    record_count: int

    @property
    def file_id(self) -> bytes:
        return f"level-{self.level:04d}-run-{self.index:08d}".encode("ascii")


def _hash_file_range_v1(path: Path, offset: int, length: int) -> bytes:
    digest = sha256()
    try:
        with path.open("rb") as source:
            source.seek(offset)
            remaining = length
            while remaining:
                payload = source.read(min(1024 * 1024, remaining))
                if not payload:
                    _fail(
                        "FAIL_FULL_NODE6_EXTERNAL_SORT",
                        "external-sort spool is truncated",
                    )
                digest.update(payload)
                remaining -= len(payload)
    except OSError as error:
        _fail("INCONCLUSIVE_FULL_NODE6_EXTERNAL_SORT", str(error))
    return digest.digest()


class _ExternalSortProjectionBuilderV1:
    """Exact frozen external-sort projection over already canonical rows."""

    def __init__(
        self,
        scratch: Path,
        channel: str,
        *,
        tracker: _ScratchTrackerV1,
        input_signature_id: int,
        stream_kind_id: _formal.ArchiveStreamKindId,
        expected_count: int,
    ) -> None:
        self.path = scratch / f"formal-{channel}-external-rows.run"
        try:
            self.target = self.path.open("xb", buffering=0)
        except OSError as error:
            _fail("INCONCLUSIVE_FULL_NODE6_EXTERNAL_SORT", str(error))
        tracker.create(self.path)
        self.tracker = tracker
        self.input_signature_id = input_signature_id
        self.stream_kind_id = stream_kind_id
        self.expected_count = expected_count
        self.previous_key: bytes | None = None
        self.count = 0
        self.payload_bytes = 0
        self.offset = 0
        self.pending_offset = 0
        self.pending_bytes = 0
        self.pending_count = 0
        self.initial: list[_ExternalRunRangeV1] = []
        self.sorted_digest = sha256()
        self.sorted_digest.update(
            _external.SORTED_STREAM_ROOT_DOMAIN.encode("utf-8") + b"\x00"
        )
        self.sorted_digest.update(_cbor_head_v1(4, expected_count))

    def _close_initial(self) -> None:
        if not self.pending_count:
            return
        self.initial.append(
            _ExternalRunRangeV1(
                0,
                len(self.initial),
                self.pending_offset,
                self.pending_bytes,
                self.pending_count,
            )
        )
        self.pending_offset += self.pending_bytes
        self.pending_bytes = 0
        self.pending_count = 0

    def append(self, key: bytes, record: bytes) -> None:
        if self.previous_key is not None and key <= self.previous_key:
            _fail(
                "FAIL_FULL_NODE6_EXTERNAL_SORT",
                "formal stream sort key repeats or decreases",
            )
        row = _external.external_sort_row_bytes_v1(key, record)
        if len(row) > _formal.EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES:
            _fail(
                "INCONCLUSIVE_FULL_NODE6_EXTERNAL_SORT",
                "one formal external-sort row exceeds the frozen run limit",
            )
        if (
            self.pending_count
            and self.pending_bytes + len(row)
            > _formal.EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES
        ):
            self._close_initial()
        try:
            self.target.write(row)
        except OSError as error:
            _fail("INCONCLUSIVE_FULL_NODE6_EXTERNAL_SORT", str(error))
        self.sorted_digest.update(_cbor_head_v1(4, 2))
        self.sorted_digest.update(_cbor_head_v1(2, len(key)))
        self.sorted_digest.update(key)
        self.sorted_digest.update(_cbor_head_v1(2, len(record)))
        self.sorted_digest.update(record)
        self.previous_key = key
        self.pending_bytes += len(row)
        self.pending_count += 1
        self.payload_bytes += len(row)
        self.offset += len(row)
        self.count += 1

    def finish(self) -> _external.Q1ExternalSortProjectionV1:
        self._close_initial()
        try:
            self.target.flush()
            self.target.close()
        except OSError as error:
            _fail("INCONCLUSIVE_FULL_NODE6_EXTERNAL_SORT", str(error))
        if self.count != self.expected_count or not self.initial:
            _fail(
                "FAIL_FULL_NODE6_EXTERNAL_SORT",
                "formal external-sort cardinality differs",
            )

        live: dict[bytes, int] = {}
        events: list[tuple[object, ...]] = []
        logical_high_water = 0
        charged_high_water = 0

        def charged(size: int) -> int:
            blocks = (
                size + _formal.SCRATCH_ALLOCATION_BLOCK_BYTES - 1
            ) // _formal.SCRATCH_ALLOCATION_BLOCK_BYTES
            return (
                blocks * _formal.SCRATCH_ALLOCATION_BLOCK_BYTES
                + _formal.SCRATCH_METADATA_RESERVE_BYTES_PER_LIVE_FILE
            )

        def event(action: int, file_id: bytes, prior: int, new: int) -> None:
            nonlocal logical_high_water, charged_high_water
            logical = sum(live.values())
            charged_bytes = sum(charged(value) for value in live.values())
            logical_high_water = max(logical_high_water, logical)
            charged_high_water = max(charged_high_water, charged_bytes)
            events.append(
                (
                    1,
                    _external.SCRATCH_EVENT_SCHEMA_ID,
                    len(events),
                    action,
                    file_id,
                    prior,
                    new,
                    logical,
                    charged_bytes,
                )
            )

        def allocate(row: _ExternalRunRangeV1) -> None:
            file_id = row.file_id
            live[file_id] = _formal.EXTERNAL_SORT_RUN_HEADER_BYTES
            event(
                _external.ScratchActionId.ALLOC,
                file_id,
                0,
                _formal.EXTERNAL_SORT_RUN_HEADER_BYTES,
            )
            prior = live[file_id]
            size = _formal.EXTERNAL_SORT_RUN_HEADER_BYTES + row.payload_bytes
            live[file_id] = size
            event(_external.ScratchActionId.GROW, file_id, prior, size)
            event(_external.ScratchActionId.SEAL, file_id, size, size)

        def free(row: _ExternalRunRangeV1) -> None:
            prior = live.pop(row.file_id)
            event(_external.ScratchActionId.FREE, row.file_id, prior, 0)

        manifests: list[tuple[object, ...]] = []

        def append_run(row: _ExternalRunRangeV1) -> None:
            allocate(row)
            manifests.append(
                (
                    1,
                    _external.RUN_MANIFEST_SCHEMA_ID,
                    self.input_signature_id,
                    int(self.stream_kind_id),
                    row.level,
                    row.index,
                    row.record_count,
                    row.payload_bytes,
                    _hash_file_range_v1(
                        self.path,
                        row.offset,
                        row.payload_bytes,
                    ),
                )
            )

        try:
            for row in self.initial:
                append_run(row)
            current = tuple(self.initial)
            level = 0
            while len(current) > 1:
                level += 1
                output: list[_ExternalRunRangeV1] = []
                for start in range(
                    0,
                    len(current),
                    _formal.EXTERNAL_SORT_MERGE_FAN_IN,
                ):
                    group = current[
                        start : start + _formal.EXTERNAL_SORT_MERGE_FAN_IN
                    ]
                    merged = _ExternalRunRangeV1(
                        level,
                        len(output),
                        group[0].offset,
                        sum(row.payload_bytes for row in group),
                        sum(row.record_count for row in group),
                    )
                    append_run(merged)
                    for child in group:
                        free(child)
                    output.append(merged)
                current = tuple(output)
            final_run_bytes = (
                _formal.EXTERNAL_SORT_RUN_HEADER_BYTES
                + current[0].payload_bytes
            )
            free(current[0])
            if live:
                raise AssertionError("formal external-sort scratch ledger remained live")
            return _external.Q1ExternalSortProjectionV1(
                self.input_signature_id,
                self.stream_kind_id,
                self.count,
                self.payload_bytes,
                len(self.initial),
                level,
                final_run_bytes,
                logical_high_water,
                charged_high_water,
                self.sorted_digest.digest(),
                rfc6962_root(tuple(manifests)),
                content_hash(
                    _external.SCRATCH_LEDGER_ROOT_DOMAIN,
                    tuple(events),
                ),
                len(events),
            )
        finally:
            _unlink_run_v1(self.path, self.tracker)


class _SameRunRecordSetRootBuilderV1:
    def __init__(self, snapshot: _snapshot.Q1PartitionSnapshotV1) -> None:
        self.digest = sha256()
        self.digest.update(RECORD_SET_ROOT_DOMAIN.encode("utf-8") + b"\x00")
        self.digest.update(_cbor_head_v1(4, 8))
        for value in (
            1,
            SCHEMA_VERSION.encode("ascii"),
            snapshot.input_signature_id,
            snapshot.universe_root,
        ):
            self.digest.update(canonical_cbor_encode(value))
        self.open_expected: int | None = None
        self.open_count = 0
        self.stream_count = 0

    def begin_stream(self, expected_count: int) -> None:
        if self.open_expected is not None or self.stream_count >= 4:
            raise AssertionError("same-run record-set stream ordering drift")
        self.digest.update(_cbor_head_v1(4, expected_count))
        self.open_expected = expected_count
        self.open_count = 0

    def append(self, canonical_record: bytes) -> None:
        if self.open_expected is None:
            raise AssertionError("same-run record-set stream is not open")
        self.digest.update(canonical_record)
        self.open_count += 1

    def end_stream(self) -> None:
        if self.open_expected is None or self.open_count != self.open_expected:
            _fail(
                "FAIL_FULL_NODE6_RECORD_SET_ROOT",
                "same-run record-set stream cardinality differs",
            )
        self.open_expected = None
        self.stream_count += 1

    def finish(self) -> bytes:
        if self.open_expected is not None or self.stream_count != 4:
            _fail(
                "FAIL_FULL_NODE6_RECORD_SET_ROOT",
                "same-run record-set stream registry differs",
            )
        return self.digest.digest()


class _FormalCountingSinkV1:
    """Old-v1 formal archive sink with bounded chunk and row spools.

    The ordinary diagnostic path leaves ``retained_chunk_consumer`` null and
    therefore preserves the original counting/discard behavior.  The formal
    materializer supplies a consumer which receives each exact frozen framed
    chunk once, at the same deterministic chunk boundary used by the counting
    path.
    """

    def __init__(
        self,
        scratch: Path,
        channel: str,
        *,
        snapshot: _snapshot.Q1PartitionSnapshotV1,
        tracker: _ScratchTrackerV1,
        stream_kind_id: _formal.ArchiveStreamKindId,
        expected_count: int,
        record_set_root: _SameRunRecordSetRootBuilderV1,
        retained_chunk_consumer: _RetainedChunkConsumerV1 | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.stream_kind_id = stream_kind_id
        self.expected_count = expected_count
        self.record_set_root = record_set_root
        self.retained_chunk_consumer = retained_chunk_consumer
        self.record_set_root.begin_stream(expected_count)
        self.archive_frontier = _Rfc6962Frontier()
        self.external = _ExternalSortProjectionBuilderV1(
            scratch,
            channel,
            tracker=tracker,
            input_signature_id=snapshot.input_signature_id,
            stream_kind_id=stream_kind_id,
            expected_count=expected_count,
        )
        self.manifests: list[_formal.Q1ArchiveChunkManifestV1] = []
        self.chunk_payloads: list[bytes] = []
        self.chunk_ids: list[bytes] = []
        self.chunk_framed_bytes = 0
        self.count = 0
        self.payload_bytes = 0
        self.framed_bytes = 0

    def _close_chunk(self) -> None:
        if not self.chunk_payloads:
            return
        blob_hash = sha256()
        blob_hash.update(_formal.FRAMED_BLOB_HASH_DOMAIN.encode("utf-8") + b"\x00")
        blob_hash.update(b"\x81")
        blob_hash.update(_cbor_head_v1(2, self.chunk_framed_bytes))
        subtree = _Rfc6962Frontier()
        for payload in self.chunk_payloads:
            blob_hash.update(len(payload).to_bytes(_formal.FRAME_LENGTH_BYTES, "big"))
            blob_hash.update(payload)
            subtree.append_canonical_record(payload)
        first_record_index = self.count - len(self.chunk_payloads)
        manifest = _formal.Q1ArchiveChunkManifestV1(
            self.snapshot.input_signature_id,
            self.snapshot.universe_root,
            self.stream_kind_id,
            len(self.manifests),
            first_record_index,
            len(self.chunk_payloads),
            self.chunk_ids[0],
            self.chunk_ids[-1],
            subtree.root(),
            blob_hash.digest(),
            self.chunk_framed_bytes,
        )
        self.manifests.append(manifest)
        if self.retained_chunk_consumer is not None:
            framed_blob = b"".join(
                len(payload).to_bytes(_formal.FRAME_LENGTH_BYTES, "big") + payload
                for payload in self.chunk_payloads
            )
            if (
                len(framed_blob) != manifest.framed_blob_length
                or _formal.framed_blob_hash_v1(framed_blob)
                != manifest.framed_blob_hash
            ):
                _fail(
                    "FAIL_FULL_NODE6_FORMAL_MATERIALIZATION",
                    "retained framed chunk differs from counting sink",
                )
            self.retained_chunk_consumer(manifest, framed_blob)
        self.chunk_payloads.clear()
        self.chunk_ids.clear()
        self.chunk_framed_bytes = 0

    def append(self, record: object) -> None:
        if (
            record.input_signature_id != self.snapshot.input_signature_id  # type: ignore[attr-defined]
            or record.universe_root != self.snapshot.universe_root  # type: ignore[attr-defined]
        ):
            _fail("FAIL_FULL_NODE6_FORMAL_STREAM", "formal record binding differs")
        canonical_object = record.canonical_object()  # type: ignore[attr-defined]
        payload = canonical_cbor_encode(canonical_object)
        record_id = record.record_id  # type: ignore[attr-defined]
        sort_key = _archive._stream_sort_key(record, self.stream_kind_id)
        frame_bytes = _formal.FRAME_LENGTH_BYTES + len(payload)
        if frame_bytes > _formal.MAX_CHUNK_FRAMED_BYTES:
            _fail(
                "INCONCLUSIVE_FULL_NODE6_FORMAL_STREAM",
                "one formal record exceeds the frozen chunk limit",
            )
        if self.chunk_payloads and (
            len(self.chunk_payloads) + 1 > _formal.MAX_RECORDS_PER_CHUNK
            or self.chunk_framed_bytes + frame_bytes
            > _formal.MAX_CHUNK_FRAMED_BYTES
        ):
            self._close_chunk()
        self.chunk_payloads.append(payload)
        self.chunk_ids.append(record_id)
        self.chunk_framed_bytes += frame_bytes
        self.archive_frontier.append_canonical_record(payload)
        self.external.append(sort_key, payload)
        self.record_set_root.append(payload)
        self.count += 1
        self.payload_bytes += len(payload)
        self.framed_bytes += frame_bytes

    def finish(self) -> _archive.Q1CountingDiscardStreamV1:
        self._close_chunk()
        if self.count != self.expected_count or not self.manifests:
            _fail(
                "FAIL_FULL_NODE6_FORMAL_STREAM",
                "formal stream cardinality differs",
            )
        self.record_set_root.end_stream()
        manifests = tuple(self.manifests)
        manifest_objects = tuple(row.canonical_object() for row in manifests)
        descriptor = _formal.Q1StreamDescriptorV1(
            self.stream_kind_id,
            self.count,
            self.archive_frontier.root(),
            self.framed_bytes,
            len(manifests),
            rfc6962_root(manifest_objects),
        )
        external = self.external.finish()
        commitment = content_hash(
            _archive.PROJECTED_STREAM_ROOT_DOMAIN,
            (
                1,
                _archive.PROJECTED_STREAM_SCHEMA_ID,
                self.snapshot.input_signature_id,
                self.snapshot.universe_root,
                int(self.stream_kind_id),
                descriptor.canonical_object(),
                manifest_objects,
                external.canonical_object(),
            ),
        )
        return _archive.Q1CountingDiscardStreamV1(
            self.snapshot.input_signature_id,
            self.snapshot.universe_root,
            self.stream_kind_id,
            self.payload_bytes,
            descriptor,
            manifests,
            external,
            commitment,
        )


def _consume_sorted_rows(
    path: Path | None,
    tracker: _ScratchTrackerV1 | None = None,
) -> tuple[dict[bytes, _Rfc6962Frontier], dict[bytes, int]]:
    roots: dict[bytes, _Rfc6962Frontier] = {}
    counts: dict[bytes, int] = {}
    if path is None:
        return roots, counts
    reader = _RunReader(path)
    previous: tuple[bytes, bytes] | None = None
    try:
        while True:
            row = reader.next_row()
            if row is None:
                break
            group, key, payload = row
            if previous == (group, key):
                _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "duplicate ordered key")
            previous = (group, key)
            roots.setdefault(group, _Rfc6962Frontier()).append_canonical_record(payload)
            counts[group] = counts.get(group, 0) + 1
    finally:
        reader.close()
    _unlink_run_v1(path, tracker)
    return roots, counts


def _consume_unique_ast_rows(
    path: Path | None,
    tracker: _ScratchTrackerV1 | None = None,
) -> dict[bytes, int]:
    output: dict[bytes, int] = {}
    if path is None:
        return output
    reader = _RunReader(path)
    previous: tuple[bytes, bytes] | None = None
    try:
        while True:
            row = reader.next_row()
            if row is None:
                break
            group, key, payload = row
            if key != payload:
                _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "AST unique row differs")
            if previous != (group, key):
                output[group] = output.get(group, 0) + 1
            previous = (group, key)
    finally:
        reader.close()
    _unlink_run_v1(path, tracker)
    return output


def _coverage_records_from_final_bank_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    programs: tuple[_formal.Q1RepresentativeProgramRecordV1, ...],
) -> tuple[_formal.Q1SemanticCoverageRecordV1, ...]:
    """Replay application identities only; no behavior vector is evaluated."""

    program_id_by_ast = {row.canonical_ast_cbor: row.program_id for row in programs}
    bank = tuple(
        _capacity._Program(
            ast=decode_shrink6_canonical_ast(row.canonical_ast_cbor),
            behavior=(),
            behavior_key=row.program_id,
        )
        for row in programs
    )
    registry = _formal.expected_coverage_registry_v1()
    counters: dict[bytes, list[int]] = {
        bytes((depth,)) + coverage_code.to_bytes(2, "big"): [0, 0]
        for depth, coverage_code in registry
    }

    with tempfile.TemporaryDirectory(prefix="hegel-q1-coverage-sort-") as directory:
        scratch = Path(directory)
        applications = _SpillSorter(scratch, "applications")
        strict = _SpillSorter(scratch, "strict")
        unique_asts = _SpillSorter(scratch, "asts")

        def append_row(
            depth: int,
            coverage_code: int,
            application_key: tuple[object, ...],
            ast_digest: bytes,
            rewrite_collapsed: bool,
        ) -> None:
            group = bytes((depth,)) + coverage_code.to_bytes(2, "big")
            application_payload = canonical_cbor_encode(application_key)
            application_id = _formal.semantic_application_id_v1(application_key)
            strict_payload = canonical_cbor_encode((application_id, ast_digest))
            applications.add(group, application_payload, application_payload)
            # Sorting by application ID is equivalent to canonical tuple order
            # for fixed-width byte strings and also detects a digest collision.
            strict.add(group, application_id, strict_payload)
            unique_asts.add(group, ast_digest, ast_digest)
            counters[group][0] += 1
            counters[group][1] += int(rewrite_collapsed)

        for coverage_code, ast in enumerate(
            _capacity._frozen_leaf_asts_v1(raw_cap=_capacity.LEAF_COUNT)
        ):
            application_key = _formal.semantic_application_key_v1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                0,
                coverage_code,
                (),
                (),
            )
            append_row(0, coverage_code, application_key, ast.digest, False)

        for depth in range(1, snapshot.limits.maximum_ast_depth + 1):
            for candidate in _capacity._operator_candidates(
                bank,
                target_depth=depth,
                limits=snapshot.limits,
            ):
                try:
                    child_ids = tuple(
                        program_id_by_ast[child.ast.cbor_bytes]
                        for child in candidate.children
                    )
                except KeyError:
                    _fail(
                        "FAIL_FULL_NODE6_COVERAGE_REPLAY",
                        "candidate child is outside final bank",
                    )
                parameters: tuple[object, ...] = (
                    (candidate.operator_code - 0x3000,)
                    if candidate.operator_code in (0x3001, 0x3002)
                    else ()
                )
                application_key = _formal.semantic_application_key_v1(
                    snapshot.input_signature_id,
                    snapshot.universe_root,
                    depth,
                    candidate.operator_code,
                    parameters,
                    child_ids,
                )
                ast = canonicalize_shrink6_source_ast(candidate.source_ast)
                append_row(
                    depth,
                    candidate.operator_code,
                    application_key,
                    ast.digest,
                    ast.value[1] != candidate.expected_node,
                )

        application_roots, application_counts = _consume_sorted_rows(
            applications.finish()
        )
        strict_roots, strict_counts = _consume_sorted_rows(strict.finish())
        unique_ast_counts = _consume_unique_ast_rows(unique_asts.finish())

    rows: list[_formal.Q1SemanticCoverageRecordV1] = []
    empty_root = sha256(b"").digest()
    for depth, coverage_code in registry:
        group = bytes((depth,)) + coverage_code.to_bytes(2, "big")
        count, rewrite_count = counters[group]
        if application_counts.get(group, 0) != count or strict_counts.get(group, 0) != count:
            _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "sorted coverage count differs")
        application_root = (
            application_roots[group].root() if group in application_roots else empty_root
        )
        strict_root = strict_roots[group].root() if group in strict_roots else empty_root
        rows.append(
            _formal.Q1SemanticCoverageRecordV1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                depth,
                coverage_code,
                count,
                application_root,
                count,
                application_root,
                count,
                strict_root,
                unique_ast_counts.get(group, 0),
                rewrite_count,
            )
        )
    output = tuple(rows)
    if (
        sum(row.eligible_application_count for row in output)
        != snapshot.raw_operator_application_count
        or sum(row.strict_admitted_count for row in output)
        != snapshot.strict_admitted_application_count
        or sum(row.rewrite_collapse_count for row in output)
        != snapshot.rewrite_collapse_count
    ):
        _fail("FAIL_FULL_NODE6_COVERAGE_REPLAY", "coverage totals differ from same-run snapshot")
    barriers = {
        row.depth: row
        for row in snapshot.depth_barriers
        if row.depth <= snapshot.limits.maximum_ast_depth
    }
    for depth in range(snapshot.limits.maximum_ast_depth + 1):
        depth_rows = tuple(row for row in output if row.construction_depth == depth)
        barrier = barriers.get(depth)
        if barrier is None or (
            sum(row.eligible_application_count for row in depth_rows)
            != barrier.eligible_raw_application_count
            or sum(row.strict_admitted_count for row in depth_rows)
            != barrier.strict_admitted_application_count
            or sum(row.rewrite_collapse_count for row in depth_rows)
            != barrier.rewrite_collapse_count
        ):
            _fail("FAIL_FULL_NODE6_COVERAGE_REPLAY", "coverage depth totals differ")
    return output


def _u64(value: int) -> bytes:
    if type(value) is not int or not 0 <= value < 1 << 64:
        _fail("REJECT_FULL_NODE6_CAPACITY_PROJECTION", "u64 value differs")
    return value.to_bytes(8, "big")


def _u32(value: int) -> bytes:
    if type(value) is not int or not 0 <= value < 1 << 32:
        _fail("REJECT_FULL_NODE6_CAPACITY_PROJECTION", "u32 value differs")
    return value.to_bytes(4, "big")


def _append_blob(output: bytearray, value: bytes) -> None:
    output.extend(_u64(len(value)))
    output.extend(value)


def _diagnostic_behavior_key_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    row: _snapshot.Q1BehaviorClassSnapshotV1,
) -> bytes:
    output = bytearray(b"HEGEL/Q1/BEHAVIOR/DIAGNOSTIC/V1\x00")
    output.extend(_u64(snapshot.input_signature_id))
    output.append(row.output_sort_id)
    output.extend(_u64(len(row.behavior_cells)))
    for cell in row.behavior_cells:
        if cell.cell_tag == 0:
            output.append(0)
            continue
        value = cell.runtime_value(row.output_sort_id)
        if row.output_sort_id == int(OutputSortId.BOOL):
            if type(value) is not bool:
                _fail("FAIL_FULL_NODE6_CAPACITY_PROJECTION", "Bool cell differs")
            output.extend((1, int(value)))
        elif row.output_sort_id == int(OutputSortId.BIT):
            if type(value) is not int or value not in (0, 1):
                _fail("FAIL_FULL_NODE6_CAPACITY_PROJECTION", "Bit cell differs")
            output.extend((2, value))
        elif row.output_sort_id == int(OutputSortId.SIGN):
            if type(value) is not int or value not in (-1, 0, 1):
                _fail("FAIL_FULL_NODE6_CAPACITY_PROJECTION", "Sign cell differs")
            output.extend((3, value + 1))
        elif row.output_sort_id == int(OutputSortId.BOUNDED_INT):
            if type(value) is not int or not -8 <= value <= 8:
                _fail("FAIL_FULL_NODE6_CAPACITY_PROJECTION", "BoundedInt cell differs")
            output.extend((4, value + 8))
        elif row.output_sort_id == int(OutputSortId.RATIONAL_VALUE):
            numerator = getattr(value, "numerator", None)
            denominator = getattr(value, "denominator", None)
            if type(numerator) is not int or type(denominator) is not int:
                _fail("FAIL_FULL_NODE6_CAPACITY_PROJECTION", "Rational cell differs")
            output.append(5)
            output.extend(numerator.to_bytes(8, "big", signed=True))
            output.extend(denominator.to_bytes(8, "big", signed=True))
        else:
            _fail("FAIL_FULL_NODE6_CAPACITY_PROJECTION", "output sort differs")
    return bytes(output)


def _signature_sort_key_v1(signature: object) -> tuple[int, ...]:
    return (
        int(signature.output_sort_id),  # type: ignore[attr-defined]
        signature.ast_depth,  # type: ignore[attr-defined]
        signature.ast_node_count,  # type: ignore[attr-defined]
        signature.scalar_parameter_occurrence_count,  # type: ignore[attr-defined]
        signature.aggregate_leaf_count,  # type: ignore[attr-defined]
        signature.distinct_bit_slot_bitmap,  # type: ignore[attr-defined]
        signature.scope_clause_count,  # type: ignore[attr-defined]
        signature.top_level_clause_count,  # type: ignore[attr-defined]
        signature.old_law_composition_depth,  # type: ignore[attr-defined]
        int(signature.normalization_profile_id),  # type: ignore[attr-defined]
        signature.mdl_length_q32,  # type: ignore[attr-defined]
    )


def _diagnostic_signature_bytes_v1(signature: object) -> bytes:
    values = _signature_sort_key_v1(signature)
    output = bytearray(b"HEGEL/Q1/CONSTRUCTION_SIGNATURE/DIAGNOSTIC/V1\x00")
    output.append(values[0])
    for value in values[1:9]:
        output.extend(_u32(value))
    output.append(values[9])
    output.extend(_u64(values[10]))
    return bytes(output)


def _capacity_projection_records_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    coverage: tuple[_formal.Q1SemanticCoverageRecordV1, ...],
) -> tuple[tuple[str, tuple[bytes, ...]], ...]:
    programs: list[bytes] = []
    cohorts: list[bytes] = []
    classes: list[bytes] = []
    class_rows = tuple(
        sorted(
            (
                (_diagnostic_behavior_key_v1(snapshot, row), row)
                for row in snapshot.behavior_classes
            ),
            key=lambda item: item[0],
        )
    )
    for behavior_key, class_row in class_rows:
        behavior_digest = sha256(behavior_key).digest()
        ordered_cohorts = tuple(
            sorted(class_row.cohorts, key=lambda row: _signature_sort_key_v1(row.signature))
        )
        class_record = bytearray(b"CLASS\x00")
        _append_blob(class_record, behavior_key)
        class_record.extend(_u64(len(ordered_cohorts)))
        classes.append(bytes(class_record))
        for cohort in ordered_cohorts:
            signature_bytes = _diagnostic_signature_bytes_v1(cohort.signature)
            cohort_record = bytearray(b"COHORT\x00")
            cohort_record.extend(behavior_digest)
            _append_blob(cohort_record, signature_bytes)
            cohort_record.append(int(cohort.visible_frontier_member))
            representatives = tuple(
                sorted(
                    cohort.representatives,
                    key=lambda row: row.canonical_ast_cbor,
                )
            )
            cohort_record.extend(_u64(len(representatives)))
            for representative in representatives:
                cohort_record.extend(representative.canonical_ast_hash)
                program_record = bytearray(b"PROGRAM\x00")
                program_record.extend(behavior_digest)
                program_record.extend(sha256(signature_bytes).digest())
                _append_blob(program_record, representative.canonical_ast_cbor)
                program_record.extend(representative.canonical_ast_hash)
                programs.append(bytes(program_record))
            cohorts.append(bytes(cohort_record))

    coverage_records: list[bytes] = []
    for row in coverage:
        record = bytearray(b"COVERAGE\x00")
        record.append(row.construction_depth)
        record.extend(row.coverage_code.to_bytes(2, "big"))
        record.extend(_u64(row.eligible_application_count))
        record.extend(_u64(row.strict_admitted_count))
        record.extend(_u64(row.rewrite_collapse_count))
        coverage_records.append(bytes(record))
    return (
        ("PROGRAM", tuple(programs)),
        ("COHORT", tuple(cohorts)),
        ("CLASS", tuple(classes)),
        ("COVERAGE", tuple(coverage_records)),
    )


def _capacity_state_projection_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    coverage: tuple[_formal.Q1SemanticCoverageRecordV1, ...],
) -> CapacityStateProjectionV1:
    streams: list[CapacityCountingDiscardStreamV1] = []
    for kind, records in _capacity_projection_records_v1(snapshot, coverage):
        digest = sha256()
        digest.update(CAPACITY_STREAM_DIGEST_DOMAIN)
        digest.update(kind.encode("ascii"))
        payload_bytes = 0
        for record in records:
            digest.update(_u64(len(record)))
            digest.update(record)
            payload_bytes += len(record)
        streams.append(
            CapacityCountingDiscardStreamV1(
                stream_kind=kind,
                record_count=len(records),
                payload_bytes=payload_bytes,
                framed_bytes=payload_bytes + 8 * len(records),
                counting_discard_sha256=digest.hexdigest(),
            )
        )
    combined = sha256()
    combined.update(CAPACITY_STATE_DIGEST_DOMAIN)
    for stream in streams:
        combined.update(stream.stream_kind.encode("ascii"))
        combined.update(_u64(stream.record_count))
        combined.update(_u64(stream.framed_bytes))
        combined.update(stream.counting_discard_sha256.encode("ascii"))
    return CapacityStateProjectionV1(
        streams=tuple(streams),
        combined_state_projection_sha256=combined.hexdigest(),
    )


class _CapacityStreamAccumulatorV1:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.digest = sha256()
        self.digest.update(CAPACITY_STREAM_DIGEST_DOMAIN)
        self.digest.update(kind.encode("ascii"))
        self.count = 0
        self.payload_bytes = 0

    def append(self, record: bytes) -> None:
        if type(record) is not bytes or not record:
            _fail("FAIL_FULL_NODE6_CAPACITY_PROJECTION", "custom record differs")
        self.digest.update(_u64(len(record)))
        self.digest.update(record)
        self.count += 1
        self.payload_bytes += len(record)

    def finish(self) -> CapacityCountingDiscardStreamV1:
        if not self.count:
            _fail("FAIL_FULL_NODE6_CAPACITY_PROJECTION", "custom stream is empty")
        return CapacityCountingDiscardStreamV1(
            stream_kind=self.kind,
            record_count=self.count,
            payload_bytes=self.payload_bytes,
            framed_bytes=self.payload_bytes + 8 * self.count,
            counting_discard_sha256=self.digest.hexdigest(),
        )


def _capacity_state_projection_from_paths_v1(
    paths: tuple[Path | None, Path | None, Path | None],
    coverage: tuple[_formal.Q1SemanticCoverageRecordV1, ...],
    *,
    tracker: _ScratchTrackerV1,
) -> CapacityStateProjectionV1:
    streams: list[CapacityCountingDiscardStreamV1] = []
    for kind, path in zip(("PROGRAM", "COHORT", "CLASS"), paths, strict=True):
        accumulator = _CapacityStreamAccumulatorV1(kind)
        previous: tuple[bytes, bytes] | None = None
        try:
            for group, key, payload in _iter_run_rows_v1(path):
                if group != b"\x00\x00\x00" or (
                    previous is not None and (group, key) <= previous
                ):
                    _fail(
                        "FAIL_FULL_NODE6_CAPACITY_PROJECTION",
                        "custom stream order repeats or decreases",
                    )
                previous = (group, key)
                accumulator.append(payload)
        finally:
            _unlink_run_v1(path, tracker)
        streams.append(accumulator.finish())

    coverage_accumulator = _CapacityStreamAccumulatorV1("COVERAGE")
    for row in coverage:
        record = bytearray(b"COVERAGE\x00")
        record.append(row.construction_depth)
        record.extend(row.coverage_code.to_bytes(2, "big"))
        record.extend(_u64(row.eligible_application_count))
        record.extend(_u64(row.strict_admitted_count))
        record.extend(_u64(row.rewrite_collapse_count))
        coverage_accumulator.append(bytes(record))
    streams.append(coverage_accumulator.finish())

    combined = sha256()
    combined.update(CAPACITY_STATE_DIGEST_DOMAIN)
    for stream in streams:
        combined.update(stream.stream_kind.encode("ascii"))
        combined.update(_u64(stream.record_count))
        combined.update(_u64(stream.framed_bytes))
        combined.update(stream.counting_discard_sha256.encode("ascii"))
    return CapacityStateProjectionV1(
        streams=tuple(streams),
        combined_state_projection_sha256=combined.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class _BankProgramV1:
    program_id: bytes
    program: _capacity._Program


def _iter_bank_programs_v1(path: Path | None) -> Iterator[_BankProgramV1]:
    previous: tuple[bytes, bytes] | None = None
    for group, key, payload in _iter_run_rows_v1(path):
        if group != b"\x00\x00\x00" or (
            previous is not None and (group, key) <= previous
        ):
            _fail("FAIL_FULL_NODE6_BANK_SORT", "bank key repeats or decreases")
        previous = (group, key)
        value = canonical_cbor_decode(payload)
        if (
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not bytes
            or len(value[0]) != 32
            or type(value[1]) is not bytes
        ):
            _fail("FAIL_FULL_NODE6_BANK_SORT", "bank payload differs")
        ast = decode_shrink6_canonical_ast(value[1])
        yield _BankProgramV1(
            value[0],
            _capacity._Program(ast=ast, behavior=(), behavior_key=value[0]),
        )


def _iter_bank_blocks_v1(path: Path | None) -> Iterator[tuple[_BankProgramV1, ...]]:
    block: list[_BankProgramV1] = []
    block_bytes = 0
    for row in _iter_bank_programs_v1(path):
        size = len(row.program.ast.cbor_bytes) + 256
        if block and block_bytes + size > COVERAGE_SORT_BUFFER_BYTES:
            yield tuple(block)
            block.clear()
            block_bytes = 0
        block.append(row)
        block_bytes += size
    if block:
        yield tuple(block)


def _coverage_records_from_bank_paths_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    bank_paths: dict[str, Path | None],
    *,
    scratch_parent: Path,
    tracker: _ScratchTrackerV1,
) -> tuple[_formal.Q1SemanticCoverageRecordV1, ...]:
    """Replay candidate identities from bounded sorted bank views."""

    registry = _formal.expected_coverage_registry_v1()
    counters: dict[bytes, list[int]] = {
        bytes((depth,)) + coverage_code.to_bytes(2, "big"): [0, 0]
        for depth, coverage_code in registry
    }
    with tempfile.TemporaryDirectory(
        prefix="hegel-q1-coverage-sort-",
        dir=scratch_parent,
    ) as directory:
        scratch = Path(directory)
        applications = _SpillSorter(scratch, "applications", tracker)
        strict = _SpillSorter(scratch, "strict", tracker)
        unique_asts = _SpillSorter(scratch, "asts", tracker)

        def append_row(
            depth: int,
            coverage_code: int,
            application_key: tuple[object, ...],
            ast_digest: bytes,
            rewrite_collapsed: bool,
        ) -> None:
            group = bytes((depth,)) + coverage_code.to_bytes(2, "big")
            application_payload = canonical_cbor_encode(application_key)
            application_id = _formal.semantic_application_id_v1(application_key)
            strict_payload = canonical_cbor_encode((application_id, ast_digest))
            applications.add(group, application_payload, application_payload)
            strict.add(group, application_id, strict_payload)
            unique_asts.add(group, ast_digest, ast_digest)
            counters[group][0] += 1
            counters[group][1] += int(rewrite_collapsed)

        for coverage_code, ast in enumerate(
            _capacity._frozen_leaf_asts_v1(raw_cap=_capacity.LEAF_COUNT)
        ):
            application_key = _formal.semantic_application_key_v1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                0,
                coverage_code,
                (),
                (),
            )
            append_row(0, coverage_code, application_key, ast.digest, False)

        def emit_candidate(depth: int, candidate: _capacity._Candidate) -> None:
            child_ids = tuple(
                child.behavior_key for child in candidate.children
            )
            parameters: tuple[object, ...] = (
                (candidate.operator_code - 0x3000,)
                if candidate.operator_code in (0x3001, 0x3002)
                else ()
            )
            application_key = _formal.semantic_application_key_v1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                depth,
                candidate.operator_code,
                parameters,
                child_ids,
            )
            ast = canonicalize_shrink6_source_ast(candidate.source_ast)
            append_row(
                depth,
                candidate.operator_code,
                application_key,
                ast.digest,
                ast.value[1] != candidate.expected_node,
            )

        def eligible(
            children: tuple[_BankProgramV1, ...],
            depth: int,
            *,
            conjunction: bool = False,
        ) -> bool:
            return _capacity._resource_eligible(
                tuple(row.program for row in children),
                target_depth=depth,
                limits=snapshot.limits,
                conjunction=conjunction,
            )

        def ordered_pairs(path: Path | None) -> Iterator[tuple[_BankProgramV1, _BankProgramV1]]:
            for left_block in _iter_bank_blocks_v1(path):
                for right_block in _iter_bank_blocks_v1(path):
                    for left in left_block:
                        for right in right_block:
                            yield left, right

        def combinations(
            path: Path | None,
            *,
            replacement: bool,
        ) -> Iterator[tuple[_BankProgramV1, _BankProgramV1]]:
            for left_block_index, left_block in enumerate(_iter_bank_blocks_v1(path)):
                for right_block_index, right_block in enumerate(
                    _iter_bank_blocks_v1(path)
                ):
                    if right_block_index < left_block_index:
                        continue
                    for left_index, left in enumerate(left_block):
                        for right_index, right in enumerate(right_block):
                            if right_block_index == left_block_index and (
                                right_index < left_index
                                or (not replacement and right_index == left_index)
                            ):
                                continue
                            yield left, right

        for depth in range(1, snapshot.limits.maximum_ast_depth + 1):
            for operator, path_name in (
                (0, "bit"),
                (1, "bounded"),
                (2, "rational"),
                (3, "rational"),
            ):
                for child in _iter_bank_programs_v1(bank_paths[path_name]):
                    if not eligible((child,), depth):
                        continue
                    source = (
                        _capacity.UNARY_NAMES[operator],
                        _capacity._canonical_node_to_source(child.program.node),
                    )
                    emit_candidate(
                        depth,
                        _capacity._Candidate(
                            0x1000 + operator,
                            source,
                            (1, operator, child.program.node),
                            (child.program,),
                        ),
                    )

            for left, right in ordered_pairs(bank_paths["rational"]):
                if not eligible((left, right), depth):
                    continue
                left_source = _capacity._canonical_node_to_source(left.program.node)
                right_source = _capacity._canonical_node_to_source(right.program.node)
                for operator in (1, 3):
                    emit_candidate(
                        depth,
                        _capacity._Candidate(
                            0x2000 + operator,
                            (
                                _capacity.BINARY_NAMES[operator],
                                left_source,
                                right_source,
                            ),
                            (2, operator, left.program.node, right.program.node),
                            (left.program, right.program),
                        ),
                    )

            for left, right in combinations(
                bank_paths["rational_comm"],
                replacement=True,
            ):
                if not eligible((left, right), depth):
                    continue
                left_source = _capacity._canonical_node_to_source(left.program.node)
                right_source = _capacity._canonical_node_to_source(right.program.node)
                emit_candidate(
                    depth,
                    _capacity._Candidate(
                        0x2002,
                        ("equal_exact", left_source, right_source),
                        (2, 2, left.program.node, right.program.node),
                        (left.program, right.program),
                    ),
                )
                for tolerance in (1, 2):
                    emit_candidate(
                        depth,
                        _capacity._Candidate(
                            0x3000 + tolerance,
                            (
                                "approx_equal",
                                left_source,
                                right_source,
                                tolerance,
                            ),
                            (
                                3,
                                0,
                                left.program.node,
                                right.program.node,
                                tolerance,
                            ),
                            (left.program, right.program),
                        ),
                    )

            for left, right in combinations(
                bank_paths["sign_comm"],
                replacement=True,
            ):
                if not eligible((left, right), depth):
                    continue
                for operator in (5, 6):
                    emit_candidate(
                        depth,
                        _capacity._Candidate(
                            0x2000 + operator,
                            (
                                _capacity.BINARY_NAMES[operator],
                                _capacity._canonical_node_to_source(left.program.node),
                                _capacity._canonical_node_to_source(right.program.node),
                            ),
                            (2, operator, left.program.node, right.program.node),
                            (left.program, right.program),
                        ),
                    )

            for left, right in combinations(
                bank_paths["bool_atom"],
                replacement=False,
            ):
                if not eligible((left, right), depth, conjunction=True):
                    continue
                emit_candidate(
                    depth,
                    _capacity._Candidate(
                        0x4002,
                        (
                            "top_level_AND",
                            _capacity._canonical_node_to_source(left.program.node),
                            _capacity._canonical_node_to_source(right.program.node),
                        ),
                        (4, (left.program.node, right.program.node)),
                        (left.program, right.program),
                    ),
                )

        application_roots, application_counts = _consume_sorted_rows(
            applications.finish(),
            tracker,
        )
        strict_roots, strict_counts = _consume_sorted_rows(
            strict.finish(),
            tracker,
        )
        unique_ast_counts = _consume_unique_ast_rows(
            unique_asts.finish(),
            tracker,
        )

    rows: list[_formal.Q1SemanticCoverageRecordV1] = []
    empty_root = sha256(b"").digest()
    for depth, coverage_code in registry:
        group = bytes((depth,)) + coverage_code.to_bytes(2, "big")
        count, rewrite_count = counters[group]
        if (
            application_counts.get(group, 0) != count
            or strict_counts.get(group, 0) != count
        ):
            _fail("FAIL_FULL_NODE6_COVERAGE_SORT", "sorted coverage count differs")
        application_root = (
            application_roots[group].root()
            if group in application_roots
            else empty_root
        )
        strict_root = strict_roots[group].root() if group in strict_roots else empty_root
        rows.append(
            _formal.Q1SemanticCoverageRecordV1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                depth,
                coverage_code,
                count,
                application_root,
                count,
                application_root,
                count,
                strict_root,
                unique_ast_counts.get(group, 0),
                rewrite_count,
            )
        )
    output = tuple(rows)
    if (
        sum(row.eligible_application_count for row in output)
        != snapshot.raw_operator_application_count
        or sum(row.strict_admitted_count for row in output)
        != snapshot.strict_admitted_application_count
        or sum(row.rewrite_collapse_count for row in output)
        != snapshot.rewrite_collapse_count
    ):
        _fail(
            "FAIL_FULL_NODE6_COVERAGE_REPLAY",
            "coverage totals differ from same-run snapshot",
        )
    barriers = {
        row.depth: row
        for row in snapshot.depth_barriers
        if row.depth <= snapshot.limits.maximum_ast_depth
    }
    for depth in range(snapshot.limits.maximum_ast_depth + 1):
        depth_rows = tuple(row for row in output if row.construction_depth == depth)
        barrier = barriers.get(depth)
        if barrier is None or (
            sum(row.eligible_application_count for row in depth_rows)
            != barrier.eligible_raw_application_count
            or sum(row.strict_admitted_count for row in depth_rows)
            != barrier.strict_admitted_application_count
            or sum(row.rewrite_collapse_count for row in depth_rows)
            != barrier.rewrite_collapse_count
        ):
            _fail("FAIL_FULL_NODE6_COVERAGE_REPLAY", "coverage depth totals differ")
    return output


def _program_id_from_same_run_metadata_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    canonical_ast_cbor: bytes,
    canonical_ast_hash: bytes,
    signature_object: tuple[object, ...],
) -> bytes:
    return content_hash(
        _formal.PROGRAM_ID_DOMAIN,
        (
            snapshot.input_signature_id,
            snapshot.universe_root,
            canonical_ast_cbor,
            canonical_ast_hash,
            signature_object,
        ),
    )


def _bounded_project_same_run_partition_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    scratch: Path,
    tracker: _ScratchTrackerV1,
    retained_chunk_consumer: _RetainedChunkConsumerV1 | None = None,
) -> FullNode6PartitionProjectionV1:
    """Project a snapshot without materializing any node6-sized record stream."""

    group = b"\x00\x00\x00"
    program_meta = _SpillSorter(scratch, "formal-program-meta", tracker)
    cohort_meta = _SpillSorter(scratch, "formal-cohort-meta", tracker)
    class_base = _SpillSorter(scratch, "formal-class-base", tracker)
    observed_programs = 0
    observed_cohorts = 0
    observed_classes = 0

    for class_row in snapshot.behavior_classes:
        behavior = _archive._formal_behavior_from_snapshot(snapshot, class_row)
        class_id = behavior.behavior_id
        behavior_bytes = behavior.canonical_bytes
        diagnostic_behavior = _diagnostic_behavior_key_v1(snapshot, class_row)
        class_base.add(
            group,
            class_id + behavior_bytes,
            canonical_cbor_encode(
                (
                    behavior.canonical_object(),
                    class_row.minimum_admitted_mdl_q32,
                    diagnostic_behavior,
                    len(class_row.cohorts),
                )
            ),
        )
        observed_classes += 1
        for cohort in class_row.cohorts:
            signature_object = _formal.construction_signature_object_v1(
                cohort.signature
            )
            signature_id = _formal.construction_signature_id_v1(cohort.signature)
            signature_bytes = canonical_cbor_encode(signature_object)
            diagnostic_signature = _diagnostic_signature_bytes_v1(cohort.signature)
            representatives = tuple(
                (
                    representative.canonical_ast_cbor,
                    representative.canonical_ast_hash,
                )
                for representative in cohort.representatives
            )
            cohort_meta.add(
                group,
                class_id + signature_id + signature_bytes,
                canonical_cbor_encode(
                    (
                        class_id,
                        signature_object,
                        representatives,
                        cohort.visible_frontier_member,
                        diagnostic_behavior,
                        diagnostic_signature,
                    )
                ),
            )
            observed_cohorts += 1
            for canonical_ast_cbor, canonical_ast_hash in representatives:
                ast = decode_shrink6_canonical_ast(canonical_ast_cbor)
                formal_key = (
                    bytes((ast.metrics.depth,))
                    + ast.metrics.node_count.to_bytes(2, "big")
                    + bytes((int(cohort.signature.output_sort_id),))
                    + ast.root_operator_id.to_bytes(2, "big")
                    + canonical_ast_cbor
                )
                program_meta.add(
                    group,
                    formal_key,
                    canonical_cbor_encode(
                        (
                            class_id,
                            canonical_ast_cbor,
                            canonical_ast_hash,
                            signature_object,
                            diagnostic_behavior,
                            diagnostic_signature,
                        )
                    ),
                )
                observed_programs += 1

    if (
        observed_programs != snapshot.continuation_bank_point_count
        or observed_cohorts != snapshot.signature_cohort_count
        or observed_classes != snapshot.behavior_class_count
    ):
        _fail(
            "FAIL_FULL_NODE6_RECORD_STREAMS",
            "same-run snapshot cardinality differs before projection",
        )

    program_meta_path = program_meta.finish()
    cohort_meta_path = cohort_meta.finish()
    class_base_path = class_base.finish()
    root_builder = _SameRunRecordSetRootBuilderV1(snapshot)
    streams: list[_archive.Q1CountingDiscardStreamV1] = []

    custom_program = _SpillSorter(scratch, "custom-program", tracker)
    bank_sorters = {
        name: _SpillSorter(scratch, f"bank-{name}", tracker)
        for name in (
            "bit",
            "bounded",
            "rational",
            "rational_comm",
            "sign_comm",
            "bool_atom",
        )
    }
    program_sink = _FormalCountingSinkV1(
        scratch,
        "program",
        snapshot=snapshot,
        tracker=tracker,
        stream_kind_id=_formal.ArchiveStreamKindId.PROGRAM,
        expected_count=snapshot.continuation_bank_point_count,
        record_set_root=root_builder,
        retained_chunk_consumer=retained_chunk_consumer,
    )
    previous_program_key: bytes | None = None
    program_index = 0
    try:
        for row_group, key, payload in _iter_run_rows_v1(program_meta_path):
            if row_group != group or (
                previous_program_key is not None and key <= previous_program_key
            ):
                _fail("FAIL_FULL_NODE6_PROGRAM_META", "program key repeats or decreases")
            previous_program_key = key
            value = canonical_cbor_decode(payload)
            if type(value) is not tuple or len(value) != 6:
                _fail("FAIL_FULL_NODE6_PROGRAM_META", "program metadata differs")
            (
                class_id,
                canonical_ast_cbor,
                canonical_ast_hash,
                signature_object,
                diagnostic_behavior,
                diagnostic_signature,
            ) = value
            if not (
                type(class_id) is bytes
                and len(class_id) == 32
                and type(canonical_ast_cbor) is bytes
                and type(canonical_ast_hash) is bytes
                and len(canonical_ast_hash) == 32
                and type(signature_object) is tuple
                and type(diagnostic_behavior) is bytes
                and type(diagnostic_signature) is bytes
            ):
                _fail("FAIL_FULL_NODE6_PROGRAM_META", "program metadata wire differs")
            signature = _archive._decode_signature_v1(signature_object)
            record = _formal.Q1RepresentativeProgramRecordV1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                program_index,
                class_id,
                canonical_ast_cbor,
                canonical_ast_hash,
                signature,
            )
            if _archive._stream_sort_key(
                record,
                _formal.ArchiveStreamKindId.PROGRAM,
            ) != key:
                _fail("FAIL_FULL_NODE6_PROGRAM_META", "program sort replay differs")
            program_sink.append(record)

            custom_record = bytearray(b"PROGRAM\x00")
            custom_record.extend(sha256(diagnostic_behavior).digest())
            custom_record.extend(sha256(diagnostic_signature).digest())
            _append_blob(custom_record, canonical_ast_cbor)
            custom_record.extend(canonical_ast_hash)
            custom_program.add(
                group,
                _lex_tuple_key_v1(
                    (
                        diagnostic_behavior,
                        diagnostic_signature,
                        canonical_ast_cbor,
                    )
                ),
                bytes(custom_record),
            )

            bank_payload = canonical_cbor_encode((record.program_id, canonical_ast_cbor))
            ast = decode_shrink6_canonical_ast(canonical_ast_cbor)
            if ast.metrics.output_sort == "Bit":
                bank_sorters["bit"].add(group, key, bank_payload)
            elif ast.metrics.output_sort == "BoundedInt":
                bank_sorters["bounded"].add(group, key, bank_payload)
            elif ast.metrics.output_sort == "RationalValue":
                bank_sorters["rational"].add(group, key, bank_payload)
                node_bytes = canonical_cbor_encode(ast.value[1])
                bank_sorters["rational_comm"].add(
                    group,
                    sha256(node_bytes).digest() + node_bytes,
                    bank_payload,
                )
            elif ast.metrics.output_sort == "Sign":
                node_bytes = canonical_cbor_encode(ast.value[1])
                bank_sorters["sign_comm"].add(
                    group,
                    sha256(node_bytes).digest() + node_bytes,
                    bank_payload,
                )
            elif ast.metrics.output_sort == "Bool" and ast.value[1][0] != 4:
                node_bytes = canonical_cbor_encode(ast.value[1])
                bank_sorters["bool_atom"].add(group, node_bytes, bank_payload)
            program_index += 1
    finally:
        _unlink_run_v1(program_meta_path, tracker)
    streams.append(program_sink.finish())
    custom_program_path = custom_program.finish()
    bank_paths = {name: sorter.finish() for name, sorter in bank_sorters.items()}

    custom_cohort = _SpillSorter(scratch, "custom-cohort", tracker)
    class_aggregate = _SpillSorter(
        scratch,
        "formal-class-aggregate",
        tracker,
    )
    cohort_sink = _FormalCountingSinkV1(
        scratch,
        "cohort",
        snapshot=snapshot,
        tracker=tracker,
        stream_kind_id=_formal.ArchiveStreamKindId.COHORT,
        expected_count=snapshot.signature_cohort_count,
        record_set_root=root_builder,
        retained_chunk_consumer=retained_chunk_consumer,
    )
    previous_cohort_key: bytes | None = None
    cohort_index = 0
    active_class_id: bytes | None = None
    first_cohort_index = 0
    class_cohort_count = 0
    class_bank_points = 0
    class_visible_cohorts = 0
    class_visible_points = 0
    class_cohort_root = _Rfc6962Frontier()
    class_visible_root = _Rfc6962Frontier()

    def flush_class_aggregate() -> None:
        nonlocal class_cohort_count, class_bank_points
        nonlocal class_visible_cohorts, class_visible_points
        nonlocal class_cohort_root, class_visible_root
        if active_class_id is None:
            return
        class_aggregate.add(
            group,
            active_class_id,
            canonical_cbor_encode(
                (
                    active_class_id,
                    first_cohort_index,
                    class_cohort_count,
                    class_cohort_root.root(),
                    class_bank_points,
                    class_visible_cohorts,
                    class_visible_points,
                    class_visible_root.root(),
                )
            ),
        )
        class_cohort_count = 0
        class_bank_points = 0
        class_visible_cohorts = 0
        class_visible_points = 0
        class_cohort_root = _Rfc6962Frontier()
        class_visible_root = _Rfc6962Frontier()

    try:
        for row_group, key, payload in _iter_run_rows_v1(cohort_meta_path):
            if row_group != group or (
                previous_cohort_key is not None and key <= previous_cohort_key
            ):
                _fail("FAIL_FULL_NODE6_COHORT_META", "cohort key repeats or decreases")
            previous_cohort_key = key
            value = canonical_cbor_decode(payload)
            if type(value) is not tuple or len(value) != 6:
                _fail("FAIL_FULL_NODE6_COHORT_META", "cohort metadata differs")
            (
                class_id,
                signature_object,
                representatives,
                visible,
                diagnostic_behavior,
                diagnostic_signature,
            ) = value
            if not (
                type(class_id) is bytes
                and len(class_id) == 32
                and type(signature_object) is tuple
                and type(representatives) is tuple
                and type(visible) is bool
                and type(diagnostic_behavior) is bytes
                and type(diagnostic_signature) is bytes
            ):
                _fail("FAIL_FULL_NODE6_COHORT_META", "cohort metadata wire differs")
            if active_class_id != class_id:
                flush_class_aggregate()
                active_class_id = class_id
                first_cohort_index = cohort_index
            signature = _archive._decode_signature_v1(signature_object)
            witnesses: list[_formal.Q1CohortWitnessV1] = []
            ast_hashes: list[bytes] = []
            for rank, representative in enumerate(representatives):
                if (
                    type(representative) is not tuple
                    or len(representative) != 2
                    or type(representative[0]) is not bytes
                    or type(representative[1]) is not bytes
                    or len(representative[1]) != 32
                ):
                    _fail("FAIL_FULL_NODE6_COHORT_META", "cohort representative differs")
                program_id = _program_id_from_same_run_metadata_v1(
                    snapshot,
                    representative[0],
                    representative[1],
                    signature_object,
                )
                witnesses.append(
                    _formal.Q1CohortWitnessV1(rank, program_id, representative[1])
                )
                ast_hashes.append(representative[1])
            record = _formal.Q1ContinuationCohortRecordV1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                cohort_index,
                class_id,
                signature,
                tuple(witnesses),
                visible,
            )
            if _archive._stream_sort_key(
                record,
                _formal.ArchiveStreamKindId.COHORT,
            ) != key:
                _fail("FAIL_FULL_NODE6_COHORT_META", "cohort sort replay differs")
            cohort_sink.append(record)
            encoded_record = canonical_cbor_encode(record.canonical_object())
            class_cohort_root.append_canonical_record(encoded_record)
            class_cohort_count += 1
            class_bank_points += len(witnesses)
            if visible:
                class_visible_root.append_canonical_record(encoded_record)
                class_visible_cohorts += 1
                class_visible_points += len(witnesses)

            custom_record = bytearray(b"COHORT\x00")
            custom_record.extend(sha256(diagnostic_behavior).digest())
            _append_blob(custom_record, diagnostic_signature)
            custom_record.append(int(visible))
            custom_record.extend(_u64(len(ast_hashes)))
            for ast_hash in ast_hashes:
                custom_record.extend(ast_hash)
            custom_cohort.add(
                group,
                _lex_tuple_key_v1((diagnostic_behavior, diagnostic_signature)),
                bytes(custom_record),
            )
            cohort_index += 1
        flush_class_aggregate()
    finally:
        _unlink_run_v1(cohort_meta_path, tracker)
    streams.append(cohort_sink.finish())
    custom_cohort_path = custom_cohort.finish()
    class_aggregate_path = class_aggregate.finish()

    custom_class = _SpillSorter(scratch, "custom-class", tracker)
    class_sink = _FormalCountingSinkV1(
        scratch,
        "class",
        snapshot=snapshot,
        tracker=tracker,
        stream_kind_id=_formal.ArchiveStreamKindId.CLASS,
        expected_count=snapshot.behavior_class_count,
        record_set_root=root_builder,
        retained_chunk_consumer=retained_chunk_consumer,
    )
    base_rows = _iter_run_rows_v1(class_base_path)
    aggregate_rows = _iter_run_rows_v1(class_aggregate_path)
    class_index = 0
    try:
        while True:
            try:
                base_group, base_key, base_payload = next(base_rows)
            except StopIteration:
                base_row = None
            else:
                base_row = (base_group, base_key, base_payload)
            try:
                aggregate_group, aggregate_key, aggregate_payload = next(
                    aggregate_rows
                )
            except StopIteration:
                aggregate_row = None
            else:
                aggregate_row = (
                    aggregate_group,
                    aggregate_key,
                    aggregate_payload,
                )
            if base_row is None or aggregate_row is None:
                if base_row is not None or aggregate_row is not None:
                    _fail("FAIL_FULL_NODE6_CLASS_META", "class joins differ")
                break
            base_value = canonical_cbor_decode(base_row[2])
            aggregate_value = canonical_cbor_decode(aggregate_row[2])
            if (
                base_row[0] != group
                or aggregate_row[0] != group
                or type(base_value) is not tuple
                or len(base_value) != 4
                or type(aggregate_value) is not tuple
                or len(aggregate_value) != 8
                or base_row[1][:32] != aggregate_row[1]
                or aggregate_value[0] != aggregate_row[1]
            ):
                _fail("FAIL_FULL_NODE6_CLASS_META", "class join wire differs")
            behavior_object, minimum_mdl, diagnostic_behavior, cohort_count = base_value
            behavior = _archive._decode_behavior_blob_v1(behavior_object)
            (
                class_id,
                first_index,
                aggregate_count,
                cohort_root,
                bank_points,
                visible_cohorts,
                visible_points,
                visible_root,
            ) = aggregate_value
            if aggregate_count != cohort_count:
                _fail("FAIL_FULL_NODE6_CLASS_META", "class cohort count differs")
            record = _formal.Q1QuotientClassRecordV1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                class_index,
                behavior,
                first_index,
                aggregate_count,
                cohort_root,
                bank_points,
                visible_cohorts,
                visible_points,
                visible_root,
                minimum_mdl,
            )
            if (
                record.class_id != class_id
                or _archive._stream_sort_key(
                    record,
                    _formal.ArchiveStreamKindId.CLASS,
                )
                != base_row[1]
            ):
                _fail("FAIL_FULL_NODE6_CLASS_META", "class sort replay differs")
            class_sink.append(record)

            custom_record = bytearray(b"CLASS\x00")
            _append_blob(custom_record, diagnostic_behavior)
            custom_record.extend(_u64(cohort_count))
            custom_class.add(
                group,
                _lex_tuple_key_v1((diagnostic_behavior,)),
                bytes(custom_record),
            )
            class_index += 1
    finally:
        base_rows.close()
        aggregate_rows.close()
        _unlink_run_v1(class_base_path, tracker)
        _unlink_run_v1(class_aggregate_path, tracker)
    streams.append(class_sink.finish())
    custom_class_path = custom_class.finish()

    try:
        coverage = _coverage_records_from_bank_paths_v1(
            snapshot,
            bank_paths,
            scratch_parent=scratch,
            tracker=tracker,
        )
    finally:
        for path in bank_paths.values():
            _unlink_run_v1(path, tracker)
    coverage_sink = _FormalCountingSinkV1(
        scratch,
        "coverage",
        snapshot=snapshot,
        tracker=tracker,
        stream_kind_id=_formal.ArchiveStreamKindId.COVERAGE,
        expected_count=_formal.EXPECTED_COVERAGE_RECORD_COUNT,
        record_set_root=root_builder,
        retained_chunk_consumer=retained_chunk_consumer,
    )
    for row in coverage:
        coverage_sink.append(row)
    streams.append(coverage_sink.finish())

    capacity_projection = _capacity_state_projection_from_paths_v1(
        (custom_program_path, custom_cohort_path, custom_class_path),
        coverage,
        tracker=tracker,
    )
    coverage_diagnostic = CoverageDiagnosticV1(
        coverage_record_count=len(coverage),
        eligible_application_count=sum(
            row.eligible_application_count for row in coverage
        ),
        processed_application_count=sum(
            row.processed_application_count for row in coverage
        ),
        strict_admitted_application_count=sum(
            row.strict_admitted_count for row in coverage
        ),
        rewrite_collapse_count=sum(row.rewrite_collapse_count for row in coverage),
        zero_eligible_record_count=sum(
            int(row.eligible_application_count == 0) for row in coverage
        ),
        all_eligible_covered=snapshot.terminal_status
        in (
            _capacity.PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY,
            _capacity.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED,
        ),
    )
    stream_tuple = tuple(streams)
    (
        fixed_point_record,
        saturation_state_root,
        signature_archive_manifest,
        projection_partition_row,
    ) = _assemble_formal_partition_v1(
        snapshot,
        stream_tuple,
        coverage_diagnostic,
    )
    tracker.require_terminal_empty()
    return FullNode6PartitionProjectionV1(
        input_signature_id=snapshot.input_signature_id,
        universe_root=snapshot.universe_root,
        same_run_record_set_root=root_builder.finish(),
        streams=stream_tuple,
        capacity_state_projection=capacity_projection,
        coverage=coverage_diagnostic,
        fixed_point_record=fixed_point_record,
        saturation_state_root=saturation_state_root,
        signature_archive_manifest=signature_archive_manifest,
        projection_partition_row=projection_partition_row,
    )


def _project_same_run_partition_with_tracker_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    *,
    scratch_root: Path,
    tracker: _ScratchTrackerV1,
    retained_chunk_consumer: _RetainedChunkConsumerV1 | None = None,
) -> FullNode6PartitionProjectionV1:
    if type(snapshot) is not _snapshot.Q1PartitionSnapshotV1:
        raise TypeError("snapshot must be exact Q1PartitionSnapshotV1")
    if (
        snapshot.q1_state != "NOT_RUN"
        or snapshot.q1_gate_count != 0
        or snapshot.q1_gate_mask != 0
        or snapshot.target_truth_accessed
        or snapshot.split_accessed
        or snapshot.role_evaluation_performed
    ):
        _fail("REJECT_FULL_NODE6_SAME_RUN", "snapshot authority boundary differs")
    with tempfile.TemporaryDirectory(
        prefix=f"hegel-q1-full-projection-{snapshot.input_signature_id}-",
        dir=scratch_root,
    ) as directory:
        try:
            if retained_chunk_consumer is None:
                return _bounded_project_same_run_partition_v1(
                    snapshot,
                    Path(directory),
                    tracker,
                )
            return _bounded_project_same_run_partition_v1(
                snapshot,
                Path(directory),
                tracker,
                retained_chunk_consumer,
            )
        except BaseException:
            # Capture the last atomic all-live state before TemporaryDirectory
            # removes the failed projection tree.
            tracker.reconcile()
            raise


def project_same_run_partition_v1(
    snapshot: _snapshot.Q1PartitionSnapshotV1,
    *,
    scratch_root: Path = SCRATCH_ROOT,
) -> FullNode6PartitionProjectionV1:
    """Bounded projection path; node-3 snapshots serve only as library KATs."""

    tracker = _ScratchTrackerV1()
    try:
        return _project_same_run_partition_with_tracker_v1(
            snapshot,
            scratch_root=scratch_root,
            tracker=tracker,
        )
    finally:
        tracker.release_after_transient_tree_cleanup()


def _assemble_formal_aggregate_v1(
    partitions: tuple[FullNode6PartitionProjectionV1, ...],
) -> tuple[_formal.Q1ClosureBundleV1, _formal.Q1ArchiveProjectionResultV1]:
    if tuple(partition.input_signature_id for partition in partitions) != (1, 2):
        _fail(
            "REJECT_FULL_NODE6_FORMAL_ASSEMBLY",
            "formal aggregate requires ordered Odd and Sink partitions",
        )
    closure = _formal.Q1ClosureBundleV1(
        semantic_binding_root=FROZEN_SEMANTIC_BINDING_ROOT,
        projection_profile_root=FROZEN_PROJECTION_PROFILE_ROOT,
        signature_rows=tuple(
            (
                partition.input_signature_id,
                partition.universe_root,
                partition.signature_archive_manifest.manifest_root,
                partition.saturation_state_root,
            )
            for partition in partitions
        ),
    )
    partition_rows = tuple(
        partition.projection_partition_row for partition in partitions
    )
    closure_frame_bytes = _framed_object_bytes_v1(closure.canonical_object())
    archive_payload_bytes = (
        sum(row.projected_partition_payload_bytes for row in partition_rows)
        + closure_frame_bytes
    )
    projection_result = _formal.Q1ArchiveProjectionResultV1(
        projection_profile_root=FROZEN_PROJECTION_PROFILE_ROOT,
        semantic_binding_root=FROZEN_SEMANTIC_BINDING_ROOT,
        partition_rows=partition_rows,
        projected_closure_bundle_frame_bytes=closure_frame_bytes,
        projected_archive_payload_bytes_per_endpoint=archive_payload_bytes,
        projected_endpoint_total_output_bytes=(
            archive_payload_bytes + _formal.ENDPOINT_RUN_METADATA_RESERVATION_BYTES
        ),
        projected_endpoint_peak_scratch_bytes=max(
            row.projected_peak_scratch_bytes for row in partition_rows
        ),
        projected_host_replay_output_bytes=(
            _formal.HOST_RUN_METADATA_RESERVATION_BYTES
        ),
        projected_host_replay_peak_scratch_bytes=0,
    )
    return closure, projection_result


def project_full_node6_same_run_v1(
    diagnostic: _full.FullNode6CapacityDiagnosticV1,
    snapshots: tuple[_snapshot.Q1PartitionSnapshotV1 | None, ...],
    *,
    scratch_root: Path = SCRATCH_ROOT,
    retained_chunk_consumer: _RetainedChunkConsumerV1 | None = None,
) -> FullNode6ProjectionResultV1:
    """Project one admitted full result without a second quotient measurement."""

    if type(diagnostic) is not _full.FullNode6CapacityDiagnosticV1:
        _fail("REJECT_FULL_NODE6_SAME_RUN", "diagnostic exact type differs")
    if (
        diagnostic.terminal_status != _full.TERMINAL_SATURATED
        or type(snapshots) is not tuple
        or len(snapshots) != 2
        or any(value is None for value in snapshots)
    ):
        _fail("REJECT_FULL_NODE6_SAME_RUN", "only two saturated same-run snapshots project")
    exact_snapshots = tuple(value for value in snapshots if value is not None)
    for row, snapshot in zip(diagnostic.partitions, exact_snapshots, strict=True):
        _require_same_run_snapshot_v1(row, snapshot)
    tracker = _ScratchTrackerV1()
    try:
        if retained_chunk_consumer is None:
            partitions = tuple(
                _project_same_run_partition_with_tracker_v1(
                    value,
                    scratch_root=scratch_root,
                    tracker=tracker,
                )
                for value in exact_snapshots
            )
        else:
            partitions = tuple(
                _project_same_run_partition_with_tracker_v1(
                    value,
                    scratch_root=scratch_root,
                    tracker=tracker,
                    retained_chunk_consumer=retained_chunk_consumer,
                )
                for value in exact_snapshots
            )
    finally:
        tracker.release_after_transient_tree_cleanup()
    closure_bundle, archive_projection_result = _assemble_formal_aggregate_v1(
        partitions
    )
    return FullNode6ProjectionResultV1(
        capsule_digest=diagnostic.capsule_digest,
        source_commit=diagnostic.source_commit,
        partitions=partitions,
        closure_bundle=closure_bundle,
        archive_projection_result=archive_projection_result,
        actual_preflight_peak_scratch_bytes=tracker.peak_logical_bytes,
        actual_preflight_scratch_live_bytes_at_terminal=0,
    )


__all__ = [
    "FullNode6PartitionProjectionV1",
    "FullNode6ProjectionError",
    "FullNode6ProjectionResultV1",
    "CapacityCountingDiscardStreamV1",
    "CapacityStateProjectionV1",
    "CoverageDiagnosticV1",
    "FROZEN_PROJECTION_PROFILE_ROOT",
    "FROZEN_PROJECTION_PROFILE_ROOT_ID",
    "FROZEN_SEMANTIC_BINDING_ROOT",
    "PROJECTION_ID",
    "SCHEMA_VERSION",
    "SIDECAR_PATHS",
    "STREAM_REGISTRY",
    "canonical_json_bytes_v1",
    "formal_wire_projection_json_v1",
    "project_full_node6_same_run_v1",
    "project_same_run_partition_v1",
]

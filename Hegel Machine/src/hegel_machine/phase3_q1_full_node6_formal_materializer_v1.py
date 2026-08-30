"""Retain the frozen Q1 formal archive produced from one same-run traversal.

The admitted runner measures Odd then Sink exactly once with the existing
full-node6 partition engine and immediately passes the returned immutable
same-run snapshots to the materializer in the same process.  The lower-level
materializer never starts a measurement itself.  Neither route reads
target/split/role material, issues a Q1 completion receipt, or publishes a
current pointer.  The retained bytes reuse the exact projection traversal,
sort schedule, canonical CBOR, record order, and deterministic chunk bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable, Final, NoReturn

from . import phase3_q1_formal_archive_contract_v1 as _formal
from . import phase3_q1_full_node6_capacity_preflight_v1 as _full
from . import phase3_q1_full_node6_projection_v1 as _projection
from . import phase3_q1_partition_snapshot_v1 as _snapshot
from . import phase3_q1_prestart_wire_v1 as _prestart


SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-formal-archive-materializer/1"
ACTOR_ID: Final = "python"
IMPLEMENTATION_ID: Final = "hegel-python-q1-formal-archive-materializer-v1"
ACTION_ID: Final = "materialize-full-node6-formal-archive-v1"
STATUS: Final = "FULL_NODE6_FORMAL_ARCHIVE_MATERIALIZED_Q1_NOT_RUN"
ADMISSION_SCHEMA_VERSION: Final = "hegel-q1-formal-archive-materializer-admission/1"
Q1_STATE: Final = "NOT_RUN"
Q1_GATE_COUNT: Final = 14
Q1_GATE_TOTAL: Final = 20
Q1_GATE_MASK: Final = 0x3FFF

_FORMAL_CAPACITY_LIMIT_ROW_FIELDS: Final = (
    "input_signature_id",
    "maximum_ast_depth",
    "maximum_ast_node_count",
    "maximum_raw_operator_applications",
    "maximum_behavior_classes",
    "maximum_visible_frontier_points",
    "maximum_visible_frontier_points_per_class",
    "maximum_continuation_bank_points",
    "maximum_continuation_bank_points_per_class",
    "maximum_work_queue_points",
    "maximum_saturation_rounds",
    "maximum_wall_time_seconds",
)


def _formal_resource_limits_v1() -> tuple[
    tuple[_full._capacity.PreflightLimitsV1, _full._capacity.PreflightLimitsV1],
    int,
    int,
]:
    """Project the immutable Gate 11 wire into exact Python actor limits."""

    envelope = _prestart.FORMAL_RESOURCE_ENVELOPE
    signature_resources = {
        signature_id: dict(resources)
        for signature_id, resources in envelope[2]
    }
    actor_resources = {
        actor_id: dict(resources)
        for _actor_number, actor_id, resources in envelope[3]
    }
    python_resources = actor_resources[b"python"]
    wall_time_seconds = python_resources[12]
    limits = tuple(
        _full._capacity.PreflightLimitsV1(
            maximum_ast_depth=3,
            maximum_ast_node_count=6,
            maximum_raw_operator_applications=signature_resources[signature_id][1],
            maximum_behavior_classes=signature_resources[signature_id][2],
            maximum_visible_frontier_points=signature_resources[signature_id][3],
            maximum_visible_frontier_points_per_class=(
                signature_resources[signature_id][4]
            ),
            maximum_continuation_bank_points=signature_resources[signature_id][5],
            maximum_continuation_bank_points_per_class=(
                signature_resources[signature_id][6]
            ),
            maximum_work_queue_points=signature_resources[signature_id][7],
            maximum_saturation_rounds=signature_resources[signature_id][8],
            maximum_wall_time_seconds=wall_time_seconds,
        )
        for signature_id in (1, 2)
    )
    return limits, python_resources[9], python_resources[10]


(
    _FORMAL_LIMITS_BY_SIGNATURE,
    FORMAL_OUTPUT_BYTES_LIMIT,
    FORMAL_SCRATCH_BYTES_LIMIT,
) = _formal_resource_limits_v1()

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class Q1FormalArchiveMaterializationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1FormalArchiveMaterializationError(code, detail)


def _sha256_id(value: bytes) -> str:
    if type(value) is not bytes or len(value) != 32:
        _fail("FAIL_Q1_FORMAL_MATERIALIZER_ROOT", "root must be exactly 32 bytes")
    return "sha256:" + value.hex()


def _digest_id(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail("REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION", f"{name} differs")
    return value


def _formal_capacity_limit_row_v1(
    input_signature_id: int,
    limits: _full._capacity.PreflightLimitsV1,
) -> dict[str, int]:
    row = {
        "input_signature_id": input_signature_id,
        "maximum_ast_depth": limits.maximum_ast_depth,
        "maximum_ast_node_count": limits.maximum_ast_node_count,
        "maximum_raw_operator_applications": (
            limits.maximum_raw_operator_applications
        ),
        "maximum_behavior_classes": limits.maximum_behavior_classes,
        "maximum_visible_frontier_points": (
            limits.maximum_visible_frontier_points
        ),
        "maximum_visible_frontier_points_per_class": (
            limits.maximum_visible_frontier_points_per_class
        ),
        "maximum_continuation_bank_points": (
            limits.maximum_continuation_bank_points
        ),
        "maximum_continuation_bank_points_per_class": (
            limits.maximum_continuation_bank_points_per_class
        ),
        "maximum_work_queue_points": limits.maximum_work_queue_points,
        "maximum_saturation_rounds": limits.maximum_saturation_rounds,
        "maximum_wall_time_seconds": limits.maximum_wall_time_seconds,
    }
    if tuple(row) != _FORMAL_CAPACITY_LIMIT_ROW_FIELDS:
        raise AssertionError("formal capacity limit row registry differs")
    return row


def _require_formal_scratch_limit_v1(actual_peak_scratch_bytes: int) -> None:
    if actual_peak_scratch_bytes > FORMAL_SCRATCH_BYTES_LIMIT:
        _fail(
            "INCONCLUSIVE_Q1_FORMAL_RESOURCE_GUARD",
            "formal projection scratch peak exceeds the frozen Python envelope",
        )


def _require_formal_output_limit_v1(
    archive_payload_bytes: int,
    endpoint_result_bytes: int,
) -> None:
    if archive_payload_bytes + endpoint_result_bytes > FORMAL_OUTPUT_BYTES_LIMIT:
        _fail(
            "INCONCLUSIVE_Q1_FORMAL_RESOURCE_GUARD",
            "formal archive plus endpoint result exceeds the frozen Python envelope",
        )


@dataclass(frozen=True, slots=True)
class Q1FormalArchiveMaterializerAdmissionV1:
    schema_version: str
    action_id: str
    capsule_digest: str
    source_commit: str
    run_genesis_root: str
    prestart_admission_receipt_root: str
    projection_profile_root: str
    q1_state: str
    q1_gate_count: int
    q1_gate_total: int
    q1_gate_mask: int
    q1_output_slots: tuple[None, ...]
    diagnostic_only: bool
    target_truth_accessed: bool
    split_accessed: bool
    role_evaluation_performed: bool

    def __post_init__(self) -> None:
        if (
            self.schema_version != ADMISSION_SCHEMA_VERSION
            or self.action_id != ACTION_ID
            or type(self.source_commit) is not str
            or _COMMIT.fullmatch(self.source_commit) is None
            or self.q1_state != Q1_STATE
            or type(self.q1_gate_count) is not int
            or self.q1_gate_count != Q1_GATE_COUNT
            or type(self.q1_gate_total) is not int
            or self.q1_gate_total != Q1_GATE_TOTAL
            or type(self.q1_gate_mask) is not int
            or self.q1_gate_mask != Q1_GATE_MASK
            or type(self.q1_output_slots) is not tuple
            or self.q1_output_slots != (None,) * 8
            or self.diagnostic_only is not False
            or self.target_truth_accessed is not False
            or self.split_accessed is not False
            or self.role_evaluation_performed is not False
        ):
            _fail(
                "REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION",
                "formal 14-of-20 admission boundary differs",
            )
        _digest_id(self.capsule_digest, "capsule_digest")
        _digest_id(self.run_genesis_root, "run_genesis_root")
        _digest_id(
            self.prestart_admission_receipt_root,
            "prestart_admission_receipt_root",
        )
        _digest_id(self.projection_profile_root, "projection_profile_root")
        if self.projection_profile_root != _projection.FROZEN_PROJECTION_PROFILE_ROOT_ID:
            _fail(
                "REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION",
                "projection profile differs from frozen formal profile",
            )


def _write_new(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes:
        raise TypeError("payload must be exact bytes")
    with path.open("xb", buffering=0) as target:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = target.write(view[offset:])
            if written is None or written <= 0:
                _fail(
                    "FAIL_Q1_FORMAL_MATERIALIZER_WRITE",
                    "short archive write",
                )
            offset += written
        target.flush()
        os.fsync(target.fileno())


class _ArchiveDirectoryWriterV1:
    """Write only the frozen archive layout; the outer runner owns publication."""

    def __init__(self, output_root: Path) -> None:
        if not isinstance(output_root, Path):
            raise TypeError("output_root must be a pathlib.Path")
        if not output_root.is_dir() or tuple(output_root.iterdir()) != ():
            _fail(
                "REJECT_Q1_FORMAL_MATERIALIZER_OUTPUT",
                "output root must be an existing empty directory",
            )
        self.output_root = output_root
        self.next_chunk: dict[tuple[int, int], int] = {}
        self.chunk_count = 0
        self.signature_directories: set[Path] = set()
        self.stream_directories: set[Path] = set()

    def retain_chunk(
        self,
        manifest: _formal.Q1ArchiveChunkManifestV1,
        framed_blob: bytes,
    ) -> None:
        if type(manifest) is not _formal.Q1ArchiveChunkManifestV1:
            _fail(
                "FAIL_Q1_FORMAL_MATERIALIZER_CHUNK",
                "chunk manifest exact type differs",
            )
        key = (manifest.input_signature_id, int(manifest.stream_kind_id))
        expected = self.next_chunk.get(key, 0)
        if manifest.chunk_index != expected:
            _fail(
                "FAIL_Q1_FORMAL_MATERIALIZER_CHUNK",
                "chunk callback order differs from frozen stream order",
            )
        signature = self.output_root / f"signature-{manifest.input_signature_id:02d}"
        stream = signature / f"stream-{int(manifest.stream_kind_id):02d}"
        stream.mkdir(parents=True, exist_ok=True)
        self.signature_directories.add(signature)
        self.stream_directories.add(stream)
        _write_new(stream / f"chunk-{manifest.chunk_index:06d}.framed", framed_blob)
        self.next_chunk[key] = expected + 1
        self.chunk_count += 1

    def write_partition_material(
        self,
        partition: _projection.FullNode6PartitionProjectionV1,
    ) -> None:
        signature = self.output_root / f"signature-{partition.input_signature_id:02d}"
        for stream in partition.streams:
            key = (partition.input_signature_id, int(stream.stream_kind_id))
            if self.next_chunk.get(key, 0) != len(stream.chunk_manifests):
                _fail(
                    "FAIL_Q1_FORMAL_MATERIALIZER_CHUNK",
                    "retained chunk count differs from frozen descriptor",
                )
        manifests = b"".join(
            _formal.frame_canonical_record_v1(manifest.canonical_object())
            for stream in partition.streams
            for manifest in stream.chunk_manifests
        )
        fixed_point = _formal.frame_canonical_record_v1(
            partition.fixed_point_record.canonical_object()
        )
        signature_manifest = _formal.frame_canonical_record_v1(
            partition.signature_archive_manifest.canonical_object()
        )
        row = partition.projection_partition_row
        if (
            len(manifests) != row.projected_chunk_manifest_stream_bytes
            or len(fixed_point) != row.projected_fixed_point_frame_bytes
            or len(signature_manifest)
            != row.projected_signature_manifest_frame_bytes
        ):
            _fail(
                "FAIL_Q1_FORMAL_MATERIALIZER_LENGTH",
                "partition terminal frame lengths differ from frozen projection",
            )
        _write_new(signature / "chunk-manifests.framed", manifests)
        _write_new(signature / "fixed-point.framed", fixed_point)
        _write_new(signature / "signature-manifest.framed", signature_manifest)

    def write_closure(self, closure: _formal.Q1ClosureBundleV1) -> int:
        payload = _formal.frame_canonical_record_v1(closure.canonical_object())
        _write_new(self.output_root / "closure-bundle.framed", payload)
        return len(payload)

    def archive_payload_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.output_root.rglob("*.framed")
            if path.is_file()
        )

    def fsync_directories(self) -> None:
        directories = (
            tuple(sorted(self.stream_directories, key=lambda path: path.as_posix()))
            + tuple(
                sorted(self.signature_directories, key=lambda path: path.as_posix())
            )
            + (self.output_root,)
        )
        for path in directories:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


@dataclass(frozen=True, slots=True)
class Q1FormalArchiveMaterializationResultV1:
    output_root: Path
    closure_bundle_root: bytes
    signature_manifest_roots: tuple[bytes, bytes]
    signature_saturation_state_roots: tuple[bytes, bytes]
    archive_payload_bytes: int
    retained_chunk_count: int
    endpoint_result_bytes: bytes

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_root, Path)
            or type(self.closure_bundle_root) is not bytes
            or len(self.closure_bundle_root) != 32
            or type(self.signature_manifest_roots) is not tuple
            or len(self.signature_manifest_roots) != 2
            or any(
                type(value) is not bytes or len(value) != 32
                for value in self.signature_manifest_roots
            )
            or type(self.signature_saturation_state_roots) is not tuple
            or len(self.signature_saturation_state_roots) != 2
            or any(
                type(value) is not bytes or len(value) != 32
                for value in self.signature_saturation_state_roots
            )
            or type(self.archive_payload_bytes) is not int
            or self.archive_payload_bytes <= 0
            or type(self.retained_chunk_count) is not int
            or self.retained_chunk_count <= 0
            or type(self.endpoint_result_bytes) is not bytes
            or not self.endpoint_result_bytes.endswith(b"\n")
        ):
            _fail(
                "FAIL_Q1_FORMAL_MATERIALIZER_RESULT",
                "materialization result boundary differs",
            )


def _stream_json_v1(stream: object) -> dict[str, object]:
    descriptor = stream.descriptor  # type: ignore[attr-defined]
    return {
        "archive_root": _sha256_id(descriptor.archive_root),
        "canonical_record_payload_bytes": (
            stream.canonical_record_payload_bytes  # type: ignore[attr-defined]
        ),
        "chunk_count": descriptor.chunk_count,
        "chunk_manifest_subtree_root": _sha256_id(
            descriptor.chunk_manifest_subtree_root
        ),
        "counting_discard_commitment": _sha256_id(
            stream.diagnostic_commitment  # type: ignore[attr-defined]
        ),
        "external_sort_projection_root": _sha256_id(
            stream.external_sort_projection.diagnostic_root  # type: ignore[attr-defined]
        ),
        "framed_bytes": descriptor.framed_stream_bytes,
        "record_count": descriptor.record_count,
        "stream_kind": descriptor.stream_kind_id.name,
        "stream_kind_id": int(descriptor.stream_kind_id),
    }


def _partition_json_v1(
    partition: _projection.FullNode6PartitionProjectionV1,
) -> dict[str, object]:
    row = partition.projection_partition_row
    return {
        "chunk_manifest_archive_root": _sha256_id(
            partition.signature_archive_manifest.chunk_manifest_archive_root
        ),
        "chunk_manifest_count": (
            partition.signature_archive_manifest.chunk_manifest_count
        ),
        "fixed_point_record_root": _sha256_id(
            partition.fixed_point_record.record_root
        ),
        "input_signature_id": partition.input_signature_id,
        "input_signature_name": (
            "OddInputV1" if partition.input_signature_id == 1 else "SinkInputV1"
        ),
        "partition_external_sort_root": _sha256_id(
            row.external_sort_projection_root
        ),
        "partition_stream_commitment_root": _sha256_id(
            row.diagnostic_stream_commitment
        ),
        "projected_partition_payload_bytes": (
            row.projected_partition_payload_bytes
        ),
        "same_run_record_set_root": _sha256_id(
            partition.same_run_record_set_root
        ),
        "saturation_state_root": _sha256_id(partition.saturation_state_root),
        "signature_manifest_root": _sha256_id(
            partition.signature_archive_manifest.manifest_root
        ),
        "streams": [_stream_json_v1(stream) for stream in partition.streams],
        "universe_root": _sha256_id(partition.universe_root),
    }


def materialize_full_node6_formal_archive_v1(
    diagnostic: _full.FullNode6CapacityDiagnosticV1,
    snapshots: tuple[_snapshot.Q1PartitionSnapshotV1 | None, ...],
    *,
    admission: Q1FormalArchiveMaterializerAdmissionV1,
    output_root: Path,
    scratch_root: Path,
    projection_runner: Callable[..., _projection.FullNode6ProjectionResultV1] = (
        _projection.project_full_node6_same_run_v1
    ),
) -> Q1FormalArchiveMaterializationResultV1:
    """Retain one full-node6 same-run archive without starting a measurement."""

    if type(admission) is not Q1FormalArchiveMaterializerAdmissionV1:
        detail = (
            "diagnostic admission is not formal materialization admission"
            if type(admission) is _full.FullNode6AdmissionV1
            else "exact formal materializer admission is required"
        )
        _fail("REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION", detail)
    if type(diagnostic) is not _full.FullNode6CapacityDiagnosticV1:
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_RESULT",
            "internal same-run carrier exact type differs",
        )
    if (
        diagnostic.capsule_digest != admission.capsule_digest
        or diagnostic.source_commit != admission.source_commit
    ):
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_RESULT",
            "admission and same-run carrier identities differ",
        )
    if (
        type(snapshots) is not tuple
        or len(snapshots) != 2
        or any(
            type(snapshot) is not _snapshot.Q1PartitionSnapshotV1
            for snapshot in snapshots
        )
    ):
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_RESULT",
            "exact Odd and Sink same-run snapshots are required",
        )
    for input_signature_id, snapshot, limits in zip(
        (1, 2),
        snapshots,
        _FORMAL_LIMITS_BY_SIGNATURE,
        strict=True,
    ):
        assert snapshot is not None
        if (
            snapshot.input_signature_id != input_signature_id
            or snapshot.limits != limits
        ):
            _fail(
                "REJECT_Q1_FORMAL_MATERIALIZER_RESULT",
                "same-run snapshot formal capacity limits differ",
            )
    if not isinstance(scratch_root, Path) or not scratch_root.is_dir():
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_SCRATCH",
            "scratch root must be an existing directory",
        )
    writer = _ArchiveDirectoryWriterV1(output_root)
    projected = projection_runner(
        diagnostic,
        snapshots,
        scratch_root=scratch_root,
        retained_chunk_consumer=writer.retain_chunk,
    )
    if type(projected) is not _projection.FullNode6ProjectionResultV1:
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_RESULT",
            "formal projection exact type differs",
        )
    _require_formal_scratch_limit_v1(
        projected.actual_preflight_peak_scratch_bytes
    )
    for partition in projected.partitions:
        writer.write_partition_material(partition)
    closure_frame_bytes = writer.write_closure(projected.closure_bundle)
    if (
        closure_frame_bytes
        != projected.archive_projection_result.projected_closure_bundle_frame_bytes
    ):
        _fail(
            "FAIL_Q1_FORMAL_MATERIALIZER_LENGTH",
            "closure frame length differs from frozen projection",
        )
    archive_payload_bytes = writer.archive_payload_bytes()
    if (
        archive_payload_bytes
        != projected.archive_projection_result.projected_archive_payload_bytes_per_endpoint
    ):
        _fail(
            "FAIL_Q1_FORMAL_MATERIALIZER_LENGTH",
            "retained archive length differs from frozen projection",
        )

    endpoint_result = {
        "action_id": ACTION_ID,
        "actor_id": ACTOR_ID,
        "actual_formal_peak_scratch_bytes": (
            projected.actual_preflight_peak_scratch_bytes
        ),
        "actual_formal_terminal_live_scratch_bytes": (
            projected.actual_preflight_scratch_live_bytes_at_terminal
        ),
        "capsule_digest": diagnostic.capsule_digest,
        "closure_bundle_root": _sha256_id(projected.closure_bundle.bundle_root),
        "closure_semantics_version": _full._capacity.CLOSURE_SEMANTICS_VERSION,
        "diagnostic_only": False,
        "dsl_version": _full._capacity.DSL_VERSION,
        "formal_archive_materialized": True,
        "formal_archive_payload_bytes": archive_payload_bytes,
        "formal_capacity_limits_by_signature": [
            _formal_capacity_limit_row_v1(input_signature_id, limits)
            for input_signature_id, limits in zip(
                (1, 2),
                _FORMAL_LIMITS_BY_SIGNATURE,
                strict=True,
            )
        ],
        "formal_output_bytes_limit": FORMAL_OUTPUT_BYTES_LIMIT,
        "formal_scratch_bytes_limit": FORMAL_SCRATCH_BYTES_LIMIT,
        "full_v16_structural_limits_applied": True,
        "implementation_id": IMPLEMENTATION_ID,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "partitions": [
            _partition_json_v1(partition) for partition in projected.partitions
        ],
        "prestart_admission_receipt_root": (
            admission.prestart_admission_receipt_root
        ),
        "q1_gate_count": Q1_GATE_COUNT,
        "q1_gate_mask": Q1_GATE_MASK,
        "q1_gate_total": Q1_GATE_TOTAL,
        "q1_output_slots": [None] * 8,
        "q1_state": Q1_STATE,
        "role_evaluation_performed": False,
        "run_genesis_root": admission.run_genesis_root,
        "schema_version": SCHEMA_VERSION,
        "source_commit": diagnostic.source_commit,
        "split_accessed": False,
        "status": STATUS,
        "target_truth_accessed": False,
    }
    endpoint_result_bytes = _projection.canonical_json_bytes_v1(endpoint_result)
    _require_formal_output_limit_v1(
        archive_payload_bytes,
        len(endpoint_result_bytes),
    )
    _write_new(output_root / "endpoint-result.json", endpoint_result_bytes)
    writer.fsync_directories()
    return Q1FormalArchiveMaterializationResultV1(
        output_root=output_root,
        closure_bundle_root=projected.closure_bundle.bundle_root,
        signature_manifest_roots=tuple(
            partition.signature_archive_manifest.manifest_root
            for partition in projected.partitions
        ),
        signature_saturation_state_roots=tuple(
            partition.saturation_state_root for partition in projected.partitions
        ),
        archive_payload_bytes=archive_payload_bytes,
        retained_chunk_count=writer.chunk_count,
        endpoint_result_bytes=endpoint_result_bytes,
    )


def run_full_node6_formal_materializer_v1(
    admission: Q1FormalArchiveMaterializerAdmissionV1,
    *,
    output_root: Path,
    scratch_root: Path,
    partition_runner: Callable[..., tuple[object, object]] = (
        _full._run_partition_once_v1
    ),
    projection_runner: Callable[..., _projection.FullNode6ProjectionResultV1] = (
        _projection.project_full_node6_same_run_v1
    ),
) -> Q1FormalArchiveMaterializationResultV1:
    """Measure Odd then Sink once and retain their same-run formal archive."""

    if type(admission) is not Q1FormalArchiveMaterializerAdmissionV1:
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION",
            "exact formal 14-of-20 admission is required",
        )
    rows: list[_full.FullNode6PartitionDiagnosticV1] = []
    snapshots: list[_snapshot.Q1PartitionSnapshotV1] = []
    for signature_id, limits in zip(
        (1, 2),
        _FORMAL_LIMITS_BY_SIGNATURE,
        strict=True,
    ):
        row, snapshot = partition_runner(signature_id, limits)
        if type(row) is not _full.FullNode6PartitionDiagnosticV1:
            _fail(
                "REJECT_Q1_FORMAL_MATERIALIZER_RESULT",
                "partition result exact type differs",
            )
        rows.append(row)
        if row.terminal_status != _full.TERMINAL_SATURATED:
            _fail(
                "INCONCLUSIVE_Q1_FORMAL_RESOURCE_GUARD",
                "same-run partition hit a formal resource guard",
            )
        if type(snapshot) is not _snapshot.Q1PartitionSnapshotV1:
            _fail(
                "REJECT_Q1_FORMAL_MATERIALIZER_RESULT",
                "saturated partition returned no exact same-run snapshot",
            )
        if snapshot.limits != limits:
            _fail(
                "REJECT_Q1_FORMAL_MATERIALIZER_RESULT",
                "same-run snapshot formal capacity limits differ",
            )
        snapshots.append(snapshot)
    diagnostic = _full.FullNode6CapacityDiagnosticV1(
        schema_version=_full.SCHEMA_VERSION,
        action_id=_full.ACTION_ID,
        capsule_digest=admission.capsule_digest,
        source_commit=admission.source_commit,
        terminal_status=_full.TERMINAL_SATURATED,
        partitions=tuple(rows),
    )
    return materialize_full_node6_formal_archive_v1(
        diagnostic,
        tuple(snapshots),
        admission=admission,
        output_root=output_root,
        scratch_root=scratch_root,
        projection_runner=projection_runner,
    )


__all__ = [
    "ACTION_ID",
    "ACTOR_ID",
    "ADMISSION_SCHEMA_VERSION",
    "FORMAL_OUTPUT_BYTES_LIMIT",
    "FORMAL_SCRATCH_BYTES_LIMIT",
    "IMPLEMENTATION_ID",
    "Q1_GATE_COUNT",
    "Q1_GATE_MASK",
    "Q1_GATE_TOTAL",
    "Q1_STATE",
    "Q1FormalArchiveMaterializerAdmissionV1",
    "Q1FormalArchiveMaterializationError",
    "Q1FormalArchiveMaterializationResultV1",
    "SCHEMA_VERSION",
    "STATUS",
    "materialize_full_node6_formal_archive_v1",
    "run_full_node6_formal_materializer_v1",
]

"""Offline trusted-host replay for materialized Q1 formal archives.

The actor reads two already-completed endpoint artifact trees.  It does not
enumerate the quotient, start Q1, read target truth or split state, evaluate a
role, publish state, or invoke another process.  Every formal frame is decoded
through the frozen strict-CBOR/archive classes, every archive root is rebuilt,
and the Python and Rust formal files must agree byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Final, NoReturn

from . import phase3_q1_archive_projection_v1 as _projection
from .phase3_q1_formal_archive_contract_v1 import (
    ARCHIVE_CHUNK_MANIFEST_SCHEMA_ID,
    CLOSURE_SEMANTICS_VERSION,
    CLOSURE_BUNDLE_SCHEMA_ID,
    DSL_VERSION,
    EXPECTED_COVERAGE_RECORD_COUNT,
    FIXED_POINT_SCHEMA_ID,
    MAX_CHUNK_FRAMED_BYTES,
    MAX_RECORDS_PER_CHUNK,
    Q1_ARCHIVE_CHUNK_MANIFEST_TAG,
    Q1_CLOSURE_BUNDLE_TAG,
    Q1_FIXED_POINT_TAG,
    Q1_RESOURCE_GUARD_REGISTRY,
    Q1_SIGNATURE_ARCHIVE_MANIFEST_TAG,
    SIGNATURE_ARCHIVE_MANIFEST_SCHEMA_ID,
    STREAM_DESCRIPTOR_SCHEMA_ID,
    ArchiveStreamKindId,
    Q1ArchiveChunkManifestV1,
    Q1ClosureBundleV1,
    Q1FixedPointRecordV1,
    Q1SignatureArchiveManifestV1,
    Q1StreamDescriptorV1,
    archive_root_v1,
    expected_coverage_registry_v1,
    frame_canonical_record_v1,
    projection_profile_root_v1,
    partition_stream_commitment_v1,
    replay_framed_records_v1,
    signature_saturation_state_root_v1,
)
from .phase3_q1_universe_v1 import production_universe_v1
from .strict_cbor_v1 import canonical_cbor_encode, content_hash, rfc6962_root


IMPLEMENTATION_ID: Final = "hegel-host-q1-formal-archive-replay-v1"
ACTION_ID: Final = "replay-full-node6-formal-archive-v1"
RESULT_SCHEMA_VERSION: Final = "hegel-q1-formal-host-replay-result/1"
RESULT_STATUS: Final = "DUAL_FORMAL_ARCHIVE_EXACT_AGREEMENT"

ENDPOINT_RESULT_SCHEMA: Final = (
    "hegel-phase3a-q1-full-node6-formal-archive-materializer/1"
)
PYTHON_RESULT_SCHEMA: Final = ENDPOINT_RESULT_SCHEMA
PYTHON_IMPLEMENTATION_ID: Final = "hegel-python-q1-formal-archive-materializer-v1"
ENDPOINT_RESULT_STATUS: Final = (
    "FULL_NODE6_FORMAL_ARCHIVE_MATERIALIZED_Q1_NOT_RUN"
)
PYTHON_RESULT_STATUS: Final = ENDPOINT_RESULT_STATUS
RUST_RESULT_SCHEMA: Final = ENDPOINT_RESULT_SCHEMA
RUST_IMPLEMENTATION_ID: Final = "hegel-rust-q1-formal-archive-materializer-v1"
RUST_RESULT_STATUS: Final = ENDPOINT_RESULT_STATUS
ENDPOINT_ACTION_ID: Final = "materialize-full-node6-formal-archive-v1"

SIGNATURE_DIRECTORY_NAMES: Final = ("signature-01", "signature-02")
STREAM_DIRECTORY_NAMES: Final = (
    "stream-01",
    "stream-02",
    "stream-03",
    "stream-04",
)
SIGNATURE_FILE_NAMES: Final = (
    "chunk-manifests.framed",
    "fixed-point.framed",
    "signature-manifest.framed",
)
ROOT_FORMAL_FILE_NAMES: Final = ("closure-bundle.framed",)
ENDPOINT_RESULT_NAME: Final = "endpoint-result.json"

ENDPOINT_RESULT_KEYS: Final = frozenset(
    (
        "action_id",
        "actor_id",
        "actual_formal_peak_scratch_bytes",
        "actual_formal_terminal_live_scratch_bytes",
        "capsule_digest",
        "closure_bundle_root",
        "closure_semantics_version",
        "diagnostic_only",
        "dsl_version",
        "formal_archive_materialized",
        "formal_archive_payload_bytes",
        "formal_capacity_limits_by_signature",
        "formal_output_bytes_limit",
        "formal_scratch_bytes_limit",
        "full_v16_structural_limits_applied",
        "implementation_id",
        "maximum_ast_depth",
        "maximum_ast_node_count",
        "partitions",
        "prestart_admission_receipt_root",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_gate_total",
        "q1_output_slots",
        "q1_state",
        "role_evaluation_performed",
        "run_genesis_root",
        "schema_version",
        "source_commit",
        "split_accessed",
        "status",
        "target_truth_accessed",
    )
)
FORMAL_CAPACITY_LIMIT_KEYS: Final = frozenset(
    (
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
)
FORMAL_CAPACITY_LIMIT_FIELDS: Final = (
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
FORMAL_CAPACITY_LIMIT_ROWS: Final = (
    (1, 3, 6, 32_768, 1_024, 4_096, 32, 8_192, 4_096, 16_384, 7, 3_600),
    (2, 3, 6, 32_768, 256, 1_024, 256, 8_192, 4_096, 16_384, 7, 3_600),
)
FORMAL_OUTPUT_BYTES_LIMIT: Final = 16 * 1024 * 1024
FORMAL_SCRATCH_BYTES_LIMIT: Final = 4 * 1024 * 1024
PARTITION_RESULT_KEYS: Final = frozenset(
    (
        "chunk_manifest_archive_root",
        "chunk_manifest_count",
        "fixed_point_record_root",
        "input_signature_id",
        "input_signature_name",
        "partition_external_sort_root",
        "partition_stream_commitment_root",
        "projected_partition_payload_bytes",
        "same_run_record_set_root",
        "saturation_state_root",
        "signature_manifest_root",
        "streams",
        "universe_root",
    )
)
STREAM_RESULT_KEYS: Final = frozenset(
    (
        "archive_root",
        "canonical_record_payload_bytes",
        "chunk_count",
        "chunk_manifest_subtree_root",
        "counting_discard_commitment",
        "external_sort_projection_root",
        "framed_bytes",
        "record_count",
        "stream_kind",
        "stream_kind_id",
    )
)
INPUT_SIGNATURE_NAMES: Final = ("OddInputV1", "SinkInputV1")
STREAM_KIND_NAMES: Final = ("PROGRAM", "COHORT", "CLASS", "COVERAGE")

_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_ID = re.compile(r"[0-9a-f]{40}\Z")


class Q1FormalHostReplayError(RuntimeError):
    """Stable rejection from the offline formal host replay."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1FormalHostReplayError(code, detail)


def _root_id(value: bytes) -> str:
    return "sha256:" + value.hex()


def _require_root_text(value: object, expected: bytes, name: str) -> None:
    if value != _root_id(expected):
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", f"{name} differs from replay")


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
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", f"canonical JSON: {error}")


def _strict_json_object(payload: bytes, name: str) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", f"{name}: duplicate {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(
            "REJECT_Q1_FORMAL_ENDPOINT_RESULT",
            f"{name}: non-finite JSON token {value}",
        )

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", f"{name}: {error}")
    if type(value) is not dict:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", f"{name} must be one object")
    if canonical_json_bytes_v1(value) != payload:
        _fail(
            "REJECT_Q1_FORMAL_ENDPOINT_RESULT",
            f"{name} is not exact canonical one-line JSON",
        )
    return value


def _read_file(path: Path, label: str) -> bytes:
    if not path.is_file():
        _fail("REJECT_Q1_FORMAL_LAYOUT", f"missing regular file: {label}")
    try:
        return path.read_bytes()
    except OSError as error:
        _fail("REJECT_Q1_FORMAL_LAYOUT", f"cannot read {label}: {error}")


def _require_directory_entries(
    directory: Path,
    expected_names: tuple[str, ...],
    label: str,
) -> None:
    if not directory.is_dir():
        _fail("REJECT_Q1_FORMAL_LAYOUT", f"missing directory: {label}")
    try:
        actual = tuple(sorted(item.name for item in directory.iterdir()))
    except OSError as error:
        _fail("REJECT_Q1_FORMAL_LAYOUT", f"cannot enumerate {label}: {error}")
    if actual != tuple(sorted(expected_names)):
        _fail("REJECT_Q1_FORMAL_LAYOUT", f"{label} file closure differs")


def _single_framed_object(path: Path, label: str) -> tuple[object, ...]:
    payload = _read_file(path, label)
    try:
        values = replay_framed_records_v1(payload)
    except ValueError as error:
        _fail("REJECT_Q1_FORMAL_FRAME", f"{label}: {error}")
    if len(values) != 1 or frame_canonical_record_v1(values[0]) != payload:
        _fail("REJECT_Q1_FORMAL_FRAME", f"{label} must contain one exact frame")
    return values[0]


def _decode_chunk_manifest(value: tuple[object, ...]) -> Q1ArchiveChunkManifestV1:
    if (
        type(value) is not tuple
        or len(value) != 14
        or value[:3]
        != (1, Q1_ARCHIVE_CHUNK_MANIFEST_TAG, ARCHIVE_CHUNK_MANIFEST_SCHEMA_ID)
    ):
        _fail("REJECT_Q1_FORMAL_MANIFEST", "chunk manifest wire differs")
    try:
        manifest = Q1ArchiveChunkManifestV1(
            value[3],
            value[4],
            ArchiveStreamKindId(value[5]),
            value[6],
            value[7],
            value[8],
            value[9],
            value[10],
            value[11],
            value[12],
            value[13],
        )
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_FORMAL_MANIFEST", f"chunk manifest: {error}")
    if manifest.canonical_object() != value:
        _fail("REJECT_Q1_FORMAL_MANIFEST", "chunk manifest replay differs")
    return manifest


def _decode_descriptor(value: object) -> Q1StreamDescriptorV1:
    if (
        type(value) is not tuple
        or len(value) != 8
        or value[:2] != (1, STREAM_DESCRIPTOR_SCHEMA_ID)
    ):
        _fail("REJECT_Q1_FORMAL_MANIFEST", "stream descriptor wire differs")
    try:
        descriptor = Q1StreamDescriptorV1(
            ArchiveStreamKindId(value[2]),
            value[3],
            value[4],
            value[5],
            value[6],
            value[7],
        )
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_FORMAL_MANIFEST", f"stream descriptor: {error}")
    if descriptor.canonical_object() != value:
        _fail("REJECT_Q1_FORMAL_MANIFEST", "stream descriptor replay differs")
    return descriptor


def _decode_fixed_point(value: tuple[object, ...]) -> Q1FixedPointRecordV1:
    if (
        type(value) is not tuple
        or len(value) != 30
        or value[:3] != (1, Q1_FIXED_POINT_TAG, FIXED_POINT_SCHEMA_ID)
    ):
        _fail("REJECT_Q1_FORMAL_FIXED_POINT", "fixed-point wire differs")
    try:
        fixed_point = Q1FixedPointRecordV1(*value[3:])
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_FORMAL_FIXED_POINT", str(error))
    if fixed_point.canonical_object() != value:
        _fail("REJECT_Q1_FORMAL_FIXED_POINT", "fixed-point replay differs")
    return fixed_point


def _decode_signature_manifest(
    value: tuple[object, ...],
) -> Q1SignatureArchiveManifestV1:
    if (
        type(value) is not tuple
        or len(value) != 15
        or value[:3]
        != (
            1,
            Q1_SIGNATURE_ARCHIVE_MANIFEST_TAG,
            SIGNATURE_ARCHIVE_MANIFEST_SCHEMA_ID,
        )
        or value[8] != MAX_RECORDS_PER_CHUNK
        or value[9] != MAX_CHUNK_FRAMED_BYTES
        or type(value[10]) is not tuple
    ):
        _fail("REJECT_Q1_FORMAL_MANIFEST", "signature manifest wire differs")
    descriptors = tuple(_decode_descriptor(item) for item in value[10])
    try:
        manifest = Q1SignatureArchiveManifestV1(
            value[3],
            value[4],
            value[5],
            value[6],
            value[7],
            descriptors,
            value[11],
            value[12],
            value[13],
            value[14],
        )
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_FORMAL_MANIFEST", f"signature manifest: {error}")
    if manifest.canonical_object() != value:
        _fail("REJECT_Q1_FORMAL_MANIFEST", "signature manifest replay differs")
    return manifest


def _decode_closure_bundle(value: tuple[object, ...]) -> Q1ClosureBundleV1:
    if (
        type(value) is not tuple
        or len(value) != 7
        or value[:3] != (1, Q1_CLOSURE_BUNDLE_TAG, CLOSURE_BUNDLE_SCHEMA_ID)
        or value[5] != 2
    ):
        _fail("REJECT_Q1_FORMAL_CLOSURE", "closure bundle wire differs")
    try:
        bundle = Q1ClosureBundleV1(value[3], value[4], value[6])
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_FORMAL_CLOSURE", str(error))
    if bundle.canonical_object() != value:
        _fail("REJECT_Q1_FORMAL_CLOSURE", "closure bundle replay differs")
    return bundle


@dataclass(frozen=True, slots=True)
class Q1FormalStreamReplayV1:
    stream_kind_id: ArchiveStreamKindId
    descriptor: Q1StreamDescriptorV1
    canonical_record_payload_bytes: int
    manifests: tuple[Q1ArchiveChunkManifestV1, ...]
    framed_relative_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Q1FormalStreamMaterialV1:
    stream_kind_id: ArchiveStreamKindId
    descriptor: Q1StreamDescriptorV1
    record_objects: tuple[tuple[object, ...], ...]
    decoded_records: tuple[object, ...]
    manifests: tuple[Q1ArchiveChunkManifestV1, ...]
    framed_relative_paths: tuple[str, ...]

    def summary(self) -> Q1FormalStreamReplayV1:
        return Q1FormalStreamReplayV1(
            self.stream_kind_id,
            self.descriptor,
            sum(len(canonical_cbor_encode(row)) for row in self.record_objects),
            self.manifests,
            self.framed_relative_paths,
        )


@dataclass(frozen=True, slots=True)
class Q1FormalSignatureReplayV1:
    input_signature_id: int
    universe_root: bytes
    streams: tuple[Q1FormalStreamReplayV1, ...]
    fixed_point: Q1FixedPointRecordV1
    signature_manifest: Q1SignatureArchiveManifestV1
    same_run_record_set_root: bytes
    payload_bytes: int
    record_stream_bytes: int
    chunk_manifest_stream_bytes: int
    fixed_point_frame_bytes: int
    signature_manifest_frame_bytes: int
    framed_relative_paths: tuple[str, ...]

    @property
    def signature_manifest_root(self) -> bytes:
        return self.signature_manifest.manifest_root

    @property
    def saturation_state_root(self) -> bytes:
        return self.signature_manifest.saturation_state_root


@dataclass(frozen=True, slots=True)
class Q1FormalEndpointReplayV1:
    root: Path
    endpoint_kind: str
    capsule_digest: str
    source_commit: str
    run_genesis_root: str
    prestart_admission_receipt_root: str
    semantic_binding_root: bytes
    projection_profile_root: bytes
    closure_bundle: Q1ClosureBundleV1
    signatures: tuple[Q1FormalSignatureReplayV1, ...]
    archive_payload_bytes: int
    formal_relative_paths: tuple[str, ...]
    common_metadata_bytes: bytes

    @property
    def closure_bundle_root(self) -> bytes:
        return self.closure_bundle.bundle_root


def _replay_stream(
    root: Path,
    *,
    input_signature_id: int,
    universe_root: bytes,
    stream_kind_id: ArchiveStreamKindId,
    manifests: tuple[Q1ArchiveChunkManifestV1, ...],
) -> _Q1FormalStreamMaterialV1:
    signature_name = SIGNATURE_DIRECTORY_NAMES[input_signature_id - 1]
    stream_name = STREAM_DIRECTORY_NAMES[int(stream_kind_id) - 1]
    directory = root / signature_name / stream_name
    expected_files = tuple(
        f"chunk-{manifest.chunk_index:06d}.framed" for manifest in manifests
    )
    _require_directory_entries(directory, expected_files, f"{signature_name}/{stream_name}")
    blobs = tuple(
        _read_file(directory / name, f"{signature_name}/{stream_name}/{name}")
        for name in expected_files
    )
    try:
        projection = _projection.Q1ChunkProjectionV1(manifests, blobs)
    except ValueError as error:
        _fail(
            "REJECT_Q1_FORMAL_STREAM",
            f"{signature_name}/{stream_name}: {error}",
        )
    objects = tuple(
        record
        for blob in blobs
        for record in replay_framed_records_v1(blob)
    )
    decoded = tuple(
        _projection._strict_decode_formal_record_v1(record, stream_kind_id)
        for record in objects
    )
    if not objects:
        _fail("REJECT_Q1_FORMAL_STREAM", f"{signature_name}/{stream_name} is empty")
    descriptor = Q1StreamDescriptorV1(
        stream_kind_id,
        len(objects),
        archive_root_v1(objects),
        sum(len(blob) for blob in blobs),
        len(manifests),
        rfc6962_root(tuple(item.canonical_object() for item in manifests)),
    )
    relative_paths = tuple(f"{signature_name}/{stream_name}/{name}" for name in expected_files)
    return _Q1FormalStreamMaterialV1(
        stream_kind_id,
        descriptor,
        objects,
        decoded,
        manifests,
        relative_paths,
    )


def _replay_signature(root: Path, input_signature_id: int) -> Q1FormalSignatureReplayV1:
    signature_name = SIGNATURE_DIRECTORY_NAMES[input_signature_id - 1]
    signature_root = root / signature_name
    _require_directory_entries(
        signature_root,
        SIGNATURE_FILE_NAMES + STREAM_DIRECTORY_NAMES,
        signature_name,
    )
    manifest_path = signature_root / "chunk-manifests.framed"
    manifest_payload = _read_file(manifest_path, f"{signature_name}/chunk-manifests.framed")
    try:
        manifest_objects = replay_framed_records_v1(manifest_payload)
        manifests = tuple(_decode_chunk_manifest(value) for value in manifest_objects)
    except ValueError as error:
        _fail("REJECT_Q1_FORMAL_MANIFEST", f"{signature_name}: {error}")
    expected_order = tuple(
        (stream_kind, chunk_index)
        for stream_kind in ArchiveStreamKindId
        for chunk_index in range(
            sum(1 for item in manifests if item.stream_kind_id is stream_kind)
        )
    )
    actual_order = tuple((item.stream_kind_id, item.chunk_index) for item in manifests)
    if actual_order != expected_order or any(
        item.input_signature_id != input_signature_id for item in manifests
    ):
        _fail(
            "REJECT_Q1_FORMAL_MANIFEST",
            f"{signature_name} aggregate chunk-manifest order differs",
        )
    universe_root = production_universe_v1(input_signature_id).universe_root
    if any(item.universe_root != universe_root for item in manifests):
        _fail("REJECT_Q1_FORMAL_MANIFEST", f"{signature_name} universe differs")
    streams = tuple(
        _replay_stream(
            root,
            input_signature_id=input_signature_id,
            universe_root=universe_root,
            stream_kind_id=stream_kind,
            manifests=tuple(
                item for item in manifests if item.stream_kind_id is stream_kind
            ),
        )
        for stream_kind in ArchiveStreamKindId
    )
    fixed_path = signature_root / "fixed-point.framed"
    signature_manifest_path = signature_root / "signature-manifest.framed"
    fixed_point = _decode_fixed_point(
        _single_framed_object(fixed_path, f"{signature_name}/fixed-point.framed")
    )
    signature_manifest = _decode_signature_manifest(
        _single_framed_object(
            signature_manifest_path,
            f"{signature_name}/signature-manifest.framed",
        )
    )
    descriptors = tuple(stream.descriptor for stream in streams)
    if (
        fixed_point.input_signature_id != input_signature_id
        or fixed_point.universe_root != universe_root
        or signature_manifest.input_signature_id != input_signature_id
        or signature_manifest.universe_root != universe_root
        or signature_manifest.stream_descriptors != descriptors
        or signature_manifest.fixed_point_record_root != fixed_point.record_root
        or signature_manifest.chunk_manifest_count != len(manifests)
        or signature_manifest.chunk_manifest_archive_root
        != rfc6962_root(manifest_objects)
    ):
        _fail("REJECT_Q1_FORMAL_SIGNATURE", f"{signature_name} DAG replay differs")

    program, cohort, classes, coverage = streams
    if (
        fixed_point.program_count != program.descriptor.record_count
        or fixed_point.cohort_count != cohort.descriptor.record_count
        or fixed_point.class_count != classes.descriptor.record_count
        or coverage.descriptor.record_count != EXPECTED_COVERAGE_RECORD_COUNT
        or fixed_point.program_archive_root != program.descriptor.archive_root
        or fixed_point.bank_archive_root != cohort.descriptor.archive_root
        or fixed_point.class_archive_root != classes.descriptor.archive_root
        or fixed_point.coverage_archive_root != coverage.descriptor.archive_root
    ):
        _fail("REJECT_Q1_FORMAL_FIXED_POINT", f"{signature_name} stream roots differ")

    class_rows = classes.decoded_records
    coverage_rows = coverage.decoded_records
    if (
        fixed_point.bank_point_count
        != sum(row.bank_point_count for row in class_rows)  # type: ignore[attr-defined]
        or fixed_point.frontier_point_count
        != sum(row.visible_frontier_point_count for row in class_rows)  # type: ignore[attr-defined]
        or fixed_point.maximum_bank_points_per_class
        != max(row.bank_point_count for row in class_rows)  # type: ignore[attr-defined]
        or fixed_point.maximum_frontier_points_per_class
        != max(row.visible_frontier_point_count for row in class_rows)  # type: ignore[attr-defined]
        or fixed_point.raw_application_count
        != sum(row.eligible_application_count for row in coverage_rows)  # type: ignore[attr-defined]
        or fixed_point.strict_admitted_count
        != sum(row.strict_admitted_count for row in coverage_rows)  # type: ignore[attr-defined]
        or fixed_point.rewrite_collapse_count
        != sum(row.rewrite_collapse_count for row in coverage_rows)  # type: ignore[attr-defined]
    ):
        _fail("REJECT_Q1_FORMAL_FIXED_POINT", f"{signature_name} counters differ")

    record_set_preimage = (
        1,
        _projection.SNAPSHOT_RECORD_SET_SCHEMA_ID,
        input_signature_id,
        universe_root,
        program.record_objects,
        cohort.record_objects,
        classes.record_objects,
    )
    same_run_record_set_root = content_hash(
        _projection.SNAPSHOT_RECORD_SET_ROOT_DOMAIN,
        record_set_preimage,
    )
    try:
        _projection.Q1SnapshotRecordSetV1(
            input_signature_id,
            universe_root,
            program.decoded_records,
            cohort.decoded_records,
            classes.decoded_records,
            same_run_record_set_root,
        )
    except ValueError as error:
        _fail("REJECT_Q1_FORMAL_RECORD_SET", f"{signature_name}: {error}")

    expected_saturation = signature_saturation_state_root_v1(
        input_signature_id=input_signature_id,
        universe_root=universe_root,
        semantic_binding_root=signature_manifest.semantic_binding_root,
        projection_profile_root=signature_manifest.projection_profile_root,
        program_archive_root=program.descriptor.archive_root,
        cohort_archive_root=cohort.descriptor.archive_root,
        class_archive_root=classes.descriptor.archive_root,
        coverage_archive_root=coverage.descriptor.archive_root,
        fixed_point_record_root=fixed_point.record_root,
    )
    if signature_manifest.saturation_state_root != expected_saturation:
        _fail("REJECT_Q1_FORMAL_SIGNATURE", f"{signature_name} saturation root differs")

    fixed_bytes = fixed_path.stat().st_size
    signature_manifest_bytes = signature_manifest_path.stat().st_size
    record_stream_bytes = sum(item.descriptor.framed_stream_bytes for item in streams)
    payload_bytes = (
        record_stream_bytes
        + len(manifest_payload)
        + fixed_bytes
        + signature_manifest_bytes
    )
    paths = tuple(
        path
        for stream in streams
        for path in stream.framed_relative_paths
    ) + (
        f"{signature_name}/chunk-manifests.framed",
        f"{signature_name}/fixed-point.framed",
        f"{signature_name}/signature-manifest.framed",
    )
    stream_summaries = tuple(stream.summary() for stream in streams)
    return Q1FormalSignatureReplayV1(
        input_signature_id,
        universe_root,
        stream_summaries,
        fixed_point,
        signature_manifest,
        same_run_record_set_root,
        payload_bytes,
        record_stream_bytes,
        len(manifest_payload),
        fixed_bytes,
        signature_manifest_bytes,
        tuple(sorted(paths)),
    )


def _exact_uint(value: object, name: str, expected: int | None = None) -> int:
    if type(value) is not int or value < 0 or (expected is not None and value != expected):
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", f"{name} differs")
    return value


def _exact_root(value: object, name: str) -> bytes:
    if type(value) is not str or _SHA256_ID.fullmatch(value) is None:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", f"{name} differs")
    return bytes.fromhex(value[7:])


def _require_key_closure(
    value: object,
    expected_keys: frozenset[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != expected_keys:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", f"{name} key closure differs")
    return value


def _validate_endpoint_result_boundary(
    value: dict[str, object],
) -> tuple[str, str, str, str, str]:
    """Reject non-formal endpoint metadata before reading the large archive."""

    _require_key_closure(value, ENDPOINT_RESULT_KEYS, "endpoint result")
    if value["schema_version"] != ENDPOINT_RESULT_SCHEMA:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "materializer schema differs")
    if value["actor_id"] == "python":
        endpoint_kind = "python"
        expected_implementation = PYTHON_IMPLEMENTATION_ID
    elif value["actor_id"] == "rust":
        endpoint_kind = "rust"
        expected_implementation = RUST_IMPLEMENTATION_ID
    else:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "materializer actor differs")
    if (
        value["implementation_id"] != expected_implementation
        or value["action_id"] != ENDPOINT_ACTION_ID
        or value["status"] != ENDPOINT_RESULT_STATUS
        or value["dsl_version"] != DSL_VERSION
        or value["closure_semantics_version"] != CLOSURE_SEMANTICS_VERSION
        or value["q1_state"] != "NOT_RUN"
        or type(value["q1_output_slots"]) is not list
        or value["q1_output_slots"] != [None] * 8
        or value["formal_archive_materialized"] is not True
        or value["diagnostic_only"] is not False
        or value["full_v16_structural_limits_applied"] is not True
        or value["target_truth_accessed"] is not False
        or value["split_accessed"] is not False
        or value["role_evaluation_performed"] is not False
    ):
        _fail(
            "REJECT_Q1_FORMAL_ENDPOINT_RESULT",
            f"{endpoint_kind} is not a completed formal materializer result",
        )
    _exact_uint(value["q1_gate_count"], "q1_gate_count", 14)
    _exact_uint(value["q1_gate_total"], "q1_gate_total", 20)
    _exact_uint(value["q1_gate_mask"], "q1_gate_mask", 0x3FFF)
    _exact_uint(value["maximum_ast_depth"], "maximum_ast_depth", 3)
    _exact_uint(value["maximum_ast_node_count"], "maximum_ast_node_count", 6)
    actual_scratch = _exact_uint(
        value["actual_formal_peak_scratch_bytes"],
        "actual scratch",
    )
    _exact_uint(
        value["actual_formal_terminal_live_scratch_bytes"],
        "terminal live scratch",
        0,
    )
    formal_archive_bytes = _exact_uint(
        value["formal_archive_payload_bytes"],
        "formal archive bytes",
    )
    _exact_uint(
        value["formal_output_bytes_limit"],
        "formal_output_bytes_limit",
        FORMAL_OUTPUT_BYTES_LIMIT,
    )
    _exact_uint(
        value["formal_scratch_bytes_limit"],
        "formal_scratch_bytes_limit",
        FORMAL_SCRATCH_BYTES_LIMIT,
    )
    if (
        actual_scratch > FORMAL_SCRATCH_BYTES_LIMIT
        or formal_archive_bytes > FORMAL_OUTPUT_BYTES_LIMIT
    ):
        _fail(
            "REJECT_Q1_FORMAL_ENDPOINT_RESULT",
            "formal materializer resource guard differs",
        )
    _exact_root(value["closure_bundle_root"], "closure_bundle_root")

    capacity_limits = value["formal_capacity_limits_by_signature"]
    if type(capacity_limits) is not list or len(capacity_limits) != 2:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "formal capacity rows differ")
    for index, (claimed, expected) in enumerate(
        zip(capacity_limits, FORMAL_CAPACITY_LIMIT_ROWS, strict=True),
        start=1,
    ):
        claimed = _require_key_closure(
            claimed,
            FORMAL_CAPACITY_LIMIT_KEYS,
            f"formal capacity row {index}",
        )
        for field, expected_value in zip(
            FORMAL_CAPACITY_LIMIT_FIELDS,
            expected,
            strict=True,
        ):
            _exact_uint(claimed[field], field, expected_value)

    capsule_digest = value["capsule_digest"]
    source_commit = value["source_commit"]
    run_genesis_root = value["run_genesis_root"]
    prestart_receipt_root = value["prestart_admission_receipt_root"]
    if type(capsule_digest) is not str or _SHA256_ID.fullmatch(capsule_digest) is None:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "capsule digest differs")
    if type(source_commit) is not str or _COMMIT_ID.fullmatch(source_commit) is None:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "source commit differs")
    _exact_root(run_genesis_root, "run_genesis_root")
    _exact_root(prestart_receipt_root, "prestart_admission_receipt_root")

    partitions = value["partitions"]
    if type(partitions) is not list or len(partitions) != 2:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "partition result closure differs")
    for input_signature_id, claimed in enumerate(partitions, start=1):
        claimed = _require_key_closure(
            claimed,
            PARTITION_RESULT_KEYS,
            f"partition {input_signature_id}",
        )
        _exact_uint(
            claimed["input_signature_id"],
            "input_signature_id",
            input_signature_id,
        )
        if claimed["input_signature_name"] != INPUT_SIGNATURE_NAMES[input_signature_id - 1]:
            _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "partition order differs")
        for name in (
            "chunk_manifest_archive_root",
            "fixed_point_record_root",
            "partition_external_sort_root",
            "partition_stream_commitment_root",
            "same_run_record_set_root",
            "saturation_state_root",
            "signature_manifest_root",
            "universe_root",
        ):
            _exact_root(claimed[name], name)
        _exact_uint(claimed["chunk_manifest_count"], "chunk_manifest_count")
        _exact_uint(
            claimed["projected_partition_payload_bytes"],
            "projected_partition_payload_bytes",
        )
        streams = claimed["streams"]
        if type(streams) is not list or len(streams) != 4:
            _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "stream result closure differs")
        for stream_kind_id, stream in enumerate(streams, start=1):
            stream = _require_key_closure(
                stream,
                STREAM_RESULT_KEYS,
                f"stream {stream_kind_id}",
            )
            _exact_uint(
                stream["stream_kind_id"],
                "stream_kind_id",
                stream_kind_id,
            )
            if stream["stream_kind"] != STREAM_KIND_NAMES[stream_kind_id - 1]:
                _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "stream order differs")
            for name in (
                "archive_root",
                "chunk_manifest_subtree_root",
                "counting_discard_commitment",
                "external_sort_projection_root",
            ):
                _exact_root(stream[name], name)
            for name in (
                "canonical_record_payload_bytes",
                "chunk_count",
                "framed_bytes",
                "record_count",
            ):
                _exact_uint(stream[name], name)
    return (
        endpoint_kind,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_receipt_root,
    )


def _validate_endpoint_result(
    value: dict[str, object],
    endpoint: Q1FormalEndpointReplayV1,
) -> tuple[str, str, str, str, str, bytes]:
    (
        endpoint_kind,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_receipt_root,
    ) = _validate_endpoint_result_boundary(value)
    _require_root_text(
        value.get("closure_bundle_root"),
        endpoint.closure_bundle_root,
        "closure_bundle_root",
    )
    _exact_uint(
        value["formal_archive_payload_bytes"],
        "formal_archive_payload_bytes",
        endpoint.archive_payload_bytes,
    )
    if (
        value["actual_formal_peak_scratch_bytes"] > FORMAL_SCRATCH_BYTES_LIMIT
        or endpoint.archive_payload_bytes + len(canonical_json_bytes_v1(value))
        > FORMAL_OUTPUT_BYTES_LIMIT
    ):
        _fail(
            "REJECT_Q1_FORMAL_ENDPOINT_RESULT",
            "formal materializer resource guard differs",
        )

    partitions = value.get("partitions")
    if type(partitions) is not list or len(partitions) != 2:
        _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "partition result closure differs")
    for expected, claimed in zip(endpoint.signatures, partitions, strict=True):
        claimed = _require_key_closure(
            claimed,
            PARTITION_RESULT_KEYS,
            f"partition {expected.input_signature_id}",
        )
        if (
            type(claimed["input_signature_id"]) is not int
            or claimed["input_signature_id"] != expected.input_signature_id
            or claimed["input_signature_name"]
            != INPUT_SIGNATURE_NAMES[expected.input_signature_id - 1]
        ):
            _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "partition order differs")
        _require_root_text(claimed["universe_root"], expected.universe_root, "universe_root")
        _require_root_text(
            claimed["fixed_point_record_root"],
            expected.fixed_point.record_root,
            "fixed_point_record_root",
        )
        _require_root_text(
            claimed["signature_manifest_root"],
            expected.signature_manifest_root,
            "signature_manifest_root",
        )
        _require_root_text(
            claimed["saturation_state_root"],
            expected.saturation_state_root,
            "saturation_state_root",
        )
        _require_root_text(
            claimed["same_run_record_set_root"],
            expected.same_run_record_set_root,
            "same_run_record_set_root",
        )
        _require_root_text(
            claimed["chunk_manifest_archive_root"],
            expected.signature_manifest.chunk_manifest_archive_root,
            "chunk_manifest_archive_root",
        )
        _exact_uint(
            claimed["chunk_manifest_count"],
            "chunk_manifest_count",
            expected.signature_manifest.chunk_manifest_count,
        )
        _exact_uint(
            claimed["projected_partition_payload_bytes"],
            "projected_partition_payload_bytes",
            expected.payload_bytes,
        )
        streams = claimed.get("streams")
        if type(streams) is not list or len(streams) != 4:
            _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "stream result closure differs")
        counting_commitments: list[bytes] = []
        for replayed, stream in zip(expected.streams, streams, strict=True):
            stream = _require_key_closure(
                stream,
                STREAM_RESULT_KEYS,
                f"stream {int(replayed.stream_kind_id)}",
            )
            kind_index = int(replayed.stream_kind_id) - 1
            if (
                type(stream["stream_kind_id"]) is not int
                or stream["stream_kind_id"] != int(replayed.stream_kind_id)
                or stream["stream_kind"] != STREAM_KIND_NAMES[kind_index]
            ):
                _fail("REJECT_Q1_FORMAL_ENDPOINT_RESULT", "stream order differs")
            descriptor = replayed.descriptor
            _exact_uint(stream["record_count"], "record_count", descriptor.record_count)
            _exact_uint(stream["chunk_count"], "chunk_count", descriptor.chunk_count)
            _exact_uint(
                stream["framed_bytes"],
                "framed_bytes",
                descriptor.framed_stream_bytes,
            )
            _exact_uint(
                stream["canonical_record_payload_bytes"],
                "canonical_record_payload_bytes",
                replayed.canonical_record_payload_bytes,
            )
            _require_root_text(stream["archive_root"], descriptor.archive_root, "archive_root")
            _require_root_text(
                stream["chunk_manifest_subtree_root"],
                descriptor.chunk_manifest_subtree_root,
                "chunk_manifest_subtree_root",
            )
            counting_commitments.append(
                _exact_root(
                    stream["counting_discard_commitment"],
                    "counting_discard_commitment",
                )
            )
            _exact_root(
                stream["external_sort_projection_root"],
                "external_sort_projection_root",
            )
        fixed = expected.fixed_point
        stream_commitment = partition_stream_commitment_v1(
            input_signature_id=expected.input_signature_id,
            universe_root=expected.universe_root,
            raw_application_count=fixed.raw_application_count,
            behavior_class_count=fixed.class_count,
            cohort_count=fixed.cohort_count,
            bank_point_count=fixed.bank_point_count,
            frontier_point_count=fixed.frontier_point_count,
            maximum_bank_points_per_class=fixed.maximum_bank_points_per_class,
            maximum_frontier_points_per_class=(
                fixed.maximum_frontier_points_per_class
            ),
            program_record_count=expected.streams[0].descriptor.record_count,
            coverage_record_count=expected.streams[3].descriptor.record_count,
            projected_record_stream_bytes=expected.record_stream_bytes,
            projected_chunk_manifest_stream_bytes=(
                expected.chunk_manifest_stream_bytes
            ),
            stream_diagnostic_commitments=tuple(counting_commitments),
        )
        _require_root_text(
            claimed["partition_stream_commitment_root"],
            stream_commitment,
            "partition_stream_commitment_root",
        )
        _exact_root(
            claimed["partition_external_sort_root"],
            "partition_external_sort_root",
        )
    excluded = {
        "schema_version",
        "actor_id",
        "implementation_id",
        "actual_formal_peak_scratch_bytes",
        "actual_formal_terminal_live_scratch_bytes",
    }
    common_metadata = canonical_json_bytes_v1(
        {key: item for key, item in value.items() if key not in excluded}
    )
    return (
        endpoint_kind,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_receipt_root,
        common_metadata,
    )


def replay_endpoint_formal_archive_v1(root: Path) -> Q1FormalEndpointReplayV1:
    """Replay one exact materializer output tree without changing it."""

    if not isinstance(root, Path):
        raise TypeError("formal archive root must be pathlib.Path")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        _fail("REJECT_Q1_FORMAL_LAYOUT", str(error))
    _require_directory_entries(
        root,
        ROOT_FORMAL_FILE_NAMES + SIGNATURE_DIRECTORY_NAMES + (ENDPOINT_RESULT_NAME,),
        "endpoint root",
    )
    endpoint_value = _strict_json_object(
        _read_file(root / ENDPOINT_RESULT_NAME, ENDPOINT_RESULT_NAME),
        ENDPOINT_RESULT_NAME,
    )
    _validate_endpoint_result_boundary(endpoint_value)
    signatures = tuple(_replay_signature(root, index) for index in (1, 2))
    bundle_path = root / "closure-bundle.framed"
    bundle = _decode_closure_bundle(
        _single_framed_object(bundle_path, "closure-bundle.framed")
    )
    if bundle.signature_rows != tuple(
        (
            item.input_signature_id,
            item.universe_root,
            item.signature_manifest_root,
            item.saturation_state_root,
        )
        for item in signatures
    ):
        _fail("REJECT_Q1_FORMAL_CLOSURE", "closure signature rows differ")
    if any(
        item.signature_manifest.semantic_binding_root != bundle.semantic_binding_root
        or item.signature_manifest.projection_profile_root != bundle.projection_profile_root
        or item.fixed_point.projection_profile_root != bundle.projection_profile_root
        for item in signatures
    ):
        _fail("REJECT_Q1_FORMAL_CLOSURE", "closure semantic/profile binding differs")
    expected_profile = projection_profile_root_v1(
        semantic_binding_root=bundle.semantic_binding_root,
        coverage_registry_root=rfc6962_root(expected_coverage_registry_v1()),
        resource_guard_registry=Q1_RESOURCE_GUARD_REGISTRY,
    )
    if bundle.projection_profile_root != expected_profile:
        _fail("REJECT_Q1_FORMAL_CLOSURE", "projection profile root differs")

    formal_paths = tuple(
        sorted(
            ("closure-bundle.framed",)
            + tuple(path for item in signatures for path in item.framed_relative_paths)
        )
    )
    archive_payload_bytes = sum((root / path).stat().st_size for path in formal_paths)
    provisional = Q1FormalEndpointReplayV1(
        root,
        "unknown",
        "",
        "",
        "",
        "",
        bundle.semantic_binding_root,
        bundle.projection_profile_root,
        bundle,
        signatures,
        archive_payload_bytes,
        formal_paths,
        b"",
    )
    (
        endpoint_kind,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_receipt_root,
        common_metadata,
    ) = _validate_endpoint_result(endpoint_value, provisional)
    return Q1FormalEndpointReplayV1(
        root,
        endpoint_kind,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_receipt_root,
        bundle.semantic_binding_root,
        bundle.projection_profile_root,
        bundle,
        signatures,
        archive_payload_bytes,
        formal_paths,
        common_metadata,
    )


def _require_exact_formal_file_agreement(
    python: Q1FormalEndpointReplayV1,
    rust: Q1FormalEndpointReplayV1,
) -> None:
    if python.formal_relative_paths != rust.formal_relative_paths:
        _fail("REJECT_Q1_FORMAL_DUAL_AGREEMENT", "formal file closure differs")
    for relative in python.formal_relative_paths:
        left_path = python.root / relative
        right_path = rust.root / relative
        if left_path.stat().st_size != right_path.stat().st_size:
            _fail("REJECT_Q1_FORMAL_DUAL_AGREEMENT", f"byte length differs: {relative}")
        with left_path.open("rb") as left, right_path.open("rb") as right:
            while True:
                left_bytes = left.read(1024 * 1024)
                right_bytes = right.read(1024 * 1024)
                if left_bytes != right_bytes:
                    _fail("REJECT_Q1_FORMAL_DUAL_AGREEMENT", f"bytes differ: {relative}")
                if not left_bytes:
                    break


def _replay_exact_matching_endpoint_v1(
    canonical: Q1FormalEndpointReplayV1,
    matching_root: Path,
) -> Q1FormalEndpointReplayV1:
    """Bind endpoint metadata after exact comparison with one replayed tree.

    Re-decoding an identical byte tree cannot change the host decision.  The
    second endpoint is therefore checked by exact layout plus streaming byte
    comparison, while the first tree supplies the already-recomputed roots.
    This reads every second-endpoint formal byte without retaining or hashing a
    duplicate copy.
    """

    if not isinstance(matching_root, Path):
        raise TypeError("formal archive root must be pathlib.Path")
    try:
        matching_root = matching_root.resolve(strict=True)
    except OSError as error:
        _fail("REJECT_Q1_FORMAL_LAYOUT", str(error))
    _require_directory_entries(
        matching_root,
        ROOT_FORMAL_FILE_NAMES + SIGNATURE_DIRECTORY_NAMES + (ENDPOINT_RESULT_NAME,),
        "matching endpoint root",
    )
    endpoint_value = _strict_json_object(
        _read_file(matching_root / ENDPOINT_RESULT_NAME, ENDPOINT_RESULT_NAME),
        ENDPOINT_RESULT_NAME,
    )
    _validate_endpoint_result_boundary(endpoint_value)
    for signature in canonical.signatures:
        signature_name = SIGNATURE_DIRECTORY_NAMES[
            signature.input_signature_id - 1
        ]
        signature_root = matching_root / signature_name
        _require_directory_entries(
            signature_root,
            SIGNATURE_FILE_NAMES + STREAM_DIRECTORY_NAMES,
            f"matching {signature_name}",
        )
        for stream in signature.streams:
            stream_name = STREAM_DIRECTORY_NAMES[int(stream.stream_kind_id) - 1]
            expected_names = tuple(
                Path(relative).name
                for relative in stream.framed_relative_paths
            )
            _require_directory_entries(
                signature_root / stream_name,
                expected_names,
                f"matching {signature_name}/{stream_name}",
            )
    shell = Q1FormalEndpointReplayV1(
        matching_root,
        "unknown",
        "",
        "",
        "",
        "",
        canonical.semantic_binding_root,
        canonical.projection_profile_root,
        canonical.closure_bundle,
        canonical.signatures,
        canonical.archive_payload_bytes,
        canonical.formal_relative_paths,
        b"",
    )
    _require_exact_formal_file_agreement(canonical, shell)
    (
        endpoint_kind,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_receipt_root,
        common_metadata,
    ) = _validate_endpoint_result(endpoint_value, shell)
    return Q1FormalEndpointReplayV1(
        matching_root,
        endpoint_kind,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_receipt_root,
        canonical.semantic_binding_root,
        canonical.projection_profile_root,
        canonical.closure_bundle,
        canonical.signatures,
        canonical.archive_payload_bytes,
        canonical.formal_relative_paths,
        common_metadata,
    )


def dual_formal_host_replay_v1(
    python_root: Path,
    rust_root: Path,
    *,
    expected_action_id: str,
    expected_capsule_digest: str,
    expected_source_commit: str,
    expected_run_genesis_root: str,
    expected_prestart_admission_receipt_root: str,
) -> dict[str, object]:
    """Replay two endpoints and return a target-blind dual agreement receipt."""

    if expected_action_id != ACTION_ID:
        _fail("REJECT_Q1_FORMAL_HOST_INVOCATION", "host replay action differs")
    if (
        type(expected_capsule_digest) is not str
        or _SHA256_ID.fullmatch(expected_capsule_digest) is None
        or type(expected_source_commit) is not str
        or _COMMIT_ID.fullmatch(expected_source_commit) is None
        or type(expected_run_genesis_root) is not str
        or _SHA256_ID.fullmatch(expected_run_genesis_root) is None
        or type(expected_prestart_admission_receipt_root) is not str
        or _SHA256_ID.fullmatch(expected_prestart_admission_receipt_root) is None
    ):
        _fail("REJECT_Q1_FORMAL_HOST_INVOCATION", "expected run identity differs")
    python = replay_endpoint_formal_archive_v1(python_root)
    rust = _replay_exact_matching_endpoint_v1(python, rust_root)
    if python.endpoint_kind != "python" or rust.endpoint_kind != "rust":
        _fail("REJECT_Q1_FORMAL_DUAL_AGREEMENT", "endpoint roles differ")
    if (
        python.capsule_digest != rust.capsule_digest
        or python.source_commit != rust.source_commit
        or python.run_genesis_root != rust.run_genesis_root
        or python.prestart_admission_receipt_root
        != rust.prestart_admission_receipt_root
        or python.capsule_digest != expected_capsule_digest
        or python.source_commit != expected_source_commit
        or python.run_genesis_root != expected_run_genesis_root
        or python.prestart_admission_receipt_root
        != expected_prestart_admission_receipt_root
    ):
        _fail("REJECT_Q1_FORMAL_DUAL_AGREEMENT", "run identity differs")
    if (
        python.semantic_binding_root != rust.semantic_binding_root
        or python.projection_profile_root != rust.projection_profile_root
        or python.closure_bundle_root != rust.closure_bundle_root
        or python.archive_payload_bytes != rust.archive_payload_bytes
        or python.common_metadata_bytes != rust.common_metadata_bytes
    ):
        _fail("REJECT_Q1_FORMAL_DUAL_AGREEMENT", "replayed roots differ")
    signatures = [
        {
            "input_signature_id": left.input_signature_id,
            "universe_root": _root_id(left.universe_root),
            "fixed_point_record_root": _root_id(left.fixed_point.record_root),
            "signature_manifest_root": _root_id(left.signature_manifest_root),
            "saturation_state_root": _root_id(left.saturation_state_root),
            "record_count_by_stream": [
                item.descriptor.record_count for item in left.streams
            ],
        }
        for left in python.signatures
    ]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "actor_id": "host",
        "implementation_id": IMPLEMENTATION_ID,
        "action_id": ACTION_ID,
        "status": RESULT_STATUS,
        "capsule_digest": python.capsule_digest,
        "source_commit": python.source_commit,
        "run_genesis_root": python.run_genesis_root,
        "prestart_admission_receipt_root": (
            python.prestart_admission_receipt_root
        ),
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 0x3FFF,
        "q1_output_slots": [None] * 8,
        "formal_capacity_limits_by_signature": [
            dict(zip(FORMAL_CAPACITY_LIMIT_FIELDS, row, strict=True))
            for row in FORMAL_CAPACITY_LIMIT_ROWS
        ],
        "formal_output_bytes_limit": FORMAL_OUTPUT_BYTES_LIMIT,
        "formal_scratch_bytes_limit": FORMAL_SCRATCH_BYTES_LIMIT,
        "semantic_binding_root": _root_id(python.semantic_binding_root),
        "projection_profile_root": _root_id(python.projection_profile_root),
        "closure_bundle_root": _root_id(python.closure_bundle_root),
        "archive_payload_bytes_per_endpoint": python.archive_payload_bytes,
        "formal_file_count_per_endpoint": len(python.formal_relative_paths),
        "signatures": signatures,
        "python_rust_formal_bytes_exact": True,
        "python_rust_replayed_roots_exact": True,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "target_split_role_absence_attested": True,
        "q1_completion_receipt_issued": False,
        "state_published": False,
    }


__all__ = [
    "ACTION_ID",
    "IMPLEMENTATION_ID",
    "Q1FormalEndpointReplayV1",
    "Q1FormalHostReplayError",
    "Q1FormalSignatureReplayV1",
    "Q1FormalStreamReplayV1",
    "RESULT_SCHEMA_VERSION",
    "RESULT_STATUS",
    "dual_formal_host_replay_v1",
    "replay_endpoint_formal_archive_v1",
]

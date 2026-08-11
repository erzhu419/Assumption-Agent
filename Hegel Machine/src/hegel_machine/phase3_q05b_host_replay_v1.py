"""Trusted-host strict replay for Q0.5b bounded-node3 sidecars.

This module is target-blind and read-only.  It replays the two endpoint
sidecar trees only after the endpoint processes have exited, independently
reconstructs every materialized stream, counting/discard stream, and
external-sort run/scratch trace, and then constructs a qualification-only
shadow assembler commitment.  The shadow object has no numeric tag and is
never a formal Q1 fixed-point record or output root.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Final, Mapping, NoReturn

from . import phase3_q1_archive_projection_v1 as _projection
from . import phase3_q1_external_sort_profile_v1 as _external_sort
from . import phase3_q1_formal_archive_contract_v1 as _formal
from . import phase3_q1_qualification_wire_v1 as _wire
from . import phase3_q1_semantic_coverage_v1 as _coverage
from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


HOST_REPLAY_VERSION: Final = "hegel-q05b-trusted-host-replay-v1.0.0"
SHADOW_ASSEMBLER_SCHEMA_ID: Final = b"hegel-q05b-shadow-partition-bundle/1"
PARTITION_STRICT_REPLAY_SCHEMA_ID: Final = (
    b"hegel-q05b-host-partition-strict-replay/1"
)
SHADOW_ASSEMBLER_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/HOST/SHADOW_PARTITION_BUNDLE/V1"
)
HOST_REPLAY_ROOT_DOMAIN: Final = "HEGEL/Q05B/HOST/STRICT_REPLAY/V1"
HOST_STREAM_REPLAY_ROOT_DOMAIN: Final = "HEGEL/Q05B/HOST/STREAM_REPLAY/V1"
HOST_PREDICATE_EVIDENCE_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/HOST/PREDICATE_EVIDENCE/V1"
)
HOST_RECORD_SET_REPLAY_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/HOST/RECORD_SET_REPLAY/V1"
)
HOST_COVERAGE_REPLAY_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/HOST/COVERAGE_REPLAY/V1"
)
HOST_MATERIALIZED_REPLAY_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/HOST/MATERIALIZED_REPLAY/V1"
)
HOST_COUNTING_REPLAY_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/HOST/COUNTING_REPLAY/V1"
)
HOST_TRACE_REPLAY_ROOT_DOMAIN: Final = "HEGEL/Q05B/HOST/TRACE_REPLAY/V1"
HOST_SCRATCH_REPLAY_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/HOST/SCRATCH_LEDGER_REPLAY/V1"
)
HOST_SEMANTIC_WITNESS_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-host-semantic-replay-witness/1"
)
HOST_SEMANTIC_WITNESS_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/HOST/SEMANTIC_REPLAY_WITNESS/V1\x00"
)
HOST_SEMANTIC_WITNESS_STATUS: Final = (
    "HOST_SEMANTIC_REPLAY_WITNESS_NOT_RECEIPT"
)
SEALED_STDOUT_SET_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-sealed-actor-stdout-set/1"
)
MAXIMUM_ACTOR_STDOUT_BYTES: Final = 1024 * 1024

MAXIMUM_SINGLE_SIDECAR_BYTES: Final = 64 * 1024 * 1024
MAXIMUM_SIDECAR_SET_BYTES: Final = 128 * 1024 * 1024
EXPECTED_OUTPUT_DIRECTORIES: Final = ("neutral", "preimages")
EXPECTED_ROOTS: Final = {
    "leaf": bytes.fromhex(
        "3fefacd3db59294f2b6d44a5d0b813e73af3ec84742a24ab846bbdacae6c1f1b"
    ),
    "odd_partition": bytes.fromhex(
        "99357fc3a5f48e8a63e6a87f4b182153c5cdae52bd911676f7b2ecc1058aa097"
    ),
    "sink_partition": bytes.fromhex(
        "51d017cd9d7e452198d9d12c53e16728c1e220e56d47f43ce3954c4e92c9ef67"
    ),
    "sidecar": bytes.fromhex(
        "1d68a6fe330f3bfe581ef37933f64d2258e1043079dae15c85607836d99ea59d"
    ),
    "golden": bytes.fromhex(
        "cbc22f6a9dc91589f77aa1564eb40d688c45ee3aa6af5a66d777ffe08a086b15"
    ),
}
EXPECTED_RAW_SHA256: Final = (
    bytes.fromhex(
        "0b2b41acce572e05cd2f201f78a5911782b1559ed31c68625eef984bbf4b39de"
    ),
    bytes.fromhex(
        "2d708648b948ac984a7632c06a71d88a6d03388ee00373c6abaf47ef8bff8756"
    ),
    bytes.fromhex(
        "318b8fb9e9ba3ce881057742d59bf43314c89891cbc37e4824349ac3f72d4ba3"
    ),
    bytes.fromhex(
        "7fd529708a068e2fa1a8d17f5cc81a41420db944120f4f1591f73e1c67f4cc05"
    ),
)

FAIL_HOST_TREE = "FAIL_Q05B_HOST_SIDECAR_TREE"
FAIL_HOST_TOCTOU = "FAIL_Q05B_HOST_TOCTOU"
FAIL_HOST_WIRE = "FAIL_Q05B_HOST_WIRE"
FAIL_HOST_STREAM = "FAIL_Q05B_HOST_STREAM_REPLAY"
FAIL_HOST_SHADOW = "FAIL_Q05B_HOST_SHADOW_ASSEMBLER"
FAIL_HOST_DISAGREEMENT = "FAIL_Q05B_HOST_DUAL_DISAGREEMENT"
FAIL_HOST_STDOUT = "FAIL_Q05B_HOST_STDOUT_SET"


class Q05BHostReplayError(RuntimeError):
    """Stable fail-closed trusted-host replay error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q05BHostReplayError(code, detail)


def sealed_actor_stdout_manifest_bytes_v1(
    python_stdout: bytes,
    rust_stdout: bytes,
) -> bytes:
    """Return the exact manifest for two separately captured stdout files."""

    rows: list[list[object]] = []
    for ordinal, actor_id, payload in (
        (1, "PYTHON_ENDPOINT", python_stdout),
        (2, "RUST_ENDPOINT", rust_stdout),
    ):
        if (
            type(payload) is not bytes
            or not payload
            or len(payload) > MAXIMUM_ACTOR_STDOUT_BYTES
        ):
            _fail(FAIL_HOST_STDOUT, f"{actor_id} stdout size/type differs")
        envelope = _wire.validate_actor_stdout_envelope_v1(payload)
        if envelope["actor_id"] != actor_id:
            _fail(FAIL_HOST_STDOUT, f"{actor_id} stdout actor binding differs")
        rows.append([ordinal, actor_id, len(payload), sha256(payload).hexdigest()])
    value = {
        "rows": rows,
        "schema_version": SEALED_STDOUT_SET_SCHEMA_VERSION,
    }
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


def validate_sealed_actor_stdout_set_v1(
    python_stdout: bytes,
    rust_stdout: bytes,
    manifest_payload: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    """Replay two standalone stdout files and their standalone manifest."""

    if type(manifest_payload) is not bytes:
        _fail(FAIL_HOST_STDOUT, "stdout manifest type differs")
    expected = sealed_actor_stdout_manifest_bytes_v1(python_stdout, rust_stdout)
    if manifest_payload != expected:
        _fail(FAIL_HOST_STDOUT, "stdout manifest bytes differ")
    return (
        _wire.validate_actor_stdout_envelope_v1(python_stdout),
        _wire.validate_actor_stdout_envelope_v1(rust_stdout),
    )


def _root32(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        _fail(FAIL_HOST_WIRE, f"{name} must be exactly 32 bytes")
    return value


def _snapshot_stat(value: os.stat_result) -> tuple[int, ...]:
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


def read_frozen_file_v1(root: Path, relative_path: bytes) -> bytes:
    """Read one exact read-only sidecar with no-follow and TOCTOU checks."""

    if not isinstance(root, Path) or type(relative_path) is not bytes:
        raise TypeError("root must be Path and relative_path must be bytes")
    try:
        relative_text = relative_path.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(FAIL_HOST_TREE, f"relative path is not ASCII: {error}")
    if (
        relative_text.startswith("/")
        or ".." in Path(relative_text).parts
        or Path(relative_text).as_posix() != relative_text
    ):
        _fail(FAIL_HOST_TREE, "sidecar relative path is not canonical")
    try:
        root_status = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        _fail(FAIL_HOST_TREE, f"sidecar root is unavailable: {error}")
    if root.is_symlink() or not stat.S_ISDIR(root_status.st_mode) or root != resolved_root:
        _fail(FAIL_HOST_TREE, "sidecar root must be a canonical nonsymlink directory")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    root_descriptor: int | None = None
    parent_descriptor: int | None = None
    descriptor: int | None = None
    try:
        root_descriptor = os.open(root, directory_flags)
        root_before = os.fstat(root_descriptor)
        if (
            root_before.st_dev != root_status.st_dev
            or root_before.st_ino != root_status.st_ino
        ):
            _fail(FAIL_HOST_TOCTOU, "sidecar root changed before anchored open")
        parent_descriptor = os.dup(root_descriptor)
        parts = Path(relative_text).parts
        for component in parts[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        descriptor = os.open(parts[-1], file_flags, dir_fd=parent_descriptor)
    except OSError as error:
        _fail(FAIL_HOST_TREE, f"cannot open frozen sidecar: {error}")
    try:
        assert descriptor is not None
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != _wire.OUTPUT_FILE_MODE
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAXIMUM_SINGLE_SIDECAR_BYTES
        ):
            _fail(FAIL_HOST_TREE, "sidecar type/mode/link/size policy differs")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(FAIL_HOST_TOCTOU, "sidecar truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_HOST_TOCTOU, "sidecar grew during read")
        after = os.fstat(descriptor)
        assert root_descriptor is not None
        root_after = os.fstat(root_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
    if _snapshot_stat(before) != _snapshot_stat(after):
        _fail(FAIL_HOST_TOCTOU, "sidecar identity changed during read")
    if _snapshot_stat(root_before) != _snapshot_stat(root_after):
        _fail(FAIL_HOST_TOCTOU, "sidecar root changed during anchored read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        _fail(FAIL_HOST_TOCTOU, "sidecar length changed during read")
    return payload


def read_exact_sidecar_tree_v1(root: Path) -> tuple[bytes, ...]:
    try:
        entries = tuple(root.rglob("*"))
    except OSError as error:
        _fail(FAIL_HOST_TREE, f"cannot enumerate sidecar tree: {error}")
    entry_snapshots: dict[str, tuple[int, ...]] = {}
    for path in entries:
        try:
            value = path.lstat()
        except OSError as error:
            _fail(FAIL_HOST_TREE, f"cannot stat sidecar tree entry: {error}")
        if path.is_symlink() or not (
            stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)
        ):
            _fail(FAIL_HOST_TREE, "sidecar tree contains a symlink or special file")
        entry_snapshots[path.relative_to(root).as_posix()] = _snapshot_stat(value)
    observed_files = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in entries
            if path.is_file()
        )
    )
    expected_files = tuple(
        sorted(path.decode("ascii") for path in _wire.ORDERED_OUTPUT_RELATIVE_PATHS)
    )
    observed_directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in entries
            if path.is_dir()
        )
    )
    if observed_files != expected_files or observed_directories != EXPECTED_OUTPUT_DIRECTORIES:
        _fail(FAIL_HOST_TREE, "sidecar file/directory set differs")
    payloads = tuple(
        read_frozen_file_v1(root, relative)
        for relative in _wire.ORDERED_OUTPUT_RELATIVE_PATHS
    )
    if sum(len(payload) for payload in payloads) > MAXIMUM_SIDECAR_SET_BYTES:
        _fail(FAIL_HOST_TREE, "sidecar set exceeds aggregate byte guard")
    try:
        entries_after = tuple(root.rglob("*"))
    except OSError as error:
        _fail(FAIL_HOST_TOCTOU, f"cannot re-enumerate sidecar tree: {error}")
    after_snapshots: dict[str, tuple[int, ...]] = {}
    for path in entries_after:
        try:
            value = path.lstat()
        except OSError as error:
            _fail(FAIL_HOST_TOCTOU, f"cannot restat sidecar entry: {error}")
        after_snapshots[path.relative_to(root).as_posix()] = _snapshot_stat(value)
    if entry_snapshots != after_snapshots:
        _fail(FAIL_HOST_TOCTOU, "sidecar tree changed across complete replay read")
    return payloads


def _expected_record_objects_v1(
    evidence: _wire.Q05BNode3PartitionEvidenceV1,
    stream_kind_id: _formal.ArchiveStreamKindId,
) -> tuple[object, ...]:
    if stream_kind_id is _formal.ArchiveStreamKindId.PROGRAM:
        value = evidence.record_set_object[4]
    elif stream_kind_id is _formal.ArchiveStreamKindId.COHORT:
        value = evidence.record_set_object[5]
    elif stream_kind_id is _formal.ArchiveStreamKindId.CLASS:
        value = evidence.record_set_object[6]
    else:
        value = tuple(row[0] for row in evidence.coverage_rows)
    if type(value) is not tuple:
        _fail(FAIL_HOST_STREAM, "partition record preimage is not an exact tuple")
    return value


@dataclass(frozen=True, slots=True)
class PartitionStrictReplayV1:
    input_signature_id: int
    record_set_replay_root: bytes
    coverage_replay_root: bytes
    stream_replay_roots: tuple[bytes, ...]
    materialized_replay_roots: tuple[bytes, ...]
    counting_replay_roots: tuple[bytes, ...]
    trace_replay_roots: tuple[bytes, ...]
    scratch_ledger_roots: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.input_signature_id) is not int or self.input_signature_id not in (1, 2):
            _fail(FAIL_HOST_STREAM, "partition replay signature differs")
        _root32(self.record_set_replay_root, "record-set replay root")
        _root32(self.coverage_replay_root, "coverage replay root")
        for name in (
            "stream_replay_roots",
            "materialized_replay_roots",
            "counting_replay_roots",
            "trace_replay_roots",
            "scratch_ledger_roots",
        ):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != 4:
                _fail(FAIL_HOST_STREAM, f"{name} must contain four roots")
            for index, root in enumerate(value):
                _root32(root, f"{name}[{index}]")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            PARTITION_STRICT_REPLAY_SCHEMA_ID,
            self.input_signature_id,
            self.record_set_replay_root,
            self.coverage_replay_root,
            self.stream_replay_roots,
            self.materialized_replay_roots,
            self.counting_replay_roots,
            self.trace_replay_roots,
            self.scratch_ledger_roots,
        )


def strict_replay_partition_streams_v1(
    evidence: _wire.Q05BNode3PartitionEvidenceV1,
) -> PartitionStrictReplayV1:
    """Independently replay all four stream paths and exposed trace ledgers."""

    roots: list[bytes] = []
    materialized_roots: list[bytes] = []
    counting_roots: list[bytes] = []
    trace_roots: list[bytes] = []
    scratch_roots: list[bytes] = []
    decoded_streams: list[tuple[object, ...]] = []
    for expected_kind, row in zip(
        _formal.ArchiveStreamKindId,
        evidence.stream_rows,
        strict=True,
    ):
        if type(row) is not tuple or len(row) != 5 or row[0] != int(expected_kind):
            _fail(FAIL_HOST_STREAM, "stream row kind/order differs")
        try:
            record_objects = tuple(
                record
                for blob in row[2]
                for record in _formal.replay_framed_records_v1(blob)
            )
            decoded_records = tuple(
                _projection._strict_decode_formal_record_v1(  # noqa: SLF001
                    record,
                    expected_kind,
                )
                for record in record_objects
            )
            materialized = _projection.project_record_stream_v1(
                decoded_records,
                input_signature_id=evidence.input_signature_id,
                universe_root=evidence.universe_root,
                stream_kind_id=expected_kind,
            )
            counting = _projection.counting_discard_record_stream_v1(
                decoded_records,
                input_signature_id=evidence.input_signature_id,
                universe_root=evidence.universe_root,
                stream_kind_id=expected_kind,
            )
            _projection.validate_counting_discard_matches_materialized_v1(
                counting,
                materialized,
            )
            trace = _external_sort.project_external_sort_trace_v1(
                tuple(
                    (
                        _projection._stream_sort_key(  # noqa: SLF001
                            record,
                            expected_kind,
                        ),
                        canonical_cbor_encode(record.canonical_object()),
                    )
                    for record in decoded_records
                ),
                input_signature_id=evidence.input_signature_id,
                stream_kind_id=expected_kind,
            )
        except (TypeError, ValueError, IndexError) as error:
            _fail(
                FAIL_HOST_STREAM,
                f"signature {evidence.input_signature_id} stream {int(expected_kind)}: {error}",
            )
        if (
            record_objects != _expected_record_objects_v1(evidence, expected_kind)
            or row[1] != materialized.canonical_diagnostic_object()
            or row[2] != materialized.chunks.framed_blobs
            or row[3] != trace.canonical_object()
            or row[4] != counting.canonical_object()
            or trace.projection != materialized.external_sort_projection
        ):
            _fail(
                FAIL_HOST_STREAM,
                f"signature {evidence.input_signature_id} stream {int(expected_kind)} differs",
            )
        decoded_streams.append(decoded_records)
        roots.append(
            content_hash(
                HOST_STREAM_REPLAY_ROOT_DOMAIN,
                (
                    evidence.input_signature_id,
                    int(expected_kind),
                    row[1],
                    row[3],
                    row[4],
                ),
            )
        )
        materialized_roots.append(
            content_hash(
                HOST_MATERIALIZED_REPLAY_ROOT_DOMAIN,
                (materialized.canonical_diagnostic_object(), materialized.chunks.framed_blobs),
            )
        )
        counting_roots.append(
            content_hash(
                HOST_COUNTING_REPLAY_ROOT_DOMAIN,
                counting.canonical_object(),
            )
        )
        trace_roots.append(
            content_hash(HOST_TRACE_REPLAY_ROOT_DOMAIN, trace.canonical_object())
        )
        scratch_roots.append(trace.projection.scratch_event_ledger_root)

    try:
        record_set_root = content_hash(
            _projection.SNAPSHOT_RECORD_SET_ROOT_DOMAIN,
            evidence.record_set_object,
        )
        record_set = _projection.Q1SnapshotRecordSetV1(
            evidence.input_signature_id,
            evidence.universe_root,
            decoded_streams[0],
            decoded_streams[1],
            decoded_streams[2],
            record_set_root,
        )
        coverage_records = decoded_streams[3]
        coverage_preimages = tuple(
            _coverage.Q1SemanticCoveragePreimageV1(
                record.construction_depth,
                record.coverage_code,
                row[1],
                row[2],
                row[3],
            )
            for record, row in zip(
                coverage_records,
                evidence.coverage_rows,
                strict=True,
            )
        )
        coverage_archive = _coverage.Q1SemanticCoverageArchiveV1(
            schema_version=_coverage.SEMANTIC_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
            diagnostic_id=_coverage.SEMANTIC_COVERAGE_DIAGNOSTIC_ID,
            input_signature_id=evidence.input_signature_id,
            universe_root=evidence.universe_root,
            coverage_records=coverage_records,
            coverage_preimages=coverage_preimages,
            eligible_application_count=sum(
                row.eligible_application_count for row in coverage_records
            ),
            processed_application_count=sum(
                row.processed_application_count for row in coverage_records
            ),
            strict_admitted_count=sum(
                row.strict_admitted_count for row in coverage_records
            ),
            rewrite_collapse_count=sum(
                row.rewrite_collapse_count for row in coverage_records
            ),
            formal_coverage_archive_root=None,
            q1_state="NOT_RUN",
            q1_gate_count=0,
            q1_gate_mask=0,
            target_truth_accessed=False,
            split_accessed=False,
            role_evaluation_performed=False,
        )
    except (TypeError, ValueError, IndexError) as error:
        _fail(FAIL_HOST_STREAM, f"partition semantic assembly failed: {error}")
    if (
        record_set.canonical_diagnostic_object() != evidence.record_set_object
        or tuple(record.canonical_object() for record in coverage_archive.coverage_records)
        != tuple(row[0] for row in evidence.coverage_rows)
    ):
        _fail(FAIL_HOST_STREAM, "record-set or coverage archive cross-stream replay differs")
    coverage_root = content_hash(
        HOST_COVERAGE_REPLAY_ROOT_DOMAIN,
        (
            tuple(record.canonical_object() for record in coverage_archive.coverage_records),
            tuple(
                (
                    item.construction_depth,
                    item.coverage_code,
                    item.eligible_application_keys,
                    item.processed_application_keys,
                    item.strict_admission_preimages,
                )
                for item in coverage_archive.coverage_preimages
            ),
            coverage_archive.eligible_application_count,
            coverage_archive.processed_application_count,
            coverage_archive.strict_admitted_count,
            coverage_archive.rewrite_collapse_count,
        ),
    )
    return PartitionStrictReplayV1(
        evidence.input_signature_id,
        content_hash(HOST_RECORD_SET_REPLAY_ROOT_DOMAIN, record_set.canonical_diagnostic_object()),
        coverage_root,
        tuple(roots),
        tuple(materialized_roots),
        tuple(counting_roots),
        tuple(trace_roots),
        tuple(scratch_roots),
    )


@dataclass(frozen=True, slots=True)
class ShadowAssemblerEvidenceV1:
    canonical_object: tuple[object, ...]
    root: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_object) is not tuple or len(self.canonical_object) != 15:
            _fail(FAIL_HOST_SHADOW, "shadow assembler object shape differs")
        _root32(self.root, "shadow assembler root")
        if self.canonical_object[:3] != (
            1,
            SHADOW_ASSEMBLER_SCHEMA_ID,
            HOST_REPLAY_VERSION.encode("ascii"),
        ):
            _fail(FAIL_HOST_SHADOW, "shadow assembler header differs")
        if self.canonical_object[-5:] != (False, None, 0, 0, _wire.Q1_NULL_OUTPUT_SLOTS):
            _fail(FAIL_HOST_SHADOW, "shadow assembler authority boundary differs")
        expected = content_hash(SHADOW_ASSEMBLER_ROOT_DOMAIN, self.canonical_object)
        if self.root != expected:
            _fail(FAIL_HOST_SHADOW, "shadow assembler root differs")


@dataclass(frozen=True, slots=True)
class ActorSidecarReplayV1:
    actor_id: str
    implementation_id: str
    stdout_payload: bytes
    payloads: tuple[bytes, ...]
    leaf_manifest: _wire.Q05BFullLeafManifestV1
    partitions: tuple[_wire.Q05BNode3PartitionEvidenceV1, ...]
    sidecar_manifest: _wire.Q05BSidecarManifestV1
    golden_manifest: _wire.Q05BNode3GoldenManifestV1
    partition_replays: tuple[PartitionStrictReplayV1, ...]
    shadow_assembler: ShadowAssemblerEvidenceV1
    host_replay_root: bytes

    def __post_init__(self) -> None:
        if self.actor_id not in ("PYTHON_ENDPOINT", "RUST_ENDPOINT"):
            _fail(FAIL_HOST_WIRE, "actor identity differs")
        if len(self.payloads) != 5 or len(self.partitions) != 2:
            _fail(FAIL_HOST_WIRE, "actor sidecar cardinality differs")
        _root32(self.host_replay_root, "host replay root")


def _shadow_assembler_v1(
    leaf: _wire.Q05BFullLeafManifestV1,
    partitions: tuple[_wire.Q05BNode3PartitionEvidenceV1, ...],
    golden: _wire.Q05BNode3GoldenManifestV1,
    partition_replays: tuple[PartitionStrictReplayV1, ...],
) -> ShadowAssemblerEvidenceV1:
    if tuple(item.input_signature_id for item in partitions) != (1, 2):
        _fail(FAIL_HOST_SHADOW, "shadow partition order differs")
    partition_rows = tuple(
        (
            evidence.input_signature_id,
            evidence.evidence_root,
            golden.bounded_state_rows[index][2],
            rfc6962_root(tuple(row[0] for row in evidence.coverage_rows)),
            replay.record_set_replay_root,
            replay.coverage_replay_root,
            replay.canonical_object(),
        )
        for index, (evidence, replay) in enumerate(
            zip(partitions, partition_replays, strict=True)
        )
    )
    value = (
        1,
        SHADOW_ASSEMBLER_SCHEMA_ID,
        HOST_REPLAY_VERSION.encode("ascii"),
        _wire.qualification_wire_profile_root_v1(),
        golden.q1_semantic_binding_root,
        golden.q1_projection_profile_root,
        leaf.manifest_root,
        golden.sidecar_manifest_root,
        golden.manifest_root,
        partition_rows,
        False,
        None,
        0,
        0,
        _wire.Q1_NULL_OUTPUT_SLOTS,
    )
    return ShadowAssemblerEvidenceV1(
        value,
        content_hash(SHADOW_ASSEMBLER_ROOT_DOMAIN, value),
    )


def replay_actor_sidecars_v1(
    actor_id: str,
    stdout_payload: bytes,
    output_root: Path,
) -> ActorSidecarReplayV1:
    envelope = _wire.validate_actor_stdout_envelope_v1(stdout_payload)
    expected_implementation = dict(_wire.ACTOR_IMPLEMENTATION_ID_REGISTRY).get(actor_id)
    if (
        envelope["actor_id"] != actor_id
        or envelope["implementation_id"] != expected_implementation
    ):
        _fail(FAIL_HOST_WIRE, "actor envelope identity differs")
    payloads = read_exact_sidecar_tree_v1(output_root)
    leaf = _wire.decode_full_v16_leaf_manifest_v1(payloads[0])
    odd = _wire.decode_node3_partition_evidence_v1(payloads[1])
    sink = _wire.decode_node3_partition_evidence_v1(payloads[2])
    sidecar = _wire.replay_sidecar_manifest_v1(payloads[3], payloads[:3])
    golden = _wire.decode_node3_golden_manifest_v1(payloads[4])
    if (
        leaf.manifest_root != EXPECTED_ROOTS["leaf"]
        or odd.evidence_root != EXPECTED_ROOTS["odd_partition"]
        or sink.evidence_root != EXPECTED_ROOTS["sink_partition"]
        or sidecar.manifest_root != EXPECTED_ROOTS["sidecar"]
        or golden.manifest_root != EXPECTED_ROOTS["golden"]
        or tuple(sha256(payload).digest() for payload in payloads[1:])
        != EXPECTED_RAW_SHA256
    ):
        _fail(FAIL_HOST_WIRE, "B0.1 sidecar golden roots or raw hashes differ")
    if (
        envelope["neutral_manifest_relative_path"]
        != _wire.NODE3_GOLDEN_MANIFEST_RELATIVE_PATH.decode("ascii")
        or envelope["neutral_manifest_length"] != len(payloads[4])
        or envelope["neutral_manifest_raw_sha256"] != sha256(payloads[4]).hexdigest()
        or envelope["neutral_manifest_root"] != golden.manifest_root.hex()
        or envelope["sidecar_manifest_relative_path"]
        != _wire.SIDECAR_MANIFEST_RELATIVE_PATH.decode("ascii")
        or envelope["sidecar_manifest_length"] != len(payloads[3])
        or envelope["sidecar_manifest_raw_sha256"] != sha256(payloads[3]).hexdigest()
        or envelope["sidecar_manifest_root"] != sidecar.manifest_root.hex()
    ):
        _fail(FAIL_HOST_WIRE, "actor envelope/sidecar identity differs")
    partitions = (odd, sink)
    partition_replays = tuple(
        strict_replay_partition_streams_v1(evidence) for evidence in partitions
    )
    shadow = _shadow_assembler_v1(leaf, partitions, golden, partition_replays)
    replay_root = content_hash(
        HOST_REPLAY_ROOT_DOMAIN,
        (
            actor_id.encode("ascii"),
            tuple(sha256(payload).digest() for payload in payloads),
            tuple(item.canonical_object() for item in partition_replays),
            shadow.root,
            golden.canonical_object()[-1],
        ),
    )
    return ActorSidecarReplayV1(
        actor_id,
        expected_implementation,
        stdout_payload,
        payloads,
        leaf,
        partitions,
        sidecar,
        golden,
        partition_replays,
        shadow,
        replay_root,
    )


@dataclass(frozen=True, slots=True)
class DualHostReplayV1:
    python: ActorSidecarReplayV1
    rust: ActorSidecarReplayV1
    neutral_manifest_bytes: bytes
    host_neutral_raw_sha256: bytes
    stdout_manifest_raw_sha256: bytes
    host_source_identity_root: bytes
    host_runtime_identity_root: bytes
    shadow_assembler_root: bytes
    predicate_evidence_rows: tuple[tuple[int, bytes], ...]
    predicate11_semantic_component_root: bytes
    pending_predicate_ids: tuple[int, ...]
    dual_replay_root: bytes

    def __post_init__(self) -> None:
        if (self.python.actor_id, self.rust.actor_id) != (
            "PYTHON_ENDPOINT",
            "RUST_ENDPOINT",
        ):
            _fail(FAIL_HOST_DISAGREEMENT, "dual actor order differs")
        _root32(self.host_neutral_raw_sha256, "host neutral raw SHA-256")
        _root32(self.stdout_manifest_raw_sha256, "stdout manifest raw SHA-256")
        _root32(self.host_source_identity_root, "host source identity root")
        _root32(self.host_runtime_identity_root, "host runtime identity root")
        _root32(self.shadow_assembler_root, "shadow assembler root")
        _root32(
            self.predicate11_semantic_component_root,
            "predicate-11 semantic component root",
        )
        _root32(self.dual_replay_root, "dual replay root")
        if tuple(row[0] for row in self.predicate_evidence_rows) != (
            6,
            7,
            8,
            12,
            14,
            15,
            17,
        ):
            _fail(FAIL_HOST_DISAGREEMENT, "host predicate evidence registry differs")
        if self.pending_predicate_ids != (11, 13, 16, 18, 19):
            _fail(FAIL_HOST_DISAGREEMENT, "pending host evidence registry differs")


def _canonical_json_bytes_v1(value: object) -> bytes:
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
    except (TypeError, ValueError, UnicodeError) as error:
        _fail(FAIL_HOST_WIRE, f"host semantic witness is not canonical JSON: {error}")


def _strict_json_object_v1(payload: bytes, name: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload.endswith(b"\n"):
        _fail(FAIL_HOST_WIRE, f"{name} must be one JSON object plus LF")

    def reject_constant(value: str) -> NoReturn:
        _fail(FAIL_HOST_WIRE, f"{name} contains a non-finite number: {value}")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if type(key) is not str or key in value:
                _fail(FAIL_HOST_WIRE, f"{name} contains a duplicate/non-string key")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
            parse_float=lambda _value: _fail(
                FAIL_HOST_WIRE, f"{name} contains a floating-point number"
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(FAIL_HOST_WIRE, f"{name} JSON decode failed: {error}")
    if type(value) is not dict or _canonical_json_bytes_v1(value) != payload:
        _fail(FAIL_HOST_WIRE, f"{name} is not exact canonical JSON")
    return value


def _negative_vector_binding_v1(
    corpus_cbor: bytes,
    corpus_root: bytes,
    category_roots: tuple[tuple[int, bytes], ...],
) -> dict[str, object]:
    if (
        type(corpus_cbor) is not bytes
        or not corpus_cbor
        or type(corpus_root) is not bytes
        or len(corpus_root) != 32
        or type(category_roots) is not tuple
        or tuple(row[0] for row in category_roots) != (13, 18)
        or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not int
            or type(row[1]) is not bytes
            or len(row[1]) != 32
            for row in category_roots
        )
    ):
        _fail(FAIL_HOST_WIRE, "negative vector witness binding types differ")
    try:
        value = canonical_cbor_decode(corpus_cbor)
    except Exception as error:
        _fail(FAIL_HOST_WIRE, f"negative vector corpus CBOR differs: {error}")
    closed_authority = (
        b"q1_state",
        b"NOT_RUN",
        b"q1_gate_count",
        0,
        b"q1_gate_mask",
        0,
        b"q1_output_slots",
        (None,) * 8,
        b"certificate_active",
        False,
    )
    if (
        type(value) is not tuple
        or len(value) != 5
        or value[0:2] != (1, b"hegel-q05b-negative-vector-corpus/1")
        or type(value[2]) is not tuple
        or len(value[2]) < 10
        or value[3] != category_roots
        or value[4] != closed_authority
        or canonical_cbor_encode(value) != corpus_cbor
    ):
        _fail(FAIL_HOST_WIRE, "negative vector corpus wire differs")
    rows = value[2]
    for row in rows:
        if (
            type(row) is not tuple
            or len(row) != 5
            or type(row[0]) is not bytes
            or not row[0]
            or type(row[1]) is not int
            or row[1] not in (13, 18)
            or type(row[2]) is not bytes
            or type(row[3]) is not bytes
            or row[2] != row[3]
            or type(row[4]) is not bytes
            or len(row[4]) != 32
        ):
            _fail(FAIL_HOST_WIRE, "negative vector corpus row differs")
    expected_categories = tuple(
        (
            category,
            content_hash(
                "HEGEL/Q05B/QUALIFICATION/NEGATIVE_VECTOR_CATEGORY/V1",
                (category, tuple(row for row in rows if row[1] == category)),
            ),
        )
        for category in (13, 18)
    )
    expected_corpus_root = content_hash(
        "HEGEL/Q05B/QUALIFICATION/NEGATIVE_VECTOR_CORPUS/V1",
        value,
    )
    if category_roots != expected_categories or corpus_root != expected_corpus_root:
        _fail(FAIL_HOST_WIRE, "negative vector corpus/category root differs")
    return {
        "negative_vector_category_root_rows": [
            [category, root.hex()] for category, root in category_roots
        ],
        "negative_vector_corpus_cbor_hex": corpus_cbor.hex(),
        "negative_vector_corpus_length": len(corpus_cbor),
        "negative_vector_corpus_root": corpus_root.hex(),
        "negative_vector_corpus_sha256": sha256(corpus_cbor).hexdigest(),
    }


def _semantic_witness_body_v1(
    replay: DualHostReplayV1,
    negative_corpus_cbor: bytes,
    negative_corpus_root: bytes,
    negative_category_roots: tuple[tuple[int, bytes], ...],
) -> dict[str, object]:
    if type(replay) is not DualHostReplayV1:
        _fail(FAIL_HOST_WIRE, "semantic witness input is not a dual replay")
    negative = _negative_vector_binding_v1(
        negative_corpus_cbor,
        negative_corpus_root,
        negative_category_roots,
    )
    base_rows = dict(replay.predicate_evidence_rows)
    base_rows[13] = negative_category_roots[0][1]
    base_rows[18] = negative_category_roots[1][1]
    python_scratch = tuple(
        partition.scratch_ledger_roots
        for partition in replay.python.partition_replays
    )
    rust_scratch = tuple(
        partition.scratch_ledger_roots
        for partition in replay.rust.partition_replays
    )
    if (
        len(python_scratch) != 2
        or python_scratch != rust_scratch
        or any(len(partition) != 4 for partition in python_scratch)
    ):
        _fail(FAIL_HOST_DISAGREEMENT, "host scratch ledger replays differ")
    host_scratch_evidence_root = content_hash(
        HOST_SCRATCH_REPLAY_ROOT_DOMAIN,
        (
            b"TRUSTED_HOST_REPLAY",
            python_scratch,
            replay.shadow_assembler_root,
            replay.dual_replay_root,
        ),
    )
    base_rows[16] = host_scratch_evidence_root
    value = {
        "host_scratch_evidence_root": host_scratch_evidence_root.hex(),
        "host_scratch_partition_roots": [
            [root.hex() for root in partition]
            for partition in python_scratch
        ],
        "host_runtime_identity_root": replay.host_runtime_identity_root.hex(),
        "host_source_identity_root": replay.host_source_identity_root.hex(),
        "neutral_manifest_sha256": replay.host_neutral_raw_sha256.hex(),
        "pending_predicate_ids": [11, 19],
        "predicate11_semantic_component_root": (
            replay.predicate11_semantic_component_root.hex()
        ),
        "predicate_evidence_rows": [
            [predicate_id, evidence_root.hex()]
            for predicate_id, evidence_root in sorted(base_rows.items())
        ],
        "q1_authority": {
            "certificate_active": False,
            "formal_output_roots": [None] * 8,
            "gate_count": 0,
            "gate_mask": 0,
            "q2_state": "NOT_RUN",
            "state": "NOT_RUN",
        },
        "schema_version": HOST_SEMANTIC_WITNESS_SCHEMA_VERSION,
        "semantic_replay_root": replay.dual_replay_root.hex(),
        "shadow_assembler_root": replay.shadow_assembler_root.hex(),
        "status": HOST_SEMANTIC_WITNESS_STATUS,
        "stdout_manifest_sha256": replay.stdout_manifest_raw_sha256.hex(),
    }
    value.update(negative)
    return value


def host_semantic_witness_bytes_v1(
    replay: DualHostReplayV1,
    negative_corpus_cbor: bytes,
    negative_corpus_root: bytes,
    negative_category_roots: tuple[tuple[int, bytes], ...],
) -> bytes:
    """Encode the host-child semantic witness without future isolation state.

    This object is deliberately not a qualification receipt.  In particular,
    it contains neither the trusted-host container's final resource transcript
    nor a Predicate-19 isolation claim.  The outer supervisor may bind those
    only after the held host container has exited and its final inspect has
    been replayed.
    """

    body = _semantic_witness_body_v1(
        replay,
        negative_corpus_cbor,
        negative_corpus_root,
        negative_category_roots,
    )
    body_bytes = _canonical_json_bytes_v1(body)
    value = dict(body)
    value["witness_root"] = sha256(
        HOST_SEMANTIC_WITNESS_ROOT_DOMAIN + body_bytes
    ).hexdigest()
    return _canonical_json_bytes_v1(value)


def decode_host_semantic_witness_v1(
    payload: bytes,
    expected_replay: DualHostReplayV1 | None = None,
    expected_negative_corpus_cbor: bytes | None = None,
    expected_negative_corpus_root: bytes | None = None,
    expected_negative_category_roots: tuple[tuple[int, bytes], ...] | None = None,
) -> dict[str, object]:
    """Strictly replay a child witness and optionally require byte equality."""

    value = _strict_json_object_v1(payload, "host semantic witness")
    expected_binding_complete = (
        expected_replay is not None
        and expected_negative_corpus_cbor is not None
        and expected_negative_corpus_root is not None
        and expected_negative_category_roots is not None
    )
    if any(
        item is not None
        for item in (
            expected_replay,
            expected_negative_corpus_cbor,
            expected_negative_corpus_root,
            expected_negative_category_roots,
        )
    ) and not expected_binding_complete:
        _fail(FAIL_HOST_WIRE, "expected host witness binding is incomplete")
    expected_keys = set(
        _semantic_witness_body_v1(
            expected_replay,
            expected_negative_corpus_cbor,
            expected_negative_corpus_root,
            expected_negative_category_roots,
        )
    ) | {"witness_root"} if expected_binding_complete else {
            "host_scratch_evidence_root",
            "host_scratch_partition_roots",
            "host_runtime_identity_root",
            "host_source_identity_root",
            "neutral_manifest_sha256",
            "negative_vector_category_root_rows",
            "negative_vector_corpus_cbor_hex",
            "negative_vector_corpus_length",
            "negative_vector_corpus_root",
            "negative_vector_corpus_sha256",
            "pending_predicate_ids",
            "predicate11_semantic_component_root",
            "predicate_evidence_rows",
            "q1_authority",
            "schema_version",
            "semantic_replay_root",
            "shadow_assembler_root",
            "status",
            "stdout_manifest_sha256",
            "witness_root",
        }
    if set(value) != expected_keys:
        _fail(FAIL_HOST_WIRE, "host semantic witness field registry differs")
    root_fields = (
        "host_scratch_evidence_root",
        "host_runtime_identity_root",
        "host_source_identity_root",
        "neutral_manifest_sha256",
        "negative_vector_corpus_root",
        "negative_vector_corpus_sha256",
        "predicate11_semantic_component_root",
        "semantic_replay_root",
        "shadow_assembler_root",
        "stdout_manifest_sha256",
        "witness_root",
    )
    if any(
        type(value.get(field)) is not str
        or len(value[field]) != 64
        or any(character not in "0123456789abcdef" for character in value[field])
        for field in root_fields
    ):
        _fail(FAIL_HOST_WIRE, "host semantic witness root encoding differs")
    if (
        value.get("schema_version") != HOST_SEMANTIC_WITNESS_SCHEMA_VERSION
        or value.get("status") != HOST_SEMANTIC_WITNESS_STATUS
        or value.get("pending_predicate_ids") != [11, 19]
        or value.get("q1_authority")
        != {
            "certificate_active": False,
            "formal_output_roots": [None] * 8,
            "gate_count": 0,
            "gate_mask": 0,
            "q2_state": "NOT_RUN",
            "state": "NOT_RUN",
        }
    ):
        _fail(FAIL_HOST_WIRE, "host semantic witness closed authority differs")
    rows = value.get("predicate_evidence_rows")
    expected_ids = (6, 7, 8, 12, 13, 14, 15, 16, 17, 18)
    if type(rows) is not list or len(rows) != len(expected_ids):
        _fail(FAIL_HOST_WIRE, "host semantic predicate rows differ")
    for expected_id, row in zip(expected_ids, rows, strict=True):
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not int
            or row[0] != expected_id
            or type(row[1]) is not str
            or len(row[1]) != 64
            or any(character not in "0123456789abcdef" for character in row[1])
        ):
            _fail(FAIL_HOST_WIRE, "host semantic predicate row differs")
    scratch_rows = value.get("host_scratch_partition_roots")
    if (
        type(scratch_rows) is not list
        or len(scratch_rows) != 2
        or any(
            type(partition) is not list or len(partition) != 4
            for partition in scratch_rows
        )
    ):
        _fail(FAIL_HOST_WIRE, "host scratch partition root registry differs")
    if any(
        type(root) is not str
        or len(root) != 64
        or any(character not in "0123456789abcdef" for character in root)
        for partition in scratch_rows
        for root in partition
    ):
        _fail(FAIL_HOST_WIRE, "host scratch root encoding differs")
    try:
        scratch_roots = tuple(
            tuple(bytes.fromhex(root) for root in partition)
            for partition in scratch_rows
        )
        shadow_root = bytes.fromhex(value["shadow_assembler_root"])
        semantic_root = bytes.fromhex(value["semantic_replay_root"])
    except (TypeError, ValueError):
        _fail(FAIL_HOST_WIRE, "host scratch root encoding differs")
    expected_scratch_evidence = content_hash(
        HOST_SCRATCH_REPLAY_ROOT_DOMAIN,
        (
            b"TRUSTED_HOST_REPLAY",
            scratch_roots,
            shadow_root,
            semantic_root,
        ),
    ).hex()
    predicate16 = next((row[1] for row in rows if row[0] == 16), None)
    if (
        value.get("host_scratch_evidence_root") != expected_scratch_evidence
        or predicate16 != expected_scratch_evidence
    ):
        _fail(FAIL_HOST_WIRE, "host scratch evidence root differs")
    negative_hex = value.get("negative_vector_corpus_cbor_hex")
    negative_rows = value.get("negative_vector_category_root_rows")
    if (
        type(negative_hex) is not str
        or len(negative_hex) % 2
        or any(character not in "0123456789abcdef" for character in negative_hex)
        or type(negative_rows) is not list
        or len(negative_rows) != 2
    ):
        _fail(FAIL_HOST_WIRE, "negative vector witness JSON encoding differs")
    try:
        negative_cbor = bytes.fromhex(negative_hex)
        category_roots = tuple(
            (row[0], bytes.fromhex(row[1]))
            for row in negative_rows
            if type(row) is list
            and len(row) == 2
            and type(row[0]) is int
            and type(row[1]) is str
        )
        corpus_root = bytes.fromhex(value["negative_vector_corpus_root"])
    except (ValueError, TypeError):
        _fail(FAIL_HOST_WIRE, "negative vector witness root decoding differs")
    negative_binding = _negative_vector_binding_v1(
        negative_cbor,
        corpus_root,
        category_roots,
    )
    if any(value.get(key) != item for key, item in negative_binding.items()):
        _fail(FAIL_HOST_WIRE, "negative vector witness binding differs")
    body = dict(value)
    observed_root = body.pop("witness_root")
    expected_root = sha256(
        HOST_SEMANTIC_WITNESS_ROOT_DOMAIN + _canonical_json_bytes_v1(body)
    ).hexdigest()
    if observed_root != expected_root:
        _fail(FAIL_HOST_WIRE, "host semantic witness root differs")
    if expected_binding_complete and payload != host_semantic_witness_bytes_v1(
        expected_replay,
        expected_negative_corpus_cbor,
        expected_negative_corpus_root,
        expected_negative_category_roots,
    ):
        _fail(FAIL_HOST_DISAGREEMENT, "host child/outer semantic witness differs")
    return value


def require_neutral_byte_agreement_v1(
    python_payloads: tuple[bytes, ...],
    rust_payloads: tuple[bytes, ...],
) -> bytes:
    if (
        type(python_payloads) is not tuple
        or type(rust_payloads) is not tuple
        or len(python_payloads) != 5
        or len(rust_payloads) != 5
        or python_payloads != rust_payloads
    ):
        _fail(
            FAIL_HOST_DISAGREEMENT,
            "Python and Rust five-sidecar canonical bytes differ",
        )
    return python_payloads[4]


def dual_actor_host_replay_v1(
    python_stdout: bytes,
    python_output_root: Path,
    rust_stdout: bytes,
    rust_output_root: Path,
    stdout_manifest: bytes,
    host_source_identity_root: bytes,
    host_runtime_identity_root: bytes,
) -> DualHostReplayV1:
    _root32(host_source_identity_root, "host source identity root")
    _root32(host_runtime_identity_root, "host runtime identity root")
    validate_sealed_actor_stdout_set_v1(
        python_stdout,
        rust_stdout,
        stdout_manifest,
    )
    python = replay_actor_sidecars_v1(
        "PYTHON_ENDPOINT",
        python_stdout,
        python_output_root,
    )
    rust = replay_actor_sidecars_v1(
        "RUST_ENDPOINT",
        rust_stdout,
        rust_output_root,
    )
    neutral = require_neutral_byte_agreement_v1(python.payloads, rust.payloads)
    if (
        python.shadow_assembler.root != rust.shadow_assembler.root
        or python.golden_manifest.canonical_bytes != neutral
        or rust.golden_manifest.canonical_bytes != neutral
    ):
        _fail(FAIL_HOST_DISAGREEMENT, "endpoint/host neutral replay differs")
    python_envelope = _wire.validate_actor_stdout_envelope_v1(python.stdout_payload)
    rust_envelope = _wire.validate_actor_stdout_envelope_v1(rust.stdout_payload)
    file_identity_rows = tuple(
        (
            index,
            _wire.ORDERED_OUTPUT_RELATIVE_PATHS[index],
            len(payload),
            sha256(payload).digest(),
        )
        for index, payload in enumerate(python.payloads)
    )
    materialized_roots = tuple(
        replay.materialized_replay_roots for replay in python.partition_replays
    )
    counting_roots = tuple(
        replay.counting_replay_roots for replay in python.partition_replays
    )
    trace_roots = tuple(
        replay.trace_replay_roots for replay in python.partition_replays
    )
    predicate_preimages = {
        6: (
            b"NEUTRAL_GOLDEN_MANIFEST_PYTHON_RUST_HOST_BYTE_EQUAL",
            sha256(python.payloads[4]).digest(),
            sha256(rust.payloads[4]).digest(),
            sha256(neutral).digest(),
        ),
        7: (
            b"SIDECAR_MANIFEST_PYTHON_RUST_HOST_BYTE_EQUAL",
            sha256(python.payloads[3]).digest(),
            sha256(rust.payloads[3]).digest(),
            python.sidecar_manifest.manifest_root,
            rust.sidecar_manifest.manifest_root,
        ),
        8: (
            b"SIDECAR_RAW_SHA_LENGTH_CONTENT_ROOT_REPLAY",
            file_identity_rows,
            python.sidecar_manifest.canonical_object(),
            sha256(python_stdout).digest(),
            sha256(rust_stdout).digest(),
            sha256(stdout_manifest).digest(),
        ),
        11: (
            b"TRUSTED_HOST_READ_ONLY_REPLAY_QUALIFIED",
            python.host_replay_root,
            rust.host_replay_root,
            sha256(neutral).digest(),
            sha256(stdout_manifest).digest(),
            host_source_identity_root,
            host_runtime_identity_root,
        ),
        12: (
            b"STRICT_PARTITION_MANIFEST_BUNDLE_ASSEMBLER_REPLAY",
            python.shadow_assembler.canonical_object,
            python.shadow_assembler.root,
            tuple(item.record_set_replay_root for item in python.partition_replays),
            tuple(item.coverage_replay_root for item in python.partition_replays),
        ),
        14: (
            b"COUNTING_DISCARD_AND_MATERIALIZED_ENCODER_EQUAL",
            materialized_roots,
            counting_roots,
        ),
        15: (
            b"EXTERNAL_SORT_RUN_AND_MERGE_REPLAY_PASS",
            trace_roots,
            tuple(
                replay.scratch_ledger_roots for replay in python.partition_replays
            ),
        ),
        17: (
            b"OUTPUT_AND_METADATA_FORMULA_REPLAY_PASS",
            file_identity_rows,
            python_envelope["neutral_manifest_length"],
            python_envelope["sidecar_manifest_length"],
            rust_envelope["neutral_manifest_length"],
            rust_envelope["sidecar_manifest_length"],
            python.golden_manifest.canonical_object()[-1],
        ),
    }
    predicate11_semantic_component_root = content_hash(
        HOST_PREDICATE_EVIDENCE_ROOT_DOMAIN,
        (11, predicate_preimages[11]),
    )
    predicate_rows = tuple(
        (
            predicate_id,
            content_hash(
                HOST_PREDICATE_EVIDENCE_ROOT_DOMAIN,
                (predicate_id, predicate_preimages[predicate_id]),
            ),
        )
        for predicate_id in (6, 7, 8, 12, 14, 15, 17)
    )
    dual_root = content_hash(
        HOST_REPLAY_ROOT_DOMAIN,
        (
            python.host_replay_root,
            rust.host_replay_root,
            sha256(neutral).digest(),
            sha256(stdout_manifest).digest(),
            host_source_identity_root,
            host_runtime_identity_root,
            python.shadow_assembler.root,
            predicate_rows,
            predicate11_semantic_component_root,
            (11, 13, 16, 18, 19),
            python.golden_manifest.canonical_object()[-1],
        ),
    )
    return DualHostReplayV1(
        python,
        rust,
        neutral,
        sha256(neutral).digest(),
        sha256(stdout_manifest).digest(),
        host_source_identity_root,
        host_runtime_identity_root,
        python.shadow_assembler.root,
        predicate_rows,
        predicate11_semantic_component_root,
        (11, 13, 16, 18, 19),
        dual_root,
    )


__all__ = [
    "ActorSidecarReplayV1",
    "DualHostReplayV1",
    "EXPECTED_OUTPUT_DIRECTORIES",
    "EXPECTED_RAW_SHA256",
    "EXPECTED_ROOTS",
    "FAIL_HOST_DISAGREEMENT",
    "FAIL_HOST_SHADOW",
    "FAIL_HOST_STREAM",
    "FAIL_HOST_STDOUT",
    "FAIL_HOST_TOCTOU",
    "FAIL_HOST_TREE",
    "FAIL_HOST_WIRE",
    "HOST_REPLAY_VERSION",
    "HOST_SCRATCH_REPLAY_ROOT_DOMAIN",
    "HOST_SEMANTIC_WITNESS_SCHEMA_VERSION",
    "HOST_SEMANTIC_WITNESS_STATUS",
    "SEALED_STDOUT_SET_SCHEMA_VERSION",
    "Q05BHostReplayError",
    "SHADOW_ASSEMBLER_ROOT_DOMAIN",
    "SHADOW_ASSEMBLER_SCHEMA_ID",
    "ShadowAssemblerEvidenceV1",
    "decode_host_semantic_witness_v1",
    "dual_actor_host_replay_v1",
    "host_semantic_witness_bytes_v1",
    "read_exact_sidecar_tree_v1",
    "read_frozen_file_v1",
    "replay_actor_sidecars_v1",
    "require_neutral_byte_agreement_v1",
    "sealed_actor_stdout_manifest_bytes_v1",
    "strict_replay_partition_streams_v1",
    "validate_sealed_actor_stdout_set_v1",
]

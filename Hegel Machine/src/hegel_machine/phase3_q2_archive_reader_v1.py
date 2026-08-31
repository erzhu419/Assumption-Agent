"""Read the sealed Q1 class streams for a future Q2 membership campaign.

The reader validates one existing Q1 endpoint tree in place, selects only the
class stream, and returns typed Q1 class records.  It never copies the Q1 actor
output and it has no target, split, matcher, or result-publication input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from . import phase3_q1_archive_projection_v1 as _projection
from . import phase3_q1_formal_archive_contract_v1 as _formal
from . import phase3_q1_formal_host_replay_v1 as _host_replay
from .phase3_q2_admission_v1 import Q1CompletionAdmissionV1
from .strict_cbor_v1 import rfc6962_root


class Q2ArchiveReaderError(ValueError):
    """The selected sealed Q1 class stream did not replay exactly."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q2ArchiveReaderError(code, detail)


@dataclass(frozen=True, slots=True)
class Q1ClassStreamV1:
    endpoint_root: Path
    endpoint_kind: str
    input_signature_id: int
    universe_root: bytes
    signature_manifest_root: bytes
    saturation_state_root: bytes
    class_archive_root: bytes
    records: tuple[_formal.Q1QuotientClassRecordV1, ...]
    framed_relative_paths: tuple[str, ...]
    framed_bytes: int


def decode_q1_class_records_v1(
    framed_payload: bytes,
    *,
    input_signature_id: int,
    universe_root: bytes,
) -> tuple[_formal.Q1QuotientClassRecordV1, ...]:
    """Decode one canonical class-stream payload through the frozen Q1 decoder."""

    if input_signature_id not in (1, 2):
        _fail("REJECT_Q2_CLASS_STREAM", "input signature is not registered")
    if type(universe_root) is not bytes or len(universe_root) != 32:
        _fail("REJECT_Q2_CLASS_STREAM", "universe root must be 32 bytes")
    try:
        objects = _formal.replay_framed_records_v1(framed_payload)
        # Reuse the exact frozen Q1 typed decoder instead of creating a second
        # authority for the Q1 archive wire in Q2.
        records = tuple(
            _projection._strict_decode_formal_record_v1(
                row,
                _formal.ArchiveStreamKindId.CLASS,
            )
            for row in objects
        )
        _formal.canonical_archive_order_v1(
            records,
            stream_kind_id=_formal.ArchiveStreamKindId.CLASS,
        )
    except ValueError as error:
        _fail("REJECT_Q2_CLASS_STREAM", str(error))
    if not records or any(
        type(record) is not _formal.Q1QuotientClassRecordV1
        or record.input_signature_id != input_signature_id
        or record.universe_root != universe_root
        for record in records
    ):
        _fail("REJECT_Q2_CLASS_STREAM", "class record binding differs")
    return records


def read_q1_class_stream_v1(
    endpoint_root: Path,
    admission: Q1CompletionAdmissionV1,
    *,
    input_signature_id: int,
    expected_endpoint_kind: str = "python",
) -> Q1ClassStreamV1:
    """Validate one sealed endpoint tree and return one selected class stream."""

    if input_signature_id not in (1, 2):
        _fail("REJECT_Q2_CLASS_STREAM", "input signature is not registered")
    return read_q1_class_streams_v1(
        endpoint_root,
        admission,
        expected_endpoint_kind=expected_endpoint_kind,
    )[input_signature_id - 1]


def _read_signature_class_stream_v1(
    endpoint_root: Path,
    endpoint_kind: str,
    admission: Q1CompletionAdmissionV1,
    input_signature_id: int,
) -> tuple[Q1ClassStreamV1, _formal.Q1SignatureArchiveManifestV1]:
    signature_name = f"signature-{input_signature_id:02d}"
    signature_root = endpoint_root / signature_name
    try:
        manifest = _host_replay._decode_signature_manifest(
            _host_replay._single_framed_object(
                signature_root / "signature-manifest.framed",
                f"{signature_name}/signature-manifest.framed",
            )
        )
        manifest_payload = (signature_root / "chunk-manifests.framed").read_bytes()
        manifest_objects = _formal.replay_framed_records_v1(manifest_payload)
        manifests = tuple(
            _host_replay._decode_chunk_manifest(row) for row in manifest_objects
        )
    except (OSError, ValueError) as error:
        _fail("REJECT_Q2_Q1_ARCHIVE", str(error))
    if (
        manifest.input_signature_id != input_signature_id
        or manifest.chunk_manifest_count != len(manifests)
        or manifest.chunk_manifest_archive_root != rfc6962_root(manifest_objects)
        or any(
            row.input_signature_id != input_signature_id
            or row.universe_root != manifest.universe_root
            for row in manifests
        )
    ):
        _fail("REJECT_Q2_Q1_ARCHIVE", "Q1 signature manifest closure differs")
    manifest_slot = 0 if input_signature_id == 1 else 2
    saturation_slot = manifest_slot + 1
    if (
        manifest.manifest_root
        != admission.ordered_output_slots[manifest_slot][2]
        or manifest.saturation_state_root
        != admission.ordered_output_slots[saturation_slot][2]
    ):
        _fail("REJECT_Q2_Q1_ARCHIVE", "Q1 receipt/archive roots differ")

    class_descriptor = next(
        row
        for row in manifest.stream_descriptors
        if row.stream_kind_id is _formal.ArchiveStreamKindId.CLASS
    )
    class_manifests = tuple(
        row
        for row in manifests
        if row.stream_kind_id is _formal.ArchiveStreamKindId.CLASS
    )
    relative_paths = tuple(
        f"{signature_name}/stream-03/chunk-{row.chunk_index:06d}.framed"
        for row in class_manifests
    )
    try:
        framed_blobs = tuple(
            (endpoint_root / relative_path).read_bytes()
            for relative_path in relative_paths
        )
        _projection.Q1ChunkProjectionV1(class_manifests, framed_blobs)
        records = tuple(
            record
            for blob in framed_blobs
            for record in decode_q1_class_records_v1(
                blob,
                input_signature_id=input_signature_id,
                universe_root=manifest.universe_root,
            )
        )
        _formal.canonical_archive_order_v1(
            records,
            stream_kind_id=_formal.ArchiveStreamKindId.CLASS,
        )
    except (OSError, ValueError) as error:
        _fail("REJECT_Q2_CLASS_STREAM", str(error))
    if (
        len(records) != class_descriptor.record_count
        or _formal.archive_root_v1(tuple(row.canonical_object() for row in records))
        != class_descriptor.archive_root
        or sum(len(blob) for blob in framed_blobs)
        != class_descriptor.framed_stream_bytes
        or len(class_manifests) != class_descriptor.chunk_count
        or rfc6962_root(tuple(row.canonical_object() for row in class_manifests))
        != class_descriptor.chunk_manifest_subtree_root
    ):
        _fail("REJECT_Q2_CLASS_STREAM", "class stream descriptor differs")
    return (
        Q1ClassStreamV1(
            endpoint_root,
            endpoint_kind,
            input_signature_id,
            manifest.universe_root,
            manifest.manifest_root,
            manifest.saturation_state_root,
            class_descriptor.archive_root,
            records,
            relative_paths,
            sum(len(blob) for blob in framed_blobs),
        ),
        manifest,
    )


def read_q1_class_streams_v1(
    endpoint_root: Path,
    admission: Q1CompletionAdmissionV1,
    *,
    expected_endpoint_kind: str = "python",
) -> tuple[Q1ClassStreamV1, Q1ClassStreamV1]:
    """Replay one endpoint once and return its Odd and Sink class streams."""

    if type(admission) is not Q1CompletionAdmissionV1:
        raise TypeError("admission must be Q1CompletionAdmissionV1")
    if expected_endpoint_kind not in ("python", "rust"):
        _fail("REJECT_Q2_CLASS_STREAM", "endpoint kind is not registered")
    try:
        if not isinstance(endpoint_root, Path):
            raise TypeError("endpoint root must be pathlib.Path")
        endpoint_root = endpoint_root.resolve(strict=True)
        endpoint_result = _host_replay._strict_json_object(
            (endpoint_root / "endpoint-result.json").read_bytes(),
            "endpoint-result.json",
        )
    except (OSError, ValueError) as error:
        _fail("REJECT_Q2_Q1_ARCHIVE", str(error))
    if endpoint_result.get("actor_id") != expected_endpoint_kind:
        _fail("REJECT_Q2_Q1_ARCHIVE", "Q1 endpoint kind differs")
    try:
        closure = _host_replay._decode_closure_bundle(
            _host_replay._single_framed_object(
                endpoint_root / "closure-bundle.framed",
                "closure-bundle.framed",
            )
        )
    except (OSError, ValueError) as error:
        _fail("REJECT_Q2_Q1_ARCHIVE", str(error))
    if closure.bundle_root != admission.ordered_output_slots[4][2]:
        _fail("REJECT_Q2_Q1_ARCHIVE", "Q1 receipt/archive roots differ")
    odd, odd_manifest = _read_signature_class_stream_v1(
        endpoint_root,
        expected_endpoint_kind,
        admission,
        1,
    )
    sink, sink_manifest = _read_signature_class_stream_v1(
        endpoint_root,
        expected_endpoint_kind,
        admission,
        2,
    )
    expected_rows = (
        (
            odd.input_signature_id,
            odd.universe_root,
            odd.signature_manifest_root,
            odd.saturation_state_root,
        ),
        (
            sink.input_signature_id,
            sink.universe_root,
            sink.signature_manifest_root,
            sink.saturation_state_root,
        ),
    )
    if (
        closure.signature_rows != expected_rows
        or odd_manifest.semantic_binding_root != closure.semantic_binding_root
        or sink_manifest.semantic_binding_root != closure.semantic_binding_root
        or odd_manifest.projection_profile_root != closure.projection_profile_root
        or sink_manifest.projection_profile_root != closure.projection_profile_root
    ):
        _fail("REJECT_Q2_Q1_ARCHIVE", "Q1 closure/signature binding differs")
    return odd, sink


__all__ = [
    "Q1ClassStreamV1",
    "Q2ArchiveReaderError",
    "decode_q1_class_records_v1",
    "read_q1_class_stream_v1",
    "read_q1_class_streams_v1",
]

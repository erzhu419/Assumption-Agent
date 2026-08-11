"""Deterministic qualification-only negative vectors for predicates 13 and 18.

The corpus exercises only the bounded node-three archive/qualification
surface.  Every rejecting row invokes a production validator and records its
exact stable failure code.  The one non-rejecting row is the exact 16 MiB
boundary control paired with the first rejected byte length.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Callable, Final, NoReturn

from . import phase3_q05b_host_replay_v1 as _host
from . import phase3_q1_archive_projection_v1 as _projection
from . import phase3_q1_capacity_preflight_v1 as _capacity
from . import phase3_q1_external_sort_profile_v1 as _external_sort
from . import phase3_q1_formal_archive_contract_v1 as _formal
from . import phase3_q1_partition_snapshot_v1 as _snapshot
from . import phase3_q1_qualification_wire_v1 as _wire
from . import phase3_q1_semantic_coverage_v1 as _coverage
from .strict_cbor_v1 import StrictCborError, canonical_cbor_decode, content_hash


NEGATIVE_VECTOR_CORPUS_SCHEMA_ID: Final = (
    b"hegel-q05b-negative-vector-corpus/1"
)
NEGATIVE_VECTOR_EVIDENCE_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/NEGATIVE_VECTOR_EVIDENCE/V1"
)
NEGATIVE_VECTOR_CORPUS_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/NEGATIVE_VECTOR_CORPUS/V1"
)
NEGATIVE_VECTOR_CATEGORY_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/NEGATIVE_VECTOR_CATEGORY/V1"
)
NO_FAILURE: Final = b"NO_FAILURE"

CLOSED_QUALIFICATION_AUTHORITY: Final = (
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

_PRODUCTION_ERRORS: Final = (
    StrictCborError,
    _projection.Q1ArchiveProjectionError,
    _external_sort.Q1ExternalSortError,
    _formal.Q1ArchiveContractError,
    _wire.Q05BWireQualificationError,
    _host.Q05BHostReplayError,
)


class Q05BNegativeVectorError(RuntimeError):
    """Fail-closed construction error for the frozen negative corpus."""


def _fail(detail: str) -> NoReturn:
    raise Q05BNegativeVectorError(detail)


@dataclass(frozen=True, slots=True)
class Q05BNegativeVectorRowV1:
    vector_id: bytes
    category: int
    expected_failure: bytes
    observed_failure: bytes
    evidence_root: bytes

    def __post_init__(self) -> None:
        if (
            type(self.vector_id) is not bytes
            or not self.vector_id
            or type(self.category) is not int
            or self.category not in (13, 18)
            or type(self.expected_failure) is not bytes
            or not self.expected_failure
            or type(self.observed_failure) is not bytes
            or self.observed_failure != self.expected_failure
            or type(self.evidence_root) is not bytes
            or len(self.evidence_root) != 32
        ):
            _fail("negative vector row differs type-exactly")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            self.vector_id,
            self.category,
            self.expected_failure,
            self.observed_failure,
            self.evidence_root,
        )


@dataclass(frozen=True, slots=True)
class Q05BNegativeVectorCorpusV1:
    rows: tuple[Q05BNegativeVectorRowV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.rows) is not tuple
            or len(self.rows) < 10
            or any(type(row) is not Q05BNegativeVectorRowV1 for row in self.rows)
            or tuple(row.vector_id for row in self.rows)
            != tuple(sorted(row.vector_id for row in self.rows))
            or len({row.vector_id for row in self.rows}) != len(self.rows)
            or {row.category for row in self.rows} != {13, 18}
        ):
            _fail("negative vector corpus registry differs")
        if sum(row.expected_failure == NO_FAILURE for row in self.rows) != 1:
            _fail("negative vector corpus requires one exact-boundary control")

    @property
    def category_roots(self) -> tuple[tuple[int, bytes], ...]:
        return tuple(
            (
                category,
                content_hash(
                    NEGATIVE_VECTOR_CATEGORY_ROOT_DOMAIN,
                    (
                        category,
                        tuple(
                            row.canonical_object()
                            for row in self.rows
                            if row.category == category
                        ),
                    ),
                ),
            )
            for category in (13, 18)
        )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            NEGATIVE_VECTOR_CORPUS_SCHEMA_ID,
            tuple(row.canonical_object() for row in self.rows),
            self.category_roots,
            CLOSED_QUALIFICATION_AUTHORITY,
        )

    @property
    def corpus_root(self) -> bytes:
        return content_hash(
            NEGATIVE_VECTOR_CORPUS_ROOT_DOMAIN,
            self.canonical_object(),
        )


def _observe_failure_v1(action: Callable[[], object]) -> bytes:
    try:
        action()
    except _PRODUCTION_ERRORS as error:
        code = getattr(error, "code", None)
        if type(code) is not str or not code.isascii() or not code:
            _fail("production rejection lacked one stable ASCII failure code")
        return code.encode("ascii")
    return NO_FAILURE


def _row_v1(
    vector_id: bytes,
    category: int,
    expected_failure: bytes,
    evidence_preimage: tuple[object, ...],
    action: Callable[[], object],
) -> Q05BNegativeVectorRowV1:
    observed = _observe_failure_v1(action)
    if observed != expected_failure:
        _fail(
            f"{vector_id.decode('ascii', 'replace')} expected "
            f"{expected_failure!r}, observed {observed!r}"
        )
    evidence_root = content_hash(
        NEGATIVE_VECTOR_EVIDENCE_ROOT_DOMAIN,
        (
            vector_id,
            category,
            expected_failure,
            observed,
            evidence_preimage,
        ),
    )
    return Q05BNegativeVectorRowV1(
        vector_id,
        category,
        expected_failure,
        observed,
        evidence_root,
    )


def _odd_partition_evidence_v1() -> _wire.Q05BNode3PartitionEvidenceV1:
    limits = _capacity.PreflightLimitsV1(maximum_ast_node_count=3)
    bounded_snapshot = _snapshot.build_q1_partition_snapshot_v1(1, limits=limits)
    records = _projection.records_from_partition_snapshot_v1(bounded_snapshot)
    coverage = _coverage.build_q1_semantic_coverage_v1(bounded_snapshot)
    return _wire.node3_partition_evidence_v1(
        bounded_snapshot,
        records,
        coverage,
    )


def _swap_first_two_frames_v1(blob: bytes) -> bytes:
    frames: list[bytes] = []
    offset = 0
    while offset < len(blob):
        length = int.from_bytes(blob[offset : offset + 4], "big")
        end = offset + 4 + length
        if end > len(blob):
            _fail("baseline framed blob is truncated")
        frames.append(blob[offset:end])
        offset = end
    if len(frames) < 2:
        _fail("baseline framed blob lacks two records")
    frames[0], frames[1] = frames[1], frames[0]
    return b"".join(frames)


def _candidate_then_host_v1(
    evidence: _wire.Q05BNode3PartitionEvidenceV1,
) -> None:
    decoded = _wire.decode_node3_partition_evidence_v1(evidence.canonical_bytes)
    if decoded.canonical_bytes != evidence.canonical_bytes:
        _fail("candidate partition decoder did not replay its accepted bytes")
    _host.strict_replay_partition_streams_v1(decoded)


def run_q05b_negative_vector_corpus_v1() -> Q05BNegativeVectorCorpusV1:
    """Execute and return the frozen predicate-13/18 evidence corpus."""

    accepted_raw = _wire.MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES
    accepted_payload = b"x" * accepted_raw
    rejected_payload = accepted_payload + b"x"

    baseline = _odd_partition_evidence_v1()

    record_set = list(baseline.record_set_object)
    programs = list(record_set[4])
    programs[1] = programs[0]
    record_set[4] = tuple(programs)
    duplicated_record_set = replace(
        baseline,
        record_set_object=tuple(record_set),
    )

    streams = list(baseline.stream_rows)
    manifest_stream = list(streams[0])
    manifest_trace = list(manifest_stream[3])
    manifests = list(manifest_trace[4])
    manifest = list(manifests[0])
    flipped = bytearray(manifest[8])
    flipped[0] ^= 1
    manifest[8] = bytes(flipped)
    manifests[0] = tuple(manifest)
    manifest_trace[4] = tuple(manifests)
    manifest_stream[3] = tuple(manifest_trace)
    streams[0] = tuple(manifest_stream)
    tampered_manifest = replace(baseline, stream_rows=tuple(streams))

    streams = list(baseline.stream_rows)
    framed_stream = list(streams[0])
    blobs = list(framed_stream[2])
    blobs[0] = _swap_first_two_frames_v1(blobs[0])
    framed_stream[2] = tuple(blobs)
    streams[0] = tuple(framed_stream)
    reordered_frames = replace(baseline, stream_rows=tuple(streams))

    small_trace = _external_sort.project_external_sort_trace_v1(
        ((b"a", b"\x81\x00"), (b"b", b"\x81\x01")),
        input_signature_id=1,
        stream_kind_id=_formal.ArchiveStreamKindId.PROGRAM,
    )
    scratch_events = list(small_trace.scratch_events)
    scratch_events[0] = replace(
        scratch_events[0],
        new_size=scratch_events[0].new_size + 1,
    )

    collision_seen: dict[bytes, bytes] = {}

    rows = (
        _row_v1(
            b"p13-boundary-accepted-exact-16mib",
            13,
            NO_FAILURE,
            (
                accepted_raw,
                _wire.framed_bstr_record_length_v1(accepted_raw),
            ),
            lambda: (
                _wire.bstr_record_fits_frozen_chunk_v1(accepted_raw) is True
                and len(_formal.frame_canonical_record_v1(accepted_payload))
                == _wire.MAX_CHUNK_FRAMED_BYTES
            )
            or _fail("exact boundary control was not accepted"),
        ),
        _row_v1(
            b"p13-boundary-reject-plus-one",
            13,
            b"INCONCLUSIVE_RESOURCE_LIMIT",
            (
                accepted_raw + 1,
                _wire.framed_bstr_record_length_v1(accepted_raw + 1),
            ),
            lambda: _projection.chunk_canonical_records_v1(
                (rejected_payload,),
                (b"r" * 32,),
                input_signature_id=1,
                universe_root=baseline.universe_root,
                stream_kind_id=_formal.ArchiveStreamKindId.PROGRAM,
            ),
        ),
        _row_v1(
            b"p13-cbor-bool-is-not-uint-alias",
            13,
            b"REJECT_Q05B_UINT",
            (b"bool-must-not-alias-int",),
            lambda: _wire.framed_bstr_record_length_v1(True),
        ),
        _row_v1(
            b"p13-cbor-noncanonical-uint",
            13,
            b"REJECT_NONCANONICAL_CBOR",
            (b"\x18\x17",),
            lambda: canonical_cbor_decode(b"\x18\x17"),
        ),
        _row_v1(
            b"p13-framed-record-reorder-host-reject",
            13,
            _host.FAIL_HOST_STREAM.encode("ascii"),
            (sha256(reordered_frames.canonical_bytes).digest(),),
            lambda: _candidate_then_host_v1(reordered_frames),
        ),
        _row_v1(
            b"p18-candidate-gap-external-manifest-hash",
            18,
            _host.FAIL_HOST_STREAM.encode("ascii"),
            (sha256(tampered_manifest.canonical_bytes).digest(),),
            lambda: _candidate_then_host_v1(tampered_manifest),
        ),
        _row_v1(
            b"p18-candidate-gap-framed-reorder",
            18,
            _host.FAIL_HOST_STREAM.encode("ascii"),
            (sha256(reordered_frames.canonical_bytes).digest(),),
            lambda: _candidate_then_host_v1(reordered_frames),
        ),
        _row_v1(
            b"p18-candidate-gap-record-set-duplicate",
            18,
            _host.FAIL_HOST_STREAM.encode("ascii"),
            (sha256(duplicated_record_set.canonical_bytes).digest(),),
            lambda: _candidate_then_host_v1(duplicated_record_set),
        ),
        _row_v1(
            b"p18-digest-preimage-collision",
            18,
            b"FAIL_SHA256_PREIMAGE_COLLISION",
            (b"same-digest", b"different-preimages"),
            lambda: (
                _projection._register_digest_preimage_v1(  # noqa: SLF001
                    collision_seen,
                    digest=b"d" * 32,
                    preimage=b"first",
                    label="negative vector",
                ),
                _projection._register_digest_preimage_v1(  # noqa: SLF001
                    collision_seen,
                    digest=b"d" * 32,
                    preimage=b"second",
                    label="negative vector",
                ),
            ),
        ),
        _row_v1(
            b"p18-external-sort-duplicate-key",
            18,
            b"REJECT_Q1_SORT_INPUT",
            (b"duplicate-key",),
            lambda: _external_sort.project_external_sort_trace_v1(
                ((b"same", b"a"), (b"same", b"b")),
                input_signature_id=1,
                stream_kind_id=_formal.ArchiveStreamKindId.PROGRAM,
            ),
        ),
        _row_v1(
            b"p18-scratch-event-tamper",
            18,
            b"REJECT_Q1_SORT_TRACE",
            (
                small_trace.projection.scratch_event_ledger_root,
                scratch_events[0].canonical_object(),
            ),
            lambda: _external_sort.Q1ExternalSortTraceV1(
                small_trace.projection,
                small_trace.ordered_rows,
                small_trace.run_manifests,
                tuple(scratch_events),
            ),
        ),
    )
    return Q05BNegativeVectorCorpusV1(tuple(sorted(rows, key=lambda row: row.vector_id)))


__all__ = [
    "CLOSED_QUALIFICATION_AUTHORITY",
    "NEGATIVE_VECTOR_CATEGORY_ROOT_DOMAIN",
    "NEGATIVE_VECTOR_CORPUS_ROOT_DOMAIN",
    "NEGATIVE_VECTOR_CORPUS_SCHEMA_ID",
    "NEGATIVE_VECTOR_EVIDENCE_ROOT_DOMAIN",
    "NO_FAILURE",
    "Q05BNegativeVectorCorpusV1",
    "Q05BNegativeVectorError",
    "Q05BNegativeVectorRowV1",
    "run_q05b_negative_vector_corpus_v1",
]

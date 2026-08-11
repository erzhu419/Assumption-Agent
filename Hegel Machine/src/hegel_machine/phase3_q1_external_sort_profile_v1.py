"""Deterministic external-sort and scratch projection for Q1 archive streams."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Final, Iterable, NoReturn

from .phase3_q1_formal_archive_contract_v1 import (
    ArchiveStreamKindId,
    EXTERNAL_SORT_MERGE_FAN_IN,
    EXTERNAL_SORT_RUN_HEADER_BYTES,
    EXTERNAL_SORT_RUN_MAGIC,
    EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES,
    SCRATCH_ALLOCATION_BLOCK_BYTES,
    SCRATCH_METADATA_RESERVE_BYTES_PER_LIVE_FILE,
)
from .strict_cbor_v1 import content_hash, rfc6962_root


EXTERNAL_SORT_PROJECTION_SCHEMA_ID: Final = (
    b"hegel-q1-external-sort-projection/1"
)
EXTERNAL_SORT_TRACE_SCHEMA_ID: Final = b"hegel-q1-external-sort-trace/1"
SCRATCH_EVENT_SCHEMA_ID: Final = b"hegel-q1-scratch-event/1"
RUN_MANIFEST_SCHEMA_ID: Final = b"hegel-q1-external-sort-run/1"
SORTED_STREAM_ROOT_DOMAIN: Final = "HEGEL/Q1/PREFLIGHT/SORTED_STREAM/V1"
SCRATCH_LEDGER_ROOT_DOMAIN: Final = "HEGEL/Q1/PREFLIGHT/SCRATCH_LEDGER/V1"
EXTERNAL_SORT_PROJECTION_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PREFLIGHT/EXTERNAL_SORT_PROJECTION/V1"
)


class ScratchActionId:
    ALLOC: Final = 1
    GROW: Final = 2
    SEAL: Final = 3
    FREE: Final = 4


class Q1ExternalSortError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1ExternalSortError(code, detail)


def _charged_file_bytes(size: int) -> int:
    if type(size) is not int or size < 0:
        _fail("REJECT_Q1_SCRATCH_SIZE", "file size must be nonnegative int")
    blocks = (size + SCRATCH_ALLOCATION_BLOCK_BYTES - 1) // (
        SCRATCH_ALLOCATION_BLOCK_BYTES
    )
    return (
        blocks * SCRATCH_ALLOCATION_BLOCK_BYTES
        + SCRATCH_METADATA_RESERVE_BYTES_PER_LIVE_FILE
    )


def external_sort_row_bytes_v1(key: bytes, record: bytes) -> bytes:
    if type(key) is not bytes or not key:
        _fail("REJECT_Q1_SORT_ROW", "sort key must be nonempty bytes")
    if type(record) is not bytes or not record:
        _fail("REJECT_Q1_SORT_ROW", "record must be nonempty bytes")
    if len(key) > 0xFFFFFFFF or len(record) > 0xFFFFFFFF:
        _fail("REJECT_Q1_SORT_ROW", "sort key or record exceeds u32 framing")
    return (
        len(key).to_bytes(4, "big")
        + key
        + len(record).to_bytes(4, "big")
        + record
    )


def external_sort_merge_shape_v1(initial_run_count: int) -> tuple[int, ...]:
    """Return deterministic run counts from initial generation to one run."""

    if type(initial_run_count) is not int or initial_run_count < 0:
        _fail("REJECT_Q1_SORT_SHAPE", "initial run count must be uint")
    if initial_run_count == 0:
        return (0,)
    shape = [initial_run_count]
    current = initial_run_count
    while current > 1:
        current = (current + EXTERNAL_SORT_MERGE_FAN_IN - 1) // (
            EXTERNAL_SORT_MERGE_FAN_IN
        )
        shape.append(current)
    return tuple(shape)


def run_header_v1(
    *,
    input_signature_id: int,
    stream_kind_id: ArchiveStreamKindId,
    level: int,
    run_index: int,
    record_count: int,
    payload: bytes,
) -> bytes:
    if type(input_signature_id) is not int or input_signature_id not in (1, 2):
        _fail("REJECT_Q1_SORT_HEADER", "input signature must be exact int 1 or 2")
    if not isinstance(stream_kind_id, ArchiveStreamKindId):
        _fail("REJECT_Q1_SORT_HEADER", "stream kind is unregistered")
    for value, name, width in (
        (level, "level", 2),
        (run_index, "run_index", 4),
        (record_count, "record_count", 8),
    ):
        if type(value) is not int or not 0 <= value < 1 << (8 * width):
            _fail("REJECT_Q1_SORT_HEADER", f"{name} is outside wire range")
    if type(payload) is not bytes:
        _fail("REJECT_Q1_SORT_HEADER", "payload must be bytes")
    header = (
        EXTERNAL_SORT_RUN_MAGIC
        + (1).to_bytes(2, "big")
        + input_signature_id.to_bytes(2, "big")
        + int(stream_kind_id).to_bytes(2, "big")
        + level.to_bytes(2, "big")
        + run_index.to_bytes(4, "big")
        + record_count.to_bytes(8, "big")
        + len(payload).to_bytes(8, "big")
        + sha256(payload).digest()
    )
    if len(header) != EXTERNAL_SORT_RUN_HEADER_BYTES:
        raise AssertionError("Q1 external-sort header length drift")
    return header


def replay_run_file_v1(
    run_file: bytes,
    *,
    input_signature_id: int,
    stream_kind_id: ArchiveStreamKindId,
    level: int,
    run_index: int,
) -> tuple[tuple[bytes, bytes], ...]:
    """Strictly reopen one complete run image and replay every framed row."""

    if type(run_file) is not bytes or len(run_file) < EXTERNAL_SORT_RUN_HEADER_BYTES:
        _fail("REJECT_Q1_SORT_RUN_REPLAY", "run file is truncated")
    header = run_file[:EXTERNAL_SORT_RUN_HEADER_BYTES]
    payload = run_file[EXTERNAL_SORT_RUN_HEADER_BYTES:]
    if header[:8] != EXTERNAL_SORT_RUN_MAGIC:
        _fail("REJECT_Q1_SORT_RUN_REPLAY", "run magic differs")
    numeric = (
        int.from_bytes(header[8:10], "big"),
        int.from_bytes(header[10:12], "big"),
        int.from_bytes(header[12:14], "big"),
        int.from_bytes(header[14:16], "big"),
        int.from_bytes(header[16:20], "big"),
        int.from_bytes(header[20:28], "big"),
        int.from_bytes(header[28:36], "big"),
    )
    if numeric[:5] != (
        1,
        input_signature_id,
        int(stream_kind_id),
        level,
        run_index,
    ):
        _fail("REJECT_Q1_SORT_RUN_REPLAY", "run header binding differs")
    record_count, payload_bytes = numeric[5:]
    if payload_bytes != len(payload) or header[36:68] != sha256(payload).digest():
        _fail("REJECT_Q1_SORT_RUN_REPLAY", "run payload commitment differs")

    rows: list[tuple[bytes, bytes]] = []
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            _fail("REJECT_Q1_SORT_RUN_REPLAY", "truncated key length")
        key_length = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        if key_length == 0 or offset + key_length + 4 > len(payload):
            _fail("REJECT_Q1_SORT_RUN_REPLAY", "truncated or empty key")
        key = payload[offset : offset + key_length]
        offset += key_length
        record_length = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        if record_length == 0 or offset + record_length > len(payload):
            _fail("REJECT_Q1_SORT_RUN_REPLAY", "truncated or empty record")
        record = payload[offset : offset + record_length]
        offset += record_length
        if external_sort_row_bytes_v1(key, record) != (
            key_length.to_bytes(4, "big")
            + key
            + record_length.to_bytes(4, "big")
            + record
        ):
            raise AssertionError("Q1 run-row replay drift")
        rows.append((key, record))
    replayed = tuple(rows)
    if len(replayed) != record_count:
        _fail("REJECT_Q1_SORT_RUN_REPLAY", "run record count differs")
    if replayed != tuple(sorted(replayed, key=lambda row: (row[0], row[1]))):
        _fail("REJECT_Q1_SORT_RUN_REPLAY", "run rows are not sorted")
    if len({row[0] for row in replayed}) != len(replayed):
        _fail("REJECT_Q1_SORT_RUN_REPLAY", "run contains duplicate sort key")
    expected_header = run_header_v1(
        input_signature_id=input_signature_id,
        stream_kind_id=stream_kind_id,
        level=level,
        run_index=run_index,
        record_count=len(replayed),
        payload=payload,
    )
    if header != expected_header:
        _fail("REJECT_Q1_SORT_RUN_REPLAY", "run header re-encode differs")
    return replayed


@dataclass(frozen=True, slots=True)
class ScratchEventV1:
    sequence: int
    action_id: int
    file_id: bytes
    prior_size: int
    new_size: int
    live_logical_bytes_after: int
    live_charged_bytes_after: int

    def canonical_object(self) -> tuple[object, ...]:
        if type(self.sequence) is not int or self.sequence < 0:
            _fail("REJECT_Q1_SCRATCH_EVENT", "sequence must be uint")
        if type(self.action_id) is not int or self.action_id not in (
            ScratchActionId.ALLOC,
            ScratchActionId.GROW,
            ScratchActionId.SEAL,
            ScratchActionId.FREE,
        ):
            _fail("REJECT_Q1_SCRATCH_EVENT", "action is unregistered")
        if type(self.file_id) is not bytes or not self.file_id:
            _fail("REJECT_Q1_SCRATCH_EVENT", "file id must be nonempty bytes")
        for name in (
            "prior_size",
            "new_size",
            "live_logical_bytes_after",
            "live_charged_bytes_after",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail("REJECT_Q1_SCRATCH_EVENT", f"{name} must be uint")
        return (
            1,
            SCRATCH_EVENT_SCHEMA_ID,
            self.sequence,
            self.action_id,
            self.file_id,
            self.prior_size,
            self.new_size,
            self.live_logical_bytes_after,
            self.live_charged_bytes_after,
        )


@dataclass(frozen=True, slots=True)
class _Run:
    level: int
    index: int
    rows: tuple[tuple[bytes, bytes], ...]

    @property
    def payload(self) -> bytes:
        return b"".join(external_sort_row_bytes_v1(key, record) for key, record in self.rows)

    @property
    def size(self) -> int:
        return EXTERNAL_SORT_RUN_HEADER_BYTES + len(self.payload)

    @property
    def file_id(self) -> bytes:
        return f"level-{self.level:04d}-run-{self.index:08d}".encode("ascii")

    def file_bytes(
        self,
        input_signature_id: int,
        stream_kind_id: ArchiveStreamKindId,
    ) -> bytes:
        payload = self.payload
        return run_header_v1(
            input_signature_id=input_signature_id,
            stream_kind_id=stream_kind_id,
            level=self.level,
            run_index=self.index,
            record_count=len(self.rows),
            payload=payload,
        ) + payload

    def manifest_object(
        self,
        input_signature_id: int,
        stream_kind_id: ArchiveStreamKindId,
    ) -> tuple[object, ...]:
        payload = self.payload
        return (
            1,
            RUN_MANIFEST_SCHEMA_ID,
            input_signature_id,
            int(stream_kind_id),
            self.level,
            self.index,
            len(self.rows),
            len(payload),
            sha256(payload).digest(),
        )


@dataclass(frozen=True, slots=True)
class Q1ExternalSortProjectionV1:
    input_signature_id: int
    stream_kind_id: ArchiveStreamKindId
    record_count: int
    input_payload_bytes: int
    initial_run_count: int
    merge_level_count: int
    final_run_bytes: int
    logical_scratch_high_water_bytes: int
    charged_scratch_high_water_bytes: int
    sorted_stream_root: bytes
    run_manifest_archive_root: bytes
    scratch_event_ledger_root: bytes
    scratch_event_count: int

    def canonical_object(self) -> tuple[object, ...]:
        if type(self.input_signature_id) is not int or self.input_signature_id not in (1, 2):
            _fail("REJECT_Q1_SORT_PROJECTION", "input signature differs")
        if not isinstance(self.stream_kind_id, ArchiveStreamKindId):
            _fail("REJECT_Q1_SORT_PROJECTION", "stream kind differs")
        for name in (
            "record_count",
            "input_payload_bytes",
            "initial_run_count",
            "merge_level_count",
            "final_run_bytes",
            "logical_scratch_high_water_bytes",
            "charged_scratch_high_water_bytes",
            "scratch_event_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail("REJECT_Q1_SORT_PROJECTION", f"{name} must be uint")
        for name in (
            "sorted_stream_root",
            "run_manifest_archive_root",
            "scratch_event_ledger_root",
        ):
            value = getattr(self, name)
            if type(value) is not bytes or len(value) != 32:
                _fail("REJECT_Q1_SORT_PROJECTION", f"{name} must be root")
        return (
            1,
            EXTERNAL_SORT_PROJECTION_SCHEMA_ID,
            self.input_signature_id,
            int(self.stream_kind_id),
            self.record_count,
            self.input_payload_bytes,
            self.initial_run_count,
            self.merge_level_count,
            self.final_run_bytes,
            self.logical_scratch_high_water_bytes,
            self.charged_scratch_high_water_bytes,
            self.sorted_stream_root,
            self.run_manifest_archive_root,
            self.scratch_event_ledger_root,
            self.scratch_event_count,
        )

    @property
    def diagnostic_root(self) -> bytes:
        return content_hash(
            EXTERNAL_SORT_PROJECTION_ROOT_DOMAIN,
            self.canonical_object(),
        )


@dataclass(frozen=True, slots=True)
class Q1ExternalSortTraceV1:
    """Immutable locally materialized preimages for actor/host replay."""

    projection: Q1ExternalSortProjectionV1
    ordered_rows: tuple[tuple[bytes, bytes], ...]
    run_manifests: tuple[tuple[object, ...], ...]
    scratch_events: tuple[ScratchEventV1, ...]

    def __post_init__(self) -> None:
        if type(self.projection) is not Q1ExternalSortProjectionV1:
            _fail("REJECT_Q1_SORT_TRACE", "projection has wrong exact type")
        self.projection.canonical_object()
        if type(self.ordered_rows) is not tuple or any(
            type(row) is not tuple or len(row) != 2
            for row in self.ordered_rows
        ):
            _fail("REJECT_Q1_SORT_TRACE", "ordered rows are malformed")
        for row in self.ordered_rows:
            external_sort_row_bytes_v1(row[0], row[1])
        if self.ordered_rows != tuple(
            sorted(self.ordered_rows, key=lambda row: (row[0], row[1]))
        ) or len({row[0] for row in self.ordered_rows}) != len(self.ordered_rows):
            _fail("REJECT_Q1_SORT_TRACE", "ordered rows are not canonical")
        projection = self.projection
        if (
            projection.record_count != len(self.ordered_rows)
            or projection.input_payload_bytes
            != sum(len(external_sort_row_bytes_v1(*row)) for row in self.ordered_rows)
            or projection.sorted_stream_root
            != content_hash(SORTED_STREAM_ROOT_DOMAIN, self.ordered_rows)
        ):
            _fail("REJECT_Q1_SORT_TRACE", "ordered-row projection replay differs")

        # Reconstruct every run and scratch event from the ordered input.  The
        # trace is used across an actor boundary, so merely checking that the
        # supplied manifest/event roots hash back to the projection would be
        # insufficient: arbitrary but self-consistent preimages must not pass.
        expected_initial: list[_Run] = []
        pending: list[tuple[bytes, bytes]] = []
        pending_bytes = 0
        for row in self.ordered_rows:
            encoded_length = len(external_sort_row_bytes_v1(*row))
            if encoded_length > EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES:
                _fail("REJECT_Q1_SORT_TRACE", "one row exceeds frozen run limit")
            if (
                pending
                and pending_bytes + encoded_length
                > EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES
            ):
                expected_initial.append(
                    _Run(0, len(expected_initial), tuple(pending))
                )
                pending = []
                pending_bytes = 0
            pending.append(row)
            pending_bytes += encoded_length
        if pending:
            expected_initial.append(_Run(0, len(expected_initial), tuple(pending)))

        expected_ledger = _ScratchLedger()
        expected_manifests: list[tuple[object, ...]] = []

        def append_expected_run(run: _Run) -> None:
            expected_ledger.allocate(run.file_id)
            expected_ledger.grow(run.file_id, run.size)
            expected_ledger.seal(run.file_id)
            expected_manifests.append(
                run.manifest_object(
                    projection.input_signature_id,
                    projection.stream_kind_id,
                )
            )

        for run in expected_initial:
            append_expected_run(run)
        expected_current = tuple(expected_initial)
        expected_level = 0
        while len(expected_current) > 1:
            expected_level += 1
            expected_output: list[_Run] = []
            for start in range(0, len(expected_current), EXTERNAL_SORT_MERGE_FAN_IN):
                group = expected_current[
                    start : start + EXTERNAL_SORT_MERGE_FAN_IN
                ]
                merged_rows = tuple(
                    sorted(
                        (row for child in group for row in child.rows),
                        key=lambda row: (row[0], row[1]),
                    )
                )
                merged = _Run(expected_level, len(expected_output), merged_rows)
                append_expected_run(merged)
                for child in group:
                    expected_ledger.free(child.file_id)
                expected_output.append(merged)
            expected_current = tuple(expected_output)
        if expected_current:
            expected_ledger.free(expected_current[0].file_id)
        if expected_ledger.live:
            raise AssertionError("Q1 reconstructed scratch ledger did not close")
        if self.run_manifests != tuple(expected_manifests):
            _fail("REJECT_Q1_SORT_TRACE", "run manifests differ from exact replay")
        if self.scratch_events != tuple(expected_ledger.events):
            _fail("REJECT_Q1_SORT_TRACE", "scratch events differ from exact replay")

        if type(self.run_manifests) is not tuple:
            _fail("REJECT_Q1_SORT_TRACE", "run manifests must be exact tuple")
        shape = external_sort_merge_shape_v1(projection.initial_run_count)
        expected_run_coordinates = tuple(
            (level, index)
            for level, count in enumerate(shape)
            for index in range(count)
        )
        manifest_coordinates: list[tuple[int, int]] = []
        manifest_sizes: dict[bytes, int] = {}
        for manifest in self.run_manifests:
            if (
                type(manifest) is not tuple
                or len(manifest) != 9
                or manifest[:2] != (1, RUN_MANIFEST_SCHEMA_ID)
                or type(manifest[2]) is not int
                or manifest[2] != projection.input_signature_id
                or type(manifest[3]) is not int
                or manifest[3] != int(projection.stream_kind_id)
                or any(type(manifest[index]) is not int for index in range(4, 8))
                or any(manifest[index] < 0 for index in range(4, 8))
                or type(manifest[8]) is not bytes
                or len(manifest[8]) != 32
            ):
                _fail("REJECT_Q1_SORT_TRACE", "run manifest wire differs")
            level = manifest[4]
            index = manifest[5]
            record_count = manifest[6]
            payload_bytes = manifest[7]
            file_id = f"level-{level:04d}-run-{index:08d}".encode("ascii")
            if file_id in manifest_sizes:
                _fail("REJECT_Q1_SORT_TRACE", "run manifest file repeats")
            manifest_coordinates.append((level, index))
            manifest_sizes[file_id] = EXTERNAL_SORT_RUN_HEADER_BYTES + payload_bytes
            if record_count < 1 or payload_bytes < record_count * 10:
                _fail("REJECT_Q1_SORT_TRACE", "run manifest cardinality differs")
        if tuple(manifest_coordinates) != expected_run_coordinates:
            _fail("REJECT_Q1_SORT_TRACE", "run manifest merge shape differs")
        for level, count in enumerate(shape):
            if count and sum(
                manifest[6]
                for manifest in self.run_manifests
                if manifest[4] == level
            ) != len(self.ordered_rows):
                _fail("REJECT_Q1_SORT_TRACE", "run level record total differs")
        if (
            projection.run_manifest_archive_root
            != rfc6962_root(self.run_manifests)
            or projection.initial_run_count != (shape[0] if shape != (0,) else 0)
            or projection.merge_level_count != (len(shape) - 1)
        ):
            _fail("REJECT_Q1_SORT_TRACE", "run manifest projection replay differs")
        expected_final_bytes = (
            manifest_sizes[
                f"level-{len(shape) - 1:04d}-run-{0:08d}".encode("ascii")
            ]
            if self.ordered_rows
            else 0
        )
        if projection.final_run_bytes != expected_final_bytes:
            _fail("REJECT_Q1_SORT_TRACE", "final run byte count differs")

        if type(self.scratch_events) is not tuple or any(
            type(event) is not ScratchEventV1 for event in self.scratch_events
        ):
            _fail("REJECT_Q1_SORT_TRACE", "scratch events are malformed")
        live: dict[bytes, int] = {}
        action_counts: dict[tuple[bytes, int], int] = {}
        logical_high_water = 0
        charged_high_water = 0
        for sequence, event in enumerate(self.scratch_events):
            event.canonical_object()
            if event.sequence != sequence:
                _fail("REJECT_Q1_SORT_TRACE", "scratch event sequence differs")
            prior = live.get(event.file_id)
            if event.action_id == ScratchActionId.ALLOC:
                if prior is not None or event.prior_size != 0 or event.new_size != EXTERNAL_SORT_RUN_HEADER_BYTES:
                    _fail("REJECT_Q1_SORT_TRACE", "scratch allocation differs")
                live[event.file_id] = event.new_size
            elif event.action_id == ScratchActionId.GROW:
                if prior is None or event.prior_size != prior or event.new_size != manifest_sizes.get(event.file_id):
                    _fail("REJECT_Q1_SORT_TRACE", "scratch growth differs")
                live[event.file_id] = event.new_size
            elif event.action_id == ScratchActionId.SEAL:
                if prior is None or event.prior_size != prior or event.new_size != prior:
                    _fail("REJECT_Q1_SORT_TRACE", "scratch seal differs")
            elif event.action_id == ScratchActionId.FREE:
                if prior is None or event.prior_size != prior or event.new_size != 0:
                    _fail("REJECT_Q1_SORT_TRACE", "scratch free differs")
                del live[event.file_id]
            else:
                _fail("REJECT_Q1_SORT_TRACE", "scratch action is unregistered")
            logical = sum(live.values())
            charged = sum(_charged_file_bytes(size) for size in live.values())
            if (
                event.live_logical_bytes_after != logical
                or event.live_charged_bytes_after != charged
            ):
                _fail("REJECT_Q1_SORT_TRACE", "scratch high-water preimage differs")
            logical_high_water = max(logical_high_water, logical)
            charged_high_water = max(charged_high_water, charged)
            key = (event.file_id, event.action_id)
            action_counts[key] = action_counts.get(key, 0) + 1
        if live:
            _fail("REJECT_Q1_SORT_TRACE", "scratch trace leaves live files")
        if any(
            action_counts.get((file_id, action), 0) != 1
            for file_id in manifest_sizes
            for action in (
                ScratchActionId.ALLOC,
                ScratchActionId.GROW,
                ScratchActionId.SEAL,
                ScratchActionId.FREE,
            )
        ) or len(action_counts) != 4 * len(manifest_sizes):
            _fail("REJECT_Q1_SORT_TRACE", "scratch lifecycle is incomplete")
        event_objects = tuple(event.canonical_object() for event in self.scratch_events)
        if (
            projection.scratch_event_count != len(self.scratch_events)
            or projection.scratch_event_ledger_root
            != content_hash(SCRATCH_LEDGER_ROOT_DOMAIN, event_objects)
            or projection.logical_scratch_high_water_bytes != logical_high_water
            or projection.charged_scratch_high_water_bytes != charged_high_water
        ):
            _fail("REJECT_Q1_SORT_TRACE", "scratch projection replay differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            EXTERNAL_SORT_TRACE_SCHEMA_ID,
            self.projection.canonical_object(),
            self.ordered_rows,
            self.run_manifests,
            tuple(event.canonical_object() for event in self.scratch_events),
        )


class _ScratchLedger:
    def __init__(self) -> None:
        self.live: dict[bytes, int] = {}
        self.events: list[ScratchEventV1] = []
        self.logical_high_water = 0
        self.charged_high_water = 0

    def _append(self, action: int, file_id: bytes, prior: int, new: int) -> None:
        logical = sum(self.live.values())
        charged = sum(_charged_file_bytes(size) for size in self.live.values())
        self.logical_high_water = max(self.logical_high_water, logical)
        self.charged_high_water = max(self.charged_high_water, charged)
        self.events.append(
            ScratchEventV1(
                len(self.events),
                action,
                file_id,
                prior,
                new,
                logical,
                charged,
            )
        )

    def allocate(self, file_id: bytes) -> None:
        if file_id in self.live:
            _fail("FAIL_Q1_SCRATCH_LEDGER", "file allocated twice")
        self.live[file_id] = EXTERNAL_SORT_RUN_HEADER_BYTES
        self._append(ScratchActionId.ALLOC, file_id, 0, EXTERNAL_SORT_RUN_HEADER_BYTES)

    def grow(self, file_id: bytes, final_size: int) -> None:
        prior = self.live.get(file_id)
        if prior is None or type(final_size) is not int or final_size < prior:
            _fail("FAIL_Q1_SCRATCH_LEDGER", "invalid file growth")
        self.live[file_id] = final_size
        self._append(ScratchActionId.GROW, file_id, prior, final_size)

    def seal(self, file_id: bytes) -> None:
        prior = self.live.get(file_id)
        if prior is None:
            _fail("FAIL_Q1_SCRATCH_LEDGER", "seal references absent file")
        self._append(ScratchActionId.SEAL, file_id, prior, prior)

    def free(self, file_id: bytes) -> None:
        prior = self.live.pop(file_id, None)
        if prior is None:
            _fail("FAIL_Q1_SCRATCH_LEDGER", "free references absent file")
        self._append(ScratchActionId.FREE, file_id, prior, 0)


def _project_external_sort_trace_v1(
    rows: Iterable[tuple[bytes, bytes]],
    *,
    input_signature_id: int,
    stream_kind_id: ArchiveStreamKindId,
) -> Q1ExternalSortTraceV1:
    if type(input_signature_id) is not int or input_signature_id not in (1, 2):
        _fail("REJECT_Q1_SORT_INPUT", "input signature must be exact int 1 or 2")
    if not isinstance(stream_kind_id, ArchiveStreamKindId):
        raise TypeError("stream_kind_id must be ArchiveStreamKindId")
    material = tuple(rows)
    for row in material:
        if type(row) is not tuple or len(row) != 2:
            _fail("REJECT_Q1_SORT_INPUT", "row must be exact key/record tuple")
        external_sort_row_bytes_v1(row[0], row[1])
    ordered = tuple(sorted(material, key=lambda row: (row[0], row[1])))
    if len({row[0] for row in ordered}) != len(ordered):
        _fail("REJECT_Q1_SORT_INPUT", "duplicate sort key")

    initial: list[_Run] = []
    pending: list[tuple[bytes, bytes]] = []
    pending_bytes = 0
    for row in ordered:
        encoded_length = len(external_sort_row_bytes_v1(*row))
        if encoded_length > EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES:
            _fail("INCONCLUSIVE_Q1_SORT_ROW_TOO_LARGE", "one row exceeds run limit")
        if pending and pending_bytes + encoded_length > EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES:
            initial.append(_Run(0, len(initial), tuple(pending)))
            pending = []
            pending_bytes = 0
        pending.append(row)
        pending_bytes += encoded_length
    if pending:
        initial.append(_Run(0, len(initial), tuple(pending)))

    ledger = _ScratchLedger()
    manifests: list[tuple[object, ...]] = []

    def seal_and_replay(run: _Run) -> tuple[tuple[bytes, bytes], ...]:
        run_file = run.file_bytes(input_signature_id, stream_kind_id)
        ledger.allocate(run.file_id)
        ledger.grow(run.file_id, len(run_file))
        ledger.seal(run.file_id)
        replayed = replay_run_file_v1(
            run_file,
            input_signature_id=input_signature_id,
            stream_kind_id=stream_kind_id,
            level=run.level,
            run_index=run.index,
        )
        if replayed != run.rows:
            _fail("REJECT_Q1_SORT_RUN_REPLAY", "run rows differ after reopen")
        manifests.append(run.manifest_object(input_signature_id, stream_kind_id))
        return replayed

    for run in initial:
        seal_and_replay(run)

    current = tuple(initial)
    merge_level_count = 0
    while len(current) > 1:
        merge_level_count += 1
        output: list[_Run] = []
        for start in range(0, len(current), EXTERNAL_SORT_MERGE_FAN_IN):
            group = current[start : start + EXTERNAL_SORT_MERGE_FAN_IN]
            merged_rows = tuple(
                sorted(
                    (
                        row
                        for child in group
                        for row in replay_run_file_v1(
                            child.file_bytes(input_signature_id, stream_kind_id),
                            input_signature_id=input_signature_id,
                            stream_kind_id=stream_kind_id,
                            level=child.level,
                            run_index=child.index,
                        )
                    ),
                    key=lambda row: (row[0], row[1]),
                )
            )
            run = _Run(merge_level_count, len(output), merged_rows)
            seal_and_replay(run)
            for child in group:
                ledger.free(child.file_id)
            output.append(run)
        current = tuple(output)

    final_size = current[0].size if current else 0
    if current:
        ledger.free(current[0].file_id)
    stream_root = content_hash(SORTED_STREAM_ROOT_DOMAIN, ordered)
    run_root = rfc6962_root(manifests)
    event_objects = tuple(event.canonical_object() for event in ledger.events)
    ledger_root = content_hash(SCRATCH_LEDGER_ROOT_DOMAIN, event_objects)
    projection = Q1ExternalSortProjectionV1(
        input_signature_id,
        stream_kind_id,
        len(ordered),
        sum(len(external_sort_row_bytes_v1(*row)) for row in ordered),
        len(initial),
        merge_level_count,
        final_size,
        ledger.logical_high_water,
        ledger.charged_high_water,
        stream_root,
        run_root,
        ledger_root,
        len(ledger.events),
    )
    if ledger.live:
        raise AssertionError("Q1 scratch ledger did not release all run files")
    return Q1ExternalSortTraceV1(
        projection,
        ordered,
        tuple(manifests),
        tuple(ledger.events),
    )


def project_external_sort_trace_v1(
    rows: Iterable[tuple[bytes, bytes]],
    *,
    input_signature_id: int,
    stream_kind_id: ArchiveStreamKindId,
) -> Q1ExternalSortTraceV1:
    """Project one stream and expose immutable preimages for host replay."""

    return _project_external_sort_trace_v1(
        rows,
        input_signature_id=input_signature_id,
        stream_kind_id=stream_kind_id,
    )


def project_external_sort_v1(
    rows: Iterable[tuple[bytes, bytes]],
    *,
    input_signature_id: int,
    stream_kind_id: ArchiveStreamKindId,
) -> Q1ExternalSortProjectionV1:
    return project_external_sort_trace_v1(
        rows,
        input_signature_id=input_signature_id,
        stream_kind_id=stream_kind_id,
    ).projection


__all__ = [
    "EXTERNAL_SORT_PROJECTION_SCHEMA_ID",
    "EXTERNAL_SORT_TRACE_SCHEMA_ID",
    "Q1ExternalSortError",
    "Q1ExternalSortProjectionV1",
    "Q1ExternalSortTraceV1",
    "ScratchActionId",
    "ScratchEventV1",
    "external_sort_row_bytes_v1",
    "external_sort_merge_shape_v1",
    "project_external_sort_v1",
    "project_external_sort_trace_v1",
    "replay_run_file_v1",
    "run_header_v1",
]

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


def project_external_sort_v1(
    rows: Iterable[tuple[bytes, bytes]],
    *,
    input_signature_id: int,
    stream_kind_id: ArchiveStreamKindId,
) -> Q1ExternalSortProjectionV1:
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
    return projection


__all__ = [
    "EXTERNAL_SORT_PROJECTION_SCHEMA_ID",
    "Q1ExternalSortError",
    "Q1ExternalSortProjectionV1",
    "ScratchActionId",
    "ScratchEventV1",
    "external_sort_row_bytes_v1",
    "external_sort_merge_shape_v1",
    "project_external_sort_v1",
    "replay_run_file_v1",
    "run_header_v1",
]

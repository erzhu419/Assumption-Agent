"""Exact-cover multiprocessing identity selection for formal FEVEROUS P6/E2.

Eight spawned workers scan disjoint, contiguous physical-rowid intervals from
the already SHA-qualified SQLite source.  Every worker independently rechecks
the immutable file identity and exact schema, exhausts its interval in strict
rowid order, enumerates all eligible atomic identities, applies the formal
private HMAC rank, and retains only its local bottom-k plus canonical gold
hits.  The coordinator requires a complete gap-free cover and performs the
mathematically exact local-bottom-k to global-bottom-k merge.

The formal secret is passed only over multiprocessing pipes after process
creation.  Python's spawn transport necessarily serializes the task payload in
transit, but the secret is never accepted on argv, logged, persisted by this
module, or serialized in any receipt.  Chunk receipts contain aggregate hashes
and counts; page ids and retained unit ids remain private in memory until the
bounded :class:`CorpusSelectionPlan` is formed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import hashlib
import heapq
import multiprocessing
import os
from pathlib import Path
import resource
import sqlite3
import stat
from types import MappingProxyType
from typing import Any

from assumption_agent.benchmarks import feverous_atomic_corpus_v1 as atomic
from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as source_adapter
from assumption_agent.benchmarks.feverous_p6_e2_acquisition_v1 import (
    CORPUS_UNIT_COUNT,
    CorpusIdentity,
    CorpusSelectionPlan,
    hmac_digest,
)
from assumption_agent.benchmarks.feverous_wikipedia_source_qualification_v1 import (
    FeverousWikipediaQualificationError,
    open_immutable_wiki_db,
)


VERSION = "feverous_p6_e2_parallel_identity_selection_v1"
RECEIPT_SCHEMA = f"{VERSION}_receipt"
FROZEN_WORKER_COUNT = 8
MAXIMUM_WORKER_COUNT = 16
SAMPLE_PAGE_COUNT = source_adapter.REAL_IDENTITY_COMPILER_SAMPLE_PAGE_COUNT


class FeverousParallelIdentitySelectionError(RuntimeError):
    """The formal source cover, worker aggregate, or bounded merge drifted."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_hash(value: object) -> str:
    return acquisition.stable_hash(value)


def _source_sha256(module: object) -> str:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        raise FeverousParallelIdentitySelectionError(
            "parallel selection source binding is unavailable"
        )
    try:
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    except OSError as exc:
        raise FeverousParallelIdentitySelectionError(
            "parallel selection source cannot be hashed"
        ) from exc


@dataclass(frozen=True)
class BoundDatabase:
    basename: str
    size_bytes: int
    declared_sha256: str
    row_count: int
    schema: str
    required_mode: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    source_spec_sha256: str
    source_binding_sha256: str
    formal_source_opener_source_sha256: str
    formal_source: bool

    def __post_init__(self) -> None:
        if (
            not self.basename
            or Path(self.basename).name != self.basename
            or type(self.size_bytes) is not int
            or self.size_bytes < 1
            or type(self.row_count) is not int
            or self.row_count < 1
            or not _is_sha256(self.declared_sha256)
            or not self.schema
            or type(self.required_mode) is not int
            or not 0 <= self.required_mode <= 0o777
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.device,
                    self.inode,
                    self.mtime_ns,
                    self.ctime_ns,
                )
            )
            or any(
                not _is_sha256(value)
                for value in (
                    self.source_spec_sha256,
                    self.source_binding_sha256,
                    self.formal_source_opener_source_sha256,
                )
            )
            or type(self.formal_source) is not bool
        ):
            raise FeverousParallelIdentitySelectionError(
                "parallel database binding is invalid"
            )


@dataclass(frozen=True)
class RowidInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 1
            or self.end < self.start
        ):
            raise FeverousParallelIdentitySelectionError(
                "parallel rowid interval is invalid"
            )

    @property
    def page_count(self) -> int:
        return self.end - self.start + 1


def partition_exact_cover(
    row_count: int, worker_count: int
) -> tuple[RowidInterval, ...]:
    if (
        type(row_count) is not int
        or row_count < 1
        or type(worker_count) is not int
        or not 1 <= worker_count <= min(row_count, MAXIMUM_WORKER_COUNT)
    ):
        raise FeverousParallelIdentitySelectionError(
            "parallel exact-cover dimensions are invalid"
        )
    quotient, remainder = divmod(row_count, worker_count)
    output: list[RowidInterval] = []
    cursor = 1
    for ordinal in range(worker_count):
        size = quotient + int(ordinal < remainder)
        output.append(RowidInterval(cursor, cursor + size - 1))
        cursor += size
    if cursor != row_count + 1:
        raise FeverousParallelIdentitySelectionError(
            "parallel exact-cover partition drifted"
        )
    return tuple(output)


def _observed_file_state(path: Path, binding: BoundDatabase) -> tuple[int, ...]:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise FeverousParallelIdentitySelectionError(
            "parallel database cannot be stated"
        ) from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_size != binding.size_bytes
        or stat.S_IMODE(observed.st_mode) != binding.required_mode
    ):
        raise FeverousParallelIdentitySelectionError(
            "parallel database file binding drifted"
        )
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
        stat.S_IMODE(observed.st_mode),
    )


def _expected_file_state(binding: BoundDatabase) -> tuple[int, ...]:
    return (
        binding.device,
        binding.inode,
        binding.size_bytes,
        binding.mtime_ns,
        binding.ctime_ns,
        binding.required_mode,
    )


def _open_bound_database(
    path: Path, binding: BoundDatabase
) -> sqlite3.Connection:
    if path.name != binding.basename or _observed_file_state(
        path, binding
    ) != _expected_file_state(binding):
        raise FeverousParallelIdentitySelectionError(
            "parallel database identity differs from qualified parent"
        )
    try:
        connection = open_immutable_wiki_db(path)
        schema_rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' ORDER BY name"
        ).fetchall()
        columns = connection.execute("PRAGMA table_info(wiki)").fetchall()
    except (sqlite3.Error, FeverousWikipediaQualificationError) as exc:
        raise FeverousParallelIdentitySelectionError(
            "parallel immutable SQLite open failed"
        ) from exc
    if (
        schema_rows != [("wiki", binding.schema)]
        or [row[1] for row in columns] != ["id", "data"]
    ):
        connection.close()
        raise FeverousParallelIdentitySelectionError(
            "parallel SQLite schema drifted"
        )
    return connection


@dataclass(order=False)
class _ReverseIdentityRank:
    rank: tuple[bytes, str, str, str]
    identity: CorpusIdentity = field(compare=False)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReverseIdentityRank):
            return NotImplemented
        return self.rank > other.rank


@dataclass(order=False)
class _ReversePageRank:
    rank: tuple[bytes, bytes]
    page_id: str = field(compare=False)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReversePageRank):
            return NotImplemented
        return self.rank > other.rank


@dataclass(frozen=True)
class _WorkerResult:
    interval: RowidInterval
    page_count: int
    payload_utf8_bytes: int
    eligible_identity_count: int
    excluded_empty_count: int
    hmac_evaluation_count: int
    logical_page_stream_sha256: str
    eligible_identity_stream_sha256: str
    retained_distractors: tuple[CorpusIdentity, ...]
    retained_gold: tuple[CorpusIdentity, ...]
    sample_page_ids: tuple[str, ...]
    peak_rss_kib: int


def _scan_worker(
    database_path: str,
    binding: BoundDatabase,
    interval: RowidInterval,
    secret: bytes,
    gold_keys: frozenset[str],
    forbidden_alternatives: frozenset[str],
    needed: int,
) -> _WorkerResult:
    path = Path(database_path)
    connection = _open_bound_database(path, binding)
    page_count = 0
    payload_bytes = 0
    identity_count = 0
    empty_count = 0
    hmac_count = 0
    logical_hasher = hashlib.sha256()
    identity_hasher = hashlib.sha256()
    distractor_heap: list[_ReverseIdentityRank] = []
    sample_heap: list[_ReversePageRank] = []
    retained_gold: dict[str, CorpusIdentity] = {}
    try:
        cursor = connection.execute(
            "SELECT rowid, id, data FROM wiki "
            "WHERE rowid BETWEEN ? AND ? ORDER BY rowid",
            (interval.start, interval.end),
        )
        for row in cursor:
            expected_rowid = interval.start + page_count
            if not isinstance(row, tuple) or len(row) != 3:
                raise FeverousParallelIdentitySelectionError(
                    "parallel formal row shape drifted"
                )
            rowid, page_id, raw_page = row
            if (
                type(rowid) is not int
                or rowid != expected_rowid
                or not isinstance(page_id, str)
                or not page_id
                or "\x00" in page_id
                or not isinstance(raw_page, str)
                or "\x00" in raw_page
            ):
                raise FeverousParallelIdentitySelectionError(
                    "parallel formal rowid stream is not strict consecutive"
                )
            page_utf8 = page_id.encode("utf-8", errors="strict")
            raw_utf8 = raw_page.encode("utf-8", errors="strict")
            logical_row = [
                rowid,
                page_id,
                len(raw_utf8),
                hashlib.sha256(raw_utf8).hexdigest(),
            ]
            encoded_logical = acquisition.canonical_json_bytes(logical_row)
            logical_hasher.update(len(encoded_logical).to_bytes(8, "big"))
            logical_hasher.update(encoded_logical)
            sample_rank = (
                hashlib.sha256(
                    b"feverous_p6_e2/identity_compiler_real_sample/v1\x00"
                    + page_utf8
                ).digest(),
                page_utf8,
            )
            sample_entry = _ReversePageRank(sample_rank, page_id)
            if len(sample_heap) < SAMPLE_PAGE_COUNT:
                heapq.heappush(sample_heap, sample_entry)
            elif sample_rank < sample_heap[0].rank:
                heapq.heapreplace(sample_heap, sample_entry)

            enumeration = atomic.enumerate_official_page_atomic_identities(
                page_id, raw_page
            )
            previous_ordinal = -1
            local_ids: set[str] = set()
            for raw_identity in enumeration.identities:
                if (
                    raw_identity.page != page_id
                    or not raw_identity.normalized_target
                    or hashlib.sha256(
                        raw_identity.normalized_target.encode("utf-8")
                    ).hexdigest()
                    != raw_identity.target_sha256
                ):
                    raise FeverousParallelIdentitySelectionError(
                        "parallel enumerated target binding drifted"
                    )
                identity = CorpusIdentity(
                    unit_key=f"{page_id}_{raw_identity.local_id}",
                    page=page_id,
                    local_id=raw_identity.local_id,
                    unit_type=raw_identity.unit_type,
                    official_ordinal=raw_identity.official_ordinal,
                    target_sha256=raw_identity.target_sha256,
                )
                if (
                    identity.official_ordinal <= previous_ordinal
                    or identity.local_id in local_ids
                ):
                    raise FeverousParallelIdentitySelectionError(
                        "parallel identities are not strict official order"
                    )
                previous_ordinal = identity.official_ordinal
                local_ids.add(identity.local_id)
                encoded_identity = acquisition.canonical_json_bytes(
                    identity.commitment_row
                )
                identity_hasher.update(
                    len(encoded_identity).to_bytes(8, "big")
                )
                identity_hasher.update(encoded_identity)
                identity_count += 1
                if identity.unit_key in gold_keys:
                    if identity.unit_key in retained_gold:
                        raise FeverousParallelIdentitySelectionError(
                            "parallel canonical gold identity is duplicated"
                        )
                    retained_gold[identity.unit_key] = identity
                    continue
                if identity.unit_key in forbidden_alternatives:
                    continue
                digest = hmac_digest(
                    secret,
                    "distractor_order",
                    identity.page,
                    identity.local_id,
                )
                hmac_count += 1
                rank = (
                    digest,
                    identity.page,
                    identity.local_id,
                    identity.unit_key,
                )
                entry = _ReverseIdentityRank(rank, identity)
                if len(distractor_heap) < needed:
                    heapq.heappush(distractor_heap, entry)
                elif needed and rank < distractor_heap[0].rank:
                    heapq.heapreplace(distractor_heap, entry)
            page_count += 1
            payload_bytes += len(raw_utf8)
            empty_count += len(enumeration.excluded_empty_local_ids)
        cursor.close()
        if page_count != interval.page_count:
            raise FeverousParallelIdentitySelectionError(
                "parallel formal worker did not exhaust its exact interval"
            )
        if _observed_file_state(path, binding) != _expected_file_state(binding):
            raise FeverousParallelIdentitySelectionError(
                "parallel database changed during worker scan"
            )
    finally:
        connection.close()
    return _WorkerResult(
        interval=interval,
        page_count=page_count,
        payload_utf8_bytes=payload_bytes,
        eligible_identity_count=identity_count,
        excluded_empty_count=empty_count,
        hmac_evaluation_count=hmac_count,
        logical_page_stream_sha256=logical_hasher.hexdigest(),
        eligible_identity_stream_sha256=identity_hasher.hexdigest(),
        retained_distractors=tuple(
            entry.identity
            for entry in sorted(distractor_heap, key=lambda value: value.rank)
        ),
        retained_gold=tuple(
            retained_gold[key]
            for key in sorted(retained_gold, key=lambda value: value.encode("utf-8"))
        ),
        sample_page_ids=tuple(entry.page_id for entry in sample_heap),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


@dataclass(frozen=True)
class _ExactCoverSelection:
    results: tuple[_WorkerResult, ...]
    retained_distractors: tuple[CorpusIdentity, ...]
    retained_gold: tuple[CorpusIdentity, ...]
    qualification_page_ids: tuple[str, ...]
    page_count: int
    payload_utf8_bytes: int
    eligible_identity_count: int
    excluded_empty_count: int
    hmac_evaluation_count: int
    logical_page_stream_sha256: str
    eligible_identity_stream_sha256: str
    chunk_receipt_set_sha256: str
    maximum_worker_peak_rss_kib: int
    sum_worker_peak_rss_kib: int


def _run_exact_cover_selection(
    *,
    database_path: str | os.PathLike[str],
    binding: BoundDatabase,
    secret: bytes,
    gold_keys: frozenset[str],
    forbidden_alternatives: frozenset[str],
    needed: int,
    worker_count: int,
) -> _ExactCoverSelection:
    if (
        not isinstance(binding, BoundDatabase)
        or not isinstance(secret, bytes)
        or len(secret) != 32
        or type(needed) is not int
        or needed < 0
        or needed > CORPUS_UNIT_COUNT
        or type(worker_count) is not int
        or not 1 <= worker_count <= MAXIMUM_WORKER_COUNT
        or gold_keys.intersection(forbidden_alternatives)
    ):
        raise FeverousParallelIdentitySelectionError(
            "parallel selection inputs are invalid"
        )
    path = Path(database_path)
    parent_state = _observed_file_state(path, binding)
    if path.name != binding.basename or parent_state != _expected_file_state(binding):
        raise FeverousParallelIdentitySelectionError(
            "parallel selection parent source binding drifted"
        )
    intervals = partition_exact_cover(binding.row_count, worker_count)
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    _scan_worker,
                    str(path),
                    binding,
                    interval,
                    secret,
                    gold_keys,
                    forbidden_alternatives,
                    needed,
                )
                for interval in intervals
            ]
            unordered = tuple(future.result() for future in futures)
    except Exception as exc:
        raise FeverousParallelIdentitySelectionError(
            "parallel formal selection worker failed closed"
        ) from exc
    results = tuple(sorted(unordered, key=lambda value: value.interval.start))
    if (
        tuple(result.interval for result in results) != intervals
        or _observed_file_state(path, binding) != parent_state
        or any(
            result.page_count != result.interval.page_count
            or not _is_sha256(result.logical_page_stream_sha256)
            or not _is_sha256(result.eligible_identity_stream_sha256)
            or len(result.retained_distractors)
            != min(needed, result.hmac_evaluation_count)
            for result in results
        )
    ):
        raise FeverousParallelIdentitySelectionError(
            "parallel formal result is not an exact exhausted cover"
        )

    local_distractors = [
        identity
        for result in results
        for identity in result.retained_distractors
    ]
    if len({row.unit_key for row in local_distractors}) != len(local_distractors):
        raise FeverousParallelIdentitySelectionError(
            "parallel local heaps contain a duplicate identity"
        )
    ranked: list[tuple[tuple[bytes, str, str, str], CorpusIdentity]] = []
    for identity in local_distractors:
        rank = (
            hmac_digest(
                secret,
                "distractor_order",
                identity.page,
                identity.local_id,
            ),
            identity.page,
            identity.local_id,
            identity.unit_key,
        )
        ranked.append((rank, identity))
    retained_distractors = tuple(
        identity for _rank, identity in sorted(ranked, key=lambda row: row[0])[:needed]
    )
    if len(retained_distractors) != needed:
        raise FeverousParallelIdentitySelectionError(
            "parallel formal distractor capacity is unavailable"
        )

    retained_gold_rows = [
        identity for result in results for identity in result.retained_gold
    ]
    if (
        len({row.unit_key for row in retained_gold_rows})
        != len(retained_gold_rows)
        or {row.unit_key for row in retained_gold_rows} != set(gold_keys)
    ):
        raise FeverousParallelIdentitySelectionError(
            "parallel formal canonical gold coverage drifted"
        )
    retained_gold = tuple(
        sorted(retained_gold_rows, key=lambda row: row.unit_key.encode("utf-8"))
    )

    sample_candidates = {
        page_id
        for result in results
        for page_id in result.sample_page_ids
    }
    qualification_page_ids = tuple(
        sorted(
            sorted(
                sample_candidates,
                key=lambda page_id: (
                    hashlib.sha256(
                        b"feverous_p6_e2/identity_compiler_real_sample/v1\x00"
                        + page_id.encode("utf-8")
                    ).digest(),
                    page_id.encode("utf-8"),
                ),
            )[: min(SAMPLE_PAGE_COUNT, binding.row_count)],
            key=lambda page_id: page_id.encode("utf-8"),
        )
    )
    if len(qualification_page_ids) != min(SAMPLE_PAGE_COUNT, binding.row_count):
        raise FeverousParallelIdentitySelectionError(
            "parallel identity/compiler sample capacity drifted"
        )

    chunk_rows = [
        {
            "end_rowid": result.interval.end,
            "eligible_atomic_identity_count": result.eligible_identity_count,
            "eligible_atomic_identity_stream_sha256": (
                result.eligible_identity_stream_sha256
            ),
            "excluded_empty_atomic_identity_count": result.excluded_empty_count,
            "hmac_evaluation_count": result.hmac_evaluation_count,
            "logical_page_stream_sha256": result.logical_page_stream_sha256,
            "observed_page_count": result.page_count,
            "payload_utf8_bytes": result.payload_utf8_bytes,
            "start_rowid": result.interval.start,
        }
        for result in results
    ]
    return _ExactCoverSelection(
        results=results,
        retained_distractors=retained_distractors,
        retained_gold=retained_gold,
        qualification_page_ids=qualification_page_ids,
        page_count=sum(result.page_count for result in results),
        payload_utf8_bytes=sum(result.payload_utf8_bytes for result in results),
        eligible_identity_count=sum(
            result.eligible_identity_count for result in results
        ),
        excluded_empty_count=sum(result.excluded_empty_count for result in results),
        hmac_evaluation_count=sum(
            result.hmac_evaluation_count for result in results
        ),
        logical_page_stream_sha256=_stable_hash(
            [
                [
                    result.interval.start,
                    result.interval.end,
                    result.page_count,
                    result.logical_page_stream_sha256,
                ]
                for result in results
            ]
        ),
        eligible_identity_stream_sha256=_stable_hash(
            [
                [
                    result.interval.start,
                    result.interval.end,
                    result.eligible_identity_count,
                    result.eligible_identity_stream_sha256,
                ]
                for result in results
            ]
        ),
        chunk_receipt_set_sha256=_stable_hash(chunk_rows),
        maximum_worker_peak_rss_kib=max(result.peak_rss_kib for result in results),
        sum_worker_peak_rss_kib=sum(result.peak_rss_kib for result in results),
    )


@dataclass(frozen=True)
class ParallelSelectionOutcome:
    plan: CorpusSelectionPlan
    database_receipt: Mapping[str, Any]
    receipt: Mapping[str, Any]


def plan_fixed_corpus_parallel(
    *,
    database_path: str | os.PathLike[str],
    database_binding: BoundDatabase,
    blocks: Mapping[str, Sequence[acquisition.AssignedRecord]],
    secret: bytes,
    identity_full_compile_equivalence_qualification_sha256: str,
) -> ParallelSelectionOutcome:
    """Form the formal bounded plan with the frozen eight-worker topology."""

    if not _is_sha256(
        identity_full_compile_equivalence_qualification_sha256
    ):
        raise FeverousParallelIdentitySelectionError(
            "identity/full-compiler qualification is absent"
        )
    selected = acquisition._validated_selected_rows(blocks)
    gold_keys = frozenset(
        key for row in selected for key in row.canonical_gold_keys
    )
    official_keys = frozenset(
        key
        for row in selected
        for key in row.record.all_official_evidence_keys
    )
    forbidden = official_keys.difference(gold_keys)
    needed = CORPUS_UNIT_COUNT - len(gold_keys)
    if needed < 0:
        raise FeverousParallelIdentitySelectionError(
            "fixed corpus capacity is unavailable"
        )
    selection = _run_exact_cover_selection(
        database_path=database_path,
        binding=database_binding,
        secret=secret,
        gold_keys=gold_keys,
        forbidden_alternatives=forbidden,
        needed=needed,
        worker_count=FROZEN_WORKER_COUNT,
    )
    if selection.page_count != database_binding.row_count:
        raise FeverousParallelIdentitySelectionError(
            "parallel selection did not exhaust the database universe"
        )

    database_body: dict[str, Any] = {
        "schema": "feverous_p6_e2_formal_source_v1_database_page_stream_receipt",
        "version": "feverous_p6_e2_formal_source_v1",
        "status": "complete_database_page_stream_exhausted",
        "source_split": "TRAIN",
        "source_spec_sha256": database_binding.source_spec_sha256,
        "source_binding_sha256": database_binding.source_binding_sha256,
        "formal_source_opener_source_sha256": (
            database_binding.formal_source_opener_source_sha256
        ),
        "formal_source": database_binding.formal_source,
        "database_basename": database_binding.basename,
        "database_size_bytes": database_binding.size_bytes,
        "database_file_sha256": database_binding.declared_sha256,
        "database_schema_sha256": _stable_hash(database_binding.schema),
        "expected_database_row_count": database_binding.row_count,
        "observed_database_row_count": selection.page_count,
        "page_order": (
            "strict_consecutive_rowid_physical_order_8_worker_exact_cover"
        ),
        "logical_page_stream_sha256": selection.logical_page_stream_sha256,
        "logical_page_stream_commitment_topology": (
            "ordered_exact_cover_chunk_sha256_v1"
        ),
        "parallel_worker_count": FROZEN_WORKER_COUNT,
        "parallel_chunk_receipt_set_sha256": selection.chunk_receipt_set_sha256,
        "observed_payload_utf8_bytes": selection.payload_utf8_bytes,
        "stream_fully_exhausted": True,
        "maximum_buffered_database_rows": FROZEN_WORKER_COUNT,
        "all_page_ids_or_pages_materialized": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    database_receipt = acquisition.self_hashed(
        database_body, "database_page_stream_receipt_sha256"
    )

    parallel_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity_body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "version": VERSION,
        "status": (
            "complete_atomic_identity_universe_exhausted_with_bounded_"
            "private_selection"
        ),
        "source_split": "TRAIN",
        "source_binding_sha256": database_binding.source_binding_sha256,
        "formal_source": database_binding.formal_source,
        "source_spec_sha256": database_binding.source_spec_sha256,
        "formal_source_opener_source_sha256": (
            database_binding.formal_source_opener_source_sha256
        ),
        "database_page_stream_receipt_sha256": database_receipt[
            "database_page_stream_receipt_sha256"
        ],
        "database_size_bytes": database_binding.size_bytes,
        "database_file_sha256": database_binding.declared_sha256,
        "expected_database_row_count": database_binding.row_count,
        "observed_database_row_count": selection.page_count,
        "logical_page_stream_sha256": selection.logical_page_stream_sha256,
        "atomic_compiler_version": atomic.VERSION,
        "identity_enumerator_version": atomic.IDENTITY_ENUMERATOR_VERSION,
        "atomic_compiler_source_sha256": _source_sha256(atomic),
        "identity_enumerator_source_sha256": _source_sha256(atomic),
        "source_adapter_source_sha256": _source_sha256(source_adapter),
        "acquisition_source_sha256": _source_sha256(acquisition),
        "parallel_selection_source_sha256": parallel_source_sha256,
        "identity_full_compile_equivalence_qualification_sha256": (
            identity_full_compile_equivalence_qualification_sha256
        ),
        "real_identity_compiler_sample_policy": (
            "lowest_sha256_domain_page_id_then_binary_page_id"
        ),
        "real_identity_compiler_sample_page_count": len(
            selection.qualification_page_ids
        ),
        "real_identity_compiler_sample_page_set_sha256": _stable_hash(
            list(selection.qualification_page_ids)
        ),
        "stream_fully_exhausted": True,
        "adapted_page_count": selection.page_count,
        "eligible_atomic_identity_count": selection.eligible_identity_count,
        "excluded_empty_atomic_identity_count": selection.excluded_empty_count,
        "eligible_atomic_identity_stream_sha256": (
            selection.eligible_identity_stream_sha256
        ),
        "identity_stream_commitment_topology": (
            "ordered_exact_cover_chunk_sha256_v1"
        ),
        "parallel_worker_count": FROZEN_WORKER_COUNT,
        "parallel_chunk_receipt_set_sha256": selection.chunk_receipt_set_sha256,
        "production_private_hmac_evaluation_count": (
            selection.hmac_evaluation_count
        ),
        "maximum_worker_peak_rss_kib": selection.maximum_worker_peak_rss_kib,
        "sum_worker_peak_rss_kib": selection.sum_worker_peak_rss_kib,
        "maximum_resident_enumerated_pages": FROZEN_WORKER_COUNT,
        "maximum_resident_distractor_identities": (
            FROZEN_WORKER_COUNT * needed
        ),
        "all_identities_or_pages_materialized": False,
        "full_atomic_text_or_sidecar_linearized": False,
        "bounded_private_selection_performed_concurrently": True,
        "formal_secret_logged_persisted_or_exposed_on_argv": False,
        "formal_secret_serialized_only_in_spawn_pipe_transit": True,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    identity_receipt = acquisition.self_hashed(
        identity_body, "corpus_identity_stream_receipt_sha256"
    )

    retained = [*selection.retained_gold, *selection.retained_distractors]
    if (
        len(retained) != CORPUS_UNIT_COUNT
        or len({row.unit_key for row in retained}) != CORPUS_UNIT_COUNT
    ):
        raise FeverousParallelIdentitySelectionError(
            "parallel retained identity plan shape drifted"
        )
    retained.sort(key=lambda row: row.unit_key.encode("utf-8"))
    selected_pages = tuple(
        sorted({row.page for row in retained}, key=lambda value: value.encode("utf-8"))
    )
    plan_body: dict[str, Any] = {
        "schema": f"{acquisition.VERSION}_corpus_identity_plan",
        "version": acquisition.VERSION,
        "status": "complete_universe_scanned_bounded_identity_plan_formed",
        "identity_stream_receipt_sha256": identity_receipt[
            "corpus_identity_stream_receipt_sha256"
        ],
        "formal_source_bound": database_binding.formal_source,
        "complete_identity_scan_count": selection.eligible_identity_count,
        "complete_identity_stream_sha256": (
            selection.eligible_identity_stream_sha256
        ),
        "identity_stream_commitment_verified_from_exhausted_adapter_receipt": False,
        "identity_stream_commitment_verified_from_parallel_exact_cover_receipt": True,
        "selected_atomic_identity_count": len(retained),
        "selected_page_count": len(selected_pages),
        "real_identity_compiler_qualification_page_count": len(
            selection.qualification_page_ids
        ),
        "real_identity_compiler_qualification_page_set_sha256": _stable_hash(
            list(selection.qualification_page_ids)
        ),
        "unique_canonical_gold_identity_count": len(gold_keys),
        "distractor_identity_count": needed,
        "known_noncanonical_official_evidence_excluded": len(forbidden),
        "maximum_retained_distractor_identities": CORPUS_UNIT_COUNT,
        "parallel_worker_count": FROZEN_WORKER_COUNT,
        "parallel_chunk_receipt_set_sha256": selection.chunk_receipt_set_sha256,
        "all_identity_keys_or_page_ids_serialized": False,
        "full_atomic_text_or_sidecar_linearized_during_scan": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    plan_receipt = acquisition.self_hashed(
        plan_body, "corpus_identity_plan_sha256"
    )
    plan = CorpusSelectionPlan(
        identities=tuple(retained),
        selected_page_ids=selected_pages,
        qualification_page_ids=selection.qualification_page_ids,
        identity_stream_receipt=MappingProxyType(identity_receipt),
        receipt=MappingProxyType(plan_receipt),
    )
    receipt_body: dict[str, Any] = {
        "schema": f"{VERSION}_formation_receipt",
        "version": VERSION,
        "status": "formal_parallel_identity_plan_formed_from_complete_exact_cover",
        "database_page_stream_receipt_sha256": database_receipt[
            "database_page_stream_receipt_sha256"
        ],
        "corpus_identity_stream_receipt_sha256": identity_receipt[
            "corpus_identity_stream_receipt_sha256"
        ],
        "corpus_identity_plan_sha256": plan.plan_sha256,
        "parallel_worker_count": FROZEN_WORKER_COUNT,
        "observed_database_row_count": selection.page_count,
        "eligible_atomic_identity_count": selection.eligible_identity_count,
        "production_private_hmac_evaluation_count": selection.hmac_evaluation_count,
        "private_secret_logged_persisted_or_exposed_on_argv": False,
        "private_secret_serialized_only_in_spawn_pipe_transit": True,
        "full_atomic_text_or_sidecar_linearized": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    receipt = acquisition.self_hashed(
        receipt_body, "parallel_identity_selection_receipt_sha256"
    )
    return ParallelSelectionOutcome(
        plan=plan,
        database_receipt=MappingProxyType(database_receipt),
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "BoundDatabase",
    "FROZEN_WORKER_COUNT",
    "FeverousParallelIdentitySelectionError",
    "ParallelSelectionOutcome",
    "RowidInterval",
    "partition_exact_cover",
    "plan_fixed_corpus_parallel",
]

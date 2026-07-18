"""Bounded multiprocessing diagnostic for FEVEROUS identity acquisition.

This module measures the actual lightweight identity enumerator, validated
``CorpusIdentity`` construction, production-domain HMAC ranking, and a bounded
bottom-k heap.  It is deliberately nonformal: the ranking key is a fixed,
public diagnostic constant; no annotation, claim, label, cohort, outcome, or
formal selection secret is accepted or opened.

Each worker owns one exact contiguous physical-rowid interval and opens its
own immutable, query-only SQLite connection.  A worker must exhaust every
row in its interval in strict order.  The coordinator requires an exact,
gap-free cover before it merges the local bottom-k heaps.  The merge is exact:
the global bottom-k of disjoint chunks is contained in the union of each
chunk's local bottom-k.  Page leaf digests are merged in physical order only
in memory, so the aggregate stream commitment is independent of chunking;
neither page digests nor retained unit ids appear in the public receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import sqlite3
import time
from typing import Any

from assumption_agent.benchmarks import feverous_atomic_corpus_v1 as atomic
from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks.feverous_p6_e2_acquisition_v1 import (
    CORPUS_UNIT_COUNT,
    CorpusIdentity,
    canonical_json_bytes,
    hmac_digest,
)
from assumption_agent.benchmarks import (
    feverous_p6_e2_identity_performance_diagnostic_v1 as serial_diagnostic,
)
from assumption_agent.benchmarks.feverous_p6_e2_identity_performance_diagnostic_v1 import (
    DiagnosticDatabaseSpec,
    FROZEN_DATABASE_SPEC,
)
from assumption_agent.benchmarks.feverous_p6_e2_source_adapter_v1 import (
    DESIGN_SHA256,
    WIKIPEDIA_QUALIFICATION_SHA256,
)


VERSION = "feverous_p6_e2_identity_parallel_performance_diagnostic_v1"
RECEIPT_SCHEMA = f"{VERSION}_aggregate_receipt"
FROZEN_WORKER_COUNTS = (1, 4, 8, 16)
FROZEN_FIRST_ROWID = 100_001
FROZEN_PAGES_PER_CONFIGURATION = 20_000
FROZEN_HEAP_CAPACITY = CORPUS_UNIT_COUNT
MAXIMUM_DIAGNOSTIC_WORKERS = 32

# This constant is intentionally public and reproducible.  It must never be
# substituted for, derived from, or compared with a formal acquisition secret.
PUBLIC_DIAGNOSTIC_RANK_KEY = hashlib.sha256(
    b"feverous-p6-e2/public-multiprocessing-performance-key/v1"
).digest()
PUBLIC_DIAGNOSTIC_RANK_KEY_SHA256 = hashlib.sha256(
    PUBLIC_DIAGNOSTIC_RANK_KEY
).hexdigest()


class FeverousIdentityParallelPerformanceDiagnosticError(RuntimeError):
    """The bounded parallel diagnostic or its aggregate receipt drifted."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _hash_digest_sequence(values: Sequence[bytes]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        if not isinstance(value, bytes) or len(value) != 32:
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "page digest sequence is malformed"
            )
        hasher.update(len(value).to_bytes(8, "big"))
        hasher.update(value)
    return hasher.hexdigest()


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
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "rowid interval is invalid"
            )

    @property
    def page_count(self) -> int:
        return self.end - self.start + 1


def partition_rowid_interval(
    interval: RowidInterval, part_count: int
) -> tuple[RowidInterval, ...]:
    """Return a deterministic balanced, exact, gap-free interval cover."""

    if (
        not isinstance(interval, RowidInterval)
        or type(part_count) is not int
        or not 1 <= part_count <= min(interval.page_count, MAXIMUM_DIAGNOSTIC_WORKERS)
    ):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel interval partition is invalid"
        )
    quotient, remainder = divmod(interval.page_count, part_count)
    output: list[RowidInterval] = []
    cursor = interval.start
    for ordinal in range(part_count):
        size = quotient + int(ordinal < remainder)
        output.append(RowidInterval(cursor, cursor + size - 1))
        cursor += size
    if cursor != interval.end + 1:
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel interval cover drifted"
        )
    return tuple(output)


@dataclass(frozen=True)
class _RankCandidate:
    digest: bytes
    page: str
    local_id: str
    unit_key: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.digest, bytes)
            or len(self.digest) != 32
            or not isinstance(self.page, str)
            or not self.page
            or "\x00" in self.page
            or not isinstance(self.local_id, str)
            or not self.local_id
            or "\x00" in self.local_id
            or self.unit_key != f"{self.page}_{self.local_id}"
        ):
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "diagnostic rank candidate is malformed"
            )

    @property
    def rank(self) -> tuple[bytes, str, str, str]:
        return (self.digest, self.page, self.local_id, self.unit_key)


@dataclass(order=False)
class _ReverseRank:
    candidate: _RankCandidate = field(compare=False)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReverseRank):
            return NotImplemented
        return self.candidate.rank > other.candidate.rank


@dataclass(frozen=True)
class _ChunkResult:
    interval: RowidInterval
    page_count: int
    payload_utf8_bytes: int
    eligible_identity_count: int
    excluded_empty_count: int
    hmac_evaluation_count: int
    page_leaf_digests: tuple[bytes, ...]
    retained_candidates: tuple[_RankCandidate, ...]
    worker_wall_seconds: float
    worker_peak_rss_kib: int


@dataclass(frozen=True)
class ConfigurationMeasurement:
    """Internal result; candidate ids are never copied into the receipt."""

    worker_count: int
    interval: RowidInterval
    chunk_intervals: tuple[RowidInterval, ...]
    page_count: int
    payload_utf8_bytes: int
    eligible_identity_count: int
    excluded_empty_count: int
    hmac_evaluation_count: int
    ordered_page_stream_sha256: str
    retained_bottom_k_sha256: str
    retained_bottom_k_count: int
    retained_candidates: tuple[_RankCandidate, ...]
    chunk_receipt_set_sha256: str
    wall_seconds: float
    maximum_worker_peak_rss_kib: int
    sum_worker_peak_rss_kib: int

    @property
    def pages_per_second(self) -> float:
        return self.page_count / self.wall_seconds

    @property
    def identities_per_second(self) -> float:
        return self.eligible_identity_count / self.wall_seconds

    @property
    def payload_mib_per_second(self) -> float:
        return self.payload_utf8_bytes / (1024 * 1024) / self.wall_seconds


def _scan_chunk(
    database_path: str,
    spec: DiagnosticDatabaseSpec,
    interval: RowidInterval,
    heap_capacity: int,
) -> _ChunkResult:
    """Worker entry point: exactly exhaust one immutable rowid interval."""

    path = Path(database_path)
    connection, source_state = serial_diagnostic._open_bound_database(path, spec)
    page_count = 0
    payload_utf8_bytes = 0
    identity_count = 0
    empty_count = 0
    hmac_count = 0
    leaves: list[bytes] = []
    heap: list[_ReverseRank] = []
    started = time.perf_counter()
    try:
        cursor = connection.execute(
            "SELECT rowid, id, data FROM wiki "
            "WHERE rowid BETWEEN ? AND ? ORDER BY rowid",
            (interval.start, interval.end),
        )
        for row in cursor:
            expected_rowid = interval.start + page_count
            if not isinstance(row, tuple) or len(row) != 3:
                raise FeverousIdentityParallelPerformanceDiagnosticError(
                    "parallel chunk row shape drifted"
                )
            rowid, page_id, raw_page = row
            if (
                type(rowid) is not int
                or rowid != expected_rowid
                or not isinstance(page_id, str)
                or not page_id
                or "\x00" in page_id
                or not isinstance(raw_page, str)
            ):
                raise FeverousIdentityParallelPerformanceDiagnosticError(
                    "parallel chunk is not strict consecutive physical rowid"
                )
            enumeration = atomic.enumerate_official_page_atomic_identities(
                page_id, raw_page
            )
            for raw_identity in enumeration.identities:
                identity = CorpusIdentity(
                    unit_key=f"{page_id}_{raw_identity.local_id}",
                    page=page_id,
                    local_id=raw_identity.local_id,
                    unit_type=raw_identity.unit_type,
                    official_ordinal=raw_identity.official_ordinal,
                    target_sha256=raw_identity.target_sha256,
                )
                digest = hmac_digest(
                    PUBLIC_DIAGNOSTIC_RANK_KEY,
                    "distractor_order",
                    identity.page,
                    identity.local_id,
                )
                candidate = _RankCandidate(
                    digest=digest,
                    page=identity.page,
                    local_id=identity.local_id,
                    unit_key=identity.unit_key,
                )
                entry = _ReverseRank(candidate)
                if len(heap) < heap_capacity:
                    heapq.heappush(heap, entry)
                elif candidate.rank < heap[0].candidate.rank:
                    heapq.heapreplace(heap, entry)
                identity_count += 1
                hmac_count += 1

            payload = raw_page.encode("utf-8", errors="strict")
            leaf_payload = [
                rowid,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                enumeration.commitment(),
            ]
            leaves.append(hashlib.sha256(canonical_json_bytes(leaf_payload)).digest())
            page_count += 1
            payload_utf8_bytes += len(payload)
            empty_count += len(enumeration.excluded_empty_local_ids)
        cursor.close()
        if page_count != interval.page_count:
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "parallel chunk ended before exact interval exhaustion"
            )
        if serial_diagnostic._file_state(path, spec) != source_state:
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "diagnostic database changed during parallel chunk"
            )
    finally:
        connection.close()
    elapsed = time.perf_counter() - started
    return _ChunkResult(
        interval=interval,
        page_count=page_count,
        payload_utf8_bytes=payload_utf8_bytes,
        eligible_identity_count=identity_count,
        excluded_empty_count=empty_count,
        hmac_evaluation_count=hmac_count,
        page_leaf_digests=tuple(leaves),
        retained_candidates=tuple(
            entry.candidate for entry in sorted(heap, key=lambda row: row.candidate.rank)
        ),
        worker_wall_seconds=elapsed,
        worker_peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


def _validate_exact_chunk_cover(
    interval: RowidInterval,
    expected_chunks: Sequence[RowidInterval],
    results: Sequence[_ChunkResult],
) -> tuple[_ChunkResult, ...]:
    ordered = tuple(sorted(results, key=lambda row: row.interval.start))
    if (
        tuple(row.interval for row in ordered) != tuple(expected_chunks)
        or not ordered
        or ordered[0].interval.start != interval.start
        or ordered[-1].interval.end != interval.end
    ):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel results do not form the frozen exact interval cover"
        )
    for result in ordered:
        if (
            result.page_count != result.interval.page_count
            or len(result.page_leaf_digests) != result.page_count
            or result.hmac_evaluation_count != result.eligible_identity_count
            or len(result.retained_candidates) > FROZEN_HEAP_CAPACITY
        ):
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "parallel chunk aggregate is malformed"
            )
    return ordered


def _merge_bottom_k(
    chunks: Sequence[_ChunkResult], heap_capacity: int
) -> tuple[_RankCandidate, ...]:
    candidates = [
        candidate for chunk in chunks for candidate in chunk.retained_candidates
    ]
    for candidate in candidates:
        expected = hmac_digest(
            PUBLIC_DIAGNOSTIC_RANK_KEY,
            "distractor_order",
            candidate.page,
            candidate.local_id,
        )
        if candidate.digest != expected:
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "worker rank disagrees with the public diagnostic HMAC"
            )
    if len({candidate.unit_key for candidate in candidates}) != len(candidates):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel chunk heaps contain a duplicate identity"
        )
    return tuple(sorted(candidates, key=lambda row: row.rank)[:heap_capacity])


def scan_parallel_configuration(
    database_path: str | os.PathLike[str],
    *,
    spec: DiagnosticDatabaseSpec,
    interval: RowidInterval,
    worker_count: int,
    heap_capacity: int,
) -> ConfigurationMeasurement:
    """Measure one exact interval with one deterministic process topology."""

    if (
        not isinstance(spec, DiagnosticDatabaseSpec)
        or not isinstance(interval, RowidInterval)
        or type(worker_count) is not int
        or not 1 <= worker_count <= min(interval.page_count, MAXIMUM_DIAGNOSTIC_WORKERS)
        or type(heap_capacity) is not int
        or not 1 <= heap_capacity <= FROZEN_HEAP_CAPACITY
    ):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel configuration is invalid"
        )
    path = Path(database_path)
    if path.name != spec.basename or interval.end > spec.row_count:
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel configuration is outside the database binding"
        )
    chunks = partition_rowid_interval(interval, worker_count)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    _scan_chunk,
                    str(path),
                    spec,
                    chunk,
                    heap_capacity,
                )
                for chunk in chunks
            ]
            results = tuple(future.result() for future in futures)
    except Exception as exc:
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel worker failed closed"
        ) from exc
    wall_seconds = time.perf_counter() - started
    ordered = _validate_exact_chunk_cover(interval, chunks, results)
    # The local-capacity check depends on the caller's bounded diagnostic cap.
    if any(
        len(result.retained_candidates)
        != min(result.eligible_identity_count, heap_capacity)
        for result in ordered
    ):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel worker retained heap has the wrong capacity"
        )
    merged = _merge_bottom_k(ordered, heap_capacity)
    page_leaves = tuple(
        digest for result in ordered for digest in result.page_leaf_digests
    )
    retained_sha256 = _stable_hash(
        [
            [
                candidate.digest.hex(),
                candidate.page,
                candidate.local_id,
                candidate.unit_key,
            ]
            for candidate in merged
        ]
    )
    chunk_receipts = [
        {
            "end_rowid": result.interval.end,
            "eligible_identity_count": result.eligible_identity_count,
            "excluded_empty_count": result.excluded_empty_count,
            "hmac_evaluation_count": result.hmac_evaluation_count,
            "ordered_page_leaf_sha256": _hash_digest_sequence(
                result.page_leaf_digests
            ),
            "page_count": result.page_count,
            "payload_utf8_bytes": result.payload_utf8_bytes,
            "retained_bottom_k_sha256": _stable_hash(
                [
                    [candidate.digest.hex(), candidate.unit_key]
                    for candidate in result.retained_candidates
                ]
            ),
            "start_rowid": result.interval.start,
        }
        for result in ordered
    ]
    return ConfigurationMeasurement(
        worker_count=worker_count,
        interval=interval,
        chunk_intervals=chunks,
        page_count=sum(result.page_count for result in ordered),
        payload_utf8_bytes=sum(result.payload_utf8_bytes for result in ordered),
        eligible_identity_count=sum(
            result.eligible_identity_count for result in ordered
        ),
        excluded_empty_count=sum(result.excluded_empty_count for result in ordered),
        hmac_evaluation_count=sum(
            result.hmac_evaluation_count for result in ordered
        ),
        ordered_page_stream_sha256=_hash_digest_sequence(page_leaves),
        retained_bottom_k_sha256=retained_sha256,
        retained_bottom_k_count=len(merged),
        retained_candidates=merged,
        chunk_receipt_set_sha256=_stable_hash(chunk_receipts),
        wall_seconds=wall_seconds,
        maximum_worker_peak_rss_kib=max(
            result.worker_peak_rss_kib for result in ordered
        ),
        sum_worker_peak_rss_kib=sum(
            result.worker_peak_rss_kib for result in ordered
        ),
    )


def run_parallel_performance_diagnostic(
    database_path: str | os.PathLike[str],
    *,
    spec: DiagnosticDatabaseSpec = FROZEN_DATABASE_SPEC,
    worker_counts: Sequence[int] = FROZEN_WORKER_COUNTS,
    first_rowid: int = FROZEN_FIRST_ROWID,
    pages_per_configuration: int = FROZEN_PAGES_PER_CONFIGURATION,
    heap_capacity: int = FROZEN_HEAP_CAPACITY,
) -> Mapping[str, Any]:
    """Run disjoint real-source intervals and return an aggregate-only receipt."""

    counts = tuple(worker_counts)
    if (
        not isinstance(spec, DiagnosticDatabaseSpec)
        or not counts
        or any(type(value) is not int for value in counts)
        or len(set(counts)) != len(counts)
        or any(not 1 <= value <= MAXIMUM_DIAGNOSTIC_WORKERS for value in counts)
        or type(first_rowid) is not int
        or first_rowid < 1
        or type(pages_per_configuration) is not int
        or pages_per_configuration < max(counts)
        or type(heap_capacity) is not int
        or not 1 <= heap_capacity <= FROZEN_HEAP_CAPACITY
        or first_rowid + pages_per_configuration * len(counts) - 1 > spec.row_count
    ):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel performance diagnostic bounds are invalid"
        )
    path = Path(database_path)
    if path.name != spec.basename:
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "diagnostic database basename differs from binding"
        )
    # Validate once in the coordinator before any process is created.  Every
    # child repeats the same exact file/schema binding.
    connection, source_state = serial_diagnostic._open_bound_database(path, spec)
    connection.close()

    measurements: list[ConfigurationMeasurement] = []
    for ordinal, worker_count in enumerate(counts):
        start = first_rowid + ordinal * pages_per_configuration
        interval = RowidInterval(start, start + pages_per_configuration - 1)
        measurements.append(
            scan_parallel_configuration(
                path,
                spec=spec,
                interval=interval,
                worker_count=worker_count,
                heap_capacity=heap_capacity,
            )
        )
    if serial_diagnostic._file_state(path, spec) != source_state:
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "diagnostic database changed across configurations"
        )

    baseline = measurements[0].pages_per_second
    best = max(
        measurements,
        key=lambda value: (value.pages_per_second, -value.worker_count),
    )
    configuration_rows: list[dict[str, Any]] = []
    for measurement in measurements:
        projected_seconds = spec.row_count / measurement.pages_per_second
        configuration_rows.append(
            {
                "worker_count": measurement.worker_count,
                "start_rowid": measurement.interval.start,
                "end_rowid": measurement.interval.end,
                "chunk_count": len(measurement.chunk_intervals),
                "exact_gap_free_nonoverlap_cover_verified": True,
                "observed_page_count": measurement.page_count,
                "observed_payload_utf8_bytes": measurement.payload_utf8_bytes,
                "observed_eligible_identity_count": (
                    measurement.eligible_identity_count
                ),
                "observed_excluded_empty_count": measurement.excluded_empty_count,
                "public_hmac_evaluation_count": measurement.hmac_evaluation_count,
                "retained_bottom_k_count": measurement.retained_bottom_k_count,
                "ordered_page_stream_sha256": (
                    measurement.ordered_page_stream_sha256
                ),
                "retained_bottom_k_sha256": measurement.retained_bottom_k_sha256,
                "chunk_receipt_set_sha256": measurement.chunk_receipt_set_sha256,
                "wall_seconds": round(measurement.wall_seconds, 6),
                "pages_per_second": round(measurement.pages_per_second, 6),
                "eligible_identities_per_second": round(
                    measurement.identities_per_second, 6
                ),
                "payload_mib_per_second": round(
                    measurement.payload_mib_per_second, 6
                ),
                "raw_page_throughput_relative_to_first_configuration": round(
                    measurement.pages_per_second / baseline, 6
                ),
                "projected_complete_identity_plus_public_hmac_seconds": round(
                    projected_seconds, 6
                ),
                "projected_complete_identity_plus_public_hmac_hours": round(
                    projected_seconds / 3600, 6
                ),
                "maximum_worker_peak_rss_kib": (
                    measurement.maximum_worker_peak_rss_kib
                ),
                "sum_worker_peak_rss_kib": measurement.sum_worker_peak_rss_kib,
            }
        )

    best_speedup = best.pages_per_second / baseline
    if best_speedup < 1.10:
        conclusion = "multiprocessing_not_materially_faster_on_disjoint_intervals"
    elif best.worker_count == max(counts):
        conclusion = "multiprocessing_improves_throughput_through_maximum_tested_workers"
    else:
        conclusion = "multiprocessing_improves_throughput_but_saturates_before_maximum"
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "passed_nonformal_parallel_identity_public_hmac_diagnostic",
        "formal_valid": False,
        "implementation_performance_only": True,
        "source_split_or_annotation_opened": False,
        "database_spec_sha256": spec.spec_sha256,
        "design_sha256": DESIGN_SHA256,
        "wikipedia_qualification_sha256": WIKIPEDIA_QUALIFICATION_SHA256,
        "database_basename": spec.basename,
        "database_size_bytes": spec.size_bytes,
        "database_declared_sha256": spec.declared_sha256,
        "database_sha256_recomputed_this_diagnostic": False,
        "database_expected_row_count": spec.row_count,
        "database_schema_sha256": _stable_hash(spec.schema),
        "database_open_mode_per_worker": "mode=ro&immutable=1_query_only",
        "process_start_method": "spawn",
        "configuration_execution_order": list(counts),
        "configuration_interval_policy": (
            "equal_size_consecutive_mutually_disjoint_physical_rowid_intervals"
        ),
        "configuration_intervals_mutually_disjoint": True,
        "cross_configuration_cache_overlap_by_rowid": False,
        "cross_configuration_density_control": (
            "equal_page_counts; pages_identities_and_payload_bytes_reported_separately"
        ),
        "first_rowid": first_rowid,
        "pages_per_configuration": pages_per_configuration,
        "heap_capacity": heap_capacity,
        "public_diagnostic_rank_key": True,
        "public_diagnostic_rank_key_sha256": PUBLIC_DIAGNOSTIC_RANK_KEY_SHA256,
        "hmac_domain_and_purpose": (
            "production_hmac_sha256_domain_with_distractor_order_purpose"
        ),
        "public_diagnostic_hmac_heap_included": True,
        "private_formal_hmac_heap_included": False,
        "all_eligible_identities_ranked_per_interval": True,
        "deterministic_local_bottom_k_then_exact_global_merge": True,
        "page_stream_commitment_independent_of_chunk_partition": True,
        "configurations": configuration_rows,
        "throughput_best_worker_count": best.worker_count,
        "throughput_best_raw_speedup_over_first_configuration": round(
            best_speedup, 6
        ),
        "throughput_conclusion": conclusion,
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "atomic_compiler_and_enumerator_source_sha256": (
            serial_diagnostic._source_sha256(atomic)
        ),
        "acquisition_identity_source_sha256": (
            serial_diagnostic._source_sha256(acquisition)
        ),
        "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "platform_system": platform.system(),
        "logical_cpu_count": os.cpu_count(),
        "page_ids_payloads_atomic_ids_or_targets_serialized": False,
        "per_page_digests_serialized": False,
        "bottom_k_unit_ids_serialized": False,
        "corpus_or_cohort_retained_after_diagnostic": False,
        "formal_selection_secret_created_derived_or_accessed": False,
        "claim_label_evidence_family_or_outcome_accessed": False,
        "development_or_test_source_accessed": False,
        "model_action_or_online_evaluator_calls": 0,
    }
    body["diagnostic_receipt_sha256"] = _stable_hash(body)
    return body


def verify_parallel_performance_diagnostic_receipt(
    receipt: Mapping[str, Any],
) -> str:
    if not isinstance(receipt, Mapping):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel diagnostic receipt must be an object"
        )
    body = dict(receipt)
    declared = body.pop("diagnostic_receipt_sha256", None)
    configurations = receipt.get("configurations")
    sha_fields = (
        "database_spec_sha256",
        "design_sha256",
        "wikipedia_qualification_sha256",
        "database_declared_sha256",
        "database_schema_sha256",
        "public_diagnostic_rank_key_sha256",
        "diagnostic_source_sha256",
        "atomic_compiler_and_enumerator_source_sha256",
        "acquisition_identity_source_sha256",
    )
    if (
        not _is_sha256(declared)
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("version") != VERSION
        or receipt.get("status")
        != "passed_nonformal_parallel_identity_public_hmac_diagnostic"
        or receipt.get("formal_valid") is not False
        or receipt.get("implementation_performance_only") is not True
        or receipt.get("source_split_or_annotation_opened") is not False
        or receipt.get("database_sha256_recomputed_this_diagnostic") is not False
        or receipt.get("public_diagnostic_rank_key") is not True
        or receipt.get("public_diagnostic_rank_key_sha256")
        != PUBLIC_DIAGNOSTIC_RANK_KEY_SHA256
        or receipt.get("public_diagnostic_hmac_heap_included") is not True
        or receipt.get("private_formal_hmac_heap_included") is not False
        or receipt.get("all_eligible_identities_ranked_per_interval") is not True
        or receipt.get("deterministic_local_bottom_k_then_exact_global_merge")
        is not True
        or receipt.get("configuration_intervals_mutually_disjoint") is not True
        or receipt.get("page_ids_payloads_atomic_ids_or_targets_serialized")
        is not False
        or receipt.get("per_page_digests_serialized") is not False
        or receipt.get("bottom_k_unit_ids_serialized") is not False
        or receipt.get("corpus_or_cohort_retained_after_diagnostic") is not False
        or receipt.get("formal_selection_secret_created_derived_or_accessed")
        is not False
        or receipt.get("claim_label_evidence_family_or_outcome_accessed")
        is not False
        or receipt.get("development_or_test_source_accessed") is not False
        or receipt.get("model_action_or_online_evaluator_calls") != 0
        or any(not _is_sha256(receipt.get(field)) for field in sha_fields)
        or not isinstance(configurations, list)
        or not configurations
        or _stable_hash(body) != declared
    ):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel diagnostic receipt drifted"
        )
    heap_capacity = receipt.get("heap_capacity")
    pages_per_configuration = receipt.get("pages_per_configuration")
    execution_order = receipt.get("configuration_execution_order")
    if (
        type(heap_capacity) is not int
        or heap_capacity < 1
        or type(pages_per_configuration) is not int
        or pages_per_configuration < 1
        or not isinstance(execution_order, list)
        or len(execution_order) != len(configurations)
    ):
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel diagnostic dimensions drifted"
        )
    previous_end: int | None = None
    measured_best: tuple[float, int] | None = None
    for ordinal, row in enumerate(configurations):
        if not isinstance(row, Mapping):
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "parallel diagnostic configuration is malformed"
            )
        start = row.get("start_rowid")
        end = row.get("end_rowid")
        workers = row.get("worker_count")
        page_count = row.get("observed_page_count")
        identity_count = row.get("observed_eligible_identity_count")
        hmac_count = row.get("public_hmac_evaluation_count")
        retained_count = row.get("retained_bottom_k_count")
        rate = row.get("pages_per_second")
        if (
            type(start) is not int
            or type(end) is not int
            or end - start + 1 != pages_per_configuration
            or (previous_end is not None and start != previous_end + 1)
            or workers != execution_order[ordinal]
            or type(workers) is not int
            or workers < 1
            or page_count != pages_per_configuration
            or type(identity_count) is not int
            or identity_count < 1
            or hmac_count != identity_count
            or retained_count != min(heap_capacity, identity_count)
            or not isinstance(rate, (int, float))
            or rate <= 0
            or row.get("exact_gap_free_nonoverlap_cover_verified") is not True
            or any(
                not _is_sha256(row.get(field))
                for field in (
                    "ordered_page_stream_sha256",
                    "retained_bottom_k_sha256",
                    "chunk_receipt_set_sha256",
                )
            )
        ):
            raise FeverousIdentityParallelPerformanceDiagnosticError(
                "parallel diagnostic configuration drifted"
            )
        previous_end = end
        contender = (float(rate), -workers)
        if measured_best is None or contender > measured_best:
            measured_best = contender
    assert measured_best is not None
    if receipt.get("throughput_best_worker_count") != -measured_best[1]:
        raise FeverousIdentityParallelPerformanceDiagnosticError(
            "parallel diagnostic best-worker conclusion drifted"
        )
    return str(declared)


__all__ = [
    "ConfigurationMeasurement",
    "DiagnosticDatabaseSpec",
    "FROZEN_DATABASE_SPEC",
    "FROZEN_FIRST_ROWID",
    "FROZEN_HEAP_CAPACITY",
    "FROZEN_PAGES_PER_CONFIGURATION",
    "FROZEN_WORKER_COUNTS",
    "FeverousIdentityParallelPerformanceDiagnosticError",
    "PUBLIC_DIAGNOSTIC_RANK_KEY_SHA256",
    "RowidInterval",
    "partition_rowid_interval",
    "run_parallel_performance_diagnostic",
    "scan_parallel_configuration",
    "verify_parallel_performance_diagnostic_receipt",
]


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded nonformal FEVEROUS parallel diagnostic."
    )
    parser.add_argument("--database", required=True)
    parser.add_argument("--first-rowid", type=int, default=FROZEN_FIRST_ROWID)
    parser.add_argument(
        "--pages-per-configuration",
        type=int,
        default=FROZEN_PAGES_PER_CONFIGURATION,
    )
    parser.add_argument(
        "--worker-counts",
        type=int,
        nargs="+",
        default=list(FROZEN_WORKER_COUNTS),
    )
    parser.add_argument(
        "--heap-capacity", type=int, default=FROZEN_HEAP_CAPACITY
    )
    arguments = parser.parse_args()
    receipt = run_parallel_performance_diagnostic(
        arguments.database,
        worker_counts=tuple(arguments.worker_counts),
        first_rowid=arguments.first_rowid,
        pages_per_configuration=arguments.pages_per_configuration,
        heap_capacity=arguments.heap_capacity,
    )
    verify_parallel_performance_diagnostic_receipt(receipt)
    print(
        json.dumps(
            receipt,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

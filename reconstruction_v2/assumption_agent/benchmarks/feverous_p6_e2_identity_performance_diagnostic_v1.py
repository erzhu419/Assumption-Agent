"""Non-scoring performance diagnostic for the FEVEROUS identity fast path.

The diagnostic is intentionally not a formal acquisition.  It opens only an
explicit SQLite file, never accepts annotations, claims, labels, a cohort, or
a selection secret, and never calls a model/evaluator.  It measures a bounded
physical-rowid prefix with the lightweight atomic identity enumerator, then
cross-checks a frozen content-independent set of rowids with the full compiler.

The public receipt is aggregate-only.  Page ids, payloads, atomic ids, targets,
and per-page digests are absent.  The known qualified database SHA-256 is bound
as metadata but is deliberately not recomputed here; consequently every
receipt says ``formal_valid=False`` regardless of which source spec is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sqlite3
import stat
import time
from typing import Any

from assumption_agent.benchmarks import feverous_atomic_corpus_v1 as atomic
from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks.feverous_p6_e2_acquisition_v1 import (
    CorpusIdentity,
    canonical_json_bytes,
)
from assumption_agent.benchmarks.feverous_p6_e2_source_adapter_v1 import (
    DESIGN_SHA256,
    WIKIPEDIA_QUALIFICATION_SHA256,
)
from assumption_agent.benchmarks.feverous_wikipedia_source_qualification_v1 import (
    FeverousWikipediaQualificationError,
    open_immutable_wiki_db,
)


VERSION = "feverous_p6_e2_identity_performance_diagnostic_v1"
RECEIPT_SCHEMA = f"{VERSION}_aggregate_receipt"
FROZEN_DATABASE_BASENAME = "feverous_wikiv1.db"
FROZEN_DATABASE_SIZE_BYTES = 53_486_538_752
FROZEN_DATABASE_SHA256 = (
    "a980581f55d46a252090b29269954503735b6f00274d05225476a650ab940276"
)
FROZEN_DATABASE_ROW_COUNT = 5_421_406
FROZEN_DATABASE_SCHEMA = "CREATE TABLE wiki (id PRIMARY KEY, data json)"
FROZEN_PREFIX_MINIMUM = 10_000
FROZEN_PREFIX_MAXIMUM = 100_000
FROZEN_CONTINUE_THRESHOLD_SECONDS = 60.0
FROZEN_REAL_CROSSCHECK_PAGE_COUNT = 64


class FeverousIdentityPerformanceDiagnosticError(RuntimeError):
    """The bounded diagnostic source, topology, or receipt drifted."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic receipt is not canonical JSON"
        ) from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_sha256(module: object) -> str:
    source = inspect.getsourcefile(module)
    if not isinstance(source, str):
        raise FeverousIdentityPerformanceDiagnosticError(
            "runtime source is unavailable"
        )
    try:
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    except OSError as exc:
        raise FeverousIdentityPerformanceDiagnosticError(
            "runtime source cannot be hashed"
        ) from exc


@dataclass(frozen=True)
class DiagnosticDatabaseSpec:
    basename: str
    size_bytes: int
    declared_sha256: str
    row_count: int
    schema: str = FROZEN_DATABASE_SCHEMA
    required_mode: int = 0o600

    def __post_init__(self) -> None:
        if not self.basename or Path(self.basename).name != self.basename:
            raise FeverousIdentityPerformanceDiagnosticError(
                "diagnostic database basename is invalid"
            )
        if (
            type(self.size_bytes) is not int
            or self.size_bytes < 1
            or type(self.row_count) is not int
            or self.row_count < 1
            or not _is_sha256(self.declared_sha256)
            or self.schema != FROZEN_DATABASE_SCHEMA
            or type(self.required_mode) is not int
            or not 0 <= self.required_mode <= 0o777
        ):
            raise FeverousIdentityPerformanceDiagnosticError(
                "diagnostic database spec drifted"
            )

    @property
    def spec_sha256(self) -> str:
        return _stable_hash(asdict(self))


FROZEN_DATABASE_SPEC = DiagnosticDatabaseSpec(
    basename=FROZEN_DATABASE_BASENAME,
    size_bytes=FROZEN_DATABASE_SIZE_BYTES,
    declared_sha256=FROZEN_DATABASE_SHA256,
    row_count=FROZEN_DATABASE_ROW_COUNT,
)


def _file_state(path: Path, spec: DiagnosticDatabaseSpec) -> tuple[int, ...]:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic database cannot be stated"
        ) from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_size != spec.size_bytes
        or stat.S_IMODE(observed.st_mode) != spec.required_mode
    ):
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic database file binding drifted"
        )
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
        stat.S_IMODE(observed.st_mode),
    )


def _rss_kib() -> int:
    try:
        for line in Path("/proc/self/status").read_text(
            encoding="ascii", errors="strict"
        ).splitlines():
            if line.startswith("VmRSS:"):
                value = line.split()
                if len(value) == 3 and value[2] == "kB":
                    return int(value[1])
    except (OSError, UnicodeError, ValueError):
        pass
    return 0


def _sample_rowids(row_count: int, sample_count: int) -> tuple[int, ...]:
    count = min(row_count, sample_count)
    values = tuple(
        min(row_count, ((2 * index + 1) * row_count) // (2 * count) + 1)
        for index in range(count)
    )
    if (
        not values
        or values[0] < 1
        or values[-1] > row_count
        or any(left >= right for left, right in zip(values, values[1:]))
    ):
        raise FeverousIdentityPerformanceDiagnosticError(
            "content-independent rowid sample drifted"
        )
    return values


def _open_bound_database(
    path: Path, spec: DiagnosticDatabaseSpec
) -> tuple[sqlite3.Connection, tuple[int, ...]]:
    state = _file_state(path, spec)
    try:
        connection = open_immutable_wiki_db(path)
        schema_rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' ORDER BY name"
        ).fetchall()
        columns = connection.execute("PRAGMA table_info(wiki)").fetchall()
    except (sqlite3.Error, FeverousWikipediaQualificationError) as exc:
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic SQLite open failed"
        ) from exc
    if (
        schema_rows != [("wiki", spec.schema)]
        or [row[1] for row in columns] != ["id", "data"]
        or _file_state(path, spec) != state
    ):
        connection.close()
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic SQLite schema or file identity drifted"
        )
    return connection, state


def run_identity_performance_diagnostic(
    database_path: str | os.PathLike[str],
    *,
    spec: DiagnosticDatabaseSpec = FROZEN_DATABASE_SPEC,
    prefix_minimum: int = FROZEN_PREFIX_MINIMUM,
    prefix_maximum: int = FROZEN_PREFIX_MAXIMUM,
    continue_threshold_seconds: float = FROZEN_CONTINUE_THRESHOLD_SECONDS,
    sample_page_count: int = FROZEN_REAL_CROSSCHECK_PAGE_COUNT,
) -> Mapping[str, Any]:
    """Run one bounded, aggregate-only, explicitly nonformal diagnostic."""

    if not isinstance(spec, DiagnosticDatabaseSpec):
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic source spec is absent"
        )
    if (
        type(prefix_minimum) is not int
        or type(prefix_maximum) is not int
        or not 1 <= prefix_minimum <= prefix_maximum <= spec.row_count
        or not isinstance(continue_threshold_seconds, (int, float))
        or continue_threshold_seconds < 0
        or type(sample_page_count) is not int
        or not 1 <= sample_page_count <= 64
    ):
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic bounds are invalid"
        )
    path = Path(database_path)
    if path.name != spec.basename:
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic database basename differs from binding"
        )
    connection, source_state = _open_bound_database(path, spec)
    atomic_source_sha256 = _source_sha256(atomic)
    acquisition_source_sha256 = _source_sha256(acquisition)
    diagnostic_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prefix_hasher = hashlib.sha256()
    identity_stream_hasher = hashlib.sha256()
    prefix_page_count = 0
    prefix_payload_bytes = 0
    prefix_identity_count = 0
    prefix_empty_count = 0
    prefix_decision = "stopped_at_minimum_elapsed_threshold"
    rss_before_kib = _rss_kib()
    peak_before_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    prefix_started = time.perf_counter()
    try:
        cursor = connection.execute(
            "SELECT rowid, id, data FROM wiki "
            "WHERE rowid BETWEEN 1 AND ? ORDER BY rowid",
            (prefix_maximum,),
        )
        for row in cursor:
            if not isinstance(row, tuple) or len(row) != 3:
                raise FeverousIdentityPerformanceDiagnosticError(
                    "diagnostic prefix row shape drifted"
                )
            rowid, page_id, raw_page = row
            if (
                type(rowid) is not int
                or rowid != prefix_page_count + 1
                or not isinstance(page_id, str)
                or not page_id
                or "\x00" in page_id
                or not isinstance(raw_page, str)
            ):
                raise FeverousIdentityPerformanceDiagnosticError(
                    "diagnostic prefix row identity drifted"
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
                identity_row = canonical_json_bytes(identity.commitment_row)
                identity_stream_hasher.update(
                    len(identity_row).to_bytes(8, "big")
                )
                identity_stream_hasher.update(identity_row)
            payload_utf8 = raw_page.encode("utf-8", errors="strict")
            commitment = enumeration.commitment()
            aggregate_row = _canonical_json(
                [
                    rowid,
                    len(payload_utf8),
                    hashlib.sha256(payload_utf8).hexdigest(),
                    commitment,
                ]
            )
            prefix_hasher.update(len(aggregate_row).to_bytes(8, "big"))
            prefix_hasher.update(aggregate_row)
            prefix_page_count += 1
            prefix_payload_bytes += len(payload_utf8)
            prefix_identity_count += len(enumeration.identities)
            prefix_empty_count += len(enumeration.excluded_empty_local_ids)
            if prefix_page_count == prefix_minimum:
                elapsed = time.perf_counter() - prefix_started
                if elapsed < continue_threshold_seconds:
                    prefix_decision = "continued_to_frozen_maximum_below_threshold"
                else:
                    break
            if prefix_page_count == prefix_maximum:
                break
        cursor.close()
        if prefix_page_count < prefix_minimum:
            raise FeverousIdentityPerformanceDiagnosticError(
                "diagnostic source ended before the minimum prefix"
            )
        prefix_elapsed = time.perf_counter() - prefix_started

        sample_rowids = _sample_rowids(spec.row_count, sample_page_count)
        sample_hasher = hashlib.sha256()
        sample_payload_bytes = 0
        sample_identity_count = 0
        sample_empty_count = 0
        sample_started = time.perf_counter()
        for rowid in sample_rowids:
            rows = connection.execute(
                "SELECT rowid, id, data FROM wiki WHERE rowid = ? LIMIT 2",
                (rowid,),
            ).fetchall()
            if (
                len(rows) != 1
                or len(rows[0]) != 3
                or rows[0][0] != rowid
                or not isinstance(rows[0][1], str)
                or not isinstance(rows[0][2], str)
            ):
                raise FeverousIdentityPerformanceDiagnosticError(
                    "content-independent sample row is absent"
                )
            page_id, raw_page = rows[0][1], rows[0][2]
            enumeration = atomic.enumerate_official_page_atomic_identities(
                page_id, raw_page
            )
            compilation = atomic.compile_official_page(page_id, raw_page)
            atomic.crosscheck_identity_enumeration(enumeration, compilation)
            payload_utf8 = raw_page.encode("utf-8", errors="strict")
            aggregate_row = _canonical_json(
                [rowid, len(payload_utf8), enumeration.commitment()]
            )
            sample_hasher.update(len(aggregate_row).to_bytes(8, "big"))
            sample_hasher.update(aggregate_row)
            sample_payload_bytes += len(payload_utf8)
            sample_identity_count += len(enumeration.identities)
            sample_empty_count += len(enumeration.excluded_empty_local_ids)
        sample_elapsed = time.perf_counter() - sample_started
        if _file_state(path, spec) != source_state:
            raise FeverousIdentityPerformanceDiagnosticError(
                "diagnostic source changed while open"
            )
    finally:
        connection.close()

    peak_after_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_after_kib = _rss_kib()
    seconds_per_page = prefix_elapsed / prefix_page_count
    projected_seconds = seconds_per_page * spec.row_count
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "passed_nonformal_identity_performance_no_selection",
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
        "database_open_mode": "mode=ro&immutable=1_query_only",
        "prefix_query_order": "strict_consecutive_physical_rowid_from_1",
        "prefix_minimum_page_count": prefix_minimum,
        "prefix_maximum_page_count": prefix_maximum,
        "continue_threshold_seconds": round(float(continue_threshold_seconds), 6),
        "prefix_stop_decision": prefix_decision,
        "observed_prefix_page_count": prefix_page_count,
        "observed_prefix_payload_utf8_bytes": prefix_payload_bytes,
        "observed_prefix_eligible_identity_count": prefix_identity_count,
        "observed_prefix_excluded_empty_count": prefix_empty_count,
        "observed_prefix_aggregate_sha256": prefix_hasher.hexdigest(),
        "observed_prefix_identity_stream_sha256": (
            identity_stream_hasher.hexdigest()
        ),
        "observed_prefix_wall_seconds": round(prefix_elapsed, 6),
        "observed_prefix_pages_per_second": round(
            prefix_page_count / prefix_elapsed, 6
        ),
        "observed_prefix_payload_mib_per_second": round(
            prefix_payload_bytes / (1024 * 1024) / prefix_elapsed, 6
        ),
        "projected_complete_identity_scan_seconds": round(projected_seconds, 6),
        "projected_complete_identity_scan_hours": round(
            projected_seconds / 3600, 6
        ),
        "projection_method": "observed_prefix_wall_seconds_per_page_times_bound_row_count",
        "projection_scope": (
            "lightweight_enumerator_plus_CorpusIdentity_validation_plus_exact_"
            "identity_stream_commitment"
        ),
        "private_HMAC_heap_selection_included": False,
        "real_crosscheck_sample_policy": (
            "64_equal_width_rowid_bin_midpoints_independent_of_page_content"
        ),
        "real_crosscheck_sample_page_count": len(sample_rowids),
        "real_crosscheck_payload_utf8_bytes": sample_payload_bytes,
        "real_crosscheck_eligible_identity_count": sample_identity_count,
        "real_crosscheck_excluded_empty_count": sample_empty_count,
        "real_crosscheck_aggregate_sha256": sample_hasher.hexdigest(),
        "real_crosscheck_wall_seconds": round(sample_elapsed, 6),
        "identity_full_compiler_mismatch_count": 0,
        "rss_before_kib": rss_before_kib,
        "rss_after_kib": rss_after_kib,
        "process_peak_rss_before_kib": peak_before_kib,
        "process_peak_rss_after_kib": peak_after_kib,
        "process_peak_rss_growth_kib": max(0, peak_after_kib - peak_before_kib),
        "atomic_compiler_version": atomic.VERSION,
        "identity_enumerator_version": atomic.IDENTITY_ENUMERATOR_VERSION,
        "atomic_compiler_and_enumerator_source_sha256": atomic_source_sha256,
        "acquisition_identity_source_sha256": acquisition_source_sha256,
        "diagnostic_source_sha256": diagnostic_source_sha256,
        "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "platform_system": platform.system(),
        "logical_cpu_count": os.cpu_count(),
        "all_page_ids_payloads_atomic_ids_or_targets_serialized": False,
        "per_page_digest_serialized": False,
        "full_database_identity_scan_performed": False,
        "cohort_candidate_canonical_set_or_corpus_selected": False,
        "selection_secret_created_or_accessed": False,
        "claim_label_evidence_family_or_outcome_accessed": False,
        "development_or_test_source_accessed": False,
        "model_action_or_online_evaluator_calls": 0,
    }
    body["diagnostic_receipt_sha256"] = _stable_hash(body)
    return body


def verify_identity_performance_diagnostic_receipt(
    receipt: Mapping[str, Any],
) -> str:
    if not isinstance(receipt, Mapping):
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic receipt must be an object"
        )
    body = dict(receipt)
    declared = body.pop("diagnostic_receipt_sha256", None)
    sha_fields = (
        "database_spec_sha256",
        "design_sha256",
        "wikipedia_qualification_sha256",
        "database_declared_sha256",
        "database_schema_sha256",
        "observed_prefix_aggregate_sha256",
        "observed_prefix_identity_stream_sha256",
        "real_crosscheck_aggregate_sha256",
        "atomic_compiler_and_enumerator_source_sha256",
        "acquisition_identity_source_sha256",
        "diagnostic_source_sha256",
    )
    if (
        not _is_sha256(declared)
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("version") != VERSION
        or receipt.get("status")
        != "passed_nonformal_identity_performance_no_selection"
        or receipt.get("formal_valid") is not False
        or receipt.get("implementation_performance_only") is not True
        or receipt.get("source_split_or_annotation_opened") is not False
        or receipt.get("database_sha256_recomputed_this_diagnostic") is not False
        or receipt.get("identity_full_compiler_mismatch_count") != 0
        or receipt.get("all_page_ids_payloads_atomic_ids_or_targets_serialized")
        is not False
        or receipt.get("per_page_digest_serialized") is not False
        or receipt.get("full_database_identity_scan_performed") is not False
        or receipt.get("private_HMAC_heap_selection_included") is not False
        or receipt.get("cohort_candidate_canonical_set_or_corpus_selected")
        is not False
        or receipt.get("selection_secret_created_or_accessed") is not False
        or receipt.get("claim_label_evidence_family_or_outcome_accessed")
        is not False
        or receipt.get("development_or_test_source_accessed") is not False
        or receipt.get("model_action_or_online_evaluator_calls") != 0
        or any(not _is_sha256(receipt.get(field)) for field in sha_fields)
        or _stable_hash(body) != declared
    ):
        raise FeverousIdentityPerformanceDiagnosticError(
            "diagnostic receipt drifted"
        )
    return str(declared)


__all__ = [
    "DiagnosticDatabaseSpec",
    "FROZEN_DATABASE_SPEC",
    "FeverousIdentityPerformanceDiagnosticError",
    "run_identity_performance_diagnostic",
    "verify_identity_performance_diagnostic_receipt",
]

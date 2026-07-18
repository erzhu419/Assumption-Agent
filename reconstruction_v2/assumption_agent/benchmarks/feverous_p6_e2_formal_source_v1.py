"""Controlled one-shot custody for the formal FEVEROUS P6/E2 TRAIN source.

This module is deliberately narrower than a general dataset loader.  A formal
instance can name only the already-qualified TRAIN JSONL file and the frozen
FEVEROUS SQLite database.  Both files are regular, non-symlink files and are
checked against their public size/SHA-256 bindings at the point of use.  The
annotation file is parsed and hashed in one read.  The database has one
deterministic, page-id ordered full-table stream; a receipt exists only after
that stream reaches normal exhaustion.

Synthetic specifications are supported solely for implementation tests.  A
synthetic receipt carries ``formal_source=False`` and therefore cannot satisfy
the downstream formal-acquisition verifier even if it is self-consistent.
No directory is accepted or enumerated, so DEV/TEST neighbours are outside the
opener's authority.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from types import MappingProxyType
from typing import Any

from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as source_adapter
from assumption_agent.benchmarks import (
    feverous_p6_e2_parallel_identity_selection_v1 as parallel_selection,
)
from assumption_agent.benchmarks.feverous_p6_e2_source_adapter_v1 import (
    ANNOTATION_QUALIFICATION_SHA256,
    DESIGN_SHA256,
    FROZEN_TRAIN_BINDING,
    WIKIPEDIA_QUALIFICATION_SHA256,
)
from assumption_agent.benchmarks.feverous_wikipedia_source_qualification_v1 import (
    FeverousWikiResolver,
    FeverousWikipediaQualificationError,
    open_immutable_wiki_db,
)


VERSION = "feverous_p6_e2_formal_source_v1"
ANNOTATION_RECEIPT_SCHEMA = f"{VERSION}_annotation_receipt"
DATABASE_RECEIPT_SCHEMA = f"{VERSION}_database_page_stream_receipt"
SELECTED_LOOKUP_RECEIPT_SCHEMA = f"{VERSION}_selected_page_lookup_receipt"

FROZEN_ANNOTATION_BASENAME = "feverous_train_challenges.jsonl"
FROZEN_ANNOTATION_SIZE_BYTES = 177_565_233
FROZEN_ANNOTATION_SHA256 = (
    "0c29ccba41e27c5b988ca5132085e8d67c7921f265707bea170bfbde12bceee7"
)
FROZEN_ANNOTATION_NONBLANK_ROWS = 71_291
FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS = 1
FROZEN_DATABASE_BASENAME = "feverous_wikiv1.db"
FROZEN_DATABASE_SIZE_BYTES = 53_486_538_752
FROZEN_DATABASE_SHA256 = (
    "a980581f55d46a252090b29269954503735b6f00274d05225476a650ab940276"
)
FROZEN_DATABASE_ROW_COUNT = 5_421_406
FROZEN_DATABASE_SCHEMA = "CREATE TABLE wiki (id PRIMARY KEY, data json)"

_READ_CHUNK_BYTES = 8 * 1024 * 1024
_FETCH_BATCH_ROWS = 256
_SHA256_HEX = frozenset("0123456789abcdef")


class FeverousFormalSourceError(RuntimeError):
    """A frozen source, one-shot boundary, or exhaustion receipt drifted."""


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
        raise FeverousFormalSourceError("receipt is not canonical JSON") from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_HEX)
    )


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise FeverousFormalSourceError("self-hash field already exists")
    output = dict(body)
    output[field] = _stable_hash(body)
    return output


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    if not isinstance(payload, Mapping):
        raise FeverousFormalSourceError("receipt must be an object")
    body = dict(payload)
    declared = body.pop(field, None)
    if not _is_sha256(declared) or _stable_hash(body) != declared:
        raise FeverousFormalSourceError("receipt self-hash mismatch")
    return str(declared)


@dataclass(frozen=True)
class TrainSourceSpec:
    """Content bindings for one controlled source pair.

    Equality with :data:`FROZEN_TRAIN_SOURCE_SPEC`, rather than a caller-set
    boolean, determines formal status.  This prevents a small fixture from
    declaring itself formal.
    """

    source_split: str
    design_sha256: str
    annotation_qualification_sha256: str
    wikipedia_qualification_sha256: str
    annotation_basename: str
    annotation_size_bytes: int
    annotation_sha256: str
    annotation_nonblank_rows: int
    annotation_blank_sentinel_rows: int
    database_basename: str
    database_size_bytes: int
    database_sha256: str
    database_row_count: int
    database_schema: str = FROZEN_DATABASE_SCHEMA
    required_mode: int = 0o600

    def __post_init__(self) -> None:
        if self.source_split != "TRAIN":
            raise FeverousFormalSourceError("only TRAIN may enter the source spec")
        if any(
            not _is_sha256(value)
            for value in (
                self.design_sha256,
                self.annotation_qualification_sha256,
                self.wikipedia_qualification_sha256,
                self.annotation_sha256,
                self.database_sha256,
            )
        ):
            raise FeverousFormalSourceError("source spec has an invalid SHA-256")
        if (
            not self.annotation_basename
            or Path(self.annotation_basename).name != self.annotation_basename
            or not self.database_basename
            or Path(self.database_basename).name != self.database_basename
        ):
            raise FeverousFormalSourceError("source spec basename is invalid")
        integer_fields = (
            self.annotation_size_bytes,
            self.annotation_nonblank_rows,
            self.annotation_blank_sentinel_rows,
            self.database_size_bytes,
            self.database_row_count,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise FeverousFormalSourceError("source spec count is invalid")
        if self.annotation_blank_sentinel_rows != 1:
            raise FeverousFormalSourceError(
                "TRAIN source must contain one exact all-empty-field sentinel"
            )
        if self.database_schema != FROZEN_DATABASE_SCHEMA:
            raise FeverousFormalSourceError("wiki schema binding drifted")
        if type(self.required_mode) is not int or not 0 <= self.required_mode <= 0o777:
            raise FeverousFormalSourceError("source mode binding is invalid")

    @property
    def spec_sha256(self) -> str:
        return _stable_hash(asdict(self))


FROZEN_TRAIN_SOURCE_SPEC = TrainSourceSpec(
    source_split="TRAIN",
    design_sha256=DESIGN_SHA256,
    annotation_qualification_sha256=ANNOTATION_QUALIFICATION_SHA256,
    wikipedia_qualification_sha256=WIKIPEDIA_QUALIFICATION_SHA256,
    annotation_basename=FROZEN_ANNOTATION_BASENAME,
    annotation_size_bytes=FROZEN_ANNOTATION_SIZE_BYTES,
    annotation_sha256=FROZEN_ANNOTATION_SHA256,
    annotation_nonblank_rows=FROZEN_ANNOTATION_NONBLANK_ROWS,
    annotation_blank_sentinel_rows=FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS,
    database_basename=FROZEN_DATABASE_BASENAME,
    database_size_bytes=FROZEN_DATABASE_SIZE_BYTES,
    database_sha256=FROZEN_DATABASE_SHA256,
    database_row_count=FROZEN_DATABASE_ROW_COUNT,
)


@dataclass(frozen=True)
class _FileState:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    mode: int


def _file_state(path: Path, *, expected_size: int, required_mode: int) -> _FileState:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise FeverousFormalSourceError("source file cannot be stated") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise FeverousFormalSourceError(
            "source must be a regular, non-symlink file"
        )
    if observed.st_size != expected_size:
        raise FeverousFormalSourceError("source byte size differs from its binding")
    if stat.S_IMODE(observed.st_mode) != required_mode:
        raise FeverousFormalSourceError("source file mode differs from qualification")
    return _FileState(
        device=observed.st_dev,
        inode=observed.st_ino,
        size_bytes=observed.st_size,
        mtime_ns=observed.st_mtime_ns,
        ctime_ns=observed.st_ctime_ns,
        mode=stat.S_IMODE(observed.st_mode),
    )


def _require_same_file(path: Path, expected: _FileState) -> None:
    observed = _file_state(
        path,
        expected_size=expected.size_bytes,
        required_mode=expected.mode,
    )
    if observed != expected:
        raise FeverousFormalSourceError("source changed during controlled use")


def _hash_file_once(path: Path, *, state: _FileState) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FeverousFormalSourceError("source cannot be opened safely") from exc
    try:
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino, observed.st_size) != (
            state.device,
            state.inode,
            state.size_bytes,
        ):
            raise FeverousFormalSourceError("opened source identity drifted")
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_size != state.size_bytes
            or after.st_mtime_ns != state.mtime_ns
            or after.st_ctime_ns != state.ctime_ns
        ):
            raise FeverousFormalSourceError("source changed while hashing")
    finally:
        os.close(descriptor)
    _require_same_file(path, state)
    return digest.hexdigest()


def _decode_json_line(raw: bytes) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FeverousFormalSourceError("TRAIN annotation is not strict UTF-8") from exc

    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise FeverousFormalSourceError(
                    "TRAIN annotation contains a duplicate JSON key"
                )
            output[key] = value
        return output

    def reject_constant(_value: str) -> None:
        raise FeverousFormalSourceError(
            "TRAIN annotation contains a non-finite value"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except FeverousFormalSourceError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise FeverousFormalSourceError("TRAIN annotation line is not JSON") from exc
    if not isinstance(value, Mapping):
        raise FeverousFormalSourceError("TRAIN annotation row must be an object")
    return value


class DatabasePageStream(Iterator[tuple[str, str]]):
    """The sole deterministic full-table iterator for one controlled DB.

    The frozen table is scanned in physical ``rowid`` order.  Sorting 53 GB by
    the page-id primary-key index would turn the mandatory universe pass into
    random I/O; selection ranks do not depend on encounter order.  Exact
    schema/PK and consecutive rowids retain uniqueness and determinism.
    """

    def __init__(
        self,
        *,
        owner: "ControlledTrainSource",
        connection: sqlite3.Connection,
        source_state: _FileState,
    ) -> None:
        self._owner = owner
        self._connection = connection
        self._source_state = source_state
        try:
            self._cursor = connection.execute(
                "SELECT rowid, id, data FROM wiki ORDER BY rowid"
            )
        except sqlite3.Error as exc:
            raise FeverousFormalSourceError(
                "deterministic wiki traversal cannot start"
            ) from exc
        self._batch: list[tuple[Any, ...]] = []
        self._batch_index = 0
        self._previous_rowid = 0
        self._row_count = 0
        self._logical_hasher = hashlib.sha256()
        self._complete = False
        self._receipt: Mapping[str, Any] | None = None

    def __iter__(self) -> "DatabasePageStream":
        return self

    def __next__(self) -> tuple[str, str]:
        if self._complete:
            raise StopIteration
        if self._batch_index >= len(self._batch):
            try:
                self._batch = self._cursor.fetchmany(_FETCH_BATCH_ROWS)
            except sqlite3.Error as exc:
                raise FeverousFormalSourceError("wiki traversal failed") from exc
            self._batch_index = 0
            if not self._batch:
                self._finish()
                raise StopIteration
        row = self._batch[self._batch_index]
        self._batch_index += 1
        if not isinstance(row, tuple) or len(row) != 3:
            raise FeverousFormalSourceError("wiki row shape drifted")
        rowid, page_id, raw_page = row
        if (
            type(rowid) is not int
            or rowid != self._previous_rowid + 1
            or not isinstance(page_id, str)
            or not page_id
            or "\x00" in page_id
            or not isinstance(raw_page, str)
            or "\x00" in raw_page
        ):
            raise FeverousFormalSourceError("wiki row types drifted")
        try:
            page_id_utf8 = page_id.encode("utf-8", errors="strict")
            raw_page_utf8 = raw_page.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise FeverousFormalSourceError("wiki row is not strict UTF-8") from exc
        self._previous_rowid = rowid
        logical_row = [
            rowid,
            page_id,
            len(raw_page_utf8),
            hashlib.sha256(raw_page_utf8).hexdigest(),
        ]
        encoded = _canonical_json(logical_row)
        self._logical_hasher.update(len(encoded).to_bytes(8, "big"))
        self._logical_hasher.update(encoded)
        self._row_count += 1
        return page_id, raw_page

    def _finish(self) -> None:
        if self._row_count != self._owner.spec.database_row_count:
            raise FeverousFormalSourceError(
                "wiki traversal did not exhaust the bound row universe"
            )
        _require_same_file(self._owner.database_path, self._source_state)
        body: dict[str, Any] = {
            "schema": DATABASE_RECEIPT_SCHEMA,
            "version": VERSION,
            "status": "complete_database_page_stream_exhausted",
            "source_split": "TRAIN",
            "source_spec_sha256": self._owner.spec.spec_sha256,
            "source_binding_sha256": FROZEN_TRAIN_BINDING.binding_sha256,
            "formal_source_opener_source_sha256": (
                self._owner._formal_source_module_sha256
            ),
            "formal_source": self._owner.formal_source,
            "database_basename": self._owner.spec.database_basename,
            "database_size_bytes": self._owner.spec.database_size_bytes,
            "database_file_sha256": self._owner.spec.database_sha256,
            "database_schema_sha256": _stable_hash(
                self._owner.spec.database_schema
            ),
            "expected_database_row_count": self._owner.spec.database_row_count,
            "observed_database_row_count": self._row_count,
            "page_order": "strict_consecutive_rowid_physical_table_order",
            "logical_page_stream_sha256": self._logical_hasher.hexdigest(),
            "stream_fully_exhausted": True,
            "maximum_buffered_database_rows": _FETCH_BATCH_ROWS,
            "all_page_ids_or_pages_materialized": False,
            "development_or_test_source_accessed": False,
            "online_evaluator_calls": 0,
        }
        receipt = _self_hashed(body, "database_page_stream_receipt_sha256")
        self._receipt = MappingProxyType(receipt)
        self._complete = True
        self._cursor.close()
        self._owner._mark_database_stream_exhausted(self)

    def aggregate_receipt(self) -> Mapping[str, Any]:
        if not self._complete or self._receipt is None:
            raise FeverousFormalSourceError(
                "database stream receipt is unavailable before normal exhaustion"
            )
        return self._receipt


class SelectedPageLookupStream(Iterator[tuple[str, str]]):
    """One post-exhaustion indexed lookup pass over a bounded selected set."""

    def __init__(
        self,
        *,
        owner: "ControlledTrainSource",
        page_ids: Sequence[str],
    ) -> None:
        if (
            isinstance(page_ids, (str, bytes, bytearray))
            or any(not isinstance(value, str) or not value for value in page_ids)
        ):
            raise FeverousFormalSourceError("selected page ids are invalid")
        page_id_tuple = tuple(page_ids)
        encoded = tuple(value.encode("utf-8", errors="strict") for value in page_id_tuple)
        if any(left >= right for left, right in zip(encoded, encoded[1:])):
            raise FeverousFormalSourceError(
                "selected page ids must be unique strict binary order"
            )
        maximum_page_ids = 8192 + source_adapter.REAL_IDENTITY_COMPILER_SAMPLE_PAGE_COUNT
        if len(page_id_tuple) > maximum_page_ids:
            raise FeverousFormalSourceError("selected page lookup is not bounded")
        self._owner = owner
        self._maximum_page_ids = maximum_page_ids
        self._page_ids = page_id_tuple
        self._index = 0
        self._hasher = hashlib.sha256()
        self._complete = False
        self._receipt: Mapping[str, Any] | None = None

    def __iter__(self) -> "SelectedPageLookupStream":
        return self

    def __next__(self) -> tuple[str, str]:
        if self._complete:
            raise StopIteration
        if self._index >= len(self._page_ids):
            self._finish()
            raise StopIteration
        page_id = self._page_ids[self._index]
        self._index += 1
        try:
            rows = self._owner._connection.execute(
                "SELECT id, data FROM wiki "
                "WHERE id COLLATE BINARY = ? COLLATE BINARY LIMIT 2",
                (page_id,),
            ).fetchall()
        except (AttributeError, sqlite3.Error) as exc:
            raise FeverousFormalSourceError("selected page lookup failed") from exc
        if len(rows) != 1 or len(rows[0]) != 2 or rows[0][0] != page_id:
            raise FeverousFormalSourceError(
                "selected page is missing or non-unique in frozen SQLite"
            )
        raw_page = rows[0][1]
        if not isinstance(raw_page, str):
            raise FeverousFormalSourceError("selected page payload is not text")
        row_commitment = _canonical_json(
            [page_id, hashlib.sha256(raw_page.encode("utf-8")).hexdigest()]
        )
        self._hasher.update(len(row_commitment).to_bytes(8, "big"))
        self._hasher.update(row_commitment)
        return page_id, raw_page

    def _finish(self) -> None:
        assert self._owner._database_state is not None
        _require_same_file(self._owner.database_path, self._owner._database_state)
        body: dict[str, Any] = {
            "schema": SELECTED_LOOKUP_RECEIPT_SCHEMA,
            "version": VERSION,
            "status": "selected_pages_materialized_after_full_universe_exhaustion",
            "source_split": "TRAIN",
            "source_spec_sha256": self._owner.spec.spec_sha256,
            "formal_source_opener_source_sha256": (
                self._owner._formal_source_module_sha256
            ),
            "formal_source": self._owner.formal_source,
            "database_page_stream_receipt_sha256": self._owner.database_receipt[
                "database_page_stream_receipt_sha256"
            ],
            "selected_page_count": len(self._page_ids),
            "selected_page_lookup_sha256": self._hasher.hexdigest(),
            "lookup_order": "strict_ascending_utf8_equivalent_SQLite_BINARY_id",
            "full_database_rescan": False,
            "maximum_selected_page_ids_resident": self._maximum_page_ids,
            "development_or_test_source_accessed": False,
            "online_evaluator_calls": 0,
        }
        self._receipt = MappingProxyType(
            _self_hashed(body, "selected_page_lookup_receipt_sha256")
        )
        self._complete = True
        self._owner._mark_selected_lookup_exhausted(self)

    def aggregate_receipt(self) -> Mapping[str, Any]:
        if not self._complete or self._receipt is None:
            raise FeverousFormalSourceError(
                "selected-page receipt is unavailable before normal exhaustion"
            )
        return self._receipt


class ControlledTrainSource:
    """Stateful custody object; every source operation is available once."""

    def __init__(
        self,
        *,
        annotation_path: str | os.PathLike[str],
        database_path: str | os.PathLike[str],
        spec: TrainSourceSpec = FROZEN_TRAIN_SOURCE_SPEC,
    ) -> None:
        if not isinstance(spec, TrainSourceSpec):
            raise FeverousFormalSourceError("source spec is absent")
        self.annotation_path = Path(annotation_path)
        self.database_path = Path(database_path)
        self.spec = spec
        if self.annotation_path.name != spec.annotation_basename:
            raise FeverousFormalSourceError("annotation basename differs from binding")
        if self.database_path.name != spec.database_basename:
            raise FeverousFormalSourceError("database basename differs from binding")
        self._formal_source_module_sha256 = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        self._annotation_used = False
        self._annotation_receipt: Mapping[str, Any] | None = None
        self._database_opened = False
        self._candidate_resolver_created = False
        self._database_stream_created = False
        self._database_stream_exhausted = False
        self._selected_lookup_created = False
        self._selected_lookup_exhausted = False
        self._connection: sqlite3.Connection | None = None
        self._database_state: _FileState | None = None
        self._database_receipt: Mapping[str, Any] | None = None
        self._selected_lookup_receipt: Mapping[str, Any] | None = None
        self._closed = False

    @property
    def formal_source(self) -> bool:
        return self.spec == FROZEN_TRAIN_SOURCE_SPEC

    def read_annotations_once(self) -> tuple[Mapping[str, Any], ...]:
        if self._closed or self._annotation_used:
            raise FeverousFormalSourceError("TRAIN annotations are one-shot")
        self._annotation_used = True
        state = _file_state(
            self.annotation_path,
            expected_size=self.spec.annotation_size_bytes,
            required_mode=self.spec.required_mode,
        )
        digest = hashlib.sha256()
        records: list[Mapping[str, Any]] = []
        nonblank = 0
        blank = 0
        total_bytes = 0
        try:
            with self.annotation_path.open("rb", buffering=1024 * 1024) as handle:
                opened = os.fstat(handle.fileno())
                if (opened.st_dev, opened.st_ino, opened.st_size) != (
                    state.device,
                    state.inode,
                    state.size_bytes,
                ):
                    raise FeverousFormalSourceError(
                        "opened annotation identity drifted"
                    )
                for raw_line in handle:
                    total_bytes += len(raw_line)
                    digest.update(raw_line)
                    content = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
                    if content.endswith(b"\r"):
                        content = content[:-1]
                    if not content:
                        raise FeverousFormalSourceError(
                            "TRAIN JSONL contains an empty physical line"
                        )
                    record = _decode_json_line(content)
                    if source_adapter._is_blank_sentinel(record):
                        blank += 1
                    else:
                        nonblank += 1
                    records.append(record)
        except OSError as exc:
            raise FeverousFormalSourceError("TRAIN annotation read failed") from exc
        _require_same_file(self.annotation_path, state)
        if (
            total_bytes != self.spec.annotation_size_bytes
            or digest.hexdigest() != self.spec.annotation_sha256
            or nonblank != self.spec.annotation_nonblank_rows
            or blank != self.spec.annotation_blank_sentinel_rows
        ):
            raise FeverousFormalSourceError(
                "TRAIN annotation content differs from its frozen binding"
            )
        body: dict[str, Any] = {
            "schema": ANNOTATION_RECEIPT_SCHEMA,
            "version": VERSION,
            "status": "train_annotation_read_once_and_verified",
            "source_split": "TRAIN",
            "source_spec_sha256": self.spec.spec_sha256,
            "source_binding_sha256": FROZEN_TRAIN_BINDING.binding_sha256,
            "formal_source_opener_source_sha256": (
                self._formal_source_module_sha256
            ),
            "formal_source": self.formal_source,
            "annotation_basename": self.spec.annotation_basename,
            "annotation_size_bytes": total_bytes,
            "annotation_file_sha256": digest.hexdigest(),
            "annotation_nonblank_rows": nonblank,
            "annotation_blank_sentinel_rows": blank,
            "annotation_file_read_count": 1,
            "development_or_test_source_accessed": False,
            "online_evaluator_calls": 0,
        }
        self._annotation_receipt = MappingProxyType(
            _self_hashed(body, "annotation_receipt_sha256")
        )
        return tuple(records)

    @property
    def annotation_receipt(self) -> Mapping[str, Any]:
        if self._annotation_receipt is None:
            raise FeverousFormalSourceError("annotation receipt is unavailable")
        return self._annotation_receipt

    def _open_database_once(self) -> sqlite3.Connection:
        if self._closed:
            raise FeverousFormalSourceError("source custody is closed")
        if self._database_opened:
            assert self._connection is not None
            return self._connection
        self._database_opened = True
        state = _file_state(
            self.database_path,
            expected_size=self.spec.database_size_bytes,
            required_mode=self.spec.required_mode,
        )
        observed_sha256 = _hash_file_once(self.database_path, state=state)
        if observed_sha256 != self.spec.database_sha256:
            raise FeverousFormalSourceError(
                "SQLite SHA-256 differs from source qualification"
            )
        try:
            connection = open_immutable_wiki_db(self.database_path)
            schema_rows = connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' ORDER BY name"
            ).fetchall()
            columns = connection.execute("PRAGMA table_info(wiki)").fetchall()
        except (sqlite3.Error, FeverousWikipediaQualificationError) as exc:
            raise FeverousFormalSourceError("qualified SQLite open failed") from exc
        if schema_rows != [("wiki", self.spec.database_schema)]:
            connection.close()
            raise FeverousFormalSourceError("SQLite schema differs from qualification")
        if [row[1] for row in columns] != ["id", "data"]:
            connection.close()
            raise FeverousFormalSourceError("SQLite wiki columns drifted")
        _require_same_file(self.database_path, state)
        self._database_state = state
        self._connection = connection
        return connection

    def exact_resolver_for_candidate_screen(self) -> FeverousWikiResolver:
        """Return the qualified resolver; no raw filesystem capability leaks."""

        if self._candidate_resolver_created:
            raise FeverousFormalSourceError("candidate resolver is one-shot")
        if self._database_stream_created:
            raise FeverousFormalSourceError(
                "candidate screening must precede the full universe stream"
            )
        self._candidate_resolver_created = True
        return FeverousWikiResolver(self._open_database_once())

    def iter_database_pages_once(self) -> DatabasePageStream:
        if self._database_stream_created:
            raise FeverousFormalSourceError("database page traversal is one-shot")
        self._database_stream_created = True
        connection = self._open_database_once()
        assert self._database_state is not None
        return DatabasePageStream(
            owner=self,
            connection=connection,
            source_state=self._database_state,
        )

    def iter_corpus_identities_once(
        self,
        *,
        identity_full_compile_equivalence_qualification_sha256: str,
    ) -> source_adapter.CorpusIdentityStream:
        """Open the sole fast full-universe stream through the frozen seam."""

        pages = self.iter_database_pages_once()
        return source_adapter.iter_qualified_corpus_identities(
            pages,
            binding=FROZEN_TRAIN_BINDING,
            identity_full_compile_equivalence_qualification_sha256=(
                identity_full_compile_equivalence_qualification_sha256
            ),
        )

    def plan_corpus_identities_parallel_once(
        self,
        *,
        blocks: Mapping[str, Sequence[Any]],
        secret: bytes,
        identity_full_compile_equivalence_qualification_sha256: str,
    ) -> source_adapter.CorpusSelectionPlan:
        """Exhaust and select the full universe with the frozen 8-worker cover."""

        if self._database_stream_created:
            raise FeverousFormalSourceError("database page traversal is one-shot")
        if not self._candidate_resolver_created:
            raise FeverousFormalSourceError(
                "candidate screening must precede parallel universe selection"
            )
        self._database_stream_created = True
        self._open_database_once()
        assert self._database_state is not None
        state = self._database_state
        binding = parallel_selection.BoundDatabase(
            basename=self.spec.database_basename,
            size_bytes=self.spec.database_size_bytes,
            declared_sha256=self.spec.database_sha256,
            row_count=self.spec.database_row_count,
            schema=self.spec.database_schema,
            required_mode=self.spec.required_mode,
            device=state.device,
            inode=state.inode,
            mtime_ns=state.mtime_ns,
            ctime_ns=state.ctime_ns,
            source_spec_sha256=self.spec.spec_sha256,
            source_binding_sha256=FROZEN_TRAIN_BINDING.binding_sha256,
            formal_source_opener_source_sha256=(
                self._formal_source_module_sha256
            ),
            formal_source=self.formal_source,
        )
        try:
            outcome = parallel_selection.plan_fixed_corpus_parallel(
                database_path=self.database_path,
                database_binding=binding,
                blocks=blocks,
                secret=secret,
                identity_full_compile_equivalence_qualification_sha256=(
                    identity_full_compile_equivalence_qualification_sha256
                ),
            )
            verify_database_page_stream_receipt(outcome.database_receipt)
            if self.formal_source:
                require_formal_database_page_stream_receipt(
                    outcome.database_receipt
                )
        except Exception as exc:
            raise FeverousFormalSourceError(
                "parallel identity selection failed closed"
            ) from exc
        self._database_stream_exhausted = True
        self._database_receipt = outcome.database_receipt
        return outcome.plan

    def _mark_database_stream_exhausted(self, stream: DatabasePageStream) -> None:
        if self._database_stream_exhausted:
            raise FeverousFormalSourceError("database exhaustion was duplicated")
        self._database_stream_exhausted = True
        self._database_receipt = stream.aggregate_receipt()

    @property
    def database_receipt(self) -> Mapping[str, Any]:
        if not self._database_stream_exhausted or self._database_receipt is None:
            raise FeverousFormalSourceError(
                "database receipt is unavailable before full exhaustion"
            )
        return self._database_receipt

    def iter_selected_pages_once(
        self, page_ids: Sequence[str]
    ) -> SelectedPageLookupStream:
        if not self._database_stream_exhausted:
            raise FeverousFormalSourceError(
                "selected pages cannot open before full-universe exhaustion"
            )
        if self._selected_lookup_created:
            raise FeverousFormalSourceError("selected-page lookup is one-shot")
        self._selected_lookup_created = True
        return SelectedPageLookupStream(owner=self, page_ids=page_ids)

    def iter_selected_corpus_units_once(
        self,
        plan: "source_adapter.CorpusSelectionPlan",
    ) -> source_adapter.SelectedCorpusUnitStream:
        """After universe exhaustion, full-compile only plan-selected pages."""

        if not isinstance(plan, source_adapter.CorpusSelectionPlan):
            raise FeverousFormalSourceError("identity selection plan is absent")
        pages = self.iter_selected_pages_once(plan.full_compile_page_ids)
        assert self._connection is not None
        resolver = FeverousWikiResolver(self._connection)
        return source_adapter.iter_selected_corpus_units(
            pages,
            resolver=resolver,
            binding=FROZEN_TRAIN_BINDING,
            plan=plan,
        )

    def _mark_selected_lookup_exhausted(
        self, stream: SelectedPageLookupStream
    ) -> None:
        if self._selected_lookup_exhausted:
            raise FeverousFormalSourceError("selected lookup exhaustion duplicated")
        self._selected_lookup_exhausted = True
        self._selected_lookup_receipt = stream.aggregate_receipt()

    @property
    def selected_lookup_receipt(self) -> Mapping[str, Any]:
        if not self._selected_lookup_exhausted or self._selected_lookup_receipt is None:
            raise FeverousFormalSourceError(
                "selected lookup receipt is unavailable before exhaustion"
            )
        return self._selected_lookup_receipt

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._closed = True

    def __enter__(self) -> "ControlledTrainSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def verify_annotation_receipt(receipt: Mapping[str, Any]) -> str:
    declared = _verify_self_hash(receipt, "annotation_receipt_sha256")
    body = dict(receipt)
    if (
        body.get("schema") != ANNOTATION_RECEIPT_SCHEMA
        or body.get("version") != VERSION
        or body.get("status") != "train_annotation_read_once_and_verified"
        or body.get("source_split") != "TRAIN"
        or body.get("source_binding_sha256")
        != FROZEN_TRAIN_BINDING.binding_sha256
        or not _is_sha256(body.get("formal_source_opener_source_sha256"))
        or body.get("annotation_file_read_count") != 1
        or body.get("development_or_test_source_accessed") is not False
        or body.get("online_evaluator_calls") != 0
    ):
        raise FeverousFormalSourceError("annotation receipt drifted")
    return declared


def verify_database_page_stream_receipt(receipt: Mapping[str, Any]) -> str:
    declared = _verify_self_hash(receipt, "database_page_stream_receipt_sha256")
    body = dict(receipt)
    if (
        body.get("schema") != DATABASE_RECEIPT_SCHEMA
        or body.get("version") != VERSION
        or body.get("status") != "complete_database_page_stream_exhausted"
        or body.get("source_split") != "TRAIN"
        or body.get("source_binding_sha256")
        != FROZEN_TRAIN_BINDING.binding_sha256
        or not _is_sha256(body.get("formal_source_opener_source_sha256"))
        or body.get("stream_fully_exhausted") is not True
        or body.get("expected_database_row_count")
        != body.get("observed_database_row_count")
        or body.get("all_page_ids_or_pages_materialized") is not False
        or body.get("development_or_test_source_accessed") is not False
        or body.get("online_evaluator_calls") != 0
    ):
        raise FeverousFormalSourceError("database page-stream receipt drifted")
    return declared


def require_formal_database_page_stream_receipt(
    receipt: Mapping[str, Any],
) -> str:
    declared = verify_database_page_stream_receipt(receipt)
    body = dict(receipt)
    if (
        body.get("formal_source") is not True
        or body.get("source_spec_sha256") != FROZEN_TRAIN_SOURCE_SPEC.spec_sha256
        or body.get("database_basename") != FROZEN_DATABASE_BASENAME
        or body.get("database_size_bytes") != FROZEN_DATABASE_SIZE_BYTES
        or body.get("database_file_sha256") != FROZEN_DATABASE_SHA256
        or body.get("expected_database_row_count") != FROZEN_DATABASE_ROW_COUNT
    ):
        raise FeverousFormalSourceError(
            "synthetic or partial database source is not formal-valid"
        )
    return declared


def verify_selected_page_lookup_receipt(receipt: Mapping[str, Any]) -> str:
    declared = _verify_self_hash(receipt, "selected_page_lookup_receipt_sha256")
    body = dict(receipt)
    if (
        body.get("schema") != SELECTED_LOOKUP_RECEIPT_SCHEMA
        or body.get("version") != VERSION
        or body.get("status")
        != "selected_pages_materialized_after_full_universe_exhaustion"
        or body.get("source_split") != "TRAIN"
        or not _is_sha256(body.get("database_page_stream_receipt_sha256"))
        or not _is_sha256(body.get("formal_source_opener_source_sha256"))
        or type(body.get("selected_page_count")) is not int
        or body["selected_page_count"] < 0
        or body.get("full_database_rescan") is not False
        or body.get("development_or_test_source_accessed") is not False
        or body.get("online_evaluator_calls") != 0
    ):
        raise FeverousFormalSourceError("selected-page lookup receipt drifted")
    return declared

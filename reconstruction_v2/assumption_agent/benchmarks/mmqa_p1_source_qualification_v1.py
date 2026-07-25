"""One-shot aggregate-only qualification of the pinned MMQA P1 source.

The formal path opens exactly four commit-pinned gzip JSONL files.  It first
verifies every compressed byte stream by size and Git-blob SHA-1, then parses
the files with bounded gzip and JSONL readers.  The only public result is an
aggregate schema/capacity receipt.  Questions, documents, answers, support
annotations, source identifiers, and identifier-derived digests are never
serialized.

This module does not run a model, form an action, score a prediction, invoke an
online evaluator, or select formal items.  Tests exercise synthetic fixtures;
they must never open any formal source file.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
import unicodedata
from urllib.parse import unquote, urlsplit


VERSION = "mmqa_p1_source_qualification_v1"
STUDY_ID = "MMQA_P1_LOCAL_PROOF_E5_V1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUALIFIER_PATH = Path(__file__).resolve()
TEST_PATH = PROJECT_ROOT / "tests/test_mmqa_p1_source_qualification_v1.py"
SOURCE_ROOT = PROJECT_ROOT / "artifacts/mmqa_p1_official_source_v1"
CUSTODY_PATH = PROJECT_ROOT / "manifests/mmqa_p1_source_custody_v1.json"
DESIGN_PATH = (
    PROJECT_ROOT / "manifests/mmqa_p1_local_proof_e5_study_design_v1.json"
)
DOWNLOAD_AUTHORIZATION_PATH = (
    PROJECT_ROOT / "manifests/mmqa_p1_source_download_authorization_v1.json"
)
FREEZE_PATH = (
    PROJECT_ROOT / "manifests/mmqa_p1_source_qualification_freeze_v1.json"
)
MARKER_PATH = (
    PROJECT_ROOT
    / "artifacts/mmqa_p1_source_qualification_v1/qualification.one_shot_marker.json"
)
SOURCE_OPEN_MARKER_PATH = (
    PROJECT_ROOT
    / "artifacts/mmqa_p1_source_qualification_v1/source_open.one_shot_marker.json"
)
FAILURE_PATH = (
    PROJECT_ROOT
    / "artifacts/mmqa_p1_source_qualification_v1/qualification.terminal_failure.json"
)
RESULT_PATH = PROJECT_ROOT / "manifests/mmqa_p1_source_qualification_result_v1.json"

EXPECTED_CUSTODY_SELF_SHA256 = (
    "e82cb94e54a3020d1f2e41f47ed4141d19b448db985479551b1d933b43bf15f5"
)
EXPECTED_DESIGN_SELF_SHA256 = (
    "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
)
EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256 = (
    "08f4bbc25c7d15182b16da909d535a4492e80c302940742e1e92c2828d7360cb"
)

FAMILY_BY_EXACT_TYPE = {
    "Compose(TextQ,TableQ)": "FIRST_TABLE_THEN_TEXT",
    "Compose(TableQ,TextQ)": "FIRST_TEXT_THEN_TABLE",
    "Intersect(TableQ,TextQ)": "TABLE_TEXT_INTERSECTION",
}
FAMILIES = tuple(FAMILY_BY_EXACT_TYPE.values())
DEV_BLOCK_QUOTAS = {
    "F_search": 5,
    "A_hold": 15,
    "M_search": 15,
}
TRAIN_QUOTA_PER_FAMILY = 40

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WHITESPACE = re.compile(r"\s+")


class MMQAP1SourceQualificationError(RuntimeError):
    """The frozen source or aggregate-only lifecycle violated its contract."""


@dataclass(frozen=True)
class SourceFileContract:
    file_name: str
    size_bytes: int
    git_blob_sha1: str
    maximum_uncompressed_bytes: int
    maximum_records: int
    maximum_line_bytes: int = 16_000_000


@dataclass(frozen=True)
class QualificationContract:
    files: Mapping[str, SourceFileContract]
    expected_train_rows: int
    expected_dev_rows: int
    train_quota_per_family: int
    dev_block_quotas: Mapping[str, int]
    maximum_identifier_characters: int = 2048
    maximum_candidate_texts: int = 48
    maximum_gold_rows: int = 4
    maximum_gold_texts: int = 4


FORMAL_CONTRACT = QualificationContract(
    files={
        "MMQA_train.jsonl.gz": SourceFileContract(
            "MMQA_train.jsonl.gz",
            11_698_210,
            "a6f55fedf35225a217defa3777338f66716304a2",
            750_000_000,
            100_000,
        ),
        "MMQA_dev.jsonl.gz": SourceFileContract(
            "MMQA_dev.jsonl.gz",
            1_310_976,
            "7b268187629fe10e2f7678b039baf49c50b29e80",
            100_000_000,
            25_000,
        ),
        "MMQA_tables.jsonl.gz": SourceFileContract(
            "MMQA_tables.jsonl.gz",
            10_344_191,
            "c2a8c4add0f12c60cdedd91ab193483bfe0ffa6f",
            2_000_000_000,
            100_000,
        ),
        "MMQA_texts.jsonl.gz": SourceFileContract(
            "MMQA_texts.jsonl.gz",
            45_851_194,
            "debfcc4389f2ddd84647f8b6a2bde3ef41431343",
            2_000_000_000,
            500_000,
        ),
    },
    expected_train_rows=23_817,
    expected_dev_rows=2_441,
    train_quota_per_family=TRAIN_QUOTA_PER_FAMILY,
    dev_block_quotas=DEV_BLOCK_QUOTAS,
)


@dataclass(frozen=True)
class TextInfo:
    exact_link_keys: frozenset[str]


@dataclass(frozen=True)
class TableInfo:
    row_exact_link_keys: tuple[frozenset[str], ...]
    row_cell_counts: tuple[int, ...]


@dataclass(frozen=True)
class EligibleItem:
    family: str
    component_resources: frozenset[str]


@dataclass(frozen=True)
class ComponentVector:
    counts: tuple[int, ...]
    private_order_key: str


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MMQAP1SourceQualificationError(
            "aggregate receipt is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()  # nosec B324: immutable Git object identity
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise MMQAP1SourceQualificationError(
                "aggregate receipt parent is unsafe"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise MMQAP1SourceQualificationError(
                    "aggregate receipt parent is unavailable"
                )
            missing.append(cursor)
            cursor = cursor.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MMQAP1SourceQualificationError(
                "aggregate receipt parent is unsafe"
            )
        break
    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    _ensure_durable_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _load_verified_manifest(
    path: Path, expected_self_sha256: str
) -> Mapping[str, Any]:
    if not _HEX64.fullmatch(expected_self_sha256):
        raise MMQAP1SourceQualificationError("manifest binding is not frozen")
    try:
        value = json.loads(path.read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MMQAP1SourceQualificationError(
            "bound manifest is unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise MMQAP1SourceQualificationError("bound manifest shape drifted")
    if value.get("self_sha256") != expected_self_sha256:
        raise MMQAP1SourceQualificationError("bound manifest self hash drifted")
    body = dict(value)
    body.pop("self_sha256", None)
    if _semantic_hash(body) != expected_self_sha256:
        raise MMQAP1SourceQualificationError(
            "bound manifest semantic hash drifted"
        )
    if value.get("study_id") != STUDY_ID:
        raise MMQAP1SourceQualificationError("bound study identity drifted")
    return value


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise MMQAP1SourceQualificationError(
            "bound implementation file is unavailable"
        ) from exc


def _load_and_verify_freeze() -> tuple[Mapping[str, Any], str]:
    try:
        value = json.loads(FREEZE_PATH.read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MMQAP1SourceQualificationError(
            "qualification freeze is unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise MMQAP1SourceQualificationError(
            "qualification freeze shape drifted"
        )
    claimed = value.get("self_sha256")
    if not isinstance(claimed, str) or not _HEX64.fullmatch(claimed):
        raise MMQAP1SourceQualificationError(
            "qualification freeze self hash is invalid"
        )
    body = dict(value)
    body.pop("self_sha256", None)
    if _semantic_hash(body) != claimed:
        raise MMQAP1SourceQualificationError(
            "qualification freeze semantic hash drifted"
        )
    required = {
        "schema": "mmqa_p1_source_qualification_freeze_v1",
        "status": "frozen_before_unique_formal_qualification",
        "study_id": STUDY_ID,
        "source_custody_self_sha256": EXPECTED_CUSTODY_SELF_SHA256,
        "study_design_self_sha256": EXPECTED_DESIGN_SELF_SHA256,
        "download_authorization_self_sha256": (
            EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256
        ),
        "qualifier_sha256": _file_sha256(QUALIFIER_PATH),
        "test_sha256": _file_sha256(TEST_PATH),
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise MMQAP1SourceQualificationError(
                "qualification freeze binding drifted"
            )
    source_sha256 = value.get("source_sha256_by_file")
    if not isinstance(source_sha256, Mapping) or set(source_sha256) != set(
        FORMAL_CONTRACT.files
    ):
        raise MMQAP1SourceQualificationError(
            "qualification freeze source binding drifted"
        )
    if any(
        not isinstance(digest, str) or not _HEX64.fullmatch(digest)
        for digest in source_sha256.values()
    ):
        raise MMQAP1SourceQualificationError(
            "qualification freeze source digest is invalid"
        )
    return value, claimed


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise MMQAP1SourceQualificationError(
                "source JSON contains a duplicate object key"
            )
        value[key] = item
    return value


def _reject_nonfinite(_value: str) -> None:
    raise MMQAP1SourceQualificationError(
        "source JSON contains a non-finite number"
    )


def _parse_json_line(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite,
        )
    except MMQAP1SourceQualificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise MMQAP1SourceQualificationError(
            "source JSONL record is invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise MMQAP1SourceQualificationError(
            "source JSONL record is not an object"
        )
    return value


def _read_verified_regular_file(
    path: Path, contract: SourceFileContract
) -> tuple[bytes, str]:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise MMQAP1SourceQualificationError(
            "fixed source file is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o077
        ):
            raise MMQAP1SourceQualificationError(
                "fixed source file is not a private regular file"
            )
        if before.st_size != contract.size_bytes:
            raise MMQAP1SourceQualificationError(
                "fixed source file size drifted"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 << 20, remaining))
            if not chunk:
                raise MMQAP1SourceQualificationError(
                    "fixed source file ended early"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise MMQAP1SourceQualificationError(
                "fixed source file grew during read"
            )
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if not stable:
            raise MMQAP1SourceQualificationError(
                "fixed source file changed during read"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if not _HEX40.fullmatch(contract.git_blob_sha1):
        raise MMQAP1SourceQualificationError(
            "fixed source Git-blob binding is invalid"
        )
    if _git_blob_sha1(raw) != contract.git_blob_sha1:
        raise MMQAP1SourceQualificationError(
            "fixed source Git-blob identity drifted"
        )
    return raw, hashlib.sha256(raw).hexdigest()


def _iter_gzip_jsonl(
    raw: bytes, contract: SourceFileContract
) -> Iterable[Mapping[str, Any]]:
    total = 0
    count = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
            while True:
                line = handle.readline(contract.maximum_line_bytes + 1)
                if not line:
                    break
                total += len(line)
                count += 1
                if (
                    len(line) > contract.maximum_line_bytes
                    or total > contract.maximum_uncompressed_bytes
                    or count > contract.maximum_records
                ):
                    raise MMQAP1SourceQualificationError(
                        "bounded gzip JSONL contract was exceeded"
                    )
                if not line.endswith(b"\n") and len(line) == (
                    contract.maximum_line_bytes + 1
                ):
                    raise MMQAP1SourceQualificationError(
                        "bounded gzip JSONL line was exceeded"
                    )
                stripped = line.strip()
                if not stripped:
                    raise MMQAP1SourceQualificationError(
                        "source JSONL contains a blank record"
                    )
                yield _parse_json_line(stripped)
    except MMQAP1SourceQualificationError:
        raise
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise MMQAP1SourceQualificationError(
            "fixed source gzip stream is invalid"
        ) from exc
    if count == 0:
        raise MMQAP1SourceQualificationError("source JSONL is empty")


def _safe_identifier(value: object, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise MMQAP1SourceQualificationError(
            "source identifier schema drifted"
        )
    return value


def _require_text(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise MMQAP1SourceQualificationError("source text schema drifted")
    if not allow_empty and not value.strip():
        raise MMQAP1SourceQualificationError("source text schema drifted")
    return value


def _canonical_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("_", " ")
    return _WHITESPACE.sub(" ", normalized).strip().casefold()


def _canonical_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise MMQAP1SourceQualificationError("source URL schema drifted") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise MMQAP1SourceQualificationError("source URL schema drifted")
    host = parsed.netloc.casefold()
    path = unquote(parsed.path)
    if host.endswith("wikipedia.org") and path.startswith("/wiki/"):
        return "wikipedia:" + _canonical_title(path[len("/wiki/") :])
    return "url:" + host + path.rstrip("/")


def _document_exact_link_keys(title: str, url: str) -> frozenset[str]:
    return frozenset(
        {
            "title:" + _canonical_title(title),
            "url:" + _canonical_url(url),
        }
    )


def _link_exact_keys(value: Mapping[str, Any]) -> frozenset[str]:
    keys: set[str] = set()
    if "wiki_title" in value:
        title = _require_text(value.get("wiki_title"))
        keys.add("title:" + _canonical_title(title))
    if "url" in value:
        url = _require_text(value.get("url"))
        keys.add("url:" + _canonical_url(url))
    if not keys:
        raise MMQAP1SourceQualificationError(
            "table link schema has no exact-link field"
        )
    return frozenset(keys)


def _load_text_corpus(
    raw: bytes,
    source_contract: SourceFileContract,
    contract: QualificationContract,
) -> tuple[dict[str, TextInfo], int]:
    texts: dict[str, TextInfo] = {}
    for row in _iter_gzip_jsonl(raw, source_contract):
        if not {"id", "title", "url", "text"}.issubset(row):
            raise MMQAP1SourceQualificationError(
                "text corpus required fields are missing"
            )
        identifier = _safe_identifier(
            row.get("id"), contract.maximum_identifier_characters
        )
        if identifier in texts:
            raise MMQAP1SourceQualificationError(
                "text corpus contains a duplicate identifier"
            )
        title = _require_text(row.get("title"))
        url = _require_text(row.get("url"))
        _require_text(row.get("text"))
        texts[identifier] = TextInfo(_document_exact_link_keys(title, url))
    return texts, len(texts)


def _load_table_corpus(
    raw: bytes,
    source_contract: SourceFileContract,
    contract: QualificationContract,
) -> tuple[dict[str, TableInfo], int, int]:
    tables: dict[str, TableInfo] = {}
    total_rows = 0
    for row in _iter_gzip_jsonl(raw, source_contract):
        if not {"id", "title", "url", "table"}.issubset(row):
            raise MMQAP1SourceQualificationError(
                "table corpus required fields are missing"
            )
        identifier = _safe_identifier(
            row.get("id"), contract.maximum_identifier_characters
        )
        if identifier in tables:
            raise MMQAP1SourceQualificationError(
                "table corpus contains a duplicate identifier"
            )
        _require_text(row.get("title"))
        _require_text(row.get("url"))
        table = row.get("table")
        if not isinstance(table, Mapping):
            raise MMQAP1SourceQualificationError("table schema drifted")
        table_rows = table.get("table_rows")
        header = table.get("header")
        _require_text(table.get("table_name"), allow_empty=True)
        if (
            not isinstance(table_rows, list)
            or not table_rows
            or not isinstance(header, list)
            or not header
        ):
            raise MMQAP1SourceQualificationError("table schema drifted")
        row_keys: list[frozenset[str]] = []
        row_cell_counts: list[int] = []
        for source_row in table_rows:
            if not isinstance(source_row, list) or not source_row:
                raise MMQAP1SourceQualificationError(
                    "table row schema drifted"
                )
            exact_keys: set[str] = set()
            for cell in source_row:
                if not isinstance(cell, Mapping):
                    raise MMQAP1SourceQualificationError(
                        "table cell schema drifted"
                    )
                _require_text(cell.get("text"), allow_empty=True)
                links = cell.get("links")
                if not isinstance(links, list):
                    raise MMQAP1SourceQualificationError(
                        "table cell link schema drifted"
                    )
                for link in links:
                    if not isinstance(link, Mapping):
                        raise MMQAP1SourceQualificationError(
                            "table link schema drifted"
                        )
                    exact_keys.update(_link_exact_keys(link))
            row_keys.append(frozenset(exact_keys))
            row_cell_counts.append(len(source_row))
        for column in header:
            if not isinstance(column, Mapping):
                raise MMQAP1SourceQualificationError(
                    "table header schema drifted"
                )
            _require_text(column.get("column_name"), allow_empty=True)
        tables[identifier] = TableInfo(
            tuple(row_keys), tuple(row_cell_counts)
        )
        total_rows += len(row_keys)
    return tables, len(tables), total_rows


def _optional_identifier_list(
    value: object, maximum_identifier_characters: int
) -> list[str] | None:
    if not isinstance(value, list):
        return None
    result: list[str] = []
    try:
        for item in value:
            result.append(
                _safe_identifier(item, maximum_identifier_characters)
            )
    except MMQAP1SourceQualificationError:
        return None
    if len(set(result)) != len(result):
        return None
    return result


def _eligible_item(
    row: Mapping[str, Any],
    texts: Mapping[str, TextInfo],
    tables: Mapping[str, TableInfo],
    contract: QualificationContract,
) -> EligibleItem | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise MMQAP1SourceQualificationError(
            "question metadata schema drifted"
        )
    exact_type = metadata.get("type")
    if not isinstance(exact_type, str):
        raise MMQAP1SourceQualificationError(
            "question type schema drifted"
        )
    family = FAMILY_BY_EXACT_TYPE.get(exact_type)
    if family is None:
        return None

    modalities = metadata.get("modalities")
    if (
        not isinstance(modalities, list)
        or any(not isinstance(item, str) for item in modalities)
        or len(modalities) != 2
        or set(modalities) != {"text", "table"}
    ):
        return None
    try:
        table_id = _safe_identifier(
            metadata.get("table_id"), contract.maximum_identifier_characters
        )
    except MMQAP1SourceQualificationError:
        return None
    table = tables.get(table_id)
    if table is None:
        return None
    text_ids = _optional_identifier_list(
        metadata.get("text_doc_ids"), contract.maximum_identifier_characters
    )
    if (
        not text_ids
        or len(text_ids) > contract.maximum_candidate_texts
        or any(identifier not in texts for identifier in text_ids)
    ):
        return None

    supporting = row.get("supporting_context")
    if not isinstance(supporting, list) or not supporting:
        return None
    gold_text_ids: set[str] = set()
    gold_table_ids: set[str] = set()
    seen_support: set[tuple[str, str]] = set()
    for support in supporting:
        if not isinstance(support, Mapping):
            return None
        try:
            doc_id = _safe_identifier(
                support.get("doc_id"), contract.maximum_identifier_characters
            )
        except MMQAP1SourceQualificationError:
            return None
        part = support.get("doc_part")
        if part not in {"text", "table"}:
            return None
        pair = (part, doc_id)
        if pair in seen_support:
            return None
        seen_support.add(pair)
        if part == "text":
            gold_text_ids.add(doc_id)
        else:
            gold_table_ids.add(doc_id)
    if (
        gold_table_ids != {table_id}
        or not 1 <= len(gold_text_ids) <= contract.maximum_gold_texts
        or not gold_text_ids.issubset(text_ids)
        or any(identifier not in texts for identifier in gold_text_ids)
        or not (set(text_ids) - gold_text_ids)
    ):
        return None

    answers = row.get("answers")
    if not isinstance(answers, list) or not answers:
        return None
    answer_rows: set[int] = set()
    for answer in answers:
        if not isinstance(answer, Mapping):
            return None
        indices = answer.get("table_indices", [])
        if indices is None:
            indices = []
        if not isinstance(indices, list):
            return None
        for index in indices:
            if (
                not isinstance(index, list)
                or len(index) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in index)
            ):
                return None
            row_index, column_index = index
            if (
                row_index < 0
                or row_index >= len(table.row_exact_link_keys)
                or column_index < 0
                or column_index >= table.row_cell_counts[row_index]
            ):
                return None
            answer_rows.add(row_index)

    gold_text_link_keys: set[str] = set()
    for identifier in gold_text_ids:
        gold_text_link_keys.update(texts[identifier].exact_link_keys)
    linked_rows = {
        index
        for index, row_keys in enumerate(table.row_exact_link_keys)
        if row_keys.intersection(gold_text_link_keys)
    }
    gold_rows = answer_rows | linked_rows
    if (
        not linked_rows
        or not 1 <= len(gold_rows) <= contract.maximum_gold_rows
    ):
        return None

    resources = {"table:" + table_id}
    resources.update("text:" + identifier for identifier in text_ids)
    return EligibleItem(family, frozenset(resources))


def _load_question_split(
    raw: bytes,
    source_contract: SourceFileContract,
    contract: QualificationContract,
    texts: Mapping[str, TextInfo],
    tables: Mapping[str, TableInfo],
    global_question_ids: set[str],
) -> tuple[list[EligibleItem], Counter[str], int]:
    eligible: list[EligibleItem] = []
    counts: Counter[str] = Counter()
    row_count = 0
    for row in _iter_gzip_jsonl(raw, source_contract):
        row_count += 1
        if not {"qid", "question", "answers", "metadata", "supporting_context"}.issubset(
            row
        ):
            raise MMQAP1SourceQualificationError(
                "question required fields are missing"
            )
        question_id = _safe_identifier(
            row.get("qid"), contract.maximum_identifier_characters
        )
        if question_id in global_question_ids:
            raise MMQAP1SourceQualificationError(
                "question identifier is duplicated across formal splits"
            )
        global_question_ids.add(question_id)
        _require_text(row.get("question"))
        item = _eligible_item(row, texts, tables, contract)
        if item is not None:
            eligible.append(item)
            counts[item.family] += 1
    return eligible, counts, row_count


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.weight[left] < self.weight[right]:
            left, right = right, left
        self.parent[right] = left
        self.weight[left] += self.weight[right]


def _component_vectors(items: Sequence[EligibleItem]) -> list[ComponentVector]:
    disjoint = _DisjointSet(len(items))
    owner: dict[str, int] = {}
    for index, item in enumerate(items):
        for resource in item.component_resources:
            previous = owner.setdefault(resource, index)
            disjoint.union(index, previous)
    member_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        member_indices[disjoint.find(index)].append(index)
    family_index = {family: index for index, family in enumerate(FAMILIES)}
    result: list[ComponentVector] = []
    for indices in member_indices.values():
        counts = [0] * len(FAMILIES)
        resources: set[str] = set()
        for index in indices:
            item = items[index]
            counts[family_index[item.family]] += 1
            resources.update(item.component_resources)
        private_order_key = hashlib.sha256(
            _canonical_bytes({"resources": sorted(resources)}).rstrip(b"\n")
        ).hexdigest()
        result.append(ComponentVector(tuple(counts), private_order_key))
    return sorted(result, key=lambda component: component.private_order_key)


def _construct_dev_allocation(
    components: Sequence[ComponentVector],
    block_quotas: Mapping[str, int],
) -> dict[str, Any]:
    """Return one conservative constructive component-disjoint certificate.

    Component assignment is atomic; unused rows inside an assigned component
    remain unused.  All six block orders and four frozen scoring policies are
    attempted.  This is a sufficient certificate, not an estimator or a gate
    tuned to source contents.
    """

    block_names = tuple(block_quotas)
    strategies = ("balanced", "scarce", "compact", "coverage")

    def attempt(order: Sequence[str], strategy: str) -> dict[str, list[int]] | None:
        remaining = set(range(len(components)))
        assigned: dict[str, list[int]] = {}
        for position, block in enumerate(order):
            quota = block_quotas[block]
            capacity = [0] * len(FAMILIES)
            chosen: list[int] = []
            future_required = sum(
                block_quotas[name] for name in order[position + 1 :]
            )
            while any(value < quota for value in capacity):
                remaining_totals = [
                    sum(components[index].counts[family] for index in remaining)
                    for family in range(len(FAMILIES))
                ]
                candidates: list[tuple[tuple[Any, ...], int]] = []
                for index in remaining:
                    vector = components[index].counts
                    gains = tuple(
                        min(max(quota - capacity[family], 0), vector[family])
                        for family in range(len(FAMILIES))
                    )
                    if not any(gains):
                        continue
                    if any(
                        remaining_totals[family] - vector[family]
                        < future_required
                        for family in range(len(FAMILIES))
                    ):
                        continue
                    gain_total = sum(gains)
                    overshoot = sum(
                        max(capacity[family] + vector[family] - quota, 0)
                        for family in range(len(FAMILIES))
                    )
                    scarce_gain = sum(
                        gains[family] / max(remaining_totals[family], 1)
                        for family in range(len(FAMILIES))
                    )
                    covered_deficits = sum(value > 0 for value in gains)
                    if strategy == "balanced":
                        primary = (covered_deficits, gain_total, scarce_gain)
                    elif strategy == "scarce":
                        primary = (scarce_gain, covered_deficits, gain_total)
                    elif strategy == "compact":
                        primary = (-overshoot, gain_total, covered_deficits)
                    else:
                        primary = (gain_total, covered_deficits, -overshoot)
                    key = (
                        *primary,
                        -overshoot,
                        -sum(vector),
                        components[index].private_order_key,
                    )
                    candidates.append((key, index))
                if not candidates:
                    return None
                _, selected = max(candidates)
                chosen.append(selected)
                remaining.remove(selected)
                vector = components[selected].counts
                capacity = [
                    min(quota, capacity[family] + vector[family])
                    for family in range(len(FAMILIES))
                ]
            assigned[block] = chosen
        return assigned

    allocation: dict[str, list[int]] | None = None
    winning_order: Sequence[str] | None = None
    winning_strategy: str | None = None
    for order in itertools.permutations(block_names):
        for strategy in strategies:
            allocation = attempt(order, strategy)
            if allocation is not None:
                winning_order = order
                winning_strategy = strategy
                break
        if allocation is not None:
            break
    if allocation is None or winning_order is None or winning_strategy is None:
        raise MMQAP1SourceQualificationError(
            "DEV component-disjoint quota capacity is insufficient"
        )
    assigned_indices = {
        index for indices in allocation.values() for index in indices
    }
    largest = max((sum(component.counts) for component in components), default=0)
    return {
        "allocation_method": (
            "frozen_constructive_component_atomic_all_block_orders_v1"
        ),
        "assigned_component_count": len(assigned_indices),
        "block_capacity": {
            block: {family: block_quotas[block] for family in FAMILIES}
            for block in block_names
        },
        "component_count": len(components),
        "largest_component_eligible_item_count": largest,
        "qualified": True,
        "winning_block_order": list(winning_order),
        "winning_scoring_policy": winning_strategy,
    }


def _qualify_sources(
    source_paths: Mapping[str, Path],
    contract: QualificationContract,
    *,
    expected_sha256_by_file: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if set(source_paths) != set(contract.files):
        raise MMQAP1SourceQualificationError(
            "formal source file set drifted"
        )
    compressed: dict[str, bytes] = {}
    observed_sha256: dict[str, str] = {}
    # No gzip or JSONL semantic access occurs until all four byte identities pass.
    for file_name in sorted(contract.files):
        raw, sha256 = _read_verified_regular_file(
            source_paths[file_name], contract.files[file_name]
        )
        compressed[file_name] = raw
        observed_sha256[file_name] = sha256
    if expected_sha256_by_file is not None and observed_sha256 != dict(
        expected_sha256_by_file
    ):
        raise MMQAP1SourceQualificationError(
            "fixed source SHA256 freeze binding drifted"
        )

    texts, text_count = _load_text_corpus(
        compressed["MMQA_texts.jsonl.gz"],
        contract.files["MMQA_texts.jsonl.gz"],
        contract,
    )
    tables, table_count, table_row_count = _load_table_corpus(
        compressed["MMQA_tables.jsonl.gz"],
        contract.files["MMQA_tables.jsonl.gz"],
        contract,
    )
    question_ids: set[str] = set()
    train_items, train_counts, train_rows = _load_question_split(
        compressed["MMQA_train.jsonl.gz"],
        contract.files["MMQA_train.jsonl.gz"],
        contract,
        texts,
        tables,
        question_ids,
    )
    dev_items, dev_counts, dev_rows = _load_question_split(
        compressed["MMQA_dev.jsonl.gz"],
        contract.files["MMQA_dev.jsonl.gz"],
        contract,
        texts,
        tables,
        question_ids,
    )
    if train_rows != contract.expected_train_rows or dev_rows != contract.expected_dev_rows:
        raise MMQAP1SourceQualificationError(
            "formal question split count drifted"
        )
    if any(
        train_counts[family] < contract.train_quota_per_family
        for family in FAMILIES
    ):
        raise MMQAP1SourceQualificationError(
            "TRAIN exact-support family capacity is insufficient"
        )
    dev_required = sum(contract.dev_block_quotas.values())
    if any(dev_counts[family] < dev_required for family in FAMILIES):
        raise MMQAP1SourceQualificationError(
            "DEV exact-support family capacity is insufficient"
        )
    components = _component_vectors(dev_items)
    dev_certificate = _construct_dev_allocation(
        components, contract.dev_block_quotas
    )
    return {
        "DEV": {
            "component_disjoint_capacity": dev_certificate,
            "eligible_count_by_family": {
                family: dev_counts[family] for family in FAMILIES
            },
            "question_record_count": dev_rows,
            "required_total_per_family": dev_required,
        },
        "TRAIN": {
            "eligible_count_by_family": {
                family: train_counts[family] for family in FAMILIES
            },
            "question_record_count": train_rows,
            "required_per_family": contract.train_quota_per_family,
        },
        "exact_type_family_count": len(FAMILIES),
        "qualified": True,
        "schema_aggregates": {
            "table_record_count": table_count,
            "table_row_count": table_row_count,
            "text_record_count": text_count,
        },
        "source_identity": {
            file_name: {
                "git_blob_sha1": contract.files[file_name].git_blob_sha1,
                "sha256": observed_sha256[file_name],
                "size_bytes": contract.files[file_name].size_bytes,
            }
            for file_name in sorted(contract.files)
        },
        "support_contract": {
            "answer_table_index_rows_union_exact_linked_rows": True,
            "gold_row_bounds_inclusive": [1, contract.maximum_gold_rows],
            "gold_text_bounds_inclusive": [1, contract.maximum_gold_texts],
            "identifier_or_content_output_count": 0,
            "requires_exact_gold_row_text_pair": True,
            "support_parts": ["table", "text"],
        },
    }


def _consume_marker() -> str:
    body = {
        "model_action_embedding_reranking_or_score_count": 0,
        "online_evaluator_call_count": 0,
        "retry_replay_resample_or_contract_revision": 0,
        "schema": f"{VERSION}_one_shot_marker_v1",
        "source_file_count": len(FORMAL_CONTRACT.files),
        "source_item_query_document_answer_support_or_identifier_output_count": 0,
        "status": "started_before_manifest_validation_or_source_open",
        "study_id": STUDY_ID,
    }
    value = {**body, "self_sha256": _semantic_hash(body)}
    return _write_exclusive(MARKER_PATH, value)


def _consume_source_open_marker() -> str:
    body = {
        "model_action_embedding_reranking_or_score_count": 0,
        "online_evaluator_call_count": 0,
        "schema": f"{VERSION}_source_open_marker_v1",
        "source_item_query_document_answer_support_or_identifier_output_count": 0,
        "status": "consumed_immediately_before_four_fixed_source_opens",
        "study_id": STUDY_ID,
    }
    value = {**body, "self_sha256": _semantic_hash(body)}
    return _write_exclusive(SOURCE_OPEN_MARKER_PATH, value)


def run_formal_qualification() -> Mapping[str, Any]:
    marker_file_sha256 = _consume_marker()
    stage = "validate_frozen_bindings"
    try:
        _load_verified_manifest(CUSTODY_PATH, EXPECTED_CUSTODY_SELF_SHA256)
        _load_verified_manifest(DESIGN_PATH, EXPECTED_DESIGN_SELF_SHA256)
        _load_verified_manifest(
            DOWNLOAD_AUTHORIZATION_PATH,
            EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256,
        )
        freeze, freeze_self_sha256 = _load_and_verify_freeze()
        expected_sha256 = freeze["source_sha256_by_file"]
        stage = "open_and_aggregate_four_fixed_sources"
        source_open_marker_file_sha256 = _consume_source_open_marker()
        aggregate = _qualify_sources(
            {
                file_name: SOURCE_ROOT / file_name
                for file_name in FORMAL_CONTRACT.files
            },
            FORMAL_CONTRACT,
            expected_sha256_by_file=expected_sha256,
        )
        stage = "write_aggregate_qualification_result"
        body: dict[str, Any] = {
            **aggregate,
            "binding_self_sha256": {
                "download_authorization": (
                    EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256
                ),
                "qualification_freeze": freeze_self_sha256,
                "source_custody": EXPECTED_CUSTODY_SELF_SHA256,
                "study_design": EXPECTED_DESIGN_SELF_SHA256,
            },
            "marker_file_sha256": marker_file_sha256,
            "model_action_embedding_reranking_or_score_count": 0,
            "online_evaluator_call_count": 0,
            "schema": f"{VERSION}_result_v1",
            "source_item_query_document_answer_support_or_identifier_output_count": 0,
            "source_open_marker_file_sha256": source_open_marker_file_sha256,
            "status": "qualified_aggregate_only",
            "study_id": STUDY_ID,
        }
        value = {**body, "self_sha256": _semantic_hash(body)}
        _write_exclusive(RESULT_PATH, value)
        return value
    except Exception:
        failure_body = {
            "error_code": "frozen_aggregate_only_contract_failure",
            "failure_stage": stage,
            "marker_file_sha256": marker_file_sha256,
            "model_action_embedding_reranking_or_score_count": 0,
            "online_evaluator_call_count": 0,
            "qualified": False,
            "retry_replay_resample_or_contract_revision": 0,
            "schema": f"{VERSION}_terminal_failure_v1",
            "source_item_query_document_answer_support_or_identifier_output_count": 0,
            "status": "terminal_failure_no_retry",
            "study_id": STUDY_ID,
        }
        failure = {
            **failure_body,
            "self_sha256": _semantic_hash(failure_body),
        }
        with suppress(FileExistsError):
            _write_exclusive(FAILURE_PATH, failure)
        raise


def main() -> int:
    value = run_formal_qualification()
    print(_canonical_bytes(value).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

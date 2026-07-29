"""One-shot source compiler for the frozen WikiSQL UAO reality study.

The compiler accepts an already downloaded, caller-hash-bound WikiSQL 1.1
``data.tar.bz2``.  It never extracts the archive.  Only the seven authorized
train/test JSONL, table, SQLite, and version members are read into memory.
Eligible items are joined to their item-local tables, denied if they are the
public README example, and selected once by the HMAC primitives in
``wikisql_uao_reality_v1``.

Outputs are deliberately split:

* action-view packs contain only opaque ID, question, headers, types, and rows;
* minimal label packs contain only scorer-required commitments, family, and
  authoritative SQLite row ordinals (plus the sealed A_form fold);
* a controller-only provenance pack retains source lineage and SQL, and is
  never an Agent or scorer input;
* a public-safe receipt contains only aggregate counts and commitments.

There is no model action, HippoRAG, scorer, network, or online evaluator here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import sqlite3
import stat
import tarfile
import tempfile
from typing import Any, Callable, Mapping, Sequence
import unicodedata

import babel
from babel.numbers import NumberFormatError, parse_decimal

from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality


VERSION = "wikisql_uao_source_compiler_v1"
STUDY_ID = "UAO_P4_WIKISQL_TYPED_RELATIONAL_ROW_EVIDENCE_V1"
RELEASE_VERSION = "1.1"
PRODUCTION_ARCHIVE_SIZE_BYTES = 26_164_664
PRODUCTION_ARCHIVE_GIT_BLOB_SHA1 = (
    "941de4cb2ad5fa7aeb2e37d314468636ce070af7"
)
FOLD_COUNT = 4
MAX_COLUMNS = 64
MAX_QUESTION_CHARACTERS = 16_000
MAX_HEADER_OR_CELL_CHARACTERS = 16_000
BABEL_LOCALE = "zh_CN"
PRODUCTION_BABEL_VERSION = "2.10.3"
REQUIRED_MEMBERS = (
    "data/train.jsonl",
    "data/train.tables.jsonl",
    "data/train.db",
    "data/test.jsonl",
    "data/test.tables.jsonl",
    "data/test.db",
    "data/version.txt",
)
JSONL_MEMBERS = {
    "train": "data/train.jsonl",
    "test": "data/test.jsonl",
}
TABLE_MEMBERS = {
    "train": "data/train.tables.jsonl",
    "test": "data/test.tables.jsonl",
}
DB_MEMBERS = {
    "train": "data/train.db",
    "test": "data/test.db",
}
README_DENYLIST = frozenset(
    {
        (
            "1-10007452-3",
            "who is the manufacturer for the order year 1998?",
        )
    }
)
ACTION_VIEW_FIELDS = frozenset(
    {
        "opaque_item_id",
        "question",
        "table_header",
        "table_types",
        "physical_rows",
    }
)
ACTION_FORBIDDEN_FIELDS = frozenset(
    {
        "source_table_id",
        "table_id",
        "sql",
        "agg",
        "sel",
        "conds",
        "condition_operator",
        "condition_value",
        "family",
        "gold_rows",
        "gold_row_ids",
        "answer",
        "utility",
        "score",
    }
)
LABEL_VIEW_FIELDS = frozenset(
    {
        "opaque_item_id",
        "action_view_sha256",
        "item_commitment_sha256",
        "family",
        "gold_row_ids",
        "table_row_count",
        "sqlite_rowid_cross_checked",
    }
)
PROVENANCE_FIELDS = frozenset(
    {
        "block",
        "opaque_item_id",
        "item_commitment_sha256",
        "table_commitment_sha256",
        "split",
        "source_line_number",
        "source_table_id",
        "sql",
    }
)
MAX_AUTHORIZED_MEMBER_BYTES = 1_000_000_000
MAX_JSONL_LINES = 1_000_000
MAX_JSON_LINE_BYTES = 8_000_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SQLITE_TABLE_ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
_OFFICIAL_NUMERIC_RE = re.compile(r"[-+]?\d*\.\d+|\d+")


class WikiSQLSourceCompilerError(RuntimeError):
    """The archive, compiler boundary, or output contract drifted."""


class WikiSQLSourceIneligibleError(WikiSQLSourceCompilerError):
    """Frozen source capacity cannot satisfy the preregistered cohort."""


@dataclass(frozen=True, slots=True)
class CompilerConfig:
    """Production is fixed; small quotas exist only in explicit test mode."""

    mode: str
    a_form_quota_per_family: int
    a_hold_quota_per_family: int
    expected_archive_size_bytes: int | None
    expected_archive_git_blob_sha1: str | None

    def __post_init__(self) -> None:
        if self.mode == "production":
            if (
                self.a_form_quota_per_family
                != reality.COHORT_QUOTAS["A_form"]["EQ"]
                or self.a_hold_quota_per_family
                != reality.COHORT_QUOTAS["A_hold"]["EQ"]
                or self.expected_archive_size_bytes
                != PRODUCTION_ARCHIVE_SIZE_BYTES
                or self.expected_archive_git_blob_sha1
                != PRODUCTION_ARCHIVE_GIT_BLOB_SHA1
            ):
                raise WikiSQLSourceCompilerError(
                    "production compiler quotas or archive identity drifted"
                )
        elif self.mode == "synthetic_test":
            if (
                self.expected_archive_size_bytes is not None
                or self.expected_archive_git_blob_sha1 is not None
            ):
                raise WikiSQLSourceCompilerError(
                    "synthetic config cannot freeze a production archive identity"
                )
        else:
            raise WikiSQLSourceCompilerError("compiler mode is invalid")
        if (
            type(self.a_form_quota_per_family) is not int
            or self.a_form_quota_per_family <= 0
            or self.a_form_quota_per_family % FOLD_COUNT != 0
            or type(self.a_hold_quota_per_family) is not int
            or self.a_hold_quota_per_family <= 0
        ):
            raise WikiSQLSourceCompilerError(
                "cohort quotas must be positive and A_form divisible by four"
            )

    @classmethod
    def production(cls) -> "CompilerConfig":
        return cls(
            mode="production",
            a_form_quota_per_family=64,
            a_hold_quota_per_family=24,
            expected_archive_size_bytes=PRODUCTION_ARCHIVE_SIZE_BYTES,
            expected_archive_git_blob_sha1=(
                PRODUCTION_ARCHIVE_GIT_BLOB_SHA1
            ),
        )

    @classmethod
    def synthetic_test(
        cls,
        *,
        a_form_quota_per_family: int = 4,
        a_hold_quota_per_family: int = 2,
    ) -> "CompilerConfig":
        """Explicitly authorize reduced quotas for synthetic unit tests only."""

        return cls(
            mode="synthetic_test",
            a_form_quota_per_family=a_form_quota_per_family,
            a_hold_quota_per_family=a_hold_quota_per_family,
            expected_archive_size_bytes=None,
            expected_archive_git_blob_sha1=None,
        )

    def quota(self, block: str) -> int:
        if block == "A_form":
            return self.a_form_quota_per_family
        if block == "A_hold":
            return self.a_hold_quota_per_family
        raise WikiSQLSourceCompilerError("cohort block is invalid")


@dataclass(frozen=True, slots=True)
class ArchiveRead:
    members: Mapping[str, bytes]
    member_sha256: tuple[tuple[str, str], ...]
    archive_git_blob_sha1: str
    regular_member_count: int
    directory_member_count: int
    ignored_regular_member_count: int


@dataclass(frozen=True, slots=True)
class StructuralItem:
    split: str
    line_number: int
    item_commitment_sha256: str
    table_commitment_sha256: str
    question: str
    source_table_id: str
    table: reality.WikiSQLTable
    query: reality.WikiSQLQuery
    family: str
    raw_sql: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CompiledItem(StructuralItem):
    gold_row_ids: tuple[int, ...]

    def candidate(self) -> reality.SelectionCandidate:
        return reality.SelectionCandidate(
            item_commitment_sha256=self.item_commitment_sha256,
            table_commitment_sha256=self.table_commitment_sha256,
            family=self.family,
            table_row_count=len(self.table.rows),
            gold_row_count=len(self.gold_row_ids),
        )


@dataclass(frozen=True, slots=True)
class SQLiteCrossCheckRequest:
    split: str
    source_table_id: str
    table: reality.WikiSQLTable
    query: reality.WikiSQLQuery
    expected_gold_row_ids: tuple[int, ...]
    item_commitment_sha256: str


@dataclass(frozen=True, slots=True)
class SQLiteDerivationRequest:
    split: str
    source_table_id: str
    table: reality.WikiSQLTable
    query: reality.WikiSQLQuery
    item_commitment_sha256: str


SQLiteRowIDDeriver = Callable[
    [Mapping[str, bytes], Sequence[SQLiteDerivationRequest]],
    Mapping[str, tuple[int, ...]],
]
SQLiteCrossChecker = Callable[
    [Mapping[str, bytes], Sequence[SQLiteCrossCheckRequest]],
    None,
]


@dataclass(frozen=True, slots=True)
class CompilationBundle:
    selection_secret: bytes
    a_form_action_pack: Mapping[str, object]
    a_form_label_pack: Mapping[str, object]
    a_hold_action_pack: Mapping[str, object]
    a_hold_label_pack: Mapping[str, object]
    controller_provenance_pack: Mapping[str, object]
    safe_receipt: Mapping[str, object]

    def action_pack(self, block: str) -> Mapping[str, object]:
        if block == "A_form":
            return self.a_form_action_pack
        if block == "A_hold":
            return self.a_hold_action_pack
        raise WikiSQLSourceCompilerError("action block is invalid")

    def label_pack(self, block: str) -> Mapping[str, object]:
        if block == "A_form":
            return self.a_form_label_pack
        if block == "A_hold":
            return self.a_hold_label_pack
        raise WikiSQLSourceCompilerError("label block is invalid")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WikiSQLSourceCompilerError(f"{field} is not a SHA-256 commitment")
    return value


def _hash_file(path: Path) -> tuple[str, str, int]:
    if path.is_symlink() or not path.is_file():
        raise WikiSQLSourceCompilerError("source archive is not a regular file")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise WikiSQLSourceCompilerError(
            "source archive metadata cannot be read"
        ) from exc
    digest = hashlib.sha256()
    git_blob_digest = hashlib.sha1(  # noqa: S324 - official Git identity
        f"blob {size}\0".encode("ascii")
    )
    observed_size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                observed_size += len(block)
                digest.update(block)
                git_blob_digest.update(block)
    except OSError as exc:
        raise WikiSQLSourceCompilerError("source archive cannot be hashed") from exc
    if observed_size != size:
        raise WikiSQLSourceCompilerError(
            "source archive size changed while hashing"
        )
    return digest.hexdigest(), git_blob_digest.hexdigest(), size


def _safe_tar_parts(name: object) -> tuple[str, ...]:
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
    ):
        raise WikiSQLSourceCompilerError("archive contains an unsafe member name")
    path = PurePosixPath(name)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or any(part == ".." for part in parts):
        raise WikiSQLSourceCompilerError("archive contains an unsafe member name")
    return parts


def _read_member(bundle: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if (
        not member.isfile()
        or member.size < 0
        or member.size > MAX_AUTHORIZED_MEMBER_BYTES
    ):
        raise WikiSQLSourceCompilerError(
            "authorized archive member is not a bounded regular file"
        )
    handle = bundle.extractfile(member)
    if handle is None:
        raise WikiSQLSourceCompilerError("authorized member cannot be opened")
    raw = handle.read(member.size + 1)
    if len(raw) != member.size:
        raise WikiSQLSourceCompilerError("authorized member size drifted")
    return raw


def read_authorized_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    config: CompilerConfig,
) -> ArchiveRead:
    """Verify the archive and read exactly seven authorized members."""

    expected = _required_sha256(
        expected_archive_sha256, field="expected archive SHA-256"
    )
    observed, git_blob_sha1, size = _hash_file(archive_path)
    if observed != expected:
        raise WikiSQLSourceCompilerError("source archive SHA-256 drifted")
    if (
        config.expected_archive_size_bytes is not None
        and size != config.expected_archive_size_bytes
    ):
        raise WikiSQLSourceCompilerError("source archive size drifted")
    if (
        config.expected_archive_git_blob_sha1 is not None
        and (
            _SHA1_RE.fullmatch(config.expected_archive_git_blob_sha1)
            is None
            or git_blob_sha1 != config.expected_archive_git_blob_sha1
        )
    ):
        raise WikiSQLSourceCompilerError(
            "source archive official Git blob identity drifted"
        )

    authorized: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    regular_count = 0
    directory_count = 0
    ignored_regular_count = 0
    seen_required: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:bz2", errorlevel=2) as bundle:
            for member in bundle:
                _safe_tar_parts(member.name)
                if member.isdir():
                    directory_count += 1
                    continue
                if not member.isfile():
                    raise WikiSQLSourceCompilerError(
                        "archive contains a non-regular non-directory member"
                    )
                regular_count += 1
                if member.name not in REQUIRED_MEMBERS:
                    ignored_regular_count += 1
                    continue
                if member.name in seen_required:
                    raise WikiSQLSourceCompilerError(
                        "archive duplicates an authorized member"
                    )
                raw = _read_member(bundle, member)
                authorized[member.name] = raw
                hashes[member.name] = _sha256_bytes(raw)
                seen_required.add(member.name)
    except WikiSQLSourceCompilerError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise WikiSQLSourceCompilerError("source archive scan failed") from exc
    if set(authorized) != set(REQUIRED_MEMBERS):
        missing = sorted(set(REQUIRED_MEMBERS) - set(authorized))
        raise WikiSQLSourceCompilerError(
            f"source archive lacks authorized members: {missing}"
        )
    version_raw = authorized["data/version.txt"]
    try:
        version = version_raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise WikiSQLSourceCompilerError("version.txt is not UTF-8") from exc
    if version != RELEASE_VERSION or "\x00" in version:
        raise WikiSQLSourceCompilerError("WikiSQL release version drifted")
    for member_name in DB_MEMBERS.values():
        if not authorized[member_name].startswith(b"SQLite format 3\x00"):
            raise WikiSQLSourceCompilerError("authorized DB member is not SQLite3")
    return ArchiveRead(
        members=authorized,
        member_sha256=tuple(
            (name, hashes[name]) for name in REQUIRED_MEMBERS
        ),
        archive_git_blob_sha1=git_blob_sha1,
        regular_member_count=regular_count,
        directory_member_count=directory_count,
        ignored_regular_member_count=ignored_regular_count,
    )


def _reject_json_constant(value: str) -> None:
    raise WikiSQLSourceCompilerError(f"JSON contains forbidden constant {value}")


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WikiSQLSourceCompilerError("JSON object repeats a key")
        result[key] = value
    return result


def _jsonl(raw: bytes, *, member_name: str) -> tuple[Mapping[str, object], ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WikiSQLSourceCompilerError(
            f"{member_name} is not UTF-8"
        ) from exc
    lines = text.splitlines()
    if not lines or len(lines) > MAX_JSONL_LINES:
        raise WikiSQLSourceCompilerError(
            f"{member_name} has an invalid line count"
        )
    result: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        if not line or len(line.encode("utf-8")) > MAX_JSON_LINE_BYTES:
            raise WikiSQLSourceCompilerError(
                f"{member_name}:{line_number} is blank or oversized"
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except WikiSQLSourceCompilerError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise WikiSQLSourceCompilerError(
                f"{member_name}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise WikiSQLSourceCompilerError(
                f"{member_name}:{line_number} is not an object"
            )
        result.append(value)
    return tuple(result)


def _load_tables(
    raw: bytes,
    *,
    member_name: str,
) -> dict[str, reality.WikiSQLTable]:
    result: dict[str, reality.WikiSQLTable] = {}
    for line_number, value in enumerate(
        _jsonl(raw, member_name=member_name), 1
    ):
        try:
            table = reality.table_from_documented_schema(value)
        except reality.WikiSQLUAORealityError as exc:
            raise WikiSQLSourceCompilerError(
                f"{member_name}:{line_number} table schema drifted"
            ) from exc
        if table.table_id in result:
            raise WikiSQLSourceCompilerError("table registry repeats an id")
        result[table.table_id] = table
    return result


def _normalized_question(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise WikiSQLSourceCompilerError("question is invalid")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _outer_item(
    value: Mapping[str, object],
    *,
    member_name: str,
    line_number: int,
) -> tuple[str, str, Mapping[str, object]]:
    if set(value) != reality.OFFICIAL_ITEM_FIELDS:
        raise WikiSQLSourceCompilerError(
            f"{member_name}:{line_number} item schema drifted"
        )
    if type(value["phase"]) is not int or value["phase"] < 1:
        raise WikiSQLSourceCompilerError("item phase is invalid")
    question = value["question"]
    table_id = value["table_id"]
    sql = value["sql"]
    _normalized_question(question)
    if not isinstance(table_id, str) or not table_id.strip() or "\x00" in table_id:
        raise WikiSQLSourceCompilerError("item table_id is invalid")
    if not isinstance(sql, Mapping) or set(sql) != reality.OFFICIAL_SQL_FIELDS:
        raise WikiSQLSourceCompilerError("item SQL schema drifted")
    return question, table_id, sql


def _condition_eligibility_reason(sql: Mapping[str, object]) -> str | None:
    conditions = sql["conds"]
    if (
        isinstance(conditions, (str, bytes, bytearray))
        or not isinstance(conditions, Sequence)
    ):
        raise WikiSQLSourceCompilerError("SQL conds is not an array")
    if len(conditions) != 1:
        return "condition_count_not_one"
    condition = conditions[0]
    if (
        isinstance(condition, (str, bytes, bytearray))
        or not isinstance(condition, Sequence)
        or len(condition) != 3
    ):
        raise WikiSQLSourceCompilerError("SQL condition schema drifted")
    operator = condition[1]
    if type(operator) is not int:
        raise WikiSQLSourceCompilerError("SQL condition operator is not an integer")
    if operator not in (0, 1, 2):
        return "condition_operator_not_EQ_GT_LT"
    return None


def _cell_character_count(value: str | int | float) -> int:
    if isinstance(value, str):
        return len(value)
    return len(
        json.dumps(value, allow_nan=False, ensure_ascii=False)
    )


def _structural_eligibility_reason(
    *,
    question: str,
    table: reality.WikiSQLTable,
) -> str | None:
    if not reality.MIN_TABLE_ROWS <= len(table.rows) <= reality.MAX_TABLE_ROWS:
        return "table_row_count_outside_11_80"
    if not 1 <= len(table.header) <= MAX_COLUMNS:
        return "column_count_outside_1_64"
    if len(question) > MAX_QUESTION_CHARACTERS:
        return "question_characters_over_16000"
    if any(
        len(header) > MAX_HEADER_OR_CELL_CHARACTERS
        for header in table.header
    ) or any(
        _cell_character_count(cell) > MAX_HEADER_OR_CELL_CHARACTERS
        for row in table.rows
        for cell in row
    ):
        return "header_or_cell_characters_over_16000"
    try:
        serialized = reality.serialize_table_rows(table)
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLSourceCompilerError(
            "canonical table-row serialization failed"
        ) from exc
    if any(
        len(document) > reality.MAX_SERIALIZED_ROW_CHARACTERS
        for document in serialized
    ):
        return "canonical_serialized_row_characters_over_16000"
    if len(set(serialized)) != len(serialized):
        return "duplicate_canonical_serialized_rows"
    try:
        validated = reality.validated_retrieval_documents(table)
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLSourceCompilerError(
            "canonical table row is not shared-arm representable"
        ) from exc
    if validated != serialized:
        raise WikiSQLSourceCompilerError(
            "shared-arm row representation changed canonical serialization"
        )
    return None


def _parse_structural_split_items(
    *,
    split: str,
    source_revision_sha256: str,
    query_raw: bytes,
    query_member_name: str,
    tables: Mapping[str, reality.WikiSQLTable],
) -> tuple[tuple[StructuralItem, ...], Counter[str]]:
    eligible: list[StructuralItem] = []
    dispositions: Counter[str] = Counter()
    for line_number, raw_item in enumerate(
        _jsonl(query_raw, member_name=query_member_name), 1
    ):
        question, table_id, sql = _outer_item(
            raw_item,
            member_name=query_member_name,
            line_number=line_number,
        )
        table = tables.get(table_id)
        if table is None:
            raise WikiSQLSourceCompilerError(
                "query references a table outside its split registry"
            )
        if (table_id, _normalized_question(question)) in README_DENYLIST:
            dispositions["README_example_denied"] += 1
            continue
        condition_reason = _condition_eligibility_reason(sql)
        if condition_reason is not None:
            dispositions[condition_reason] += 1
            continue
        try:
            query = reality.query_from_documented_sql(
                sql, column_count=len(table.header)
            )
        except reality.WikiSQLUAORealityError as exc:
            raise WikiSQLSourceCompilerError(
                "eligible SQL cannot be parsed under the frozen schema"
            ) from exc
        structure_reason = _structural_eligibility_reason(
            question=question,
            table=table,
        )
        if structure_reason is not None:
            dispositions[structure_reason] += 1
            continue
        try:
            identity = reality.item_identity_commitment(
                source_revision_sha256=source_revision_sha256,
                split=split,
                line_number=line_number,
                raw_item=raw_item,
            )
        except reality.WikiSQLUAORealityError as exc:
            raise WikiSQLSourceCompilerError(
                "eligible item identity cannot be committed"
            ) from exc
        eligible.append(
            StructuralItem(
                split=split,
                line_number=line_number,
                item_commitment_sha256=identity,
                table_commitment_sha256=reality.canonical_sha256(
                    {
                        "schema": f"{VERSION}_table_identity_v1",
                        "source_revision_sha256": source_revision_sha256,
                        "table_id": table_id,
                    }
                ),
                question=question,
                source_table_id=table_id,
                table=table,
                query=query,
                family=query.family,
                raw_sql=sql,
            )
        )
        dispositions["structurally_eligible_for_SQLite_derivation"] += 1
    if len({row.item_commitment_sha256 for row in eligible}) != len(eligible):
        raise WikiSQLSourceCompilerError("eligible item identities collide")
    return tuple(eligible), dispositions


def _select_block(
    *,
    secret: bytes,
    block: str,
    items: Sequence[CompiledItem],
    quota_per_family: int,
) -> tuple[CompiledItem, ...]:
    by_commitment = {row.item_commitment_sha256: row for row in items}
    if len(by_commitment) != len(items):
        raise WikiSQLSourceCompilerError("selection input repeats an item")
    by_table: dict[str, list[reality.SelectionCandidate]] = defaultdict(list)
    for row in items:
        candidate = row.candidate()
        by_table[candidate.table_commitment_sha256].append(candidate)
    table_winners = tuple(
        reality.hmac_order(
            secret,
            block=f"{block}:table",
            candidates=tuple(rows),
        )[0]
        for _, rows in sorted(by_table.items())
    )
    selected: list[CompiledItem] = []
    for family in reality.FAMILY_ORDER:
        family_rows = tuple(
            row for row in table_winners if row.family == family
        )
        if len(family_rows) < quota_per_family:
            raise WikiSQLSourceIneligibleError(
                f"{block}/{family} lacks the frozen one-table quota"
            )
        ordered = reality.hmac_order(
            secret,
            block=block,
            candidates=family_rows,
        )
        selected.extend(
            by_commitment[row.item_commitment_sha256]
            for row in ordered[:quota_per_family]
        )
    if (
        len({row.item_commitment_sha256 for row in selected}) != len(selected)
        or len({row.table_commitment_sha256 for row in selected}) != len(selected)
    ):
        raise WikiSQLSourceCompilerError("selected block is not item/table unique")
    return tuple(selected)


def _fold_assignment(
    secret: bytes,
    selected: Sequence[CompiledItem],
    *,
    quota_per_family: int,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for family in reality.FAMILY_ORDER:
        family_items = tuple(row for row in selected if row.family == family)
        ordered = reality.hmac_order(
            secret,
            block="A_form:fold",
            candidates=tuple(row.candidate() for row in family_items),
        )
        if len(ordered) != quota_per_family:
            raise WikiSQLSourceCompilerError("A_form family width drifted")
        for rank, candidate in enumerate(ordered):
            result[candidate.item_commitment_sha256] = rank % FOLD_COUNT
    counts = Counter(
        (row.family, result[row.item_commitment_sha256])
        for row in selected
    )
    expected = quota_per_family // FOLD_COUNT
    if any(
        counts[(family, fold)] != expected
        for family in reality.FAMILY_ORDER
        for fold in range(FOLD_COUNT)
    ):
        raise WikiSQLSourceCompilerError("four-fold family balance drifted")
    return result


def _sqlite_condition_value(
    value: str | int | float, column_type: str
) -> str | float:
    if column_type == "text":
        return str(value).lower()
    if isinstance(value, bool):
        raise WikiSQLSourceCompilerError("boolean SQLite condition is invalid")
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        # Match the frozen WikiSQL 1.1 DBEngine: parse the original value with
        # Babel/zh_CN first, then take the first official-regex match.
        text = value
        try:
            result = float(parse_decimal(text, locale=BABEL_LOCALE))
        except NumberFormatError:
            match = _OFFICIAL_NUMERIC_RE.search(text)
            if match is None:
                raise WikiSQLSourceCompilerError(
                    "SQLite numeric condition cannot be coerced"
                )
            result = float(match.group(0))
    if not math.isfinite(result):
        raise WikiSQLSourceCompilerError("SQLite numeric condition is non-finite")
    return result


def _verify_sqlite_table_matches_json(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    table: reality.WikiSQLTable,
) -> None:
    """Bind SQLite rowid order and normalized cells to the JSON table rows."""

    try:
        schema = tuple(
            connection.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
        )
        quoted_columns = ", ".join(
            f'"col{index}"' for index in range(len(table.header))
        )
        db_rows = tuple(
            connection.execute(
                f'SELECT rowid, {quoted_columns} FROM "{table_name}" '
                "ORDER BY rowid"
            ).fetchall()
        )
    except sqlite3.Error as exc:
        raise WikiSQLSourceCompilerError(
            "SQLite/JSON table binding query failed"
        ) from exc
    expected_names = tuple(
        f"col{index}" for index in range(len(table.header))
    )
    observed_names = tuple(str(row[1]) for row in schema)
    observed_types = tuple(str(row[2]).casefold() for row in schema)
    if (
        observed_names != expected_names
        or observed_types != tuple(table.types)
        or len(db_rows) != len(table.rows)
        or tuple(row[0] for row in db_rows)
        != tuple(range(1, len(table.rows) + 1))
    ):
        raise WikiSQLSourceCompilerError(
            "SQLite schema or rowid order disagrees with JSON table rows"
        )
    for db_row, json_row in zip(db_rows, table.rows, strict=True):
        for (
            db_value,
            json_value,
            column_type,
        ) in zip(db_row[1:], json_row, table.types, strict=True):
            if column_type == "text":
                expected: str | float = str(json_value).lower()
                matches = isinstance(db_value, str) and db_value == expected
            else:
                expected = _sqlite_condition_value(
                    json_value, column_type
                )
                matches = (
                    not isinstance(db_value, bool)
                    and isinstance(db_value, (int, float))
                    and math.isfinite(float(db_value))
                    and float(db_value) == float(expected)
                )
            if not matches:
                raise WikiSQLSourceCompilerError(
                    "SQLite normalized cell disagrees with JSON table rows"
                )


def _sqlite_rowid_results(
    db_members: Mapping[str, bytes],
    requests: Sequence[SQLiteDerivationRequest],
    *,
    connection_verifier: Callable[
        [Mapping[str, sqlite3.Connection]],
        None,
    ]
    | None = None,
    table_verifier: Callable[
        [sqlite3.Connection, str, reality.WikiSQLTable],
        None,
    ]
    | None = None,
) -> dict[str, tuple[int, ...]]:
    """Derive authoritative physical row ordinals in one DB-open session.

    The two exact DB members are materialized only to mode-0600 files inside a
    mode-0700 temporary directory; no archive path is ever extracted.
    """

    if set(db_members) != {"train", "test"}:
        raise WikiSQLSourceCompilerError("SQLite member registry drifted")
    rows = tuple(requests)
    if not rows:
        raise WikiSQLSourceCompilerError("SQLite rowid request is empty")
    if (
        any(not isinstance(row, SQLiteDerivationRequest) for row in rows)
        or len({row.item_commitment_sha256 for row in rows}) != len(rows)
    ):
        raise WikiSQLSourceCompilerError(
            "SQLite rowid requests are malformed or duplicated"
        )
    posix_tmp = Path("/tmp")
    temporary_parent = (
        posix_tmp
        if posix_tmp.is_dir()
        and os.access(posix_tmp, os.W_OK | os.X_OK)
        else None
    )
    verifier = table_verifier
    if verifier is None:
        verifier = lambda connection, table_name, table: (
            _verify_sqlite_table_matches_json(
                connection,
                table_name=table_name,
                table=table,
            )
        )
    with tempfile.TemporaryDirectory(
        prefix="wikisql-uao-sqlite-",
        dir=temporary_parent,
    ) as raw_root:
        root = Path(raw_root)
        root.chmod(0o700)
        connections: dict[str, sqlite3.Connection] = {}
        result: dict[str, tuple[int, ...]] = {}
        verified_tables: dict[tuple[str, str], str] = {}
        try:
            for split in ("train", "test"):
                path = root / f"{split}.db"
                _exclusive_write(path, db_members[split], mode=0o600)
                connection = sqlite3.connect(
                    f"file:{path}?mode=ro&immutable=1",
                    uri=True,
                )
                connection.execute("PRAGMA query_only=ON")
                connections[split] = connection
            if connection_verifier is not None:
                connection_verifier(connections)
            for request in rows:
                if request.split not in connections:
                    raise WikiSQLSourceCompilerError(
                        "SQLite request split is invalid"
                    )
                if _SQLITE_TABLE_ID_RE.fullmatch(request.source_table_id) is None:
                    raise WikiSQLSourceCompilerError(
                        "SQLite table id cannot be quoted safely"
                    )
                table_name = "table_" + request.source_table_id.replace("-", "_")
                condition = request.query.conditions[0]
                column_type = request.table.types[condition.column_index]
                parameter = _sqlite_condition_value(
                    condition.value, column_type
                )
                operator = reality.CONDITION_OPERATORS[
                    condition.operator_index
                ]
                connection = connections[request.split]
                table_key = (request.split, request.source_table_id)
                table_sha256 = reality.canonical_sha256(
                    {
                        "header": list(request.table.header),
                        "rows": [
                            list(row) for row in request.table.rows
                        ],
                        "types": list(request.table.types),
                    }
                )
                previous_table_sha256 = verified_tables.get(table_key)
                if previous_table_sha256 is None:
                    verifier(connection, table_name, request.table)
                    verified_tables[table_key] = table_sha256
                elif previous_table_sha256 != table_sha256:
                    raise WikiSQLSourceCompilerError(
                        "repeated SQLite table request changed JSON content"
                    )
                try:
                    observed = tuple(
                        int(row[0]) - 1
                        for row in connection.execute(
                            f'SELECT rowid FROM "{table_name}" '
                            f'WHERE "col{condition.column_index}" {operator} ? '
                            "ORDER BY rowid",
                            (parameter,),
                        )
                    )
                except sqlite3.Error as exc:
                    raise WikiSQLSourceCompilerError(
                        "SQLite rowid cross-check query failed"
                    ) from exc
                result[request.item_commitment_sha256] = observed
        finally:
            for connection in connections.values():
                connection.close()
    return result


def sqlite_rowid_derive(
    db_members: Mapping[str, bytes],
    requests: Sequence[SQLiteDerivationRequest],
    *,
    connection_verifier: Callable[
        [Mapping[str, sqlite3.Connection]],
        None,
    ]
    | None = None,
    table_verifier: Callable[
        [sqlite3.Connection, str, reality.WikiSQLTable],
        None,
    ]
    | None = None,
) -> Mapping[str, tuple[int, ...]]:
    """Public injectable authority for pre-HMAC gold-row derivation."""

    return _sqlite_rowid_results(
        db_members,
        requests,
        connection_verifier=connection_verifier,
        table_verifier=table_verifier,
    )


def sqlite_rowid_cross_check(
    db_members: Mapping[str, bytes],
    requests: Sequence[SQLiteCrossCheckRequest],
    *,
    connection_verifier: Callable[
        [Mapping[str, sqlite3.Connection]],
        None,
    ]
    | None = None,
    table_verifier: Callable[
        [sqlite3.Connection, str, reality.WikiSQLTable],
        None,
    ]
    | None = None,
) -> None:
    """Independently re-query selected items and assert exact consistency."""

    rows = tuple(requests)
    if not rows or any(not isinstance(row, SQLiteCrossCheckRequest) for row in rows):
        raise WikiSQLSourceCompilerError(
            "selected SQLite consistency requests are malformed"
        )
    derived = _sqlite_rowid_results(
        db_members,
        tuple(
            SQLiteDerivationRequest(
                split=row.split,
                source_table_id=row.source_table_id,
                table=row.table,
                query=row.query,
                item_commitment_sha256=row.item_commitment_sha256,
            )
            for row in rows
        ),
        connection_verifier=connection_verifier,
        table_verifier=table_verifier,
    )
    if any(
        derived[row.item_commitment_sha256] != row.expected_gold_row_ids
        for row in rows
    ):
        raise WikiSQLSourceCompilerError(
            "selected SQLite consistency assert failed"
        )


def _opaque_item_id(
    secret: bytes,
    *,
    block: str,
    item: CompiledItem,
) -> str:
    return reality.hmac_selection_digest(
        secret,
        block=f"{block}:opaque",
        family=item.family,
        item_commitment_sha256=item.item_commitment_sha256,
    ).hex()


def _self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if "self_sha256" in result:
        raise WikiSQLSourceCompilerError("payload already contains self_sha256")
    result["self_sha256"] = reality.canonical_sha256(result)
    return result


def _build_packs(
    *,
    secret: bytes,
    block: str,
    items: Sequence[CompiledItem],
    folds: Mapping[str, int] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    action_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    opaque_seen: set[str] = set()
    for item in items:
        opaque = _opaque_item_id(secret, block=block, item=item)
        if opaque in opaque_seen:
            raise WikiSQLSourceCompilerError("opaque item id collides")
        opaque_seen.add(opaque)
        action = {
            "opaque_item_id": opaque,
            "question": item.question,
            "table_header": list(item.table.header),
            "table_types": list(item.table.types),
            "physical_rows": [list(row) for row in item.table.rows],
        }
        if set(action) != ACTION_VIEW_FIELDS:
            raise WikiSQLSourceCompilerError("action-view schema drifted")
        label: dict[str, object] = {
            "opaque_item_id": opaque,
            "action_view_sha256": reality.canonical_sha256(action),
            "item_commitment_sha256": item.item_commitment_sha256,
            "family": item.family,
            "gold_row_ids": list(item.gold_row_ids),
            "table_row_count": len(item.table.rows),
            "sqlite_rowid_cross_checked": True,
        }
        if folds is not None:
            label["fold_index"] = folds[item.item_commitment_sha256]
        expected_label_fields = set(LABEL_VIEW_FIELDS)
        if folds is not None:
            expected_label_fields.add("fold_index")
        if set(label) != expected_label_fields:
            raise WikiSQLSourceCompilerError("minimal label-view schema drifted")
        action_rows.append(action)
        label_rows.append(label)
    paired = sorted(
        zip(action_rows, label_rows, strict=True),
        key=lambda row: row[0]["opaque_item_id"],
    )
    action_rows = [row[0] for row in paired]
    label_rows = [row[1] for row in paired]
    action_pack = _self_hashed(
        {
            "schema": f"{VERSION}_private_action_view_pack_v1",
            "study_id": STUDY_ID,
            "block": block,
            "item_count": len(action_rows),
            "items": action_rows,
            "contains_labels": False,
        }
    )
    label_pack = _self_hashed(
        {
            "schema": f"{VERSION}_private_label_pack_v1",
            "study_id": STUDY_ID,
            "block": block,
            "item_count": len(label_rows),
            "items": label_rows,
            "release_policy": (
                "A_form_train_only"
                if block == "A_form"
                else "after_all_A_hold_three_arm_actions_are_sealed"
            ),
        }
    )
    return action_pack, label_pack


def _build_controller_provenance_pack(
    *,
    secret: bytes,
    blocks: Sequence[tuple[str, Sequence[CompiledItem]]],
) -> dict[str, object]:
    """Build lineage held by the controller, never by Agent or scorer."""

    rows: list[dict[str, object]] = []
    for block, items in blocks:
        if block not in {"A_form", "A_hold"}:
            raise WikiSQLSourceCompilerError("provenance block is invalid")
        for item in items:
            row: dict[str, object] = {
                "block": block,
                "opaque_item_id": _opaque_item_id(
                    secret,
                    block=block,
                    item=item,
                ),
                "item_commitment_sha256": item.item_commitment_sha256,
                "table_commitment_sha256": item.table_commitment_sha256,
                "split": item.split,
                "source_line_number": item.line_number,
                "source_table_id": item.source_table_id,
                "sql": item.raw_sql,
            }
            if set(row) != PROVENANCE_FIELDS:
                raise WikiSQLSourceCompilerError(
                    "controller provenance schema drifted"
                )
            rows.append(row)
    rows.sort(key=lambda row: (str(row["block"]), str(row["opaque_item_id"])))
    if (
        len({row["opaque_item_id"] for row in rows}) != len(rows)
        or len({row["item_commitment_sha256"] for row in rows}) != len(rows)
    ):
        raise WikiSQLSourceCompilerError(
            "controller provenance identities collide"
        )
    return _self_hashed(
        {
            "schema": f"{VERSION}_controller_only_provenance_pack_v1",
            "study_id": STUDY_ID,
            "access_policy": "controller_only_never_Agent_or_scorer_input",
            "item_count": len(rows),
            "items": rows,
        }
    )


def _pack_commitment(pack: Mapping[str, object]) -> dict[str, object]:
    self_sha256 = pack.get("self_sha256")
    _required_sha256(self_sha256, field="pack self SHA-256")
    return {
        "self_sha256": self_sha256,
        "canonical_payload_sha256": reality.canonical_sha256(pack),
    }


def compile_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    config: CompilerConfig | None = None,
    secret_factory: Callable[[int], bytes] = secrets.token_bytes,
    sqlite_rowid_deriver: SQLiteRowIDDeriver = sqlite_rowid_derive,
    sqlite_cross_checker: SQLiteCrossChecker = sqlite_rowid_cross_check,
) -> CompilationBundle:
    """Compile one verified source archive into isolated private packs."""

    frozen = CompilerConfig.production() if config is None else config
    if not isinstance(frozen, CompilerConfig):
        raise WikiSQLSourceCompilerError("compiler config is invalid")
    if (
        frozen.mode == "production"
        and babel.__version__ != PRODUCTION_BABEL_VERSION
    ):
        raise WikiSQLSourceCompilerError(
            "production compile requires Babel 2.10.3"
        )
    archive = read_authorized_archive(
        archive_path,
        expected_archive_sha256=expected_archive_sha256,
        config=frozen,
    )
    source_revision = _required_sha256(
        expected_archive_sha256, field="source revision"
    )
    tables = {
        split: _load_tables(
            archive.members[TABLE_MEMBERS[split]],
            member_name=TABLE_MEMBERS[split],
        )
        for split in ("train", "test")
    }
    overlap = set(tables["train"]).intersection(tables["test"])
    if overlap:
        raise WikiSQLSourceCompilerError(
            "train/test table registries are not disjoint"
        )
    structural: dict[str, tuple[StructuralItem, ...]] = {}
    dispositions: dict[str, Counter[str]] = {}
    for split in ("train", "test"):
        structural[split], dispositions[split] = _parse_structural_split_items(
            split=split,
            source_revision_sha256=source_revision,
            query_raw=archive.members[JSONL_MEMBERS[split]],
            query_member_name=JSONL_MEMBERS[split],
            tables=tables[split],
        )

    db_members = {
        split: archive.members[DB_MEMBERS[split]]
        for split in ("train", "test")
    }
    derivation_requests = tuple(
        SQLiteDerivationRequest(
            split=row.split,
            source_table_id=row.source_table_id,
            table=row.table,
            query=row.query,
            item_commitment_sha256=row.item_commitment_sha256,
        )
        for split in ("train", "test")
        for row in structural[split]
    )
    if not callable(sqlite_rowid_deriver):
        raise WikiSQLSourceCompilerError("SQLite rowid deriver is not callable")
    derived = sqlite_rowid_deriver(db_members, derivation_requests)
    expected_derivation_keys = {
        row.item_commitment_sha256 for row in derivation_requests
    }
    if (
        not isinstance(derived, Mapping)
        or set(derived) != expected_derivation_keys
    ):
        raise WikiSQLSourceCompilerError(
            "SQLite rowid derivation coverage drifted"
        )
    compiled: dict[str, tuple[CompiledItem, ...]] = {}
    for split in ("train", "test"):
        accepted: list[CompiledItem] = []
        for row in structural[split]:
            value = derived[row.item_commitment_sha256]
            if (
                not isinstance(value, tuple)
                or any(type(index) is not int for index in value)
                or tuple(sorted(set(value))) != value
                or any(not 0 <= index < len(row.table.rows) for index in value)
            ):
                raise WikiSQLSourceCompilerError(
                    "SQLite rowid derivation returned invalid physical ordinals"
                )
            if not reality.MIN_GOLD_ROWS <= len(value) <= reality.MAX_GOLD_ROWS:
                dispositions[split]["sqlite_gold_row_count_outside_1_5"] += 1
                continue
            accepted.append(
                CompiledItem(
                    split=row.split,
                    line_number=row.line_number,
                    item_commitment_sha256=row.item_commitment_sha256,
                    table_commitment_sha256=row.table_commitment_sha256,
                    question=row.question,
                    source_table_id=row.source_table_id,
                    table=row.table,
                    query=row.query,
                    family=row.family,
                    raw_sql=row.raw_sql,
                    gold_row_ids=value,
                )
            )
            dispositions[split]["eligible"] += 1
        compiled[split] = tuple(accepted)

    secret = secret_factory(reality.HMAC_SECRET_BYTES)
    if type(secret) is not bytes or len(secret) != reality.HMAC_SECRET_BYTES:
        raise WikiSQLSourceCompilerError(
            "secret factory did not return one exact 32-byte secret"
        )
    a_form = _select_block(
        secret=secret,
        block="A_form",
        items=compiled["train"],
        quota_per_family=frozen.a_form_quota_per_family,
    )
    a_hold = _select_block(
        secret=secret,
        block="A_hold",
        items=compiled["test"],
        quota_per_family=frozen.a_hold_quota_per_family,
    )
    if {
        row.source_table_id for row in a_form
    }.intersection(row.source_table_id for row in a_hold):
        raise WikiSQLSourceCompilerError(
            "selected A_form/A_hold tables are not disjoint"
        )
    folds = _fold_assignment(
        secret,
        a_form,
        quota_per_family=frozen.a_form_quota_per_family,
    )
    selected = a_form + a_hold
    requests = tuple(
        SQLiteCrossCheckRequest(
            split=row.split,
            source_table_id=row.source_table_id,
            table=row.table,
            query=row.query,
            expected_gold_row_ids=row.gold_row_ids,
            item_commitment_sha256=row.item_commitment_sha256,
        )
        for row in selected
    )
    if not callable(sqlite_cross_checker):
        raise WikiSQLSourceCompilerError("SQLite cross-checker is not callable")
    sqlite_cross_checker(
        db_members,
        requests,
    )

    a_form_action, a_form_label = _build_packs(
        secret=secret,
        block="A_form",
        items=a_form,
        folds=folds,
    )
    a_hold_action, a_hold_label = _build_packs(
        secret=secret,
        block="A_hold",
        items=a_hold,
        folds=None,
    )
    controller_provenance = _build_controller_provenance_pack(
        secret=secret,
        blocks=(("A_form", a_form), ("A_hold", a_hold)),
    )
    family_counts = {
        block: {
            family: sum(row.family == family for row in block_items)
            for family in reality.FAMILY_ORDER
        }
        for block, block_items in (("A_form", a_form), ("A_hold", a_hold))
    }
    fold_counts = {
        str(fold): {
            family: sum(
                row.family == family
                and folds[row.item_commitment_sha256] == fold
                for row in a_form
            )
            for family in reality.FAMILY_ORDER
        }
        for fold in range(FOLD_COUNT)
    }
    safe_receipt = _self_hashed(
        {
            "schema": f"{VERSION}_safe_aggregate_receipt_v1",
            "study_id": STUDY_ID,
            "status": "compiled_source_and_sealed_private_packs",
            "compiler_mode": frozen.mode,
            "release_version": RELEASE_VERSION,
            "source_archive_sha256": source_revision,
            "source_archive_git_blob_sha1": (
                archive.archive_git_blob_sha1
            ),
            "source_member_sha256": [
                {"member": name, "sha256": digest}
                for name, digest in archive.member_sha256
            ],
            "authorized_member_open_count": len(REQUIRED_MEMBERS),
            "regular_member_count": archive.regular_member_count,
            "directory_member_count": archive.directory_member_count,
            "ignored_regular_member_count": archive.ignored_regular_member_count,
            "selection_secret_sha256": reality.hmac_secret_commitment(secret),
            "family_counts": family_counts,
            "A_form_fold_family_counts": fold_counts,
            "selected_item_count": len(selected),
            "selected_table_count": len(
                {row.source_table_id for row in selected}
            ),
            "train_test_table_overlap_count": 0,
            "sqlite_rowid_derivation_candidate_count": len(
                derivation_requests
            ),
            "sqlite_rowid_eligible_count": sum(
                len(compiled[split]) for split in ("train", "test")
            ),
            "selected_sqlite_consistency_assert_count": len(requests),
            "sqlite_runtime_version": sqlite3.sqlite_version,
            "babel_runtime_version": babel.__version__,
            "babel_required_production_version": PRODUCTION_BABEL_VERSION,
            "babel_locale": BABEL_LOCALE,
            "eligibility_contract": {
                "condition_count": 1,
                "condition_operator_indices": [0, 1, 2],
                "table_physical_row_count_minimum": reality.MIN_TABLE_ROWS,
                "table_physical_row_count_maximum": reality.MAX_TABLE_ROWS,
                "column_count_minimum": 1,
                "column_count_maximum": MAX_COLUMNS,
                "question_character_count_maximum": MAX_QUESTION_CHARACTERS,
                "header_or_cell_character_count_maximum": (
                    MAX_HEADER_OR_CELL_CHARACTERS
                ),
                "canonical_serialized_row_character_count_maximum": (
                    reality.MAX_SERIALIZED_ROW_CHARACTERS
                ),
                "canonical_serialized_rows_must_round_trip": True,
                "canonical_serialized_rows_must_be_unique": True,
                "sqlite_schema_rowid_order_and_normalized_cells_must_match_json_before_gold_derivation": True,
                "sqlite_gold_row_count_minimum": reality.MIN_GOLD_ROWS,
                "sqlite_gold_row_count_maximum": reality.MAX_GOLD_ROWS,
                "sqlite_gold_authoritative_before_HMAC": True,
            },
            "source_dispositions": {
                split: dict(sorted(dispositions[split].items()))
                for split in ("train", "test")
            },
            "pack_commitments": {
                "A_form_action_view": _pack_commitment(a_form_action),
                "A_form_label": _pack_commitment(a_form_label),
                "A_hold_action_view": _pack_commitment(a_hold_action),
                "A_hold_label": _pack_commitment(a_hold_label),
                "controller_only_provenance": _pack_commitment(
                    controller_provenance
                ),
            },
            "actual_item_question_table_SQL_family_gold_disclosed": False,
        }
    )
    return CompilationBundle(
        selection_secret=secret,
        a_form_action_pack=a_form_action,
        a_form_label_pack=a_form_label,
        a_hold_action_pack=a_hold_action,
        a_hold_label_pack=a_hold_label,
        controller_provenance_pack=controller_provenance,
        safe_receipt=safe_receipt,
    )


def _exclusive_write(path: Path, raw: bytes, *, mode: int = 0o600) -> str:
    if path.exists() or path.is_symlink():
        raise WikiSQLSourceCompilerError("one-shot output already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
        try:
            os.fchmod(descriptor, mode)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("short output write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise WikiSQLSourceCompilerError("one-shot output write failed") from exc
    metadata = path.stat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != mode
        or path.read_bytes() != raw
    ):
        raise WikiSQLSourceCompilerError("one-shot output verification failed")
    return _sha256_bytes(raw)


def _canonical_json_file(value: Mapping[str, object]) -> bytes:
    return reality.canonical_json_bytes(value) + b"\n"


def write_compilation(
    output_root: Path,
    bundle: CompilationBundle,
) -> Mapping[str, str]:
    """Write private packs first and the safe terminal receipt last."""

    if not isinstance(bundle, CompilationBundle):
        raise WikiSQLSourceCompilerError("compilation bundle is invalid")
    try:
        output_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(output_root, 0o700)
        private = output_root / "private"
        safe = output_root / "safe"
        private.mkdir(mode=0o700)
        safe.mkdir(mode=0o700)
    except OSError as exc:
        raise WikiSQLSourceCompilerError(
            "one-shot output root cannot be created"
        ) from exc
    paths_and_raw = (
        (
            private / "selection_secret.bin",
            bundle.selection_secret,
        ),
        (
            private / "A_form.action_views.json",
            _canonical_json_file(bundle.a_form_action_pack),
        ),
        (
            private / "A_form.labels.json",
            _canonical_json_file(bundle.a_form_label_pack),
        ),
        (
            private / "A_hold.action_views.json",
            _canonical_json_file(bundle.a_hold_action_pack),
        ),
        (
            private / "A_hold.labels.json",
            _canonical_json_file(bundle.a_hold_label_pack),
        ),
        (
            private / "controller_only.provenance.json",
            _canonical_json_file(bundle.controller_provenance_pack),
        ),
        (
            safe / "source_compiler_receipt.json",
            _canonical_json_file(bundle.safe_receipt),
        ),
    )
    result: dict[str, str] = {}
    for path, raw in paths_and_raw:
        result[path.relative_to(output_root).as_posix()] = _exclusive_write(
            path, raw, mode=0o600
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Production CLI: fixed quotas only; no synthetic or quota flags."""

    arguments = _parser().parse_args(argv)
    bundle = compile_archive(
        arguments.archive,
        expected_archive_sha256=arguments.archive_sha256,
        config=CompilerConfig.production(),
    )
    write_compilation(arguments.output_root, bundle)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTION_FORBIDDEN_FIELDS",
    "ACTION_VIEW_FIELDS",
    "BABEL_LOCALE",
    "PRODUCTION_BABEL_VERSION",
    "PROVENANCE_FIELDS",
    "LABEL_VIEW_FIELDS",
    "ArchiveRead",
    "CompilationBundle",
    "CompilerConfig",
    "DB_MEMBERS",
    "FOLD_COUNT",
    "JSONL_MEMBERS",
    "PRODUCTION_ARCHIVE_SIZE_BYTES",
    "README_DENYLIST",
    "RELEASE_VERSION",
    "REQUIRED_MEMBERS",
    "SQLiteCrossCheckRequest",
    "SQLiteDerivationRequest",
    "STUDY_ID",
    "TABLE_MEMBERS",
    "VERSION",
    "WikiSQLSourceCompilerError",
    "WikiSQLSourceIneligibleError",
    "compile_archive",
    "main",
    "read_authorized_archive",
    "sqlite_rowid_cross_check",
    "sqlite_rowid_derive",
    "write_compilation",
]

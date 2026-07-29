"""Public-source-qualified WikiSQL compiler for the amended v5 lineage.

The original v5 attempt stopped before JSON parsing because the official
archive has no ``data/version.txt``.  This module binds the release through
the already frozen archive/Git-blob identity, accepts only the three observed
official table metadata envelopes, validates SQLite storage classes without
assuming that every ``REAL`` cell is stored as a number, and replaces the
capacity-unsafe cross-family table winner with a deterministic constrained
matching.

``qualify_archive`` performs the complete public-source adapter pass without
creating a secret, HMAC selection, cohort, action, or score.  ``compile_archive``
reuses exactly that pass and only then creates the single formal secret.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import hmac
import math
from pathlib import Path
import secrets
import sqlite3
import tarfile
from typing import Callable, Mapping, Sequence

import babel
import pytz

from assumption_agent.benchmarks import (
    wikisql_uao_reality_v1 as reality,
)
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as base,
)


# Pack schemas remain byte-compatible with the already frozen action runtime.
VERSION = base.VERSION
STUDY_ID = base.STUDY_ID
RELEASE_VERSION = base.RELEASE_VERSION
PRODUCTION_ARCHIVE_SIZE_BYTES = base.PRODUCTION_ARCHIVE_SIZE_BYTES
PRODUCTION_ARCHIVE_GIT_BLOB_SHA1 = base.PRODUCTION_ARCHIVE_GIT_BLOB_SHA1
PRODUCTION_BABEL_VERSION = base.PRODUCTION_BABEL_VERSION
PRODUCTION_PYTZ_VERSION = "2022.1"
BABEL_LOCALE = base.BABEL_LOCALE
FOLD_COUNT = base.FOLD_COUNT
MAX_COLUMNS = base.MAX_COLUMNS
MAX_QUESTION_CHARACTERS = base.MAX_QUESTION_CHARACTERS
MAX_HEADER_OR_CELL_CHARACTERS = base.MAX_HEADER_OR_CELL_CHARACTERS
JSONL_MEMBERS = base.JSONL_MEMBERS
TABLE_MEMBERS = base.TABLE_MEMBERS
DB_MEMBERS = base.DB_MEMBERS
REQUIRED_MEMBERS = (
    "data/train.jsonl",
    "data/train.tables.jsonl",
    "data/train.db",
    "data/test.jsonl",
    "data/test.tables.jsonl",
    "data/test.db",
)
DEV_MEMBERS = (
    "data/dev.jsonl",
    "data/dev.tables.jsonl",
    "data/dev.db",
)
EXACT_REGULAR_MEMBERS = frozenset((*REQUIRED_MEMBERS, *DEV_MEMBERS))
EXACT_DIRECTORY_MEMBERS = frozenset({"data"})
TABLE_ENVELOPES = frozenset(
    {
        frozenset({"id", "header", "types", "rows", "name"}),
        frozenset(
            {
                "id",
                "header",
                "types",
                "rows",
                "name",
                "caption",
                "page_title",
                "section_title",
            }
        ),
        frozenset(
            {
                "id",
                "header",
                "types",
                "rows",
                "page_id",
                "caption",
                "page_title",
                "section_title",
            }
        ),
    }
)
EXPECTED_TABLE_COUNTS = {"train": 18_585, "test": 5_230}
EXPECTED_QUERY_COUNTS = {"train": 56_355, "test": 15_878}
QUALIFICATION_SCHEMA = "wikisql_uao_v5_repair_source_qualification_v1"
SELECTOR_NAME = "hmac_priority_capacity_preserving_bipartite_matching_v1"

CompilerConfig = base.CompilerConfig
CompilationBundle = base.CompilationBundle
CompiledItem = base.CompiledItem
SQLiteCrossCheckRequest = base.SQLiteCrossCheckRequest
SQLiteDerivationRequest = base.SQLiteDerivationRequest
WikiSQLSourceCompilerError = base.WikiSQLSourceCompilerError
WikiSQLSourceIneligibleError = base.WikiSQLSourceIneligibleError
write_compilation = base.write_compilation


@dataclass(frozen=True, slots=True)
class RepairArchive:
    members: Mapping[str, bytes]
    member_sha256: tuple[tuple[str, str], ...]
    archive_git_blob_sha1: str
    regular_member_count: int
    directory_member_count: int
    ignored_regular_member_count: int


@dataclass(frozen=True, slots=True)
class TableRegistry:
    eligible: Mapping[str, reality.WikiSQLTable]
    database_validation: Mapping[str, reality.WikiSQLTable]
    schema_variant_counts: Mapping[str, int]
    unreferenced_blank_header_table_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class QualifiedSource:
    archive: RepairArchive
    compiled: Mapping[str, tuple[CompiledItem, ...]]
    dispositions: Mapping[str, Counter[str]]
    sqlite_derivation_count: int
    safe_receipt: Mapping[str, object]


def _read_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    config: CompilerConfig,
) -> RepairArchive:
    expected = base._required_sha256(
        expected_archive_sha256,
        field="expected archive SHA-256",
    )
    observed, git_blob_sha1, size = base._hash_file(archive_path)
    if observed != expected:
        raise WikiSQLSourceCompilerError("source archive SHA-256 drifted")
    if (
        config.expected_archive_size_bytes is not None
        and config.expected_archive_size_bytes != size
    ) or (
        config.expected_archive_git_blob_sha1 is not None
        and config.expected_archive_git_blob_sha1 != git_blob_sha1
    ):
        raise WikiSQLSourceCompilerError(
            "source archive official identity drifted"
        )

    authorized: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    regular: set[str] = set()
    directories: set[str] = set()
    try:
        with tarfile.open(
            archive_path,
            mode="r:bz2",
            errorlevel=2,
        ) as bundle:
            for member in bundle:
                base._safe_tar_parts(member.name)
                if member.isdir():
                    directories.add(member.name.rstrip("/"))
                    continue
                if not member.isfile():
                    raise WikiSQLSourceCompilerError(
                        "archive contains a non-regular non-directory member"
                    )
                if member.name in regular:
                    raise WikiSQLSourceCompilerError(
                        "archive duplicates a regular member"
                    )
                regular.add(member.name)
                if member.name in REQUIRED_MEMBERS:
                    raw = base._read_member(bundle, member)
                    authorized[member.name] = raw
                    hashes[member.name] = hashlib.sha256(raw).hexdigest()
    except WikiSQLSourceCompilerError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise WikiSQLSourceCompilerError(
            "source archive scan failed"
        ) from exc
    if (
        regular != set(EXACT_REGULAR_MEMBERS)
        or directories != set(EXACT_DIRECTORY_MEMBERS)
        or set(authorized) != set(REQUIRED_MEMBERS)
    ):
        raise WikiSQLSourceCompilerError(
            "official archive topology drifted"
        )
    for member_name in DB_MEMBERS.values():
        if not authorized[member_name].startswith(b"SQLite format 3\x00"):
            raise WikiSQLSourceCompilerError(
                "authorized DB member is not SQLite3"
            )
    return RepairArchive(
        members=authorized,
        member_sha256=tuple(
            (name, hashes[name]) for name in REQUIRED_MEMBERS
        ),
        archive_git_blob_sha1=git_blob_sha1,
        regular_member_count=len(regular),
        directory_member_count=len(directories),
        ignored_regular_member_count=len(regular - set(REQUIRED_MEMBERS)),
    )


def _metadata_envelope(value: Mapping[str, object]) -> None:
    keys = frozenset(value)
    if keys not in TABLE_ENVELOPES:
        raise WikiSQLSourceCompilerError(
            "official table metadata envelope drifted"
        )
    for field in ("name", "caption", "page_title", "section_title"):
        if field in value and (
            not isinstance(value[field], str)
            or "\x00" in value[field]
        ):
            raise WikiSQLSourceCompilerError(
                "official table text metadata drifted"
            )
    if "page_id" in value and (
        type(value["page_id"]) is not int or value["page_id"] < 0
    ):
        raise WikiSQLSourceCompilerError(
            "official table page_id metadata drifted"
        )


def _load_table_registry(
    raw: bytes,
    *,
    member_name: str,
) -> TableRegistry:
    eligible: dict[str, reality.WikiSQLTable] = {}
    database_validation: dict[str, reality.WikiSQLTable] = {}
    variants: Counter[str] = Counter()
    blank_header_ids: set[str] = set()
    rows = base._jsonl(raw, member_name=member_name)
    for line_number, value in enumerate(rows, 1):
        _metadata_envelope(value)
        variants["|".join(sorted(value))] += 1
        projected = {
            field: value[field]
            for field in reality.OFFICIAL_TABLE_FIELDS
        }
        raw_header = projected["header"]
        if (
            isinstance(raw_header, (str, bytes, bytearray))
            or not isinstance(raw_header, Sequence)
        ):
            raise WikiSQLSourceCompilerError(
                f"{member_name}:{line_number} table header drifted"
            )
        normalized_header = list(raw_header)
        has_blank = False
        for index, header in enumerate(normalized_header):
            if isinstance(header, str) and not header.strip():
                normalized_header[index] = f"__blank_column_{index}__"
                has_blank = True
        validation_projection = {
            **projected,
            "header": normalized_header,
        }
        try:
            table = reality.table_from_documented_schema(
                validation_projection
            )
        except reality.WikiSQLUAORealityError as exc:
            raise WikiSQLSourceCompilerError(
                f"{member_name}:{line_number} table core schema drifted"
            ) from exc
        if table.table_id in database_validation:
            raise WikiSQLSourceCompilerError(
                "table registry repeats an id"
            )
        database_validation[table.table_id] = table
        if has_blank:
            blank_header_ids.add(table.table_id)
        else:
            eligible[table.table_id] = table
    return TableRegistry(
        eligible=eligible,
        database_validation=database_validation,
        schema_variant_counts=dict(sorted(variants.items())),
        unreferenced_blank_header_table_ids=frozenset(blank_header_ids),
    )


def _query_table_ids(
    raw: bytes,
    *,
    member_name: str,
) -> tuple[str, ...]:
    result: list[str] = []
    rows = base._jsonl(raw, member_name=member_name)
    for line_number, value in enumerate(rows, 1):
        _, table_id, _ = base._outer_item(
            value,
            member_name=member_name,
            line_number=line_number,
        )
        result.append(table_id)
    return tuple(result)


def _verify_official_storage(
    connection: sqlite3.Connection,
    table_name: str,
    table: reality.WikiSQLTable,
) -> None:
    try:
        schema = tuple(
            connection.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
        )
        quoted_columns = ", ".join(
            f'"col{index}"' for index in range(len(table.header))
        )
        rows = tuple(
            connection.execute(
                f'SELECT rowid, {quoted_columns} FROM "{table_name}" '
                "ORDER BY rowid"
            ).fetchall()
        )
    except sqlite3.Error as exc:
        raise WikiSQLSourceCompilerError(
            "SQLite/JSON table binding query failed"
        ) from exc
    if (
        tuple(str(row[1]) for row in schema)
        != tuple(f"col{index}" for index in range(len(table.header)))
        or tuple(str(row[2]).casefold() for row in schema)
        != tuple(table.types)
        or len(rows) != len(table.rows)
        or tuple(row[0] for row in rows)
        != tuple(range(1, len(table.rows) + 1))
    ):
        raise WikiSQLSourceCompilerError(
            "SQLite schema or rowid order disagrees with JSON table rows"
        )
    for database_row, json_row in zip(rows, table.rows, strict=True):
        for database_value, json_value, column_type in zip(
            database_row[1:],
            json_row,
            table.types,
            strict=True,
        ):
            if column_type == "text":
                matches = (
                    isinstance(database_value, str)
                    and database_value == str(json_value).lower()
                )
            elif isinstance(database_value, str):
                matches = (
                    isinstance(json_value, str)
                    and database_value == json_value.lower()
                )
            else:
                try:
                    expected = base._sqlite_condition_value(
                        json_value,
                        column_type,
                    )
                except WikiSQLSourceCompilerError:
                    matches = False
                else:
                    matches = (
                        not isinstance(database_value, bool)
                        and isinstance(database_value, (int, float))
                        and math.isfinite(float(database_value))
                        and float(database_value) == float(expected)
                    )
            if not matches:
                raise WikiSQLSourceCompilerError(
                    "SQLite normalized cell disagrees with JSON table rows"
                )


def _database_verifier(
    registries: Mapping[str, TableRegistry],
) -> Callable[[Mapping[str, sqlite3.Connection]], None]:
    def verify(connections: Mapping[str, sqlite3.Connection]) -> None:
        if set(connections) != {"train", "test"}:
            raise WikiSQLSourceCompilerError(
                "SQLite connection registry drifted"
            )
        for split in ("train", "test"):
            connection = connections[split]
            try:
                quick = tuple(
                    connection.execute("PRAGMA quick_check").fetchall()
                )
                observed = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table'"
                    )
                }
            except sqlite3.Error as exc:
                raise WikiSQLSourceCompilerError(
                    "SQLite registry qualification failed"
                ) from exc
            registry = registries[split].database_validation
            expected = {
                "table_" + table_id.replace("-", "_")
                for table_id in registry
            }
            if quick != (("ok",),) or observed != expected:
                raise WikiSQLSourceCompilerError(
                    "SQLite registry or quick_check drifted"
                )
            for table_id, table in registry.items():
                _verify_official_storage(
                    connection,
                    "table_" + table_id.replace("-", "_"),
                    table,
                )

    return verify


def _hall_capacity(
    rows: Sequence[CompiledItem],
    *,
    quota_per_family: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    families_by_table: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        families_by_table[row.table_commitment_sha256].add(row.family)
    family_support = {
        family: sum(
            family in families
            for families in families_by_table.values()
        )
        for family in reality.FAMILY_ORDER
    }
    single_family_floor = {
        family: sum(
            families == {family}
            for families in families_by_table.values()
        )
        for family in reality.FAMILY_ORDER
    }
    hall: dict[str, int] = {}
    family_order = tuple(reality.FAMILY_ORDER)
    for mask in range(1, 1 << len(family_order)):
        subset = tuple(
            family_order[index]
            for index in range(len(family_order))
            if mask & (1 << index)
        )
        neighbor_count = sum(
            bool(set(subset) & families)
            for families in families_by_table.values()
        )
        required = quota_per_family * len(subset)
        hall["+".join(subset)] = neighbor_count
        if neighbor_count < required:
            raise WikiSQLSourceIneligibleError(
                "capacity-preserving selector Hall condition failed"
            )
    return family_support, single_family_floor, hall


def _qualify(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    config: CompilerConfig,
) -> QualifiedSource:
    if config.mode == "production":
        if babel.__version__ != PRODUCTION_BABEL_VERSION:
            raise WikiSQLSourceCompilerError(
                "production compile requires Babel 2.10.3"
            )
        if pytz.__version__ != PRODUCTION_PYTZ_VERSION:
            raise WikiSQLSourceCompilerError(
                "production compile requires pytz 2022.1"
            )
    archive = _read_archive(
        archive_path,
        expected_archive_sha256=expected_archive_sha256,
        config=config,
    )
    source_revision = base._required_sha256(
        expected_archive_sha256,
        field="source revision",
    )
    registries = {
        split: _load_table_registry(
            archive.members[TABLE_MEMBERS[split]],
            member_name=TABLE_MEMBERS[split],
        )
        for split in ("train", "test")
    }
    if config.mode == "production" and any(
        len(registries[split].database_validation)
        != EXPECTED_TABLE_COUNTS[split]
        for split in ("train", "test")
    ):
        raise WikiSQLSourceCompilerError(
            "official table registry count drifted"
        )
    if set(registries["train"].database_validation).intersection(
        registries["test"].database_validation
    ):
        raise WikiSQLSourceCompilerError(
            "train/test table registries are not disjoint"
        )
    query_ids = {
        split: _query_table_ids(
            archive.members[JSONL_MEMBERS[split]],
            member_name=JSONL_MEMBERS[split],
        )
        for split in ("train", "test")
    }
    if config.mode == "production" and any(
        len(query_ids[split]) != EXPECTED_QUERY_COUNTS[split]
        for split in ("train", "test")
    ):
        raise WikiSQLSourceCompilerError(
            "official query registry count drifted"
        )
    if any(
        set(query_ids[split])
        & registries[split].unreferenced_blank_header_table_ids
        for split in ("train", "test")
    ):
        raise WikiSQLSourceCompilerError(
            "blank-header table is query referenced"
        )
    if any(
        not set(query_ids[split]).issubset(
            registries[split].database_validation
        )
        for split in ("train", "test")
    ):
        raise WikiSQLSourceCompilerError(
            "query references an absent table"
        )

    structural: dict[str, tuple[base.StructuralItem, ...]] = {}
    dispositions: dict[str, Counter[str]] = {}
    for split in ("train", "test"):
        structural[split], dispositions[split] = (
            base._parse_structural_split_items(
                split=split,
                source_revision_sha256=source_revision,
                query_raw=archive.members[JSONL_MEMBERS[split]],
                query_member_name=JSONL_MEMBERS[split],
                tables=registries[split].eligible,
            )
        )
    requests = tuple(
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
    database_members = {
        split: archive.members[DB_MEMBERS[split]]
        for split in ("train", "test")
    }
    derived = base.sqlite_rowid_derive(
        database_members,
        requests,
        connection_verifier=_database_verifier(registries),
        table_verifier=lambda _connection, _name, _table: None,
    )
    compiled: dict[str, tuple[CompiledItem, ...]] = {}
    family_support: dict[str, dict[str, int]] = {}
    single_family_floor: dict[str, dict[str, int]] = {}
    hall_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        accepted: list[CompiledItem] = []
        for row in structural[split]:
            gold = derived[row.item_commitment_sha256]
            if (
                tuple(sorted(set(gold))) != gold
                or any(not 0 <= index < len(row.table.rows) for index in gold)
            ):
                raise WikiSQLSourceCompilerError(
                    "SQLite rowid derivation returned invalid physical ordinals"
                )
            if not reality.MIN_GOLD_ROWS <= len(gold) <= reality.MAX_GOLD_ROWS:
                dispositions[split][
                    "sqlite_gold_row_count_outside_1_5"
                ] += 1
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
                    gold_row_ids=gold,
                )
            )
            dispositions[split]["eligible"] += 1
        compiled[split] = tuple(accepted)
        quota = config.quota(
            "A_form" if split == "train" else "A_hold"
        )
        support, floor, hall = _hall_capacity(
            compiled[split],
            quota_per_family=quota,
        )
        family_support[split] = support
        single_family_floor[split] = floor
        hall_counts[split] = hall
    exhaustive_checks = tuple(
        SQLiteCrossCheckRequest(
            split=row.split,
            source_table_id=row.source_table_id,
            table=row.table,
            query=row.query,
            expected_gold_row_ids=row.gold_row_ids,
            item_commitment_sha256=row.item_commitment_sha256,
        )
        for split in ("train", "test")
        for row in compiled[split]
    )
    base.sqlite_rowid_cross_check(
        database_members,
        exhaustive_checks,
        table_verifier=_verify_official_storage,
    )
    safe_receipt = base._self_hashed(
        {
            "schema": QUALIFICATION_SCHEMA,
            "study_id": STUDY_ID,
            "status": "passed_full_public_source_adapter",
            "qualification_runtime_mode": config.mode,
            "release_version_bound_by_archive_identity": RELEASE_VERSION,
            "source_archive_sha256": source_revision,
            "source_archive_git_blob_sha1": archive.archive_git_blob_sha1,
            "source_member_sha256": [
                {"member": name, "sha256": digest}
                for name, digest in archive.member_sha256
            ],
            "archive_regular_member_count": archive.regular_member_count,
            "archive_directory_member_count": archive.directory_member_count,
            "archive_ignored_dev_member_count": (
                archive.ignored_regular_member_count
            ),
            "authorized_payload_member_open_count": len(REQUIRED_MEMBERS),
            "table_registry_count": {
                split: len(registries[split].database_validation)
                for split in ("train", "test")
            },
            "query_registry_count": {
                split: len(query_ids[split])
                for split in ("train", "test")
            },
            "unreferenced_blank_header_table_count": {
                split: len(
                    registries[
                        split
                    ].unreferenced_blank_header_table_ids
                )
                for split in ("train", "test")
            },
            "table_schema_variant_counts": {
                split: dict(
                    registries[split].schema_variant_counts
                )
                for split in ("train", "test")
            },
            "structural_eligible_count": {
                split: len(structural[split])
                for split in ("train", "test")
            },
            "sqlite_eligible_count": {
                split: len(compiled[split])
                for split in ("train", "test")
            },
            "family_unique_table_support": family_support,
            "legacy_selector_single_family_floor": single_family_floor,
            "capacity_preserving_selector_Hall_neighbor_counts": hall_counts,
            "quota_per_family": {
                "train": config.a_form_quota_per_family,
                "test": config.a_hold_quota_per_family,
            },
            "sqlite_derivation_count": len(requests),
            "sqlite_exhaustive_crosscheck_count": len(exhaustive_checks),
            "source_dispositions": {
                split: dict(sorted(dispositions[split].items()))
                for split in ("train", "test")
            },
            "babel_runtime_version": babel.__version__,
            "pytz_runtime_version": pytz.__version__,
            "selector": SELECTOR_NAME,
            "secret_generation_count": 0,
            "HMAC_selection_count": 0,
            "cohort_selection_count": 0,
            "action_count": 0,
            "scorer_count": 0,
            "score_count": 0,
            "API_or_online_evaluation_count": 0,
            "per_item_question_SQL_family_or_gold_disclosed": False,
        }
    )
    return QualifiedSource(
        archive=archive,
        compiled=compiled,
        dispositions=dispositions,
        sqlite_derivation_count=len(requests),
        safe_receipt=safe_receipt,
    )


def qualify_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    config: CompilerConfig | None = None,
) -> Mapping[str, object]:
    """Return an aggregate-only, zero-secret source qualification receipt."""

    frozen = CompilerConfig.production() if config is None else config
    if not isinstance(frozen, CompilerConfig):
        raise WikiSQLSourceCompilerError("compiler config is invalid")
    return _qualify(
        archive_path,
        expected_archive_sha256=expected_archive_sha256,
        config=frozen,
    ).safe_receipt


def _slot_priority(
    secret: bytes,
    *,
    block: str,
    family: str,
    slot: int,
    item: CompiledItem,
) -> bytes:
    return hmac.new(
        secret,
        (
            f"{STUDY_ID}\0{SELECTOR_NAME}\0{block}\0{family}\0{slot}\0"
            f"{item.table_commitment_sha256}\0"
            f"{item.item_commitment_sha256}"
        ).encode("ascii"),
        hashlib.sha256,
    ).digest()


def _select_block(
    *,
    secret: bytes,
    block: str,
    items: Sequence[CompiledItem],
    quota_per_family: int,
) -> tuple[CompiledItem, ...]:
    by_table_family: dict[tuple[str, str], list[CompiledItem]] = defaultdict(
        list
    )
    by_commitment = {
        row.item_commitment_sha256: row for row in items
    }
    if len(by_commitment) != len(items):
        raise WikiSQLSourceCompilerError(
            "selection input repeats an item"
        )
    for row in items:
        by_table_family[
            (row.table_commitment_sha256, row.family)
        ].append(row)
    winners: dict[tuple[str, str], CompiledItem] = {}
    for key, rows in by_table_family.items():
        ordered = reality.hmac_order(
            secret,
            block=f"{block}:table_family_item",
            candidates=tuple(row.candidate() for row in rows),
        )
        winners[key] = by_commitment[
            ordered[0].item_commitment_sha256
        ]
    slots = tuple(
        (family, slot)
        for family in reality.FAMILY_ORDER
        for slot in range(quota_per_family)
    )
    neighbors: dict[tuple[str, int], tuple[str, ...]] = {}
    for family, slot in slots:
        rows = tuple(
            row
            for (table_id, candidate_family), row in winners.items()
            if candidate_family == family
        )
        neighbors[(family, slot)] = tuple(
            row.table_commitment_sha256
            for row in sorted(
                rows,
                key=lambda row: (
                    _slot_priority(
                        secret,
                        block=block,
                        family=family,
                        slot=slot,
                        item=row,
                    ),
                    row.table_commitment_sha256,
                ),
            )
        )
    table_to_slot: dict[str, tuple[str, int]] = {}
    slot_to_table: dict[tuple[str, int], str] = {}

    def augment(
        slot_key: tuple[str, int],
        visited_tables: set[str],
    ) -> bool:
        for table_id in neighbors[slot_key]:
            if table_id in visited_tables:
                continue
            visited_tables.add(table_id)
            previous = table_to_slot.get(table_id)
            if previous is None or augment(previous, visited_tables):
                table_to_slot[table_id] = slot_key
                slot_to_table[slot_key] = table_id
                return True
        return False

    for slot_key in slots:
        if not augment(slot_key, set()):
            raise WikiSQLSourceIneligibleError(
                "capacity-preserving selector could not fill a quota"
            )
    selected = tuple(
        winners[(slot_to_table[(family, slot)], family)]
        for family, slot in slots
    )
    if (
        len(selected) != quota_per_family * len(reality.FAMILY_ORDER)
        or len({row.table_commitment_sha256 for row in selected})
        != len(selected)
        or any(
            sum(row.family == family for row in selected)
            != quota_per_family
            for family in reality.FAMILY_ORDER
        )
    ):
        raise WikiSQLSourceCompilerError(
            "capacity-preserving selection invariant drifted"
        )
    return selected


def compile_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    config: CompilerConfig | None = None,
    secret_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> CompilationBundle:
    """Qualify the exact source, then form the sole amended-v5 cohort."""

    frozen = CompilerConfig.production() if config is None else config
    if not isinstance(frozen, CompilerConfig):
        raise WikiSQLSourceCompilerError("compiler config is invalid")
    qualified = _qualify(
        archive_path,
        expected_archive_sha256=expected_archive_sha256,
        config=frozen,
    )
    secret = secret_factory(reality.HMAC_SECRET_BYTES)
    if type(secret) is not bytes or len(secret) != reality.HMAC_SECRET_BYTES:
        raise WikiSQLSourceCompilerError(
            "secret factory did not return one exact 32-byte secret"
        )
    a_form = _select_block(
        secret=secret,
        block="A_form",
        items=qualified.compiled["train"],
        quota_per_family=frozen.a_form_quota_per_family,
    )
    a_hold = _select_block(
        secret=secret,
        block="A_hold",
        items=qualified.compiled["test"],
        quota_per_family=frozen.a_hold_quota_per_family,
    )
    if {
        row.source_table_id for row in a_form
    }.intersection(row.source_table_id for row in a_hold):
        raise WikiSQLSourceCompilerError(
            "selected A_form/A_hold tables are not disjoint"
        )
    selected = a_form + a_hold
    base.sqlite_rowid_cross_check(
        {
            split: qualified.archive.members[DB_MEMBERS[split]]
            for split in ("train", "test")
        },
        tuple(
            SQLiteCrossCheckRequest(
                split=row.split,
                source_table_id=row.source_table_id,
                table=row.table,
                query=row.query,
                expected_gold_row_ids=row.gold_row_ids,
                item_commitment_sha256=row.item_commitment_sha256,
            )
            for row in selected
        ),
        table_verifier=_verify_official_storage,
    )
    folds = base._fold_assignment(
        secret,
        a_form,
        quota_per_family=frozen.a_form_quota_per_family,
    )
    a_form_action, a_form_label = base._build_packs(
        secret=secret,
        block="A_form",
        items=a_form,
        folds=folds,
    )
    a_hold_action, a_hold_label = base._build_packs(
        secret=secret,
        block="A_hold",
        items=a_hold,
        folds=None,
    )
    controller_provenance = base._build_controller_provenance_pack(
        secret=secret,
        blocks=(("A_form", a_form), ("A_hold", a_hold)),
    )
    family_counts = {
        block: {
            family: sum(row.family == family for row in rows)
            for family in reality.FAMILY_ORDER
        }
        for block, rows in (("A_form", a_form), ("A_hold", a_hold))
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
    safe_receipt = base._self_hashed(
        {
            "schema": f"{VERSION}_safe_aggregate_receipt_v1",
            "study_id": STUDY_ID,
            "status": "compiled_source_and_sealed_private_packs",
            "compiler_mode": frozen.mode,
            "release_version": RELEASE_VERSION,
            "source_archive_sha256": expected_archive_sha256,
            "source_archive_git_blob_sha1": (
                qualified.archive.archive_git_blob_sha1
            ),
            "source_member_sha256": [
                {"member": name, "sha256": digest}
                for name, digest in qualified.archive.member_sha256
            ],
            "authorized_member_open_count": len(REQUIRED_MEMBERS),
            "regular_member_count": qualified.archive.regular_member_count,
            "directory_member_count": (
                qualified.archive.directory_member_count
            ),
            "ignored_regular_member_count": (
                qualified.archive.ignored_regular_member_count
            ),
            "selection_secret_sha256": (
                reality.hmac_secret_commitment(secret)
            ),
            "family_counts": family_counts,
            "A_form_fold_family_counts": fold_counts,
            "selected_item_count": len(a_form) + len(a_hold),
            "selected_table_count": len(
                {
                    row.source_table_id
                    for row in selected
                }
            ),
            "train_test_table_overlap_count": 0,
            "sqlite_rowid_derivation_candidate_count": (
                qualified.sqlite_derivation_count
            ),
            "sqlite_rowid_eligible_count": sum(
                len(qualified.compiled[split])
                for split in ("train", "test")
            ),
            "selected_sqlite_consistency_assert_count": (
                len(a_form) + len(a_hold)
            ),
            "sqlite_runtime_version": sqlite3.sqlite_version,
            "babel_runtime_version": babel.__version__,
            "babel_required_production_version": (
                PRODUCTION_BABEL_VERSION
            ),
            "pytz_runtime_version": pytz.__version__,
            "pytz_required_production_version": (
                PRODUCTION_PYTZ_VERSION
            ),
            "babel_locale": BABEL_LOCALE,
            "eligibility_contract": {
                "condition_count": 1,
                "condition_operator_indices": [0, 1, 2],
                "table_physical_row_count_minimum": (
                    reality.MIN_TABLE_ROWS
                ),
                "table_physical_row_count_maximum": (
                    reality.MAX_TABLE_ROWS
                ),
                "column_count_minimum": 1,
                "column_count_maximum": MAX_COLUMNS,
                "question_character_count_maximum": (
                    MAX_QUESTION_CHARACTERS
                ),
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
                split: dict(
                    sorted(qualified.dispositions[split].items())
                )
                for split in ("train", "test")
            },
            "pack_commitments": {
                "A_form_action_view": base._pack_commitment(
                    a_form_action
                ),
                "A_form_label": base._pack_commitment(a_form_label),
                "A_hold_action_view": base._pack_commitment(
                    a_hold_action
                ),
                "A_hold_label": base._pack_commitment(a_hold_label),
                "controller_only_provenance": base._pack_commitment(
                    controller_provenance
                ),
            },
            "source_qualification_receipt_sha256": (
                qualified.safe_receipt["self_sha256"]
            ),
            "selector": SELECTOR_NAME,
            "protocol_amendment": "v5_repair_r1_pre_secret",
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


__all__ = [
    "BABEL_LOCALE",
    "CompilerConfig",
    "DEV_MEMBERS",
    "EXACT_REGULAR_MEMBERS",
    "PRODUCTION_ARCHIVE_GIT_BLOB_SHA1",
    "PRODUCTION_BABEL_VERSION",
    "PRODUCTION_PYTZ_VERSION",
    "REQUIRED_MEMBERS",
    "RELEASE_VERSION",
    "SELECTOR_NAME",
    "STUDY_ID",
    "VERSION",
    "WikiSQLSourceCompilerError",
    "compile_archive",
    "qualify_archive",
    "write_compilation",
]

"""Aggregate-only source qualification for Spider schema-linking.

This module opens only the four public TRAIN/DEV/schema members authorized by
the committed source-access receipt.  It never opens TEST, creates a selection
secret, executes an action or baseline, fits an evaluator, or computes an
effect score.  Individual database IDs, questions, SQL strings, schema names,
and values never appear in its output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
import zipfile


VERSION = "spider_p0_public_source_qualification_v1"
STUDY_ID = "SPIDER_P1_TYPED_SCHEMA_EXPANSION_EVALUATOR_L5_V1"
FAMILIES = (
    "ONE_FOREIGN_KEY_EDGE",
    "MULTI_FOREIGN_KEY_PATH",
    "NESTED_OR_SET_RELATION",
)
ALLOWED_MEMBERS = (
    "spider_data/train_spider.json",
    "spider_data/train_others.json",
    "spider_data/dev.json",
    "spider_data/tables.json",
)
FORBIDDEN_PREFIXES = (
    "spider_data/test.json",
    "spider_data/test_tables.json",
    "spider_data/test_database/",
)
QUALIFICATION_TIERS = (
    {
        "name": "floor",
        "train": {"A_form": 24, "F_search": 8, "A_hold": 12},
        "M_search": 12,
        "minimum_M_database_count_per_family": 6,
    },
    {
        "name": "target",
        "train": {"A_form": 36, "F_search": 12, "A_hold": 18},
        "M_search": 18,
        "minimum_M_database_count_per_family": 8,
    },
    {
        "name": "stretch",
        "train": {"A_form": 48, "F_search": 16, "A_hold": 24},
        "M_search": 24,
        "minimum_M_database_count_per_family": 10,
    },
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class SpiderQualificationError(RuntimeError):
    """The public source qualification failed closed."""


class RowContractError(ValueError):
    """One public row is ineligible under a safe reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class DatabaseSchema:
    database_id: str
    table_count: int
    column_table_ids: tuple[int, ...]
    foreign_table_edges: frozenset[tuple[int, int]]


@dataclass
class SQLFacts:
    table_ids: set[int]
    column_ids: set[int]
    nested_sql_count: int = 0
    set_operation_count: int = 0

    def merge(self, other: "SQLFacts", *, nested: bool = False) -> None:
        self.table_ids.update(other.table_ids)
        self.column_ids.update(other.column_ids)
        self.nested_sql_count += other.nested_sql_count + int(nested)
        self.set_operation_count += other.set_operation_count


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise SpiderQualificationError("self hash supplied twice")
    if os.path.lexists(path):
        raise SpiderQualificationError(f"output already exists: {path.name}")
    value = dict(body)
    value["self_sha256"] = stable_hash(body)
    raw = _canonical_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path):
            raise SpiderQualificationError(f"output raced: {path.name}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return value


def _read_self_hashed_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SpiderQualificationError("source-access receipt is unavailable")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise SpiderQualificationError("source-access receipt is not an object")
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if not isinstance(claimed, str) or not HEX64.fullmatch(claimed):
        raise SpiderQualificationError("source-access receipt lacks self hash")
    if claimed != stable_hash(body):
        raise SpiderQualificationError("source-access receipt self hash drifted")
    return value


def _required_list(value: object, code: str) -> list[Any]:
    if not isinstance(value, list):
        raise RowContractError(code)
    return value


def _required_mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RowContractError(code)
    return value


def _required_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RowContractError(code)
    return value


def _parse_schemas(value: object) -> dict[str, DatabaseSchema]:
    rows = _required_list(value, "tables_root_not_list")
    result: dict[str, DatabaseSchema] = {}
    for row in rows:
        item = _required_mapping(row, "table_row_not_object")
        database_id = item.get("db_id")
        if not isinstance(database_id, str) or not database_id:
            raise RowContractError("database_id_invalid")
        if database_id in result:
            raise RowContractError("database_id_duplicate")
        table_names = _required_list(
            item.get("table_names_original"), "table_names_invalid"
        )
        normalized_table_names = _required_list(
            item.get("table_names"), "normalized_table_names_invalid"
        )
        if len(table_names) != len(normalized_table_names) or not table_names:
            raise RowContractError("table_name_count_invalid")
        if not all(isinstance(name, str) for name in table_names):
            raise RowContractError("table_name_type_invalid")
        columns = _required_list(
            item.get("column_names_original"), "column_names_invalid"
        )
        normalized_columns = _required_list(
            item.get("column_names"), "normalized_column_names_invalid"
        )
        column_types = _required_list(
            item.get("column_types"), "column_types_invalid"
        )
        if not (
            len(columns) == len(normalized_columns) == len(column_types)
        ):
            raise RowContractError("column_count_invalid")
        column_table_ids: list[int] = []
        for column in columns:
            pair = _required_list(column, "column_pair_invalid")
            if len(pair) != 2 or not isinstance(pair[1], str):
                raise RowContractError("column_pair_invalid")
            table_id = _required_int(pair[0], "column_table_id_invalid")
            if table_id < -1 or table_id >= len(table_names):
                raise RowContractError("column_table_id_out_of_range")
            column_table_ids.append(table_id)
        primary_keys = _required_list(
            item.get("primary_keys"), "primary_keys_invalid"
        )
        for column_id in primary_keys:
            index = _required_int(column_id, "primary_key_invalid")
            if index <= 0 or index >= len(columns):
                raise RowContractError("primary_key_out_of_range")
        foreign_table_edges: set[tuple[int, int]] = set()
        foreign_keys = _required_list(
            item.get("foreign_keys"), "foreign_keys_invalid"
        )
        for relation in foreign_keys:
            pair = _required_list(relation, "foreign_key_pair_invalid")
            if len(pair) != 2:
                raise RowContractError("foreign_key_pair_invalid")
            left = _required_int(pair[0], "foreign_key_column_invalid")
            right = _required_int(pair[1], "foreign_key_column_invalid")
            if (
                left <= 0
                or right <= 0
                or left >= len(columns)
                or right >= len(columns)
            ):
                raise RowContractError("foreign_key_column_out_of_range")
            left_table = column_table_ids[left]
            right_table = column_table_ids[right]
            if left_table < 0 or right_table < 0:
                raise RowContractError("foreign_key_table_invalid")
            if left_table == right_table:
                continue
            foreign_table_edges.add(tuple(sorted((left_table, right_table))))
        result[database_id] = DatabaseSchema(
            database_id=database_id,
            table_count=len(table_names),
            column_table_ids=tuple(column_table_ids),
            foreign_table_edges=frozenset(foreign_table_edges),
        )
    return result


def _parse_column_unit(value: object, facts: SQLFacts) -> None:
    unit = _required_list(value, "column_unit_invalid")
    if len(unit) != 3:
        raise RowContractError("column_unit_invalid")
    _required_int(unit[0], "column_aggregate_invalid")
    column_id = _required_int(unit[1], "column_id_invalid")
    if not isinstance(unit[2], bool):
        raise RowContractError("column_distinct_invalid")
    facts.column_ids.add(column_id)


def _parse_value_unit(value: object, facts: SQLFacts) -> None:
    unit = _required_list(value, "value_unit_invalid")
    if len(unit) != 3:
        raise RowContractError("value_unit_invalid")
    _required_int(unit[0], "unit_operator_invalid")
    _parse_column_unit(unit[1], facts)
    if unit[2] is not None:
        _parse_column_unit(unit[2], facts)


def _looks_like_column_unit(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and isinstance(value[0], int)
        and not isinstance(value[0], bool)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
        and isinstance(value[2], bool)
    )


def _parse_condition_value(value: object, facts: SQLFacts) -> None:
    if isinstance(value, Mapping):
        nested = _parse_sql(value)
        facts.merge(nested, nested=True)
        return
    if _looks_like_column_unit(value):
        _parse_column_unit(value, facts)
        return
    if isinstance(value, list):
        for member in value:
            if isinstance(member, Mapping):
                nested = _parse_sql(member)
                facts.merge(nested, nested=True)
            elif _looks_like_column_unit(member):
                _parse_column_unit(member, facts)


def _parse_conditions(value: object, facts: SQLFacts) -> None:
    conditions = _required_list(value, "conditions_invalid")
    for index, condition in enumerate(conditions):
        if index % 2 == 1:
            if condition not in {"and", "or"}:
                raise RowContractError("condition_conjunction_invalid")
            continue
        item = _required_list(condition, "condition_invalid")
        if len(item) != 5 or not isinstance(item[0], bool):
            raise RowContractError("condition_invalid")
        _required_int(item[1], "condition_operator_invalid")
        _parse_value_unit(item[2], facts)
        _parse_condition_value(item[3], facts)
        _parse_condition_value(item[4], facts)


def _parse_sql(value: object) -> SQLFacts:
    sql = _required_mapping(value, "sql_not_object")
    facts = SQLFacts(table_ids=set(), column_ids=set())

    select = _required_list(sql.get("select"), "select_invalid")
    if len(select) != 2 or not isinstance(select[0], bool):
        raise RowContractError("select_invalid")
    for item in _required_list(select[1], "select_items_invalid"):
        pair = _required_list(item, "select_item_invalid")
        if len(pair) != 2:
            raise RowContractError("select_item_invalid")
        _required_int(pair[0], "select_aggregate_invalid")
        _parse_value_unit(pair[1], facts)

    from_clause = _required_mapping(sql.get("from"), "from_invalid")
    table_units = _required_list(
        from_clause.get("table_units"), "table_units_invalid"
    )
    for value_unit in table_units:
        pair = _required_list(value_unit, "table_unit_invalid")
        if len(pair) != 2:
            raise RowContractError("table_unit_invalid")
        if pair[0] == "table_unit":
            facts.table_ids.add(_required_int(pair[1], "table_id_invalid"))
        elif pair[0] == "sql":
            nested = _parse_sql(pair[1])
            facts.merge(nested, nested=True)
        else:
            raise RowContractError("table_unit_type_invalid")
    _parse_conditions(from_clause.get("conds"), facts)

    _parse_conditions(sql.get("where"), facts)
    _parse_conditions(sql.get("having"), facts)
    for column_unit in _required_list(sql.get("groupBy"), "group_by_invalid"):
        _parse_column_unit(column_unit, facts)

    order_by = sql.get("orderBy")
    if order_by:
        pair = _required_list(order_by, "order_by_invalid")
        if len(pair) != 2 or pair[0] not in {"asc", "desc"}:
            raise RowContractError("order_by_invalid")
        for value_unit in _required_list(pair[1], "order_values_invalid"):
            _parse_value_unit(value_unit, facts)

    for key in ("intersect", "union", "except"):
        nested_sql = sql.get(key)
        if nested_sql is not None:
            nested = _parse_sql(nested_sql)
            facts.merge(nested, nested=True)
            facts.set_operation_count += 1
    return facts


def _connected_by_declared_foreign_keys(
    table_ids: set[int],
    edges: frozenset[tuple[int, int]],
) -> bool:
    if not table_ids:
        return False
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        if left in table_ids and right in table_ids:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen = {next(iter(table_ids))}
    frontier = list(seen)
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return seen == table_ids


def _classify_row(
    row: object,
    schemas: Mapping[str, DatabaseSchema],
) -> tuple[str, str, int, int]:
    item = _required_mapping(row, "annotation_row_not_object")
    database_id = item.get("db_id")
    if not isinstance(database_id, str) or database_id not in schemas:
        raise RowContractError("annotation_database_unknown")
    if not isinstance(item.get("question"), str):
        raise RowContractError("question_invalid")
    if not isinstance(item.get("query"), str):
        raise RowContractError("query_invalid")
    schema = schemas[database_id]
    facts = _parse_sql(item.get("sql"))
    if any(
        table_id < 0 or table_id >= schema.table_count
        for table_id in facts.table_ids
    ):
        raise RowContractError("used_table_out_of_range")
    if any(
        column_id < 0 or column_id >= len(schema.column_table_ids)
        for column_id in facts.column_ids
    ):
        raise RowContractError("used_column_out_of_range")
    evidence = {
        column_id
        for column_id in facts.column_ids
        if column_id > 0 and schema.column_table_ids[column_id] >= 0
    }
    evidence_count = len(evidence)
    if not 2 <= evidence_count <= 5:
        raise RowContractError("gold_schema_evidence_count_outside_2_5")

    if facts.nested_sql_count > 0 or facts.set_operation_count > 0:
        family = "NESTED_OR_SET_RELATION"
    elif (
        len(facts.table_ids) == 2
        and tuple(sorted(facts.table_ids)) in schema.foreign_table_edges
    ):
        family = "ONE_FOREIGN_KEY_EDGE"
    elif (
        len(facts.table_ids) >= 3
        and _connected_by_declared_foreign_keys(
            facts.table_ids, schema.foreign_table_edges
        )
    ):
        family = "MULTI_FOREIGN_KEY_PATH"
    else:
        raise RowContractError("not_in_preregistered_relation_families")
    return family, database_id, evidence_count, len(facts.table_ids)


def _load_authorized_members(
    archive: Path,
    access: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    bindings = _required_mapping(
        access.get("key_member_bindings"), "source_access_member_bindings_invalid"
    )
    parsed: dict[str, Any] = {}
    receipts: dict[str, dict[str, Any]] = {}
    opened: list[str] = []
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        for forbidden in FORBIDDEN_PREFIXES:
            if not any(
                name == forbidden or name.startswith(forbidden)
                for name in names
            ):
                raise SpiderQualificationError(
                    "a frozen forbidden TEST boundary is absent"
                )
        for member in ALLOWED_MEMBERS:
            if member not in names:
                raise SpiderQualificationError(
                    "an authorized public member is absent"
                )
            info = package.getinfo(member)
            binding = _required_mapping(
                bindings.get(member), "authorized member binding absent"
            )
            if (
                info.file_size != binding.get("uncompressed_bytes")
                or info.compress_size != binding.get("compressed_bytes")
                or f"{info.CRC:08x}" != binding.get("crc32")
            ):
                raise SpiderQualificationError(
                    "authorized member central-directory binding drifted"
                )
            with package.open(member, "r") as handle:
                raw = handle.read()
            opened.append(member)
            if len(raw) != info.file_size:
                raise SpiderQualificationError(
                    "authorized member payload size drifted"
                )
            try:
                parsed[member] = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SpiderQualificationError(
                    "authorized member JSON decode failed"
                ) from exc
            receipts[member] = {
                "byte_count": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
    if tuple(opened) != ALLOWED_MEMBERS:
        raise SpiderQualificationError("authorized member open order drifted")
    return parsed, receipts


def _partition_for_seed(
    database_ids: Sequence[str],
    seed: int,
    block_demands: Mapping[str, int],
) -> dict[str, str]:
    ordered_blocks = tuple(block_demands)
    total = sum(block_demands.values())
    cumulative: list[tuple[int, str]] = []
    running = 0
    for block in ordered_blocks:
        running += block_demands[block]
        cumulative.append((running, block))
    assignment: dict[str, str] = {}
    for database_id in database_ids:
        bucket = int.from_bytes(
            hashlib.sha256(f"{seed}:{database_id}".encode("utf-8")).digest()[:8],
            "big",
        ) % total
        for boundary, block in cumulative:
            if bucket < boundary:
                assignment[database_id] = block
                break
    return assignment


def _find_database_disjoint_allocation(
    capacity: Mapping[str, Mapping[str, int]],
    block_demands: Mapping[str, int],
) -> dict[str, Any] | None:
    database_ids = sorted(capacity)
    for seed in range(65536):
        assignment = _partition_for_seed(database_ids, seed, block_demands)
        available = {
            block: {family: 0 for family in FAMILIES}
            for block in block_demands
        }
        database_counts = Counter(assignment.values())
        family_database_counts = {
            block: {family: 0 for family in FAMILIES}
            for block in block_demands
        }
        for database_id, block in assignment.items():
            for family in FAMILIES:
                count = int(capacity[database_id].get(family, 0))
                available[block][family] += count
                if count:
                    family_database_counts[block][family] += 1
        if all(
            available[block][family] >= demand
            for block, demand in block_demands.items()
            for family in FAMILIES
        ):
            commitment = stable_hash(
                sorted(assignment.items(), key=lambda item: item[0])
            )
            return {
                "partition_seed": seed,
                "assignment_commitment_sha256": commitment,
                "database_count_by_block": {
                    block: database_counts[block] for block in block_demands
                },
                "eligible_item_capacity_by_block_and_family": available,
                "eligible_database_count_by_block_and_family": (
                    family_database_counts
                ),
            }
    return None


def _qualify_rows(
    rows_by_split: Mapping[str, Sequence[object]],
    schemas: Mapping[str, DatabaseSchema],
) -> tuple[dict[str, Any], dict[str, dict[str, Counter[str]]]]:
    aggregates: dict[str, Any] = {}
    capacity: dict[str, dict[str, Counter[str]]] = {}
    for split, rows in rows_by_split.items():
        family_counts = Counter()
        evidence_histogram = Counter()
        used_table_histogram = Counter()
        invalid_reasons = Counter()
        family_databases: dict[str, set[str]] = {
            family: set() for family in FAMILIES
        }
        by_database: dict[str, Counter[str]] = defaultdict(Counter)
        row_commitments: set[str] = set()
        duplicate_row_count = 0
        for row in rows:
            if isinstance(row, Mapping):
                commitment = stable_hash(
                    {
                        "db_id": row.get("db_id"),
                        "question": row.get("question"),
                        "query": row.get("query"),
                    }
                )
                if commitment in row_commitments:
                    duplicate_row_count += 1
                row_commitments.add(commitment)
            try:
                family, database_id, evidence_count, used_table_count = (
                    _classify_row(row, schemas)
                )
            except RowContractError as exc:
                invalid_reasons[exc.code] += 1
                continue
            family_counts[family] += 1
            evidence_histogram[str(evidence_count)] += 1
            used_table_histogram[str(used_table_count)] += 1
            family_databases[family].add(database_id)
            by_database[database_id][family] += 1
        aggregates[split] = {
            "row_count": len(rows),
            "duplicate_row_commitment_count": duplicate_row_count,
            "eligible_item_count_by_family": {
                family: family_counts[family] for family in FAMILIES
            },
            "eligible_database_count_by_family": {
                family: len(family_databases[family]) for family in FAMILIES
            },
            "eligible_evidence_count_histogram": dict(
                sorted(evidence_histogram.items())
            ),
            "eligible_used_table_count_histogram": dict(
                sorted(used_table_histogram.items())
            ),
            "ineligible_reason_counts": dict(sorted(invalid_reasons.items())),
        }
        capacity[split] = by_database
    return aggregates, capacity


def qualify(
    *,
    archive: Path,
    source_access_path: Path,
    expected_source_access_self_sha256: str,
) -> dict[str, Any]:
    if not HEX64.fullmatch(expected_source_access_self_sha256):
        raise SpiderQualificationError("expected source-access hash is invalid")
    access = _read_self_hashed_json(source_access_path)
    if access.get("self_sha256") != expected_source_access_self_sha256:
        raise SpiderQualificationError("source-access receipt identity drifted")
    if access.get("status") != (
        "official_archive_and_code_acquired_central_directory_qualified_no_member_payload_open"
    ):
        raise SpiderQualificationError("source-access status is not authorized")
    acquisition = _required_mapping(
        access.get("acquisition"), "source_access_acquisition_invalid"
    )
    if archive.is_symlink() or not archive.is_file():
        raise SpiderQualificationError("archive is unavailable")
    if stat.S_IMODE(archive.stat().st_mode) != 0o400:
        raise SpiderQualificationError("archive mode drifted")
    if (
        archive.stat().st_size != acquisition.get("archive_byte_count")
        or file_sha256(archive) != acquisition.get("archive_sha256")
    ):
        raise SpiderQualificationError("archive byte identity drifted")

    parsed, member_receipts = _load_authorized_members(archive, access)
    schemas = _parse_schemas(parsed["spider_data/tables.json"])
    train_spider = _required_list(
        parsed["spider_data/train_spider.json"], "train_spider_not_list"
    )
    train_others = _required_list(
        parsed["spider_data/train_others.json"], "train_others_not_list"
    )
    dev = _required_list(parsed["spider_data/dev.json"], "dev_not_list")
    rows_by_split = {
        "train": [*train_spider, *train_others],
        "dev": dev,
    }
    row_aggregates, capacity = _qualify_rows(rows_by_split, schemas)
    train_database_ids = {
        row.get("db_id")
        for row in rows_by_split["train"]
        if isinstance(row, Mapping) and isinstance(row.get("db_id"), str)
    }
    dev_database_ids = {
        row.get("db_id")
        for row in rows_by_split["dev"]
        if isinstance(row, Mapping) and isinstance(row.get("db_id"), str)
    }
    overlap = train_database_ids & dev_database_ids

    tier_results: dict[str, Any] = {}
    highest_feasible: str | None = None
    for tier in QUALIFICATION_TIERS:
        train_allocation = _find_database_disjoint_allocation(
            capacity["train"], tier["train"]
        )
        dev_counts = row_aggregates["dev"]["eligible_item_count_by_family"]
        dev_database_counts = row_aggregates["dev"][
            "eligible_database_count_by_family"
        ]
        m_feasible = all(
            dev_counts[family] >= tier["M_search"]
            and dev_database_counts[family]
            >= tier["minimum_M_database_count_per_family"]
            for family in FAMILIES
        )
        feasible = (
            train_allocation is not None
            and m_feasible
            and not overlap
        )
        tier_results[tier["name"]] = {
            "feasible": feasible,
            "train_database_disjoint_allocation": train_allocation,
            "train_item_demand_per_family_by_block": tier["train"],
            "M_search_item_demand_per_family": tier["M_search"],
            "M_search_minimum_database_count_per_family": tier[
                "minimum_M_database_count_per_family"
            ],
            "M_search_capacity_passed": m_feasible,
        }
        if feasible:
            highest_feasible = tier["name"]

    schema_aggregates = {
        "database_count": len(schemas),
        "table_count": sum(schema.table_count for schema in schemas.values()),
        "column_count": sum(
            len(schema.column_table_ids) for schema in schemas.values()
        ),
        "declared_foreign_table_edge_count": sum(
            len(schema.foreign_table_edges) for schema in schemas.values()
        ),
        "train_database_count": len(train_database_ids),
        "dev_database_count": len(dev_database_ids),
        "train_dev_database_overlap_count": len(overlap),
    }
    status = (
        "passed_public_topology_capacity_qualification"
        if highest_feasible is not None
        else "terminal_public_topology_capacity_insufficient"
    )
    return {
        "schema": VERSION,
        "study_id": STUDY_ID,
        "status": status,
        "qualified_at_utc": _utc_now(),
        "source_binding": {
            "archive_byte_count": archive.stat().st_size,
            "archive_sha256": acquisition["archive_sha256"],
            "source_access_self_sha256": access["self_sha256"],
            "opened_public_member_count": len(ALLOWED_MEMBERS),
            "opened_public_members": list(ALLOWED_MEMBERS),
            "opened_public_member_receipts": member_receipts,
            "train_dev_SQLite_member_payload_open_count": 0,
            "test_annotation_table_or_database_payload_open_count": 0,
        },
        "schema_aggregates": schema_aggregates,
        "row_aggregates": row_aggregates,
        "qualification_tiers": tier_results,
        "highest_feasible_tier": highest_feasible,
        "claim_boundary": {
            "individual_database_question_SQL_schema_or_value_output_count": 0,
            "selection_HMAC_secret_or_cohort_count": 0,
            "action_RAW_HippoRAG_evaluator_or_score_count": 0,
            "model_GPU_provider_API_or_online_evaluator_call_count": 0,
            "effect_claim_authorized": False,
        },
        "next_stage_policy": {
            "formal_design_must_choose_highest_feasible_tier": True,
            "source_qualification_replay_allowed": False,
            "candidate_action_must_expand_beyond_RAW_top5": True,
            "test_payload_must_remain_unopened": True,
        },
    }


def _run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve(strict=False)
    if os.path.lexists(output_root):
        raise SpiderQualificationError("output root was already consumed")
    output_root.mkdir(mode=0o700)
    attempt = _write_new_json(
        output_root / "qualification.attempt.json",
        {
            "schema": f"{VERSION}_attempt_v1",
            "study_id": STUDY_ID,
            "status": "attempt_reserved_before_public_member_open",
            "reserved_at_utc": _utc_now(),
            "attempt_count": 1,
            "retry_replay_count": 0,
            "expected_source_access_self_sha256": (
                args.expected_source_access_self_sha256
            ),
            "test_payload_open_count": 0,
        },
    )
    result = qualify(
        archive=Path(args.archive).resolve(strict=True),
        source_access_path=Path(args.source_access).resolve(strict=True),
        expected_source_access_self_sha256=(
            args.expected_source_access_self_sha256
        ),
    )
    result["attempt_self_sha256"] = attempt["self_sha256"]
    terminal = _write_new_json(
        output_root / "qualification.result.json", result
    )
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "self_sha256": terminal["self_sha256"],
                "highest_feasible_tier": terminal["highest_feasible_tier"],
                "test_payload_open_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if terminal["highest_feasible_tier"] is not None else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--source-access", required=True)
    parser.add_argument("--expected-source-access-self-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return _run(_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SpiderQualificationError, RowContractError) as exc:
        print(
            f"Spider public source qualification failed: {type(exc).__name__}",
            file=os.sys.stderr,
        )
        raise SystemExit(1) from exc

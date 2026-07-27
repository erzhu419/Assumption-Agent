"""Aggregate-only BIRD topology/capacity qualification.

Only four members bound by the committed BIRD source-access receipt are
retrieved.  Each member uses one exact local-header Range GET and one exact
filename/extra/compressed-stream Range GET.  No database value, test split,
selection secret, action, baseline, evaluator, model, or effect score is
opened.  Output contains aggregate counts and commitments only.
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
import struct
from typing import Any
from urllib.request import Request, urlopen
import zlib

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError

from assumption_agent.benchmarks.spider_p0_public_source_qualification_v1 import (
    FAMILIES,
    QUALIFICATION_TIERS,
    RowContractError,
    _canonical_bytes,
    _connected_by_declared_foreign_keys,
    _find_database_disjoint_allocation,
    _required_int,
    _required_list,
    _required_mapping,
    _write_new_json,
    stable_hash,
)


VERSION = "bird_p0_public_source_qualification_v1"
STUDY_ID = "BIRD_P1_TYPED_SCHEMA_EXPANSION_EVALUATOR_L5_V1"
SQLGLOT_VERSION = "30.13.0"
LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ARCHIVES = {
    "train": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
    },
    "dev": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
    },
}
MEMBER_ORDER = (
    "train/train.json",
    "train/train_tables.json",
    "dev_20240627/dev.json",
    "dev_20240627/dev_tables.json",
)


class BirdQualificationError(RuntimeError):
    """The BIRD public qualification failed closed."""


@dataclass(frozen=True)
class BirdDatabaseSchema:
    database_id: str
    table_count: int
    table_lookup: Mapping[str, int]
    table_names: tuple[str, ...]
    column_lookup_by_table: tuple[Mapping[str, int], ...]
    column_table_ids: tuple[int, ...]
    foreign_table_edges: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class BirdSQLFacts:
    table_ids: frozenset[int]
    column_ids: frozenset[int]
    nested_or_set: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_etag(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"')


def _canonical_identifier(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _read_self_hashed_json(
    path: Path,
    *,
    expected_self_sha256: str,
    label: str,
) -> dict[str, Any]:
    if not HEX64.fullmatch(expected_self_sha256):
        raise BirdQualificationError(f"{label} expected self hash is invalid")
    if path.is_symlink() or not path.is_file():
        raise BirdQualificationError(f"{label} is unavailable")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise BirdQualificationError(f"{label} is not an object")
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if not isinstance(claimed, str) or not HEX64.fullmatch(claimed):
        raise BirdQualificationError(f"{label} lacks a self hash")
    if claimed != stable_hash(body):
        raise BirdQualificationError(f"{label} self hash drifted")
    if claimed != expected_self_sha256:
        raise BirdQualificationError(f"{label} identity drifted")
    return value


def _range_get(
    *,
    url: str,
    start: int,
    end: int,
    expected_total: int,
    expected_etag: str,
    expected_last_modified: str,
) -> bytes:
    request = Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": f"{VERSION}/1",
        },
        method="GET",
    )
    with urlopen(request, timeout=120) as response:
        if getattr(response, "status", None) != 206:
            raise BirdQualificationError("member Range GET did not return 206")
        expected_range = f"bytes {start}-{end}/{expected_total}"
        if response.headers.get("Content-Range") != expected_range:
            raise BirdQualificationError("member Content-Range drifted")
        if (
            _normalize_etag(response.headers.get("ETag"))
            != _normalize_etag(expected_etag)
        ):
            raise BirdQualificationError("member archive ETag drifted")
        if (
            response.headers.get("Last-Modified", "").strip()
            != expected_last_modified
        ):
            raise BirdQualificationError(
                "member archive Last-Modified drifted"
            )
        raw = response.read()
    if len(raw) != end - start + 1:
        raise BirdQualificationError("member Range GET byte count drifted")
    return raw


def _open_bound_member(
    member: str,
    binding: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    archive_name = binding.get("archive")
    if archive_name not in ARCHIVES:
        raise BirdQualificationError("member archive binding is invalid")
    url = ARCHIVES[str(archive_name)]["url"]
    local_header_offset = _required_int(
        binding.get("local_header_offset"), "member_header_offset_invalid"
    )
    archive_size = _required_int(
        archive.get("archive_byte_count"), "archive_size_invalid"
    )
    expected_etag = archive.get("etag")
    expected_last_modified = archive.get("last_modified")
    if not isinstance(expected_etag, str) or not isinstance(
        expected_last_modified, str
    ):
        raise BirdQualificationError("archive HTTP validator is invalid")

    header = _range_get(
        url=url,
        start=local_header_offset,
        end=local_header_offset + 29,
        expected_total=archive_size,
        expected_etag=expected_etag,
        expected_last_modified=expected_last_modified,
    )
    (
        signature,
        version_needed,
        flags,
        method,
        modified_time,
        modified_date,
        local_crc32,
        local_compressed_size,
        local_uncompressed_size,
        name_length,
        extra_length,
    ) = struct.unpack("<4s5H3L2H", header)
    if signature != LOCAL_FILE_SIGNATURE:
        raise BirdQualificationError("local member header signature drifted")
    if flags & 0x0001:
        raise BirdQualificationError("encrypted member is not authorized")
    if method != binding.get("compression_method") or method != 8:
        raise BirdQualificationError("member compression method drifted")
    if flags != binding.get("flags"):
        raise BirdQualificationError("member general-purpose flags drifted")
    if version_needed < 20:
        raise BirdQualificationError("member ZIP version is invalid")

    compressed_size = _required_int(
        binding.get("compressed_bytes"), "member_compressed_size_invalid"
    )
    uncompressed_size = _required_int(
        binding.get("uncompressed_bytes"), "member_size_invalid"
    )
    if not flags & 0x0008:
        if (
            local_crc32 != int(str(binding.get("crc32")), 16)
            or local_compressed_size not in {compressed_size, 0xFFFFFFFF}
            or local_uncompressed_size not in {uncompressed_size, 0xFFFFFFFF}
        ):
            raise BirdQualificationError("local member size or CRC drifted")

    variable_and_payload_start = local_header_offset + 30
    variable_and_payload_size = (
        name_length + extra_length + compressed_size
    )
    variable_and_payload = _range_get(
        url=url,
        start=variable_and_payload_start,
        end=variable_and_payload_start + variable_and_payload_size - 1,
        expected_total=archive_size,
        expected_etag=expected_etag,
        expected_last_modified=expected_last_modified,
    )
    raw_name = variable_and_payload[:name_length]
    name_encoding = "utf-8" if flags & 0x0800 else "cp437"
    try:
        local_name = raw_name.decode(name_encoding)
    except UnicodeDecodeError as exc:
        raise BirdQualificationError("local member name decode failed") from exc
    if local_name != member:
        raise BirdQualificationError("local member name drifted")
    compressed_start = name_length + extra_length
    compressed = variable_and_payload[
        compressed_start : compressed_start + compressed_size
    ]
    if len(compressed) != compressed_size:
        raise BirdQualificationError("compressed member stream is truncated")
    try:
        raw = zlib.decompress(compressed, -15)
    except zlib.error as exc:
        raise BirdQualificationError("member DEFLATE decode failed") from exc
    if (
        len(raw) != uncompressed_size
        or f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}" != binding.get("crc32")
    ):
        raise BirdQualificationError("member payload size or CRC drifted")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BirdQualificationError("member JSON decode failed") from exc
    receipt = {
        "compressed_byte_count": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "local_extra_byte_count": extra_length,
        "local_header_byte_count": 30,
        "local_header_sha256": hashlib.sha256(header).hexdigest(),
        "payload_byte_count": len(raw),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "range_GET_attempt_count": 2,
    }
    return parsed, receipt


def _register_unique(
    lookup: dict[str, int | None],
    name: str,
    index: int,
) -> None:
    key = _canonical_identifier(name)
    if not key:
        raise RowContractError("empty_schema_identifier")
    if key not in lookup:
        lookup[key] = index
    elif lookup[key] != index:
        lookup[key] = None


def _parse_schemas(value: object) -> dict[str, BirdDatabaseSchema]:
    rows = _required_list(value, "tables_root_not_list")
    result: dict[str, BirdDatabaseSchema] = {}
    for row in rows:
        item = _required_mapping(row, "table_row_not_object")
        database_id = item.get("db_id")
        if not isinstance(database_id, str) or not database_id:
            raise RowContractError("database_id_invalid")
        if database_id in result:
            raise RowContractError("database_id_duplicate")
        original_tables = _required_list(
            item.get("table_names_original"), "table_names_invalid"
        )
        normalized_tables = _required_list(
            item.get("table_names"), "normalized_table_names_invalid"
        )
        if (
            not original_tables
            or len(original_tables) != len(normalized_tables)
            or not all(isinstance(name, str) for name in original_tables)
            or not all(isinstance(name, str) for name in normalized_tables)
        ):
            raise RowContractError("table_name_count_invalid")
        table_lookup_mutable: dict[str, int | None] = {}
        for table_id, names in enumerate(
            zip(original_tables, normalized_tables, strict=True)
        ):
            for name in set(names):
                _register_unique(table_lookup_mutable, name, table_id)
        table_lookup = {
            key: value
            for key, value in table_lookup_mutable.items()
            if value is not None
        }

        original_columns = _required_list(
            item.get("column_names_original"), "column_names_invalid"
        )
        normalized_columns = _required_list(
            item.get("column_names"), "normalized_column_names_invalid"
        )
        column_types = _required_list(
            item.get("column_types"), "column_types_invalid"
        )
        if not (
            len(original_columns)
            == len(normalized_columns)
            == len(column_types)
        ):
            raise RowContractError("column_count_invalid")
        column_table_ids: list[int] = []
        column_maps: list[dict[str, int | None]] = [
            {} for _ in original_tables
        ]
        for column_id, (original, normalized) in enumerate(
            zip(original_columns, normalized_columns, strict=True)
        ):
            original_pair = _required_list(original, "column_pair_invalid")
            normalized_pair = _required_list(
                normalized, "normalized_column_pair_invalid"
            )
            if (
                len(original_pair) != 2
                or len(normalized_pair) != 2
                or original_pair[0] != normalized_pair[0]
                or not isinstance(original_pair[1], str)
                or not isinstance(normalized_pair[1], str)
            ):
                raise RowContractError("column_pair_invalid")
            table_id = _required_int(
                original_pair[0], "column_table_id_invalid"
            )
            if table_id < -1 or table_id >= len(original_tables):
                raise RowContractError("column_table_id_out_of_range")
            column_table_ids.append(table_id)
            if table_id >= 0:
                for name in {original_pair[1], normalized_pair[1]}:
                    _register_unique(
                        column_maps[table_id], name, column_id
                    )

        primary_keys = _required_list(
            item.get("primary_keys"), "primary_keys_invalid"
        )
        for column_id in primary_keys:
            index = _required_int(column_id, "primary_key_invalid")
            if index <= 0 or index >= len(original_columns):
                raise RowContractError("primary_key_out_of_range")
        foreign_edges: set[tuple[int, int]] = set()
        for relation in _required_list(
            item.get("foreign_keys"), "foreign_keys_invalid"
        ):
            pair = _required_list(relation, "foreign_key_pair_invalid")
            if len(pair) != 2:
                raise RowContractError("foreign_key_pair_invalid")
            left = _required_int(pair[0], "foreign_key_column_invalid")
            right = _required_int(pair[1], "foreign_key_column_invalid")
            if (
                left <= 0
                or right <= 0
                or left >= len(original_columns)
                or right >= len(original_columns)
            ):
                raise RowContractError("foreign_key_column_out_of_range")
            left_table = column_table_ids[left]
            right_table = column_table_ids[right]
            if left_table < 0 or right_table < 0:
                raise RowContractError("foreign_key_table_invalid")
            if left_table != right_table:
                foreign_edges.add(tuple(sorted((left_table, right_table))))
        result[database_id] = BirdDatabaseSchema(
            database_id=database_id,
            table_count=len(original_tables),
            table_lookup=table_lookup,
            table_names=tuple(str(name) for name in original_tables),
            column_lookup_by_table=tuple(
                {
                    key: value
                    for key, value in column_map.items()
                    if value is not None
                }
                for column_map in column_maps
            ),
            column_table_ids=tuple(column_table_ids),
            foreign_table_edges=frozenset(foreign_edges),
        )
    return result


def _parse_sql_facts(sql: str, schema: BirdDatabaseSchema) -> BirdSQLFacts:
    try:
        expression = sqlglot.parse_one(
            sql,
            read="sqlite",
            error_level=ErrorLevel.RAISE,
        )
    except (ParseError, ValueError, TypeError) as exc:
        raise RowContractError("SQL_parse_failed") from exc
    if expression is None:
        raise RowContractError("SQL_parse_failed")

    cte_names = {
        _canonical_identifier(node.alias_or_name)
        for node in expression.find_all(exp.CTE)
        if node.alias_or_name
    }
    alias_to_table: dict[str, int | None] = {}
    table_ids: set[int] = set()
    for table in expression.find_all(exp.Table):
        name = table.name
        if not name:
            raise RowContractError("SQL_table_name_missing")
        canonical_name = _canonical_identifier(name)
        if canonical_name in cte_names:
            continue
        table_id = schema.table_lookup.get(canonical_name)
        if table_id is None:
            raise RowContractError("SQL_table_unknown_or_ambiguous")
        table_ids.add(table_id)
        for alias in {table.alias_or_name, name}:
            if not alias:
                continue
            key = _canonical_identifier(alias)
            previous = alias_to_table.get(key)
            if previous is None and key not in alias_to_table:
                alias_to_table[key] = table_id
            elif previous != table_id:
                alias_to_table[key] = None
    if not table_ids:
        raise RowContractError("SQL_table_set_empty")

    column_ids: set[int] = set()
    for column in expression.find_all(exp.Column):
        name = column.name
        if not name or name == "*":
            continue
        column_name = _canonical_identifier(name)
        qualifier = column.table
        if qualifier:
            table_id = alias_to_table.get(_canonical_identifier(qualifier))
            if table_id is None:
                raise RowContractError(
                    "SQL_column_qualifier_unknown_or_ambiguous"
                )
            column_id = schema.column_lookup_by_table[table_id].get(
                column_name
            )
            if column_id is None:
                raise RowContractError("SQL_column_unknown_or_ambiguous")
            column_ids.add(column_id)
            continue
        candidates = {
            schema.column_lookup_by_table[table_id][column_name]
            for table_id in table_ids
            if column_name in schema.column_lookup_by_table[table_id]
        }
        if len(candidates) != 1:
            raise RowContractError("SQL_unqualified_column_ambiguous")
        column_ids.update(candidates)

    nested_or_set = (
        sum(1 for _ in expression.find_all(exp.Select)) > 1
        or any(
            isinstance(node, (exp.Union, exp.Except, exp.Intersect))
            for node in expression.walk()
        )
    )
    return BirdSQLFacts(
        table_ids=frozenset(table_ids),
        column_ids=frozenset(column_ids),
        nested_or_set=nested_or_set,
    )


def _classify_row(
    row: object,
    schemas: Mapping[str, BirdDatabaseSchema],
) -> tuple[str, str, int, int]:
    item = _required_mapping(row, "annotation_row_not_object")
    database_id = item.get("db_id")
    if not isinstance(database_id, str) or database_id not in schemas:
        raise RowContractError("annotation_database_unknown")
    if not isinstance(item.get("question"), str):
        raise RowContractError("question_invalid")
    sql = item.get("SQL")
    if not isinstance(sql, str) or not sql.strip():
        raise RowContractError("SQL_invalid")
    schema = schemas[database_id]
    facts = _parse_sql_facts(sql, schema)
    evidence_count = len(facts.column_ids)
    if not 2 <= evidence_count <= 5:
        raise RowContractError("gold_schema_evidence_count_outside_2_5")

    table_ids = set(facts.table_ids)
    if facts.nested_or_set:
        family = "NESTED_OR_SET_RELATION"
    elif (
        len(table_ids) == 2
        and tuple(sorted(table_ids)) in schema.foreign_table_edges
    ):
        family = "ONE_FOREIGN_KEY_EDGE"
    elif (
        len(table_ids) >= 3
        and _connected_by_declared_foreign_keys(
            table_ids, schema.foreign_table_edges
        )
    ):
        family = "MULTI_FOREIGN_KEY_PATH"
    else:
        raise RowContractError("not_in_preregistered_relation_families")
    return family, database_id, evidence_count, len(table_ids)


def _qualify_rows(
    rows_by_split: Mapping[str, Sequence[object]],
    schemas: Mapping[str, BirdDatabaseSchema],
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
                        "SQL": row.get("SQL"),
                        "db_id": row.get("db_id"),
                        "question": row.get("question"),
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
            "duplicate_row_commitment_count": duplicate_row_count,
            "eligible_database_count_by_family": {
                family: len(family_databases[family]) for family in FAMILIES
            },
            "eligible_evidence_count_histogram": dict(
                sorted(evidence_histogram.items())
            ),
            "eligible_item_count_by_family": {
                family: family_counts[family] for family in FAMILIES
            },
            "eligible_used_table_count_histogram": dict(
                sorted(used_table_histogram.items())
            ),
            "ineligible_reason_counts": dict(sorted(invalid_reasons.items())),
            "row_count": len(rows),
        }
        capacity[split] = by_database
    return aggregates, capacity


def qualify(
    *,
    source_access_path: Path,
    expected_source_access_self_sha256: str,
    transport_addendum_path: Path,
    expected_transport_addendum_self_sha256: str,
) -> dict[str, Any]:
    if sqlglot.__version__ != SQLGLOT_VERSION:
        raise BirdQualificationError("sqlglot runtime version drifted")
    access = _read_self_hashed_json(
        source_access_path,
        expected_self_sha256=expected_source_access_self_sha256,
        label="source access",
    )
    if access.get("status") != (
        "official_archives_central_directories_qualified_no_member_semantic_payload_open"
    ):
        raise BirdQualificationError("source access is not authorized")
    addendum = _read_self_hashed_json(
        transport_addendum_path,
        expected_self_sha256=expected_transport_addendum_self_sha256,
        label="transport addendum",
    )
    if (
        addendum.get("source_access_self_sha256")
        != expected_source_access_self_sha256
    ):
        raise BirdQualificationError("transport addendum source drifted")

    bindings = _required_mapping(
        access.get("key_member_bindings"), "member_bindings_invalid"
    )
    acquisition = _required_mapping(
        access.get("acquisition"), "source_acquisition_invalid"
    )
    archives = {
        "train": _required_mapping(
            acquisition.get("train_archive"), "train_archive_invalid"
        ),
        "dev": _required_mapping(
            acquisition.get("dev_archive"), "dev_archive_invalid"
        ),
    }
    parsed: dict[str, Any] = {}
    member_receipts: dict[str, Any] = {}
    for member in MEMBER_ORDER:
        binding = _required_mapping(
            bindings.get(member), "member_binding_absent"
        )
        archive = archives[str(binding.get("archive"))]
        parsed[member], member_receipts[member] = _open_bound_member(
            member, binding, archive
        )

    train_schemas = _parse_schemas(parsed["train/train_tables.json"])
    dev_schemas = _parse_schemas(parsed["dev_20240627/dev_tables.json"])
    overlap = set(train_schemas) & set(dev_schemas)
    schemas = {**train_schemas, **dev_schemas}
    train_rows = _required_list(
        parsed["train/train.json"], "train_rows_not_list"
    )
    dev_rows = _required_list(
        parsed["dev_20240627/dev.json"], "dev_rows_not_list"
    )
    rows_by_split = {"dev": dev_rows, "train": train_rows}
    row_aggregates, capacity = _qualify_rows(rows_by_split, schemas)

    train_database_ids = {
        row.get("db_id")
        for row in train_rows
        if isinstance(row, Mapping) and isinstance(row.get("db_id"), str)
    }
    dev_database_ids = {
        row.get("db_id")
        for row in dev_rows
        if isinstance(row, Mapping) and isinstance(row.get("db_id"), str)
    }
    annotation_overlap = train_database_ids & dev_database_ids
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
            and not annotation_overlap
        )
        tier_results[tier["name"]] = {
            "M_search_capacity_passed": m_feasible,
            "M_search_item_demand_per_family": tier["M_search"],
            "M_search_minimum_database_count_per_family": tier[
                "minimum_M_database_count_per_family"
            ],
            "feasible": feasible,
            "train_database_disjoint_allocation": train_allocation,
            "train_item_demand_per_family_by_block": tier["train"],
        }
        if feasible:
            highest_feasible = tier["name"]

    status = (
        "passed_public_topology_capacity_qualification"
        if highest_feasible is not None
        else "terminal_public_topology_capacity_insufficient"
    )
    return {
        "claim_boundary": {
            "action_RAW_HippoRAG_evaluator_or_score_count": 0,
            "effect_claim_authorized": False,
            "individual_database_question_SQL_schema_or_value_output_count": 0,
            "model_GPU_provider_API_or_online_evaluator_call_count": 0,
            "selection_HMAC_secret_or_cohort_count": 0,
        },
        "highest_feasible_tier": highest_feasible,
        "next_stage_policy": {
            "candidate_action_must_expand_beyond_RAW_top5": True,
            "if_not_feasible_close_typed_schema_expansion_without_another_source": True,
            "source_qualification_replay_allowed": False,
            "test_and_database_value_payloads_must_remain_unopened": True,
        },
        "qualification_tiers": tier_results,
        "qualified_at_utc": _utc_now(),
        "row_aggregates": row_aggregates,
        "runtime_binding": {
            "sqlglot_version": sqlglot.__version__,
        },
        "schema": VERSION,
        "schema_aggregates": {
            "column_count": sum(
                len(schema.column_table_ids) for schema in schemas.values()
            ),
            "database_count": len(schemas),
            "declared_foreign_table_edge_count": sum(
                len(schema.foreign_table_edges) for schema in schemas.values()
            ),
            "dev_database_count": len(dev_database_ids),
            "table_count": sum(
                schema.table_count for schema in schemas.values()
            ),
            "train_database_count": len(train_database_ids),
            "train_dev_annotation_database_overlap_count": len(
                annotation_overlap
            ),
            "train_dev_schema_database_overlap_count": len(overlap),
        },
        "source_binding": {
            "archive_full_GET_count": 0,
            "database_file_or_value_payload_open_count": 0,
            "opened_public_member_count": len(MEMBER_ORDER),
            "opened_public_member_receipts": member_receipts,
            "opened_public_members": list(MEMBER_ORDER),
            "source_access_self_sha256": access["self_sha256"],
            "test_payload_open_count": 0,
            "transport_addendum_self_sha256": addendum["self_sha256"],
        },
        "status": status,
        "study_id": STUDY_ID,
    }


def _run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve(strict=False)
    if os.path.lexists(output_root):
        raise BirdQualificationError("output root was already consumed")
    output_root.mkdir(mode=0o700)
    attempt = _write_new_json(
        output_root / "qualification.attempt.json",
        {
            "attempt_count": 1,
            "expected_source_access_self_sha256": (
                args.expected_source_access_self_sha256
            ),
            "expected_transport_addendum_self_sha256": (
                args.expected_transport_addendum_self_sha256
            ),
            "member_payload_open_count_before_reservation": 0,
            "reserved_at_utc": _utc_now(),
            "retry_replay_count": 0,
            "schema": f"{VERSION}_attempt_v1",
            "status": "attempt_reserved_before_public_member_open",
            "study_id": STUDY_ID,
            "test_or_database_value_payload_open_count": 0,
        },
    )
    result = qualify(
        source_access_path=Path(args.source_access).resolve(strict=True),
        expected_source_access_self_sha256=(
            args.expected_source_access_self_sha256
        ),
        transport_addendum_path=Path(
            args.transport_addendum
        ).resolve(strict=True),
        expected_transport_addendum_self_sha256=(
            args.expected_transport_addendum_self_sha256
        ),
    )
    result["attempt_self_sha256"] = attempt["self_sha256"]
    terminal = _write_new_json(
        output_root / "qualification.result.json", result
    )
    print(
        json.dumps(
            {
                "highest_feasible_tier": terminal[
                    "highest_feasible_tier"
                ],
                "self_sha256": terminal["self_sha256"],
                "status": terminal["status"],
                "test_or_database_value_payload_open_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if terminal["highest_feasible_tier"] is not None else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-access", required=True)
    parser.add_argument("--expected-source-access-self-sha256", required=True)
    parser.add_argument("--transport-addendum", required=True)
    parser.add_argument(
        "--expected-transport-addendum-self-sha256", required=True
    )
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return _run(_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BirdQualificationError, RowContractError) as exc:
        print(
            f"BIRD public source qualification failed: {type(exc).__name__}",
            file=os.sys.stderr,
        )
        raise SystemExit(1) from exc

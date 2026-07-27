from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile

from assumption_agent.benchmarks import (
    spider_p0_public_source_qualification_v1 as spider,
)


def _schema(database_id: str) -> dict[str, object]:
    return {
        "db_id": database_id,
        "table_names_original": ["alpha", "beta", "gamma"],
        "table_names": ["alpha", "beta", "gamma"],
        "column_names_original": [
            [-1, "*"],
            [0, "alpha_id"],
            [0, "alpha_name"],
            [1, "alpha_id"],
            [1, "beta_name"],
            [2, "beta_id"],
            [2, "gamma_name"],
        ],
        "column_names": [
            [-1, "*"],
            [0, "alpha id"],
            [0, "alpha name"],
            [1, "alpha id"],
            [1, "beta name"],
            [2, "beta id"],
            [2, "gamma name"],
        ],
        "column_types": [
            "text",
            "number",
            "text",
            "number",
            "text",
            "number",
            "text",
        ],
        "primary_keys": [1, 3, 5],
        "foreign_keys": [[1, 3], [3, 5]],
    }


def _column(column_id: int) -> list[object]:
    return [0, column_id, False]


def _value(column_id: int) -> list[object]:
    return [0, _column(column_id), None]


def _condition(left: int, right: int) -> list[object]:
    return [False, 2, _value(left), _column(right), None]


def _sql(
    *,
    tables: list[int],
    select_column: int,
    conditions: list[object] | None = None,
    where: list[object] | None = None,
) -> dict[str, object]:
    return {
        "select": [False, [[0, _value(select_column)]]],
        "from": {
            "table_units": [["table_unit", table] for table in tables],
            "conds": conditions or [],
        },
        "where": where or [],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }


def _row(
    database_id: str,
    family: str,
    suffix: str,
) -> dict[str, object]:
    if family == "ONE_FOREIGN_KEY_EDGE":
        sql = _sql(
            tables=[0, 1],
            select_column=4,
            conditions=[_condition(1, 3)],
        )
    elif family == "MULTI_FOREIGN_KEY_PATH":
        sql = _sql(
            tables=[0, 1, 2],
            select_column=6,
            conditions=[
                _condition(1, 3),
                "and",
                _condition(3, 5),
            ],
        )
    elif family == "NESTED_OR_SET_RELATION":
        nested = _sql(tables=[1], select_column=4)
        sql = _sql(
            tables=[0],
            select_column=2,
            where=[[False, 8, _value(2), nested, None]],
        )
    else:
        raise AssertionError(family)
    return {
        "db_id": database_id,
        "question": f"synthetic public question {database_id} {family} {suffix}",
        "question_toks": ["synthetic"],
        "query": f"SELECT synthetic_{suffix}",
        "query_toks": ["SELECT"],
        "query_toks_no_value": ["select"],
        "sql": sql,
    }


def _member_binding(info: zipfile.ZipInfo) -> dict[str, object]:
    return {
        "uncompressed_bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
    }


def _write_package(
    path: Path,
    *,
    train_spider: list[object],
    train_others: list[object],
    dev: list[object],
    tables: list[object],
) -> dict[str, dict[str, object]]:
    values = {
        "spider_data/train_spider.json": train_spider,
        "spider_data/train_others.json": train_others,
        "spider_data/dev.json": dev,
        "spider_data/tables.json": tables,
        "spider_data/test.json": [{"forbidden": True}],
        "spider_data/test_tables.json": [{"forbidden": True}],
        "spider_data/test_database/forbidden/forbidden.sqlite": b"forbidden",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, value in values.items():
            raw = (
                value
                if isinstance(value, bytes)
                else json.dumps(value, separators=(",", ":")).encode()
            )
            package.writestr(name, raw)
    with zipfile.ZipFile(path) as package:
        return {
            member: _member_binding(package.getinfo(member))
            for member in spider.ALLOWED_MEMBERS
        }


def _write_access(
    path: Path,
    archive: Path,
    bindings: dict[str, dict[str, object]],
) -> dict[str, object]:
    body = {
        "schema": "spider_p1_public_source_access_v1",
        "status": (
            "official_archive_and_code_acquired_central_directory_qualified_no_member_payload_open"
        ),
        "acquisition": {
            "archive_byte_count": archive.stat().st_size,
            "archive_sha256": spider.file_sha256(archive),
        },
        "key_member_bindings": bindings,
    }
    value = {**body, "self_sha256": spider.stable_hash(body)}
    path.write_bytes(spider._canonical_bytes(value) + b"\n")
    return value


def test_typed_sql_parser_distinguishes_three_relation_families() -> None:
    schemas = spider._parse_schemas([_schema("db")])
    for family in spider.FAMILIES:
        observed, database_id, evidence_count, table_count = (
            spider._classify_row(_row("db", family, "0"), schemas)
        )
        assert observed == family
        assert database_id == "db"
        assert 2 <= evidence_count <= 5
        assert table_count >= 1


def test_database_disjoint_partition_is_capacity_checked() -> None:
    capacity = {
        f"db_{index:03d}": {
            family: 4 for family in spider.FAMILIES
        }
        for index in range(40)
    }
    allocation = spider._find_database_disjoint_allocation(
        capacity,
        {"A_form": 24, "F_search": 8, "A_hold": 12},
    )
    assert allocation is not None
    assert len(allocation["assignment_commitment_sha256"]) == 64
    for block, demand in {
        "A_form": 24,
        "F_search": 8,
        "A_hold": 12,
    }.items():
        for family in spider.FAMILIES:
            assert (
                allocation["eligible_item_capacity_by_block_and_family"][block][
                    family
                ]
                >= demand
            )


def test_loader_opens_only_four_authorized_members(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "source.zip"
    bindings = _write_package(
        archive,
        train_spider=[],
        train_others=[],
        dev=[],
        tables=[],
    )
    opened: list[str] = []
    original = zipfile.ZipFile.open

    def tracked_open(self, name, *args, **kwargs):
        opened.append(name if isinstance(name, str) else name.filename)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", tracked_open)
    parsed, receipts = spider._load_authorized_members(
        archive, {"key_member_bindings": bindings}
    )
    assert tuple(opened) == spider.ALLOWED_MEMBERS
    assert set(parsed) == set(spider.ALLOWED_MEMBERS)
    assert set(receipts) == set(spider.ALLOWED_MEMBERS)
    assert not any(name.startswith("spider_data/test") for name in opened)


def test_full_public_qualification_selects_highest_feasible_tier(
    tmp_path: Path,
) -> None:
    train_databases = [f"train_{index:03d}" for index in range(24)]
    dev_databases = [f"dev_{index:03d}" for index in range(12)]
    tables = [
        *(_schema(database_id) for database_id in train_databases),
        *(_schema(database_id) for database_id in dev_databases),
    ]
    train_rows = [
        _row(database_id, family, str(repetition))
        for database_id in train_databases
        for family in spider.FAMILIES
        for repetition in range(3)
    ]
    dev_rows = [
        _row(database_id, family, "0")
        for database_id in dev_databases
        for family in spider.FAMILIES
    ]
    archive = tmp_path / "source.zip"
    bindings = _write_package(
        archive,
        train_spider=train_rows,
        train_others=[],
        dev=dev_rows,
        tables=tables,
    )
    os.chmod(archive, 0o400)
    access_path = tmp_path / "access.json"
    access = _write_access(access_path, archive, bindings)

    result = spider.qualify(
        archive=archive,
        source_access_path=access_path,
        expected_source_access_self_sha256=access["self_sha256"],
    )

    assert result["status"] == "passed_public_topology_capacity_qualification"
    assert result["highest_feasible_tier"] == "floor"
    assert result["schema_aggregates"]["train_dev_database_overlap_count"] == 0
    assert result["source_binding"]["opened_public_member_count"] == 4
    assert (
        result["source_binding"][
            "test_annotation_table_or_database_payload_open_count"
        ]
        == 0
    )
    assert result["claim_boundary"][
        "action_RAW_HippoRAG_evaluator_or_score_count"
    ] == 0


def test_cli_receipts_are_mode_0600_and_self_hashed(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    value = spider._write_new_json(
        output / "receipt.json",
        {"schema": "test", "count": 0},
    )
    assert stat.S_IMODE((output / "receipt.json").stat().st_mode) == 0o600
    body = dict(value)
    claimed = body.pop("self_sha256")
    assert claimed == spider.stable_hash(body)

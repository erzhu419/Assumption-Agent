from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
from typing import Mapping

import pytest

from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as base,
)
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v5_repair as repair,
)


SECRET = b"r" * reality.HMAC_SECRET_BYTES
FAMILY_OPERATOR = {"EQ": 0, "GT": 1, "LT": 2}


def _jsonl(rows: list[Mapping[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for row in rows
        )
        + "\n"
    ).encode("ascii")


def _table_document(
    table_id: str,
    *,
    header: list[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": table_id,
        "header": ["Name", "Score"] if header is None else header,
        "types": ["text", "real"],
        "rows": [[f"name-{index}", index] for index in range(11)],
        "name": f"table {table_id}",
    }
    if metadata is not None:
        result.pop("name")
        result.update(metadata)
    return result


def _query_document(table_id: str, family: str) -> dict[str, object]:
    threshold = {"EQ": 5, "GT": 8, "LT": 2}[family]
    return {
        "phase": 1,
        "question": f"Find the row whose score is {family} {threshold}.",
        "sql": {
            "sel": 0,
            "agg": 0,
            "conds": [[1, FAMILY_OPERATOR[family], threshold]],
        },
        "table_id": table_id,
    }


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _compiled_item(table_id: str, family: str, serial: int) -> base.CompiledItem:
    table = reality.WikiSQLTable(
        table_id=table_id,
        header=("Score",),
        types=("real",),
        rows=tuple((index,) for index in range(11)),
    )
    operator = FAMILY_OPERATOR[family]
    query = reality.WikiSQLQuery(
        select_index=0,
        aggregation_index=0,
        conditions=(
            reality.WikiSQLCondition(
                column_index=0,
                operator_index=operator,
                value=5,
            ),
        ),
    )
    return base.CompiledItem(
        split="train",
        line_number=serial + 1,
        item_commitment_sha256=_sha256(
            f"item:{table_id}:{family}:{serial}"
        ),
        table_commitment_sha256=_sha256(f"table:{table_id}"),
        question=f"question {serial}",
        source_table_id=table_id,
        table=table,
        query=query,
        family=family,
        raw_sql={"sel": 0, "agg": 0, "conds": [[0, operator, 5]]},
        gold_row_ids=(5,),
    )


def test_projects_all_three_official_metadata_envelopes() -> None:
    rows = [
        _table_document("minimal"),
        _table_document(
            "named-rich",
            metadata={
                "name": "display name",
                "caption": "caption",
                "page_title": "page",
                "section_title": "section",
            },
        ),
        _table_document(
            "page-rich",
            metadata={
                "page_id": 17,
                "caption": "caption",
                "page_title": "page",
                "section_title": "section",
            },
        ),
    ]

    registry = repair._load_table_registry(  # noqa: SLF001
        _jsonl(rows),
        member_name="data/train.tables.jsonl",
    )

    assert set(registry.eligible) == {
        "minimal",
        "named-rich",
        "page-rich",
    }
    assert set(registry.database_validation) == set(registry.eligible)
    assert all(
        table.header == ("Name", "Score")
        for table in registry.eligible.values()
    )
    assert sum(registry.schema_variant_counts.values()) == 3
    assert len(registry.schema_variant_counts) == 3

    drifted = dict(rows[0], undocumented="must fail closed")
    with pytest.raises(
        repair.WikiSQLSourceCompilerError,
        match="metadata envelope drifted",
    ):
        repair._load_table_registry(  # noqa: SLF001
            _jsonl([drifted]),
            member_name="data/train.tables.jsonl",
        )


def test_blank_header_is_registry_only_and_query_reference_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blank = _table_document("train-blank", header=[" ", "Score"])
    test = _table_document("test-normal")
    train_tables = _jsonl([blank])
    test_tables = _jsonl([test])
    train_queries = _jsonl([_query_document("train-blank", "EQ")])
    test_queries = _jsonl([_query_document("test-normal", "EQ")])

    registry = repair._load_table_registry(  # noqa: SLF001
        train_tables,
        member_name="data/train.tables.jsonl",
    )
    assert set(registry.database_validation) == {"train-blank"}
    assert registry.eligible == {}
    assert registry.unreferenced_blank_header_table_ids == {
        "train-blank"
    }
    assert registry.database_validation["train-blank"].header == (
        "__blank_column_0__",
        "Score",
    )
    assert repair._query_table_ids(  # noqa: SLF001
        train_queries,
        member_name="data/train.jsonl",
    ) == ("train-blank",)

    members = {
        repair.TABLE_MEMBERS["train"]: train_tables,
        repair.TABLE_MEMBERS["test"]: test_tables,
        repair.JSONL_MEMBERS["train"]: train_queries,
        repair.JSONL_MEMBERS["test"]: test_queries,
        repair.DB_MEMBERS["train"]: b"SQLite format 3\x00",
        repair.DB_MEMBERS["test"]: b"SQLite format 3\x00",
    }
    archive = repair.RepairArchive(
        members=members,
        member_sha256=(),
        archive_git_blob_sha1="0" * 40,
        regular_member_count=9,
        directory_member_count=1,
        ignored_regular_member_count=3,
    )
    monkeypatch.setattr(
        repair,
        "_read_archive",
        lambda *_args, **_kwargs: archive,
    )
    monkeypatch.setattr(
        repair,
        "EXPECTED_TABLE_COUNTS",
        {"train": 1, "test": 1},
    )
    monkeypatch.setattr(
        repair,
        "EXPECTED_QUERY_COUNTS",
        {"train": 1, "test": 1},
    )

    with pytest.raises(
        repair.WikiSQLSourceCompilerError,
        match="blank-header table is query referenced",
    ):
        repair.qualify_archive(
            tmp_path / "unused.tar.bz2",
            expected_archive_sha256="a" * 64,
            config=repair.CompilerConfig.synthetic_test(),
        )


def test_real_affinity_accepts_official_text_storage() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute('CREATE TABLE "table_mixed" (col0 REAL)')
        connection.executemany(
            'INSERT INTO "table_mixed" (col0) VALUES (?)',
            [("12-abc",), ("missing",)],
        )
        table = reality.WikiSQLTable(
            table_id="mixed",
            header=("Value",),
            types=("real",),
            rows=(("12-ABC",), ("MISSING",)),
        )

        repair._verify_official_storage(  # noqa: SLF001
            connection,
            "table_mixed",
            table,
        )

        connection.execute(
            'UPDATE "table_mixed" SET col0 = ? WHERE rowid = 2',
            ("different",),
        )
        with pytest.raises(
            repair.WikiSQLSourceCompilerError,
            match="normalized cell disagrees",
        ):
            repair._verify_official_storage(  # noqa: SLF001
                connection,
                "table_mixed",
                table,
            )
    finally:
        connection.close()


def test_hall_capacity_rejects_cross_family_table_overlap() -> None:
    rows = (
        _compiled_item("shared", "EQ", 0),
        _compiled_item("shared", "GT", 1),
        _compiled_item("lt-only", "LT", 2),
    )

    with pytest.raises(
        repair.WikiSQLSourceIneligibleError,
        match="Hall condition failed",
    ):
        repair._hall_capacity(  # noqa: SLF001
            rows,
            quota_per_family=1,
        )


def test_capacity_preserving_selector_resolves_adversarial_overlap() -> None:
    rows = (
        _compiled_item("shared", "EQ", 0),
        _compiled_item("shared", "GT", 1),
        _compiled_item("eq-only", "EQ", 2),
        _compiled_item("lt-only", "LT", 3),
    )
    support, _floor, hall = repair._hall_capacity(  # noqa: SLF001
        rows,
        quota_per_family=1,
    )
    assert support == {"EQ": 2, "GT": 1, "LT": 1}
    assert hall["EQ+GT+LT"] == 3

    selected = repair._select_block(  # noqa: SLF001
        secret=SECRET,
        block="A_form",
        items=rows,
        quota_per_family=1,
    )

    assert {row.family for row in selected} == {"EQ", "GT", "LT"}
    assert len({row.table_commitment_sha256 for row in selected}) == 3
    assert next(row for row in selected if row.family == "GT").source_table_id == (
        "shared"
    )
    assert next(row for row in selected if row.family == "EQ").source_table_id == (
        "eq-only"
    )


def _sqlite_archive_member(
    path: Path,
    tables: list[Mapping[str, object]],
) -> bytes:
    connection = sqlite3.connect(path)
    try:
        for table in tables:
            table_id = table["id"]
            assert isinstance(table_id, str)
            name = "table_" + table_id.replace("-", "_")
            connection.execute(
                f'CREATE TABLE "{name}" (col0 TEXT, col1 REAL)'
            )
            connection.executemany(
                f'INSERT INTO "{name}" (col0, col1) VALUES (?, ?)',
                [
                    (str(row[0]).lower(), row[1])
                    for row in table["rows"]  # type: ignore[union-attr]
                ],
            )
        connection.commit()
    finally:
        connection.close()
    return path.read_bytes()


def _add_member(bundle: tarfile.TarFile, name: str, raw: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.mode = 0o600
    info.size = len(raw)
    bundle.addfile(info, io.BytesIO(raw))


def _build_feasible_archive(
    tmp_path: Path,
) -> tuple[Path, str, dict[str, int], dict[str, int]]:
    quotas = {"train": 4, "test": 2}
    tables: dict[str, list[dict[str, object]]] = {}
    queries: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "test"):
        tables[split] = []
        queries[split] = []
        for family in reality.FAMILY_ORDER:
            for index in range(quotas[split]):
                table_id = f"{split}-{family}-{index}"
                tables[split].append(_table_document(table_id))
                queries[split].append(_query_document(table_id, family))

    members: dict[str, bytes] = {}
    for split in ("train", "test"):
        members[repair.TABLE_MEMBERS[split]] = _jsonl(tables[split])
        members[repair.JSONL_MEMBERS[split]] = _jsonl(queries[split])
        members[repair.DB_MEMBERS[split]] = _sqlite_archive_member(
            tmp_path / f"{split}.db",
            tables[split],
        )
    members.update(
        {
            "data/dev.jsonl": b"",
            "data/dev.tables.jsonl": b"",
            "data/dev.db": b"unused development database",
        }
    )
    archive_path = tmp_path / "source.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as bundle:
        directory = tarfile.TarInfo("data")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o700
        bundle.addfile(directory)
        for name in sorted(repair.EXACT_REGULAR_MEMBERS):
            _add_member(bundle, name, members[name])
    return (
        archive_path,
        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        {split: len(tables[split]) for split in ("train", "test")},
        {split: len(queries[split]) for split in ("train", "test")},
    )


def test_qualify_archive_is_full_source_feasible_without_generating_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path, archive_sha256, table_counts, query_counts = (
        _build_feasible_archive(tmp_path)
    )
    def forbidden_secret(_width: int) -> bytes:
        raise AssertionError("source qualification generated a secret")

    monkeypatch.setattr(repair.secrets, "token_bytes", forbidden_secret)

    receipt = repair.qualify_archive(
        archive_path,
        expected_archive_sha256=archive_sha256,
        config=repair.CompilerConfig.synthetic_test(),
    )

    assert receipt["status"] == "passed_full_public_source_adapter"
    assert receipt["qualification_runtime_mode"] == "synthetic_test"
    assert receipt["pytz_runtime_version"] == repair.PRODUCTION_PYTZ_VERSION
    assert receipt["secret_generation_count"] == 0
    assert receipt["HMAC_selection_count"] == 0
    assert receipt["cohort_selection_count"] == 0
    assert receipt["sqlite_eligible_count"] == table_counts
    assert receipt["sqlite_exhaustive_crosscheck_count"] == sum(
        table_counts.values()
    )

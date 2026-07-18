from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    feverous_p6_e2_identity_parallel_performance_diagnostic_v1 as module,
)


def _page(page_id: str, ordinal: int) -> dict[str, Any]:
    return {
        "title": page_id,
        "order": ["sentence_0", "table_0", "list_0"],
        "sentence_0": f"Sentence {ordinal}.",
        "table_0": {
            "type": "normal",
            "caption": "",
            "table": [
                [
                    {
                        "id": "cell_0_0_0",
                        "value": "" if ordinal % 5 == 0 else str(ordinal),
                        "is_header": False,
                        "row_span": "1",
                        "column_span": "1",
                    }
                ]
            ],
        },
        "list_0": {
            "type": "ordered_list",
            "list": [
                {
                    "id": "item_0_0",
                    "value": "" if ordinal % 7 == 0 else f"Item {ordinal}",
                    "level": 0,
                }
            ],
        },
    }


def _database(
    tmp_path: Path, *, rows: int = 96, gap: bool = False
) -> tuple[Path, module.DiagnosticDatabaseSpec]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "fixture_parallel_wiki.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE wiki (id PRIMARY KEY, data json)")
    for ordinal in range(rows + int(gap)):
        page = _page(f"Fixture_Parallel_Page_{ordinal:04d}", ordinal)
        connection.execute(
            "INSERT INTO wiki (id, data) VALUES (?, ?)",
            (
                page["title"],
                json.dumps(page, ensure_ascii=False, separators=(",", ":")),
            ),
        )
    if gap:
        connection.execute("DELETE FROM wiki WHERE rowid = 2")
    connection.commit()
    connection.close()
    path.chmod(0o600)
    spec = module.DiagnosticDatabaseSpec(
        basename=path.name,
        size_bytes=path.stat().st_size,
        declared_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        row_count=rows,
        required_mode=path.stat().st_mode & 0o777,
    )
    return path, spec


def test_exact_chunk_merge_matches_one_worker_on_identical_interval(
    tmp_path: Path,
) -> None:
    path, spec = _database(tmp_path)
    interval = module.RowidInterval(1, 64)
    serial = module.scan_parallel_configuration(
        path,
        spec=spec,
        interval=interval,
        worker_count=1,
        heap_capacity=16,
    )
    parallel = module.scan_parallel_configuration(
        path,
        spec=spec,
        interval=interval,
        worker_count=4,
        heap_capacity=16,
    )
    assert module.partition_rowid_interval(interval, 4) == (
        module.RowidInterval(1, 16),
        module.RowidInterval(17, 32),
        module.RowidInterval(33, 48),
        module.RowidInterval(49, 64),
    )
    assert serial.page_count == parallel.page_count == 64
    assert serial.payload_utf8_bytes == parallel.payload_utf8_bytes
    assert serial.eligible_identity_count == parallel.eligible_identity_count
    assert serial.excluded_empty_count == parallel.excluded_empty_count
    assert serial.hmac_evaluation_count == parallel.hmac_evaluation_count
    assert (
        serial.ordered_page_stream_sha256
        == parallel.ordered_page_stream_sha256
    )
    assert serial.retained_bottom_k_sha256 == parallel.retained_bottom_k_sha256
    assert [row.rank for row in serial.retained_candidates] == [
        row.rank for row in parallel.retained_candidates
    ]


def test_disjoint_configuration_receipt_is_aggregate_only_and_tamper_evident(
    tmp_path: Path,
) -> None:
    path, spec = _database(tmp_path)
    receipt = module.run_parallel_performance_diagnostic(
        path,
        spec=spec,
        worker_counts=(1, 2, 4),
        first_rowid=1,
        pages_per_configuration=24,
        heap_capacity=12,
    )
    assert module.verify_parallel_performance_diagnostic_receipt(receipt)
    assert [row["start_rowid"] for row in receipt["configurations"]] == [
        1,
        25,
        49,
    ]
    assert all(
        row["public_hmac_evaluation_count"]
        == row["observed_eligible_identity_count"]
        for row in receipt["configurations"]
    )
    serialized = json.dumps(receipt, sort_keys=True)
    assert "Fixture_Parallel_Page" not in serialized
    assert "Sentence" not in serialized
    assert receipt["formal_selection_secret_created_derived_or_accessed"] is False
    assert receipt["bottom_k_unit_ids_serialized"] is False

    tampered = copy.deepcopy(dict(receipt))
    tampered["configurations"][1]["start_rowid"] += 1
    with pytest.raises(
        module.FeverousIdentityParallelPerformanceDiagnosticError,
        match="drifted",
    ):
        module.verify_parallel_performance_diagnostic_receipt(tampered)


def test_rowid_gap_fails_closed_across_worker_boundary(tmp_path: Path) -> None:
    path, spec = _database(tmp_path, rows=32, gap=True)
    with pytest.raises(
        module.FeverousIdentityParallelPerformanceDiagnosticError,
        match="worker failed closed",
    ):
        module.scan_parallel_configuration(
            path,
            spec=spec,
            interval=module.RowidInterval(1, 32),
            worker_count=4,
            heap_capacity=8,
        )

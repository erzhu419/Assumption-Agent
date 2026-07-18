from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    feverous_p6_e2_identity_performance_diagnostic_v1 as module,
)


def _page(page_id: str, ordinal: int) -> dict[str, Any]:
    return {
        "title": page_id,
        "order": ["sentence_0", "table_0"],
        "sentence_0": f"Sentence {ordinal}.",
        "table_0": {
            "type": "normal",
            "caption": "" if ordinal % 7 == 0 else f"Caption {ordinal}",
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
    }


def _database(tmp_path: Path, *, rows: int = 100, gap: bool = False) -> tuple[Path, module.DiagnosticDatabaseSpec]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "fixture_wiki.db"
    connection = sqlite3.connect(path)
    connection.execute(module.FROZEN_DATABASE_SCHEMA)
    for ordinal in range(rows + int(gap)):
        page = _page(f"Fixture_Page_{ordinal:04d}", ordinal)
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
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = module.DiagnosticDatabaseSpec(
        basename=path.name,
        size_bytes=path.stat().st_size,
        declared_sha256=digest,
        row_count=rows,
        required_mode=path.stat().st_mode & 0o777,
    )
    return path, spec


def test_bounded_prefix_continues_to_cap_and_real_crosscheck_is_aggregate_only(
    tmp_path: Path,
) -> None:
    path, spec = _database(tmp_path)
    receipt = module.run_identity_performance_diagnostic(
        path,
        spec=spec,
        prefix_minimum=10,
        prefix_maximum=50,
        continue_threshold_seconds=60,
        sample_page_count=16,
    )
    assert receipt["observed_prefix_page_count"] == 50
    assert receipt["real_crosscheck_sample_page_count"] == 16
    assert receipt["identity_full_compiler_mismatch_count"] == 0
    assert receipt["formal_valid"] is False
    assert receipt["cohort_candidate_canonical_set_or_corpus_selected"] is False
    assert module.verify_identity_performance_diagnostic_receipt(receipt)
    serialized = json.dumps(receipt, sort_keys=True)
    assert "Fixture_Page" not in serialized
    assert "Sentence" not in serialized

    tampered = copy.deepcopy(dict(receipt))
    tampered["observed_prefix_page_count"] = 51
    with pytest.raises(
        module.FeverousIdentityPerformanceDiagnosticError,
        match="drifted",
    ):
        module.verify_identity_performance_diagnostic_receipt(tampered)


def test_elapsed_threshold_stops_at_minimum_and_rowid_gap_fails_closed(
    tmp_path: Path,
) -> None:
    path, spec = _database(tmp_path / "threshold")
    receipt = module.run_identity_performance_diagnostic(
        path,
        spec=spec,
        prefix_minimum=10,
        prefix_maximum=50,
        continue_threshold_seconds=0,
        sample_page_count=8,
    )
    assert receipt["observed_prefix_page_count"] == 10
    assert receipt["prefix_stop_decision"] == (
        "stopped_at_minimum_elapsed_threshold"
    )

    gap_path, gap_spec = _database(tmp_path / "gap", gap=True)
    with pytest.raises(
        module.FeverousIdentityPerformanceDiagnosticError,
        match="row identity drifted",
    ):
        module.run_identity_performance_diagnostic(
            gap_path,
            spec=gap_spec,
            prefix_minimum=10,
            prefix_maximum=50,
            continue_threshold_seconds=0,
            sample_page_count=8,
        )

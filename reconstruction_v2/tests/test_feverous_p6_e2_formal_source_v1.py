from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as module
from assumption_agent.benchmarks.feverous_p6_e2_source_adapter_v1 import (
    ANNOTATION_QUALIFICATION_SHA256,
    DESIGN_SHA256,
    WIKIPEDIA_QUALIFICATION_SHA256,
)


OFFICIAL_BLANK_SENTINEL = {
    "annotator_operations": "",
    "challenge": "",
    "claim": "",
    "evidence": "",
    "id": "",
    "label": "",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4096):
            digest.update(chunk)
    return digest.hexdigest()


def _write_annotation(
    path: Path,
    *,
    sentinel: dict[str, Any] | None = None,
) -> None:
    rows: list[dict[str, Any]] = [
        OFFICIAL_BLANK_SENTINEL if sentinel is None else sentinel,
        {
            "annotator_operations": [],
            "challenge": "Other",
            "claim": "Fixture claim.",
            "evidence": [],
            "id": 1,
            "label": "NOT ENOUGH INFO",
        },
    ]
    path.write_bytes(
        b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
            for row in rows
        )
    )
    path.chmod(0o600)


def _write_database(path: Path, *, with_rowid_gap: bool = False) -> None:
    connection = sqlite3.connect(path)
    connection.execute(module.FROZEN_DATABASE_SCHEMA)
    connection.execute(
        "INSERT INTO wiki (id, data) VALUES (?, ?)",
        ("Page_Z", '{"title":"Page_Z","order":[]}'),
    )
    if with_rowid_gap:
        connection.execute(
            "INSERT INTO wiki (id, data) VALUES (?, ?)",
            ("Deleted", '{"title":"Deleted","order":[]}'),
        )
    connection.execute(
        "INSERT INTO wiki (id, data) VALUES (?, ?)",
        ("Page_A", '{"title":"Page_A","order":[]}'),
    )
    if with_rowid_gap:
        connection.execute("DELETE FROM wiki WHERE id = 'Deleted'")
    connection.commit()
    connection.close()
    path.chmod(0o600)


def _fixture_spec(annotation: Path, database: Path, *, rows: int = 2) -> module.TrainSourceSpec:
    return module.TrainSourceSpec(
        source_split="TRAIN",
        design_sha256=DESIGN_SHA256,
        annotation_qualification_sha256=ANNOTATION_QUALIFICATION_SHA256,
        wikipedia_qualification_sha256=WIKIPEDIA_QUALIFICATION_SHA256,
        annotation_basename=annotation.name,
        annotation_size_bytes=annotation.stat().st_size,
        annotation_sha256=_sha256(annotation),
        annotation_nonblank_rows=1,
        annotation_blank_sentinel_rows=1,
        database_basename=database.name,
        database_size_bytes=database.stat().st_size,
        database_sha256=_sha256(database),
        database_row_count=rows,
        required_mode=annotation.stat().st_mode & 0o777,
    )


def _source(tmp_path: Path, *, rowid_gap: bool = False) -> module.ControlledTrainSource:
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotation = tmp_path / "synthetic_train.jsonl"
    database = tmp_path / "synthetic_wiki.db"
    _write_annotation(annotation)
    _write_database(database, with_rowid_gap=rowid_gap)
    return module.ControlledTrainSource(
        annotation_path=annotation,
        database_path=database,
        spec=_fixture_spec(annotation, database),
    )


def test_annotation_is_one_controlled_read_and_synthetic_never_becomes_formal(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    records = source.read_annotations_once()
    assert len(records) == 2 and records[0] == OFFICIAL_BLANK_SENTINEL
    receipt = source.annotation_receipt
    assert receipt["formal_source"] is False
    assert receipt["annotation_file_read_count"] == 1
    assert receipt["annotation_nonblank_rows"] == 1
    assert receipt["annotation_blank_sentinel_rows"] == 1
    assert module.verify_annotation_receipt(receipt)
    with pytest.raises(module.FeverousFormalSourceError, match="one-shot"):
        source.read_annotations_once()
    source.close()


@pytest.mark.parametrize(
    "near_sentinel",
    [
        {},
        {**OFFICIAL_BLANK_SENTINEL, "evidence": []},
        {**OFFICIAL_BLANK_SENTINEL, "unexpected": ""},
    ],
)
def test_falsy_near_sentinel_is_not_counted_as_official_blank_row(
    tmp_path: Path,
    near_sentinel: dict[str, Any],
) -> None:
    annotation = tmp_path / "synthetic_train.jsonl"
    database = tmp_path / "synthetic_wiki.db"
    _write_annotation(annotation, sentinel=near_sentinel)
    _write_database(database)
    source = module.ControlledTrainSource(
        annotation_path=annotation,
        database_path=database,
        spec=_fixture_spec(annotation, database),
    )
    with pytest.raises(
        module.FeverousFormalSourceError,
        match="content differs from its frozen binding",
    ):
        source.read_annotations_once()
    source.close()


def test_realistic_sqlite_partial_then_complete_exhaustion_and_selected_lookup(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    stream = source.iter_database_pages_once()
    # Physical rowid order is deliberately not lexicographic page-id order.
    assert next(stream)[0] == "Page_Z"
    with pytest.raises(module.FeverousFormalSourceError, match="before normal exhaustion"):
        stream.aggregate_receipt()
    assert [page_id for page_id, _ in stream] == ["Page_A"]
    receipt = stream.aggregate_receipt()
    assert receipt["observed_database_row_count"] == 2
    assert receipt["page_order"] == "strict_consecutive_rowid_physical_table_order"
    assert module.verify_database_page_stream_receipt(receipt)
    with pytest.raises(module.FeverousFormalSourceError, match="not formal-valid"):
        module.require_formal_database_page_stream_receipt(receipt)
    with pytest.raises(module.FeverousFormalSourceError, match="one-shot"):
        source.iter_database_pages_once()

    lookup = source.iter_selected_pages_once(["Page_A", "Page_Z"])
    assert [page_id for page_id, _ in lookup] == ["Page_A", "Page_Z"]
    lookup_receipt = lookup.aggregate_receipt()
    assert lookup_receipt["selected_page_count"] == 2
    assert module.verify_selected_page_lookup_receipt(lookup_receipt)
    source.close()


def test_duplicate_or_out_of_order_selected_ids_and_rowid_gap_fail_closed(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "ordered")
    list(source.iter_database_pages_once())
    with pytest.raises(module.FeverousFormalSourceError, match="strict binary order"):
        source.iter_selected_pages_once(["Page_A", "Page_A"])
    source.close()

    gap_source = _source(tmp_path / "gap", rowid_gap=True)
    gap_stream = gap_source.iter_database_pages_once()
    assert next(gap_stream)[0] == "Page_Z"
    with pytest.raises(module.FeverousFormalSourceError, match="row types drifted"):
        next(gap_stream)
    with pytest.raises(module.FeverousFormalSourceError, match="before normal exhaustion"):
        gap_stream.aggregate_receipt()
    gap_source.close()


def test_receipt_forgery_and_source_substitution_are_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path / "receipt")
    stream = source.iter_database_pages_once()
    list(stream)
    receipt = dict(stream.aggregate_receipt())
    tampered = copy.deepcopy(receipt)
    tampered["observed_database_row_count"] = 999
    with pytest.raises(module.FeverousFormalSourceError, match="self-hash mismatch"):
        module.verify_database_page_stream_receipt(tampered)

    # Even a re-self-hashed fixture cannot assert formal status because the
    # exact frozen source-spec digest and public file bindings are checked.
    forged = dict(receipt)
    forged.pop("database_page_stream_receipt_sha256")
    forged["formal_source"] = True
    forged["database_page_stream_receipt_sha256"] = module._stable_hash(forged)
    with pytest.raises(module.FeverousFormalSourceError, match="not formal-valid"):
        module.require_formal_database_page_stream_receipt(forged)
    source.close()

    annotation = tmp_path / "substitution" / "synthetic_train.jsonl"
    database = tmp_path / "substitution" / "synthetic_wiki.db"
    annotation.parent.mkdir(parents=True)
    _write_annotation(annotation)
    _write_database(database)
    spec = _fixture_spec(annotation, database)
    annotation.write_bytes(annotation.read_bytes() + b"{}\n")
    annotation.chmod(0o600)
    substituted = module.ControlledTrainSource(
        annotation_path=annotation,
        database_path=database,
        spec=spec,
    )
    with pytest.raises(module.FeverousFormalSourceError, match="byte size"):
        substituted.read_annotations_once()

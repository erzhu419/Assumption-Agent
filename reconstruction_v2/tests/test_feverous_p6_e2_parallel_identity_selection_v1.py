from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent.benchmarks import feverous_atomic_corpus_v1 as atomic
from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as formal_source
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as source_adapter
from assumption_agent.benchmarks import (
    feverous_p6_e2_parallel_identity_selection_v1 as module,
)


SECRET = b"P" * 32


def _page(page_id: str, ordinal: int) -> dict[str, Any]:
    return {
        "title": page_id,
        "order": ["sentence_0", "table_0", "list_0"],
        "sentence_0": f"Sentence {ordinal}",
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
            "type": "unordered_list",
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
) -> tuple[Path, module.BoundDatabase]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "parallel_formal_wiki.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE wiki (id PRIMARY KEY, data json)")
    for ordinal in range(rows + int(gap)):
        page = _page(f"Parallel_Formal_Page_{ordinal:04d}", ordinal)
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
    observed = path.stat()
    binding = module.BoundDatabase(
        basename=path.name,
        size_bytes=observed.st_size,
        declared_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        row_count=rows,
        schema="CREATE TABLE wiki (id PRIMARY KEY, data json)",
        required_mode=observed.st_mode & 0o777,
        device=observed.st_dev,
        inode=observed.st_ino,
        mtime_ns=observed.st_mtime_ns,
        ctime_ns=observed.st_ctime_ns,
        source_spec_sha256="1" * 64,
        source_binding_sha256="2" * 64,
        formal_source_opener_source_sha256="3" * 64,
        formal_source=False,
    )
    return path, binding


def _all_identities(path: Path) -> list[acquisition.CorpusIdentity]:
    connection = sqlite3.connect(path)
    output: list[acquisition.CorpusIdentity] = []
    for page_id, raw_page in connection.execute(
        "SELECT id, data FROM wiki ORDER BY rowid"
    ):
        enumeration = atomic.enumerate_official_page_atomic_identities(
            page_id, raw_page
        )
        output.extend(
            acquisition.CorpusIdentity(
                unit_key=f"{page_id}_{row.local_id}",
                page=page_id,
                local_id=row.local_id,
                unit_type=row.unit_type,
                official_ordinal=row.official_ordinal,
                target_sha256=row.target_sha256,
            )
            for row in enumeration.identities
        )
    connection.close()
    return output


def test_four_worker_exact_cover_matches_one_worker_and_serial_bottom_k(
    tmp_path: Path,
) -> None:
    path, binding = _database(tmp_path)
    identities = _all_identities(path)
    gold_key = identities[3].unit_key
    forbidden_key = identities[9].unit_key
    kwargs = {
        "database_path": path,
        "binding": binding,
        "secret": SECRET,
        "gold_keys": frozenset({gold_key}),
        "forbidden_alternatives": frozenset({forbidden_key}),
        "needed": 20,
    }
    one = module._run_exact_cover_selection(**kwargs, worker_count=1)
    four = module._run_exact_cover_selection(**kwargs, worker_count=4)
    assert module.partition_exact_cover(10, 4) == (
        module.RowidInterval(1, 3),
        module.RowidInterval(4, 6),
        module.RowidInterval(7, 8),
        module.RowidInterval(9, 10),
    )
    assert one.page_count == four.page_count == binding.row_count
    assert one.payload_utf8_bytes == four.payload_utf8_bytes
    assert one.eligible_identity_count == four.eligible_identity_count
    assert one.excluded_empty_count == four.excluded_empty_count
    assert one.hmac_evaluation_count == four.hmac_evaluation_count
    assert [row.unit_key for row in one.retained_gold] == [gold_key]
    assert [row.unit_key for row in four.retained_gold] == [gold_key]
    assert one.qualification_page_ids == four.qualification_page_ids
    page_ids = sorted({row.page for row in identities})
    expected_sample = tuple(
        sorted(
            sorted(
                page_ids,
                key=lambda page_id: (
                    hashlib.sha256(
                        b"feverous_p6_e2/identity_compiler_real_sample/v1\x00"
                        + page_id.encode("utf-8")
                    ).digest(),
                    page_id.encode("utf-8"),
                ),
            )[:64],
            key=lambda page_id: page_id.encode("utf-8"),
        )
    )
    assert one.qualification_page_ids == expected_sample
    assert [row.unit_key for row in one.retained_distractors] == [
        row.unit_key for row in four.retained_distractors
    ]

    expected = sorted(
        (
            (
                acquisition.hmac_digest(
                    SECRET, "distractor_order", row.page, row.local_id
                ),
                row.page,
                row.local_id,
                row.unit_key,
                row,
            )
            for row in identities
            if row.unit_key not in {gold_key, forbidden_key}
        ),
        key=lambda row: row[:4],
    )[:20]
    assert [row.unit_key for row in four.retained_distractors] == [
        row[4].unit_key for row in expected
    ]


def test_gap_fails_closed_in_parallel_worker(tmp_path: Path) -> None:
    path, binding = _database(tmp_path, rows=32, gap=True)
    with pytest.raises(
        module.FeverousParallelIdentitySelectionError,
        match="worker failed closed",
    ):
        module._run_exact_cover_selection(
            database_path=path,
            binding=binding,
            secret=SECRET,
            gold_keys=frozenset(),
            forbidden_alternatives=frozenset(),
            needed=8,
            worker_count=4,
        )


def test_controlled_source_delegates_only_after_hash_and_candidate_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _binding = _database(tmp_path / "source", rows=8)
    annotation = tmp_path / "source" / "train.jsonl"
    annotation.write_text("{}\n", encoding="utf-8")
    annotation.chmod(0o600)
    annotation_bytes = annotation.read_bytes()
    database_bytes = path.read_bytes()
    spec = formal_source.TrainSourceSpec(
        source_split="TRAIN",
        design_sha256="4" * 64,
        annotation_qualification_sha256="5" * 64,
        wikipedia_qualification_sha256="6" * 64,
        annotation_basename=annotation.name,
        annotation_size_bytes=len(annotation_bytes),
        annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
        annotation_nonblank_rows=0,
        annotation_blank_sentinel_rows=1,
        database_basename=path.name,
        database_size_bytes=len(database_bytes),
        database_sha256=hashlib.sha256(database_bytes).hexdigest(),
        database_row_count=8,
        required_mode=path.stat().st_mode & 0o777,
    )
    source = formal_source.ControlledTrainSource(
        annotation_path=annotation,
        database_path=path,
        spec=spec,
    )
    with pytest.raises(
        formal_source.FeverousFormalSourceError,
        match="candidate screening",
    ):
        source.plan_corpus_identities_parallel_once(
            blocks={},
            secret=SECRET,
            identity_full_compile_equivalence_qualification_sha256="7" * 64,
        )
    source.exact_resolver_for_candidate_screen()
    database_body = {
        "schema": formal_source.DATABASE_RECEIPT_SCHEMA,
        "version": formal_source.VERSION,
        "status": "complete_database_page_stream_exhausted",
        "source_split": "TRAIN",
        "source_spec_sha256": spec.spec_sha256,
        "source_binding_sha256": formal_source.FROZEN_TRAIN_BINDING.binding_sha256,
        "formal_source_opener_source_sha256": source._formal_source_module_sha256,
        "formal_source": False,
        "database_basename": spec.database_basename,
        "database_size_bytes": spec.database_size_bytes,
        "database_file_sha256": spec.database_sha256,
        "database_schema_sha256": "8" * 64,
        "expected_database_row_count": 8,
        "observed_database_row_count": 8,
        "page_order": "synthetic_parallel_exact_cover",
        "logical_page_stream_sha256": "9" * 64,
        "stream_fully_exhausted": True,
        "maximum_buffered_database_rows": 4,
        "all_page_ids_or_pages_materialized": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    database_receipt = formal_source._self_hashed(
        database_body, "database_page_stream_receipt_sha256"
    )
    sentinel = object()
    observed: dict[str, object] = {}

    def delegate(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(
            plan=sentinel,
            database_receipt=database_receipt,
        )

    monkeypatch.setattr(
        formal_source.parallel_selection,
        "plan_fixed_corpus_parallel",
        delegate,
    )
    result = source.plan_corpus_identities_parallel_once(
        blocks={"A_form": ()},
        secret=SECRET,
        identity_full_compile_equivalence_qualification_sha256="7" * 64,
    )
    assert result is sentinel
    assert observed["database_path"] == path
    assert observed["secret"] == SECRET
    assert isinstance(observed["database_binding"], module.BoundDatabase)
    assert source.database_receipt == database_receipt
    with pytest.raises(formal_source.FeverousFormalSourceError, match="one-shot"):
        source.plan_corpus_identities_parallel_once(
            blocks={},
            secret=SECRET,
            identity_full_compile_equivalence_qualification_sha256="7" * 64,
        )
    source.close()


def _dense_page(page_id: str, count: int) -> dict[str, Any]:
    page: dict[str, Any] = {
        "title": page_id,
        "order": [f"sentence_{index}" for index in range(count)],
    }
    page.update(
        {
            f"sentence_{index}": f"Dense atomic target {page_id} {index}."
            for index in range(count)
        }
    )
    return page


def _dense_source(
    tmp_path: Path,
    *,
    page_count: int = 8,
    identities_per_page: int = 1100,
) -> formal_source.ControlledTrainSource:
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotation = tmp_path / "dense_train.jsonl"
    annotation.write_bytes(b"{}\n")
    annotation.chmod(0o600)
    database = tmp_path / "dense_wiki.db"
    connection = sqlite3.connect(database)
    connection.execute(formal_source.FROZEN_DATABASE_SCHEMA)
    pages = [
        _dense_page(f"Dense_Page_{index:02d}", identities_per_page)
        for index in range(page_count)
    ]
    connection.executemany(
        "INSERT INTO wiki (id, data) VALUES (?, ?)",
        [
            (
                page["title"],
                json.dumps(page, ensure_ascii=False, separators=(",", ":")),
            )
            for page in pages
        ],
    )
    connection.commit()
    connection.close()
    database.chmod(0o600)
    mode = database.stat().st_mode & 0o777
    spec = formal_source.TrainSourceSpec(
        source_split="TRAIN",
        design_sha256=source_adapter.DESIGN_SHA256,
        annotation_qualification_sha256=(
            source_adapter.ANNOTATION_QUALIFICATION_SHA256
        ),
        wikipedia_qualification_sha256=(
            source_adapter.WIKIPEDIA_QUALIFICATION_SHA256
        ),
        annotation_basename=annotation.name,
        annotation_size_bytes=annotation.stat().st_size,
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        annotation_nonblank_rows=0,
        annotation_blank_sentinel_rows=1,
        database_basename=database.name,
        database_size_bytes=database.stat().st_size,
        database_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
        database_row_count=page_count,
        required_mode=mode,
    )
    return formal_source.ControlledTrainSource(
        annotation_path=annotation,
        database_path=database,
        spec=spec,
    )


def _complete_blocks(
    first_page_id: str,
) -> dict[str, tuple[acquisition.AssignedRecord, ...]]:
    blocks: dict[str, tuple[acquisition.AssignedRecord, ...]] = {}
    cursor = 0
    for block in acquisition.BLOCK_ORDER:
        assigned: list[acquisition.AssignedRecord] = []
        for ordinal in range(acquisition.BLOCK_COUNTS[block]):
            keys = (
                f"{first_page_id}_sentence_{cursor}",
                f"{first_page_id}_sentence_{cursor + 1}",
            )
            record = acquisition.CandidateRecord(
                source_key=f"parallel-integration:{block}:{ordinal}",
                claim=f"Parallel integration claim {block} {ordinal}.",
                family=acquisition.FAMILIES[
                    ordinal % len(acquisition.FAMILIES)
                ],
                verdict=acquisition.VERDICTS[
                    ordinal % len(acquisition.VERDICTS)
                ],
                evidence_sets=(keys,),
                all_official_evidence_keys=keys,
            )
            assigned.append(
                acquisition.AssignedRecord(
                    block=block,
                    ordinal=ordinal,
                    record=record,
                    canonical_gold_keys=keys,
                )
            )
            cursor += 2
        blocks[block] = tuple(assigned)
    return blocks


def test_real_eight_worker_plan_to_selected_materialization_integration(
    tmp_path: Path,
) -> None:
    source = _dense_source(tmp_path)
    first_page = "Dense_Page_00"
    blocks = _complete_blocks(first_page)
    selected_rows = [
        row for block in acquisition.BLOCK_ORDER for row in blocks[block]
    ]
    assert len(selected_rows) == sum(acquisition.BLOCK_COUNTS.values()) == 288
    gold_keys = {
        key for row in selected_rows for key in row.canonical_gold_keys
    }
    assert len(gold_keys) == 576

    # This call hashes and opens the synthetic DB before the worker capability
    # becomes available, mirroring formal candidate screening custody.
    source.exact_resolver_for_candidate_screen()
    plan = source.plan_corpus_identities_parallel_once(
        blocks=blocks,
        secret=SECRET,
        identity_full_compile_equivalence_qualification_sha256="a" * 64,
    )
    assert len(plan.identities) == acquisition.CORPUS_UNIT_COUNT == 8192
    assert gold_keys.issubset({row.unit_key for row in plan.identities})
    assert plan.receipt["parallel_worker_count"] == 8
    assert plan.receipt["complete_identity_scan_count"] == 8800
    assert plan.receipt["formal_source_bound"] is False
    identity_receipt_sha = acquisition.verify_self_hash(
        plan.identity_stream_receipt,
        "corpus_identity_stream_receipt_sha256",
    )
    assert plan.receipt["identity_stream_receipt_sha256"] == identity_receipt_sha
    assert plan.identity_stream_receipt["parallel_worker_count"] == 8
    assert plan.identity_stream_receipt["formal_source"] is False
    assert (
        plan.identity_stream_receipt[
            "formal_secret_logged_persisted_or_exposed_on_argv"
        ]
        is False
    )
    assert (
        plan.identity_stream_receipt[
            "formal_secret_serialized_only_in_spawn_pipe_transit"
        ]
        is True
    )
    database_receipt_sha = formal_source.verify_database_page_stream_receipt(
        source.database_receipt
    )
    assert (
        plan.identity_stream_receipt["database_page_stream_receipt_sha256"]
        == database_receipt_sha
    )

    selected_units = source.iter_selected_corpus_units_once(plan)
    corpus, index, stats = acquisition.materialize_fixed_corpus_from_selection_plan(
        plan=plan,
        units=selected_units,
        secret=SECRET,
        require_formal_source=False,
    )
    assert len(corpus) == len(index) == acquisition.CORPUS_UNIT_COUNT
    assert gold_keys.issubset({row.unit_key for row in corpus})
    materialization = selected_units.aggregate_receipt()
    materialization_sha = (
        source_adapter.verify_selected_corpus_materialization_receipt(
            materialization
        )
    )
    assert materialization["corpus_identity_plan_sha256"] == plan.plan_sha256
    assert (
        materialization["database_page_stream_receipt_sha256"]
        == database_receipt_sha
    )
    assert (
        stats["selected_corpus_materialization_receipt_sha256"]
        == materialization_sha
    )
    assert stats["source_identity_stream_receipt_sha256"] == identity_receipt_sha
    assert stats["source_atomic_identity_scan_count"] == 8800
    assert stats["formal_source_bound"] is False
    assert stats["formal_acquisition_valid"] is False
    with pytest.raises(
        acquisition.FeverousP6E2AcquisitionError,
        match="not bound",
    ):
        acquisition.verify_formal_corpus_acquisition(stats)
    source.close()

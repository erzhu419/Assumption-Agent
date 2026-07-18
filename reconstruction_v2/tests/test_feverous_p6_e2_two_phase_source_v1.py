from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as formal
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as adapter


EQUIVALENCE_QUALIFICATION_SHA256 = "ab" * 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _page(page_id: str, count: int) -> dict[str, Any]:
    page: dict[str, Any] = {
        "title": page_id,
        "order": [f"sentence_{index}" for index in range(count)],
    }
    page.update(
        {
            f"sentence_{index}": f"Atomic target {index}."
            for index in range(count)
        }
    )
    return page


def _source(tmp_path: Path, pages: list[dict[str, Any]]) -> formal.ControlledTrainSource:
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotation = tmp_path / "synthetic_train.jsonl"
    annotation.write_bytes(b"{}\n")
    annotation.chmod(0o600)
    database = tmp_path / "synthetic_wiki.db"
    connection = sqlite3.connect(database)
    connection.execute(formal.FROZEN_DATABASE_SCHEMA)
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
    mode = annotation.stat().st_mode & 0o777
    spec = formal.TrainSourceSpec(
        source_split="TRAIN",
        design_sha256=adapter.DESIGN_SHA256,
        annotation_qualification_sha256=(
            adapter.ANNOTATION_QUALIFICATION_SHA256
        ),
        wikipedia_qualification_sha256=(
            adapter.WIKIPEDIA_QUALIFICATION_SHA256
        ),
        annotation_basename=annotation.name,
        annotation_size_bytes=annotation.stat().st_size,
        annotation_sha256=_sha256(annotation),
        annotation_nonblank_rows=0,
        annotation_blank_sentinel_rows=1,
        database_basename=database.name,
        database_size_bytes=database.stat().st_size,
        database_sha256=_sha256(database),
        database_row_count=len(pages),
        required_mode=mode,
    )
    return formal.ControlledTrainSource(
        annotation_path=annotation,
        database_path=database,
        spec=spec,
    )


def test_fast_identity_stream_accepts_physical_nonlexical_order_and_receipts_only_after_exhaustion(
    tmp_path: Path,
) -> None:
    # Insertion/rowid order is Z then A.  The formal scan must stay physical;
    # sorting the 53 GB source by the page-id index would be random I/O.
    source = _source(tmp_path, [_page("Page_Z", 2), _page("Page_A", 1)])
    identities = source.iter_corpus_identities_once(
        identity_full_compile_equivalence_qualification_sha256=(
            EQUIVALENCE_QUALIFICATION_SHA256
        )
    )
    first = next(identities)
    assert first.page == "Page_Z"
    with pytest.raises(adapter.FeverousSourceAdapterError, match="before normal exhaustion"):
        identities.aggregate_receipt()
    rows = [first, *identities]
    assert [row.page for row in rows] == ["Page_Z", "Page_Z", "Page_A"]
    receipt = identities.aggregate_receipt()
    assert receipt["adapted_page_count"] == 2
    assert receipt["eligible_atomic_identity_count"] == 3
    assert receipt["full_atomic_text_or_sidecar_linearized"] is False
    assert receipt["formal_source"] is False
    assert adapter.verify_corpus_identity_stream_receipt(receipt)

    forged = copy.deepcopy(dict(receipt))
    forged["eligible_atomic_identity_count"] = 4
    with pytest.raises(adapter.FeverousSourceAdapterError, match="drifted"):
        adapter.verify_corpus_identity_stream_receipt(forged)
    source.close()


class _DuplicatePageStream:
    def __init__(self, page: dict[str, Any]) -> None:
        payload = json.dumps(page, ensure_ascii=False, separators=(",", ":"))
        self._rows = iter([(page["title"], payload), (page["title"], payload)])
        self._complete = False

    def __iter__(self) -> "_DuplicatePageStream":
        return self

    def __next__(self) -> tuple[str, str]:
        try:
            return next(self._rows)
        except StopIteration:
            self._complete = True
            raise

    def aggregate_receipt(self) -> dict[str, Any]:
        if not self._complete:
            raise RuntimeError("partial")
        body = {
            "schema": formal.DATABASE_RECEIPT_SCHEMA,
            "version": formal.VERSION,
            "status": "complete_database_page_stream_exhausted",
            "source_split": "TRAIN",
            "formal_source": False,
            "database_file_sha256": "11" * 32,
            "logical_page_stream_sha256": "22" * 32,
            "expected_database_row_count": 2,
            "observed_database_row_count": 2,
            "stream_fully_exhausted": True,
        }
        body["database_page_stream_receipt_sha256"] = adapter._stable_hash(body)
        return body


def test_fast_identity_stream_rejects_duplicate_page_without_large_seen_set() -> None:
    source = _DuplicatePageStream(_page("Duplicate_Page", 1))
    identities = adapter.iter_qualified_corpus_identities(
        source,
        binding=adapter.FROZEN_TRAIN_BINDING,
        identity_full_compile_equivalence_qualification_sha256=(
            EQUIVALENCE_QUALIFICATION_SHA256
        ),
    )
    assert next(identities).page == "Duplicate_Page"
    with pytest.raises(adapter.FeverousSourceAdapterError, match="adjacent duplicate"):
        next(identities)
    with pytest.raises(adapter.FeverousSourceAdapterError, match="before normal exhaustion"):
        identities.aggregate_receipt()


def _blocks(page_id: str) -> dict[str, tuple[acquisition.AssignedRecord, ...]]:
    blocks: dict[str, tuple[acquisition.AssignedRecord, ...]] = {}
    cursor = 0
    for block in acquisition.BLOCK_ORDER:
        assigned: list[acquisition.AssignedRecord] = []
        for ordinal in range(acquisition.BLOCK_COUNTS[block]):
            keys = (
                f"{page_id}_sentence_{cursor}",
                f"{page_id}_sentence_{cursor + 1}",
            )
            record = acquisition.CandidateRecord(
                source_key=f"fixture:{block}:{ordinal}",
                claim=f"Fixture claim {block} {ordinal}.",
                family=acquisition.FAMILIES[ordinal % len(acquisition.FAMILIES)],
                verdict=acquisition.VERDICTS[ordinal % len(acquisition.VERDICTS)],
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


def test_two_phase_plan_then_selected_only_compile_binds_receipts_but_fixture_is_not_formal(
    tmp_path: Path,
) -> None:
    page_id = "Large_Selected_Page"
    source = _source(tmp_path, [_page(page_id, 9000)])
    identity_stream = source.iter_corpus_identities_once(
        identity_full_compile_equivalence_qualification_sha256=(
            EQUIVALENCE_QUALIFICATION_SHA256
        )
    )
    plan = acquisition.plan_fixed_corpus_from_identity_stream(
        blocks=_blocks(page_id),
        identities=identity_stream,
        secret=b"s" * 32,
        require_formal_source=False,
    )
    assert len(plan.identities) == acquisition.CORPUS_UNIT_COUNT
    assert plan.selected_page_ids == (page_id,)
    assert plan.receipt["complete_identity_scan_count"] == 9000
    assert plan.receipt["maximum_retained_distractor_identities"] == 8192
    assert plan.receipt["full_atomic_text_or_sidecar_linearized_during_scan"] is False

    selected_units = source.iter_selected_corpus_units_once(plan)
    corpus, index, stats = acquisition.materialize_fixed_corpus_from_selection_plan(
        plan=plan,
        units=selected_units,
        secret=b"s" * 32,
        require_formal_source=False,
    )
    assert len(corpus) == len(index) == acquisition.CORPUS_UNIT_COUNT
    assert stats["source_atomic_identity_scan_count"] == 9000
    assert stats["selected_pages_only_full_compiled"] is True
    assert stats["full_universe_linearized_or_sidecars_built"] is False
    assert stats["formal_source_bound"] is False
    assert adapter.verify_selected_corpus_materialization_receipt(
        selected_units.aggregate_receipt()
    )
    with pytest.raises(acquisition.FeverousP6E2AcquisitionError, match="not bound"):
        acquisition.verify_formal_corpus_acquisition(stats)
    source.close()

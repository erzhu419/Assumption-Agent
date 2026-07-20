from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    beir_bridge_expansion_acquisition_v1 as acquisition,
)


def _jsonl(rows):
    return b"".join(
        json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in rows
    )


def _parsed(count: int = 120) -> acquisition.ParsedFamily:
    documents = {
        f"doc-{index}": {"title": f"title {index}", "text": f"text {index}"}
        for index in range(count + 5)
    }
    queries = {f"query-{index}": f"query text {index}" for index in range(count)}
    qrels = {
        f"query-{index}": (f"doc-{index}",)
        for index in range(count)
    }
    return acquisition.ParsedFamily(
        documents=documents,
        queries=queries,
        positive_qrels=qrels,
        member_bindings={},
        self_document_query_count=0,
    )


def test_parse_corpus_and_queries() -> None:
    corpus = acquisition.parse_corpus(
        _jsonl(
            [
                {"_id": "d1", "title": "Title", "text": "Body"},
                {"_id": "d2", "title": "", "text": "Second"},
            ]
        )
    )
    queries = acquisition.parse_queries(
        _jsonl([{"_id": "q1", "text": "Question?"}])
    )
    assert tuple(corpus) == ("d1", "d2")
    assert queries == {"q1": "Question?"}


def test_parse_corpus_rejects_duplicate_id() -> None:
    raw = _jsonl(
        [
            {"_id": "d1", "title": "", "text": "one"},
            {"_id": "d1", "title": "", "text": "two"},
        ]
    )
    with pytest.raises(acquisition.BeirAcquisitionError):
        acquisition.parse_corpus(raw)


def test_parse_qrels_binarizes_positive_and_rejects_unknown() -> None:
    raw = b"query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\n"
    assert acquisition.parse_qrels(
        raw,
        query_ids=("q1",),
        document_ids=("d1", "d2"),
    ) == {"q1": ("d1",)}
    with pytest.raises(acquisition.BeirAcquisitionError):
        acquisition.parse_qrels(
            b"query-id\tcorpus-id\tscore\nq1\tmissing\t1\n",
            query_ids=("q1",),
            document_ids=("d1",),
        )


def test_allocate_blocks_is_deterministic_disjoint_and_exact() -> None:
    parsed = _parsed()
    secret = bytes(range(32))
    first = acquisition.allocate_blocks(
        family="NFCORPUS",
        parsed=parsed,
        secret=secret,
    )
    second = acquisition.allocate_blocks(
        family="NFCORPUS",
        parsed=parsed,
        secret=secret,
    )
    assert first == second
    assert {key: len(value) for key, value in first.items()} == dict(
        acquisition.BLOCK_SIZES
    )
    all_ids = [query_id for values in first.values() for query_id in values]
    assert len(all_ids) == acquisition.TOTAL_PER_FAMILY
    assert len(set(all_ids)) == len(all_ids)


def test_allocate_blocks_refuses_insufficient_capacity() -> None:
    with pytest.raises(acquisition.BeirAcquisitionError):
        acquisition.allocate_blocks(
            family="NFCORPUS",
            parsed=_parsed(acquisition.TOTAL_PER_FAMILY - 1),
            secret=b"x" * 32,
        )


def test_materialize_blocks_separates_qrels_from_views(tmp_path: Path) -> None:
    parsed = {family: _parsed() for family in acquisition.FAMILY_ORDER}
    allocations = {
        family: acquisition.allocate_blocks(
            family=family,
            parsed=parsed[family],
            secret=b"z" * 32,
        )
        for family in acquisition.FAMILY_ORDER
    }
    receipts = acquisition.materialize_blocks(
        parsed_by_family=parsed,
        allocation_by_family=allocations,
        secret=b"z" * 32,
        block_root=tmp_path,
    )
    assert receipts["C_confirm"]["item_count"] == 72
    view = (tmp_path / "C_confirm.view.jsonl").read_text()
    labels = (tmp_path / "C_confirm.labels.jsonl").read_text()
    assert "gold_document_ids" not in view
    assert "query text" not in labels


def _write_family_zip(path: Path, root: str, *, duplicate_corpus: bool = False) -> None:
    corpus = _jsonl([{"_id": "d1", "title": "Title", "text": "Body"}])
    queries = _jsonl([{"_id": "q1", "text": "Question"}])
    qrels = b"query-id\tcorpus-id\tscore\nq1\td1\t1\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/corpus.jsonl", corpus)
        if duplicate_corpus:
            archive.writestr(f"{root}/corpus.jsonl", corpus)
        archive.writestr(f"{root}/queries.jsonl", queries)
        archive.writestr(f"{root}/qrels/test.tsv", qrels)


def test_open_family_archive_reads_only_required_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "fixture.zip"
    _write_family_zip(archive_path, "nfcorpus")
    parsed = acquisition.open_family_archive(
        archive_path,
        family="NFCORPUS",
        extraction_root=tmp_path / "members",
    )
    assert tuple(parsed.documents) == ("d1",)
    assert tuple(parsed.queries) == ("q1",)
    assert parsed.positive_qrels == {"q1": ("d1",)}


def test_open_family_archive_rejects_duplicate_required_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "fixture.zip"
    _write_family_zip(archive_path, "nfcorpus", duplicate_corpus=True)
    with pytest.raises(acquisition.BeirAcquisitionError):
        acquisition.open_family_archive(
            archive_path,
            family="NFCORPUS",
            extraction_root=tmp_path / "members",
        )


def test_self_hash_contract() -> None:
    receipt = acquisition.self_hashed({"schema": "fixture", "status": "ok"}, "hash")
    declared = receipt["hash"]
    acquisition.verify_self_hash(receipt, "hash", declared)
    tampered = dict(receipt)
    tampered["status"] = "changed"
    with pytest.raises(acquisition.BeirAcquisitionError):
        acquisition.verify_self_hash(tampered, "hash", declared)

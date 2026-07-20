from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_dev_acquisition_v1 as acquisition,
)


def _jsonl(rows):
    return b"".join(acquisition.canonical_json(row) + b"\n" for row in rows)


def _parsed(count: int = 64) -> acquisition.ParsedDevSource:
    queries = {f"q-{index}": f"query {index}" for index in range(count)}
    positive = {f"q-{index}": (f"d-{index}",) for index in range(count)}
    return acquisition.ParsedDevSource(
        queries=queries,
        positive_qrels=positive,
        shared_query_count=count,
        dev_qrel_query_count=count,
        dev_qrel_row_count=count,
        positive_qrel_row_count=count,
        source_unretrievable_positive_qrel_row_count=0,
        nonpositive_qrel_row_count=0,
        self_document_positive_qrel_row_count=0,
    )


def test_filtered_corpus_parser_requires_canonical_unique_nonempty_rows() -> None:
    raw = _jsonl(
        [
            {"_id": "d1", "text": "body", "title": ""},
            {"_id": "d2", "text": "", "title": "title"},
        ]
    )
    assert acquisition.parse_filtered_corpus_ids(raw) == ("d1", "d2")
    with pytest.raises(acquisition.FiqaDevAcquisitionError):
        acquisition.parse_filtered_corpus_ids(raw + _jsonl([{"_id": "d1", "text": "x", "title": ""}]))


def test_dev_qrels_exclude_source_unretrievable_and_self_positive_rows() -> None:
    raw = (
        b"query-id\tcorpus-id\tscore\n"
        b"q1\td1\t2\n"
        b"q1\tmissing\t1\n"
        b"q1\tq1\t1\n"
        b"q2\td2\t0\n"
    )
    positive, query_ids, counts = acquisition.parse_dev_qrels(
        raw,
        query_ids=("q1", "q2"),
        usable_document_ids=("d1", "d2", "q1"),
    )
    assert positive == {"q1": ("d1",)}
    assert query_ids == {"q1", "q2"}
    assert counts == {
        "dev_qrel_row_count": 4,
        "positive_qrel_row_count": 3,
        "source_unretrievable_positive_qrel_row_count": 1,
        "nonpositive_qrel_row_count": 1,
        "self_document_positive_qrel_row_count": 1,
    }


def test_dev_qrels_reject_unknown_query_and_duplicate_pair() -> None:
    with pytest.raises(acquisition.FiqaDevAcquisitionError):
        acquisition.parse_dev_qrels(
            b"query-id\tcorpus-id\tscore\nunknown\td1\t1\n",
            query_ids=("q1",),
            usable_document_ids=("d1",),
        )
    with pytest.raises(acquisition.FiqaDevAcquisitionError):
        acquisition.parse_dev_qrels(
            b"query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t1\n",
            query_ids=("q1",),
            usable_document_ids=("d1",),
        )


def test_private_hmac_selection_is_exact_deterministic_and_disjoint() -> None:
    parsed = _parsed()
    secret = bytes(range(32))
    first = acquisition.select_dev_cohort(parsed, secret)
    second = acquisition.select_dev_cohort(parsed, secret)
    assert first == second
    assert len(first) == acquisition.COHORT_SIZE
    assert len(set(first)) == acquisition.COHORT_SIZE


def test_materialized_dev_pack_separates_views_and_labels(tmp_path: Path) -> None:
    parsed = _parsed()
    secret = b"z" * 32
    selected = acquisition.select_dev_cohort(parsed, secret)
    pack = acquisition.materialize_dev_pack(
        parsed=parsed,
        selected=selected,
        usable_document_ids=tuple(f"d-{index}" for index in range(64)),
        secret=secret,
        run_root=tmp_path,
    )
    assert pack["item_count"] == acquisition.COHORT_SIZE
    view = (tmp_path / "C_confirm.view.jsonl").read_text()
    labels = (tmp_path / "C_confirm.labels.jsonl").read_text()
    assert "gold_document_ids" not in view
    assert "query 0" not in labels


def _write_archive(path: Path, *, duplicate_dev: bool = False) -> None:
    queries = _jsonl([{"_id": "q1", "text": "question"}])
    dev = b"query-id\tcorpus-id\tscore\nq1\td1\t1\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fiqa/corpus.jsonl", b"not opened")
        archive.writestr("fiqa/queries.jsonl", queries)
        archive.writestr("fiqa/qrels/train.tsv", b"not opened")
        archive.writestr("fiqa/qrels/dev.tsv", dev)
        if duplicate_dev:
            archive.writestr("fiqa/qrels/dev.tsv", dev)
        archive.writestr("fiqa/qrels/test.tsv", b"not opened")


def test_archive_reader_opens_only_queries_and_dev_qrels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "fiqa.zip"
    _write_archive(archive_path)
    opened = []
    original = zipfile.ZipFile.read

    def tracking(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", tracking)
    raw, bindings = acquisition.read_dev_members(archive_path)
    assert set(raw) == {"queries", "dev_qrels"}
    assert set(bindings) == {"queries", "dev_qrels"}
    assert opened == ["fiqa/queries.jsonl", "fiqa/qrels/dev.tsv"]


def test_archive_reader_rejects_duplicate_required_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "fiqa.zip"
    _write_archive(archive_path, duplicate_dev=True)
    with pytest.raises(acquisition.FiqaDevAcquisitionError):
        acquisition.read_dev_members(archive_path)


def test_self_hash_contract() -> None:
    value = acquisition.self_hashed({"schema": "fixture", "status": "ok"}, "hash")
    acquisition.verify_self_hash(value, "hash", value["hash"])
    changed = dict(value)
    changed["status"] = "changed"
    with pytest.raises(acquisition.FiqaDevAcquisitionError):
        acquisition.verify_self_hash(changed, "hash", value["hash"])

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_integration_v1 as integration,
)


def _jsonl(rows):
    return b"".join(
        json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in rows
    )


def _fixture_source(count: int = 20):
    corpus = _jsonl(
        {"_id": f"d{index}", "title": f"T{index}", "text": f"Body {index}"}
        for index in range(count + 2)
    )
    queries = _jsonl(
        [
            {"_id": f"q{index}", "text": f"TRAIN query {index}"}
            for index in range(count)
        ]
        + [{"_id": "dev-only", "text": "must not materialize"}]
    )
    qrels = (
        "query-id\tcorpus-id\tscore\n"
        + "".join(f"q{index}\td{index}\t1\n" for index in range(count))
    ).encode("utf-8")
    return corpus, queries, qrels


def test_parse_train_source_materializes_only_train_queries() -> None:
    corpus, queries, qrels = _fixture_source()
    parsed = integration.parse_train_source(
        corpus_raw=corpus,
        queries_raw=queries,
        train_qrels_raw=qrels,
    )
    assert parsed.shared_query_count == 21
    assert "dev-only" not in parsed.queries
    assert len(parsed.positive_qrels) == 20


def test_unknown_document_positive_is_counted_and_excluded() -> None:
    corpus, queries, qrels = _fixture_source()
    qrels += b"q0\tmissing\t1\n"
    parsed = integration.parse_train_source(
        corpus_raw=corpus,
        queries_raw=queries,
        train_qrels_raw=qrels,
    )
    assert parsed.unknown_document_positive_qrel_row_count == 1
    assert parsed.positive_qrels["q0"] == ("d0",)


def test_unknown_query_qrel_fails_closed() -> None:
    corpus, queries, qrels = _fixture_source()
    qrels += b"missing-query\td0\t1\n"
    with pytest.raises(integration.FiqaTrainIntegrationError):
        integration.parse_train_source(
            corpus_raw=corpus,
            queries_raw=queries,
            train_qrels_raw=qrels,
        )


def test_duplicate_qrel_pair_fails_closed() -> None:
    corpus, queries, qrels = _fixture_source()
    qrels += b"q0\td0\t1\n"
    with pytest.raises(integration.FiqaTrainIntegrationError):
        integration.parse_train_source(
            corpus_raw=corpus,
            queries_raw=queries,
            train_qrels_raw=qrels,
        )


def test_nonpositive_and_self_rows_are_not_relevant() -> None:
    corpus = _jsonl(
        [{"_id": f"q{index}", "title": "T", "text": "B"} for index in range(20)]
        + [{"_id": f"d{index}", "title": "T", "text": "B"} for index in range(20)]
    )
    queries = _jsonl(
        {"_id": f"q{index}", "text": f"query {index}"} for index in range(20)
    )
    qrels = (
        "query-id\tcorpus-id\tscore\n"
        + "".join(
            f"q{index}\td{index}\t1\nq{index}\tq{index}\t1\nq{index}\td{(index + 1) % 20}\t0\n"
            for index in range(20)
        )
    ).encode()
    parsed = integration.parse_train_source(
        corpus_raw=corpus,
        queries_raw=queries,
        train_qrels_raw=qrels,
    )
    assert parsed.self_document_positive_qrel_row_count == 20
    assert parsed.nonpositive_qrel_row_count == 20
    assert all(len(value) == 1 for value in parsed.positive_qrels.values())


def test_selection_is_public_deterministic_exact_and_unique() -> None:
    corpus, queries, qrels = _fixture_source()
    parsed = integration.parse_train_source(
        corpus_raw=corpus,
        queries_raw=queries,
        train_qrels_raw=qrels,
    )
    first = integration.select_train_diagnostic(parsed)
    second = integration.select_train_diagnostic(parsed)
    assert first == second
    assert len(first) == integration.TRAIN_DIAGNOSTIC_SIZE
    assert len(set(first)) == len(first)


def test_materialized_pack_separates_query_and_gold(tmp_path: Path) -> None:
    corpus, queries, qrels = _fixture_source()
    parsed = integration.parse_train_source(
        corpus_raw=corpus,
        queries_raw=queries,
        train_qrels_raw=qrels,
    )
    selected = integration.select_train_diagnostic(parsed)
    receipt = integration.materialize_train_pack(
        parsed=parsed,
        selected=selected,
        run_root=tmp_path,
    )
    assert receipt["item_count"] == integration.TRAIN_DIAGNOSTIC_SIZE
    assert "gold_document_ids" not in (tmp_path / "train_integration.view.jsonl").read_text()
    assert "TRAIN query" not in (tmp_path / "train_integration.labels.jsonl").read_text()


def _write_archive(path: Path, *, duplicate_train: bool = False) -> None:
    corpus, queries, qrels = _fixture_source()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fiqa/corpus.jsonl", corpus)
        archive.writestr("fiqa/queries.jsonl", queries)
        archive.writestr("fiqa/qrels/train.tsv", qrels)
        if duplicate_train:
            archive.writestr("fiqa/qrels/train.tsv", qrels)
        archive.writestr("fiqa/qrels/dev.tsv", b"SECRET DEV")
        archive.writestr("fiqa/qrels/test.tsv", b"SECRET TEST")


def test_read_train_members_never_opens_dev_or_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "fiqa.zip"
    _write_archive(path)
    opened = []
    original = zipfile.ZipFile.read

    def tracking_read(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", tracking_read)
    raw, bindings = integration.read_train_members(path)
    assert set(raw) == {"corpus", "queries", "train_qrels"}
    assert set(bindings) == set(raw)
    assert opened == [
        "fiqa/corpus.jsonl",
        "fiqa/qrels/train.tsv",
        "fiqa/queries.jsonl",
    ]


def test_duplicate_required_member_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "fiqa.zip"
    _write_archive(path, duplicate_train=True)
    with pytest.raises(integration.FiqaTrainIntegrationError):
        integration.read_train_members(path)


def test_self_hash_contract() -> None:
    receipt = integration.self_hashed({"schema": "fixture", "status": "ok"}, "hash")
    integration.verify_self_hash(receipt, "hash", receipt["hash"])
    tampered = dict(receipt)
    tampered["status"] = "changed"
    with pytest.raises(integration.FiqaTrainIntegrationError):
        integration.verify_self_hash(tampered, "hash", receipt["hash"])

from __future__ import annotations

import json

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_integration_v2 as integration,
)


def _jsonl(rows):
    return b"".join(
        json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in rows
    )


def _fixture_source():
    corpus_rows = [
        {"_id": f"d{index}", "title": "", "text": f"body {index}"}
        for index in range(20)
    ] + [
        {"_id": "empty-only", "title": "", "text": ""},
        {"_id": "empty-shared", "title": "   ", "text": ""},
    ]
    query_rows = [
        {"_id": f"q{index}", "text": f"train query {index}"}
        for index in range(20)
    ] + [
        {"_id": "q-empty-only", "text": "empty only"},
        {"_id": "q-empty-shared", "text": "empty plus usable"},
        {"_id": "dev-only", "text": "not materialized"},
    ]
    qrels = (
        "query-id\tcorpus-id\tscore\n"
        + "".join(f"q{index}\td{index}\t1\n" for index in range(20))
        + "q-empty-only\tempty-only\t1\n"
        + "q-empty-shared\tempty-shared\t1\n"
        + "q-empty-shared\td0\t1\n"
    ).encode()
    return _jsonl(corpus_rows), _jsonl(query_rows), qrels


def test_empty_documents_are_removed_from_filtered_corpus() -> None:
    corpus, _, _ = _fixture_source()
    documents, empty_ids, filtered, source_count = integration.parse_corpus_v2(corpus)
    assert source_count == 22
    assert empty_ids == frozenset({"empty-only", "empty-shared"})
    assert len(documents) == 20
    assert b"empty-only" not in filtered
    assert len(filtered.decode().splitlines()) == 20


def test_empty_document_qrels_are_counted_and_excluded() -> None:
    corpus, queries, qrels = _fixture_source()
    parsed = integration.parse_train_source_v2(
        corpus_raw=corpus,
        queries_raw=queries,
        train_qrels_raw=qrels,
    )
    assert parsed.empty_document_count == 2
    assert parsed.empty_document_positive_qrel_row_count == 2
    assert "q-empty-only" not in parsed.positive_qrels
    assert parsed.positive_qrels["q-empty-shared"] == ("d0",)
    assert "dev-only" not in parsed.queries


def test_missing_document_is_distinct_from_empty_document() -> None:
    corpus, queries, qrels = _fixture_source()
    qrels += b"q0\tmissing\t1\n"
    parsed = integration.parse_train_source_v2(
        corpus_raw=corpus,
        queries_raw=queries,
        train_qrels_raw=qrels,
    )
    assert parsed.unknown_document_positive_qrel_row_count == 1
    assert parsed.empty_document_positive_qrel_row_count == 2


def test_duplicate_document_id_still_fails_closed() -> None:
    raw = _jsonl(
        [
            {"_id": "same", "title": "", "text": "body"},
            {"_id": "same", "title": "", "text": ""},
        ]
    )
    with pytest.raises(integration.FiqaTrainIntegrationV2Error):
        integration.parse_corpus_v2(raw)


def test_public_train_selection_remains_exact_and_deterministic() -> None:
    corpus, queries, qrels = _fixture_source()
    parsed = integration.parse_train_source_v2(
        corpus_raw=corpus,
        queries_raw=queries,
        train_qrels_raw=qrels,
    )
    first = integration.v1.select_train_diagnostic(parsed)
    second = integration.v1.select_train_diagnostic(parsed)
    assert first == second
    assert len(first) == integration.v1.TRAIN_DIAGNOSTIC_SIZE


def test_filtered_corpus_is_canonical_jsonl() -> None:
    corpus, _, _ = _fixture_source()
    _, _, filtered, _ = integration.parse_corpus_v2(corpus)
    for line in filtered.splitlines(keepends=True):
        value = json.loads(line)
        assert line == integration.v1.canonical_json(value) + b"\n"

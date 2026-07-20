from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_runtime_v1 as runtime,
)


def _scores(size: int, top_rows):
    values = np.arange(size, dtype=np.int64) * -1 - 10_000
    for rank, row in enumerate(top_rows):
        values[row] = 10_000 - rank
    return values


def _fixture_documents(size: int = 400):
    ids = tuple(f"d{index}" for index in range(size))
    contents = tuple(
        f"Document {index}. Alpha mechanism relates Finance Entity{index} to market outcome."
        for index in range(size)
    )
    return ids, contents


def test_build_local_plan_emits_typed_bridge_queries() -> None:
    ids, contents = _fixture_documents()
    item = runtime.ViewItem(
        ordinal=0,
        item_key="a" * 64,
        query="How does leverage affect returns?",
        excluded_ids=(),
    )
    vectors = [
        _scores(len(ids), range(offset, offset + 64)) for offset in (0, 64, 128, 192, 256)
    ]
    expansions = (
        "leverage returns entities",
        "leverage relation returns",
        "leverage mechanism returns",
        "leverage returns constraints",
    )
    plan = runtime.build_local_plan(
        item=item,
        document_ids=ids,
        document_contents=contents,
        query_score_vectors=vectors,
        expansions=expansions,
    )
    assert len(plan.base_pool) == runtime.core.POOL_SIZE
    assert 1 <= len(plan.seed_rows) <= 4
    assert len(plan.bridge_queries) <= 4
    assert all(query.query_kind in {"relation_query", "mechanism_query"} for query in plan.bridge_queries)


def test_expand_plan_adds_full_corpus_candidates() -> None:
    ids, contents = _fixture_documents()
    item = runtime.ViewItem(0, "b" * 64, "financial query", ())
    expansions = ("entity q", "relation q", "mechanism q", "constraint q")
    vectors = [_scores(len(ids), range(offset, offset + 64)) for offset in (0, 64, 128, 192, 256)]
    plan = runtime.build_local_plan(
        item=item,
        document_ids=ids,
        document_contents=contents,
        query_score_vectors=vectors,
        expansions=expansions,
    )
    bridge_vectors = [
        _scores(len(ids), range(300, 364)) for _ in plan.bridge_queries
    ]
    expanded = runtime.expand_plan(plan, bridge_vectors)
    assert set(plan.base_pool) <= set(expanded.expanded.expanded_pool)
    assert len(expanded.expanded.expanded_pool) >= len(plan.base_pool)


def test_cross_input_preserves_expanded_pool_order() -> None:
    ids, contents = _fixture_documents()
    item = runtime.ViewItem(0, "c" * 64, "financial query", ())
    expansions = ("entity q", "relation q", "mechanism q", "constraint q")
    vectors = [_scores(len(ids), range(offset, offset + 64)) for offset in (0, 64, 128, 192, 256)]
    local = runtime.build_local_plan(
        item=item,
        document_ids=ids,
        document_contents=contents,
        query_score_vectors=vectors,
        expansions=expansions,
    )
    expanded = runtime.expand_plan(local, []) if not local.bridge_queries else runtime.expand_plan(
        local,
        [_scores(len(ids), range(300, 364)) for _ in local.bridge_queries],
    )
    payload = runtime.build_cross_input([expanded], contents)
    parsed = runtime.cross_contract.validate_items(payload["items"])
    assert len(parsed[0].documents) == len(expanded.expanded.expanded_pool)
    for ordinal, row in enumerate(expanded.expanded.expanded_pool):
        assert parsed[0].documents[ordinal].content == contents[row]


def test_local_plan_rejects_missing_typed_query() -> None:
    ids, contents = _fixture_documents()
    item = runtime.ViewItem(0, "d" * 64, "query", ())
    with pytest.raises(runtime.FiqaTrainRuntimeError):
        runtime.build_local_plan(
            item=item,
            document_ids=ids,
            document_contents=contents,
            query_score_vectors=[_scores(len(ids), range(64))] * 5,
            expansions=("one", "two", "three"),
        )


def test_load_filtered_corpus_uses_title_newline_text(tmp_path: Path) -> None:
    relative = Path("source.jsonl")
    rows = [
        {"_id": "d1", "title": " Title ", "text": " Body "},
        {"_id": "d2", "title": "", "text": "Second"},
    ]
    raw = b"".join(runtime.integration_v1.canonical_json(row) + b"\n" for row in rows)
    (tmp_path / relative).write_bytes(raw)
    integration = {
        "filtered_corpus_binding": {
            "relative_path": relative.as_posix(),
            "sha256": runtime.integration_v1.file_sha256(tmp_path / relative),
            "size_bytes": len(raw),
        },
        "source_aggregates": {"usable_corpus_document_count": 2},
    }
    ids, contents = runtime.load_filtered_corpus(tmp_path, integration)
    assert ids == ("d1", "d2")
    assert contents == ("Title\nBody", "Second")


def test_paired_descriptive_counts() -> None:
    assert runtime._paired([3, 2, 1], [1, 2, 4]) == {
        "gain": 1,
        "harm": 1,
        "net_integer_ndcg": -1,
        "tie": 1,
    }

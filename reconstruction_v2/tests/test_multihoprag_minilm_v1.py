from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from replication_runtime.multihoprag_minilm_v1 import (
    ArticleText,
    FROZEN_MINILM_RUNTIME_RECEIPT_SHA256,
    MultiHopRAGMiniLMError,
    build_corpus_embedding_index,
    compile_query_features,
    frozen_minilm_runtime_identity,
    recompute_embedding_index_sha256,
    recompute_query_feature_sha256,
    reciprocal_topic_neighbors,
    serialize_article_chunks,
    validate_corpus_embedding_index,
    validate_query_features,
)


class FakeEncoder:
    def __init__(self, *, wrong_receipt: bool = False):
        identity = frozen_minilm_runtime_identity()
        self.runtime_receipt = {
            "asset_file_sha256": identity["asset_file_sha256"],
            "asset_manifest_path": "/synthetic/fixed/asset.json",
            "asset_sha256": identity["asset_sha256"],
            "embedding_dimension": identity["embedding_dimension"],
            "maximum_sequence_length": identity["maximum_sequence_length"],
            "model_root": "/synthetic/fixed/model",
            "model_tree_sha256": identity["model_tree_sha256"],
            "runtime_versions": identity["runtime_versions"],
            "status": identity["status"],
            "weights_sha256": identity["weights_sha256"],
        }
        self.canary_receipt = {
            "float32_bytes_sha256": identity["canary_float32_bytes_sha256"],
            "quantized_embedding_matrix_sha256": identity[
                "canary_quantized_embedding_sha256"
            ],
            "qasper_rows_or_archives_accessed_by_canary": False,
            "repeat_count": 2,
            "repeat_exact": True,
            "sentence_count": identity["canary_sentence_count"],
            "status": "passed_exact_row_free_synthetic_canary",
            "text_vector_sha256": identity["canary_text_vector_sha256"],
        }
        if wrong_receipt:
            self.runtime_receipt["weights_sha256"] = "0" * 64

    def encode(self, texts):
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            first = int.from_bytes(digest[:2], "big") % 384
            second = int.from_bytes(digest[2:4], "big") % 384
            matrix[row, first] += 1.0
            matrix[row, second] += 0.5
            matrix[row] /= np.linalg.norm(matrix[row])
        return matrix


def _articles():
    return tuple(
        ArticleText(index, f"Title {index}", " ".join(f"token_{index}_{j}" for j in range(190)))
        for index in range(6)
    )


def test_chunking_is_nfkc_exact_window_stride_and_no_extra_terminal_chunk() -> None:
    full_width = "Ｔｉｔｌｅ"
    exactly = serialize_article_chunks(full_width, " ".join(f"w{i}" for i in range(160)))
    assert len(exactly) == 1
    assert exactly[0].startswith("Title\n\n")
    overlap = serialize_article_chunks("Title", " ".join(f"w{i}" for i in range(161)))
    assert len(overlap) == 2
    assert overlap[1].split("\n\n", 1)[1].split()[0] == "w128"
    assert len(overlap[1].split("\n\n", 1)[1].split()) == 33
    assert serialize_article_chunks("Title", "  \n ") == ("Title\n\n",)


def test_embedding_index_and_reciprocal_topic_graph_are_deterministic() -> None:
    first = build_corpus_embedding_index(articles=_articles(), encoder=FakeEncoder())
    second = build_corpus_embedding_index(articles=_articles(), encoder=FakeEncoder())
    assert first.index_sha256 == second.index_sha256
    assert first.encoder_receipt_sha256 == FROZEN_MINILM_RUNTIME_RECEIPT_SHA256
    assert recompute_embedding_index_sha256(first) == first.index_sha256
    assert validate_corpus_embedding_index(first) is first
    assert first.chunk_vectors.shape == (12, 384)
    assert not first.chunk_vectors.flags.writeable
    neighbors = reciprocal_topic_neighbors(first)
    assert len(neighbors) == 6
    assert all(len(values) <= 4 for values in neighbors)
    for left, values in enumerate(neighbors):
        assert all(left in neighbors[right] for right in values)


def test_index_receipt_detects_embedding_mutation() -> None:
    index = build_corpus_embedding_index(articles=_articles(), encoder=FakeEncoder())
    index.chunk_vectors.setflags(write=True)
    index.chunk_vectors[0, 0] += np.float32(0.01)
    with pytest.raises(MultiHopRAGMiniLMError, match="normalized|content"):
        reciprocal_topic_neighbors(index)


def test_query_feature_compilation_is_label_free_deterministic_and_all_corpus() -> None:
    index = build_corpus_embedding_index(articles=_articles(), encoder=FakeEncoder())
    first = compile_query_features(query="Compare Ａ and B", index=index, encoder=FakeEncoder())
    second = compile_query_features(query="Compare A and B", index=index, encoder=FakeEncoder())
    assert first == second
    assert first.embedding_index_sha256 == index.index_sha256
    assert len(first.dense_relevance_ints) == 6
    assert len(first.capability_similarity_ints) == 3
    assert first.predicted_capability in {
        "comparison_query",
        "inference_query",
        "temporal_query",
    }
    assert len(first.feature_sha256) == 64
    assert recompute_query_feature_sha256(first, index=index) == first.feature_sha256
    assert validate_query_features(first, index=index) is first


def test_feature_tamper_and_wrong_index_fail_closed() -> None:
    index = build_corpus_embedding_index(articles=_articles(), encoder=FakeEncoder())
    features = compile_query_features(
        query="Compare A and B", index=index, encoder=FakeEncoder()
    )
    tampered = replace(
        features,
        dense_relevance_ints=(features.dense_relevance_ints[0] + 1,)
        + features.dense_relevance_ints[1:],
    )
    with pytest.raises(MultiHopRAGMiniLMError, match="content drifted"):
        validate_query_features(tampered, index=index)

    changed_articles = list(_articles())
    row = changed_articles[0]
    changed_articles[0] = ArticleText(row.article_i, row.title, row.body + " changed")
    other_index = build_corpus_embedding_index(
        articles=changed_articles, encoder=FakeEncoder()
    )
    with pytest.raises(MultiHopRAGMiniLMError, match="index binding drifted"):
        validate_query_features(features, index=other_index)


def test_wrong_encoder_receipt_and_index_receipt_fail_closed() -> None:
    with pytest.raises(MultiHopRAGMiniLMError, match="not the frozen MiniLM runtime"):
        build_corpus_embedding_index(
            articles=_articles(), encoder=FakeEncoder(wrong_receipt=True)
        )
    index = build_corpus_embedding_index(articles=_articles(), encoder=FakeEncoder())
    with pytest.raises(MultiHopRAGMiniLMError, match="not the frozen MiniLM runtime"):
        compile_query_features(
            query="Compare A and B",
            index=index,
            encoder=FakeEncoder(wrong_receipt=True),
        )
    wrong_index = replace(index, encoder_receipt_sha256="0" * 64)
    with pytest.raises(MultiHopRAGMiniLMError, match="receipt|topology"):
        validate_corpus_embedding_index(wrong_index)


def test_article_ids_must_be_contiguous() -> None:
    rows = list(_articles())
    rows[2] = ArticleText(7, rows[2].title, rows[2].body)
    with pytest.raises(MultiHopRAGMiniLMError, match="contiguous"):
        build_corpus_embedding_index(articles=rows, encoder=FakeEncoder())
    rows = list(_articles())
    rows[1] = ArticleText(True, rows[1].title, rows[1].body)
    with pytest.raises(MultiHopRAGMiniLMError, match="contiguous"):
        build_corpus_embedding_index(articles=rows, encoder=FakeEncoder())

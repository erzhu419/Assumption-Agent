"""Row-agnostic MiniLM chunk, dense-score, topic, and capability algebra.

The model loader and its complete offline asset verification remain the
already-frozen :mod:`replication_runtime.qasper_minilm_v1.binding` trust root.
This module fixes only the MultiHopRAG serialization and aggregation layer.
It never accepts an answer, evidence fact, evidence URL, or question type.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Protocol, Sequence
import unicodedata

import numpy as np

from replication_runtime.qasper_minilm_v1.binding import (
    ASSET_FILE_SHA256,
    ASSET_SELF_SHA256,
    CANARY_FLOAT32_BYTES_SHA256,
    CANARY_QUANTIZED_EMBEDDING_SHA256,
    CANARY_SENTENCE_COUNT,
    CANARY_TEXT_VECTOR_SHA256,
    EMBEDDING_DIMENSION,
    EXPECTED_RUNTIME_VERSIONS,
    MAXIMUM_SEQUENCE_LENGTH,
    MODEL_TREE_SHA256,
    QUANTIZATION_SCALE,
    WEIGHTS_SHA256,
    quantized_cosine_similarity,
)


VERSION = "multihoprag_minilm_v1"
BODY_WINDOW_TOKENS = 160
BODY_WINDOW_STRIDE = 128
TOPIC_K = 4
CAPABILITY_ORDER = ("comparison_query", "inference_query", "temporal_query")
CAPABILITY_PROTOTYPES = {
    "comparison_query": (
        "Retrieve documents that compare multiple named entities, organizations, "
        "sources, or values."
    ),
    "inference_query": (
        "Retrieve multiple documents whose facts must be joined through a shared "
        "entity or topic to identify or infer an answer."
    ),
    "temporal_query": (
        "Retrieve documents needed to order events, dates, or changes over time."
    ),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MultiHopRAGMiniLMError(ValueError):
    """Raised when the frozen feature contract drifts."""


class Encoder(Protocol):
    runtime_receipt: Mapping[str, object]
    canary_receipt: Mapping[str, object]

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class ArticleText:
    article_i: int
    title: str
    body: str


@dataclass(frozen=True)
class CorpusEmbeddingIndex:
    article_count: int
    encoder_receipt_sha256: str
    article_chunk_ranges: tuple[tuple[int, int], ...]
    chunk_vectors: np.ndarray
    topic_vectors: np.ndarray
    normalized_article_sha256s: tuple[str, ...]
    index_sha256: str


@dataclass(frozen=True)
class QueryFeatures:
    embedding_index_sha256: str
    normalized_query_sha256: str
    capability_similarity_ints: tuple[int, int, int]
    predicted_capability: str
    dense_relevance_ints: tuple[int, ...]
    feature_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def frozen_minilm_runtime_identity() -> dict[str, object]:
    """Return the path-free identity of the already-frozen Qasper runtime."""

    return {
        "asset_file_sha256": ASSET_FILE_SHA256,
        "asset_sha256": ASSET_SELF_SHA256,
        "canary_float32_bytes_sha256": CANARY_FLOAT32_BYTES_SHA256,
        "canary_quantized_embedding_sha256": CANARY_QUANTIZED_EMBEDDING_SHA256,
        "canary_sentence_count": CANARY_SENTENCE_COUNT,
        "canary_text_vector_sha256": CANARY_TEXT_VECTOR_SHA256,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "maximum_sequence_length": MAXIMUM_SEQUENCE_LENGTH,
        "model_tree_sha256": MODEL_TREE_SHA256,
        "runtime_versions": dict(EXPECTED_RUNTIME_VERSIONS),
        "status": "verified_offline_immutable_qasper_minilm_runtime",
        "weights_sha256": WEIGHTS_SHA256,
    }


FROZEN_MINILM_RUNTIME_RECEIPT_SHA256 = _stable_hash(
    frozen_minilm_runtime_identity()
)


def _required_receipt_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MultiHopRAGMiniLMError(f"{field} is missing")
    return value


def encoder_receipt_sha256(encoder: Encoder) -> str:
    """Validate and hash the encoder's fixed asset, runtime, and canary receipt."""

    runtime = _required_receipt_mapping(
        getattr(encoder, "runtime_receipt", None), field="encoder runtime receipt"
    )
    canary = _required_receipt_mapping(
        getattr(encoder, "canary_receipt", None), field="encoder canary receipt"
    )
    expected_runtime_keys = {
        "asset_file_sha256",
        "asset_manifest_path",
        "asset_sha256",
        "embedding_dimension",
        "maximum_sequence_length",
        "model_root",
        "model_tree_sha256",
        "runtime_versions",
        "status",
        "weights_sha256",
    }
    expected_canary = {
        "float32_bytes_sha256": CANARY_FLOAT32_BYTES_SHA256,
        "quantized_embedding_matrix_sha256": CANARY_QUANTIZED_EMBEDDING_SHA256,
        "qasper_rows_or_archives_accessed_by_canary": False,
        "repeat_count": 2,
        "repeat_exact": True,
        "sentence_count": CANARY_SENTENCE_COUNT,
        "status": "passed_exact_row_free_synthetic_canary",
        "text_vector_sha256": CANARY_TEXT_VECTOR_SHA256,
    }
    if (
        set(runtime) != expected_runtime_keys
        or runtime.get("asset_file_sha256") != ASSET_FILE_SHA256
        or runtime.get("asset_sha256") != ASSET_SELF_SHA256
        or runtime.get("embedding_dimension") != EMBEDDING_DIMENSION
        or runtime.get("maximum_sequence_length") != MAXIMUM_SEQUENCE_LENGTH
        or runtime.get("model_tree_sha256") != MODEL_TREE_SHA256
        or runtime.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS
        or runtime.get("status")
        != "verified_offline_immutable_qasper_minilm_runtime"
        or runtime.get("weights_sha256") != WEIGHTS_SHA256
        or not isinstance(runtime.get("asset_manifest_path"), str)
        or not runtime.get("asset_manifest_path")
        or "\x00" in str(runtime.get("asset_manifest_path"))
        or not isinstance(runtime.get("model_root"), str)
        or not runtime.get("model_root")
        or "\x00" in str(runtime.get("model_root"))
        or dict(canary) != expected_canary
    ):
        raise MultiHopRAGMiniLMError("encoder receipt is not the frozen MiniLM runtime")
    receipt = _stable_hash(frozen_minilm_runtime_identity())
    if receipt != FROZEN_MINILM_RUNTIME_RECEIPT_SHA256:
        raise MultiHopRAGMiniLMError("frozen MiniLM trust root drifted")
    return receipt


def _array_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(matrix.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def canonical_text(value: str, *, field: str = "text") -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise MultiHopRAGMiniLMError(f"{field} is invalid")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized:
        raise MultiHopRAGMiniLMError(f"{field} is empty")
    return normalized


def serialize_article_chunks(title: str, body: str) -> tuple[str, ...]:
    normalized_title = canonical_text(title, field="title")
    if not isinstance(body, str) or "\x00" in body:
        raise MultiHopRAGMiniLMError("body is invalid")
    normalized_body = " ".join(unicodedata.normalize("NFKC", body).split())
    tokens = normalized_body.split() if normalized_body else []
    if not tokens:
        return (normalized_title + "\n\n",)
    chunks: list[str] = []
    start = 0
    while True:
        window = tokens[start : start + BODY_WINDOW_TOKENS]
        chunks.append(normalized_title + "\n\n" + " ".join(window))
        if start + BODY_WINDOW_TOKENS >= len(tokens):
            break
        start += BODY_WINDOW_STRIDE
    return tuple(chunks)


def _validated_articles(articles: Sequence[ArticleText]) -> tuple[ArticleText, ...]:
    if isinstance(articles, (str, bytes)) or not isinstance(articles, Sequence):
        raise MultiHopRAGMiniLMError("articles must be a sequence")
    rows = tuple(articles)
    if not rows:
        raise MultiHopRAGMiniLMError("articles are empty")
    for position, row in enumerate(rows):
        if (
            not isinstance(row, ArticleText)
            or isinstance(row.article_i, bool)
            or not isinstance(row.article_i, int)
            or row.article_i != position
        ):
            raise MultiHopRAGMiniLMError("article IDs must be contiguous corpus order")
        canonical_text(row.title, field="title")
        if not isinstance(row.body, str) or "\x00" in row.body:
            raise MultiHopRAGMiniLMError("body is invalid")
    return rows


def _validated_matrix(matrix: np.ndarray, *, rows: int, field: str) -> np.ndarray:
    if not isinstance(matrix, np.ndarray) or matrix.shape != (rows, EMBEDDING_DIMENSION):
        raise MultiHopRAGMiniLMError(f"{field} shape drifted")
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise MultiHopRAGMiniLMError(f"{field} dtype or finiteness drifted")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
        raise MultiHopRAGMiniLMError(f"{field} is not L2 normalized")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _encode_batches(encoder: Encoder, texts: Sequence[str], *, batch: int = 4096) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for start in range(0, len(texts), batch):
        values = tuple(texts[start : start + batch])
        matrix = encoder.encode(values)
        matrices.append(_validated_matrix(matrix, rows=len(values), field="encoder output"))
    return np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)


def _mean_topic_vector(matrix: np.ndarray) -> np.ndarray:
    values = [
        math.fsum(float(matrix[row, column]) for row in range(matrix.shape[0]))
        / matrix.shape[0]
        for column in range(EMBEDDING_DIMENSION)
    ]
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0:
        raise MultiHopRAGMiniLMError("article mean embedding has zero norm")
    return np.asarray([value / norm for value in values], dtype=np.float32)


def build_corpus_embedding_index(
    *, articles: Sequence[ArticleText], encoder: Encoder
) -> CorpusEmbeddingIndex:
    encoder_hash = encoder_receipt_sha256(encoder)
    rows = _validated_articles(articles)
    chunks: list[str] = []
    ranges: list[tuple[int, int]] = []
    article_hashes: list[str] = []
    for row in rows:
        serialized = serialize_article_chunks(row.title, row.body)
        start = len(chunks)
        chunks.extend(serialized)
        ranges.append((start, len(chunks)))
        article_hashes.append(
            _stable_hash(
                {
                    "article_i": row.article_i,
                    "chunks": list(serialized),
                    "version": VERSION,
                }
            )
        )
    chunk_matrix = _encode_batches(encoder, chunks)
    topics = np.stack(
        [_mean_topic_vector(chunk_matrix[start:end]) for start, end in ranges], axis=0
    ).astype(np.float32, copy=False)
    topics = _validated_matrix(topics, rows=len(rows), field="topic vectors")
    body = {
        "article_chunk_ranges": [list(value) for value in ranges],
        "article_count": len(rows),
        "article_sha256s": article_hashes,
        "chunk_matrix_sha256": _array_sha256(chunk_matrix),
        "chunk_shape": list(chunk_matrix.shape),
        "encoder_receipt_sha256": encoder_hash,
        "topic_matrix_sha256": _array_sha256(topics),
        "topic_shape": list(topics.shape),
        "version": VERSION,
    }
    chunk_matrix.setflags(write=False)
    topics.setflags(write=False)
    return CorpusEmbeddingIndex(
        article_count=len(rows),
        encoder_receipt_sha256=encoder_hash,
        article_chunk_ranges=tuple(ranges),
        chunk_vectors=chunk_matrix,
        topic_vectors=topics,
        normalized_article_sha256s=tuple(article_hashes),
        index_sha256=_stable_hash(body),
    )


def _canonical_index_body(index: CorpusEmbeddingIndex) -> dict[str, object]:
    if not isinstance(index, CorpusEmbeddingIndex):
        raise MultiHopRAGMiniLMError("embedding index receipt is invalid")
    if (
        isinstance(index.article_count, bool)
        or not isinstance(index.article_count, int)
        or index.article_count <= 0
        or index.encoder_receipt_sha256 != FROZEN_MINILM_RUNTIME_RECEIPT_SHA256
        or not isinstance(index.article_chunk_ranges, tuple)
        or len(index.article_chunk_ranges) != index.article_count
        or not isinstance(index.normalized_article_sha256s, tuple)
        or len(index.normalized_article_sha256s) != index.article_count
        or any(not _is_sha256(value) for value in index.normalized_article_sha256s)
    ):
        raise MultiHopRAGMiniLMError("embedding index topology drifted")
    if not isinstance(index.chunk_vectors, np.ndarray) or index.chunk_vectors.ndim != 2:
        raise MultiHopRAGMiniLMError("chunk vectors shape drifted")
    chunks = _validated_matrix(
        index.chunk_vectors, rows=index.chunk_vectors.shape[0], field="chunk vectors"
    )
    topics = _validated_matrix(
        index.topic_vectors, rows=index.article_count, field="topic vectors"
    )
    cursor = 0
    for raw_range in index.article_chunk_ranges:
        if (
            not isinstance(raw_range, tuple)
            or len(raw_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_range)
        ):
            raise MultiHopRAGMiniLMError("article chunk ranges drifted")
        start, end = raw_range
        if start != cursor or not start < end <= len(chunks):
            raise MultiHopRAGMiniLMError("article chunk ranges drifted")
        cursor = end
    if cursor != len(chunks):
        raise MultiHopRAGMiniLMError("article chunks are not fully partitioned")
    body = {
        "article_chunk_ranges": [list(value) for value in index.article_chunk_ranges],
        "article_count": index.article_count,
        "article_sha256s": list(index.normalized_article_sha256s),
        "chunk_matrix_sha256": _array_sha256(chunks),
        "chunk_shape": list(chunks.shape),
        "encoder_receipt_sha256": index.encoder_receipt_sha256,
        "topic_matrix_sha256": _array_sha256(topics),
        "topic_shape": list(topics.shape),
        "version": VERSION,
    }
    return body


def recompute_embedding_index_sha256(index: CorpusEmbeddingIndex) -> str:
    """Recompute the canonical content hash without trusting its declaration."""

    return _stable_hash(_canonical_index_body(index))


def validate_corpus_embedding_index(
    index: CorpusEmbeddingIndex,
    *,
    expected_encoder_receipt_sha256: str = FROZEN_MINILM_RUNTIME_RECEIPT_SHA256,
) -> CorpusEmbeddingIndex:
    """Fail closed on topology, tensor, encoder-receipt, or content drift."""

    if (
        not isinstance(index, CorpusEmbeddingIndex)
        or not _is_sha256(index.index_sha256)
        or expected_encoder_receipt_sha256 != FROZEN_MINILM_RUNTIME_RECEIPT_SHA256
        or index.encoder_receipt_sha256 != expected_encoder_receipt_sha256
    ):
        raise MultiHopRAGMiniLMError("embedding index receipt is invalid")
    if recompute_embedding_index_sha256(index) != index.index_sha256:
        raise MultiHopRAGMiniLMError("embedding index content drifted")
    return index


def _validate_index(index: CorpusEmbeddingIndex) -> CorpusEmbeddingIndex:
    return validate_corpus_embedding_index(index)


def reciprocal_topic_neighbors(index: CorpusEmbeddingIndex) -> tuple[tuple[int, ...], ...]:
    index = _validate_index(index)
    if index.article_count <= TOPIC_K:
        raise MultiHopRAGMiniLMError("corpus is too small for reciprocal topic kNN")
    directed: list[tuple[int, ...]] = []
    for left in range(index.article_count):
        scored = [
            (
                quantized_cosine_similarity(
                    index.topic_vectors[left], index.topic_vectors[right]
                ),
                right,
            )
            for right in range(index.article_count)
            if right != left
        ]
        ordered = sorted(scored, key=lambda row: (-row[0], row[1]))[:TOPIC_K]
        directed.append(tuple(row[1] for row in ordered))
    return tuple(
        tuple(right for right in directed[left] if left in directed[right])
        for left in range(index.article_count)
    )


def compile_query_features(
    *, query: str, index: CorpusEmbeddingIndex, encoder: Encoder
) -> QueryFeatures:
    index = _validate_index(index)
    if encoder_receipt_sha256(encoder) != index.encoder_receipt_sha256:
        raise MultiHopRAGMiniLMError("query encoder receipt does not match embedding index")
    normalized_query = canonical_text(query, field="query")
    texts = (normalized_query, *(CAPABILITY_PROTOTYPES[name] for name in CAPABILITY_ORDER))
    matrix = _validated_matrix(encoder.encode(texts), rows=len(texts), field="query output")
    query_vector = matrix[0]
    capability_scores = tuple(
        quantized_cosine_similarity(query_vector, matrix[offset + 1])
        for offset in range(3)
    )
    predicted = min(
        CAPABILITY_ORDER,
        key=lambda name: (
            -capability_scores[CAPABILITY_ORDER.index(name)],
            CAPABILITY_ORDER.index(name),
        ),
    )
    relevance: list[int] = []
    for start, end in index.article_chunk_ranges:
        relevance.append(
            max(
                quantized_cosine_similarity(query_vector, index.chunk_vectors[row])
                for row in range(start, end)
            )
        )
    query_sha = hashlib.sha256(normalized_query.casefold().encode("utf-8")).hexdigest()
    features = QueryFeatures(
        embedding_index_sha256=index.index_sha256,
        normalized_query_sha256=query_sha,
        capability_similarity_ints=capability_scores,
        predicted_capability=predicted,
        dense_relevance_ints=tuple(relevance),
        feature_sha256="0" * 64,
    )
    body = _canonical_query_feature_body(features, index=index)
    return QueryFeatures(
        embedding_index_sha256=features.embedding_index_sha256,
        normalized_query_sha256=features.normalized_query_sha256,
        capability_similarity_ints=features.capability_similarity_ints,
        predicted_capability=features.predicted_capability,
        dense_relevance_ints=features.dense_relevance_ints,
        feature_sha256=_stable_hash(body),
    )


def _canonical_query_feature_body(
    features: QueryFeatures, *, index: CorpusEmbeddingIndex
) -> dict[str, object]:
    index = validate_corpus_embedding_index(index)
    if not isinstance(features, QueryFeatures):
        raise MultiHopRAGMiniLMError("query features are invalid")
    if (
        features.embedding_index_sha256 != index.index_sha256
        or not _is_sha256(features.normalized_query_sha256)
        or not isinstance(features.capability_similarity_ints, tuple)
        or len(features.capability_similarity_ints) != len(CAPABILITY_ORDER)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in features.capability_similarity_ints
        )
        or not isinstance(features.dense_relevance_ints, tuple)
        or len(features.dense_relevance_ints) != index.article_count
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in features.dense_relevance_ints
        )
    ):
        raise MultiHopRAGMiniLMError("query feature topology or index binding drifted")
    expected_capability = min(
        CAPABILITY_ORDER,
        key=lambda name: (
            -features.capability_similarity_ints[CAPABILITY_ORDER.index(name)],
            CAPABILITY_ORDER.index(name),
        ),
    )
    if features.predicted_capability != expected_capability:
        raise MultiHopRAGMiniLMError("predicted capability drifted")
    return {
        "capability_similarity_ints": list(features.capability_similarity_ints),
        "dense_relevance_ints": list(features.dense_relevance_ints),
        "embedding_index_sha256": features.embedding_index_sha256,
        "normalized_query_sha256": features.normalized_query_sha256,
        "predicted_capability": features.predicted_capability,
        "quantization_scale": QUANTIZATION_SCALE,
        "version": VERSION,
    }


def recompute_query_feature_sha256(
    features: QueryFeatures, *, index: CorpusEmbeddingIndex
) -> str:
    """Recompute a query-feature receipt against the supplied immutable index."""

    return _stable_hash(_canonical_query_feature_body(features, index=index))


def validate_query_features(
    features: QueryFeatures, *, index: CorpusEmbeddingIndex
) -> QueryFeatures:
    """Consumer-side validation for feature bytes and exact index identity."""

    if (
        not isinstance(features, QueryFeatures)
        or not _is_sha256(features.feature_sha256)
        or recompute_query_feature_sha256(features, index=index)
        != features.feature_sha256
    ):
        raise MultiHopRAGMiniLMError("query feature content drifted")
    return features


__all__ = [
    "ArticleText",
    "BODY_WINDOW_STRIDE",
    "BODY_WINDOW_TOKENS",
    "CAPABILITY_ORDER",
    "CAPABILITY_PROTOTYPES",
    "CorpusEmbeddingIndex",
    "FROZEN_MINILM_RUNTIME_RECEIPT_SHA256",
    "MultiHopRAGMiniLMError",
    "QueryFeatures",
    "TOPIC_K",
    "VERSION",
    "build_corpus_embedding_index",
    "canonical_text",
    "compile_query_features",
    "encoder_receipt_sha256",
    "frozen_minilm_runtime_identity",
    "recompute_embedding_index_sha256",
    "recompute_query_feature_sha256",
    "reciprocal_topic_neighbors",
    "serialize_article_chunks",
    "validate_corpus_embedding_index",
    "validate_query_features",
]

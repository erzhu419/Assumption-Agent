"""Frozen, row-free MiniLM runtime for the QASPER graph study."""

from .binding import (
    OfflineMiniLMEncoder,
    QasperMiniLMError,
    quantize_embeddings,
    quantized_cosine_similarity,
    query_paragraph_similarities,
    run_synthetic_canary,
    synthetic_canary_texts,
    verify_runtime_asset,
    verify_runtime_binding,
)

__all__ = [
    "OfflineMiniLMEncoder",
    "QasperMiniLMError",
    "quantize_embeddings",
    "quantized_cosine_similarity",
    "query_paragraph_similarities",
    "run_synthetic_canary",
    "synthetic_canary_texts",
    "verify_runtime_asset",
    "verify_runtime_binding",
]

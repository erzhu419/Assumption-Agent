"""Frozen global-corpus official HippoRAG adapter for MultiHopRAG."""

from .adapter import (
    build_official_hipporag_global_index_v1,
    retrieve_official_hipporag_global_index_v1,
)
from .contract import RetrievalBatch

__all__ = [
    "RetrievalBatch",
    "build_official_hipporag_global_index_v1",
    "retrieve_official_hipporag_global_index_v1",
]

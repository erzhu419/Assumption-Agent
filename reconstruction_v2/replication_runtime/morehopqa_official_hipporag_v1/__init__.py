"""Frozen global-corpus official HippoRAG adapter for MoreHopQA."""

from .adapter import (
    build_morehopqa_official_hipporag_global_index_v1,
    retrieve_morehopqa_official_hipporag_global_index_v1,
)
from .contract import RetrievalBatch

__all__ = [
    "RetrievalBatch",
    "build_morehopqa_official_hipporag_global_index_v1",
    "retrieve_morehopqa_official_hipporag_global_index_v1",
]

"""Frozen 8,192-unit official HippoRAG adapter for FEVEROUS."""

from .adapter import (
    build_feverous_official_hipporag_global_index_v1,
    retrieve_feverous_official_hipporag_global_index_v1,
)
from .contract import RetrievalBatch

__all__ = [
    "RetrievalBatch",
    "build_feverous_official_hipporag_global_index_v1",
    "retrieve_feverous_official_hipporag_global_index_v1",
]

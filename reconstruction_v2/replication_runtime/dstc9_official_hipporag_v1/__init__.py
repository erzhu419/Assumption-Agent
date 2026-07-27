"""Frozen source-free official HippoRAG adapter for DSTC9 Track 1."""

from .adapter import (
    build_dstc9_official_hipporag_global_index_v1,
    retrieve_dstc9_official_hipporag_global_index_v1,
)
from .contract import (
    CorpusInput,
    QueryInput,
    RetrievalBatch,
    make_corpus_input,
    make_query_input,
)

__all__ = [
    "CorpusInput",
    "QueryInput",
    "RetrievalBatch",
    "build_dstc9_official_hipporag_global_index_v1",
    "make_corpus_input",
    "make_query_input",
    "retrieve_dstc9_official_hipporag_global_index_v1",
]

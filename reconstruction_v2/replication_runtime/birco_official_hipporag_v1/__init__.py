"""BIRCO candidate-restricted official HippoRAG runtime."""

from .contract import (
    MAX_CANDIDATE_COUNT,
    MAX_POOL_SIZE,
    MIN_CANDIDATE_COUNT,
    MIN_POOL_SIZE,
    OFFICIAL_HIPPORAG_COMMIT,
    BircoOfficialHippoRAGError,
    CandidateDocument,
    common_projection_sha256,
    core_query_text,
    parse_output,
    validate_input,
)

__all__ = [
    "MAX_CANDIDATE_COUNT",
    "MAX_POOL_SIZE",
    "MIN_CANDIDATE_COUNT",
    "MIN_POOL_SIZE",
    "OFFICIAL_HIPPORAG_COMMIT",
    "BircoOfficialHippoRAGError",
    "CandidateDocument",
    "common_projection_sha256",
    "core_query_text",
    "parse_output",
    "validate_input",
]

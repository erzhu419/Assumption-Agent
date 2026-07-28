"""Source-free QuAC block adapter for the pinned official HippoRAG core.

The package exports only the label-free contract.  The production worker is
kept separate so importing the contract never imports HippoRAG, Torch, a
model, or any benchmark source.
"""

from .contract import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    TOP_K,
    BlockInput,
    QueryRow,
    QuacP1OfficialHippoRAGError,
    UnitRow,
    build_input,
    canonical_bytes,
    canonical_unit_document,
    serialize_corpus,
    stable_complete_ranking,
    stable_hash,
    validate_input,
    validate_output,
)

__all__ = [
    "INPUT_SCHEMA",
    "OUTPUT_SCHEMA",
    "TOP_K",
    "BlockInput",
    "QueryRow",
    "QuacP1OfficialHippoRAGError",
    "UnitRow",
    "build_input",
    "canonical_bytes",
    "canonical_unit_document",
    "serialize_corpus",
    "stable_complete_ranking",
    "stable_hash",
    "validate_input",
    "validate_output",
]

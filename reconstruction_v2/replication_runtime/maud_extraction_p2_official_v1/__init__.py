"""Source-free contract runtime for the MAUD extraction P2 HippoRAG arm.

The package keeps imports lazy so ``python -m ...worker`` does not import the
worker twice.  Public names are resolved from :mod:`worker` on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "INPUT_SCHEMA",
    "OUTPUT_SCHEMA",
    "QUERY_COUNT",
    "TOP_K",
    "MaudOfficialHippoRAGError",
    "canonical_json_bytes",
    "canonical_passage_document",
    "corpus_sha256",
    "input_payload",
    "parse_output",
    "retrieve_contract_with_core",
    "validate_input",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = import_module(f"{__name__}.worker")
    return getattr(module, name)

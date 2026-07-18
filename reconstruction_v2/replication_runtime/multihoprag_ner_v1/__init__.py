"""Frozen, offline typed-entity runtime for the MultiHopRAG study."""

from .binding import (
    ASSET_RELATIVE_PATH,
    MODEL_ID,
    MODEL_REVISION,
    verify_runtime_asset,
    verify_runtime_binding,
)
from .contract import (
    CanonicalText,
    EntitySpan,
    MultiHopRAGNERError,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    synthetic_canary_inputs,
)
from .worker import (
    FrozenNERExtractor,
    compute_preasset_canary,
    compute_synthetic_canary,
    merge_window_logits,
    tokenize_windows,
)

__all__ = [
    "ASSET_RELATIVE_PATH",
    "CanonicalText",
    "EntitySpan",
    "FrozenNERExtractor",
    "MODEL_ID",
    "MODEL_REVISION",
    "MultiHopRAGNERError",
    "compute_preasset_canary",
    "compute_synthetic_canary",
    "decode_request",
    "decode_response",
    "encode_request",
    "encode_response",
    "merge_window_logits",
    "synthetic_canary_inputs",
    "tokenize_windows",
    "verify_runtime_asset",
    "verify_runtime_binding",
]

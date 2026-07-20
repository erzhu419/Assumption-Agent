"""Exact upstream HippoRAG non-finite phrase-weight backport."""

from .backport import (
    BASELINE_SOURCE_SHA256,
    PATCHED_SOURCE_SHA256,
    UNIFIED_PATCH_SHA256,
    UPSTREAM_SOURCE_SHA256,
    apply_fixed_backport,
    unified_patch_bytes,
)

__all__ = [
    "BASELINE_SOURCE_SHA256",
    "PATCHED_SOURCE_SHA256",
    "UNIFIED_PATCH_SHA256",
    "UPSTREAM_SOURCE_SHA256",
    "apply_fixed_backport",
    "unified_patch_bytes",
]

"""Materialize the one-edit zero-weight totality hardening.

The input is the already-qualified upstream nonfinite hardening.  This module
changes only the obsolete cardinality assertion in ``get_top_k_weights`` and
fails closed on any source or patch drift.
"""

from __future__ import annotations

import difflib
import hashlib

from ..hipporag_upstream_hardening_v1 import backport as upstream_hardening


INPUT_SOURCE_SHA256 = upstream_hardening.PATCHED_SOURCE_SHA256
PATCHED_SOURCE_SHA256 = (
    "6d0938da96757504e88ec15ea88f15bc6a6605e006eeb00c780598330b4c698b"
)
UNIFIED_PATCH_SHA256 = (
    "a4a5584e0906d89eb09b59b4ee244d0a80b78a64cae9dbeafb50a923f7eddce5"
)

_OLD_ASSERTION = (
    "        assert np.count_nonzero(all_phrase_weights) == "
    "len(linking_score_map.keys())\n"
)
_NEW_CHECK = (
    "        if not np.all(np.isfinite(all_phrase_weights)):\n"
    "            raise ValueError(\"phrase weights contain a nonfinite value\")\n"
    "        selected_nonzero_phrase_ids = {\n"
    "            self.node_name_to_vertex_idx[phrase_key]\n"
    "            for phrase_key in top_k_phrases_keys\n"
    "            if phrase_key in self.node_name_to_vertex_idx\n"
    "            and all_phrase_weights[self.node_name_to_vertex_idx[phrase_key]] != 0.0\n"
    "        }\n"
    "        observed_nonzero_phrase_ids = set(np.flatnonzero(all_phrase_weights).tolist())\n"
    "        if observed_nonzero_phrase_ids != selected_nonzero_phrase_ids:\n"
    "            raise ValueError(\"unselected phrase weight remained nonzero\")\n"
)


class HippoRAGZeroWeightTotalityError(RuntimeError):
    """The exact totality-hardening contract failed closed."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def apply_totality_hardening(source: bytes) -> bytes:
    """Return the frozen one-edit hardening or fail closed."""

    if sha256_bytes(source) != INPUT_SOURCE_SHA256:
        raise HippoRAGZeroWeightTotalityError("input HippoRAG source drifted")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HippoRAGZeroWeightTotalityError("input source is not UTF-8") from exc
    if text.count(_OLD_ASSERTION) != 1 or _NEW_CHECK in text:
        raise HippoRAGZeroWeightTotalityError("totality source anchor drifted")
    patched = text.replace(_OLD_ASSERTION, _NEW_CHECK, 1).encode("utf-8")
    if sha256_bytes(patched) != PATCHED_SOURCE_SHA256:
        raise HippoRAGZeroWeightTotalityError("totalized source drifted")
    if sha256_bytes(unified_patch_bytes(source, patched)) != UNIFIED_PATCH_SHA256:
        raise HippoRAGZeroWeightTotalityError("totality patch drifted")
    return patched


def unified_patch_bytes(source: bytes, patched: bytes) -> bytes:
    """Return the canonical one-edit unified diff."""

    try:
        before = source.decode("utf-8").splitlines(keepends=True)
        after = patched.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise HippoRAGZeroWeightTotalityError("patch source is not UTF-8") from exc
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile="a/src/hipporag/HippoRAG.py",
            tofile="b/src/hipporag/HippoRAG.py",
        )
    ).encode("utf-8")

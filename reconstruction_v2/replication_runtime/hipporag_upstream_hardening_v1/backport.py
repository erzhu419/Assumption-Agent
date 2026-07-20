"""Materialize the exact two-edit backport from upstream HippoRAG.

The frozen baseline remains untouched.  Formal runtimes bind the materialized
file over the baseline module inside an offline bubblewrap namespace.
"""

from __future__ import annotations

import difflib
import hashlib


BASELINE_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
UPSTREAM_COMMIT = "1e8f60981bf760b64003aa5bf5668126d0c106b3"
BASELINE_SOURCE_SHA256 = (
    "3698809e70b4a39eb75fef40fa8aacc5e86788e04a4e6c87725ae6ba6b44f635"
)
UPSTREAM_SOURCE_SHA256 = (
    "b8249045c6adaefd156715dc95edfa8931bce99cdf6897fbd1e53be2b8ad45f7"
)
PATCHED_SOURCE_SHA256 = (
    "960561b080531fe4d668bde635e81f8e65620ce50bdacdd9a25531e856fa3e05"
)
UNIFIED_PATCH_SHA256 = (
    "b0426cb70728a4cbf4985d7577303281fa3e298e68c38acda1e9dca17ef86fea"
)

_OLD_BLOCK = (
    "                phrases_and_ids.add((phrase, phrase_id))\n"
    "\n"
    "        phrase_weights /= number_of_occurs\n"
)
_NEW_BLOCK = (
    "                    phrases_and_ids.add((phrase, phrase_id))\n"
    "\n"
    "        phrase_weights = np.divide(phrase_weights, number_of_occurs, "
    "out=np.zeros_like(phrase_weights), where=number_of_occurs != 0)\n"
)


class HippoRAGUpstreamHardeningError(RuntimeError):
    """The exact source/backport contract failed closed."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def apply_fixed_backport(baseline: bytes) -> bytes:
    """Return the frozen two-edit upstream backport or fail closed."""

    if sha256_bytes(baseline) != BASELINE_SOURCE_SHA256:
        raise HippoRAGUpstreamHardeningError("baseline HippoRAG source drifted")
    try:
        text = baseline.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HippoRAGUpstreamHardeningError("baseline source is not UTF-8") from exc
    if text.count(_OLD_BLOCK) != 1 or _NEW_BLOCK in text:
        raise HippoRAGUpstreamHardeningError("backport source anchor drifted")
    patched = text.replace(_OLD_BLOCK, _NEW_BLOCK, 1).encode("utf-8")
    if sha256_bytes(patched) != PATCHED_SOURCE_SHA256:
        raise HippoRAGUpstreamHardeningError("patched HippoRAG source drifted")
    if sha256_bytes(unified_patch_bytes(baseline, patched)) != UNIFIED_PATCH_SHA256:
        raise HippoRAGUpstreamHardeningError("unified backport drifted")
    return patched


def unified_patch_bytes(baseline: bytes, patched: bytes) -> bytes:
    """Return the canonical unified diff used by the preregistration."""

    try:
        before = baseline.decode("utf-8").splitlines(keepends=True)
        after = patched.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise HippoRAGUpstreamHardeningError("patch source is not UTF-8") from exc
    value = "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile="a/src/hipporag/HippoRAG.py",
            tofile="b/src/hipporag/HippoRAG.py",
        )
    ).encode("utf-8")
    return value


def verify_upstream_contains_backport(upstream: bytes) -> None:
    """Verify the pinned official revision contains both exact semantics."""

    if sha256_bytes(upstream) != UPSTREAM_SOURCE_SHA256:
        raise HippoRAGUpstreamHardeningError("upstream HippoRAG source drifted")
    try:
        text = upstream.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HippoRAGUpstreamHardeningError("upstream source is not UTF-8") from exc
    if text.count(_NEW_BLOCK) != 1 or _OLD_BLOCK in text:
        raise HippoRAGUpstreamHardeningError(
            "pinned upstream revision does not contain the exact backport"
        )

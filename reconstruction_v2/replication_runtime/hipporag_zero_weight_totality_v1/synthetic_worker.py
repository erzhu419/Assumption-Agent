"""Exercise all three branches of the frozen phrase-weight invariant."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .backport import PATCHED_SOURCE_SHA256


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def exercise_fixture(HippoRAG: type, compute_mdhash_id: Any) -> dict[str, Any]:
    """Call the patched method without constructing model or graph state."""

    alpha_key = compute_mdhash_id(content="alpha", prefix="entity-")
    beta_key = compute_mdhash_id(content="beta", prefix="entity-")
    core = SimpleNamespace(node_name_to_vertex_idx={alpha_key: 0, beta_key: 1})
    method = HippoRAG.get_top_k_weights

    allowed_weights = np.asarray([0.75, 0.0, 0.0])
    allowed_scores = {"alpha": 0.9, "beta": 0.8}
    weights_before = allowed_weights.copy()
    scores_before = dict(allowed_scores)
    observed_weights, observed_scores = method(
        core,
        link_top_k=2,
        all_phrase_weights=allowed_weights,
        linking_score_map=allowed_scores,
    )
    allowed_unchanged = bool(
        observed_weights is allowed_weights
        and observed_scores == scores_before
        and np.array_equal(observed_weights, weights_before)
        and allowed_scores == scores_before
    )

    rejected: list[str] = []
    for name, weights in (
        ("nonfinite", np.asarray([0.75, np.nan, 0.0])),
        ("unselected_nonzero", np.asarray([0.75, 0.0, 0.25])),
    ):
        try:
            method(
                core,
                link_top_k=2,
                all_phrase_weights=weights,
                linking_score_map={"alpha": 0.9, "beta": 0.8},
            )
        except ValueError:
            rejected.append(name)

    result = {
        "allowed_linking_key_count": len(observed_scores),
        "allowed_nonzero_weight_count": int(np.count_nonzero(observed_weights)),
        "allowed_values_unchanged": allowed_unchanged,
        "rejected_cases": rejected,
        "schema": "hipporag_zero_weight_totality_synthetic_fixture_v1",
        "source_sha256": PATCHED_SOURCE_SHA256,
    }
    if (
        not allowed_unchanged
        or result["allowed_linking_key_count"] != 2
        or result["allowed_nonzero_weight_count"] != 1
        or rejected != ["nonfinite", "unselected_nonzero"]
    ):
        raise RuntimeError("synthetic totality fixture failed")
    return result


def run_fixture() -> dict[str, Any]:
    from hipporag import HippoRAG
    from hipporag.utils.misc_utils import compute_mdhash_id

    source_path = Path(inspect.getfile(HippoRAG)).resolve(strict=True)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != PATCHED_SOURCE_SHA256:
        raise RuntimeError("totalized source is not bound")
    return exercise_fixture(HippoRAG, compute_mdhash_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    _write_json_exclusive(arguments.output, run_fixture())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

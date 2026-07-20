"""Exercise the upstream fix on a deterministic absent-entity graph fixture."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .backport import PATCHED_SOURCE_SHA256


class _Vertices:
    def __getitem__(self, key: str) -> list[str]:
        if key != "name":
            raise KeyError(key)
        return ["passage-0", "passage-1"]


class _Graph:
    vs = _Vertices()


class _ChunkStore:
    def get_row(self, key: str) -> Mapping[str, str]:
        return {"content": "document zero" if key == "passage-0" else "document one"}


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


def exercise_fixture(HippoRAG: type, source_sha256: str) -> dict[str, Any]:
    """Exercise an explicitly supplied HippoRAG class on the frozen fixture."""

    if source_sha256 != PATCHED_SOURCE_SHA256:
        raise RuntimeError("hardened source is not bound")

    core = HippoRAG.__new__(HippoRAG)
    core.graph = _Graph()
    core.node_name_to_vertex_idx = {"passage-0": 0, "passage-1": 1}
    core.ent_node_to_chunk_ids = {}
    core.passage_node_keys = ["passage-0", "passage-1"]
    core.passage_node_idxs = [0, 1]
    core.chunk_embedding_store = _ChunkStore()
    core.global_config = SimpleNamespace(damping=0.5)
    core.ppr_time = 0.0
    observed_weights: list[float] = []

    def dense(_self: object, _query: str):
        return np.asarray([0, 1]), np.asarray([1.0, 0.0])

    def ppr(_self: object, weights: np.ndarray, damping: float):
        if damping != 0.5:
            raise RuntimeError("damping drifted")
        observed_weights.extend(float(value) for value in weights.tolist())
        return np.asarray([0, 1]), np.asarray([0.75, 0.25])

    core.dense_passage_retrieval = MethodType(dense, core)
    core.run_ppr = MethodType(ppr, core)
    rows, scores = core.graph_search_with_fact_entities(
        query="synthetic absent entity fixture",
        link_top_k=5,
        query_fact_scores=np.asarray([1.0]),
        top_k_facts=[("absent subject", "relation", "absent object")],
        top_k_fact_indices=[0],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_division = np.asarray([0.0, 0.0]) / np.asarray([0.0, 0.0])
    result = {
        "baseline_nonfinite_count": int(
            np.count_nonzero(~np.isfinite(baseline_division))
        ),
        "hardened_node_weights": observed_weights,
        "hardened_nonfinite_count": sum(
            not math.isfinite(value) for value in observed_weights
        ),
        "ranked_passage_rows": [int(value) for value in rows.tolist()],
        "ranked_passage_scores": [float(value) for value in scores.tolist()],
        "schema": "hipporag_upstream_hardening_synthetic_fixture_v1",
        "source_sha256": source_sha256,
    }
    if (
        result["baseline_nonfinite_count"] != 2
        or result["hardened_nonfinite_count"] != 0
        or result["ranked_passage_rows"] != [0, 1]
        or len(observed_weights) != 2
        or not math.isclose(sum(observed_weights), 0.05, abs_tol=1e-12)
    ):
        raise RuntimeError("synthetic upstream-fix fixture failed")
    return result


def run_fixture() -> dict[str, Any]:
    from hipporag import HippoRAG

    source_path = Path(inspect.getfile(HippoRAG)).resolve(strict=True)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return exercise_fixture(HippoRAG, source_sha256)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    _write_json_exclusive(arguments.output, run_fixture())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

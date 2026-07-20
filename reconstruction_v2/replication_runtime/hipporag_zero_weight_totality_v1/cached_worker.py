"""Offline cached-index worker for the totalized HippoRAG source."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Sequence

from ..bright_official_hipporag_v1 import worker as baseline_worker
from ..bright_official_hipporag_v1.contract import BrightOfficialHippoRAGError
from ..hipporag_upstream_hardening_v1 import cached_worker as qualified_worker

from .backport import PATCHED_SOURCE_SHA256


def _assert_bound_source() -> None:
    from hipporag import HippoRAG

    source = Path(inspect.getfile(HippoRAG)).resolve(strict=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != PATCHED_SOURCE_SHA256:
        raise BrightOfficialHippoRAGError("totalized HippoRAG source is not bound")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    _assert_bound_source()
    query, documents = baseline_worker._load_input(arguments.input)
    core = qualified_worker._build_cached_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
    )
    payload = qualified_worker._retrieve_from_cached_index(
        core=core,
        query=query,
        documents=documents,
    )
    qualified_worker._write_exclusive(arguments.output, payload)
    print(
        json.dumps(
            {"source_sha256": PATCHED_SOURCE_SHA256, "status": "passed"},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

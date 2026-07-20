"""Offline cached-index parity worker for the fixed HippoRAG source."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from replication_runtime.bright_official_hipporag_v1 import (
    worker as baseline_worker,
)
from replication_runtime.bright_official_hipporag_v1.contract import (
    CANDIDATE_COUNT,
    BrightOfficialHippoRAGError,
    canonical_json_bytes,
    output_payload,
    serialize_documents,
    stable_top_k,
)

from .backport import PATCHED_SOURCE_SHA256


def _assert_bound_source() -> None:
    from hipporag import HippoRAG

    source = Path(inspect.getfile(HippoRAG)).resolve(strict=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != PATCHED_SOURCE_SHA256:
        raise BrightOfficialHippoRAGError("hardened HippoRAG source is not bound")


def _build_cached_core(
    *, save_dir: Path, llm_model: Path, embedding_model: Path
) -> object:
    import torch
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    torch.set_num_threads(baseline_worker.TORCH_THREAD_COUNT)
    torch.manual_seed(0)
    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name="Transformers/" + str(llm_model),
        embedding_model_name="Transformers/" + str(embedding_model),
        openie_mode="online",
        max_new_tokens=baseline_worker.OPENIE_MAX_NEW_TOKENS,
        retrieval_top_k=CANDIDATE_COUNT,
        qa_top_k=10,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        save_openie=True,
        temperature=0,
        seed=0,
    )
    core = HippoRAG(global_config=config)
    baseline_worker._install_completion_only_backend(core)
    return core


def _retrieve_from_cached_index(
    *, core: object, query: str, documents: Sequence[Any]
) -> dict[str, Any]:
    serialized = serialize_documents(documents)
    mapping = {text: row.ordinal for text, row in zip(serialized, documents)}
    store = getattr(core, "chunk_embedding_store", None)
    get_all_texts = getattr(store, "get_all_texts", None)
    if not callable(get_all_texts) or set(get_all_texts()) != set(serialized):
        raise BrightOfficialHippoRAGError("cached index document identity drifted")
    retrieve = getattr(core, "retrieve", None)
    if not callable(retrieve):
        raise BrightOfficialHippoRAGError("official cached retrieve is unavailable")
    rows = retrieve([query], num_to_retrieve=CANDIDATE_COUNT)
    if not isinstance(rows, list) or len(rows) != 1:
        raise BrightOfficialHippoRAGError("official cached result drifted")
    solution = rows[0]
    top = stable_top_k(
        retrieved_documents=getattr(solution, "docs", None),
        retrieved_scores=getattr(solution, "doc_scores", None),
        document_to_ordinal=mapping,
    )
    graph = getattr(core, "graph", None)
    vcount = getattr(graph, "vcount", None)
    ecount = getattr(graph, "ecount", None)
    if not callable(vcount) or not callable(ecount):
        raise BrightOfficialHippoRAGError("official graph counters are unavailable")
    return output_payload(
        top_ordinals=top,
        graph_nodes=int(vcount()),
        graph_edges=int(ecount()),
    )


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


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
    core = _build_cached_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
    )
    payload = _retrieve_from_cached_index(
        core=core, query=query, documents=documents
    )
    _write_exclusive(arguments.output, payload)
    print(
        json.dumps(
            {
                "source_sha256": PATCHED_SOURCE_SHA256,
                "status": "passed",
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One-index, one-eager-batch official HippoRAG worker for QuAC P1.

This module has no QuAC loader.  It consumes only the strict source-free block
contract, indexes every canonical evidence-window unit exactly once, and
passes all full question-only query strings to one official ``retrieve`` call
requesting the complete corpus.  The production core is constructed through
the already-qualified MAUD P2 compatibility helper.  The outer formal
controller remains responsible for binding exact model/source/CUDA assets and
for placing this worker in a network-isolated sandbox.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping, Sequence

from replication_runtime.maud_extraction_p2_official_v1 import (
    worker as qualified_maud,
)

from . import contract


VERSION = "quac_p1_official_hipporag_worker_v1"
NATIVE_THREAD_KEYS = qualified_maud.NATIVE_THREAD_ENVIRONMENT_KEYS
_VISIBLE_GPU = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _graph_count(core: object, method_name: str) -> int:
    graph = getattr(core, "graph", None)
    method = getattr(graph, method_name, None)
    if not callable(method):
        return 0
    try:
        value = int(method())
    except BaseException as exc:
        raise contract.QuacP1OfficialHippoRAGError(
            "official graph count is unavailable"
        ) from exc
    if value < 0:
        raise contract.QuacP1OfficialHippoRAGError(
            "official graph count drifted"
        )
    return value


def retrieve_block_with_core(
    *,
    core: object,
    private_input: Mapping[str, object],
) -> dict[str, object]:
    """Index the complete block once and retrieve all queries in one call."""

    block = contract.validate_input(private_input)
    documents = contract.serialize_corpus(block.units)
    document_to_unit_id = {
        document: unit.unit_id
        for document, unit in zip(documents, block.units)
    }
    if len(document_to_unit_id) != len(block.units):
        raise contract.QuacP1OfficialHippoRAGError(
            "canonical document addressing collided"
        )
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise contract.QuacP1OfficialHippoRAGError(
            "official core surface drifted"
        )
    try:
        index(list(documents))
        solutions = retrieve(
            [query.text for query in block.queries],
            num_to_retrieve=len(documents),
        )
    except BaseException as exc:
        raise contract.QuacP1OfficialHippoRAGError(
            "official index or retrieve call failed"
        ) from exc
    if not isinstance(solutions, list) or len(solutions) != len(
        block.queries
    ):
        raise contract.QuacP1OfficialHippoRAGError(
            "official eager query batch drifted"
        )
    rankings = tuple(
        contract.stable_complete_ranking(
            retrieved_documents=getattr(solution, "docs", None),
            retrieved_scores=getattr(solution, "doc_scores", None),
            document_to_unit_id=document_to_unit_id,
        )
        for solution in solutions
    )
    return contract.build_output(
        input_value=private_input,
        full_rankings=rankings,
        graph_node_count=_graph_count(core, "vcount"),
        graph_edge_count=_graph_count(core, "ecount"),
    )


def _require_environment() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("TOKENIZERS_PARALLELISM") != "false"
        or not isinstance(visible, str)
        or _VISIBLE_GPU.fullmatch(visible) is None
        or any(os.environ.get(key) != "1" for key in NATIVE_THREAD_KEYS)
    ):
        raise contract.QuacP1OfficialHippoRAGError(
            "official offline/single-GPU/single-thread environment drifted"
        )


def _require_single_visible_gpu() -> None:
    try:
        import torch

        valid = torch.cuda.is_available() and torch.cuda.device_count() == 1
    except BaseException as exc:
        raise contract.QuacP1OfficialHippoRAGError(
            "CUDA runtime is unavailable"
        ) from exc
    if not valid:
        raise contract.QuacP1OfficialHippoRAGError(
            "worker must see exactly one available GPU"
        )


def build_official_core(
    *,
    index_root: Path,
    llm_model_alias: str,
    embedding_model_alias: str,
    corpus_count: int,
) -> object:
    """Construct the real core through the qualified MAUD compatibility seam."""

    if (
        type(corpus_count) is not int
        or not contract.MIN_UNIT_COUNT
        <= corpus_count
        <= contract.MAX_UNIT_COUNT
    ):
        raise contract.QuacP1OfficialHippoRAGError(
            "official corpus count drifted"
        )
    try:
        return qualified_maud._build_core(
            save_dir=index_root,
            llm_alias=qualified_maud._validate_model_alias(
                llm_model_alias, "LLM model"
            ),
            embedding_alias=qualified_maud._validate_model_alias(
                embedding_model_alias, "embedding model"
            ),
            passage_count=corpus_count,
        )
    except BaseException as exc:
        raise contract.QuacP1OfficialHippoRAGError(
            "qualified official core construction failed"
        ) from exc


def _read_private(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise contract.QuacP1OfficialHippoRAGError(
            "private worker input is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not isinstance(value, dict)
        or raw != contract.canonical_bytes(value)
    ):
        raise contract.QuacP1OfficialHippoRAGError(
            "private worker input metadata drifted"
        )
    contract.validate_input(value)
    return value


def _write_private(path: Path, value: Mapping[str, object]) -> None:
    raw = contract.canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_once(
    *,
    private_input: Mapping[str, object],
    output_path: Path,
    index_root: Path,
    llm_model_alias: str,
    embedding_model_alias: str,
) -> dict[str, object]:
    """Run the real build/retrieve path exactly once."""

    _require_environment()
    _require_single_visible_gpu()
    block = contract.validate_input(private_input)
    if index_root.exists() or index_root.is_symlink():
        raise contract.QuacP1OfficialHippoRAGError(
            "official index root is not fresh"
        )
    index_root.mkdir(mode=0o700)
    core = build_official_core(
        index_root=index_root,
        llm_model_alias=llm_model_alias,
        embedding_model_alias=embedding_model_alias,
        corpus_count=len(block.units),
    )
    result = retrieve_block_with_core(
        core=core,
        private_input=private_input,
    )
    contract.validate_output(result, expected_input=private_input)
    _write_private(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    arguments = parser.parse_args(argv)
    private_input = _read_private(arguments.input)
    result = run_once(
        private_input=private_input,
        output_path=arguments.output,
        index_root=arguments.index_root,
        llm_model_alias=arguments.llm_model,
        embedding_model_alias=arguments.embedding_model,
    )
    print(
        json.dumps(
            {
                "corpus_count": result["runtime"]["corpus_count"],  # type: ignore[index]
                "query_count": result["runtime"]["query_count"],  # type: ignore[index]
                "status": "passed",
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

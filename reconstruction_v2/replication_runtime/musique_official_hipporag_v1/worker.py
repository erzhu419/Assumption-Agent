"""Private one-item worker for the official HippoRAG retrieve-only path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import (
    FROZEN_CORE_CONFIG,
    INPUT_SCHEMA,
    MuSiQueOfficialHippoRAGError,
    serialize_candidate_corpus,
    stable_top_five_from_official_result,
    validate_single_item,
)


def retrieve_idx_with_core(
    *,
    core: object,
    question: str,
    paragraphs: Sequence[Mapping[str, object]],
) -> tuple[int, ...]:
    """Index one corpus and consume only the official core retrieval result."""

    validated_question, validated_paragraphs = validate_single_item(question, paragraphs)
    documents = serialize_candidate_corpus(validated_paragraphs)
    document_to_idx = {
        document: paragraph.idx
        for document, paragraph in zip(documents, validated_paragraphs)
    }
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise MuSiQueOfficialHippoRAGError("official core lacks index or retrieve")
    index(list(documents))
    rows = retrieve([validated_question], num_to_retrieve=len(documents))
    if not isinstance(rows, list) or len(rows) != 1:
        raise MuSiQueOfficialHippoRAGError("official core returned an invalid query batch")
    solution = rows[0]
    retrieved_documents = getattr(solution, "docs", None)
    retrieved_scores = getattr(solution, "doc_scores", None)
    return stable_top_five_from_official_result(
        retrieved_documents=retrieved_documents,
        retrieved_scores=retrieved_scores,
        document_to_idx=document_to_idx,
    )


def _load_input(path: Path) -> tuple[str, list[Mapping[str, object]]]:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueOfficialHippoRAGError("worker input is unavailable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError("worker input is invalid") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "question",
        "paragraphs",
    }:
        raise MuSiQueOfficialHippoRAGError("worker input envelope is not exact")
    if payload.get("schema") != INPUT_SCHEMA or not isinstance(payload.get("paragraphs"), list):
        raise MuSiQueOfficialHippoRAGError("worker input schema mismatch")
    question = payload.get("question")
    if not isinstance(question, str):
        raise MuSiQueOfficialHippoRAGError("worker question is malformed")
    return question, payload["paragraphs"]


def _write_idx_only(path: Path, values: Sequence[int]) -> None:
    raw = (json.dumps(list(values), separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _build_official_core(*, save_dir: Path, llm_model: Path, embedding_model: Path) -> object:
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name="Transformers/" + str(llm_model),
        embedding_model_name="Transformers/" + str(embedding_model),
        openie_mode=FROZEN_CORE_CONFIG["openie_mode"],
        max_new_tokens=FROZEN_CORE_CONFIG["max_new_tokens"],
        retrieval_top_k=FROZEN_CORE_CONFIG["retrieval_top_k"],
        qa_top_k=FROZEN_CORE_CONFIG["qa_top_k"],
        force_index_from_scratch=FROZEN_CORE_CONFIG["force_index_from_scratch"],
        save_openie=FROZEN_CORE_CONFIG["save_openie"],
    )
    core = HippoRAG(global_config=config)
    core.llm_model.llm_config.generate_params["max_tokens"] = FROZEN_CORE_CONFIG[
        "max_new_tokens"
    ]
    return core


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.index_root.exists():
        raise MuSiQueOfficialHippoRAGError("per-item index root already exists")
    arguments.index_root.mkdir(mode=0o700)
    question, paragraphs = _load_input(arguments.input)
    core = _build_official_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
    )
    result = retrieve_idx_with_core(core=core, question=question, paragraphs=paragraphs)
    _write_idx_only(arguments.output, result)
    print(json.dumps({"retrieval_count": len(result), "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Isolated build-once and reopen-retrieve worker for the global corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import (
    BUILD_RECEIPT_SCHEMA,
    CORPUS_INPUT_SCHEMA,
    CORPUS_SIZE,
    FROZEN_CORE_CONFIG,
    IndexTreeSnapshot,
    MAX_QUERY_BATCH,
    HybridQAOfficialHippoRAGError,
    QUERY_INPUT_SCHEMA,
    RETRIEVAL_OUTPUT_SCHEMA,
    canonical_json_bytes,
    corpus_text_multiplicity,
    make_build_receipt,
    make_retrieval_receipt,
    serialize_corpus,
    snapshot_index_tree,
    stable_top_five_from_official_result,
    validate_build_receipt,
    validate_corpus,
    validate_queries,
)


def build_index_with_core(
    *,
    core: object,
    articles: Sequence[Mapping[str, object]],
    index_root: Path,
    runtime_attestation_receipt_sha256: str,
) -> dict[str, Any]:
    """Call the official index method exactly once for the entire corpus."""

    validated = validate_corpus(articles)
    documents = serialize_corpus(validated)
    # Fail before the irreversible index call if distinct exact texts alias
    # under HippoRAG's pinned content-addressing hash, then index the canonical
    # first-occurrence quotient exactly once.
    corpus_text_multiplicity(documents)
    unique_documents = list(dict.fromkeys(documents))
    index = getattr(core, "index", None)
    if not callable(index):
        raise HybridQAOfficialHippoRAGError("official core lacks index")
    index(unique_documents)
    return make_build_receipt(
        documents,
        index_snapshot=snapshot_index_tree(index_root),
        runtime_attestation_receipt_sha256=(
            runtime_attestation_receipt_sha256
        ),
    )


def retrieve_batches_with_core(
    *,
    core: object,
    articles: Sequence[Mapping[str, object]],
    queries: Sequence[str],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Retrieve in batches of at most eight without invoking ``index``."""

    validated_articles = validate_corpus(articles)
    validated_queries = validate_queries(queries)
    documents = serialize_corpus(validated_articles)
    document_to_indices: dict[str, list[int]] = {}
    for document, article in zip(documents, validated_articles, strict=True):
        document_to_indices.setdefault(document, []).append(article.idx)
    retrieve = getattr(core, "retrieve", None)
    if not callable(retrieve):
        raise HybridQAOfficialHippoRAGError("official core lacks retrieve")

    result: list[tuple[int, ...]] = []
    batch_sizes: list[int] = []
    for offset in range(0, len(validated_queries), MAX_QUERY_BATCH):
        batch = list(validated_queries[offset : offset + MAX_QUERY_BATCH])
        if not 1 <= len(batch) <= MAX_QUERY_BATCH:
            raise HybridQAOfficialHippoRAGError("internal query batch bound drifted")
        rows = retrieve(batch, num_to_retrieve=CORPUS_SIZE)
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise HybridQAOfficialHippoRAGError(
                "official core returned an invalid query batch"
            )
        for solution in rows:
            result.append(
                stable_top_five_from_official_result(
                    retrieved_documents=getattr(solution, "docs", None),
                    retrieved_scores=getattr(solution, "doc_scores", None),
                    document_to_indices=document_to_indices,
                )
            )
        batch_sizes.append(len(batch))
    indices = tuple(result)
    return indices, tuple(batch_sizes)


def _load_json(path: Path, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise HybridQAOfficialHippoRAGError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridQAOfficialHippoRAGError(f"{field} is invalid") from exc
    if raw != canonical_json_bytes(value) or not isinstance(value, dict):
        raise HybridQAOfficialHippoRAGError(f"{field} is not canonical JSON")
    return value


def _load_corpus(path: Path) -> list[Mapping[str, object]]:
    payload = _load_json(path, "corpus input")
    if set(payload) != {"articles", "schema"} or payload.get(
        "schema"
    ) != CORPUS_INPUT_SCHEMA:
        raise HybridQAOfficialHippoRAGError("corpus input envelope mismatch")
    articles = payload.get("articles")
    if not isinstance(articles, list):
        raise HybridQAOfficialHippoRAGError("corpus input articles are malformed")
    validate_corpus(articles)
    return articles


def _load_queries(path: Path) -> list[str]:
    payload = _load_json(path, "query input")
    if set(payload) != {"queries", "schema"} or payload.get(
        "schema"
    ) != QUERY_INPUT_SCHEMA:
        raise HybridQAOfficialHippoRAGError("query input envelope mismatch")
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise HybridQAOfficialHippoRAGError("query input rows are malformed")
    return list(validate_queries(queries))


def _load_and_validate_build_receipt(
    path: Path,
    *,
    articles: Sequence[Mapping[str, object]],
    index_snapshot: IndexTreeSnapshot,
    runtime_attestation_receipt_sha256: str,
) -> dict[str, Any]:
    payload = _load_json(path, "build receipt")
    if payload.get("schema") != BUILD_RECEIPT_SCHEMA:
        raise HybridQAOfficialHippoRAGError("build receipt schema mismatch")
    documents = serialize_corpus(validate_corpus(articles))
    return validate_build_receipt(
        payload,
        expected_documents=documents,
        expected_index_snapshot=index_snapshot,
        expected_runtime_attestation_receipt_sha256=(
            runtime_attestation_receipt_sha256
        ),
    )


def _write_exclusive_json(path: Path, payload: object) -> None:
    raw = canonical_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _build_official_core(
    *,
    save_dir: Path,
    llm_model: Path,
    embedding_model: Path,
    force_index_from_scratch: bool,
) -> object:
    """Construct the pinned official core in either build or reopen mode."""

    if not isinstance(force_index_from_scratch, bool):
        raise HybridQAOfficialHippoRAGError(
            "force_index_from_scratch must be boolean"
        )
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
        force_index_from_scratch=force_index_from_scratch,
        save_openie=FROZEN_CORE_CONFIG["save_openie"],
    )
    core = HippoRAG(global_config=config)
    core.llm_model.llm_config.generate_params["max_tokens"] = FROZEN_CORE_CONFIG[
        "max_new_tokens"
    ]
    return core


def _run_build(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.index_root.exists():
        raise HybridQAOfficialHippoRAGError(
            "global index root must not exist before the build stage"
        )
    arguments.index_root.mkdir(mode=0o700)
    articles = _load_corpus(arguments.corpus_input)
    core = _build_official_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
        force_index_from_scratch=True,
    )
    receipt = build_index_with_core(
        core=core,
        articles=articles,
        index_root=arguments.index_root,
        runtime_attestation_receipt_sha256=(
            arguments.runtime_attestation_receipt_sha256
        ),
    )
    _write_exclusive_json(arguments.output, receipt)
    return {
        "corpus_count": CORPUS_SIZE,
        "index_call_count": 1,
        "stage": "build",
        "status": "passed",
    }


def _run_retrieve(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.index_root.is_symlink() or not arguments.index_root.is_dir():
        raise HybridQAOfficialHippoRAGError(
            "global index root is unavailable for reopen"
        )
    if arguments.query_input is None or arguments.build_receipt is None:
        raise HybridQAOfficialHippoRAGError(
            "retrieve stage requires queries and the build receipt"
        )
    articles = _load_corpus(arguments.corpus_input)
    index_snapshot_before = snapshot_index_tree(arguments.index_root)
    build_receipt = _load_and_validate_build_receipt(
        arguments.build_receipt,
        articles=articles,
        index_snapshot=index_snapshot_before,
        runtime_attestation_receipt_sha256=(
            arguments.runtime_attestation_receipt_sha256
        ),
    )
    queries = _load_queries(arguments.query_input)
    core = _build_official_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
        force_index_from_scratch=False,
    )
    indices, batch_sizes = retrieve_batches_with_core(
        core=core,
        articles=articles,
        queries=queries,
    )
    index_snapshot_after = snapshot_index_tree(arguments.index_root)
    receipt = make_retrieval_receipt(
        documents=serialize_corpus(validate_corpus(articles)),
        queries=queries,
        indices=indices,
        batch_sizes=batch_sizes,
        build_receipt=build_receipt,
        index_snapshot_before=index_snapshot_before,
        index_snapshot_after=index_snapshot_after,
    )
    output = {
        "receipt": receipt,
        "retrieved_idx": [list(row) for row in indices],
        "schema": RETRIEVAL_OUTPUT_SCHEMA,
    }
    _write_exclusive_json(arguments.output, output)
    return {
        "batch_count": receipt["retrieval_call_count"],
        "index_call_count": 0,
        "query_count": len(queries),
        "stage": "retrieve",
        "status": "passed",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "retrieve"), required=True)
    parser.add_argument("--corpus-input", required=True, type=Path)
    parser.add_argument("--query-input", type=Path)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    parser.add_argument(
        "--runtime-attestation-receipt-sha256", required=True
    )
    arguments = parser.parse_args(argv)
    status = _run_build(arguments) if arguments.stage == "build" else _run_retrieve(arguments)
    print(json.dumps(status, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_index_with_core", "retrieve_batches_with_core"]

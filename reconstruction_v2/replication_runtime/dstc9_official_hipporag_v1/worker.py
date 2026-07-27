"""Isolated GPU0 build-once/reopen-retrieve worker for DSTC9."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .contract import (
    BUILD_RECEIPT_SCHEMA,
    CORPUS_SIZE,
    CUDA_VISIBLE_DEVICES,
    FROZEN_CORE_CONFIG,
    LOGICAL_CUDA_DEVICE,
    MAX_QUERY_BATCH,
    RETRIEVAL_OUTPUT_SCHEMA,
    WORKER_ENVIRONMENT_KEYS,
    WORKER_FIXED_ENVIRONMENT_VALUES,
    CorpusInput,
    Dstc9OfficialHippoRAGError,
    IndexTreeSnapshot,
    QueryInput,
    canonical_json_bytes,
    corpus_input_projection,
    corpus_text_multiplicity,
    make_build_receipt,
    make_retrieval_receipt,
    query_input_projection,
    serialize_corpus,
    serialize_queries,
    snapshot_index_tree,
    stable_top_five_from_official_result,
    validate_build_receipt,
    validate_corpus_input,
    validate_query_input,
)
from .runtime_binding import (
    Dstc9P17RuntimeBindingError,
    verify_worker_runtime_provenance,
)


def _validate_effective_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Fail closed unless ``env --ignore-environment`` supplied the contract."""

    effective = os.environ if environment is None else environment
    if (
        not isinstance(effective, Mapping)
        or frozenset(effective) != WORKER_ENVIRONMENT_KEYS
        or any(
            effective.get(key) != value
            for key, value in WORKER_FIXED_ENVIRONMENT_VALUES.items()
        )
        or effective.get("PYTHONPYCACHEPREFIX")
        != str(Path(str(effective.get("TMPDIR"))) / "pycache")
        or sys.pycache_prefix
        != effective.get("PYTHONPYCACHEPREFIX")
    ):
        raise Dstc9OfficialHippoRAGError(
            "worker environment contract failed"
        )


def _validate_logical_cuda0(torch_module: object | None = None) -> None:
    """Prove that physical GPU0 is the sole visible logical ``cuda:0``."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_VISIBLE_DEVICES:
        raise Dstc9OfficialHippoRAGError("GPU0 visibility binding drifted")
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise Dstc9OfficialHippoRAGError(
                "CUDA runtime is unavailable"
            ) from exc
    cuda = getattr(torch_module, "cuda", None)
    try:
        available = cuda.is_available()
        count = cuda.device_count()
        cuda.set_device(0)
        current = cuda.current_device()
        sentinel = torch_module.empty(1, device=LOGICAL_CUDA_DEVICE)
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise Dstc9OfficialHippoRAGError(
            "logical cuda:0 attestation failed"
        ) from exc
    device = getattr(sentinel, "device", None)
    if (
        available is not True
        or type(count) is not int
        or count != 1
        or type(current) is not int
        or current != 0
        or str(device) != LOGICAL_CUDA_DEVICE
    ):
        raise Dstc9OfficialHippoRAGError(
            "logical cuda:0 attestation failed"
        )


def build_index_with_core(
    *,
    core: object,
    corpus_input: object,
    index_root: Path,
    runtime_attestation_receipt_sha256: str,
) -> dict[str, Any]:
    """Invoke official ``index`` exactly once for the frozen global corpus."""

    validated = validate_corpus_input(corpus_input)
    documents = serialize_corpus(validated.units)
    # The official embedding store addresses passages by MD5(text).
    corpus_text_multiplicity(documents)
    index = getattr(core, "index", None)
    if not callable(index):
        raise Dstc9OfficialHippoRAGError("official core lacks index")
    index(list(documents))
    return make_build_receipt(
        validated,
        index_snapshot=snapshot_index_tree(index_root),
        runtime_attestation_receipt_sha256=(
            runtime_attestation_receipt_sha256
        ),
    )


def retrieve_batches_with_core(
    *,
    core: object,
    corpus_input: object,
    query_input: object,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Retrieve fixed-size official results without any ``index`` call."""

    corpus = validate_corpus_input(corpus_input)
    queries = validate_query_input(
        query_input, expected_study_id=corpus.study_id
    )
    documents = serialize_corpus(corpus.units)
    query_texts = serialize_queries(queries.queries)
    document_to_ordinals: dict[str, list[int]] = {}
    for document, unit in zip(documents, corpus.units):
        document_to_ordinals.setdefault(document, []).append(unit.ordinal)
    retrieve = getattr(core, "retrieve", None)
    if not callable(retrieve):
        raise Dstc9OfficialHippoRAGError("official core lacks retrieve")

    results: list[tuple[int, ...]] = []
    batch_sizes: list[int] = []
    for offset in range(0, len(query_texts), MAX_QUERY_BATCH):
        batch = list(query_texts[offset : offset + MAX_QUERY_BATCH])
        if not 1 <= len(batch) <= MAX_QUERY_BATCH:
            raise Dstc9OfficialHippoRAGError(
                "internal query batch bound drifted"
            )
        rows = retrieve(batch, num_to_retrieve=CORPUS_SIZE)
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise Dstc9OfficialHippoRAGError(
                "official core returned an invalid query batch"
            )
        for solution in rows:
            results.append(
                stable_top_five_from_official_result(
                    retrieved_documents=getattr(solution, "docs", None),
                    retrieved_scores=getattr(solution, "doc_scores", None),
                    document_to_ordinals=document_to_ordinals,
                )
            )
        batch_sizes.append(len(batch))
    return tuple(results), tuple(batch_sizes)


def _load_json(path: Path, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Dstc9OfficialHippoRAGError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9OfficialHippoRAGError(f"{field} is invalid") from exc
    if raw != canonical_json_bytes(value) or not isinstance(value, dict):
        raise Dstc9OfficialHippoRAGError(f"{field} is not canonical JSON")
    return value


def _load_corpus_input(path: Path) -> CorpusInput:
    return validate_corpus_input(_load_json(path, "corpus input"))


def _load_query_input(path: Path, *, expected_study_id: str) -> QueryInput:
    return validate_query_input(
        _load_json(path, "query input"),
        expected_study_id=expected_study_id,
    )


def _load_and_validate_build_receipt(
    path: Path,
    *,
    corpus_input: CorpusInput,
    index_snapshot: IndexTreeSnapshot,
    runtime_attestation_receipt_sha256: str,
) -> dict[str, Any]:
    payload = _load_json(path, "build receipt")
    if payload.get("schema") != BUILD_RECEIPT_SCHEMA:
        raise Dstc9OfficialHippoRAGError("build receipt schema mismatch")
    return validate_build_receipt(
        payload,
        expected_corpus_input=corpus_input,
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
    corpus_count: int,
) -> object:
    """Construct the pinned local-only core after logical-GPU attestation."""

    if not isinstance(force_index_from_scratch, bool):
        raise Dstc9OfficialHippoRAGError(
            "force_index_from_scratch must be boolean"
        )
    if type(corpus_count) is not int or corpus_count != CORPUS_SIZE:
        raise Dstc9OfficialHippoRAGError(
            "worker corpus count must remain exactly 2900"
        )
    _validate_logical_cuda0()
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name="Transformers/" + str(llm_model),
        embedding_model_name="Transformers/" + str(embedding_model),
        openie_mode=FROZEN_CORE_CONFIG["openie_mode"],
        max_new_tokens=FROZEN_CORE_CONFIG["max_new_tokens"],
        max_retry_attempts=FROZEN_CORE_CONFIG["max_retry_attempts"],
        retrieval_top_k=CORPUS_SIZE,
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
        raise Dstc9OfficialHippoRAGError(
            "global index root must not exist before build"
        )
    corpus = _load_corpus_input(arguments.corpus_input)
    corpus_text_multiplicity(serialize_corpus(corpus.units))
    arguments.index_root.mkdir(mode=0o700)
    core = _build_official_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
        force_index_from_scratch=True,
        corpus_count=len(corpus.units),
    )
    receipt = build_index_with_core(
        core=core,
        corpus_input=corpus_input_projection(corpus),
        index_root=arguments.index_root,
        runtime_attestation_receipt_sha256=(
            arguments.runtime_binding_receipt_sha256
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
        raise Dstc9OfficialHippoRAGError(
            "global index root is unavailable for reopen"
        )
    if arguments.query_input is None or arguments.build_receipt is None:
        raise Dstc9OfficialHippoRAGError(
            "retrieve requires queries and the build receipt"
        )
    corpus = _load_corpus_input(arguments.corpus_input)
    before = snapshot_index_tree(arguments.index_root)
    build_receipt = _load_and_validate_build_receipt(
        arguments.build_receipt,
        corpus_input=corpus,
        index_snapshot=before,
        runtime_attestation_receipt_sha256=(
            arguments.runtime_binding_receipt_sha256
        ),
    )
    queries = _load_query_input(
        arguments.query_input, expected_study_id=corpus.study_id
    )
    core = _build_official_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
        force_index_from_scratch=False,
        corpus_count=CORPUS_SIZE,
    )
    indices, batch_sizes = retrieve_batches_with_core(
        core=core,
        corpus_input=corpus_input_projection(corpus),
        query_input=query_input_projection(queries),
    )
    after = snapshot_index_tree(arguments.index_root)
    receipt = make_retrieval_receipt(
        corpus_input=corpus,
        query_input=queries,
        indices=indices,
        batch_sizes=batch_sizes,
        build_receipt=build_receipt,
        index_snapshot_before=before,
        index_snapshot_after=after,
    )
    _write_exclusive_json(
        arguments.output,
        {
            "receipt": receipt,
            "retrieved_ordinals": [list(row) for row in indices],
            "schema": RETRIEVAL_OUTPUT_SCHEMA,
        },
    )
    return {
        "batch_count": len(batch_sizes),
        "index_call_count": 0,
        "query_count": len(queries.queries),
        "stage": "retrieve",
        "status": "passed",
    }


def main(argv: Sequence[str] | None = None) -> int:
    _validate_effective_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "retrieve"), required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--p17-project-root", required=True, type=Path)
    parser.add_argument("--worker-project-root", required=True, type=Path)
    parser.add_argument(
        "--current-hardware-binding", required=True, type=Path
    )
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument("--runtime-fingerprint", required=True, type=Path)
    parser.add_argument("--runtime-binding-receipt", required=True, type=Path)
    parser.add_argument("--corpus-input", required=True, type=Path)
    parser.add_argument("--query-input", type=Path)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    parser.add_argument(
        "--runtime-binding-receipt-sha256", required=True
    )
    arguments = parser.parse_args(argv)
    try:
        verify_worker_runtime_provenance(
            binding_receipt_path=arguments.runtime_binding_receipt,
            binding_receipt_file_sha256=(
                arguments.runtime_binding_receipt_sha256
            ),
            p17_project_root=arguments.p17_project_root,
            worker_project_root=arguments.worker_project_root,
            current_hardware_binding_path=(
                arguments.current_hardware_binding
            ),
            expected_study_id=arguments.study_id,
            runtime_fingerprint_path=arguments.runtime_fingerprint,
            runtime_python=arguments.runtime_python,
            local_llm_model=arguments.llm_model,
            local_embedding_model=arguments.embedding_model,
        )
    except Dstc9P17RuntimeBindingError as exc:
        raise Dstc9OfficialHippoRAGError(
            "worker P17 closure/current hardware provenance failed"
        ) from exc
    status = (
        _run_build(arguments)
        if arguments.stage == "build"
        else _run_retrieve(arguments)
    )
    print(
        json.dumps(
            status,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_build_official_core",
    "_validate_effective_environment",
    "_validate_logical_cuda0",
    "build_index_with_core",
    "retrieve_batches_with_core",
]

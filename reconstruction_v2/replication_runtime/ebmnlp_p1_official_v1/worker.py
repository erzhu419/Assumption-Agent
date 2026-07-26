"""Private offline worker for one EBM-NLP abstract and three role queries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from replication_runtime.maud_extraction_p2_official_v1 import (
    worker as _qualified_base,
)

from . import contract


OPENIE_MAX_NEW_TOKENS = 96
TORCH_THREAD_COUNT = 1
NATIVE_THREAD_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MAX_MODEL_ALIAS_CHARACTERS = 64
_MODEL_ALIAS = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_MODEL_ALIAS_CHARACTERS - 1}}}\Z"
)


def _require_qualified_base() -> None:
    if (
        _qualified_base.OPENIE_MAX_NEW_TOKENS != OPENIE_MAX_NEW_TOKENS
        or _qualified_base.TORCH_THREAD_COUNT != TORCH_THREAD_COUNT
        or not callable(_qualified_base._install_completion_only_backend)
        or not callable(_qualified_base._install_single_worker_openie_executor)
    ):
        raise contract.EBMNLPOfficialHippoRAGError(
            "qualified official-core compatibility boundary drifted"
        )


def _require_native_thread_environment() -> None:
    if (
        any(os.environ.get(key) != "1" for key in NATIVE_THREAD_KEYS)
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != CUBLAS_WORKSPACE_CONFIG
    ):
        raise contract.EBMNLPOfficialHippoRAGError(
            "native thread environment drifted"
        )


def _require_project_origins(project_root: Path) -> None:
    try:
        root = project_root.resolve(strict=True)
    except OSError as exc:
        raise contract.EBMNLPOfficialHippoRAGError(
            "project root is unavailable"
        ) from exc
    expected = {
        "replication_runtime": "replication_runtime/__init__.py",
        "replication_runtime.ebmnlp_p1_official_v1": (
            "replication_runtime/ebmnlp_p1_official_v1/__init__.py"
        ),
        contract.__name__: (
            "replication_runtime/ebmnlp_p1_official_v1/contract.py"
        ),
        __name__: (
            "replication_runtime/ebmnlp_p1_official_v1/worker.py"
        ),
        "replication_runtime.maud_extraction_p2_official_v1": (
            "replication_runtime/"
            "maud_extraction_p2_official_v1/__init__.py"
        ),
        _qualified_base.__name__: (
            "replication_runtime/"
            "maud_extraction_p2_official_v1/worker.py"
        ),
    }
    for module_name, relative in expected.items():
        module = sys.modules.get(module_name)
        origin = getattr(module, "__file__", None)
        if (
            not isinstance(origin, str)
            or Path(origin).resolve(strict=True)
            != (root / relative).resolve(strict=True)
        ):
            raise contract.EBMNLPOfficialHippoRAGError(
                "project module import origin drifted"
            )


def _validate_model_alias(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or _MODEL_ALIAS.fullmatch(value) is None
        or "/" in value
        or "\\" in value
        or ".." in value
        or Path(value).is_absolute()
    ):
        raise contract.EBMNLPOfficialHippoRAGError(
            f"{label} alias is invalid"
        )
    try:
        available = Path(value).is_dir()
    except OSError:
        available = False
    if not available:
        raise contract.EBMNLPOfficialHippoRAGError(
            f"{label} alias is unavailable"
        )
    return value


def _build_core(
    *,
    save_dir: Path,
    llm_model: str,
    embedding_model: str,
    document_count: int,
    hipporag_source_root: Path,
) -> object:
    if (
        type(document_count) is not int
        or not contract.MIN_DOCUMENT_COUNT
        <= document_count
        <= contract.MAX_DOCUMENT_COUNT
    ):
        raise contract.EBMNLPOfficialHippoRAGError(
            "document count is invalid"
        )
    _require_qualified_base()
    _require_native_thread_environment()

    import torch
    import hipporag as hipporag_package
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    try:
        expected_package_root = (
            hipporag_source_root.resolve(strict=True) / "hipporag"
        ).resolve(strict=True)
        observed_origin = Path(
            str(hipporag_package.__file__)
        ).resolve(strict=True)
        observed_origin.relative_to(expected_package_root)
    except (OSError, TypeError, ValueError) as exc:
        raise contract.EBMNLPOfficialHippoRAGError(
            "HippoRAG import origin drifted"
        ) from exc

    torch.set_num_threads(TORCH_THREAD_COUNT)
    try:
        torch.set_num_interop_threads(TORCH_THREAD_COUNT)
    except RuntimeError as exc:
        raise contract.EBMNLPOfficialHippoRAGError(
            "torch interop thread state drifted"
        ) from exc
    torch.manual_seed(0)
    _qualified_base._install_single_worker_openie_executor()
    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name="Transformers/" + llm_model,
        embedding_model_name="Transformers/" + embedding_model,
        openie_mode="online",
        max_new_tokens=OPENIE_MAX_NEW_TOKENS,
        retrieval_top_k=document_count,
        qa_top_k=min(5, document_count),
        force_index_from_scratch=True,
        save_openie=True,
        temperature=0,
        seed=0,
    )
    core = HippoRAG(global_config=config)
    try:
        _qualified_base._install_completion_only_backend(core)
    except BaseException as exc:
        raise contract.EBMNLPOfficialHippoRAGError(
            "qualified local LLM compatibility boundary failed"
        ) from exc
    return core


def _load_input(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise contract.EBMNLPOfficialHippoRAGError(
            "worker input is unavailable"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise contract.EBMNLPOfficialHippoRAGError(
            "worker input is invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or contract.canonical_json_bytes(value) != raw
    ):
        raise contract.EBMNLPOfficialHippoRAGError(
            "worker input is not canonical"
        )
    contract.validate_input(value)
    return dict(value)


def _write_output(path: Path, payload: Mapping[str, object]) -> None:
    raw = contract.canonical_json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_once(
    *,
    payload: Mapping[str, object],
    output_path: Path,
    index_root: Path,
    llm_model: str,
    embedding_model: str,
    hipporag_source_root: Path,
) -> dict[str, object]:
    if index_root.exists() or index_root.is_symlink():
        raise contract.EBMNLPOfficialHippoRAGError(
            "index root already exists"
        )
    (
        _abstract_work_id,
        _corpus_hash,
        documents,
        _queries,
    ) = contract.validate_input(payload)
    index_root.mkdir(mode=0o700)
    core = _build_core(
        save_dir=index_root,
        llm_model=llm_model,
        embedding_model=embedding_model,
        document_count=len(documents),
        hipporag_source_root=hipporag_source_root,
    )
    result = contract.retrieve_abstract_with_core(
        core=core, payload=payload
    )
    _write_output(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument(
        "--hipporag-source-root", required=True, type=Path
    )
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    llm_model = _validate_model_alias(
        arguments.llm_model, "LLM model"
    )
    embedding_model = _validate_model_alias(
        arguments.embedding_model, "embedding model"
    )
    _require_project_origins(arguments.project_root)
    payload = _load_input(arguments.input)
    result = run_once(
        payload=payload,
        output_path=arguments.output,
        index_root=arguments.index_root,
        llm_model=llm_model,
        embedding_model=embedding_model,
        hipporag_source_root=arguments.hipporag_source_root,
    )
    print(
        json.dumps(
            {
                "document_count": result["document_count"],
                "graph_edge_count": result["graph_edge_count"],
                "graph_node_count": result["graph_node_count"],
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

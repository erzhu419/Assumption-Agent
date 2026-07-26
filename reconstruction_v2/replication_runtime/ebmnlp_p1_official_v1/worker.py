"""Private offline worker for one EBM-NLP abstract and three role queries."""

from __future__ import annotations

import argparse
import hashlib
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
CUDA_RUNTIME_RECEIPT_SCHEMA = (
    "ebmnlp_p1_official_hipporag_worker_v2_private_cuda_receipt"
)
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


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise contract.EBMNLPOfficialHippoRAGError(
            "worker receipt self hash was supplied twice"
        )
    value = dict(body)
    value["self_sha256"] = hashlib.sha256(
        contract.canonical_json_bytes(value, newline=False)
    ).hexdigest()
    return value


def _module_cuda_residency(
    module: object,
    *,
    label: str,
    require_hf_device_map: bool,
) -> dict[str, object]:
    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise contract.EBMNLPOfficialHippoRAGError(
            f"{label} parameter registry is unavailable"
        )
    try:
        parameters = tuple(named_parameters())
    except BaseException as exc:
        raise contract.EBMNLPOfficialHippoRAGError(
            f"{label} parameter registry failed"
        ) from exc
    if (
        not parameters
        or len({str(name) for name, _parameter in parameters})
        != len(parameters)
    ):
        raise contract.EBMNLPOfficialHippoRAGError(
            f"{label} parameter registry drifted"
        )
    dtype_counts: dict[str, int] = {}
    parameter_numel = 0
    for name, parameter in parameters:
        device = getattr(parameter, "device", None)
        dtype = getattr(parameter, "dtype", None)
        numel = getattr(parameter, "numel", None)
        if (
            not isinstance(name, str)
            or not name
            or getattr(device, "type", None) != "cuda"
            or getattr(device, "index", None) != 0
            or not callable(numel)
        ):
            raise contract.EBMNLPOfficialHippoRAGError(
                f"{label} parameter is not resident on logical cuda:0"
            )
        try:
            count = int(numel())
        except BaseException as exc:
            raise contract.EBMNLPOfficialHippoRAGError(
                f"{label} parameter size is unavailable"
            ) from exc
        if count <= 0:
            raise contract.EBMNLPOfficialHippoRAGError(
                f"{label} parameter size drifted"
            )
        parameter_numel += count
        dtype_name = str(dtype)
        if not dtype_name.startswith("torch."):
            raise contract.EBMNLPOfficialHippoRAGError(
                f"{label} parameter dtype drifted"
            )
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

    raw_device_map = getattr(module, "hf_device_map", None)
    if raw_device_map is None:
        if require_hf_device_map:
            raise contract.EBMNLPOfficialHippoRAGError(
                f"{label} Hugging Face device map is unavailable"
            )
        device_map_present = False
        device_map_entry_count = 0
    else:
        if not isinstance(raw_device_map, Mapping) or not raw_device_map:
            raise contract.EBMNLPOfficialHippoRAGError(
                f"{label} Hugging Face device map drifted"
            )
        normalized = tuple(str(value) for value in raw_device_map.values())
        if any(value not in {"0", "cuda", "cuda:0"} for value in normalized):
            raise contract.EBMNLPOfficialHippoRAGError(
                f"{label} contains CPU disk or nonzero-GPU offload"
            )
        device_map_present = True
        device_map_entry_count = len(normalized)
    return {
        "parameter_count": len(parameters),
        "parameter_numel": parameter_numel,
        "parameter_dtype_counts": dict(sorted(dtype_counts.items())),
        "parameter_device": "cuda:0",
        "hf_device_map_present": device_map_present,
        "hf_device_map_entry_count": device_map_entry_count,
        "cpu_disk_or_nonzero_gpu_offload_count": 0,
    }


def _attest_cuda_residency(
    core: object, *, torch_module: object | None = None
) -> dict[str, object]:
    if torch_module is None:
        import torch as torch_module

    cuda = getattr(torch_module, "cuda", None)
    if (
        cuda is None
        or not callable(getattr(cuda, "is_available", None))
        or cuda.is_available() is not True
        or not callable(getattr(cuda, "device_count", None))
        or cuda.device_count() != 1
        or not callable(getattr(cuda, "current_device", None))
        or cuda.current_device() != 0
    ):
        raise contract.EBMNLPOfficialHippoRAGError(
            "exactly one visible logical CUDA device is required"
        )
    visible_binding = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_binding not in {"0", "1"}:
        raise contract.EBMNLPOfficialHippoRAGError(
            "physical CUDA lane binding drifted"
        )
    try:
        sentinel = torch_module.ones(
            (1,), dtype=torch_module.float32, device="cuda:0"
        )
        sentinel.add_(1.0)
        if (
            float(sentinel.item()) != 2.0
            or getattr(sentinel.device, "type", None) != "cuda"
            or getattr(sentinel.device, "index", None) != 0
        ):
            raise RuntimeError("CUDA allocation value or device drifted")
        cuda.synchronize(0)
        device_name = str(cuda.get_device_name(0))
        memory_allocated = int(cuda.memory_allocated(0))
    except BaseException as exc:
        raise contract.EBMNLPOfficialHippoRAGError(
            "logical CUDA allocation or synchronization failed"
        ) from exc
    if not device_name or memory_allocated <= 0:
        raise contract.EBMNLPOfficialHippoRAGError(
            "logical CUDA device identity or allocation drifted"
        )
    llm_wrapper = getattr(core, "llm_model", None)
    embedding_wrapper = getattr(core, "embedding_model", None)
    llm_model = getattr(llm_wrapper, "model", None)
    embedding_model = getattr(embedding_wrapper, "model", None)
    if llm_model is None or embedding_model is None:
        raise contract.EBMNLPOfficialHippoRAGError(
            "official HippoRAG model boundary drifted"
        )
    return {
        "torch_cuda_is_available": True,
        "visible_cuda_device_count": 1,
        "logical_cuda_current_device": 0,
        "physical_visible_gpu_binding": visible_binding,
        "cuda_device_name_sha256": hashlib.sha256(
            device_name.encode("utf-8")
        ).hexdigest(),
        "cuda_allocation_and_synchronize_succeeded": True,
        "cuda_memory_allocated_bytes": memory_allocated,
        "LLM": _module_cuda_residency(
            llm_model,
            label="LLM",
            require_hf_device_map=True,
        ),
        "embedding": _module_cuda_residency(
            embedding_model,
            label="embedding",
            require_hf_device_map=False,
        ),
    }


def run_once(
    *,
    payload: Mapping[str, object],
    output_path: Path,
    runtime_receipt_path: Path,
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
    pre_inference = _attest_cuda_residency(core)
    result = contract.retrieve_abstract_with_core(
        core=core, payload=payload
    )
    _write_output(output_path, result)
    post_inference = _attest_cuda_residency(core)
    _write_output(
        runtime_receipt_path,
        _self_hashed(
            {
                "schema": CUDA_RUNTIME_RECEIPT_SCHEMA,
                "status": (
                    "complete_output_and_pre_post_inference_cuda_"
                    "residency_attested"
                ),
                "input_sha256": hashlib.sha256(
                    contract.canonical_json_bytes(payload)
                ).hexdigest(),
                "output_file_sha256": hashlib.sha256(
                    contract.canonical_json_bytes(result)
                ).hexdigest(),
                "pre_inference": pre_inference,
                "post_inference": post_inference,
            }
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-receipt", required=True, type=Path)
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
        runtime_receipt_path=arguments.runtime_receipt,
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

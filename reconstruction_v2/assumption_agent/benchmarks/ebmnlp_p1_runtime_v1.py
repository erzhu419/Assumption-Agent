"""Concrete offline runtime for the frozen EBM-NLP P1 study.

MiniLM runs once in the controller process on visible ``cuda:0``.  Official
HippoRAG runs in two stable lanes, one subprocess per physical GPU, with every
network syscall denied and audited by ``strace``.  Each abstract gets one
fresh index shared by its three frozen role queries; the index is destroyed
after its complete rank permutations have been verified.

This module contains no API/provider path, no retry/recovery path, and no
online evaluator.  Source-free fingerprint and canary entrypoints do not
accept or open an EBM-NLP archive.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    ebmnlp_p1_formal_controller_v1 as formal,
)
from assumption_agent.benchmarks import (
    ebmnlp_p1_source_qualification_v1 as source,
)
from assumption_agent.benchmarks import (
    ebmnlp_p1_typed_pico_core_v1 as core,
)
from replication_runtime.ebmnlp_p1_official_v1 import contract as hippo
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


VERSION = "ebmnlp_p1_runtime_v1"
EMBEDDER_RECEIPT_SCHEMA = (
    "ebmnlp_p1_local_minilm_embedder_v1_safe_runtime_receipt"
)
HIPPO_RECEIPT_SCHEMA = (
    "ebmnlp_p1_official_hipporag_batch_v1_safe_runtime_receipt"
)
FINGERPRINT_SCHEMA = f"{VERSION}_source_free_fingerprint"
CANARY_SCHEMA = f"{VERSION}_source_free_full_path_canary"
IMPLEMENTATION_FREEZE_SCHEMA = "ebmnlp_p1_implementation_freeze_v1"
IMPLEMENTATION_FREEZE_STATUS = (
    "implementation_frozen_before_runtime_fingerprint_canary_or_source_access"
)
EXECUTION_FREEZE_SCHEMA = "ebmnlp_p1_execution_freeze_v1"
EXECUTION_FREEZE_STATUS = (
    "execution_frozen_after_source_free_canary_before_source_access"
)
LIVE_EXECUTION_ATTESTATION_SCHEMA = (
    f"{VERSION}_live_formal_execution_attestation"
)
HIPPO_WORKER_MODULE = (
    "replication_runtime.ebmnlp_p1_official_v1.worker"
)

MINILM_GENERIC_TREE_SHA256 = (
    "42d8d798e4f01e68d9bb10634b9c712de00f7f8495271636fd6311b2db58e506"
)
MINILM_WEIGHTS_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
MINILM_ASSET_FILE_SHA256 = minilm_binding.ASSET_FILE_SHA256
MINILM_ASSET_SELF_SHA256 = minilm_binding.ASSET_SELF_SHA256
HIPPORAG_LLM_TREE_SHA256 = formal.EXPECTED_HIPPORAG_LLM_TREE_SHA256
HIPPORAG_SOURCE_TREE_SHA256 = formal.EXPECTED_HIPPORAG_SOURCE_TREE_SHA256
MINILM_NORMATIVE_TREE_SHA256 = formal.EXPECTED_MINILM_TREE_SHA256

EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_DEVICE = "cuda:0"
GPU_ASSIGNMENT = ("0", "1")
MAXIMUM_HIPPO_PROCESSES = 2
CPU_THREADS_PER_HIPPO_PROCESS = 1
MAXIMUM_WORKER_OUTPUT_BYTES = 16 * 1024 * 1024
MAXIMUM_NETWORK_AUDIT_BYTES = 64 * 1024 * 1024
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FORMAL_CLEAN_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
    "CUDA_VISIBLE_DEVICES": "0,1",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/home/erzhu419",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "XDG_RUNTIME_DIR": "/run/user/1001",
}

_REQUIRED_IMPLEMENTATION_PATHS = frozenset(
    {
        "assumption_agent/__init__.py",
        "assumption_agent/benchmarks/__init__.py",
        "assumption_agent/benchmarks/ebmnlp_p1_formal_controller_v1.py",
        "assumption_agent/benchmarks/ebmnlp_p1_runtime_v1.py",
        "assumption_agent/benchmarks/ebmnlp_p1_source_qualification_v1.py",
        "assumption_agent/benchmarks/ebmnlp_p1_typed_pico_core_v1.py",
        "manifests/ebmnlp_p1_implementation_clarification_v1.json",
        "manifests/ebmnlp_p1_pre_source_clarification_v1.json",
        "manifests/ebmnlp_p1_source_custody_v1.json",
        "manifests/ebmnlp_p1_typed_pico_set_evaluator_study_design_v1.json",
        "manifests/qasper_minilm_runtime_asset_v1.json",
        "replication_runtime/__init__.py",
        "replication_runtime/ebmnlp_p1_official_v1/__init__.py",
        "replication_runtime/ebmnlp_p1_official_v1/contract.py",
        "replication_runtime/ebmnlp_p1_official_v1/worker.py",
        "replication_runtime/maud_extraction_p2_official_v1/__init__.py",
        "replication_runtime/maud_extraction_p2_official_v1/worker.py",
        "replication_runtime/qasper_minilm_v1/__init__.py",
        "replication_runtime/qasper_minilm_v1/binding.py",
    }
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_NETWORK_CALL = re.compile(
    r"^(?:\[pid\s+\d+\]\s+)?[A-Za-z_][A-Za-z0-9_]*\("
)
_NETWORK_RESUMED = re.compile(
    r"^(?:\[pid\s+\d+\]\s+)?<\.\.\.\s+"
    r"[A-Za-z_][A-Za-z0-9_]*\s+resumed>"
)


class EbmNlpP1RuntimeError(RuntimeError):
    """The frozen asset, process, network, or concurrency boundary drifted."""


def _install_cuda_determinism_environment() -> None:
    """Bind cuBLAS determinism before PyTorch can initialize CUDA."""

    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise EbmNlpP1RuntimeError(
            "cuBLAS deterministic workspace policy drifted"
        )
    if observed is None and "torch" in sys.modules:
        raise EbmNlpP1RuntimeError(
            "cuBLAS deterministic workspace was bound after PyTorch import"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


def canonical_json_bytes(
    value: object, *, newline: bool = True
) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EbmNlpP1RuntimeError(
            "runtime value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, newline=False)
    ).hexdigest()


def self_hashed(
    body: Mapping[str, Any], field: str = "self_sha256"
) -> dict[str, Any]:
    if field in body:
        raise EbmNlpP1RuntimeError("self-hash field was supplied twice")
    value = dict(body)
    value[field] = stable_hash(value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise EbmNlpP1RuntimeError("bound file is unavailable") from exc
    try:
        metadata_row = os.fstat(descriptor)
        if not stat.S_ISREG(metadata_row.st_mode):
            raise EbmNlpP1RuntimeError("bound file is not regular")
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def tree_receipt(root: Path) -> dict[str, object]:
    """Return the P17-compatible generic tree receipt."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise EbmNlpP1RuntimeError("frozen tree is unavailable")
    rows: list[dict[str, object]] = []
    total = 0
    for current, directories, files in os.walk(
        root, followlinks=False
    ):
        base = Path(current)
        for name in directories:
            if (base / name).is_symlink():
                raise EbmNlpP1RuntimeError(
                    "frozen tree contains a directory symlink"
                )
        for name in files:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise EbmNlpP1RuntimeError(
                    "frozen tree contains a non-file"
                )
            size = path.stat().st_size
            total += size
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": size,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "file_count": len(rows),
        "size_bytes": total,
        "tree_sha256": stable_hash(rows),
    }


def _verify_minilm_asset(
    *, asset_manifest_path: Path, model_root: Path
) -> dict[str, object]:
    """Verify the immutable snapshot without imposing another study's venv."""

    manifest = _checked_file(
        asset_manifest_path, "MiniLM asset manifest"
    )
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MINILM_ASSET_FILE_SHA256:
        raise EbmNlpP1RuntimeError(
            "MiniLM asset manifest file hash drifted"
        )
    try:
        asset = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EbmNlpP1RuntimeError(
            "MiniLM asset manifest is invalid"
        ) from exc
    if not isinstance(asset, dict):
        raise EbmNlpP1RuntimeError(
            "MiniLM asset manifest shape drifted"
        )
    body = dict(asset)
    declared = body.pop("asset_sha256", None)
    local = asset.get("local_binding")
    model = asset.get("model")
    if (
        declared != MINILM_ASSET_SELF_SHA256
        or stable_hash(body) != declared
        or not isinstance(local, Mapping)
        or not isinstance(model, Mapping)
        or local.get("snapshot_tree_sha256")
        != MINILM_NORMATIVE_TREE_SHA256
        or model.get("weights_sha256") != MINILM_WEIGHTS_SHA256
    ):
        raise EbmNlpP1RuntimeError(
            "MiniLM asset manifest binding drifted"
        )
    rows = local.get("snapshot_files")
    if not isinstance(rows, list) or len(rows) != 11:
        raise EbmNlpP1RuntimeError(
            "MiniLM asset file registry drifted"
        )
    root = _checked_directory(model_root, "MiniLM model")
    observed: list[dict[str, object]] = []
    expected_paths: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise EbmNlpP1RuntimeError(
                "MiniLM asset file row drifted"
            )
        relative_text = row.get("path")
        size = row.get("size")
        digest = row.get("sha256")
        if (
            not isinstance(relative_text, str)
            or not relative_text
            or Path(relative_text).is_absolute()
            or ".." in Path(relative_text).parts
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
        ):
            raise EbmNlpP1RuntimeError(
                "MiniLM asset file row is invalid"
            )
        path = root / relative_text
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or file_sha256(path) != digest
        ):
            raise EbmNlpP1RuntimeError(
                "MiniLM asset file content drifted"
            )
        expected_paths.append(relative_text)
        observed.append(
            {
                "path": relative_text,
                "sha256": digest,
                "size": size,
            }
        )
    live = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    if (
        live != sorted(expected_paths)
        or stable_hash(observed) != MINILM_NORMATIVE_TREE_SHA256
    ):
        raise EbmNlpP1RuntimeError(
            "MiniLM snapshot population drifted"
        )
    return {
        "asset_file_sha256": MINILM_ASSET_FILE_SHA256,
        "asset_self_sha256": MINILM_ASSET_SELF_SHA256,
        "model_tree_sha256": MINILM_NORMATIVE_TREE_SHA256,
        "weights_sha256": MINILM_WEIGHTS_SHA256,
    }


def _private_directory(
    path: Path, *, fresh: bool = False
) -> Path:
    path = Path(path)
    try:
        if fresh:
            path.mkdir(mode=0o700)
        else:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.lstat()
    except OSError as exc:
        raise EbmNlpP1RuntimeError(
            "private runtime directory is unavailable"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise EbmNlpP1RuntimeError(
            "private runtime directory is unsafe"
        )
    os.chmod(path, 0o700)
    return path


def _write_exclusive(path: Path, raw: bytes) -> str:
    _private_directory(path.parent)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise EbmNlpP1RuntimeError(
            "one-shot runtime artifact is already consumed"
        ) from exc
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _read_regular(
    path: Path,
    *,
    maximum_bytes: int,
    expected_mode: int | None = None,
) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise EbmNlpP1RuntimeError(
            "runtime artifact is unavailable"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > maximum_bytes
            or (
                expected_mode is not None
                and stat.S_IMODE(info.st_mode) != expected_mode
            )
        ):
            raise EbmNlpP1RuntimeError(
                "runtime artifact metadata drifted"
            )
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise EbmNlpP1RuntimeError(
                    "runtime artifact was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _read_proc_regular(path: Path, *, maximum_bytes: int) -> bytes:
    """Read a bounded procfs pseudo-file whose reported size is zero."""

    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise EbmNlpP1RuntimeError(
            "live process artifact is unavailable"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise EbmNlpP1RuntimeError(
                "live process artifact metadata drifted"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(
                descriptor, min(4096, maximum_bytes + 1 - observed)
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise EbmNlpP1RuntimeError(
                    "live process artifact byte bound exceeded"
                )
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _checked_directory(path: Path, label: str) -> Path:
    value = Path(path).expanduser().absolute()
    try:
        info = value.lstat()
    except OSError as exc:
        raise EbmNlpP1RuntimeError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise EbmNlpP1RuntimeError(f"{label} is not a real directory")
    return value


def _checked_file(path: Path, label: str) -> Path:
    value = Path(path).expanduser().absolute()
    try:
        info = value.lstat()
    except OSError as exc:
        raise EbmNlpP1RuntimeError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise EbmNlpP1RuntimeError(f"{label} is not a regular file")
    return value


def _checked_executable(
    path: Path, label: str, *, allow_leaf_symlink: bool = False
) -> Path:
    value = Path(path).expanduser().absolute()
    try:
        info = value.lstat()
        target = value.resolve(strict=True)
    except OSError as exc:
        raise EbmNlpP1RuntimeError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) and not allow_leaf_symlink:
        raise EbmNlpP1RuntimeError(f"{label} may not be a symlink")
    if (
        not target.is_file()
        or not os.access(value, os.X_OK)
    ):
        raise EbmNlpP1RuntimeError(f"{label} is not executable")
    return value


def _load_self_hashed_json(
    path: Path,
    *,
    label: str,
    expected_self_sha256: str,
    expected_schema: str,
    expected_status: str,
    canonical_file: bool,
) -> dict[str, object]:
    """Read and verify a semantic self-hashed prerequisite."""

    if _HEX64.fullmatch(expected_self_sha256) is None:
        raise EbmNlpP1RuntimeError(f"{label} expected hash drifted")
    raw = _read_regular(
        _checked_file(path, label),
        maximum_bytes=16 * 1024 * 1024,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EbmNlpP1RuntimeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise EbmNlpP1RuntimeError(f"{label} shape drifted")
    if canonical_file and canonical_json_bytes(value) != raw:
        raise EbmNlpP1RuntimeError(f"{label} is not canonical JSON")
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        declared != expected_self_sha256
        or stable_hash(body) != declared
        or value.get("schema") != expected_schema
        or value.get("status") != expected_status
        or value.get("study_id") != core.STUDY_ID
    ):
        raise EbmNlpP1RuntimeError(f"{label} binding drifted")
    return value


def _implementation_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise EbmNlpP1RuntimeError(
            "implementation binding path drifted"
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise EbmNlpP1RuntimeError(
            "implementation binding path is unsafe"
        )
    return relative


def _verify_live_module_origins(root: Path) -> None:
    """Reject a shadow import even when another tree matches the manifest."""

    live_modules = {
        "assumption_agent": "assumption_agent/__init__.py",
        "assumption_agent.benchmarks": (
            "assumption_agent/benchmarks/__init__.py"
        ),
        formal.__name__: (
            "assumption_agent/benchmarks/"
            "ebmnlp_p1_formal_controller_v1.py"
        ),
        source.__name__: (
            "assumption_agent/benchmarks/"
            "ebmnlp_p1_source_qualification_v1.py"
        ),
        core.__name__: (
            "assumption_agent/benchmarks/"
            "ebmnlp_p1_typed_pico_core_v1.py"
        ),
        __name__: (
            "assumption_agent/benchmarks/ebmnlp_p1_runtime_v1.py"
        ),
        "replication_runtime": "replication_runtime/__init__.py",
        "replication_runtime.ebmnlp_p1_official_v1": (
            "replication_runtime/ebmnlp_p1_official_v1/__init__.py"
        ),
        hippo.__name__: (
            "replication_runtime/ebmnlp_p1_official_v1/contract.py"
        ),
        "replication_runtime.qasper_minilm_v1": (
            "replication_runtime/qasper_minilm_v1/__init__.py"
        ),
        minilm_binding.__name__: (
            "replication_runtime/qasper_minilm_v1/binding.py"
        ),
    }
    for module_name, relative in live_modules.items():
        module = sys.modules.get(module_name)
        origin = getattr(module, "__file__", None)
        try:
            matches = (
                isinstance(origin, str)
                and Path(origin).resolve(strict=True)
                == (root / relative).resolve(strict=True)
            )
        except OSError:
            matches = False
        if not matches:
            raise EbmNlpP1RuntimeError(
                "live implementation module origin drifted"
            )


def verify_implementation_freeze(
    *,
    project_root: Path,
    manifest_path: Path,
    expected_self_sha256: str,
) -> dict[str, object]:
    """Verify every frozen study file before a model or source is opened."""

    root = _checked_directory(project_root, "project root")
    manifest = _load_self_hashed_json(
        manifest_path,
        label="implementation freeze manifest",
        expected_self_sha256=expected_self_sha256,
        expected_schema=IMPLEMENTATION_FREEZE_SCHEMA,
        expected_status=IMPLEMENTATION_FREEZE_STATUS,
        canonical_file=False,
    )
    rows = manifest.get("implementation_bindings")
    if not isinstance(rows, list) or not rows:
        raise EbmNlpP1RuntimeError(
            "implementation binding registry drifted"
        )
    observed_paths: set[str] = set()
    resolved_root = root.resolve(strict=True)
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise EbmNlpP1RuntimeError(
                "implementation binding row drifted"
            )
        relative = _implementation_relative_path(
            row["relative_path"]
        )
        relative_text = relative.as_posix()
        if relative_text in observed_paths:
            raise EbmNlpP1RuntimeError(
                "implementation binding path repeated"
            )
        expected_hash = row["sha256"]
        expected_size = row["size_bytes"]
        if (
            not isinstance(expected_hash, str)
            or _HEX64.fullmatch(expected_hash) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise EbmNlpP1RuntimeError(
                "implementation binding value drifted"
            )
        candidate = _checked_file(
            root.joinpath(*relative.parts),
            "frozen implementation file",
        )
        resolved_candidate = candidate.resolve(strict=True)
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise EbmNlpP1RuntimeError(
                "implementation file escaped the project root"
            ) from exc
        raw = _read_regular(
            candidate, maximum_bytes=16 * 1024 * 1024
        )
        if (
            len(raw) != expected_size
            or hashlib.sha256(raw).hexdigest() != expected_hash
        ):
            raise EbmNlpP1RuntimeError(
                "frozen implementation file drifted"
            )
        observed_paths.add(relative_text)
    if not _REQUIRED_IMPLEMENTATION_PATHS.issubset(
        observed_paths
    ):
        raise EbmNlpP1RuntimeError(
            "implementation freeze omitted a required file"
        )
    _verify_live_module_origins(root)
    return manifest


def _required_hash(config: Mapping[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise EbmNlpP1RuntimeError(
            f"runtime config hash {key} drifted"
        )
    return value


def _required_path(config: Mapping[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise EbmNlpP1RuntimeError(
            f"runtime config path {key} drifted"
        )
    return Path(value)


def _verify_fingerprint_receipt(
    config: Mapping[str, object],
    *,
    implementation_freeze_sha256: str,
    paths: "RuntimePaths",
) -> dict[str, object]:
    expected = _required_hash(config, "runtime_fingerprint_sha256")
    receipt = _load_self_hashed_json(
        _required_path(config, "runtime_fingerprint_receipt"),
        label="runtime fingerprint receipt",
        expected_self_sha256=expected,
        expected_schema=FINGERPRINT_SCHEMA,
        expected_status="verified_before_formal_source_open",
        canonical_file=True,
    )
    live = build_source_free_runtime_fingerprint(
        paths,
        implementation_freeze_sha256=implementation_freeze_sha256,
    )
    if receipt != live:
        raise EbmNlpP1RuntimeError(
            "runtime fingerprint prerequisite drifted"
        )
    return receipt


def _verify_canary_receipt(
    config: Mapping[str, object],
    *,
    implementation_freeze_sha256: str,
    runtime_fingerprint_sha256: str,
) -> dict[str, object]:
    expected = _required_hash(config, "source_free_canary_sha256")
    receipt = _load_self_hashed_json(
        _required_path(config, "source_free_canary_receipt"),
        label="source-free canary receipt",
        expected_self_sha256=expected,
        expected_schema=CANARY_SCHEMA,
        expected_status="passed_source_free_synthetic_full_path",
        canonical_file=True,
    )
    expected_top_keys = {
        "EBM_NLP_archive_path_or_member_access_count",
        "E0_selected_recipe_ids",
        "E1_selected_recipe_ids",
        "HippoRAG_complete_rank_permutation_count",
        "HippoRAG_runtime_receipt",
        "HippoRAG_runtime_receipt_sha256",
        "RAW_selected_recipe_ids",
        "embedding_runtime_receipt",
        "embedding_runtime_receipt_sha256",
        "external_network_call_count",
        "implementation_freeze_sha256",
        "online_or_api_evaluator_call_count",
        "provider_or_API_credential_read_count",
        "recipe_count",
        "role_count",
        "runtime_fingerprint_sha256",
        "schema",
        "self_sha256",
        "status",
        "study_id",
        "synthetic_canary_used_as_efficacy_gate",
        "window_count",
    }
    embedding = receipt.get("embedding_runtime_receipt")
    hippo_runtime = receipt.get("HippoRAG_runtime_receipt")
    selected_lists = (
        receipt.get("E0_selected_recipe_ids"),
        receipt.get("E1_selected_recipe_ids"),
        receipt.get("RAW_selected_recipe_ids"),
    )
    if (
        set(receipt) != expected_top_keys
        or
        receipt.get("implementation_freeze_sha256")
        != implementation_freeze_sha256
        or receipt.get("runtime_fingerprint_sha256")
        != runtime_fingerprint_sha256
        or receipt.get("window_count") != 5
        or receipt.get("role_count") != len(core.ROLE_ORDER)
        or receipt.get("recipe_count") != len(core.RECIPE_IDS)
        or any(
            not isinstance(values, list)
            or len(values) != len(core.ROLE_ORDER)
            or any(value not in core.RECIPE_IDS for value in values)
            for values in selected_lists
        )
        or receipt.get("HippoRAG_complete_rank_permutation_count")
        != 2 * len(core.ROLE_ORDER)
        or receipt.get("EBM_NLP_archive_path_or_member_access_count")
        != 0
        or receipt.get("provider_or_API_credential_read_count") != 0
        or receipt.get("external_network_call_count") != 0
        or receipt.get("online_or_api_evaluator_call_count") != 0
        or receipt.get("synthetic_canary_used_as_efficacy_gate")
        is not False
        or not isinstance(embedding, Mapping)
        or set(embedding)
        != {
            "CUBLAS_WORKSPACE_CONFIG",
            "batch_size",
            "call_count",
            "device",
            "dtype",
            "embedding_dimension",
            "encoded_text_count",
            "external_network_call_count",
            "maximum_sequence_length",
            "model_tree_sha256",
            "online_or_api_evaluator_call_count",
            "retry_or_replay_count",
            "schema",
            "status",
            "torch_deterministic_algorithms",
            "torch_manual_seed",
            "weights_sha256",
        }
        or embedding.get("schema") != EMBEDDER_RECEIPT_SCHEMA
        or embedding.get("status")
        != "complete_offline_deterministic_embeddings"
        or embedding.get("model_tree_sha256")
        != MINILM_NORMATIVE_TREE_SHA256
        or embedding.get("weights_sha256") != MINILM_WEIGHTS_SHA256
        or embedding.get("device") != EMBEDDING_DEVICE
        or embedding.get("dtype") != "float32"
        or embedding.get("embedding_dimension") != EMBEDDING_DIMENSION
        or embedding.get("batch_size") != EMBEDDING_BATCH_SIZE
        or embedding.get("maximum_sequence_length") != 256
        or embedding.get("CUBLAS_WORKSPACE_CONFIG")
        != CUBLAS_WORKSPACE_CONFIG
        or embedding.get("torch_deterministic_algorithms") is not True
        or embedding.get("torch_manual_seed") != 0
        or embedding.get("call_count") != 2
        or embedding.get("encoded_text_count") != 8
        or embedding.get("external_network_call_count") != 0
        or embedding.get("online_or_api_evaluator_call_count") != 0
        or embedding.get("retry_or_replay_count") != 0
        or receipt.get("embedding_runtime_receipt_sha256")
        != stable_hash(embedding)
        or not isinstance(hippo_runtime, Mapping)
        or set(hippo_runtime)
        != {
            "CUBLAS_WORKSPACE_CONFIG",
            "HippoRAG_import_origin_verified_worker_count",
            "HippoRAG_source_tree_sha256",
            "attempted_network_syscall_count",
            "batch_invocation_count",
            "configured_cpu_threads_per_process",
            "denied_network_syscall_count",
            "external_network_call_count",
            "gpu_assignment",
            "index_destroyed_count",
            "maximum_process_count",
            "maximum_processes_per_gpu",
            "observed_process_peak",
            "observed_process_peak_by_gpu",
            "online_or_api_evaluator_call_count",
            "retry_or_replay_count",
            "schema",
            "status",
            "worker_attempt_count",
            "worker_completed_count",
            "worker_completed_count_by_gpu",
        }
        or hippo_runtime.get("schema") != HIPPO_RECEIPT_SCHEMA
        or hippo_runtime.get("status")
        != "complete_offline_outputs_verified_indexes_destroyed"
        or hippo_runtime.get("gpu_assignment") != ["0", "1"]
        or hippo_runtime.get("CUBLAS_WORKSPACE_CONFIG")
        != CUBLAS_WORKSPACE_CONFIG
        or hippo_runtime.get("maximum_process_count") != 2
        or hippo_runtime.get("maximum_processes_per_gpu") != 1
        or hippo_runtime.get("configured_cpu_threads_per_process") != 1
        or hippo_runtime.get("observed_process_peak") != 2
        or hippo_runtime.get("observed_process_peak_by_gpu")
        != {"0": 1, "1": 1}
        or hippo_runtime.get("worker_attempt_count") != 2
        or hippo_runtime.get("worker_completed_count") != 2
        or hippo_runtime.get("worker_completed_count_by_gpu")
        != {"0": 1, "1": 1}
        or hippo_runtime.get("index_destroyed_count") != 2
        or hippo_runtime.get("batch_invocation_count") != 1
        or hippo_runtime.get("HippoRAG_source_tree_sha256")
        != HIPPORAG_SOURCE_TREE_SHA256
        or hippo_runtime.get(
            "HippoRAG_import_origin_verified_worker_count"
        )
        != 2
        or not isinstance(
            hippo_runtime.get("attempted_network_syscall_count"), int
        )
        or hippo_runtime.get("attempted_network_syscall_count") < 0
        or hippo_runtime.get("denied_network_syscall_count")
        != hippo_runtime.get("attempted_network_syscall_count")
        or hippo_runtime.get("external_network_call_count") != 0
        or hippo_runtime.get("online_or_api_evaluator_call_count") != 0
        or hippo_runtime.get("retry_or_replay_count") != 0
        or receipt.get("HippoRAG_runtime_receipt_sha256")
        != stable_hash(hippo_runtime)
    ):
        raise EbmNlpP1RuntimeError(
            "source-free canary prerequisite drifted"
        )
    return receipt


_FORMAL_SEMANTIC_CONFIG_KEYS = frozenset(
    {
        "archive_path",
        "env_executable",
        "formal_executor_runtime_root",
        "formal_hostname_sha256",
        "formal_unit_file",
        "formal_unit_file_sha256",
        "formal_unit_name",
        "formal_work_root",
        "hippo_embedding_model",
        "hippo_llm_model",
        "hippo_python",
        "hipporag_source",
        "implementation_freeze_manifest",
        "implementation_freeze_sha256",
        "live_attestation_output",
        "minilm_asset_manifest",
        "minilm_model",
        "project_root",
        "runtime_fingerprint_receipt",
        "runtime_fingerprint_sha256",
        "source_free_canary_receipt",
        "source_free_canary_sha256",
        "strace_executable",
        "systemctl_executable",
        "systemctl_executable_sha256",
    }
)


def formal_semantic_config_sha256(
    config: Mapping[str, object],
) -> str:
    if not _FORMAL_SEMANTIC_CONFIG_KEYS.issubset(config):
        raise EbmNlpP1RuntimeError(
            "formal semantic config is incomplete"
        )
    return stable_hash(
        {
            key: config[key]
            for key in sorted(_FORMAL_SEMANTIC_CONFIG_KEYS)
        }
    )


def _verify_execution_freeze(
    config: Mapping[str, object],
    *,
    implementation_freeze_sha256: str,
    runtime_fingerprint_sha256: str,
    source_free_canary_sha256: str,
) -> tuple[dict[str, object], str]:
    expected = _required_hash(config, "execution_freeze_sha256")
    receipt = _load_self_hashed_json(
        _required_path(config, "execution_freeze_manifest"),
        label="execution freeze manifest",
        expected_self_sha256=expected,
        expected_schema=EXECUTION_FREEZE_SCHEMA,
        expected_status=EXECUTION_FREEZE_STATUS,
        canonical_file=False,
    )
    semantic_hash = formal_semantic_config_sha256(config)
    declared_semantic_hash = _required_hash(
        config, "execution_config_sha256"
    )
    unit = receipt.get("formal_unit_contract")
    if (
        declared_semantic_hash != semantic_hash
        or receipt.get("implementation_freeze_sha256")
        != implementation_freeze_sha256
        or receipt.get("runtime_fingerprint_sha256")
        != runtime_fingerprint_sha256
        or receipt.get("source_free_canary_sha256")
        != source_free_canary_sha256
        or receipt.get("execution_config_sha256") != semantic_hash
        or receipt.get("formal_hostname_sha256")
        != config.get("formal_hostname_sha256")
        or receipt.get("formal_unit_file_sha256")
        != config.get("formal_unit_file_sha256")
        or receipt.get("systemctl_executable_sha256")
        != config.get("systemctl_executable_sha256")
        or not isinstance(unit, Mapping)
        or unit.get("unit") != config.get("formal_unit_name")
        or unit.get("CPUQuota_percent") != 800
        or unit.get("MemoryMax_bytes") != 40 * 1024**3
        or unit.get("TasksMax") != 64
        or unit.get("Restart") != "no"
        or unit.get("KillMode") != "control-group"
        or unit.get("GPU_assignment") != ["0", "1"]
        or unit.get("CUBLAS_WORKSPACE_CONFIG")
        != CUBLAS_WORKSPACE_CONFIG
        or unit.get("external_network") != "denied"
    ):
        raise EbmNlpP1RuntimeError(
            "execution freeze prerequisite drifted"
        )
    return receipt, semantic_hash


def _systemd_timespan_microseconds(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)(us|ms|s|min|h)", value)
    if match is None:
        raise EbmNlpP1RuntimeError(
            "systemd CPU quota timespan drifted"
        )
    scale = {
        "us": 1,
        "ms": 1_000,
        "s": 1_000_000,
        "min": 60_000_000,
        "h": 3_600_000_000,
    }[match.group(2)]
    return int(match.group(1)) * scale


def verify_live_formal_execution(
    *,
    config: Mapping[str, object],
    config_path: Path,
    paths: "RuntimePaths",
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """Attest the executing service/cgroup/network envelope before source."""

    unit_name = config.get("formal_unit_name")
    if unit_name != "ebmnlp-p1-formal-v1.service":
        raise EbmNlpP1RuntimeError("formal unit identity drifted")
    hostname_hash = _required_hash(config, "formal_hostname_sha256")
    if hashlib.sha256(
        os.uname().nodename.encode("utf-8")
    ).hexdigest() != hostname_hash:
        raise EbmNlpP1RuntimeError("formal host identity drifted")
    systemctl = _checked_executable(
        _required_path(config, "systemctl_executable"),
        "systemctl executable",
    )
    if file_sha256(systemctl) != _required_hash(
        config, "systemctl_executable_sha256"
    ):
        raise EbmNlpP1RuntimeError("systemctl executable drifted")
    unit_file = _checked_file(
        _required_path(config, "formal_unit_file"),
        "formal unit file",
    )
    if file_sha256(unit_file) != _required_hash(
        config, "formal_unit_file_sha256"
    ):
        raise EbmNlpP1RuntimeError("formal unit file drifted")
    if Path(sys.executable).resolve(strict=True) != (
        paths.hippo_python.resolve(strict=True)
    ):
        raise EbmNlpP1RuntimeError(
            "formal Python executable origin drifted"
        )
    try:
        cmdline_raw = _read_proc_regular(
            Path("/proc/self/cmdline"), maximum_bytes=64 * 1024
        )
        cgroup_raw = _read_proc_regular(
            Path("/proc/self/cgroup"), maximum_bytes=64 * 1024
        )
        environ_raw = _read_proc_regular(
            Path("/proc/self/environ"), maximum_bytes=64 * 1024
        )
        cmdline = [
            part.decode("utf-8")
            for part in cmdline_raw.rstrip(b"\0").split(b"\0")
        ]
        cgroup = cgroup_raw.decode("utf-8")
        environment_rows = [
            part.decode("utf-8")
            for part in environ_raw.rstrip(b"\0").split(b"\0")
            if part
        ]
    except UnicodeDecodeError as exc:
        raise EbmNlpP1RuntimeError(
            "live process envelope is not UTF-8"
        ) from exc
    expected_tail = [
        "-m",
        "assumption_agent.benchmarks.ebmnlp_p1_runtime_v1",
        "formal",
        "--config",
        str(Path(config_path).expanduser().absolute()),
    ]
    if (
        len(cmdline) != 1 + len(expected_tail)
        or Path(cmdline[0]).resolve(strict=True)
        != paths.hippo_python.resolve(strict=True)
        or cmdline[1:] != expected_tail
        or unit_name not in cgroup
    ):
        raise EbmNlpP1RuntimeError(
            "formal ExecStart or cgroup membership drifted"
        )
    property_names = (
        "ActiveState",
        "CPUQuotaPerSecUSec",
        "ControlGroup",
        "ExecStart",
        "FragmentPath",
        "IPAddressDeny",
        "KillMode",
        "MainPID",
        "MemoryMax",
        "Restart",
        "RestrictAddressFamilies",
        "SubState",
        "TasksMax",
    )
    command = [
        str(systemctl),
        "--user",
        "show",
        unit_name,
        "--no-pager",
        *[
            item
            for name in property_names
            for item in ("--property", name)
        ],
    ]
    try:
        completed = command_runner(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            text=True,
        )
    except Exception as exc:
        raise EbmNlpP1RuntimeError(
            "formal systemd attestation command failed"
        ) from exc
    properties: dict[str, str] = {}
    for line in str(completed.stdout).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in properties:
            raise EbmNlpP1RuntimeError(
                "formal systemd property repeated"
            )
        properties[key] = value
    exec_start = properties.get("ExecStart", "")
    control_group = properties.get("ControlGroup", "")
    fragment = properties.get("FragmentPath", "")
    try:
        main_pid = int(properties.get("MainPID", ""))
        memory_max = int(properties.get("MemoryMax", ""))
        tasks_max = int(properties.get("TasksMax", ""))
        quota = _systemd_timespan_microseconds(
            properties.get("CPUQuotaPerSecUSec", "")
        )
        fragment_matches = (
            Path(fragment).resolve(strict=True)
            == unit_file.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise EbmNlpP1RuntimeError(
            "formal systemd numeric or path property drifted"
        ) from exc
    observed_environment: dict[str, str] = {}
    for row in environment_rows:
        if "=" not in row:
            raise EbmNlpP1RuntimeError(
                "formal process environment row drifted"
            )
        key, value = row.split("=", 1)
        if key in observed_environment:
            raise EbmNlpP1RuntimeError(
                "formal process environment key repeated"
            )
        observed_environment[key] = value
    if (
        completed.returncode != 0
        or set(properties) != set(property_names)
        or properties["ActiveState"] != "active"
        or properties["SubState"] not in {"running", "start"}
        or main_pid != os.getpid()
        or quota != 8_000_000
        or memory_max != 40 * 1024**3
        or tasks_max != 64
        or properties["Restart"] != "no"
        or properties["KillMode"] != "control-group"
        or unit_name not in control_group
        or properties["RestrictAddressFamilies"] != "AF_UNIX"
        or properties["IPAddressDeny"]
        not in {
            "any",
            "0.0.0.0/0 ::/0",
            "::/0 0.0.0.0/0",
        }
        or not fragment_matches
        or observed_environment != FORMAL_CLEAN_ENVIRONMENT
        or str(paths.hippo_python) not in exec_start
        or str(Path(config_path).expanduser().absolute())
        not in exec_start
    ):
        raise EbmNlpP1RuntimeError(
            "live formal systemd envelope drifted"
        )
    body = {
        "schema": LIVE_EXECUTION_ATTESTATION_SCHEMA,
        "status": (
            "verified_effective_service_cgroup_and_network_before_source"
        ),
        "study_id": core.STUDY_ID,
        "formal_unit_name": unit_name,
        "hostname_sha256": hostname_hash,
        "unit_file_sha256": str(
            config["formal_unit_file_sha256"]
        ),
        "systemctl_executable_sha256": str(
            config["systemctl_executable_sha256"]
        ),
        "main_pid_sha256": hashlib.sha256(
            str(main_pid).encode("ascii")
        ).hexdigest(),
        "control_group_sha256": hashlib.sha256(
            control_group.encode("utf-8")
        ).hexdigest(),
        "CPUQuota_percent": 800,
        "MemoryMax_bytes": memory_max,
        "TasksMax": tasks_max,
        "KillMode": properties["KillMode"],
        "Restart": properties["Restart"],
        "RestrictAddressFamilies": ["AF_UNIX"],
        "IPAddressDeny": "any",
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
        "gpu_assignment": ["0", "1"],
        "clean_environment_key_count": len(
            FORMAL_CLEAN_ENVIRONMENT
        ),
        "provider_or_API_credential_environment_key_count": 0,
        "external_network_call_count_before_source": 0,
        "EBM_NLP_archive_path_or_member_access_count": 0,
    }
    receipt = self_hashed(body)
    _write_receipt(
        _required_path(config, "live_attestation_output"),
        receipt,
    )
    return receipt


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    runtime_root: Path
    minilm_asset_manifest: Path
    minilm_model: Path
    hippo_llm_model: Path
    hippo_embedding_model: Path
    hipporag_source: Path
    hippo_python: Path
    strace_executable: Path = Path("/usr/bin/strace")
    env_executable: Path = Path("/usr/bin/env")

    def checked(self) -> "RuntimePaths":
        project = _checked_directory(
            self.project_root, "project root"
        )
        manifest = _checked_file(
            self.minilm_asset_manifest, "MiniLM asset manifest"
        )
        minilm = _checked_directory(
            self.minilm_model, "MiniLM model"
        )
        llm = _checked_directory(
            self.hippo_llm_model, "HippoRAG LLM model"
        )
        hippo_embedding = _checked_directory(
            self.hippo_embedding_model,
            "HippoRAG embedding model",
        )
        hippo_source = _checked_directory(
            self.hipporag_source, "HippoRAG source"
        )
        python = _checked_executable(
            self.hippo_python,
            "HippoRAG Python",
            allow_leaf_symlink=True,
        )
        strace = _checked_executable(
            self.strace_executable, "strace"
        )
        env = _checked_executable(
            self.env_executable, "environment clearer"
        )
        runtime = Path(self.runtime_root).expanduser().absolute()
        if runtime.exists() or runtime.is_symlink():
            raise EbmNlpP1RuntimeError(
                "runtime root is already consumed"
            )
        if not runtime.parent.is_dir():
            raise EbmNlpP1RuntimeError(
                "runtime root parent is unavailable"
            )
        return RuntimePaths(
            project_root=project,
            runtime_root=runtime,
            minilm_asset_manifest=manifest,
            minilm_model=minilm,
            hippo_llm_model=llm,
            hippo_embedding_model=hippo_embedding,
            hipporag_source=hippo_source,
            hippo_python=python,
            strace_executable=strace,
            env_executable=env,
        )


def verify_frozen_assets(paths: RuntimePaths) -> dict[str, object]:
    """Passively verify all offline model/source trees."""

    checked = paths
    minilm_tree = tree_receipt(checked.minilm_model)
    hippo_embedding_tree = tree_receipt(
        checked.hippo_embedding_model
    )
    llm_tree = tree_receipt(checked.hippo_llm_model)
    source_tree = tree_receipt(checked.hipporag_source)
    if (
        minilm_tree["tree_sha256"] != MINILM_GENERIC_TREE_SHA256
        or hippo_embedding_tree["tree_sha256"]
        != MINILM_GENERIC_TREE_SHA256
        or llm_tree["tree_sha256"] != HIPPORAG_LLM_TREE_SHA256
        or source_tree["tree_sha256"]
        != HIPPORAG_SOURCE_TREE_SHA256
    ):
        raise EbmNlpP1RuntimeError("offline asset tree drifted")
    weights = checked.minilm_model / "model.safetensors"
    if (
        not weights.is_file()
        or weights.is_symlink()
        or file_sha256(weights) != MINILM_WEIGHTS_SHA256
    ):
        raise EbmNlpP1RuntimeError("MiniLM weights drifted")
    binding = _verify_minilm_asset(
        asset_manifest_path=checked.minilm_asset_manifest,
        model_root=checked.minilm_model,
    )
    if (
        binding.get("model_tree_sha256")
        != MINILM_NORMATIVE_TREE_SHA256
    ):
        raise EbmNlpP1RuntimeError(
            "normative MiniLM tree identity drifted"
        )
    return {
        "MiniLM": {
            "generic_tree": minilm_tree,
            "normative_tree_sha256": (
                MINILM_NORMATIVE_TREE_SHA256
            ),
            "weights_sha256": MINILM_WEIGHTS_SHA256,
        },
        "HippoRAG_embedding": hippo_embedding_tree,
        "HippoRAG_LLM": llm_tree,
        "HippoRAG_source": source_tree,
    }


def _load_local_minilm_model(model_root: Path) -> object:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EbmNlpP1RuntimeError(
            "offline MiniLM runtime is unavailable"
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise EbmNlpP1RuntimeError(
            "the frozen two-GPU runtime is unavailable"
        )
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    try:
        model = SentenceTransformer(
            str(model_root),
            device=EMBEDDING_DEVICE,
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={
                "local_files_only": True,
                "torch_dtype": torch.float32,
                "use_safetensors": True,
            },
            config_kwargs={
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    except Exception as exc:
        raise EbmNlpP1RuntimeError(
            "verified local MiniLM failed to load"
        ) from exc
    model.max_seq_length = 256
    model.float()
    model.eval()
    parameters = tuple(model.parameters())
    if (
        model.training
        or not parameters
        or any(
            parameter.device.type != "cuda"
            or parameter.dtype != torch.float32
            for parameter in parameters
        )
    ):
        raise EbmNlpP1RuntimeError(
            "MiniLM GPU float32 eval state drifted"
        )
    return model


class LocalMiniLMEmbedder:
    """One frozen L2-normalized float32 MiniLM stream on GPU0."""

    def __init__(
        self,
        *,
        asset_manifest: Path,
        model_root: Path,
    ) -> None:
        _install_cuda_determinism_environment()
        for key, value in {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }.items():
            os.environ[key] = value
        binding = _verify_minilm_asset(
            asset_manifest_path=Path(asset_manifest),
            model_root=Path(model_root),
        )
        if (
            binding.get("model_tree_sha256")
            != MINILM_NORMATIVE_TREE_SHA256
            or binding.get("weights_sha256")
            != MINILM_WEIGHTS_SHA256
        ):
            raise EbmNlpP1RuntimeError(
                "MiniLM runtime identity drifted"
            )
        self._model = _load_local_minilm_model(Path(model_root))
        self._lock = threading.Lock()
        self._call_count = 0
        self._encoded_text_count = 0

    def __call__(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        if (
            isinstance(texts, (str, bytes))
            or not isinstance(texts, Sequence)
            or not texts
            or any(
                not isinstance(text, str)
                or not text
                or "\x00" in text
                for text in texts
            )
        ):
            raise EbmNlpP1RuntimeError(
                "MiniLM input text population drifted"
            )
        try:
            import numpy as np
            values = self._model.encode(
                list(texts),
                batch_size=EMBEDDING_BATCH_SIZE,
                convert_to_numpy=True,
                convert_to_tensor=False,
                device=EMBEDDING_DEVICE,
                normalize_embeddings=True,
                precision="float32",
                show_progress_bar=False,
            )
            matrix = np.asarray(values)
        except Exception as exc:
            raise EbmNlpP1RuntimeError(
                "offline MiniLM encoding failed"
            ) from exc
        if (
            matrix.dtype.name != "float32"
            or matrix.shape
            != (len(texts), EMBEDDING_DIMENSION)
            or not np.isfinite(matrix).all()
        ):
            raise EbmNlpP1RuntimeError(
                "MiniLM output dtype or shape drifted"
            )
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
            raise EbmNlpP1RuntimeError(
                "MiniLM output normalization drifted"
            )
        with self._lock:
            self._call_count += 1
            self._encoded_text_count += len(texts)
        return tuple(
            tuple(float(value) for value in row) for row in matrix
        )

    def safe_runtime_receipt(self) -> Mapping[str, object]:
        with self._lock:
            calls = self._call_count
            texts = self._encoded_text_count
        return {
            "schema": EMBEDDER_RECEIPT_SCHEMA,
            "status": "complete_offline_deterministic_embeddings",
            "model_tree_sha256": MINILM_NORMATIVE_TREE_SHA256,
            "weights_sha256": MINILM_WEIGHTS_SHA256,
            "device": EMBEDDING_DEVICE,
            "dtype": "float32",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "batch_size": EMBEDDING_BATCH_SIZE,
            "maximum_sequence_length": 256,
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
            "torch_deterministic_algorithms": True,
            "torch_manual_seed": 0,
            "call_count": calls,
            "encoded_text_count": texts,
            "external_network_call_count": 0,
            "online_or_api_evaluator_call_count": 0,
            "retry_or_replay_count": 0,
        }


class OfficialHippoBatchLauncher:
    """Two stable one-GPU lanes for complete abstract payloads."""

    def __init__(self, paths: RuntimePaths) -> None:
        self.paths = paths.checked()
        _private_directory(self.paths.runtime_root, fresh=True)
        self._alias_root = _private_directory(
            self.paths.runtime_root / "model_aliases"
        )
        for alias, target in (
            ("smollm2", self.paths.hippo_llm_model),
            ("minilm", self.paths.hippo_embedding_model),
        ):
            alias_path = self._alias_root / alias
            try:
                os.symlink(str(target), alias_path)
            except OSError as exc:
                raise EbmNlpP1RuntimeError(
                    "frozen model alias creation failed"
                ) from exc
            if not alias_path.is_dir():
                raise EbmNlpP1RuntimeError(
                    "frozen model alias target drifted"
                )
        self._attempts_root = _private_directory(
            self.paths.runtime_root / "attempts"
        )
        self._lock = threading.Lock()
        self._next_ordinal = 0
        self._batch_invocation_count = 0
        self._claimed_input_sha256s: set[str] = set()
        self._failed = False
        self._active = 0
        self._active_by_gpu = {gpu: 0 for gpu in GPU_ASSIGNMENT}
        self._observed_peak = 0
        self._observed_peak_by_gpu = {
            gpu: 0 for gpu in GPU_ASSIGNMENT
        }
        self._attempt_count = 0
        self._completed_count = 0
        self._completed_by_gpu = {
            gpu: 0 for gpu in GPU_ASSIGNMENT
        }
        self._index_destroyed_count = 0
        self._attempted_network = 0
        self._denied_network = 0

    @staticmethod
    def _canonical_payload(
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        try:
            (
                abstract_work_id,
                corpus_sha256,
                documents,
                queries,
            ) = hippo.validate_input(payload)
            canonical = hippo.input_payload(
                abstract_work_id=abstract_work_id,
                documents=[
                    {
                        "ordinal": row.ordinal,
                        "text": row.text,
                        "window_id": row.window_id,
                    }
                    for row in documents
                ],
                queries=[
                    {
                        "ordinal": row.ordinal,
                        "role": row.role,
                        "text": row.text,
                        "work_id": row.work_id,
                    }
                    for row in queries
                ],
            )
        except Exception as exc:
            raise EbmNlpP1RuntimeError(
                "HippoRAG input payload drifted"
            ) from exc
        if (
            canonical != dict(payload)
            or canonical["corpus_sha256"] != corpus_sha256
        ):
            raise EbmNlpP1RuntimeError(
                "HippoRAG input is noncanonical"
            )
        return canonical

    @staticmethod
    def _audit_network(path: Path) -> dict[str, int | str]:
        raw = _read_regular(
            path,
            maximum_bytes=MAXIMUM_NETWORK_AUDIT_BYTES,
            expected_mode=0o600,
        )
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise EbmNlpP1RuntimeError(
                "network audit is not UTF-8"
            ) from exc
        attempted = 0
        denied = 0
        for line in lines:
            text = line.strip()
            if not text:
                continue
            if text.startswith("strace:"):
                raise EbmNlpP1RuntimeError(
                    "strace reported an injection failure"
                )
            if text.startswith("+++") or text.startswith("---"):
                continue
            if "<unfinished ...>" in text:
                if _NETWORK_CALL.match(text) is None:
                    raise EbmNlpP1RuntimeError(
                        "network audit line drifted"
                    )
                continue
            if _NETWORK_CALL.match(text) or _NETWORK_RESUMED.match(
                text
            ):
                attempted += 1
                if "= -1 EPERM" not in text:
                    raise EbmNlpP1RuntimeError(
                        "a network syscall was not denied"
                    )
                denied += 1
                continue
            raise EbmNlpP1RuntimeError(
                "network audit contains an unknown line"
            )
        if denied != attempted:
            raise EbmNlpP1RuntimeError(
                "network denial audit is incomplete"
            )
        return {
            "attempted_network_syscall_count": attempted,
            "denied_network_syscall_count": denied,
            "external_network_call_count": 0,
            "network_audit_file_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def _counter_enter(self, gpu: str) -> None:
        with self._lock:
            self._active += 1
            self._active_by_gpu[gpu] += 1
            if (
                self._active > MAXIMUM_HIPPO_PROCESSES
                or self._active_by_gpu[gpu] > 1
            ):
                raise EbmNlpP1RuntimeError(
                    "HippoRAG process concurrency drifted"
                )
            self._observed_peak = max(
                self._observed_peak, self._active
            )
            self._observed_peak_by_gpu[gpu] = max(
                self._observed_peak_by_gpu[gpu],
                self._active_by_gpu[gpu],
            )

    def _counter_exit(self, gpu: str) -> None:
        with self._lock:
            self._active -= 1
            self._active_by_gpu[gpu] -= 1
            if self._active < 0 or self._active_by_gpu[gpu] < 0:
                raise EbmNlpP1RuntimeError(
                    "HippoRAG process counter underflowed"
                )

    def _run_one(
        self,
        *,
        ordinal: int,
        gpu: str,
        payload: Mapping[str, object],
    ) -> tuple[str, Mapping[str, object]]:
        raw_input = hippo.canonical_json_bytes(payload)
        input_sha256 = hashlib.sha256(raw_input).hexdigest()
        directory = self._attempts_root / (
            f"{ordinal:06d}-{input_sha256}"
        )
        _private_directory(directory, fresh=True)
        input_path = directory / "input.json"
        output_path = directory / "output.json"
        index_root = directory / "index"
        audit_path = directory / "network.strace"
        _write_exclusive(input_path, raw_input)
        _write_exclusive(audit_path, b"")
        _write_exclusive(
            directory / "attempt.json",
            canonical_json_bytes(
                self_hashed(
                    {
                        "schema": f"{VERSION}_hippo_attempt_v1",
                        "status": (
                            "consumed_before_unique_offline_subprocess"
                        ),
                        "input_sha256": input_sha256,
                        "ordinal": ordinal,
                        "visible_gpu": gpu,
                        "attempt_count": 1,
                        "retry_count": 0,
                    }
                )
            ),
        )
        environment_rows = [
            f"CUDA_VISIBLE_DEVICES={gpu}",
            f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG}",
            "HF_DATASETS_OFFLINE=1",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "TOKENIZERS_PARALLELISM=false",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONHASHSEED=0",
            "PYTHONNOUSERSITE=1",
            (
                "PYTHONPATH="
                f"{self.paths.project_root}{os.pathsep}"
                f"{self.paths.hipporag_source}"
            ),
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "VECLIB_MAXIMUM_THREADS=1",
        ]
        command = [
            str(self.paths.strace_executable),
            "-f",
            "-qq",
            "-e",
            "trace=%network",
            "-e",
            "inject=%network:error=EPERM",
            "-o",
            str(audit_path),
            str(self.paths.env_executable),
            "-i",
            *environment_rows,
            str(self.paths.hippo_python),
            "-m",
            HIPPO_WORKER_MODULE,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            "smollm2",
            "--embedding-model",
            "minilm",
            "--hipporag-source-root",
            str(self.paths.hipporag_source),
            "--project-root",
            str(self.paths.project_root),
        ]
        with self._lock:
            self._attempt_count += 1
        self._counter_enter(gpu)
        try:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self._alias_root,
                    env={},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=None,
                )
            except Exception as exc:
                raise EbmNlpP1RuntimeError(
                    "HippoRAG worker process failed to terminate"
                ) from exc
            if (
                type(completed.returncode) is not int
                or completed.returncode != 0
            ):
                raise EbmNlpP1RuntimeError(
                    "HippoRAG worker process failed"
                )
            raw_output = _read_regular(
                output_path,
                maximum_bytes=MAXIMUM_WORKER_OUTPUT_BYTES,
                expected_mode=0o600,
            )
            try:
                output = hippo.parse_output(raw_output)
            except Exception as exc:
                raise EbmNlpP1RuntimeError(
                    "HippoRAG worker output drifted"
                ) from exc
            if (
                output["abstract_work_id"]
                != payload["abstract_work_id"]
                or output["corpus_sha256"]
                != payload["corpus_sha256"]
                or output["document_count"]
                != len(payload["documents"])
            ):
                raise EbmNlpP1RuntimeError(
                    "HippoRAG worker output binding drifted"
                )
            network = self._audit_network(audit_path)
            if index_root.is_symlink() or not index_root.is_dir():
                raise EbmNlpP1RuntimeError(
                    "HippoRAG worker index is unavailable"
                )
            shutil.rmtree(index_root)
            if index_root.exists() or index_root.is_symlink():
                raise EbmNlpP1RuntimeError(
                    "HippoRAG worker index was not destroyed"
                )
            _write_exclusive(
                directory / "safe_runtime_receipt.json",
                canonical_json_bytes(
                    self_hashed(
                        {
                            "schema": f"{VERSION}_hippo_worker_receipt_v1",
                            "status": (
                                "complete_offline_output_verified_"
                                "index_destroyed"
                            ),
                            "input_sha256": input_sha256,
                            "output_file_sha256": hashlib.sha256(
                                raw_output
                            ).hexdigest(),
                            "ordinal": ordinal,
                            "visible_gpu": gpu,
                            "configured_cpu_threads": 1,
                            "index_destroyed": True,
                            "subprocess_environment_was_env_i": True,
                            "retry_or_replay_count": 0,
                            **network,
                        }
                    )
                ),
            )
            with self._lock:
                self._completed_count += 1
                self._completed_by_gpu[gpu] += 1
                self._index_destroyed_count += 1
                self._attempted_network += int(
                    network["attempted_network_syscall_count"]
                )
                self._denied_network += int(
                    network["denied_network_syscall_count"]
                )
            return str(payload["abstract_work_id"]), output
        finally:
            self._counter_exit(gpu)

    def __call__(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> Mapping[str, Mapping[str, object]]:
        if (
            isinstance(payloads, (str, bytes))
            or not isinstance(payloads, Sequence)
            or not payloads
        ):
            raise EbmNlpP1RuntimeError(
                "HippoRAG batch is empty or malformed"
            )
        canonical = tuple(
            self._canonical_payload(payload) for payload in payloads
        )
        identities = tuple(
            str(payload["abstract_work_id"]) for payload in canonical
        )
        if len(set(identities)) != len(identities):
            raise EbmNlpP1RuntimeError(
                "HippoRAG batch work identities are duplicated"
            )
        input_hashes = {
            hashlib.sha256(
                hippo.canonical_json_bytes(payload)
            ).hexdigest()
            for payload in canonical
        }
        if len(input_hashes) != len(canonical):
            raise EbmNlpP1RuntimeError(
                "HippoRAG batch payloads are duplicated"
            )
        with self._lock:
            if self._failed:
                raise EbmNlpP1RuntimeError(
                    "HippoRAG launcher is terminally consumed"
                )
            if (
                self._batch_invocation_count >= 2
                or input_hashes & self._claimed_input_sha256s
            ):
                raise EbmNlpP1RuntimeError(
                    "HippoRAG batch retry or replay is forbidden"
                )
            self._batch_invocation_count += 1
            self._claimed_input_sha256s.update(input_hashes)
            start = self._next_ordinal
            self._next_ordinal += len(canonical)
        lanes: list[list[tuple[int, Mapping[str, object]]]] = [
            [],
            [],
        ]
        for offset, payload in enumerate(canonical):
            lane = offset % len(GPU_ASSIGNMENT)
            lanes[lane].append((start + offset, payload))

        def run_lane(
            lane: int,
        ) -> list[tuple[str, Mapping[str, object]]]:
            return [
                self._run_one(
                    ordinal=ordinal,
                    gpu=GPU_ASSIGNMENT[lane],
                    payload=payload,
                )
                for ordinal, payload in lanes[lane]
            ]

        results: dict[str, Mapping[str, object]] = {}
        active_lanes = [
            lane for lane, rows in enumerate(lanes) if rows
        ]
        try:
            with ThreadPoolExecutor(
                max_workers=len(active_lanes),
                thread_name_prefix="ebmnlp-hippo",
            ) as executor:
                futures = {
                    lane: executor.submit(run_lane, lane)
                    for lane in active_lanes
                }
                for lane in active_lanes:
                    for identity, output in futures[lane].result():
                        if identity in results:
                            raise EbmNlpP1RuntimeError(
                                "HippoRAG batch result collided"
                            )
                        results[identity] = output
        except BaseException:
            with self._lock:
                self._failed = True
            raise
        if set(results) != set(identities):
            raise EbmNlpP1RuntimeError(
                "HippoRAG batch result population drifted"
            )
        return results

    def safe_runtime_receipt(self) -> Mapping[str, object]:
        with self._lock:
            if self._active != 0 or any(self._active_by_gpu.values()):
                raise EbmNlpP1RuntimeError(
                    "HippoRAG workers remain active"
                )
            return {
                "schema": HIPPO_RECEIPT_SCHEMA,
                "status": (
                    "complete_offline_outputs_verified_indexes_destroyed"
                ),
                "gpu_assignment": list(GPU_ASSIGNMENT),
                "CUBLAS_WORKSPACE_CONFIG": (
                    CUBLAS_WORKSPACE_CONFIG
                ),
                "maximum_process_count": MAXIMUM_HIPPO_PROCESSES,
                "maximum_processes_per_gpu": 1,
                "configured_cpu_threads_per_process": (
                    CPU_THREADS_PER_HIPPO_PROCESS
                ),
                "observed_process_peak": self._observed_peak,
                "observed_process_peak_by_gpu": dict(
                    self._observed_peak_by_gpu
                ),
                "worker_attempt_count": self._attempt_count,
                "worker_completed_count": self._completed_count,
                "worker_completed_count_by_gpu": dict(
                    self._completed_by_gpu
                ),
                "index_destroyed_count": self._index_destroyed_count,
                "batch_invocation_count": (
                    self._batch_invocation_count
                ),
                "HippoRAG_source_tree_sha256": (
                    HIPPORAG_SOURCE_TREE_SHA256
                ),
                "HippoRAG_import_origin_verified_worker_count": (
                    self._completed_count
                ),
                "attempted_network_syscall_count": (
                    self._attempted_network
                ),
                "denied_network_syscall_count": self._denied_network,
                "external_network_call_count": 0,
                "online_or_api_evaluator_call_count": 0,
                "retry_or_replay_count": 0,
            }


def _module_binding(module: object) -> dict[str, object]:
    path_value = getattr(module, "__file__", None)
    if not isinstance(path_value, str):
        raise EbmNlpP1RuntimeError("runtime module origin is absent")
    path = _checked_file(Path(path_value), "runtime module origin")
    return {
        "module": getattr(module, "__name__", type(module).__name__),
        "origin_path_sha256": hashlib.sha256(
            str(path).encode("utf-8")
        ).hexdigest(),
        "origin_file_sha256": file_sha256(path),
    }


def build_source_free_runtime_fingerprint(
    paths: RuntimePaths,
    *,
    implementation_freeze_sha256: str,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """Build a source-free asset/runtime/hardware fingerprint."""

    if _HEX64.fullmatch(implementation_freeze_sha256) is None:
        raise EbmNlpP1RuntimeError(
            "implementation freeze hash drifted"
        )
    _install_cuda_determinism_environment()
    checked = paths.checked()
    assets = verify_frozen_assets(checked)
    distributions = {}
    for name in (
        "hipporag",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "transformers",
    ):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise EbmNlpP1RuntimeError(
                f"runtime distribution {name} is absent"
            ) from exc
    try:
        gpu_process = command_runner(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            text=True,
        )
    except Exception as exc:
        raise EbmNlpP1RuntimeError(
            "GPU inventory command failed"
        ) from exc
    lines = [
        line.strip()
        for line in str(gpu_process.stdout).splitlines()
        if line.strip()
    ]
    if gpu_process.returncode != 0 or len(lines) != 2:
        raise EbmNlpP1RuntimeError(
            "exact two-GPU inventory is unavailable"
        )
    gpus: list[dict[str, object]] = []
    for expected_index, line in enumerate(lines):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5 or parts[0] != str(expected_index):
            raise EbmNlpP1RuntimeError(
                "GPU inventory row drifted"
            )
        gpus.append(
            {
                "index": expected_index,
                "name": parts[1],
                "uuid_sha256": hashlib.sha256(
                    parts[2].encode("utf-8")
                ).hexdigest(),
                "memory_total_MiB": int(parts[3]),
                "driver_version": parts[4],
            }
        )
    import sklearn
    import torch
    import transformers
    import sentence_transformers
    body = {
        "schema": FINGERPRINT_SCHEMA,
        "status": "verified_before_formal_source_open",
        "study_id": core.STUDY_ID,
        "implementation_freeze_sha256": (
            implementation_freeze_sha256
        ),
        "remote_host_alias": "jtl311linux",
        "hostname_sha256": hashlib.sha256(
            os.uname().nodename.encode("utf-8")
        ).hexdigest(),
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
        "platform": {
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "system": platform.system(),
        },
        "gpu_assignment": list(GPU_ASSIGNMENT),
        "gpus": gpus,
        "distributions": distributions,
        "module_bindings": [
            _module_binding(module)
            for module in (
                torch,
                sklearn,
                transformers,
                sentence_transformers,
            )
        ],
        "python_executable_target_sha256": file_sha256(
            checked.hippo_python.resolve(strict=True)
        ),
        "strace_executable_sha256": file_sha256(
            checked.strace_executable
        ),
        "environment_clearer_sha256": file_sha256(
            checked.env_executable
        ),
        "assets": assets,
        "EBM_NLP_archive_path_or_member_access_count": 0,
        "model_inference_call_count": 0,
        "provider_or_API_credential_read_count": 0,
        "external_network_call_count": 0,
        "online_or_api_evaluator_call_count": 0,
    }
    return self_hashed(body)


def run_source_free_full_path_canary(
    *,
    embedder: LocalMiniLMEmbedder,
    hippo_launcher: OfficialHippoBatchLauncher,
    implementation_freeze_sha256: str,
    runtime_fingerprint_sha256: str,
) -> dict[str, object]:
    """Exercise MiniLM, typed probes/recipes/E0/E1, and official HippoRAG."""

    if (
        _HEX64.fullmatch(implementation_freeze_sha256) is None
        or _HEX64.fullmatch(runtime_fingerprint_sha256) is None
    ):
        raise EbmNlpP1RuntimeError(
            "source-free canary prerequisite hash drifted"
        )
    tokens = tuple(
        f"synthetic_token_{index:03d}" for index in range(144)
    )
    windows = core.build_evidence_windows(tokens)
    window_embeddings = embedder(
        tuple(window.text for window in windows)
    )
    labels = {
        role: tuple(
            int((ordinal + role_index) % 2 == 0)
            for ordinal in range(len(windows))
        )
        for role_index, role in enumerate(core.ROLE_ORDER)
    }
    probes = core.fit_independent_role_probes(
        window_embeddings, labels
    )
    query_embeddings = embedder(
        tuple(core.ROLE_QUERIES[role] for role in core.ROLE_ORDER)
    )
    probabilities = probes.score_quantized(window_embeddings)
    quantized_embeddings = tuple(
        tuple(core.quantize_half_even(value) for value in row)
        for row in window_embeddings
    )
    slates: list[core.RecipeSlate] = []
    utilities: list[tuple[object, ...]] = []
    role_work_ids: dict[str, str] = {}
    for role_index, role in enumerate(core.ROLE_ORDER):
        query = query_embeddings[role_index]
        cosines = tuple(
            core.quantize_half_even(
                min(
                    1.0,
                    max(
                        0.0,
                        (
                            math.fsum(
                                float(left) * float(right)
                                for left, right in zip(row, query)
                            )
                            + 1.0
                        )
                        / 2.0,
                    ),
                ),
                unit_interval=True,
            )
            for row in window_embeddings
        )
        slate = core.build_recipe_slate(
            windows=windows,
            target_role=role,
            role_probabilities=probabilities,
            query_cosines=cosines,
            embeddings=quantized_embeddings,
        )
        positives = tuple(
            position
            for position in range(len(tokens))
            if (position + role_index) % 7 == 0
        )
        slates.append(slate)
        utilities.append(
            tuple(
                core.score_ranked_token_coverage(
                    windows=windows,
                    ranking=action.window_ordinals,
                    positive_token_positions=positives,
                ).primary_utility
                for action in slate.actions
            )
        )
        role_work_ids[role] = hashlib.sha256(
            f"canary-role-{role}".encode("ascii")
        ).hexdigest()
    e1 = core.fit_e1_deepsets(
        slates,
        utilities,
        standardization_slates=slates,
    )
    payloads = tuple(
        hippo.input_payload(
            abstract_work_id=hashlib.sha256(
                f"ebmnlp-source-free-canary-{lane}".encode("ascii")
            ).hexdigest(),
            documents=[
                {
                    "ordinal": window.ordinal,
                    "text": f"{window.text} synthetic_lane_{lane}",
                    "window_id": window.window_id,
                }
                for window in windows
            ],
            queries=[
                {
                    "ordinal": ordinal,
                    "role": role,
                    "text": core.ROLE_QUERIES[role],
                    "work_id": hashlib.sha256(
                        f"{role_work_ids[role]}:{lane}".encode("ascii")
                    ).hexdigest(),
                }
                for ordinal, role in enumerate(core.ROLE_ORDER)
            ],
        )
        for lane in range(2)
    )
    outputs = hippo_launcher(payloads)
    for payload in payloads:
        output = outputs[payload["abstract_work_id"]]
        hippo.parse_output(hippo.canonical_json_bytes(output))
    embedding_receipt = embedder.safe_runtime_receipt()
    hippo_receipt = hippo_launcher.safe_runtime_receipt()
    if (
        hippo_receipt["worker_completed_count"] != 2
        or hippo_receipt["worker_completed_count_by_gpu"]
        != {"0": 1, "1": 1}
        or hippo_receipt["observed_process_peak"] != 2
        or hippo_receipt["observed_process_peak_by_gpu"]
        != {"0": 1, "1": 1}
        or hippo_receipt["external_network_call_count"] != 0
        or embedding_receipt["external_network_call_count"] != 0
    ):
        raise EbmNlpP1RuntimeError(
            "source-free canary runtime receipt drifted"
        )
    body = {
        "schema": CANARY_SCHEMA,
        "status": "passed_source_free_synthetic_full_path",
        "study_id": core.STUDY_ID,
        "implementation_freeze_sha256": (
            implementation_freeze_sha256
        ),
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "window_count": len(windows),
        "role_count": len(core.ROLE_ORDER),
        "recipe_count": len(core.RECIPE_IDS),
        "E0_selected_recipe_ids": [
            core.select_e0(slate).recipe_id for slate in slates
        ],
        "E1_selected_recipe_ids": [
            core.select_e1(e1, slate).recipe_id for slate in slates
        ],
        "RAW_selected_recipe_ids": [
            core.raw_probe_ranking(slate).recipe_id for slate in slates
        ],
        "HippoRAG_complete_rank_permutation_count": len(
            payloads
        )
        * len(
            core.ROLE_ORDER
        ),
        "embedding_runtime_receipt_sha256": stable_hash(
            embedding_receipt
        ),
        "embedding_runtime_receipt": dict(embedding_receipt),
        "HippoRAG_runtime_receipt_sha256": stable_hash(
            hippo_receipt
        ),
        "HippoRAG_runtime_receipt": dict(hippo_receipt),
        "EBM_NLP_archive_path_or_member_access_count": 0,
        "provider_or_API_credential_read_count": 0,
        "external_network_call_count": 0,
        "online_or_api_evaluator_call_count": 0,
        "synthetic_canary_used_as_efficacy_gate": False,
    }
    return self_hashed(body)


def _load_config(path: Path) -> dict[str, object]:
    raw = _read_regular(
        _checked_file(path, "runtime config"),
        maximum_bytes=1024 * 1024,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EbmNlpP1RuntimeError(
            "runtime config is invalid"
        ) from exc
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) != raw
    ):
        raise EbmNlpP1RuntimeError(
            "runtime config is not canonical JSON"
        )
    return value


def _paths_from_config(
    config: Mapping[str, object], *, runtime_root_key: str
) -> RuntimePaths:
    required = (
        "project_root",
        runtime_root_key,
        "minilm_asset_manifest",
        "minilm_model",
        "hippo_llm_model",
        "hippo_embedding_model",
        "hipporag_source",
        "hippo_python",
        "strace_executable",
        "env_executable",
    )
    if any(
        not isinstance(config.get(key), str) or not config[key]
        for key in required
    ):
        raise EbmNlpP1RuntimeError(
            "runtime config path binding drifted"
        )
    return RuntimePaths(
        project_root=Path(str(config["project_root"])),
        runtime_root=Path(str(config[runtime_root_key])),
        minilm_asset_manifest=Path(
            str(config["minilm_asset_manifest"])
        ),
        minilm_model=Path(str(config["minilm_model"])),
        hippo_llm_model=Path(str(config["hippo_llm_model"])),
        hippo_embedding_model=Path(
            str(config["hippo_embedding_model"])
        ),
        hipporag_source=Path(str(config["hipporag_source"])),
        hippo_python=Path(str(config["hippo_python"])),
        strace_executable=Path(str(config["strace_executable"])),
        env_executable=Path(str(config["env_executable"])),
    ).checked()


def _write_receipt(path: Path, value: Mapping[str, object]) -> None:
    _write_exclusive(path, canonical_json_bytes(value))


_RUNTIME_PATH_CONFIG_KEYS = frozenset(
    {
        "env_executable",
        "hippo_embedding_model",
        "hippo_llm_model",
        "hippo_python",
        "hipporag_source",
        "minilm_asset_manifest",
        "minilm_model",
        "project_root",
        "strace_executable",
    }
)
_IMPLEMENTATION_CONFIG_KEYS = frozenset(
    {
        "implementation_freeze_manifest",
        "implementation_freeze_sha256",
    }
)


def _require_exact_config_keys(
    config: Mapping[str, object], expected: frozenset[str]
) -> None:
    if set(config) != expected:
        raise EbmNlpP1RuntimeError(
            "runtime config key registry drifted"
        )


def _verify_implementation_from_config(
    config: Mapping[str, object],
) -> str:
    expected = _required_hash(
        config, "implementation_freeze_sha256"
    )
    verify_implementation_freeze(
        project_root=_required_path(config, "project_root"),
        manifest_path=_required_path(
            config, "implementation_freeze_manifest"
        ),
        expected_self_sha256=expected,
    )
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    _install_cuda_determinism_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("fingerprint", "canary", "formal")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    config = _load_config(arguments.config)

    if arguments.mode == "fingerprint":
        _require_exact_config_keys(
            config,
            _RUNTIME_PATH_CONFIG_KEYS
            | _IMPLEMENTATION_CONFIG_KEYS
            | {"fingerprint_unused_runtime_root"},
        )
        if arguments.output is None:
            raise EbmNlpP1RuntimeError(
                "fingerprint output path is required"
            )
        implementation_freeze_sha256 = (
            _verify_implementation_from_config(config)
        )
        # Fingerprinting reserves no executor work root.
        paths = _paths_from_config(
            config, runtime_root_key="fingerprint_unused_runtime_root"
        )
        receipt = build_source_free_runtime_fingerprint(
            paths,
            implementation_freeze_sha256=(
                implementation_freeze_sha256
            ),
        )
        _write_receipt(arguments.output, receipt)
    elif arguments.mode == "canary":
        _require_exact_config_keys(
            config,
            _RUNTIME_PATH_CONFIG_KEYS
            | _IMPLEMENTATION_CONFIG_KEYS
            | {
                "canary_runtime_root",
                "runtime_fingerprint_receipt",
                "runtime_fingerprint_sha256",
            },
        )
        if arguments.output is None:
            raise EbmNlpP1RuntimeError(
                "canary output path is required"
            )
        implementation_freeze_sha256 = (
            _verify_implementation_from_config(config)
        )
        paths = _paths_from_config(
            config, runtime_root_key="canary_runtime_root"
        )
        fingerprint = _verify_fingerprint_receipt(
            config,
            implementation_freeze_sha256=(
                implementation_freeze_sha256
            ),
            paths=paths,
        )
        embedder = LocalMiniLMEmbedder(
            asset_manifest=paths.minilm_asset_manifest,
            model_root=paths.minilm_model,
        )
        launcher = OfficialHippoBatchLauncher(paths)
        receipt = run_source_free_full_path_canary(
            embedder=embedder,
            hippo_launcher=launcher,
            implementation_freeze_sha256=(
                implementation_freeze_sha256
            ),
            runtime_fingerprint_sha256=str(
                fingerprint["self_sha256"]
            ),
        )
        _write_receipt(arguments.output, receipt)
    else:
        _require_exact_config_keys(
            config,
            _FORMAL_SEMANTIC_CONFIG_KEYS
            | {
                "execution_config_sha256",
                "execution_freeze_manifest",
                "execution_freeze_sha256",
            },
        )
        implementation_freeze_sha256 = (
            _verify_implementation_from_config(config)
        )
        paths = _paths_from_config(
            config, runtime_root_key="formal_executor_runtime_root"
        )
        fingerprint = _verify_fingerprint_receipt(
            config,
            implementation_freeze_sha256=(
                implementation_freeze_sha256
            ),
            paths=paths,
        )
        canary = _verify_canary_receipt(
            config,
            implementation_freeze_sha256=(
                implementation_freeze_sha256
            ),
            runtime_fingerprint_sha256=str(
                fingerprint["self_sha256"]
            ),
        )
        execution_freeze, execution_config_sha256 = (
            _verify_execution_freeze(
                config,
                implementation_freeze_sha256=(
                    implementation_freeze_sha256
                ),
                runtime_fingerprint_sha256=str(
                    fingerprint["self_sha256"]
                ),
                source_free_canary_sha256=str(
                    canary["self_sha256"]
                ),
            )
        )
        live_attestation = verify_live_formal_execution(
            config=config,
            config_path=arguments.config,
            paths=paths,
        )
        if (
            not isinstance(config.get("archive_path"), str)
            or not isinstance(config.get("formal_work_root"), str)
        ):
            raise EbmNlpP1RuntimeError(
                "formal execution binding is incomplete"
            )
        embedder = LocalMiniLMEmbedder(
            asset_manifest=paths.minilm_asset_manifest,
            model_root=paths.minilm_model,
        )
        launcher = OfficialHippoBatchLauncher(paths)
        binding = formal.FormalExecutionBinding(
            implementation_freeze_sha256=str(
                implementation_freeze_sha256
            ),
            runtime_fingerprint_sha256=str(
                fingerprint["self_sha256"]
            ),
            source_free_canary_sha256=str(
                canary["self_sha256"]
            ),
            execution_config_sha256=execution_config_sha256,
            execution_freeze_sha256=str(
                execution_freeze["self_sha256"]
            ),
            live_execution_attestation_sha256=str(
                live_attestation["self_sha256"]
            ),
            source_archive_sha256=source.FORMAL_CONTRACT.archive_sha256,
            minilm_tree_sha256=MINILM_NORMATIVE_TREE_SHA256,
            hipporag_source_tree_sha256=HIPPORAG_SOURCE_TREE_SHA256,
            hipporag_llm_tree_sha256=HIPPORAG_LLM_TREE_SHA256,
        )
        terminal = formal.run_formal_study(
            archive_path=Path(str(config["archive_path"])),
            work_root=Path(str(config["formal_work_root"])),
            contract=source.FORMAL_CONTRACT,
            embedder=embedder,
            hippo_launcher=launcher,
            execution_binding=binding,
        )
        if arguments.output is not None:
            _write_receipt(arguments.output, terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EbmNlpP1RuntimeError",
    "LocalMiniLMEmbedder",
    "OfficialHippoBatchLauncher",
    "RuntimePaths",
    "build_source_free_runtime_fingerprint",
    "run_source_free_full_path_canary",
    "verify_frozen_assets",
]

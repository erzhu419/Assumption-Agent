"""One-shot, offline production closure for the HiTab P1 formal lifecycle.

This module deliberately has no HiTab source reader.  It verifies a source-free
implementation freeze, binds the already-existing local BRIGHT planner,
cross-encoder and MiniLM assets, and launches the item-local official HippoRAG
worker as a fresh ``env -i`` subprocess.  The outer user service is required to
deny IP networking; the child receives only an exact offline environment and
therefore inherits that systemd network boundary without an API credential,
proxy, or network configuration channel.

The formal acquisition boundary is injected through a frozen factory reference.
That narrow seam lets the private custodian evolve independently without
putting a source path, qrel, family, or item value in this production module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence


VERSION = "hitab_p1_production_runtime_v1"
STUDY_ID = "HITAB_P1_DMC1_HIERARCHICAL_SET_EVALUATOR_V1"
IMPLEMENTATION_REVISION = (
    "direct_transformers_minilm_v4_sealed_child_sys_path"
)
IMPLEMENTATION_FREEZE_SCHEMA = f"{VERSION}_implementation_freeze"
ACQUISITION_FREEZE_SCHEMA = f"{VERSION}_source_acquisition_freeze"
EXECUTION_FREEZE_SCHEMA = f"{VERSION}_execution_freeze"
CANARY_RECEIPT_SCHEMA = f"{VERSION}_source_free_canary_receipt"
CANARY_ATTEMPT_SCHEMA = f"{VERSION}_source_free_canary_attempt"
EXECUTION_CLAIM_SCHEMA = f"{VERSION}_exclusive_execution_claim"

HIPPORAG_WORKER_MODULE = (
    "replication_runtime.birco_official_hipporag_v1.worker"
)
# ``python -m ...runner`` first imports this package's eager ``__init__`` and
# then re-executes runner as ``__main__``, producing two incompatible dataclass
# identities.  The canonical import-only entrypoint executes exactly one module.
OUTER_ENTRYPOINT_SCRIPT = (
    'import os;project=os.environ.get("PYTHONPATH","");'
    'bad=any(name=="__pycache__" or name.endswith((".pyc",".pyo")) '
    "for current,dirs,files in os.walk(project,followlinks=False) "
    "for name in dirs+files);"
    '(_ for _ in ()).throw(RuntimeError("unbound project bytecode")) '
    "if bad else None;"
    'getattr(__import__("replication_runtime.hitab_p1_formal_v1.runner",'
    'fromlist=["main"]),"main")()'
)
ACQUISITION_FACTORY_MODULE = (
    "assumption_agent.benchmarks.hitab_p1_source_acquisition_v1"
)
ACQUISITION_FACTORY_ATTRIBUTE = "build_production_boundary_from_execution"
ACQUISITION_FACTORY_FILE_LABEL = "hitab_source_acquisition"
PHYSICAL_GPUS = (0, 1)
CPU_THREADS_PER_GPU_LANE = 4
MAXIMUM_JSON_BYTES = 64 * 1024 * 1024
MAXIMUM_WORKER_OUTPUT_BYTES = 16 * 1024 * 1024
MAXIMUM_DIAGNOSTIC_PERSISTED_BYTES = 1 * 1024 * 1024

REQUIRED_PROJECT_FILES: dict[str, str] = {
    "assumption_agent_init": "assumption_agent/__init__.py",
    "assumption_agent_models": "assumption_agent/models.py",
    "benchmarks_init": "assumption_agent/benchmarks/__init__.py",
    "bright_cross_encoder_contract": (
        "replication_runtime/bright_cross_encoder_v1/contract.py"
    ),
    "bright_cross_encoder_init": (
        "replication_runtime/bright_cross_encoder_v1/__init__.py"
    ),
    "bright_cross_encoder_worker": (
        "replication_runtime/bright_cross_encoder_v1/worker.py"
    ),
    "bright_minilm_encoder": (
        "replication_runtime/bright_minilm_v1/encoder.py"
    ),
    "bright_minilm_init": (
        "replication_runtime/bright_minilm_v1/__init__.py"
    ),
    "bright_query_v1_contract": (
        "replication_runtime/bright_query_generator_v1/contract.py"
    ),
    "bright_query_v1_init": (
        "replication_runtime/bright_query_generator_v1/__init__.py"
    ),
    "bright_query_v1_worker": (
        "replication_runtime/bright_query_generator_v1/worker.py"
    ),
    "bright_query_v2_init": (
        "replication_runtime/bright_query_generator_v2/__init__.py"
    ),
    "bright_query_v2_worker": (
        "replication_runtime/bright_query_generator_v2/worker.py"
    ),
    "dependency_closure": (
        "replication_runtime/hitab_p1_formal_v1/dependency_closure.py"
    ),
    "hitab_acquire_unit": (
        "manifests/hitab_p1_source_acquisition_unit_v1.service"
    ),
    "hitab_canary": (
        "assumption_agent/benchmarks/hitab_p1_public_canary_v1.py"
    ),
    "hitab_canary_unit": (
        "manifests/hitab_p1_source_free_canary_unit_v1.service"
    ),
    "hitab_core": (
        "assumption_agent/benchmarks/hitab_p1_dmc1_core_v1.py"
    ),
    "hitab_formal_controller": (
        "assumption_agent/benchmarks/hitab_p1_formal_controller_v1.py"
    ),
    "hitab_formal_init": (
        "replication_runtime/hitab_p1_formal_v1/__init__.py"
    ),
    "hitab_formal_unit": "manifests/hitab_p1_formal_unit_v1.service",
    "hitab_runtime": (
        "assumption_agent/benchmarks/hitab_p1_runtime_v1.py"
    ),
    "hitab_source_acquisition": (
        "assumption_agent/benchmarks/hitab_p1_source_acquisition_v1.py"
    ),
    "hitab_source_custody": (
        "manifests/hitab_p1_public_source_custody_v1.json"
    ),
    "hitab_study_design": (
        "manifests/hitab_p1_dmc1_hierarchical_set_evaluator_design_v1.json"
    ),
    "hitab_v2_implementation_addendum": (
        "manifests/hitab_p1_direct_transformers_minilm_addendum_v2.json"
    ),
    "hitab_v3_implementation_addendum": (
        "manifests/hitab_p1_child_cwd_sanitization_addendum_v3.json"
    ),
    "hitab_v4_implementation_addendum": (
        "manifests/hitab_p1_sealed_child_sys_path_addendum_v4.json"
    ),
    "hippo_contract": (
        "replication_runtime/birco_official_hipporag_v1/contract.py"
    ),
    "hippo_init": (
        "replication_runtime/birco_official_hipporag_v1/__init__.py"
    ),
    "hippo_runtime_attestation_builder": (
        "replication_runtime/tatqa_p19_v1/runtime_attestation_v1.py"
    ),
    "hippo_runtime_attestation_receipt": (
        "manifests/tatqa_p19_hipporag_runtime_attestation_v1.json"
    ),
    "hippo_worker": (
        "replication_runtime/birco_official_hipporag_v1/worker.py"
    ),
    "production_runner": (
        "replication_runtime/hitab_p1_formal_v1/runner.py"
    ),
    "qasper_minilm_binding": (
        "replication_runtime/qasper_minilm_v1/binding.py"
    ),
    "qasper_minilm_init": (
        "replication_runtime/qasper_minilm_v1/__init__.py"
    ),
    "replication_runtime_init": "replication_runtime/__init__.py",
}
REQUIRED_FILE_LABELS = frozenset(REQUIRED_PROJECT_FILES)
REQUIRED_MODEL_LABELS = frozenset(
    {
        "cross_encoder",
        "hippo_embedding",
        "hippo_llm",
        "minilm",
        "planner",
    }
)

OUTER_REQUIRED_DISTRIBUTIONS: dict[str, tuple[str, str]] = {
    "huggingface_hub": ("huggingface-hub", "1.11.0"),
    "numpy": ("numpy", "2.2.6"),
    "safetensors": ("safetensors", "0.7.0"),
    "tokenizers": ("tokenizers", "0.22.2"),
    "torch": ("torch", "2.8.0+cu128"),
    "transformers": ("transformers", "5.10.1"),
}
HIPPORAG_REQUIRED_DISTRIBUTIONS: dict[str, tuple[str, str]] = {
    "gritlm": ("gritlm", "1.0.2"),
    "hipporag": ("hipporag", "2.0.0a4"),
    "huggingface_hub": ("huggingface-hub", "0.25.2"),
    "igraph": ("python-igraph", "0.11.8"),
    "litellm": ("litellm", "1.73.1"),
    "networkx": ("networkx", "3.3"),
    "numpy": ("numpy", "2.1.3"),
    "openai": ("openai", "1.91.0"),
    "pydantic": ("pydantic", "2.10.4"),
    "safetensors": ("safetensors", "0.4.5"),
    "scipy": ("scipy", "1.14.1"),
    "sklearn": ("scikit-learn", "1.5.2"),
    "sentence_transformers": ("sentence-transformers", "3.1.1"),
    "tenacity": ("tenacity", "8.5.0"),
    "tiktoken": ("tiktoken", "0.7.0"),
    "tokenizers": ("tokenizers", "0.20.3"),
    "torch": ("torch", "2.4.1+cu118"),
    "transformers": ("transformers", "4.45.2"),
}
OUTER_REQUIRED_PROJECT_IMPORTS: dict[str, str] = {
    "assumption_agent.benchmarks.hitab_p1_runtime_v1": "hitab_runtime",
    "replication_runtime.bright_cross_encoder_v1.worker": (
        "bright_cross_encoder_worker"
    ),
    "replication_runtime.bright_minilm_v1.encoder": "bright_minilm_encoder",
    "replication_runtime.bright_query_generator_v1.worker": (
        "bright_query_v1_worker"
    ),
    "replication_runtime.bright_query_generator_v2.worker": (
        "bright_query_v2_worker"
    ),
    "replication_runtime.qasper_minilm_v1.binding": (
        "qasper_minilm_binding"
    ),
}
HIPPORAG_REQUIRED_PROJECT_IMPORTS: dict[str, str] = {
    HIPPORAG_WORKER_MODULE: "hippo_worker",
}
REQUIRED_STDLIB_IMPORTS = frozenset({"hashlib", "json", "ssl"})
EXPECTED_HIPPORAG_ATTESTATION_BUILDER_SHA256 = (
    "8344353e326e0c5f986bd29a6aea65903a1271c3444e8ca95d372317a072be07"
)
EXPECTED_HIPPORAG_ATTESTATION_FILE_SHA256 = (
    "96479f597bbf6ae9f69998df375816db9d870634d787976513ccb5bbef173955"
)
EXPECTED_HIPPORAG_ATTESTATION_RECEIPT_SHA256 = (
    "f12863b59a83e19188ccbf35208cafdf2b7c857daf404749a58e7f7787a07618"
)
EXPECTED_V2_IMPLEMENTATION_ADDENDUM_SELF_SHA256 = (
    "b5cb382ec40b4ffbac0648968b286665bed6de64ec705b644fcdff4607174149"
)
EXPECTED_V3_IMPLEMENTATION_ADDENDUM_SELF_SHA256 = (
    "fe55b40f9612510751d6dc837ff35076159d68ca25dd7486f12a5e86a61ca506"
)
EXPECTED_V4_IMPLEMENTATION_ADDENDUM_SELF_SHA256 = (
    "b15c6f807ac51f4f84c3ce58c8be68a4dda0ecdfc47ae75b58d919d1072c91c9"
)
EXPECTED_HIPPORAG_SOURCE_LEGACY_TREE_SHA256 = (
    "a644ab2811db2739db3cfbdc051561e2cfdf2ed87286f8ebd00a5971d189cdd5"
)
# The reusable P17 receipt intentionally included 36 path-sensitive ``pyc``
# files.  Five independently preserved copies agree byte-for-byte on the 60
# non-bytecode files below, while one generated ``HippoRAG.pyc`` has drifted.
# Production therefore imports a study-local bytecode-free projection and
# binds both this portable source commitment and the original P17 receipt.
EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT = 60
EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES = 332110
EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256 = (
    "925e2a305659cc7ae39464b09e64c800b28455fcf878caebbe81c9f783ec3e4c"
)
HIPPORAG_SOURCE_PROJECTION_POLICY = (
    "same_host_direct_copy_excluding_every___pycache___component_and_"
    "pyc_or_pyo_file_before_any_hitab_source_access_v1"
)
EXPECTED_HIPPORAG_LLM_TREE_SHA256 = (
    "d626d755c99c006761d5e069aa85a73fe8b011c6c0f5d0323a6f8de85246bcb5"
)
EXPECTED_HIPPORAG_EMBEDDING_TREE_SHA256 = (
    "42d8d798e4f01e68d9bb10634b9c712de00f7f8495271636fd6311b2db58e506"
)

_COMMON_OFFLINE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/home/erzhu419",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPYCACHEPREFIX": "/dev/null",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "VECLIB_MAXIMUM_THREADS": "4",
    "XDG_RUNTIME_DIR": "/run/user/1001",
}
_ACQUISITION_ENVIRONMENT = {
    "HOME": "/home/erzhu419",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPYCACHEPREFIX": "/dev/null",
}
RUNTIME_POLICY: dict[str, object] = {
    "agent_formation_physical_gpu": 0,
    "cpu_threads_per_gpu_lane": CPU_THREADS_PER_GPU_LANE,
    "dependency_closure_scope": (
        "python_executable_pyvenv_stdlib_declared_project_files_and_"
        "declared_dependency_trees_excluding_external_dynamic_libraries"
    ),
    "hipporag_logical_lane_count": 2,
    "hipporag_maximum_processes_per_gpu": 1,
    "network_or_API_call_count": 0,
    "offline_evaluation_only": True,
    "physical_GPUs": [0, 1],
    "retry_replay_resample_provider_model_candidate_or_gate_change_count": 0,
    "source_free_canary_repeat_count": 2,
    "source_acquisition_systemd": {
        "CPUQuota": "800%",
        "KillMode": "control-group",
        "MemoryMax": "40G",
        "Restart": "no",
        "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6",
        "TasksMax": 64,
        "TimeoutStartSec": "infinity",
        "network_stage": "four_exact_commit_addressed_GETs_only",
    },
    "systemd": {
        "CPUQuota": "800%",
        "IPAddressDeny": "any",
        "KillMode": "control-group",
        "MemoryMax": "40G",
        "Restart": "no",
        "RestrictAddressFamilies": "AF_UNIX",
        "TasksMax": 64,
        "TimeoutStartSec": "infinity",
    },
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")


class HitabP1ProductionRuntimeError(RuntimeError):
    """A freeze, asset, process, canary, or one-shot boundary drifted."""


class ProductionBindingBuilder(Protocol):
    def __call__(
        self, implementation: "FrozenImplementation", runtime_root: Path
    ) -> "ProductionBindings": ...


class AcquisitionFactoryLoader(Protocol):
    def __call__(self, execution: "FrozenExecution") -> object: ...


class SourceAcquisitionRunner(Protocol):
    def __call__(self, acquisition: "FrozenAcquisition") -> Mapping[str, object]: ...


class RuntimePreparer(Protocol):
    def __call__(
        self,
        implementation: "FrozenImplementation",
        *,
        verify_hippo_child: bool,
    ) -> Mapping[str, object]: ...


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HitabP1ProductionRuntimeError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def _hex64(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise HitabP1ProductionRuntimeError(f"{field} is not SHA-256")
    return value


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise HitabP1ProductionRuntimeError("self hash was supplied twice")
    output = dict(body)
    output["self_sha256"] = stable_hash(output)
    return output


def _verify_self(value: Mapping[str, object], *, field: str) -> str:
    body = dict(value)
    claimed = _hex64(body.pop("self_sha256", None), field=field)
    if not hashlib.sha256(
        canonical_bytes(body, newline=False)
    ).hexdigest() == claimed:
        raise HitabP1ProductionRuntimeError(f"{field} drifted")
    return claimed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise HitabP1ProductionRuntimeError("frozen file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HitabP1ProductionRuntimeError(
                "frozen file is not regular"
            )
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise HitabP1ProductionRuntimeError(
                "frozen file changed while read"
            )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def model_tree_sha256(root: Path) -> str:
    """Commit every regular file in a direct, symlink-free model tree."""

    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != root
    ):
        raise HitabP1ProductionRuntimeError(
            "model root is not a direct absolute directory"
        )
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise HitabP1ProductionRuntimeError(
                "model tree contains a symlink"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise HitabP1ProductionRuntimeError(
                "model tree contains a nonregular entry"
            )
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not rows:
        raise HitabP1ProductionRuntimeError("model tree is empty")
    return stable_hash(rows)


def _read_canonical_json(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = MAXIMUM_JSON_BYTES,
) -> dict[str, object]:
    if not path.is_absolute() or path.is_symlink():
        raise HitabP1ProductionRuntimeError(
            f"{label} is not a direct absolute file"
        )
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise HitabP1ProductionRuntimeError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 2
            or before.st_size > maximum_bytes
        ):
            raise HitabP1ProductionRuntimeError(
                f"{label} is not a bounded regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise HitabP1ProductionRuntimeError(
                f"{label} changed while read"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HitabP1ProductionRuntimeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise HitabP1ProductionRuntimeError(
            f"{label} is not canonical JSON"
        )
    return value


def _write_exclusive(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    if not path.is_absolute():
        raise HitabP1ProductionRuntimeError(
            "exclusive output path is not absolute"
        )
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise HitabP1ProductionRuntimeError(
            "exclusive output parent is unsafe"
        )
    os.chmod(path.parent, 0o700)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
    except OSError as exc:
        raise HitabP1ProductionRuntimeError(
            "exclusive output already exists or is unavailable"
        ) from exc
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _direct_project_file(
    project_root: Path, relative: object, *, field: str
) -> Path:
    if not isinstance(relative, str):
        raise HitabP1ProductionRuntimeError(f"{field} path is invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise HitabP1ProductionRuntimeError(f"{field} path is unsafe")
    path = project_root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise HitabP1ProductionRuntimeError(
            f"{field} is not a direct project file"
        )
    return path


def _reject_unbound_project_bytecode(project_root: Path) -> None:
    """Reject project bytecode that is outside the per-source-file freeze.

    ``-B`` suppresses writes but still permits reads of a valid adjacent
    cache.  ``PYTHONPYCACHEPREFIX=/dev/null`` redirects normal cache lookup;
    this scan additionally rejects sourceless or explicitly imported project
    bytecode.  Dependency/stdlib bytecode is allowed only because those trees
    are recursively content-bound.
    """

    try:
        for current, directories, files in os.walk(
            project_root, followlinks=False
        ):
            base = Path(current)
            for name in directories:
                candidate = base / name
                if name == "__pycache__" or name.endswith((".pyc", ".pyo")):
                    raise HitabP1ProductionRuntimeError(
                        "project root contains unbound Python bytecode"
                    )
                if candidate.is_symlink():
                    continue
            for name in files:
                if name.endswith((".pyc", ".pyo")):
                    raise HitabP1ProductionRuntimeError(
                        "project root contains unbound Python bytecode"
                    )
    except HitabP1ProductionRuntimeError:
        raise
    except OSError as exc:
        raise HitabP1ProductionRuntimeError(
            "project root bytecode audit failed"
        ) from exc


def _validate_frozen_unit(
    path: Path,
    *,
    mode: str,
    project_root: Path,
    python_executable: Path,
) -> None:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise HitabP1ProductionRuntimeError(
            f"{mode} unit is unreadable"
        ) from exc
    lines = text.splitlines()
    expected_properties = {
        "CPUQuota=800%",
        "KillMode=control-group",
        "MemoryMax=40G",
        "Restart=no",
        "TasksMax=64",
        "TimeoutStartSec=infinity",
        "Type=oneshot",
        "UMask=0077",
        f"WorkingDirectory={project_root}",
    }
    if not expected_properties.issubset(lines):
        raise HitabP1ProductionRuntimeError(
            f"{mode} unit policy drifted"
        )
    exec_directives = [line for line in lines if line.startswith("Exec")]
    exec_lines = [line for line in exec_directives if line.startswith("ExecStart=")]
    if len(exec_lines) != 1:
        raise HitabP1ProductionRuntimeError(
            f"{mode} unit ExecStart count drifted"
        )
    if exec_directives != exec_lines:
        raise HitabP1ProductionRuntimeError(
            f"{mode} unit has an unfrozen auxiliary command"
        )
    try:
        command = shlex.split(exec_lines[0][len("ExecStart=") :])
    except ValueError as exc:
        raise HitabP1ProductionRuntimeError(
            f"{mode} unit ExecStart is not parseable"
        ) from exc
    environment = dict(_ACQUISITION_ENVIRONMENT)
    environment["PYTHONPATH"] = str(project_root)
    if mode == "canary":
        environment = dict(_COMMON_OFFLINE_ENVIRONMENT)
        environment["PYTHONPATH"] = str(project_root)
        expected_arguments = [
            "canary",
            "--implementation-freeze",
            str(
                project_root
                / "manifests/hitab_p1_implementation_freeze_v1.json"
            ),
            "--output",
            str(project_root.parent / "receipts/source_free_canary.json"),
        ]
        expected_network = {
            "IPAddressDeny=any",
            "RestrictAddressFamilies=AF_UNIX",
        }
    elif mode == "formal":
        environment = dict(_COMMON_OFFLINE_ENVIRONMENT)
        environment["PYTHONPATH"] = str(project_root)
        expected_arguments = [
            "formal",
            "--execution-freeze",
            str(
                project_root
                / "manifests/hitab_p1_execution_freeze_v1.json"
            ),
        ]
        expected_network = {
            "IPAddressDeny=any",
            "RestrictAddressFamilies=AF_UNIX",
        }
    elif mode == "acquire":
        expected_arguments = [
            "acquire",
            "--acquisition-freeze",
            str(
                project_root
                / "manifests/hitab_p1_source_acquisition_freeze_v1.json"
            ),
        ]
        expected_network = {
            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        }
    else:
        raise HitabP1ProductionRuntimeError("unit mode is invalid")
    network_lines = {
        line
        for line in lines
        if line.startswith("IPAddress")
        or line.startswith("RestrictAddressFamilies=")
    }
    expected_command = [
        "/usr/bin/env",
        "-i",
        *(f"{key}={value}" for key, value in sorted(environment.items())),
        str(python_executable),
        "-S",
        "-B",
        "-c",
        OUTER_ENTRYPOINT_SCRIPT,
        *expected_arguments,
    ]
    if command != expected_command or network_lines != expected_network:
        raise HitabP1ProductionRuntimeError(
            f"{mode} unit launch envelope drifted"
        )


@dataclass(frozen=True)
class FrozenPythonRuntime:
    role: str
    executable: Path
    resolved_target: Path
    resolved_target_receipt: Mapping[str, object]
    pyvenv_cfg: Path
    pyvenv_cfg_receipt: Mapping[str, object]
    python_version: str
    stdlib_root: Path
    stdlib_tree_receipt: Mapping[str, object]
    python_zip_path: Path
    ordered_roots: tuple[Path, ...]
    tree_receipts: tuple[Mapping[str, object], ...]
    import_probe: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class FrozenImplementation:
    path: Path
    self_sha256: str
    project_root: Path
    outer_runtime: FrozenPythonRuntime
    hippo_runtime: FrozenPythonRuntime
    files: Mapping[str, Path]
    file_sha256s: Mapping[str, str]
    models: Mapping[str, Path]
    model_tree_sha256s: Mapping[str, str]
    minilm_asset_manifest: Path
    minilm_asset_manifest_sha256: str
    hippo_source_root: Path
    hippo_source_tree_receipt: Mapping[str, object]
    hippo_source_file_count: int
    hippo_source_size_bytes: int
    hippo_source_tree_sha256: str
    hippo_legacy_source_root: Path
    hippo_worker_module: str
    runtime_policy: Mapping[str, object]

    @property
    def python_executable(self) -> Path:
        """Compatibility alias for the outer typed interpreter."""

        return self.outer_runtime.executable

    @property
    def python_target_sha256(self) -> str:
        """Compatibility alias for the outer resolved target digest."""

        return str(
            self.outer_runtime.resolved_target_receipt["content_sha256"]
        )


@dataclass(frozen=True)
class FrozenAcquisition:
    path: Path
    self_sha256: str
    implementation: FrozenImplementation
    canary_receipt_path: Path
    canary_receipt_self_sha256: str
    source_root: Path
    control_root: Path


@dataclass(frozen=True)
class FrozenExecution:
    path: Path
    self_sha256: str
    implementation: FrozenImplementation
    canary_receipt_path: Path
    canary_receipt_self_sha256: str
    source_receipt_path: Path
    source_receipt_self_sha256: str
    source_identity_commitment: str
    source_paths: Mapping[str, Path]
    source_sha256s: Mapping[str, str]
    formal_work_root: Path
    acquisition_factory_module: str
    acquisition_factory_attribute: str
    acquisition_factory_file_label: str


@dataclass(frozen=True)
class ProductionBindings:
    planner_runner: object
    cross_encoder_scorer: object
    minilm_encoder: object
    hippo_runner: object
    gpu0_cache_releaser: object


@dataclass(frozen=True)
class _DiagnosticResult:
    full_stream_sha256: str
    total_size_bytes: int
    persisted_size_bytes: int
    truncated: bool

    def payload(self) -> dict[str, object]:
        return {
            "full_stream_sha256": self.full_stream_sha256,
            "persisted_size_bytes": self.persisted_size_bytes,
            "total_size_bytes": self.total_size_bytes,
            "truncated": self.truncated,
        }


class _BoundedDiagnosticCapture:
    """Drain an inherited pipe while persisting at most one fixed MiB."""

    def __init__(self, path: Path) -> None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            self._persist_descriptor = os.open(path, flags, 0o600)
            os.fchmod(self._persist_descriptor, 0o600)
            self._read_descriptor, self.write_descriptor = os.pipe()
        except OSError as exc:
            try:
                os.close(self._persist_descriptor)
            except (AttributeError, OSError):
                pass
            raise HitabP1ProductionRuntimeError(
                "private diagnostic capture is unavailable"
            ) from exc
        self._result: _DiagnosticResult | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._drain,
            name=f"hitab-bounded-diagnostic-{path.name}",
            daemon=False,
        )
        self._thread.start()

    def _drain(self) -> None:
        digest = hashlib.sha256()
        total = 0
        persisted = 0
        try:
            while True:
                chunk = os.read(self._read_descriptor, 64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
                if persisted < MAXIMUM_DIAGNOSTIC_PERSISTED_BYTES:
                    remaining = (
                        MAXIMUM_DIAGNOSTIC_PERSISTED_BYTES - persisted
                    )
                    view = memoryview(chunk[:remaining])
                    offset = 0
                    while offset < len(view):
                        count = os.write(
                            self._persist_descriptor, view[offset:]
                        )
                        if count <= 0:
                            raise OSError("diagnostic write returned zero")
                        offset += count
                    persisted += len(view)
            os.fsync(self._persist_descriptor)
            self._result = _DiagnosticResult(
                full_stream_sha256=digest.hexdigest(),
                total_size_bytes=total,
                persisted_size_bytes=persisted,
                truncated=total > persisted,
            )
        except BaseException as exc:
            self._error = exc
        finally:
            for descriptor in (
                self._read_descriptor,
                self._persist_descriptor,
            ):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def finish(self) -> _DiagnosticResult:
        try:
            os.close(self.write_descriptor)
        except OSError:
            pass
        self._thread.join()
        if self._error is not None or self._result is None:
            raise HitabP1ProductionRuntimeError(
                "private diagnostic capture failed"
            ) from self._error
        return self._result


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_reusable_hipporag_attestation(
    files: Mapping[str, Path],
    file_hashes: Mapping[str, str],
) -> Mapping[str, object]:
    if (
        file_hashes["hippo_runtime_attestation_builder"]
        != EXPECTED_HIPPORAG_ATTESTATION_BUILDER_SHA256
        or file_hashes["hippo_runtime_attestation_receipt"]
        != EXPECTED_HIPPORAG_ATTESTATION_FILE_SHA256
    ):
        raise HitabP1ProductionRuntimeError(
            "reusable HippoRAG attestation implementation drifted"
        )
    value = _read_canonical_json(
        files["hippo_runtime_attestation_receipt"],
        label="reusable HippoRAG runtime attestation",
    )
    try:
        assets = value["asset_bindings"]
        source_tree = assets["HippoRAG_source"]["tree"]
        llm_tree = assets["SmolLM"]["tree"]
        embedding_tree = assets["MiniLM"]["generic_tree"]
        module_rows = value["active_module_binding"]["rows"]
        decision = value["compatibility_decision"]
    except (KeyError, TypeError) as exc:
        raise HitabP1ProductionRuntimeError(
            "reusable HippoRAG attestation is malformed"
        ) from exc
    hippo_rows = (
        [
            row
            for row in module_rows
            if isinstance(row, dict)
            and row.get("distribution_name") == "hipporag"
            and row.get("module_name") == "hipporag"
        ]
        if isinstance(module_rows, list)
        else []
    )
    if (
        value.get("schema")
        != "tatqa_p19_hipporag_runtime_attestation_v1"
        or value.get("receipt_sha256")
        != EXPECTED_HIPPORAG_ATTESTATION_RECEIPT_SHA256
        or not isinstance(source_tree, dict)
        or source_tree.get("tree_sha256")
        != EXPECTED_HIPPORAG_SOURCE_LEGACY_TREE_SHA256
        or not isinstance(llm_tree, dict)
        or llm_tree.get("tree_sha256")
        != EXPECTED_HIPPORAG_LLM_TREE_SHA256
        or not isinstance(embedding_tree, dict)
        or embedding_tree.get("tree_sha256")
        != EXPECTED_HIPPORAG_EMBEDDING_TREE_SHA256
        or len(hippo_rows) != 1
        or not isinstance(hippo_rows[0].get("origin_path"), str)
        or _HEX64.fullmatch(
            str(hippo_rows[0].get("origin_file_sha256"))
        )
        is None
        or not isinstance(decision, dict)
        or decision.get(
            "P17_HippoRAG_interpreter_is_exact_QASPER_controller_runtime"
        )
        is not False
    ):
        raise HitabP1ProductionRuntimeError(
            "reusable HippoRAG attestation decision drifted"
        )
    return {
        "embedding_tree_sha256": embedding_tree["tree_sha256"],
        "hipporag_origin_file_sha256": hippo_rows[0][
            "origin_file_sha256"
        ],
        "hipporag_origin_path": hippo_rows[0]["origin_path"],
        "llm_tree_sha256": llm_tree["tree_sha256"],
        "source_tree_sha256": source_tree["tree_sha256"],
    }


def _validate_clean_hipporag_source_tree(
    source_root: Path,
    *,
    expected_file_count: int,
    expected_size_bytes: int,
    expected_tree_sha256: str,
) -> None:
    """Revalidate the portable source projection, including hardlink state."""

    source_root = Path(source_root)
    try:
        root_metadata = source_root.lstat()
        entries = tuple(source_root.iterdir())
    except OSError as exc:
        raise HitabP1ProductionRuntimeError(
            "clean HippoRAG source projection is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or source_root.resolve() != source_root
        or {entry.name for entry in entries} != {"src"}
        or (source_root / "src").is_symlink()
        or not (source_root / "src").is_dir()
    ):
        raise HitabP1ProductionRuntimeError(
            "clean HippoRAG source projection root drifted"
        )
    try:
        nodes = tuple(source_root.rglob("*"))
        file_rows = tuple(
            (path, path.lstat())
            for path in nodes
            if stat.S_ISREG(path.lstat().st_mode)
        )
    except OSError as exc:
        raise HitabP1ProductionRuntimeError(
            "clean HippoRAG source projection could not be scanned"
        ) from exc
    if (
        any(
            stat.S_ISLNK(path.lstat().st_mode)
            or not (
                stat.S_ISDIR(path.lstat().st_mode)
                or stat.S_ISREG(path.lstat().st_mode)
            )
            for path in nodes
        )
        or any(
            "__pycache__" in path.relative_to(source_root).parts
            for path in nodes
        )
        or any(metadata.st_nlink != 1 for _path, metadata in file_rows)
        or any(
            path.suffix in {".pyc", ".pyo"}
            or "__pycache__" in path.relative_to(source_root).parts
            for path, _metadata in file_rows
        )
        or len(file_rows)
        != expected_file_count
        or sum(metadata.st_size for _path, metadata in file_rows)
        != expected_size_bytes
        or model_tree_sha256(source_root)
        != expected_tree_sha256
    ):
        raise HitabP1ProductionRuntimeError(
            "current HippoRAG source is not the frozen clean projection"
        )


def _validate_current_hipporag_attestation_linkage(
    attestation: Mapping[str, object],
    *,
    runtime: FrozenPythonRuntime,
    model_hashes: Mapping[str, str],
    clean_source_root: Path,
) -> None:
    """Bind the portable source projection to the reused origin and models."""

    probe = runtime.import_probe.get("hipporag")
    if not isinstance(probe, Mapping):
        raise HitabP1ProductionRuntimeError(
            "current HippoRAG import binding is absent"
        )
    origin = Path(str(probe.get("origin_path")))
    if (
        probe.get("origin_receipt", {}).get("content_sha256")
        != attestation["hipporag_origin_file_sha256"]
        or model_hashes.get("hippo_llm")
        != attestation["llm_tree_sha256"]
        or model_hashes.get("hippo_embedding")
        != attestation["embedding_tree_sha256"]
        or origin.name != "__init__.py"
        or origin.parent.name != "hipporag"
        or origin.parent.parent.name != "src"
    ):
        raise HitabP1ProductionRuntimeError(
            "current HippoRAG source or model binding is not attested"
        )
    source_root = origin.parent.parent.parent
    clean_source_root = Path(clean_source_root)
    if (
        source_root != clean_source_root
        or source_root / "src" not in runtime.ordered_roots
    ):
        raise HitabP1ProductionRuntimeError(
            "current HippoRAG import is outside the clean projection"
        )
    _validate_clean_hipporag_source_tree(
        source_root,
        expected_file_count=EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT,
        expected_size_bytes=EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES,
        expected_tree_sha256=EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256,
    )


def _load_python_runtime(
    *,
    role: str,
    raw_python: object,
    raw_dependency: object,
    required_distributions: Mapping[str, tuple[str, str]],
    required_project_imports: Mapping[str, str],
    project_root: Path,
    project_files: Mapping[str, Path],
    closure: object,
) -> FrozenPythonRuntime:
    if role not in {"outer", "hippo_child"}:
        raise HitabP1ProductionRuntimeError("Python runtime role is invalid")
    if not isinstance(raw_python, dict) or set(raw_python) != {
        "executable_path",
        "lexical_symlink_target",
        "pyvenv_cfg",
        "python_version",
        "resolved_target",
        "stdlib_root",
    }:
        raise HitabP1ProductionRuntimeError(
            f"{role} Python binding is malformed"
        )
    executable_raw = raw_python["executable_path"]
    if not isinstance(executable_raw, str):
        raise HitabP1ProductionRuntimeError(
            f"{role} Python path is invalid"
        )
    executable = Path(executable_raw)
    if not executable.is_absolute():
        raise HitabP1ProductionRuntimeError(
            f"{role} Python path is not absolute"
        )
    try:
        executable_lstat = executable.lstat()
        resolved = executable.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HitabP1ProductionRuntimeError(
            f"{role} Python executable is unavailable"
        ) from exc
    declared_link = raw_python["lexical_symlink_target"]
    if stat.S_ISLNK(executable_lstat.st_mode):
        try:
            actual_link: str | None = os.readlink(executable)
        except OSError as exc:
            raise HitabP1ProductionRuntimeError(
                f"{role} Python symlink is unreadable"
            ) from exc
    elif stat.S_ISREG(executable_lstat.st_mode):
        actual_link = None
    else:
        raise HitabP1ProductionRuntimeError(
            f"{role} Python lexical path is unsafe"
        )
    if declared_link != actual_link or not os.access(executable, os.X_OK):
        raise HitabP1ProductionRuntimeError(
            f"{role} Python lexical binding drifted"
        )

    target_row = raw_python["resolved_target"]
    pyvenv_row = raw_python["pyvenv_cfg"]
    if (
        not isinstance(target_row, dict)
        or set(target_row) != {"path", "receipt"}
        or not isinstance(target_row["path"], str)
        or not isinstance(target_row["receipt"], dict)
        or not isinstance(pyvenv_row, dict)
        or set(pyvenv_row) != {"path", "receipt"}
        or not isinstance(pyvenv_row["path"], str)
        or not isinstance(pyvenv_row["receipt"], dict)
    ):
        raise HitabP1ProductionRuntimeError(
            f"{role} Python file receipts are malformed"
        )
    target = Path(target_row["path"])
    pyvenv_cfg = Path(pyvenv_row["path"])
    if (
        target != resolved
        or pyvenv_cfg
        != executable.parent.parent / "pyvenv.cfg"
    ):
        raise HitabP1ProductionRuntimeError(
            f"{role} Python target or structural pyvenv binding drifted"
        )
    try:
        target_receipt = closure.verify_regular_file_receipt(
            target, target_row["receipt"]
        )
        pyvenv_receipt = closure.verify_regular_file_receipt(
            pyvenv_cfg, pyvenv_row["receipt"]
        )
    except Exception as exc:
        raise HitabP1ProductionRuntimeError(
            f"{role} Python file closure drifted"
        ) from exc
    python_version = raw_python["python_version"]
    if python_version != "3.10.12":
        raise HitabP1ProductionRuntimeError(
            f"{role} Python version is not the frozen 3.10.12 runtime"
        )
    raw_stdlib = raw_python["stdlib_root"]
    if (
        not isinstance(raw_stdlib, dict)
        or set(raw_stdlib) != {"path", "tree_receipt"}
        or not isinstance(raw_stdlib["path"], str)
        or not isinstance(raw_stdlib["tree_receipt"], dict)
    ):
        raise HitabP1ProductionRuntimeError(
            f"{role} stdlib closure is malformed"
        )
    stdlib_root = Path(raw_stdlib["path"])
    expected_stdlib_root = (
        target.parent.parent / "lib" / "python3.10"
    )
    python_zip = stdlib_root.parent / "python310.zip"
    if (
        stdlib_root != expected_stdlib_root
        or python_zip.exists()
        or python_zip.is_symlink()
    ):
        raise HitabP1ProductionRuntimeError(
            f"{role} stdlib root or automatic Python zip path drifted"
        )
    try:
        stdlib_tree_receipt = closure.verify_tree_receipt(
            stdlib_root, raw_stdlib["tree_receipt"]
        )
    except Exception as exc:
        raise HitabP1ProductionRuntimeError(
            f"{role} stdlib tree drifted"
        ) from exc

    if not isinstance(raw_dependency, dict) or set(raw_dependency) != {
        "import_probe",
        "ordered_roots",
    }:
        raise HitabP1ProductionRuntimeError(
            f"{role} dependency closure is malformed"
        )
    raw_roots = raw_dependency["ordered_roots"]
    if not isinstance(raw_roots, list) or not raw_roots:
        raise HitabP1ProductionRuntimeError(
            f"{role} dependency roots are absent"
        )
    roots: list[Path] = []
    receipts: list[Mapping[str, object]] = []
    for ordinal, row in enumerate(raw_roots):
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "tree_receipt"}
            or not isinstance(row["path"], str)
            or not isinstance(row["tree_receipt"], dict)
        ):
            raise HitabP1ProductionRuntimeError(
                f"{role} dependency root {ordinal} is malformed"
            )
        root = Path(row["path"])
        if (
            not root.is_absolute()
            or root == project_root
            or root == stdlib_root
            or _path_is_within(root, stdlib_root)
            or _path_is_within(stdlib_root, root)
            or root in roots
        ):
            raise HitabP1ProductionRuntimeError(
                f"{role} dependency root {ordinal} is unsafe"
            )
        try:
            receipt = closure.verify_tree_receipt(
                root, row["tree_receipt"]
            )
        except Exception as exc:
            raise HitabP1ProductionRuntimeError(
                f"{role} dependency root {ordinal} drifted"
            ) from exc
        roots.append(root)
        receipts.append(receipt)

    raw_probe = raw_dependency["import_probe"]
    required_modules = (
        set(required_distributions)
        | set(required_project_imports)
        | set(REQUIRED_STDLIB_IMPORTS)
    )
    if not isinstance(raw_probe, dict) or set(raw_probe) != required_modules:
        raise HitabP1ProductionRuntimeError(
            f"{role} import probe closure is not exact"
        )
    probe: dict[str, Mapping[str, object]] = {}
    for module_name, row in sorted(raw_probe.items()):
        if (
            not isinstance(module_name, str)
            or not isinstance(row, dict)
            or set(row)
            != {
                "distribution",
                "origin_path",
                "origin_receipt",
                "version",
            }
            or not isinstance(row["origin_path"], str)
            or not isinstance(row["origin_receipt"], dict)
        ):
            raise HitabP1ProductionRuntimeError(
                f"{role} import probe row is malformed"
            )
        origin = Path(row["origin_path"])
        try:
            origin_receipt = closure.verify_regular_file_receipt(
                origin, row["origin_receipt"]
            )
        except Exception as exc:
            raise HitabP1ProductionRuntimeError(
                f"{role} import origin {module_name} drifted"
            ) from exc
        if module_name in required_distributions:
            distribution, version = required_distributions[module_name]
            if (
                row["distribution"] != distribution
                or row["version"] != version
                or not any(_path_is_within(origin, root) for root in roots)
            ):
                raise HitabP1ProductionRuntimeError(
                    f"{role} distribution {module_name} drifted"
                )
        elif module_name in required_project_imports:
            label = required_project_imports[module_name]
            if (
                row["distribution"] is not None
                or row["version"] is not None
                or origin != project_files[label]
            ):
                raise HitabP1ProductionRuntimeError(
                    f"{role} project import {module_name} drifted"
                )
        elif (
            row["distribution"] is not None
            or row["version"] is not None
            or _path_is_within(origin, project_root)
            or any(_path_is_within(origin, root) for root in roots)
            or not _path_is_within(origin, stdlib_root)
        ):
            raise HitabP1ProductionRuntimeError(
                f"{role} stdlib import {module_name} drifted"
            )
        probe[module_name] = {
            "distribution": row["distribution"],
            "origin_path": str(origin),
            "origin_receipt": origin_receipt,
            "version": row["version"],
        }
    return FrozenPythonRuntime(
        role=role,
        executable=executable,
        resolved_target=target,
        resolved_target_receipt=target_receipt,
        pyvenv_cfg=pyvenv_cfg,
        pyvenv_cfg_receipt=pyvenv_receipt,
        python_version=python_version,
        stdlib_root=stdlib_root,
        stdlib_tree_receipt=stdlib_tree_receipt,
        python_zip_path=python_zip,
        ordered_roots=tuple(roots),
        tree_receipts=tuple(receipts),
        import_probe=probe,
    )


def load_implementation_freeze(path: Path) -> FrozenImplementation:
    path = Path(path)
    value = _read_canonical_json(path, label="implementation freeze")
    self_sha256 = _verify_self(value, field="implementation freeze self hash")
    if (
        value.get("schema") != IMPLEMENTATION_FREEZE_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("implementation_revision")
        != IMPLEMENTATION_REVISION
        or value.get("runtime_policy") != RUNTIME_POLICY
    ):
        raise HitabP1ProductionRuntimeError(
            "implementation freeze identity or policy drifted"
        )

    raw_project = value.get("project_root")
    if not isinstance(raw_project, str):
        raise HitabP1ProductionRuntimeError("project root is invalid")
    project = Path(raw_project)
    if (
        not project.is_absolute()
        or project.is_symlink()
        or not project.is_dir()
        or project.resolve() != project
    ):
        raise HitabP1ProductionRuntimeError(
            "project root is not a direct absolute directory"
        )
    _reject_unbound_project_bytecode(project)

    raw_files = value.get("files")
    if not isinstance(raw_files, dict) or set(raw_files) != REQUIRED_FILE_LABELS:
        raise HitabP1ProductionRuntimeError(
            "implementation file closure is not exact"
        )
    files: dict[str, Path] = {}
    file_hashes: dict[str, str] = {}
    for label, row in sorted(raw_files.items()):
        if (
            not isinstance(label, str)
            or _SAFE_LABEL.fullmatch(label) is None
            or not isinstance(row, dict)
            or set(row) != {"relative_path", "sha256"}
            or row.get("relative_path") != REQUIRED_PROJECT_FILES.get(label)
        ):
            raise HitabP1ProductionRuntimeError(
                "implementation file binding is malformed"
            )
        candidate = _direct_project_file(
            project, row["relative_path"], field=f"file {label}"
        )
        expected = _hex64(row["sha256"], field=f"file {label}")
        if file_sha256(candidate) != expected:
            raise HitabP1ProductionRuntimeError(
                f"implementation file {label} drifted"
            )
        files[label] = candidate
        file_hashes[label] = expected
    addendum_path = files["hitab_v2_implementation_addendum"]
    try:
        addendum = json.loads(addendum_path.read_bytes().decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HitabP1ProductionRuntimeError(
            "HiTab v2 implementation addendum is invalid"
        ) from exc
    if (
        not isinstance(addendum, dict)
        or file_sha256(addendum_path)
        != file_hashes["hitab_v2_implementation_addendum"]
    ):
        raise HitabP1ProductionRuntimeError(
            "HiTab v2 implementation addendum changed while read"
        )
    addendum_self = _verify_self(
        addendum, field="HiTab v2 implementation addendum self hash"
    )
    if (
        addendum_self
        != EXPECTED_V2_IMPLEMENTATION_ADDENDUM_SELF_SHA256
        or addendum.get("schema")
        != "hitab_p1_direct_transformers_minilm_implementation_addendum_v2"
        or addendum.get("study_id") != STUDY_ID
        or addendum.get("implementation_revision")
        != "direct_transformers_minilm_v2"
        or addendum.get("status")
        != (
            "implementation_addendum_frozen_before_any_HiTab_source_"
            "body_secret_action_qrel_or_score"
        )
    ):
        raise HitabP1ProductionRuntimeError(
            "HiTab v2 implementation addendum drifted"
        )
    v3_addendum_path = files["hitab_v3_implementation_addendum"]
    try:
        v3_addendum = json.loads(
            v3_addendum_path.read_bytes().decode("ascii")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HitabP1ProductionRuntimeError(
            "HiTab v3 implementation addendum is invalid"
        ) from exc
    if (
        not isinstance(v3_addendum, dict)
        or file_sha256(v3_addendum_path)
        != file_hashes["hitab_v3_implementation_addendum"]
    ):
        raise HitabP1ProductionRuntimeError(
            "HiTab v3 implementation addendum changed while read"
        )
    v3_addendum_self = _verify_self(
        v3_addendum,
        field="HiTab v3 implementation addendum self hash",
    )
    if (
        v3_addendum_self
        != EXPECTED_V3_IMPLEMENTATION_ADDENDUM_SELF_SHA256
        or v3_addendum.get("schema")
        != "hitab_p1_child_cwd_sanitization_implementation_addendum_v3"
        or v3_addendum.get("study_id") != STUDY_ID
        or v3_addendum.get("implementation_revision")
        != "direct_transformers_minilm_v3_child_cwd_sanitized"
        or v3_addendum.get("status")
        != (
            "implementation_addendum_frozen_before_any_HiTab_source_"
            "body_secret_action_qrel_or_score"
        )
    ):
        raise HitabP1ProductionRuntimeError(
            "HiTab v3 implementation addendum drifted"
        )
    v4_addendum_path = files["hitab_v4_implementation_addendum"]
    try:
        v4_addendum = json.loads(
            v4_addendum_path.read_bytes().decode("ascii")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HitabP1ProductionRuntimeError(
            "HiTab v4 implementation addendum is invalid"
        ) from exc
    if (
        not isinstance(v4_addendum, dict)
        or file_sha256(v4_addendum_path)
        != file_hashes["hitab_v4_implementation_addendum"]
    ):
        raise HitabP1ProductionRuntimeError(
            "HiTab v4 implementation addendum changed while read"
        )
    v4_addendum_self = _verify_self(
        v4_addendum,
        field="HiTab v4 implementation addendum self hash",
    )
    if (
        v4_addendum_self
        != EXPECTED_V4_IMPLEMENTATION_ADDENDUM_SELF_SHA256
        or v4_addendum.get("schema")
        != "hitab_p1_sealed_child_sys_path_implementation_addendum_v4"
        or v4_addendum.get("study_id") != STUDY_ID
        or v4_addendum.get("implementation_revision")
        != IMPLEMENTATION_REVISION
        or v4_addendum.get("status")
        != (
            "implementation_addendum_frozen_before_any_HiTab_source_"
            "body_secret_action_qrel_or_score"
        )
    ):
        raise HitabP1ProductionRuntimeError(
            "HiTab v4 implementation addendum drifted"
        )
    hipporag_attestation = _validate_reusable_hipporag_attestation(
        files, file_hashes
    )
    try:
        from . import dependency_closure as closure
    except ImportError as exc:
        raise HitabP1ProductionRuntimeError(
            "dependency closure verifier is unavailable"
        ) from exc
    if (
        file_sha256(Path(closure.__file__).resolve())
        != file_hashes["dependency_closure"]
    ):
        raise HitabP1ProductionRuntimeError(
            "loaded dependency closure verifier drifted"
        )

    raw_source_projection = value.get("hippo_source_projection")
    if (
        not isinstance(raw_source_projection, dict)
        or set(raw_source_projection)
        != {
            "clean_root",
            "file_count",
            "legacy_attested_root",
            "projection_policy",
            "size_bytes",
            "tree_receipt",
            "tree_sha256",
        }
        or raw_source_projection.get("projection_policy")
        != HIPPORAG_SOURCE_PROJECTION_POLICY
        or not isinstance(raw_source_projection.get("clean_root"), str)
        or not isinstance(
            raw_source_projection.get("legacy_attested_root"), str
        )
        or not isinstance(raw_source_projection.get("tree_receipt"), dict)
        or raw_source_projection.get("file_count")
        != EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT
        or raw_source_projection.get("size_bytes")
        != EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES
        or raw_source_projection.get("tree_sha256")
        != EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256
    ):
        raise HitabP1ProductionRuntimeError(
            "HippoRAG source projection binding is malformed"
        )
    clean_source_root = Path(raw_source_projection["clean_root"])
    expected_clean_source_root = (
        project.parent.parent / "runtime/hipporag_clean/HippoRAG"
    )
    legacy_source_root = Path(
        str(hipporag_attestation["hipporag_origin_path"])
    ).parents[2]
    if (
        clean_source_root != expected_clean_source_root
        or Path(raw_source_projection["legacy_attested_root"])
        != legacy_source_root
    ):
        raise HitabP1ProductionRuntimeError(
            "HippoRAG source relocation is not the exact study-local mapping"
        )
    try:
        clean_source_tree_receipt = closure.verify_tree_receipt(
            clean_source_root,
            raw_source_projection["tree_receipt"],
        )
        _validate_clean_hipporag_source_tree(
            clean_source_root,
            expected_file_count=(
                EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT
            ),
            expected_size_bytes=(
                EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES
            ),
            expected_tree_sha256=(
                EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256
            ),
        )
    except Exception as exc:
        raise HitabP1ProductionRuntimeError(
            "HippoRAG full clean source closure drifted"
        ) from exc

    raw_models = value.get("models")
    if not isinstance(raw_models, dict) or set(raw_models) != REQUIRED_MODEL_LABELS:
        raise HitabP1ProductionRuntimeError(
            "model asset closure is not exact"
        )
    models: dict[str, Path] = {}
    model_hashes: dict[str, str] = {}
    for label, row in sorted(raw_models.items()):
        if not isinstance(row, dict) or set(row) != {"path", "tree_sha256"}:
            raise HitabP1ProductionRuntimeError(
                "model binding is malformed"
            )
        raw_path = row["path"]
        if not isinstance(raw_path, str):
            raise HitabP1ProductionRuntimeError("model path is invalid")
        root = Path(raw_path)
        expected = _hex64(
            row["tree_sha256"], field=f"model tree {label}"
        )
        if model_tree_sha256(root) != expected:
            raise HitabP1ProductionRuntimeError(
                f"model tree {label} drifted"
            )
        models[label] = root
        model_hashes[label] = expected

    raw_python = value.get("python")
    raw_dependencies = value.get("dependency_closure")
    if (
        not isinstance(raw_python, dict)
        or set(raw_python) != {"hippo_child", "outer"}
        or not isinstance(raw_dependencies, dict)
        or set(raw_dependencies) != {"hippo_child", "outer"}
    ):
        raise HitabP1ProductionRuntimeError(
            "dual Python dependency closure is malformed"
        )
    outer_runtime = _load_python_runtime(
        role="outer",
        raw_python=raw_python["outer"],
        raw_dependency=raw_dependencies["outer"],
        required_distributions=OUTER_REQUIRED_DISTRIBUTIONS,
        required_project_imports=OUTER_REQUIRED_PROJECT_IMPORTS,
        project_root=project,
        project_files=files,
        closure=closure,
    )
    hippo_runtime = _load_python_runtime(
        role="hippo_child",
        raw_python=raw_python["hippo_child"],
        raw_dependency=raw_dependencies["hippo_child"],
        required_distributions=HIPPORAG_REQUIRED_DISTRIBUTIONS,
        required_project_imports=HIPPORAG_REQUIRED_PROJECT_IMPORTS,
        project_root=project,
        project_files=files,
        closure=closure,
    )
    if outer_runtime.executable == hippo_runtime.executable:
        raise HitabP1ProductionRuntimeError(
            "outer and HippoRAG runtimes are not separated"
        )
    _validate_current_hipporag_attestation_linkage(
        hipporag_attestation,
        runtime=hippo_runtime,
        model_hashes=model_hashes,
        clean_source_root=clean_source_root,
    )

    raw_manifest = value.get("minilm_asset_manifest")
    if not isinstance(raw_manifest, dict) or set(raw_manifest) != {
        "path",
        "sha256",
    }:
        raise HitabP1ProductionRuntimeError(
            "MiniLM asset manifest binding is malformed"
        )
    if not isinstance(raw_manifest["path"], str):
        raise HitabP1ProductionRuntimeError(
            "MiniLM asset manifest path is invalid"
        )
    minilm_manifest = Path(raw_manifest["path"])
    if (
        not minilm_manifest.is_absolute()
        or minilm_manifest.is_symlink()
        or not minilm_manifest.is_file()
        or minilm_manifest.resolve() != minilm_manifest
    ):
        raise HitabP1ProductionRuntimeError(
            "MiniLM asset manifest is not a direct file"
        )
    minilm_manifest_hash = _hex64(
        raw_manifest["sha256"], field="MiniLM asset manifest"
    )
    if file_sha256(minilm_manifest) != minilm_manifest_hash:
        raise HitabP1ProductionRuntimeError(
            "MiniLM asset manifest drifted"
        )

    raw_worker = value.get("hippo_worker")
    if not isinstance(raw_worker, dict) or set(raw_worker) != {
        "file_label",
        "module",
    }:
        raise HitabP1ProductionRuntimeError(
            "HippoRAG worker binding is malformed"
        )
    if (
        raw_worker["file_label"] != "hippo_worker"
        or raw_worker["module"] != HIPPORAG_WORKER_MODULE
    ):
        raise HitabP1ProductionRuntimeError(
            "HippoRAG worker identity drifted"
        )

    _validate_frozen_unit(
        files["hitab_canary_unit"],
        mode="canary",
        project_root=project,
        python_executable=outer_runtime.executable,
    )
    _validate_frozen_unit(
        files["hitab_acquire_unit"],
        mode="acquire",
        project_root=project,
        python_executable=outer_runtime.executable,
    )
    _validate_frozen_unit(
        files["hitab_formal_unit"],
        mode="formal",
        project_root=project,
        python_executable=outer_runtime.executable,
    )

    return FrozenImplementation(
        path=path,
        self_sha256=self_sha256,
        project_root=project,
        outer_runtime=outer_runtime,
        hippo_runtime=hippo_runtime,
        files=files,
        file_sha256s=file_hashes,
        models=models,
        model_tree_sha256s=model_hashes,
        minilm_asset_manifest=minilm_manifest,
        minilm_asset_manifest_sha256=minilm_manifest_hash,
        hippo_source_root=clean_source_root,
        hippo_source_tree_receipt=clean_source_tree_receipt,
        hippo_source_file_count=(
            EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT
        ),
        hippo_source_size_bytes=(
            EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES
        ),
        hippo_source_tree_sha256=(
            EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256
        ),
        hippo_legacy_source_root=legacy_source_root,
        hippo_worker_module=HIPPORAG_WORKER_MODULE,
        runtime_policy=dict(RUNTIME_POLICY),
    )


_SEAL_CHILD_SYS_PATH_SCRIPT = (
    "cwd=os.path.realpath(os.getcwd())\n"
    "sys.path[:]=[path for path in sys.path if path and "
    "os.path.isabs(path) and os.path.exists(path) and "
    "os.path.realpath(path)!=cwd]\n"
    "_frozen_child_sys_path=tuple(sys.path)\n"
    "class _FrozenChildSysPath(list):\n"
    " def _add(self,value):\n"
    "  if not isinstance(value,str):"
    " raise RuntimeError('non-string child import path')\n"
    "  if (not value) or os.path.realpath(value)==cwd"
    " or value in _frozen_child_sys_path: return\n"
    "  raise RuntimeError('unfrozen child import path mutation')\n"
    " def append(self,value): self._add(value)\n"
    " def insert(self,index,value): self._add(value)\n"
    " def extend(self,values):\n"
    "  for value in values: self._add(value)\n"
    " def __iadd__(self,values): self.extend(values); return self\n"
    " def _immutable(self,*args,**kwargs):"
    " raise RuntimeError('frozen child import path mutation')\n"
    " __setitem__=__delitem__=__imul__=clear=pop=remove=reverse=sort="
    "_immutable\n"
    "sys.path=_FrozenChildSysPath(_frozen_child_sys_path)\n"
    "_sealed_child_sys_path=sys.path\n"
)


_IMPORT_PROBE_SCRIPT = (
    "import hashlib,importlib,importlib.metadata,json,os,sys,sysconfig\n"
    + _SEAL_CHILD_SYS_PATH_SCRIPT
    + "request=json.loads(sys.argv[1])\n"
    "rows={}\n"
    "for name in sorted(request):\n"
    " module=importlib.import_module(name)\n"
    " if (sys.path is not _sealed_child_sys_path or "
    "tuple(sys.path)!=_frozen_child_sys_path):"
    " raise RuntimeError('child import path seal drifted')\n"
    " raw_origin=getattr(module,'__file__',None)\n"
    " if not isinstance(raw_origin,str): raise RuntimeError('origin absent')\n"
    " origin=os.path.realpath(raw_origin)\n"
    " digest=hashlib.sha256()\n"
    " with open(origin,'rb') as handle:\n"
    "  while True:\n"
    "   block=handle.read(1048576)\n"
    "   if not block: break\n"
    "   digest.update(block)\n"
    " distribution=request[name]\n"
    " version=(importlib.metadata.version(distribution)"
    " if distribution is not None else None)\n"
    " rows[name]={'origin_path':origin,'content_sha256':digest.hexdigest(),"
    "'version':version}\n"
    "invalid_cached=[]\n"
    "for module in tuple(sys.modules.values()):\n"
    " cached=vars(module).get('__cached__')\n"
    " if isinstance(cached,str) and"
    " (not cached.startswith('/dev/null/') or os.path.exists(cached)):\n"
    "  invalid_cached.append(cached)\n"
    "if (sys.path is not _sealed_child_sys_path or "
    "tuple(sys.path)!=_frozen_child_sys_path):"
    " raise RuntimeError('child import path seal drifted before output')\n"
    "value={'dont_write_bytecode':sys.dont_write_bytecode,"
    "'invalid_cached':sorted(set(invalid_cached)),"
    "'no_site':sys.flags.no_site,'python_version':"
    "'.'.join(str(v) for v in sys.version_info[:3]),"
    "'pycache_prefix':sys.pycache_prefix,"
    "'resolved_executable':os.path.realpath(sys.executable),"
    "'rows':rows,'stdlib_root':os.path.realpath("
    "sysconfig.get_path('stdlib')),'sys_path':sys.path}\n"
    "sys.stdout.write(json.dumps(value,allow_nan=False,ensure_ascii=True,"
    "separators=(',',':'),sort_keys=True)+'\\n')\n"
)


def _verify_runtime_filesystem_again(
    runtime: FrozenPythonRuntime,
    *,
    hippo_source_root: Path | None = None,
    hippo_source_tree_receipt: Mapping[str, object] | None = None,
    hippo_source_file_count: int | None = None,
    hippo_source_size_bytes: int | None = None,
    hippo_source_tree_sha256: str | None = None,
) -> None:
    try:
        from . import dependency_closure as closure

        closure.verify_regular_file_receipt(
            runtime.resolved_target, runtime.resolved_target_receipt
        )
        closure.verify_regular_file_receipt(
            runtime.pyvenv_cfg, runtime.pyvenv_cfg_receipt
        )
        closure.verify_tree_receipt(
            runtime.stdlib_root, runtime.stdlib_tree_receipt
        )
        if (
            runtime.python_zip_path.exists()
            or runtime.python_zip_path.is_symlink()
        ):
            raise HitabP1ProductionRuntimeError(
                f"{runtime.role} automatic Python zip appeared"
            )
        for root, receipt in zip(
            runtime.ordered_roots, runtime.tree_receipts, strict=True
        ):
            closure.verify_tree_receipt(root, receipt)
        for row in runtime.import_probe.values():
            closure.verify_regular_file_receipt(
                Path(str(row["origin_path"])),
                row["origin_receipt"],
            )
        source_bindings = (
            hippo_source_root,
            hippo_source_tree_receipt,
            hippo_source_file_count,
            hippo_source_size_bytes,
            hippo_source_tree_sha256,
        )
        if any(value is None for value in source_bindings) and not all(
            value is None for value in source_bindings
        ):
            raise HitabP1ProductionRuntimeError(
                "HippoRAG full source recheck binding is incomplete"
            )
        if (
            hippo_source_root is not None
            and hippo_source_tree_receipt is not None
            and hippo_source_file_count is not None
            and hippo_source_size_bytes is not None
            and hippo_source_tree_sha256 is not None
        ):
            closure.verify_tree_receipt(
                hippo_source_root, hippo_source_tree_receipt
            )
            _validate_clean_hipporag_source_tree(
                hippo_source_root,
                expected_file_count=hippo_source_file_count,
                expected_size_bytes=hippo_source_size_bytes,
                expected_tree_sha256=hippo_source_tree_sha256,
            )
    except Exception as exc:
        raise HitabP1ProductionRuntimeError(
            f"{runtime.role} runtime filesystem drifted after claim"
        ) from exc


def _validate_live_probe(
    runtime: FrozenPythonRuntime,
    value: object,
    *,
    project_root: Path,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "dont_write_bytecode",
        "invalid_cached",
        "no_site",
        "pycache_prefix",
        "python_version",
        "resolved_executable",
        "rows",
        "stdlib_root",
        "sys_path",
    }:
        raise HitabP1ProductionRuntimeError(
            f"{runtime.role} live import probe is malformed"
        )
    if (
        value["dont_write_bytecode"] is not True
        or value["invalid_cached"] != []
        or value["no_site"] != 1
        or value["pycache_prefix"] != "/dev/null"
        or value["python_version"] != runtime.python_version
        or value["resolved_executable"] != str(runtime.resolved_target)
        or value["stdlib_root"] != str(runtime.stdlib_root)
        or not isinstance(value["rows"], dict)
        or set(value["rows"]) != set(runtime.import_probe)
        or not isinstance(value["sys_path"], list)
        or not all(isinstance(row, str) for row in value["sys_path"])
    ):
        raise HitabP1ProductionRuntimeError(
            f"{runtime.role} live interpreter identity drifted"
        )
    selected_paths = [str(project_root), *map(str, runtime.ordered_roots)]
    positions: list[int] = []
    for selected in selected_paths:
        try:
            positions.append(value["sys_path"].index(selected))
        except ValueError as exc:
            raise HitabP1ProductionRuntimeError(
                f"{runtime.role} live path closure is incomplete"
            ) from exc
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise HitabP1ProductionRuntimeError(
            f"{runtime.role} live dependency root order drifted"
        )
    allowed_package_roots = set(map(str, runtime.ordered_roots))
    for raw_path in value["sys_path"]:
        path = Path(raw_path)
        if not (
            raw_path == str(project_root)
            or raw_path in allowed_package_roots
            or _path_is_within(path, runtime.stdlib_root)
        ):
            raise HitabP1ProductionRuntimeError(
                f"{runtime.role} imported an unfrozen Python path"
            )
    for module_name, expected in runtime.import_probe.items():
        actual = value["rows"].get(module_name)
        if (
            not isinstance(actual, dict)
            or set(actual)
            != {"content_sha256", "origin_path", "version"}
            or actual["origin_path"] != expected["origin_path"]
            or actual["content_sha256"]
            != expected["origin_receipt"]["content_sha256"]
            or actual["version"] != expected["version"]
        ):
            raise HitabP1ProductionRuntimeError(
                f"{runtime.role} live import {module_name} drifted"
            )
    return value


def _probe_child_runtime(
    implementation: FrozenImplementation,
) -> dict[str, object]:
    runtime = implementation.hippo_runtime
    request = {
        module: row["distribution"]
        for module, row in runtime.import_probe.items()
    }
    environment = dict(_COMMON_OFFLINE_ENVIRONMENT)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(implementation.project_root), *map(str, runtime.ordered_roots)]
    )
    command = [
        "/usr/bin/env",
        "-i",
        *(f"{key}={value}" for key, value in sorted(environment.items())),
        str(runtime.executable),
        "-S",
        "-B",
        "-c",
        _IMPORT_PROBE_SCRIPT,
        canonical_bytes(request, newline=False).decode("ascii"),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path("/"),
            env={},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise HitabP1ProductionRuntimeError(
            "HippoRAG child runtime probe could not start"
        ) from exc
    if (
        completed.returncode != 0
        or len(completed.stdout) > 4 * 1024 * 1024
        or len(completed.stderr) > 4 * 1024 * 1024
    ):
        raise HitabP1ProductionRuntimeError(
            "HippoRAG child runtime probe failed"
        )
    try:
        value = json.loads(completed.stdout.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HitabP1ProductionRuntimeError(
            "HippoRAG child runtime probe output is invalid"
        ) from exc
    return _validate_live_probe(
        runtime, value, project_root=implementation.project_root
    )


def _invalid_module_cache_paths(
    modules: Sequence[object],
) -> list[str]:
    """Inspect module dictionaries without invoking lazy ``__getattr__``."""

    return sorted(
        {
            cached
            for module in modules
            for cached in [vars(module).get("__cached__")]
            if isinstance(cached, str)
            and (
                not cached.startswith("/dev/null/")
                or os.path.exists(cached)
            )
        }
    )


def _temporary_module_paths(
    modules: Sequence[object],
) -> list[str]:
    """Return filesystem-bearing module paths rooted in shared temp space."""

    values: set[str] = set()

    def record(raw: object) -> None:
        if not isinstance(raw, str) or not raw.startswith("/"):
            return
        path = Path(raw)
        if _path_is_within(path, Path("/tmp")) or _path_is_within(
            path, Path("/var/tmp")
        ):
            values.add(raw)

    for module in modules:
        try:
            namespace = vars(module)
        except TypeError:
            continue
        record(namespace.get("__file__"))
        spec = namespace.get("__spec__")
        if spec is not None:
            record(getattr(spec, "origin", None))
            locations = getattr(
                spec, "submodule_search_locations", None
            )
            if locations is not None:
                for value in locations:
                    record(value)
        package_paths = namespace.get("__path__")
        if package_paths is not None:
            for value in package_paths:
                record(value)
    return sorted(values)


def _reject_outer_dynamic_python() -> None:
    temporary_paths = set(
        _temporary_module_paths(tuple(sys.modules.values()))
    )
    for raw in sys.path:
        if not isinstance(raw, str) or not raw.startswith("/"):
            continue
        path = Path(raw)
        if _path_is_within(path, Path("/tmp")) or _path_is_within(
            path, Path("/var/tmp")
        ):
            temporary_paths.add(raw)
    if temporary_paths:
        raise HitabP1ProductionRuntimeError(
            "outer runtime executed Python from shared temporary space"
        )
    if any(
        name == "sentence_transformers"
        or name.startswith("sentence_transformers.")
        for name in sys.modules
    ):
        raise HitabP1ProductionRuntimeError(
            "retired SentenceTransformer backend entered outer runtime"
        )


def prepare_implementation_runtime(
    implementation: FrozenImplementation,
    *,
    verify_hippo_child: bool,
) -> Mapping[str, object]:
    """Activate a verified typed closure without executing any ``.pth`` file."""

    runtime = implementation.outer_runtime
    if (
        sys.flags.no_site != 1
        or sys.dont_write_bytecode is not True
        or sys.pycache_prefix != "/dev/null"
        or Path(sys.executable).resolve() != runtime.resolved_target
        or os.environ.get("PYTHONPATH") != str(implementation.project_root)
    ):
        raise HitabP1ProductionRuntimeError(
            "outer runtime was not launched through the frozen -S -B unit"
        )
    for key in os.environ:
        folded = key.casefold()
        if (
            "proxy" in folded
            or "api_key" in folded
            or "assumption_v2_api" in folded
            or "ruoli" in folded
        ):
            raise HitabP1ProductionRuntimeError(
                "outer runtime inherited a forbidden credential channel"
            )
    _reject_unbound_project_bytecode(implementation.project_root)
    _verify_runtime_filesystem_again(runtime)
    sys.path[:] = [
        path for path in sys.path if path and Path(path).exists()
    ]
    import sysconfig

    if (
        Path(sysconfig.get_path("stdlib")).resolve()
        != runtime.stdlib_root
    ):
        raise HitabP1ProductionRuntimeError(
            "outer live stdlib root drifted"
        )
    unexpected = [
        path
        for path in sys.path
        if path != str(implementation.project_root)
        and not _path_is_within(Path(path), runtime.stdlib_root)
    ]
    if unexpected or any(
        str(root) in sys.path for root in runtime.ordered_roots
    ):
        raise HitabP1ProductionRuntimeError(
            "outer runtime dependency roots were active before verification"
        )
    sys.path.extend(map(str, runtime.ordered_roots))
    from importlib import metadata

    rows: dict[str, object] = {}
    for module_name, expected in sorted(runtime.import_probe.items()):
        try:
            module = importlib.import_module(module_name)
            raw_origin = getattr(module, "__file__", None)
            if not isinstance(raw_origin, str):
                raise TypeError("module origin absent")
            origin = Path(raw_origin).resolve(strict=True)
            distribution = expected["distribution"]
            version = (
                metadata.version(str(distribution))
                if distribution is not None
                else None
            )
        except Exception as exc:
            raise HitabP1ProductionRuntimeError(
                f"outer live import {module_name} failed"
            ) from exc
        actual = {
            "content_sha256": file_sha256(origin),
            "origin_path": str(origin),
            "version": version,
        }
        rows[module_name] = actual
        _reject_outer_dynamic_python()
    outer_probe = {
        "dont_write_bytecode": sys.dont_write_bytecode,
        "invalid_cached": _invalid_module_cache_paths(
            tuple(sys.modules.values())
        ),
        "no_site": sys.flags.no_site,
        "pycache_prefix": sys.pycache_prefix,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "resolved_executable": str(Path(sys.executable).resolve()),
        "rows": rows,
        "stdlib_root": str(
            Path(sysconfig.get_path("stdlib")).resolve()
        ),
        "sys_path": list(sys.path),
    }
    _validate_live_probe(
        runtime, outer_probe, project_root=implementation.project_root
    )
    child_probe = None
    if verify_hippo_child:
        _verify_runtime_filesystem_again(
            implementation.hippo_runtime,
            hippo_source_root=implementation.hippo_source_root,
            hippo_source_tree_receipt=(
                implementation.hippo_source_tree_receipt
            ),
            hippo_source_file_count=(
                implementation.hippo_source_file_count
            ),
            hippo_source_size_bytes=(
                implementation.hippo_source_size_bytes
            ),
            hippo_source_tree_sha256=(
                implementation.hippo_source_tree_sha256
            ),
        )
        child_probe = _probe_child_runtime(implementation)
    return _self_hashed(
        {
            "dependency_closure_scope": implementation.runtime_policy[
                "dependency_closure_scope"
            ],
            "hippo_child_verified": verify_hippo_child,
            "hippo_child_probe_sha256": (
                stable_hash(child_probe) if child_probe is not None else None
            ),
            "outer_probe_sha256": stable_hash(outer_probe),
            "pth_execution_count": 0,
            "schema": f"{VERSION}_dual_runtime_live_binding_v1",
            "study_id": STUDY_ID,
        }
    )


def load_acquisition_freeze(path: Path) -> FrozenAcquisition:
    """Verify the committed, source-body-free authorization for four GETs."""

    path = Path(path)
    value = _read_canonical_json(path, label="source acquisition freeze")
    self_sha256 = _verify_self(
        value, field="source acquisition freeze self hash"
    )
    if (
        value.get("schema") != ACQUISITION_FREEZE_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("network_attempt_count") != 4
        or value.get("parallel_transport_count") != 4
        or value.get("json_decode_count") != 0
        or value.get(
            "retry_resume_range_mirror_or_provider_switch_count"
        )
        != 0
    ):
        raise HitabP1ProductionRuntimeError(
            "source acquisition freeze policy drifted"
        )
    raw_impl = value.get("implementation_freeze")
    if not isinstance(raw_impl, dict) or set(raw_impl) != {
        "file_sha256",
        "path",
        "self_sha256",
    }:
        raise HitabP1ProductionRuntimeError(
            "source acquisition implementation binding is malformed"
        )
    if not isinstance(raw_impl["path"], str):
        raise HitabP1ProductionRuntimeError(
            "source acquisition implementation path is invalid"
        )
    implementation_path = Path(raw_impl["path"])
    if file_sha256(implementation_path) != _hex64(
        raw_impl["file_sha256"], field="source acquisition implementation file"
    ):
        raise HitabP1ProductionRuntimeError(
            "source acquisition implementation file drifted"
        )
    implementation = load_implementation_freeze(implementation_path)
    if implementation.self_sha256 != _hex64(
        raw_impl["self_sha256"],
        field="source acquisition implementation self hash",
    ):
        raise HitabP1ProductionRuntimeError(
            "source acquisition implementation self binding drifted"
        )
    raw_canary = value.get("canary_receipt")
    if not isinstance(raw_canary, dict) or set(raw_canary) != {
        "file_sha256",
        "path",
        "self_sha256",
    }:
        raise HitabP1ProductionRuntimeError(
            "source acquisition canary binding is malformed"
        )
    if not isinstance(raw_canary["path"], str):
        raise HitabP1ProductionRuntimeError(
            "source acquisition canary path is invalid"
        )
    canary_path = Path(raw_canary["path"])
    _canary, canary_self = _validate_canary_receipt(
        canary_path,
        implementation=implementation,
        expected_file_sha256=str(raw_canary["file_sha256"]),
        expected_self_sha256=str(raw_canary["self_sha256"]),
    )
    raw_source = value.get("source_root")
    raw_control = value.get("control_root")
    if not isinstance(raw_source, str) or not isinstance(raw_control, str):
        raise HitabP1ProductionRuntimeError(
            "source acquisition roots are invalid"
        )
    source_root = Path(raw_source)
    control_root = Path(raw_control)
    if (
        not source_root.is_absolute()
        or not control_root.is_absolute()
        or source_root == control_root
        or source_root.is_symlink()
        or control_root.is_symlink()
    ):
        raise HitabP1ProductionRuntimeError(
            "source acquisition roots are unsafe"
        )
    for root in (source_root, control_root):
        if root.exists() and not root.is_dir():
            raise HitabP1ProductionRuntimeError(
                "source acquisition root is not a directory"
            )
    return FrozenAcquisition(
        path=path,
        self_sha256=self_sha256,
        implementation=implementation,
        canary_receipt_path=canary_path,
        canary_receipt_self_sha256=canary_self,
        source_root=source_root,
        control_root=control_root,
    )


def _git_blob_sha1(path: Path, *, size_bytes: int) -> str:
    digest = hashlib.sha1()  # nosec B324: immutable Git object identity
    digest.update(f"blob {size_bytes}\0".encode("ascii"))
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise HitabP1ProductionRuntimeError(
            "source file is unavailable"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != size_bytes
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise HitabP1ProductionRuntimeError(
                "source file metadata drifted"
            )
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _validate_source_download_receipt(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
    expected_self_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    if expected_file_sha256 is not None and file_sha256(path) != _hex64(
        expected_file_sha256, field="source receipt file"
    ):
        raise HitabP1ProductionRuntimeError("source receipt file drifted")
    value = _read_canonical_json(path, label="source download receipt")
    self_sha256 = _verify_self(value, field="source receipt self hash")
    if expected_self_sha256 is not None and self_sha256 != _hex64(
        expected_self_sha256, field="source receipt expected self hash"
    ):
        raise HitabP1ProductionRuntimeError(
            "source receipt self binding drifted"
        )
    files = value.get("files")
    if (
        value.get("schema") != "hitab_p1_source_download_receipt_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("status") != "four_exact_sources_acquired_once"
        or value.get("file_count") != 4
        or value.get("network_attempt_count") != 4
        or value.get("parallel_transport_count") != 4
        or value.get("json_decode_count") != 0
        or value.get("test_json_decode_count") != 0
        or value.get(
            "retry_resume_range_mirror_or_provider_switch_count"
        )
        != 0
        or not isinstance(files, dict)
        or set(files) != {"TRAIN", "DEV", "TEST", "TABLES"}
        or _HEX64.fullmatch(
            str(value.get("source_identity_commitment"))
        )
        is None
    ):
        raise HitabP1ProductionRuntimeError(
            "source download receipt policy drifted"
        )
    return value, self_sha256


def _validate_canary_receipt(
    path: Path,
    *,
    implementation: FrozenImplementation,
    expected_file_sha256: str | None = None,
    expected_self_sha256: str | None = None,
    validate_inner: bool = False,
) -> tuple[dict[str, object], str]:
    if expected_file_sha256 is not None and file_sha256(path) != _hex64(
        expected_file_sha256, field="canary receipt file"
    ):
        raise HitabP1ProductionRuntimeError("canary receipt file drifted")
    value = _read_canonical_json(path, label="source-free canary receipt")
    self_sha256 = _verify_self(value, field="canary receipt self hash")
    if expected_self_sha256 is not None and self_sha256 != _hex64(
        expected_self_sha256, field="canary receipt expected self hash"
    ):
        raise HitabP1ProductionRuntimeError("canary receipt binding drifted")
    if (
        value.get("schema") != CANARY_RECEIPT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("implementation_freeze_self_sha256")
        != implementation.self_sha256
        or value.get("qualified") is not True
        or value.get("source_or_HiTab_rows_accessed") is not False
        or value.get("online_or_API_call_count") != 0
        or _HEX64.fullmatch(
            str(value.get("canary_attempt_self_sha256"))
        )
        is None
    ):
        raise HitabP1ProductionRuntimeError(
            "canary receipt identity drifted"
        )
    attempt_path = (
        path.parent / "source_free_canary.attempt.private.json"
    )
    attempt = _read_canonical_json(
        attempt_path, label="source-free canary attempt"
    )
    attempt_self = _verify_self(
        attempt, field="source-free canary attempt self hash"
    )
    if (
        attempt.get("schema") != CANARY_ATTEMPT_SCHEMA
        or attempt.get("study_id") != STUDY_ID
        or attempt.get(
            "retry_replay_resample_provider_model_candidate_or_gate_change_count"
        )
        != 0
        or attempt.get("implementation_freeze_file_sha256")
        != file_sha256(implementation.path)
        or attempt.get("implementation_freeze_path_sha256")
        != hashlib.sha256(
            os.fsencode(os.fspath(implementation.path))
        ).hexdigest()
        or attempt.get("output_path_sha256")
        != hashlib.sha256(os.fsencode(os.fspath(path))).hexdigest()
        or value.get("canary_attempt_self_sha256") != attempt_self
    ):
        raise HitabP1ProductionRuntimeError(
            "source-free canary attempt binding drifted"
        )
    inner = value.get("canary")
    if not isinstance(inner, dict):
        raise HitabP1ProductionRuntimeError(
            "canary receipt payload is absent"
        )
    if validate_inner:
        from assumption_agent.benchmarks import (
            hitab_p1_public_canary_v1 as canary,
        )

        try:
            canary.validate_receipt(inner)
        except Exception as exc:
            raise HitabP1ProductionRuntimeError(
                "inner public canary receipt drifted"
            ) from exc
    return value, self_sha256


def load_execution_freeze(
    path: Path,
    *,
    runtime_preparer: RuntimePreparer = prepare_implementation_runtime,
) -> FrozenExecution:
    path = Path(path)
    value = _read_canonical_json(path, label="execution freeze")
    self_sha256 = _verify_self(value, field="execution freeze self hash")
    if (
        value.get("schema") != EXECUTION_FREEZE_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("runtime_policy") != RUNTIME_POLICY
        or value.get(
            "retry_replay_resample_provider_model_candidate_or_gate_change_count"
        )
        != 0
    ):
        raise HitabP1ProductionRuntimeError(
            "execution freeze identity or policy drifted"
        )

    raw_impl = value.get("implementation_freeze")
    if not isinstance(raw_impl, dict) or set(raw_impl) != {
        "file_sha256",
        "path",
        "self_sha256",
    }:
        raise HitabP1ProductionRuntimeError(
            "execution implementation binding is malformed"
        )
    if not isinstance(raw_impl["path"], str):
        raise HitabP1ProductionRuntimeError(
            "implementation freeze path is invalid"
        )
    implementation_path = Path(raw_impl["path"])
    if file_sha256(implementation_path) != _hex64(
        raw_impl["file_sha256"], field="implementation freeze file"
    ):
        raise HitabP1ProductionRuntimeError(
            "implementation freeze file drifted"
        )
    implementation = load_implementation_freeze(implementation_path)
    if implementation.self_sha256 != _hex64(
        raw_impl["self_sha256"], field="implementation freeze expected self"
    ):
        raise HitabP1ProductionRuntimeError(
            "implementation freeze self binding drifted"
        )
    runtime_preparer(implementation, verify_hippo_child=True)

    raw_canary = value.get("canary_receipt")
    if not isinstance(raw_canary, dict) or set(raw_canary) != {
        "file_sha256",
        "path",
        "self_sha256",
    }:
        raise HitabP1ProductionRuntimeError(
            "execution canary binding is malformed"
        )
    if not isinstance(raw_canary["path"], str):
        raise HitabP1ProductionRuntimeError(
            "canary receipt path is invalid"
        )
    canary_path = Path(raw_canary["path"])
    _canary, canary_self = _validate_canary_receipt(
        canary_path,
        implementation=implementation,
        expected_file_sha256=str(raw_canary["file_sha256"]),
        expected_self_sha256=str(raw_canary["self_sha256"]),
    )

    raw_source_receipt = value.get("source_receipt")
    if not isinstance(raw_source_receipt, dict) or set(raw_source_receipt) != {
        "file_sha256",
        "path",
        "self_sha256",
        "source_identity_commitment",
    }:
        raise HitabP1ProductionRuntimeError(
            "execution source receipt binding is malformed"
        )
    if not isinstance(raw_source_receipt["path"], str):
        raise HitabP1ProductionRuntimeError(
            "source receipt path is invalid"
        )
    source_receipt_path = Path(raw_source_receipt["path"])
    source_receipt, source_receipt_self = (
        _validate_source_download_receipt(
            source_receipt_path,
            expected_file_sha256=str(raw_source_receipt["file_sha256"]),
            expected_self_sha256=str(raw_source_receipt["self_sha256"]),
        )
    )
    source_identity_commitment = _hex64(
        raw_source_receipt["source_identity_commitment"],
        field="source identity commitment",
    )
    if (
        source_receipt.get("source_identity_commitment")
        != source_identity_commitment
    ):
        raise HitabP1ProductionRuntimeError(
            "source identity commitment drifted"
        )
    receipt_files = source_receipt["files"]
    if not isinstance(receipt_files, dict):
        raise HitabP1ProductionRuntimeError(
            "source receipt files are invalid"
        )
    raw_source_files = value.get("source_files")
    if not isinstance(raw_source_files, dict) or set(raw_source_files) != {
        "TRAIN",
        "DEV",
        "TEST",
        "TABLES",
    }:
        raise HitabP1ProductionRuntimeError(
            "execution source file closure is not exact"
        )
    source_paths: dict[str, Path] = {}
    source_sha256s: dict[str, str] = {}
    for key in ("TRAIN", "DEV", "TEST", "TABLES"):
        row = raw_source_files[key]
        receipt_row = receipt_files[key]
        if (
            not isinstance(row, dict)
            or set(row)
            != {"git_blob_sha1", "path", "sha256", "size_bytes"}
            or not isinstance(receipt_row, dict)
            or not isinstance(row["path"], str)
            or row["sha256"] != receipt_row.get("sha256")
            or row["git_blob_sha1"] != receipt_row.get("git_blob_sha1")
            or row["size_bytes"] != receipt_row.get("size_bytes")
        ):
            raise HitabP1ProductionRuntimeError(
                f"source file {key} binding is malformed"
            )
        source_path = Path(row["path"])
        expected_sha = _hex64(
            row["sha256"], field=f"source file {key}"
        )
        size = row["size_bytes"]
        blob = row["git_blob_sha1"]
        if (
            not source_path.is_absolute()
            or source_path.is_symlink()
            or type(size) is not int
            or size < 1
            or not isinstance(blob, str)
            or re.fullmatch(r"[0-9a-f]{40}", blob) is None
            or source_path.stat().st_size != size
            or file_sha256(source_path) != expected_sha
            or _git_blob_sha1(source_path, size_bytes=size) != blob
        ):
            raise HitabP1ProductionRuntimeError(
                f"source file {key} identity drifted"
            )
        source_paths[key] = source_path
        source_sha256s[key] = expected_sha

    raw_root = value.get("formal_work_root")
    if not isinstance(raw_root, str):
        raise HitabP1ProductionRuntimeError("formal work root is invalid")
    formal_root = Path(raw_root)
    if not formal_root.is_absolute() or formal_root.is_symlink():
        raise HitabP1ProductionRuntimeError("formal work root is unsafe")
    if formal_root.exists() and not formal_root.is_dir():
        raise HitabP1ProductionRuntimeError(
            "formal work root is not a directory"
        )

    raw_factory = value.get("acquisition_factory")
    if not isinstance(raw_factory, dict) or set(raw_factory) != {
        "attribute",
        "file_label",
        "module",
    }:
        raise HitabP1ProductionRuntimeError(
            "acquisition factory binding is malformed"
        )
    module = raw_factory["module"]
    attribute = raw_factory["attribute"]
    label = raw_factory["file_label"]
    if (
        module != ACQUISITION_FACTORY_MODULE
        or attribute != ACQUISITION_FACTORY_ATTRIBUTE
        or label != ACQUISITION_FACTORY_FILE_LABEL
    ):
        raise HitabP1ProductionRuntimeError(
            "acquisition factory reference is not the frozen exact factory"
        )
    return FrozenExecution(
        path=path,
        self_sha256=self_sha256,
        implementation=implementation,
        canary_receipt_path=canary_path,
        canary_receipt_self_sha256=canary_self,
        source_receipt_path=source_receipt_path,
        source_receipt_self_sha256=source_receipt_self,
        source_identity_commitment=source_identity_commitment,
        source_paths=source_paths,
        source_sha256s=source_sha256s,
        formal_work_root=formal_root,
        acquisition_factory_module=module,
        acquisition_factory_attribute=attribute,
        acquisition_factory_file_label=label,
    )


_HIPPO_CHILD_BOOTSTRAP_SCRIPT = (
    "import os,runpy,sys,sysconfig\n"
    "project=os.path.realpath(sys.argv[1]);"
    "stdlib=os.path.realpath(sys.argv[2]);module=sys.argv[3]\n"
    "if (sys.flags.no_site!=1 or not sys.dont_write_bytecode or "
    "sys.pycache_prefix!='/dev/null' or "
    "os.path.realpath(sysconfig.get_path('stdlib'))!=stdlib):"
    " raise RuntimeError('frozen child interpreter drifted')\n"
    "for current,dirs,files in os.walk(project,followlinks=False):\n"
    " if '__pycache__' in dirs or any("
    "name.endswith(('.pyc','.pyo')) for name in dirs+files):"
    "  raise RuntimeError('unbound project bytecode')\n"
    + _SEAL_CHILD_SYS_PATH_SCRIPT
    + "sys.argv=[module,*sys.argv[4:]]\n"
    "if (sys.path is not _sealed_child_sys_path or "
    "tuple(sys.path)!=_frozen_child_sys_path):"
    " raise RuntimeError('child import path seal drifted before module')\n"
    "try:\n"
    " runpy.run_module(module,run_name='__main__')\n"
    "finally:\n"
    " if (sys.path is not _sealed_child_sys_path or "
    "tuple(sys.path)!=_frozen_child_sys_path):"
    "  raise RuntimeError('child import path seal drifted after module')\n"
)


class HippoFreshProcessRunner:
    """One fresh, private, item-local official process per byte request."""

    def __init__(
        self,
        *,
        project_root: Path,
        runtime_root: Path,
        python_executable: Path,
        dependency_roots: Sequence[Path],
        stdlib_root: Path,
        hippo_source_root: Path,
        hippo_source_tree_receipt: Mapping[str, object],
        hippo_source_file_count: int,
        hippo_source_size_bytes: int,
        hippo_source_tree_sha256: str,
        worker_module: str,
        worker_file: Path,
        worker_file_sha256: str,
        llm_model_root: Path,
        llm_model_tree_sha256: str,
        embedding_model_root: Path,
        embedding_model_tree_sha256: str,
        subprocess_runner: Callable[..., object] = subprocess.run,
    ) -> None:
        if (
            worker_module != HIPPORAG_WORKER_MODULE
            or file_sha256(worker_file)
            != _hex64(worker_file_sha256, field="HippoRAG worker")
            or model_tree_sha256(llm_model_root)
            != _hex64(llm_model_tree_sha256, field="HippoRAG LLM tree")
            or model_tree_sha256(embedding_model_root)
            != _hex64(
                embedding_model_tree_sha256,
                field="HippoRAG embedding tree",
            )
        ):
            raise HitabP1ProductionRuntimeError(
                "HippoRAG process closure drifted"
            )
        self.project_root = project_root
        self.runtime_root = runtime_root
        self.python_executable = python_executable
        self.dependency_roots = tuple(map(Path, dependency_roots))
        self.stdlib_root = Path(stdlib_root)
        self.hippo_source_root = Path(hippo_source_root)
        self.hippo_source_tree_receipt = dict(
            hippo_source_tree_receipt
        )
        self.hippo_source_file_count = hippo_source_file_count
        self.hippo_source_size_bytes = hippo_source_size_bytes
        self.hippo_source_tree_sha256 = hippo_source_tree_sha256
        if (
            not self.dependency_roots
            or len(set(self.dependency_roots))
            != len(self.dependency_roots)
            or any(not root.is_dir() for root in self.dependency_roots)
            or not self.stdlib_root.is_dir()
        ):
            raise HitabP1ProductionRuntimeError(
                "HippoRAG child dependency roots are invalid"
            )
        self.worker_module = worker_module
        self.llm_model_root = llm_model_root
        self.embedding_model_root = embedding_model_root
        self.subprocess_runner = subprocess_runner
        self._gpu_locks = {gpu: threading.Lock() for gpu in PHYSICAL_GPUS}
        self._counter_lock = threading.Lock()
        self._counter = 0
        runtime_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if runtime_root.is_symlink() or not runtime_root.is_dir():
            raise HitabP1ProductionRuntimeError(
                "HippoRAG runtime root is unsafe"
            )
        os.chmod(runtime_root, 0o700)
        self._verify_source_projection()

    def _verify_source_projection(self) -> None:
        try:
            from . import dependency_closure as closure

            closure.verify_tree_receipt(
                self.hippo_source_root,
                self.hippo_source_tree_receipt,
            )
            _validate_clean_hipporag_source_tree(
                self.hippo_source_root,
                expected_file_count=self.hippo_source_file_count,
                expected_size_bytes=self.hippo_source_size_bytes,
                expected_tree_sha256=self.hippo_source_tree_sha256,
            )
        except Exception as exc:
            raise HitabP1ProductionRuntimeError(
                "HippoRAG clean source drifted before child launch"
            ) from exc

    def _next_ordinal(self) -> int:
        with self._counter_lock:
            value = self._counter
            self._counter += 1
            return value

    def __call__(
        self,
        canonical_input: bytes,
        *,
        physical_gpu: int,
        cpu_thread_limit: int,
        launch_ack: Callable[[], None],
    ) -> bytes:
        if (
            physical_gpu not in PHYSICAL_GPUS
            or cpu_thread_limit != CPU_THREADS_PER_GPU_LANE
            or not isinstance(canonical_input, bytes)
            or not callable(launch_ack)
        ):
            raise HitabP1ProductionRuntimeError(
                "HippoRAG lane contract drifted"
            )
        from replication_runtime.birco_official_hipporag_v1 import (
            contract as hippo_contract,
        )

        try:
            payload = json.loads(canonical_input.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HitabP1ProductionRuntimeError(
                "HippoRAG input is invalid JSON"
            ) from exc
        if (
            not isinstance(payload, dict)
            or canonical_input != hippo_contract.canonical_json_bytes(payload)
        ):
            raise HitabP1ProductionRuntimeError(
                "HippoRAG input is not canonical"
            )
        try:
            hippo_contract.validate_input(
                payload.get("work_id"),
                payload.get("objective"),
                payload.get("query"),
                payload.get("documents"),
                payload.get("common_projection_sha256"),
            )
        except Exception as exc:
            raise HitabP1ProductionRuntimeError(
                "HippoRAG input contract drifted"
            ) from exc

        lock = self._gpu_locks[physical_gpu]
        if not lock.acquire(blocking=False):
            raise HitabP1ProductionRuntimeError(
                "more than one HippoRAG process targeted one physical GPU"
            )
        try:
            ordinal = self._next_ordinal()
            digest = hashlib.sha256(canonical_input).hexdigest()
            attempt = (
                self.runtime_root
                / f"attempt-{ordinal:06d}-gpu{physical_gpu}-{digest[:16]}"
            )
            try:
                attempt.mkdir(mode=0o700)
            except OSError as exc:
                raise HitabP1ProductionRuntimeError(
                    "HippoRAG attempt is not exclusive"
                ) from exc
            model_cwd = attempt / "models"
            model_cwd.mkdir(mode=0o700)
            (model_cwd / "smollm2").symlink_to(
                self.llm_model_root, target_is_directory=True
            )
            (model_cwd / "minilm").symlink_to(
                self.embedding_model_root, target_is_directory=True
            )
            home = attempt / "home"
            home.mkdir(mode=0o700)
            cache = attempt / "cache"
            cache.mkdir(mode=0o700)
            input_path = attempt / "input.private.json"
            output_path = attempt / "output.private.json"
            index_root = attempt / "index.private"
            stdout_path = attempt / "stdout.private.log"
            stderr_path = attempt / "stderr.private.log"
            terminal_path = attempt / "attempt.terminal.private.json"
            _write_exclusive(input_path, canonical_input)
            _reject_unbound_project_bytecode(self.project_root)

            environment = {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_VISIBLE_DEVICES": str(physical_gpu),
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "HOME": str(home),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "MKL_NUM_THREADS": str(CPU_THREADS_PER_GPU_LANE),
                "NUMEXPR_NUM_THREADS": str(CPU_THREADS_PER_GPU_LANE),
                "OMP_NUM_THREADS": str(CPU_THREADS_PER_GPU_LANE),
                "OPENBLAS_NUM_THREADS": str(CPU_THREADS_PER_GPU_LANE),
                "PATH": (
                    f"{self.python_executable.parent}:/usr/bin:/bin"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(self.project_root),
                        *map(str, self.dependency_roots),
                    ]
                ),
                "PYTHONPYCACHEPREFIX": "/dev/null",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "VECLIB_MAXIMUM_THREADS": str(CPU_THREADS_PER_GPU_LANE),
                "XDG_CACHE_HOME": str(cache),
            }
            command = [
                "/usr/bin/env",
                "-i",
                *(f"{key}={value}" for key, value in sorted(environment.items())),
                str(self.python_executable),
                "-S",
                "-B",
                "-c",
                _HIPPO_CHILD_BOOTSTRAP_SCRIPT,
                str(self.project_root),
                str(self.stdlib_root),
                self.worker_module,
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
            ]
            stdout_capture = _BoundedDiagnosticCapture(stdout_path)
            try:
                stderr_capture = _BoundedDiagnosticCapture(stderr_path)
            except Exception:
                stdout_capture.finish()
                raise
            returncode: int | None = None
            stdout_result: _DiagnosticResult | None = None
            stderr_result: _DiagnosticResult | None = None
            stage = "launch_fresh_official_worker"
            try:
                try:
                    self._verify_source_projection()
                    launch_ack()
                    completed = self.subprocess_runner(
                        command,
                        cwd=model_cwd,
                        env={},
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_capture.write_descriptor,
                        stderr=stderr_capture.write_descriptor,
                        check=False,
                    )
                    raw_returncode = getattr(completed, "returncode", None)
                    if type(raw_returncode) is not int:
                        raise HitabP1ProductionRuntimeError(
                            "HippoRAG subprocess return code is unavailable"
                        )
                    returncode = raw_returncode
                finally:
                    stdout_result = stdout_capture.finish()
                    stderr_result = stderr_capture.finish()
                if returncode != 0:
                    raise HitabP1ProductionRuntimeError(
                        "HippoRAG subprocess exited unsuccessfully"
                    )
                stage = "validate_official_worker_output"
                raw = _read_bounded_regular(
                    output_path,
                    label="HippoRAG output",
                    maximum_bytes=MAXIMUM_WORKER_OUTPUT_BYTES,
                    expected_mode=0o600,
                )
                try:
                    parsed = hippo_contract.parse_output(raw)
                except Exception as exc:
                    raise HitabP1ProductionRuntimeError(
                        "HippoRAG output contract drifted"
                    ) from exc
                if raw != hippo_contract.canonical_json_bytes(parsed):
                    raise HitabP1ProductionRuntimeError(
                        "HippoRAG output is not canonical"
                    )
                stage = "remove_validated_success_ephemeral_index"
                _remove_ephemeral_index_tree(
                    index_root, attempt_root=attempt
                )
                terminal = _self_hashed(
                    {
                        "CPU_threads_per_lane": (
                            CPU_THREADS_PER_GPU_LANE
                        ),
                        "env_i": True,
                        "ephemeral_index_preserved": False,
                        "ephemeral_index_removed_after_validated_success": True,
                        "input_sha256": hashlib.sha256(
                            canonical_input
                        ).hexdigest(),
                        "network_or_API_call_count": 0,
                        "output_present": True,
                        "output_sha256": hashlib.sha256(raw).hexdigest(),
                        "physical_gpu": physical_gpu,
                        "returncode": returncode,
                        "schema": (
                            f"{VERSION}_official_hipporag_attempt_terminal_v1"
                        ),
                        "status": "validated_success_index_removed",
                        "stderr": stderr_result.payload(),
                        "stdout": stdout_result.payload(),
                        "subprocess_timeout_seconds": None,
                        "worker_module": self.worker_module,
                    }
                )
                _write_exclusive(
                    terminal_path, canonical_bytes(terminal)
                )
                return raw
            except Exception as exc:
                if not terminal_path.exists() and not terminal_path.is_symlink():
                    failure = _self_hashed(
                        {
                            "CPU_threads_per_lane": (
                                CPU_THREADS_PER_GPU_LANE
                            ),
                            "env_i": True,
                            "ephemeral_index_preserved": (
                                index_root.exists()
                                or index_root.is_symlink()
                            ),
                            "ephemeral_index_removed_after_validated_success": False,
                            "failure_exception_type_sha256": hashlib.sha256(
                                type(exc).__name__.encode(
                                    "ascii", errors="replace"
                                )
                            ).hexdigest(),
                            "failure_stage": stage,
                            "input_sha256": hashlib.sha256(
                                canonical_input
                            ).hexdigest(),
                            "network_or_API_call_count": 0,
                            "output_present": (
                                output_path.exists()
                                and not output_path.is_symlink()
                            ),
                            "output_sha256": (
                                file_sha256(output_path)
                                if output_path.exists()
                                and not output_path.is_symlink()
                                and output_path.stat().st_size
                                <= MAXIMUM_WORKER_OUTPUT_BYTES
                                else None
                            ),
                            "physical_gpu": physical_gpu,
                            "returncode": returncode,
                            "schema": (
                                f"{VERSION}_official_hipporag_attempt_terminal_v1"
                            ),
                            "status": "terminal_failure_no_retry",
                            "stderr": (
                                stderr_result.payload()
                                if stderr_result is not None
                                else None
                            ),
                            "stdout": (
                                stdout_result.payload()
                                if stdout_result is not None
                                else None
                            ),
                            "subprocess_timeout_seconds": None,
                            "worker_module": self.worker_module,
                        }
                    )
                    _write_exclusive(
                        terminal_path, canonical_bytes(failure)
                    )
                if isinstance(exc, HitabP1ProductionRuntimeError):
                    raise
                raise HitabP1ProductionRuntimeError(
                    "HippoRAG subprocess failed"
                ) from exc
        finally:
            lock.release()


def _read_bounded_regular(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    expected_mode: int | None = None,
) -> bytes:
    if path.is_symlink():
        raise HitabP1ProductionRuntimeError(f"{label} is a symlink")
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise HitabP1ProductionRuntimeError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 2
            or metadata.st_size > maximum_bytes
            or (
                expected_mode is not None
                and stat.S_IMODE(metadata.st_mode) != expected_mode
            )
        ):
            raise HitabP1ProductionRuntimeError(
                f"{label} is not a bounded regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _remove_ephemeral_index_tree(root: Path, *, attempt_root: Path) -> None:
    """Delete only a validated successful item's private ephemeral index."""

    if (
        root.parent != attempt_root
        or root.name != "index.private"
        or not root.exists()
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise HitabP1ProductionRuntimeError(
            "successful HippoRAG index root is unsafe or absent"
        )

    def remove(path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise HitabP1ProductionRuntimeError(
                "HippoRAG index cleanup encountered a missing entry"
            ) from exc
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
            metadata.st_mode
        ):
            try:
                children = tuple(
                    sorted(
                        (Path(entry.path) for entry in os.scandir(path)),
                        key=lambda value: value.name,
                    )
                )
            except OSError as exc:
                raise HitabP1ProductionRuntimeError(
                    "HippoRAG index cleanup could not scan a directory"
                ) from exc
            for child in children:
                remove(child)
            try:
                path.rmdir()
            except OSError as exc:
                raise HitabP1ProductionRuntimeError(
                    "HippoRAG index cleanup could not remove a directory"
                ) from exc
        elif stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            try:
                path.unlink()
            except OSError as exc:
                raise HitabP1ProductionRuntimeError(
                    "HippoRAG index cleanup could not remove an entry"
                ) from exc
        else:
            raise HitabP1ProductionRuntimeError(
                "HippoRAG index contains a special file"
            )

    remove(root)
    if root.exists() or root.is_symlink():
        raise HitabP1ProductionRuntimeError(
            "HippoRAG index cleanup was incomplete"
        )


def build_production_bindings(
    implementation: FrozenImplementation, runtime_root: Path
) -> ProductionBindings:
    """Load only frozen local assets; never read a benchmark source."""

    from assumption_agent.benchmarks import hitab_p1_runtime_v1 as runtime

    original = {
        key: os.environ.get(key)
        for key in (
            *runtime.PRODUCTION_OFFLINE_ENVIRONMENT,
            "CUDA_VISIBLE_DEVICES",
        )
    }
    try:
        for key, value in runtime.PRODUCTION_OFFLINE_ENVIRONMENT.items():
            os.environ[key] = value
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        planner = runtime.BrightPlannerProductionRunner(
            implementation.models["planner"], physical_gpu=0
        )
        scorer = runtime.BrightCrossEncoderProductionScorer(
            implementation.models["cross_encoder"], physical_gpu=0
        )
        minilm = runtime.bind_bright_minilm_production_encoder(
            asset_manifest=implementation.minilm_asset_manifest,
            model_root=implementation.models["minilm"],
            physical_gpu=0,
        )
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    hippo = HippoFreshProcessRunner(
        project_root=implementation.project_root,
        runtime_root=runtime_root / "official_hipporag",
        python_executable=implementation.hippo_runtime.executable,
        dependency_roots=implementation.hippo_runtime.ordered_roots,
        stdlib_root=implementation.hippo_runtime.stdlib_root,
        hippo_source_root=implementation.hippo_source_root,
        hippo_source_tree_receipt=(
            implementation.hippo_source_tree_receipt
        ),
        hippo_source_file_count=implementation.hippo_source_file_count,
        hippo_source_size_bytes=implementation.hippo_source_size_bytes,
        hippo_source_tree_sha256=implementation.hippo_source_tree_sha256,
        worker_module=implementation.hippo_worker_module,
        worker_file=implementation.files["hippo_worker"],
        worker_file_sha256=implementation.file_sha256s["hippo_worker"],
        llm_model_root=implementation.models["hippo_llm"],
        llm_model_tree_sha256=implementation.model_tree_sha256s["hippo_llm"],
        embedding_model_root=implementation.models["hippo_embedding"],
        embedding_model_tree_sha256=(
            implementation.model_tree_sha256s["hippo_embedding"]
        ),
    )
    cache_releaser = build_gpu0_cache_releaser()
    _reject_outer_dynamic_python()
    return ProductionBindings(
        planner_runner=planner,
        cross_encoder_scorer=scorer,
        minilm_encoder=minilm,
        hippo_runner=hippo,
        gpu0_cache_releaser=cache_releaser,
    )


def build_gpu0_cache_releaser() -> Callable[[], Mapping[str, object]]:
    """Return the frozen unused-cache release action required before GPU0 Hippo."""

    def release() -> Mapping[str, object]:
        try:
            import torch

            if not torch.cuda.is_available():
                raise HitabP1ProductionRuntimeError(
                    "GPU0 cache release CUDA capability is unavailable"
                )
            with torch.cuda.device(0):
                torch.cuda.empty_cache()
        except HitabP1ProductionRuntimeError:
            raise
        except Exception as exc:
            raise HitabP1ProductionRuntimeError(
                "GPU0 unused CUDA cache release failed"
            ) from exc
        return _self_hashed(
            {
                "model_offload_or_reload": False,
                "physical_gpu": 0,
                "schema": "hitab_p1_gpu0_unused_cuda_cache_release_v1",
                "study_id": STUDY_ID,
                "torch_cuda_empty_cache_called": True,
            }
        )

    return release


def _default_source_acquisition(
    acquisition: FrozenAcquisition,
) -> Mapping[str, object]:
    before = file_sha256(
        acquisition.implementation.files["hitab_source_acquisition"]
    )
    from assumption_agent.benchmarks import (
        hitab_p1_source_acquisition_v1 as source,
    )

    after = file_sha256(
        acquisition.implementation.files["hitab_source_acquisition"]
    )
    if (
        before != after
        or before
        != acquisition.implementation.file_sha256s[
            "hitab_source_acquisition"
        ]
    ):
        raise HitabP1ProductionRuntimeError(
            "source acquisition implementation drifted during import"
        )
    downloaded = source.download_source_set_once(
        source_root=acquisition.source_root,
        control_root=acquisition.control_root,
        custody_path=acquisition.implementation.files[
            "hitab_source_custody"
        ],
        design_path=acquisition.implementation.files["hitab_study_design"],
    )
    receipt = downloaded.verified_sources.safe_receipt
    if not isinstance(receipt, Mapping):
        raise HitabP1ProductionRuntimeError(
            "source acquisition returned no safe receipt"
        )
    return receipt


def run_source_acquisition_once(
    *,
    acquisition_freeze_path: Path,
    acquisition_runner: SourceAcquisitionRunner = _default_source_acquisition,
    runtime_preparer: RuntimePreparer = prepare_implementation_runtime,
) -> Mapping[str, object]:
    """Run four pinned GETs once, verify bytes, and perform zero JSON decodes."""

    acquisition = load_acquisition_freeze(Path(acquisition_freeze_path))
    runtime_preparer(
        acquisition.implementation, verify_hippo_child=False
    )
    receipt_path = (
        acquisition.control_root / "source_download.receipt.safe.json"
    )
    attempt_path = (
        acquisition.control_root / "source_download.attempt.private.json"
    )
    if (
        receipt_path.exists()
        or receipt_path.is_symlink()
        or attempt_path.exists()
        or attempt_path.is_symlink()
    ):
        raise HitabP1ProductionRuntimeError(
            "source acquisition attempt was already consumed"
        )
    try:
        returned = acquisition_runner(acquisition)
    except Exception as exc:
        raise HitabP1ProductionRuntimeError(
            "source acquisition terminated; retry is forbidden"
        ) from exc
    receipt, _self = _validate_source_download_receipt(receipt_path)
    if dict(returned) != receipt:
        raise HitabP1ProductionRuntimeError(
            "source acquisition receipt return drifted"
        )
    return receipt


def run_source_free_canary_once(
    *,
    implementation_freeze_path: Path,
    output_path: Path,
    binding_builder: ProductionBindingBuilder = build_production_bindings,
    runtime_preparer: RuntimePreparer = prepare_implementation_runtime,
) -> Mapping[str, object]:
    """Run the public synthetic full path twice and seal one exclusive receipt."""

    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise HitabP1ProductionRuntimeError(
            "source-free canary receipt already exists"
        )
    implementation_path = Path(implementation_freeze_path)
    attempt = _self_hashed(
        {
            "implementation_freeze_file_sha256": file_sha256(
                implementation_path
            ),
            "implementation_freeze_path_sha256": hashlib.sha256(
                os.fsencode(os.fspath(implementation_path))
            ).hexdigest(),
            "output_path_sha256": hashlib.sha256(
                os.fsencode(os.fspath(output))
            ).hexdigest(),
            "retry_replay_resample_provider_model_candidate_or_gate_change_count": 0,
            "schema": CANARY_ATTEMPT_SCHEMA,
            "study_id": STUDY_ID,
        }
    )
    _write_exclusive(
        output.parent / "source_free_canary.attempt.private.json",
        canonical_bytes(attempt),
    )
    implementation = load_implementation_freeze(implementation_path)
    runtime_preparer(implementation, verify_hippo_child=True)
    bindings = binding_builder(
        implementation, output.parent / "source_free_canary_runtime"
    )
    _reject_outer_dynamic_python()
    from assumption_agent.benchmarks import hitab_p1_public_canary_v1 as canary

    try:
        inner = canary.run_public_canary(
            planner_runner=bindings.planner_runner,
            cross_encoder_scorer=bindings.cross_encoder_scorer,
            minilm_encoder=bindings.minilm_encoder,
            hippo_runner=bindings.hippo_runner,
            gpu0_cache_releaser=bindings.gpu0_cache_releaser,
        )
        canary.validate_receipt(inner)
        _reject_outer_dynamic_python()
    except Exception as exc:
        raise HitabP1ProductionRuntimeError(
            "source-free production canary failed"
        ) from exc
    if (
        inner.get("repeat_count") != 2
        or inner.get("repeat_exact") is not True
        or inner.get("source_or_HiTab_rows_accessed") is not False
    ):
        raise HitabP1ProductionRuntimeError(
            "source-free production canary policy drifted"
        )
    receipt = _self_hashed(
        {
            "canary": inner,
            "canary_attempt_self_sha256": attempt["self_sha256"],
            "implementation_freeze_self_sha256": (
                implementation.self_sha256
            ),
            "network_isolation": {
                "IPAddressDeny": "any",
                "RestrictAddressFamilies": "AF_UNIX",
            },
            "online_or_API_call_count": 0,
            "qualified": True,
            "schema": CANARY_RECEIPT_SCHEMA,
            "source_or_HiTab_rows_accessed": False,
            "study_id": STUDY_ID,
        }
    )
    _write_exclusive(output, canonical_bytes(receipt))
    _validate_canary_receipt(output, implementation=implementation)
    return receipt


def _load_frozen_acquisition_factory(execution: FrozenExecution) -> object:
    before = file_sha256(
        execution.implementation.files[
            execution.acquisition_factory_file_label
        ]
    )
    try:
        module = importlib.import_module(execution.acquisition_factory_module)
        factory = getattr(module, execution.acquisition_factory_attribute)
    except (ImportError, AttributeError) as exc:
        raise HitabP1ProductionRuntimeError(
            "acquisition factory is unavailable"
        ) from exc
    after = file_sha256(
        execution.implementation.files[
            execution.acquisition_factory_file_label
        ]
    )
    if (
        before != after
        or before
        != execution.implementation.file_sha256s[
            execution.acquisition_factory_file_label
        ]
        or not callable(factory)
    ):
        raise HitabP1ProductionRuntimeError(
            "acquisition factory drifted during import"
        )
    try:
        return factory(execution)
    except Exception as exc:
        raise HitabP1ProductionRuntimeError(
            "acquisition factory failed"
        ) from exc


def run_formal_once(
    *,
    execution_freeze_path: Path,
    binding_builder: ProductionBindingBuilder = build_production_bindings,
    acquisition_factory_loader: AcquisitionFactoryLoader = (
        _load_frozen_acquisition_factory
    ),
    controller_runner: Callable[..., Mapping[str, object]] | None = None,
    runtime_preparer: RuntimePreparer = prepare_implementation_runtime,
) -> Mapping[str, object]:
    """Consume one frozen canary/execution tuple and invoke the controller once."""

    execution = load_execution_freeze(
        Path(execution_freeze_path),
        runtime_preparer=runtime_preparer,
    )
    root = execution.formal_work_root
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise HitabP1ProductionRuntimeError("formal work root is unsafe")
    os.chmod(root, 0o700)
    claim = _self_hashed(
        {
            "execution_freeze_self_sha256": execution.self_sha256,
            "implementation_freeze_self_sha256": (
                execution.implementation.self_sha256
            ),
            "source_identity_commitment": (
                execution.source_identity_commitment
            ),
            "retry_replay_resample_provider_model_candidate_or_gate_change_count": 0,
            "schema": EXECUTION_CLAIM_SCHEMA,
            "study_id": STUDY_ID,
        }
    )
    _write_exclusive(
        root / "production_execution.claim.json", canonical_bytes(claim)
    )
    # The execution attempt is consumed before model/runtime binding.  A
    # dependency or model-load failure is a formal terminal failure and must
    # not leave the same frozen root replayable.
    bindings = binding_builder(
        execution.implementation, root / "production_runtime"
    )
    _reject_outer_dynamic_python()
    acquisition = acquisition_factory_loader(execution)
    if controller_runner is None:
        from assumption_agent.benchmarks import (
            hitab_p1_formal_controller_v1 as controller,
        )

        controller_runner = controller.run_formal_controller
    try:
        result = controller_runner(
            work_root=root,
            execution_binding_sha256=execution.self_sha256,
            acquisition=acquisition,
            planner_runner=bindings.planner_runner,
            cross_encoder_scorer=bindings.cross_encoder_scorer,
            minilm_encoder=bindings.minilm_encoder,
            hippo_runner=bindings.hippo_runner,
            gpu0_cache_releaser=bindings.gpu0_cache_releaser,
        )
        _reject_outer_dynamic_python()
        return result
    except Exception as exc:
        raise HitabP1ProductionRuntimeError(
            "formal controller terminated; replay is forbidden"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    canary_parser = subparsers.add_parser("canary")
    canary_parser.add_argument(
        "--implementation-freeze", required=True, type=Path
    )
    canary_parser.add_argument("--output", required=True, type=Path)
    acquisition_parser = subparsers.add_parser("acquire")
    acquisition_parser.add_argument(
        "--acquisition-freeze", required=True, type=Path
    )
    formal_parser = subparsers.add_parser("formal")
    formal_parser.add_argument("--execution-freeze", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "canary":
        run_source_free_canary_once(
            implementation_freeze_path=arguments.implementation_freeze,
            output_path=arguments.output,
        )
    elif arguments.command == "acquire":
        run_source_acquisition_once(
            acquisition_freeze_path=arguments.acquisition_freeze,
        )
    else:
        run_formal_once(
            execution_freeze_path=arguments.execution_freeze,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANARY_ATTEMPT_SCHEMA",
    "CANARY_RECEIPT_SCHEMA",
    "ACQUISITION_FREEZE_SCHEMA",
    "ACQUISITION_FACTORY_ATTRIBUTE",
    "ACQUISITION_FACTORY_FILE_LABEL",
    "ACQUISITION_FACTORY_MODULE",
    "CPU_THREADS_PER_GPU_LANE",
    "EXECUTION_FREEZE_SCHEMA",
    "FrozenExecution",
    "FrozenAcquisition",
    "FrozenImplementation",
    "FrozenPythonRuntime",
    "HIPPORAG_WORKER_MODULE",
    "HitabP1ProductionRuntimeError",
    "HippoFreshProcessRunner",
    "IMPLEMENTATION_FREEZE_SCHEMA",
    "ProductionBindings",
    "REQUIRED_FILE_LABELS",
    "REQUIRED_MODEL_LABELS",
    "RUNTIME_POLICY",
    "STUDY_ID",
    "VERSION",
    "build_production_bindings",
    "build_gpu0_cache_releaser",
    "canonical_bytes",
    "file_sha256",
    "load_execution_freeze",
    "load_acquisition_freeze",
    "load_implementation_freeze",
    "main",
    "model_tree_sha256",
    "prepare_implementation_runtime",
    "run_formal_once",
    "run_source_acquisition_once",
    "run_source_free_canary_once",
    "stable_hash",
]

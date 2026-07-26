"""Source-free runtime and official-comparator building blocks for MAUD P2.

This module is deliberately benchmark-blind.  It has no MAUD downloader,
SQuAD parser, answer/span field, scoring function, evaluator, provider, or API
surface.  It supports three prospective operations only:

* passively fingerprint the post-reboot 311 runtime and frozen local assets;
* run the one allowed fixed public-synthetic contract canary; and
* bulk-submit private one-contract official HippoRAG jobs with two GPU lanes.

Each official child uses ``python -S -B -m`` with the deployed project root
first in an explicit import closure.  Its model argv values are the short
cwd-local aliases ``smollm2`` and ``minilm``.  Worker stdout and stderr are
created with mode 0600 outside the destroyed scratch tree; public failure
metadata contains hashes, byte counts, return code, and a safe phase only.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Callable, Mapping, Sequence

from replication_runtime.maud_extraction_p2_official_v1 import (
    worker as official_worker,
)


VERSION = "maud_extraction_p2_runtime_v1"
STUDY_ID = "MAUD_EXTRACTION_P2_CGROUP_BOUNDED_EVALUATOR_V1"
STUDY_DESIGN_SELF_SHA256 = (
    "01a1d2ef33eb9721f1644ca748ed13b26b9a6d3b96fba62c603363a104a87cbd"
)
PRE_SOURCE_CLARIFICATION_SELF_SHA256 = (
    "e774e12b44611c89f2de7efc4a23ce28a92a07e8cbd839ea5f450cac877b3aca"
)
RUNTIME_FINGERPRINT_SCHEMA = f"{VERSION}_source_free_fingerprint"
SYNTHETIC_CANARY_SCHEMA = f"{VERSION}_source_free_synthetic_canary"
SAFE_TERMINAL_SCHEMA = f"{VERSION}_safe_worker_terminal"

MAX_GPU_LANES = 2
MAX_CPU_WORKERS = 4
OUTER_CGROUP_CPU_QUOTA_PERCENT = 400
OUTER_CGROUP_MEMORY_MAX_BYTES = 40 * 1024 * 1024 * 1024
OUTER_CGROUP_TASKS_MAX = 64
WORKER_TIMEOUT_SECONDS = 7_200
PHYSICAL_GPU_IDS = ("0", "1")
LLM_ALIAS = "smollm2"
EMBEDDING_ALIAS = "minilm"
ABSOLUTE_MODEL_ARGV_COUNT = 0

CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

EXPECTED_KERNEL = "7.0.0-28-generic"
EXPECTED_NVIDIA_DRIVER = "595.84"
EXPECTED_NVIDIA_SMI_SHA256 = (
    "dd0cbc1a839dae1cfadb5ba1ffb8e3bfed99ddd8f3d1dca8e986d68ce7d0515c"
)
EXPECTED_GPU_ROWS = (
    {
        "index": 0,
        "memory_total_mib": 8192,
        "name": "NVIDIA GeForce RTX 2080",
        "uuid": "GPU-32d6e292-70cd-50a0-405b-e344d2da8d39",
    },
    {
        "index": 1,
        "memory_total_mib": 8192,
        "name": "NVIDIA GeForce RTX 2080",
        "uuid": "GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb",
    },
)

EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_PYTHON_FILE_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
EXPECTED_OFFICIAL_PYVENV_CFG_SHA256 = (
    "973ff55fad570c3922d91779ff66db497b7fdf69c55ec102ecfd9f3b6b711e45"
)
EXPECTED_TYPED_PYVENV_CFG_SHA256 = (
    "7b20ce176e7bef11f2724ad78c24cfdd77c072b3d5dd28d075d74ed63fed9a42"
)
EXPECTED_PACKAGE_VERSIONS = {
    "click": "8.0.3",
    "distro": "1.7.0",
    "gritlm": "1.0.2",
    "hipporag": "2.0.0a4",
    "huggingface-hub": "0.25.2",
    "litellm": "1.73.1",
    "networkx": "3.3",
    "numpy": "2.1.3",
    "openai": "1.91.0",
    "pydantic": "2.10.4",
    "python-igraph": "0.11.8",
    "safetensors": "0.4.5",
    "scikit-learn": "1.5.2",
    "scipy": "1.14.1",
    "sentence-transformers": "3.1.1",
    "tenacity": "8.5.0",
    "tiktoken": "0.7.0",
    "tokenizers": "0.20.3",
    "torch": "2.4.1+cu118",
    "transformers": "4.45.2",
}
EXPECTED_TYPED_PACKAGE_VERSIONS = {
    "numpy": "2.2.6",
    "scikit-learn": "1.7.2",
    "sentence-transformers": "5.5.1",
    "torch": "2.8.0+cu128",
    "transformers": "5.10.1",
}
EXPECTED_TYPED_MODULES = (
    "assumption_agent.benchmarks.maud_extraction_p2_coordinate_worker_v1",
    "numpy",
    "sentence_transformers",
    "sklearn",
    "torch",
    "transformers",
)
EXPECTED_MODULE_IMPORT_ROOTS = {
    "click": "official_base_site_root",
    "distro": "official_base_site_root",
    "hipporag": "hipporag_source_root",
    "hipporag.HippoRAG": "hipporag_source_root",
    "hipporag.llm.transformers_llm": "hipporag_source_root",
    "hipporag.utils.config_utils": "hipporag_source_root",
    "igraph": "overlay_root",
    "networkx": "p16_site_root",
    "numpy": "p16_site_root",
    "openai": "overlay_root",
    "replication_runtime": "deployed_project_root",
    "replication_runtime.maud_extraction_p2_official_v1": (
        "deployed_project_root"
    ),
    "replication_runtime.maud_extraction_p2_official_v1.worker": (
        "deployed_project_root"
    ),
    "sentence_transformers": "p16_site_root",
    "torch": "p16_site_root",
    "transformers": "overlay_root",
}

EXPECTED_HIPPORAG_SOURCE_TREE = {
    "file_count": 60,
    "size_bytes": 332_110,
    "tree_sha256": (
        "342505c3aaa8dc5e57718e8ac695ac28f60aa66837ba717f52d6f7b536527b1f"
    ),
}
EXPECTED_HIPPORAG_PY_SHA256 = (
    "960561b080531fe4d668bde635e81f8e65620ce50bdacdd9a25531e856fa3e05"
)
EXPECTED_SMOLLM_TREE = {
    "file_count": 23,
    "size_bytes": 272_031_142,
    "tree_sha256": (
        "d626d755c99c006761d5e069aa85a73fe8b011c6c0f5d0323a6f8de85246bcb5"
    ),
}
EXPECTED_MINILM_TREE = {
    "file_count": 11,
    "size_bytes": 91_578_415,
    "tree_sha256": (
        "42d8d798e4f01e68d9bb10634b9c712de00f7f8495271636fd6311b2db58e506"
    ),
}
EXPECTED_CROSS_ENCODER_TREE = {
    "file_count": 14,
    "size_bytes": 91_816_579,
    "tree_sha256": (
        "3c72ad94f790f807ed4ef5dde918b00e3493ddbd66ea557fcb22dc3cf6910cca"
    ),
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class MaudExtractionP2RuntimeError(RuntimeError):
    """The source-free runtime or isolated official worker failed closed."""

    def __init__(
        self, message: str, *, safe_terminal: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.safe_terminal = (
            dict(safe_terminal) if safe_terminal is not None else None
        )


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MaudExtractionP2RuntimeError(
            "runtime value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validated_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MaudExtractionP2RuntimeError(f"{field} is invalid")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise MaudExtractionP2RuntimeError(
            "a frozen runtime file is unreadable"
        ) from exc
    return digest.hexdigest()


def _absolute_path(value: object, field: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise MaudExtractionP2RuntimeError(f"{field} path is invalid")
    raw = os.fspath(value)
    if not raw or "\x00" in raw or not os.path.isabs(raw):
        raise MaudExtractionP2RuntimeError(f"{field} path is not absolute")
    return os.path.abspath(raw)


@dataclass(frozen=True)
class RuntimePaths:
    """Exact lexical roots for both typed and official 311 runtimes."""

    deployed_project_root: str
    official_python: str
    official_pyvenv_cfg: str
    overlay_root: str
    hipporag_source_root: str
    p16_site_root: str
    official_base_site_root: str
    smollm_model_root: str
    minilm_model_root: str
    typed_python: str
    typed_pyvenv_cfg: str
    typed_site_root: str
    cross_encoder_model_root: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field,
                _absolute_path(getattr(self, field), field),
            )

    def path_commitments(self) -> dict[str, str]:
        return {
            f"{field}_sha256": hashlib.sha256(
                getattr(self, field).encode("utf-8")
            ).hexdigest()
            for field in self.__dataclass_fields__
        }

    def pythonpath(self) -> str:
        return os.pathsep.join(
            (
                self.deployed_project_root,
                self.overlay_root,
                self.hipporag_source_root,
                self.p16_site_root,
                self.official_base_site_root,
            )
        )

    def typed_pythonpath(self) -> str:
        return os.pathsep.join(
            (self.deployed_project_root, self.typed_site_root)
        )


def _tree_receipt(
    root: Path, *, exclude_bytecode: bool = False
) -> dict[str, object]:
    lexical = root.absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise MaudExtractionP2RuntimeError(
            "frozen tree root is unavailable or a symlink"
        )
    rows: list[dict[str, object]] = []
    try:
        for current, directories, files in os.walk(
            lexical, followlinks=False
        ):
            base = Path(current)
            for name in directories:
                if (base / name).is_symlink():
                    raise MaudExtractionP2RuntimeError(
                        "frozen tree contains a directory symlink"
                    )
            if exclude_bytecode:
                directories[:] = [
                    name for name in directories if name != "__pycache__"
                ]
            for name in files:
                path = base / name
                if exclude_bytecode and path.suffix == ".pyc":
                    continue
                metadata = path.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                ):
                    raise MaudExtractionP2RuntimeError(
                        "frozen tree contains a non-regular file"
                    )
                rows.append(
                    {
                        "path": path.relative_to(lexical).as_posix(),
                        "sha256": _sha256_file(path),
                        "size_bytes": metadata.st_size,
                    }
                )
    except OSError as exc:
        raise MaudExtractionP2RuntimeError(
            "frozen tree cannot be inspected"
        ) from exc
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "file_count": len(rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "tree_sha256": semantic_sha256(rows),
    }


def _require_tree(
    path: str,
    expected: Mapping[str, object],
    *,
    exclude_bytecode: bool = False,
) -> dict[str, object]:
    observed = _tree_receipt(
        Path(path), exclude_bytecode=exclude_bytecode
    )
    if observed != dict(expected):
        raise MaudExtractionP2RuntimeError("frozen asset tree drifted")
    return observed


def production_filesystem_inspector(
    paths: RuntimePaths,
) -> dict[str, object]:
    """Passively bind executable, source, model, and implementation bytes."""

    if not isinstance(paths, RuntimePaths):
        raise MaudExtractionP2RuntimeError("runtime paths drifted")
    official_python = Path(paths.official_python)
    typed_python = Path(paths.typed_python)
    try:
        official_resolved = official_python.resolve(strict=True)
        typed_resolved = typed_python.resolve(strict=True)
    except OSError as exc:
        raise MaudExtractionP2RuntimeError(
            "runtime Python is unavailable"
        ) from exc
    official_hash = _sha256_file(official_resolved)
    typed_hash = _sha256_file(typed_resolved)
    if (
        official_hash != EXPECTED_PYTHON_FILE_SHA256
        or typed_hash != EXPECTED_PYTHON_FILE_SHA256
        or _sha256_file(Path(paths.official_pyvenv_cfg))
        != EXPECTED_OFFICIAL_PYVENV_CFG_SHA256
        or _sha256_file(Path(paths.typed_pyvenv_cfg))
        != EXPECTED_TYPED_PYVENV_CFG_SHA256
    ):
        raise MaudExtractionP2RuntimeError(
            "runtime Python identity drifted"
        )

    source = _require_tree(
        paths.hipporag_source_root,
        EXPECTED_HIPPORAG_SOURCE_TREE,
        exclude_bytecode=True,
    )
    hipporag_py = (
        Path(paths.hipporag_source_root) / "hipporag" / "HippoRAG.py"
    )
    if _sha256_file(hipporag_py) != EXPECTED_HIPPORAG_PY_SHA256:
        raise MaudExtractionP2RuntimeError(
            "patched HippoRAG implementation drifted"
        )
    worker_path = (
        Path(paths.deployed_project_root)
        / "replication_runtime"
        / "maud_extraction_p2_official_v1"
        / "worker.py"
    )
    package_init_path = worker_path.with_name("__init__.py")
    replication_init_path = (
        Path(paths.deployed_project_root)
        / "replication_runtime"
        / "__init__.py"
    )
    controller_path = (
        Path(paths.deployed_project_root)
        / "assumption_agent"
        / "benchmarks"
        / "maud_extraction_p2_runtime_v1.py"
    )
    return {
        "cross_encoder": _require_tree(
            paths.cross_encoder_model_root,
            EXPECTED_CROSS_ENCODER_TREE,
        ),
        "hipporag_source": source,
        "hipporag_source_excludes_bytecode": True,
        "hipporag_py_sha256": EXPECTED_HIPPORAG_PY_SHA256,
        "minilm": _require_tree(
            paths.minilm_model_root, EXPECTED_MINILM_TREE
        ),
        "official_python_file_sha256": official_hash,
        "official_pyvenv_cfg_sha256": (
            EXPECTED_OFFICIAL_PYVENV_CFG_SHA256
        ),
        "runtime_controller_file_sha256": _sha256_file(controller_path),
        "runtime_package_init_file_sha256": _sha256_file(
            package_init_path
        ),
        "runtime_replication_init_file_sha256": _sha256_file(
            replication_init_path
        ),
        "runtime_worker_file_sha256": _sha256_file(worker_path),
        "smollm2": _require_tree(
            paths.smollm_model_root, EXPECTED_SMOLLM_TREE
        ),
        "typed_python_file_sha256": typed_hash,
        "typed_pyvenv_cfg_sha256": EXPECTED_TYPED_PYVENV_CFG_SHA256,
    }


def _origin_root(
    origin: object, paths: RuntimePaths
) -> tuple[str, str]:
    if not isinstance(origin, str) or not os.path.isabs(origin):
        raise MaudExtractionP2RuntimeError(
            "module origin is not an absolute path"
        )
    try:
        candidate = Path(origin).resolve(strict=True)
    except OSError as exc:
        raise MaudExtractionP2RuntimeError(
            "module origin is unavailable"
        ) from exc
    roots = {
        "deployed_project_root": Path(
            paths.deployed_project_root
        ).resolve(strict=True),
        "overlay_root": Path(paths.overlay_root).resolve(strict=True),
        "hipporag_source_root": Path(
            paths.hipporag_source_root
        ).resolve(strict=True),
        "p16_site_root": Path(paths.p16_site_root).resolve(strict=True),
        "official_base_site_root": Path(
            paths.official_base_site_root
        ).resolve(strict=True),
    }
    matches: list[tuple[int, str]] = []
    for label, root in roots.items():
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        matches.append((len(root.parts), label))
    if not matches:
        raise MaudExtractionP2RuntimeError(
            "module origin escaped the explicit import closure"
        )
    label = max(matches)[1]
    return label, _sha256_file(candidate)


def _worker_environment(
    paths: RuntimePaths, *, physical_gpu: str, private_root: Path
) -> dict[str, str]:
    if physical_gpu not in PHYSICAL_GPU_IDS:
        raise MaudExtractionP2RuntimeError("physical GPU lane drifted")
    return {
        "PATH": f"{Path(paths.official_python).parent}:/usr/bin:/bin",
        "HOME": str(private_root / "home"),
        "HF_HOME": str(private_root / "cache"),
        "TMPDIR": str(private_root / "tmp"),
        "TMP": str(private_root / "tmp"),
        "TEMP": str(private_root / "tmp"),
        "PYTHONPATH": paths.pythonpath(),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": physical_gpu,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **CPU_THREAD_ENV,
    }


def production_runtime_inspector(
    paths: RuntimePaths,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Import the exact transitive module closure without loading a model."""

    modules = tuple(sorted(EXPECTED_MODULE_IMPORT_ROOTS))
    packages = tuple(sorted(EXPECTED_PACKAGE_VERSIONS))
    script = (
        "import hashlib,importlib,importlib.metadata,json,os,sys\n"
        "modules=json.loads(sys.argv[1]);packages=json.loads(sys.argv[2])\n"
        "loaded={name:importlib.import_module(name) for name in modules}\n"
        "origins={name:str(getattr(module,'__file__',None)) "
        "for name,module in loaded.items()}\n"
        "value={'python_version':'.'.join(map(str,sys.version_info[:3])),"
        "'package_versions':{name:importlib.metadata.version(name) "
        "for name in packages},'module_origins':origins,"
        "'sys_path':list(sys.path),"
        "'pythonpath':os.environ.get('PYTHONPATH'),"
        "'pythondontwritebytecode':os.environ.get("
        "'PYTHONDONTWRITEBYTECODE')}\n"
        "print(json.dumps(value,sort_keys=True,separators=(',',':')))\n"
    )
    private_root = Path("/tmp")
    environment = _worker_environment(
        paths, physical_gpu="0", private_root=private_root
    )
    environment.update(
        {
            "HOME": "/tmp",
            "HF_HOME": "/tmp",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
        }
    )
    command = [
        paths.official_python,
        "-S",
        "-B",
        "-c",
        script,
        json.dumps(list(modules), separators=(",", ":")),
        json.dumps(list(packages), separators=(",", ":")),
    ]
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MaudExtractionP2RuntimeError(
            "source-free import closure inspection failed"
        ) from exc
    returncode = getattr(completed, "returncode", None)
    stdout = getattr(completed, "stdout", None)
    stderr = getattr(completed, "stderr", None)
    if returncode != 0 or not isinstance(stdout, bytes) or not isinstance(
        stderr, bytes
    ):
        raise MaudExtractionP2RuntimeError(
            "source-free import closure inspection failed"
        )
    try:
        value = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaudExtractionP2RuntimeError(
            "source-free import closure output is invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("python_version") != EXPECTED_PYTHON_VERSION
        or value.get("package_versions") != EXPECTED_PACKAGE_VERSIONS
        or value.get("pythonpath") != paths.pythonpath()
        or value.get("pythondontwritebytecode") != "1"
        or not isinstance(value.get("module_origins"), Mapping)
        or set(value["module_origins"]) != set(modules)
        or not isinstance(value.get("sys_path"), list)
    ):
        raise MaudExtractionP2RuntimeError(
            "source-free import closure identity drifted"
        )
    sys_path = value["sys_path"]
    declared_roots = [
        paths.deployed_project_root,
        paths.overlay_root,
        paths.hipporag_source_root,
        paths.p16_site_root,
        paths.official_base_site_root,
    ]
    positions = []
    for root in declared_roots:
        try:
            positions.append(sys_path.index(root))
        except ValueError as exc:
            raise MaudExtractionP2RuntimeError(
                "explicit import root is absent from sys.path"
            ) from exc
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise MaudExtractionP2RuntimeError(
            "explicit import root order drifted"
        )
    origin_rows = {}
    for name, raw_origin in value["module_origins"].items():
        label, file_hash = _origin_root(raw_origin, paths)
        if EXPECTED_MODULE_IMPORT_ROOTS.get(name) != label:
            raise MaudExtractionP2RuntimeError(
                "transitive module import root drifted"
            )
        origin_rows[name] = {
            "file_sha256": file_hash,
            "import_root": label,
            "origin_path_sha256": hashlib.sha256(
                str(raw_origin).encode("utf-8")
            ).hexdigest(),
        }
    return {
        "command_flags": ["-S", "-B", "-c"],
        "explicit_pythonpath_order": [
            "deployed_project_root",
            "overlay_root",
            "hipporag_source_root",
            "p16_site_root",
            "official_base_site_root",
        ],
        "module_origins": origin_rows,
        "package_versions": dict(EXPECTED_PACKAGE_VERSIONS),
        "python_version": EXPECTED_PYTHON_VERSION,
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "sys_path_sha256": semantic_sha256(sys_path),
    }


def production_typed_runtime_inspector(
    paths: RuntimePaths,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Bind the isolated typed-model venv without official-site fallback."""

    typed_site = Path(paths.typed_site_root)
    if typed_site.is_symlink() or not typed_site.is_dir():
        raise MaudExtractionP2RuntimeError(
            "typed site-packages root is unavailable"
        )
    modules = tuple(EXPECTED_TYPED_MODULES)
    packages = tuple(sorted(EXPECTED_TYPED_PACKAGE_VERSIONS))
    script = (
        "import importlib,importlib.metadata,json,os,sys\n"
        "modules=json.loads(sys.argv[1]);packages=json.loads(sys.argv[2])\n"
        "loaded={name:importlib.import_module(name) for name in modules}\n"
        "value={'python_version':'.'.join(map(str,sys.version_info[:3])),"
        "'package_versions':{name:importlib.metadata.version(name) "
        "for name in packages},"
        "'module_origins':{name:str(getattr(module,'__file__',None)) "
        "for name,module in loaded.items()},"
        "'pythonpath':os.environ.get('PYTHONPATH'),"
        "'sys_path':list(sys.path),"
        "'torch_cuda_version':loaded['torch'].version.cuda}\n"
        "print(json.dumps(value,sort_keys=True,separators=(',',':')))\n"
    )
    environment = {
        "PATH": f"{Path(paths.typed_python).parent}:/usr/bin:/bin",
        "HOME": "/tmp",
        "HF_HOME": "/tmp",
        "TMPDIR": "/tmp",
        "TMP": "/tmp",
        "TEMP": "/tmp",
        "PYTHONPATH": paths.typed_pythonpath(),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **CPU_THREAD_ENV,
    }
    command = [
        paths.typed_python,
        "-S",
        "-B",
        "-c",
        script,
        json.dumps(list(modules), separators=(",", ":")),
        json.dumps(list(packages), separators=(",", ":")),
    ]
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MaudExtractionP2RuntimeError(
            "typed import closure inspection failed"
        ) from exc
    stdout = getattr(completed, "stdout", None)
    stderr = getattr(completed, "stderr", None)
    if (
        getattr(completed, "returncode", None) != 0
        or not isinstance(stdout, bytes)
        or not isinstance(stderr, bytes)
    ):
        raise MaudExtractionP2RuntimeError(
            "typed import closure inspection failed"
        )
    try:
        value = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaudExtractionP2RuntimeError(
            "typed import closure output is invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("python_version") != EXPECTED_PYTHON_VERSION
        or value.get("package_versions")
        != EXPECTED_TYPED_PACKAGE_VERSIONS
        or value.get("pythonpath") != paths.typed_pythonpath()
        or value.get("torch_cuda_version") != "12.8"
        or not isinstance(value.get("module_origins"), Mapping)
        or set(value["module_origins"]) != set(modules)
        or not isinstance(value.get("sys_path"), list)
    ):
        raise MaudExtractionP2RuntimeError(
            "typed import closure identity drifted"
        )
    expected_roots = {
        modules[0]: Path(paths.deployed_project_root).resolve(strict=True),
        **{
            name: typed_site.resolve(strict=True)
            for name in modules[1:]
        },
    }
    origin_rows: dict[str, object] = {}
    for name, raw_origin in value["module_origins"].items():
        if not isinstance(raw_origin, str) or not os.path.isabs(raw_origin):
            raise MaudExtractionP2RuntimeError(
                "typed module origin drifted"
            )
        origin = Path(raw_origin).resolve(strict=True)
        try:
            origin.relative_to(expected_roots[name])
        except ValueError as exc:
            raise MaudExtractionP2RuntimeError(
                "typed module escaped isolated site closure"
            ) from exc
        origin_rows[name] = {
            "file_sha256": _sha256_file(origin),
            "origin_path_sha256": hashlib.sha256(
                raw_origin.encode("utf-8")
            ).hexdigest(),
            "import_root": (
                "deployed_project_root"
                if name == modules[0]
                else "typed_site_root"
            ),
        }
    sys_path = value["sys_path"]
    if any(
        forbidden in sys_path
        for forbidden in (
            paths.overlay_root,
            paths.hipporag_source_root,
            paths.p16_site_root,
            paths.official_base_site_root,
        )
    ):
        raise MaudExtractionP2RuntimeError(
            "official import root leaked into typed runtime"
        )
    return {
        "command_flags": ["-S", "-B", "-c"],
        "explicit_pythonpath_order": [
            "deployed_project_root",
            "typed_site_root",
        ],
        "module_origins": origin_rows,
        "package_versions": dict(EXPECTED_TYPED_PACKAGE_VERSIONS),
        "python_version": EXPECTED_PYTHON_VERSION,
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "sys_path_sha256": semantic_sha256(sys_path),
        "torch_cuda_version": "12.8",
    }


def _parse_gpu_rows(raw: bytes) -> tuple[dict[str, object], ...]:
    try:
        lines = [
            line.strip()
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except UnicodeDecodeError as exc:
        raise MaudExtractionP2RuntimeError(
            "nvidia-smi output is invalid"
        ) from exc
    rows = []
    drivers = set()
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise MaudExtractionP2RuntimeError(
                "nvidia-smi row shape drifted"
            )
        try:
            index = int(parts[0])
            memory = int(parts[3])
        except ValueError as exc:
            raise MaudExtractionP2RuntimeError(
                "nvidia-smi numeric field drifted"
            ) from exc
        rows.append(
            {
                "index": index,
                "memory_total_mib": memory,
                "name": parts[2],
                "uuid": parts[1],
            }
        )
        drivers.add(parts[4])
    if tuple(rows) != EXPECTED_GPU_ROWS or drivers != {
        EXPECTED_NVIDIA_DRIVER
    }:
        raise MaudExtractionP2RuntimeError(
            "post-reboot GPU identity drifted"
        )
    return tuple(rows)


def production_hardware_inspector(
    nvidia_smi: str | os.PathLike[str],
    *,
    runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    executable = Path(_absolute_path(nvidia_smi, "nvidia-smi"))
    if _sha256_file(executable) != EXPECTED_NVIDIA_SMI_SHA256:
        raise MaudExtractionP2RuntimeError("nvidia-smi identity drifted")
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}
    try:
        kernel_run = runner(
            ["/usr/bin/uname", "-r"],
            check=False,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
        gpu_run = runner(
            [
                str(executable),
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MaudExtractionP2RuntimeError(
            "post-reboot hardware inspection failed"
        ) from exc
    if (
        getattr(kernel_run, "returncode", None) != 0
        or getattr(gpu_run, "returncode", None) != 0
        or not isinstance(getattr(kernel_run, "stdout", None), bytes)
        or not isinstance(getattr(gpu_run, "stdout", None), bytes)
    ):
        raise MaudExtractionP2RuntimeError(
            "post-reboot hardware inspection failed"
        )
    kernel = kernel_run.stdout.decode("ascii").strip()
    if kernel != EXPECTED_KERNEL:
        raise MaudExtractionP2RuntimeError("post-reboot kernel drifted")
    rows = _parse_gpu_rows(gpu_run.stdout)
    return {
        "gpu_count": len(rows),
        "gpus": list(rows),
        "kernel": kernel,
        "nvidia_driver": EXPECTED_NVIDIA_DRIVER,
        "nvidia_smi_file_sha256": EXPECTED_NVIDIA_SMI_SHA256,
    }


def build_source_free_runtime_fingerprint(
    paths: RuntimePaths,
    *,
    nvidia_smi: str | os.PathLike[str],
    filesystem_inspector: Callable[[RuntimePaths], Mapping[str, object]] = (
        production_filesystem_inspector
    ),
    runtime_inspector: Callable[[RuntimePaths], Mapping[str, object]] = (
        production_runtime_inspector
    ),
    typed_runtime_inspector: Callable[
        [RuntimePaths], Mapping[str, object]
    ] = production_typed_runtime_inspector,
    hardware_inspector: Callable[
        [str | os.PathLike[str]], Mapping[str, object]
    ] = production_hardware_inspector,
) -> dict[str, object]:
    """Build a path-free, source-free post-reboot runtime receipt."""

    if not isinstance(paths, RuntimePaths):
        raise MaudExtractionP2RuntimeError("runtime paths drifted")
    filesystem = dict(filesystem_inspector(paths))
    imports = dict(runtime_inspector(paths))
    typed_imports = dict(typed_runtime_inspector(paths))
    hardware = dict(hardware_inspector(nvidia_smi))
    body = {
        "schema": RUNTIME_FINGERPRINT_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "pre_source_clarification_self_sha256": (
            PRE_SOURCE_CLARIFICATION_SELF_SHA256
        ),
        "status": "verified_source_free_post_reboot_runtime_fingerprint",
        "path_commitments": paths.path_commitments(),
        "filesystem_binding": filesystem,
        "filesystem_binding_sha256": semantic_sha256(filesystem),
        "import_closure": imports,
        "import_closure_sha256": semantic_sha256(imports),
        "typed_import_closure": typed_imports,
        "typed_import_closure_sha256": semantic_sha256(typed_imports),
        "hardware": hardware,
        "hardware_sha256": semantic_sha256(hardware),
        "execution_limits": {
            "absolute_model_argv_count": ABSOLUTE_MODEL_ARGV_COUNT,
            "cpu_thread_cap": MAX_CPU_WORKERS,
            "gpu_lane_cap": MAX_GPU_LANES,
            "hipporag_processes_per_gpu": 1,
            "outer_cgroup_CPUQuota_percent": (
                OUTER_CGROUP_CPU_QUOTA_PERCENT
            ),
            "outer_cgroup_MemoryMax_bytes": (
                OUTER_CGROUP_MEMORY_MAX_BYTES
            ),
            "outer_cgroup_TasksMax": OUTER_CGROUP_TASKS_MAX,
            "queries_per_contract_index": official_worker.QUERY_COUNT,
        },
        "claim_boundary": {
            "api_or_online_evaluator_call_count": 0,
            "formal_MAUD_file_or_row_access_count": 0,
            "formal_action_or_score_count": 0,
            "model_inference_call_count": 0,
            "source_free_import_probe_count": 2,
        },
    }
    return {**body, "self_sha256": semantic_sha256(body)}


def _create_private_directory(path: Path) -> None:
    if os.path.lexists(path):
        raise MaudExtractionP2RuntimeError(
            "private runtime directory is not fresh"
        )
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise MaudExtractionP2RuntimeError(
            "private runtime directory parent is unsafe"
        )
    os.mkdir(path, 0o700)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise MaudExtractionP2RuntimeError(
            "private runtime directory mode drifted"
        )


def _write_private_file(path: Path, raw: bytes) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MaudExtractionP2RuntimeError(
            "private runtime file cannot be created"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_private_log(path: Path):
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "wb")


def _destroy_private_scratch(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise MaudExtractionP2RuntimeError(
            "private runtime scratch cleanup failed; no retry permitted"
        ) from exc
    if os.path.lexists(path):
        raise MaudExtractionP2RuntimeError(
            "private runtime scratch survived cleanup; no retry permitted"
        )


def _private_file_receipt(path: Path) -> dict[str, object]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise MaudExtractionP2RuntimeError(
            "private custody file mode drifted"
        )
    return {
        "bytes": metadata.st_size,
        "mode": "0600",
        "sha256": _sha256_file(path),
    }


def _safe_terminal(
    *,
    phase: str,
    returncode: int,
    stdout_path: Path,
    stderr_path: Path,
    output_raw: bytes | None,
) -> dict[str, object]:
    value = {
        "schema": SAFE_TERMINAL_SCHEMA,
        "safe_phase": phase,
        "returncode": returncode,
        "stdout": _private_file_receipt(stdout_path),
        "stderr": _private_file_receipt(stderr_path),
        "output": (
            None
            if output_raw is None
            else {
                "bytes": len(output_raw),
                "sha256": hashlib.sha256(output_raw).hexdigest(),
            }
        ),
        "private_content_exposed": False,
    }
    return {**value, "terminal_sha256": semantic_sha256(value)}


def _validate_output_binding(
    payload: Mapping[str, object], output: Mapping[str, object]
) -> None:
    contract_id, corpus_hash, documents, queries = (
        official_worker.validate_input(payload)
    )
    rows = output.get("rows")
    if (
        output.get("contract_work_id") != contract_id
        or output.get("corpus_sha256") != corpus_hash
        or output.get("passage_count") != len(documents)
        or not isinstance(rows, list)
        or len(rows) != len(queries)
    ):
        raise MaudExtractionP2RuntimeError(
            "official output escaped its contract input binding"
        )
    for query, row in zip(queries, rows):
        if (
            not isinstance(row, Mapping)
            or row.get("query_ordinal") != query.ordinal
            or row.get("work_id") != query.work_id
        ):
            raise MaudExtractionP2RuntimeError(
                "official output escaped its query input binding"
            )


def _create_short_aliases(
    alias_cwd: Path, paths: RuntimePaths
) -> dict[str, object]:
    _create_private_directory(alias_cwd)
    aliases = {
        LLM_ALIAS: Path(paths.smollm_model_root),
        EMBEDDING_ALIAS: Path(paths.minilm_model_root),
    }
    rows = {}
    for alias, target in aliases.items():
        if not target.is_absolute() or target.is_symlink() or not target.is_dir():
            raise MaudExtractionP2RuntimeError(
                "model alias target is unavailable"
            )
        link = alias_cwd / alias
        try:
            os.symlink(str(target), link, target_is_directory=True)
            metadata = link.lstat()
            resolved = link.resolve(strict=True)
        except OSError as exc:
            raise MaudExtractionP2RuntimeError(
                "short model alias cannot be created"
            ) from exc
        if (
            not stat.S_ISLNK(metadata.st_mode)
            or os.readlink(link) != str(target)
            or not os.path.samefile(resolved, target)
        ):
            raise MaudExtractionP2RuntimeError(
                "short model alias binding drifted"
            )
        rows[alias] = {
            "alias_is_single_component": True,
            "link_target_sha256": hashlib.sha256(
                str(target).encode("utf-8")
            ).hexdigest(),
            "resolved_path_sha256": hashlib.sha256(
                str(resolved).encode("utf-8")
            ).hexdigest(),
            "samefile": True,
        }
    return rows


@dataclass(frozen=True)
class WorkerRun:
    output: dict[str, object]
    safe_terminal: dict[str, object]


def production_contract_launcher(
    *,
    payload: Mapping[str, object],
    runtime_paths: RuntimePaths,
    scratch_root: str | os.PathLike[str],
    private_custody_root: str | os.PathLike[str],
    physical_gpu: str,
    timeout_seconds: int = WORKER_TIMEOUT_SECONDS,
    runner: Callable[..., object] = subprocess.run,
) -> WorkerRun:
    """Run one contract once, preserving private logs outside scratch."""

    official_worker.validate_input(payload)
    if not isinstance(runtime_paths, RuntimePaths):
        raise MaudExtractionP2RuntimeError("runtime paths drifted")
    if physical_gpu not in PHYSICAL_GPU_IDS:
        raise MaudExtractionP2RuntimeError("physical GPU lane drifted")
    if timeout_seconds != WORKER_TIMEOUT_SECONDS:
        raise MaudExtractionP2RuntimeError("worker timeout drifted")
    scratch = Path(_absolute_path(scratch_root, "scratch root"))
    custody = Path(
        _absolute_path(private_custody_root, "private custody root")
    )
    for left, right in ((scratch, custody), (custody, scratch)):
        try:
            left.relative_to(right)
        except ValueError:
            continue
        raise MaudExtractionP2RuntimeError(
            "private custody must be outside destroyed scratch"
        )
    _create_private_directory(custody)
    _create_private_directory(scratch)
    stdout_path = custody / "worker.stdout.private.bin"
    stderr_path = custody / "worker.stderr.private.bin"
    output_raw: bytes | None = None
    try:
        for name in ("home", "cache", "tmp"):
            os.mkdir(scratch / name, 0o700)
        alias_cwd = scratch / "model_aliases"
        _create_short_aliases(alias_cwd, runtime_paths)
        input_path = scratch / "contract.input.private.json"
        output_path = scratch / "contract.output.private.json"
        index_root = scratch / "hipporag_contract_index"
        _write_private_file(
            input_path,
            official_worker.canonical_json_bytes(dict(payload)),
        )
        environment = _worker_environment(
            runtime_paths,
            physical_gpu=physical_gpu,
            private_root=scratch,
        )
        command = [
            runtime_paths.official_python,
            "-S",
            "-B",
            "-m",
            "replication_runtime.maud_extraction_p2_official_v1.worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            LLM_ALIAS,
            "--embedding-model",
            EMBEDDING_ALIAS,
        ]
        if any(
            argument
            in {
                runtime_paths.smollm_model_root,
                runtime_paths.minilm_model_root,
            }
            for argument in command
        ):
            raise MaudExtractionP2RuntimeError(
                "absolute model path escaped into worker argv"
            )
        try:
            with _open_private_log(stdout_path) as stdout_handle, (
                _open_private_log(stderr_path)
            ) as stderr_handle:
                completed = runner(
                    command,
                    check=False,
                    cwd=alias_cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=WORKER_TIMEOUT_SECONDS,
                )
                stdout_handle.flush()
                stderr_handle.flush()
                os.fsync(stdout_handle.fileno())
                os.fsync(stderr_handle.fileno())
        except subprocess.TimeoutExpired as exc:
            terminal = _safe_terminal(
                phase="worker_timeout",
                returncode=-124,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_raw=None,
            )
            raise MaudExtractionP2RuntimeError(
                "official worker timed out; no retry permitted",
                safe_terminal=terminal,
            ) from exc
        except OSError as exc:
            terminal = _safe_terminal(
                phase="worker_launch_failed",
                returncode=-125,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_raw=None,
            )
            raise MaudExtractionP2RuntimeError(
                "official worker launch failed; no retry permitted",
                safe_terminal=terminal,
            ) from exc
        returncode = getattr(completed, "returncode", None)
        if type(returncode) is not int:
            raise MaudExtractionP2RuntimeError(
                "official worker return code drifted"
            )
        if returncode != 0:
            terminal = _safe_terminal(
                phase="worker_failed",
                returncode=returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_raw=None,
            )
            raise MaudExtractionP2RuntimeError(
                "official worker failed; no retry permitted",
                safe_terminal=terminal,
            )
        if output_path.is_symlink() or not output_path.is_file():
            terminal = _safe_terminal(
                phase="worker_output_absent",
                returncode=returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_raw=None,
            )
            raise MaudExtractionP2RuntimeError(
                "official worker output is absent",
                safe_terminal=terminal,
            )
        output_metadata = output_path.lstat()
        if (
            not stat.S_ISREG(output_metadata.st_mode)
            or stat.S_IMODE(output_metadata.st_mode) != 0o600
        ):
            terminal = _safe_terminal(
                phase="worker_output_mode_invalid",
                returncode=returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_raw=None,
            )
            raise MaudExtractionP2RuntimeError(
                "official worker output mode failed closed",
                safe_terminal=terminal,
            )
        output_raw = output_path.read_bytes()
        try:
            output = official_worker.parse_output(output_raw)
        except official_worker.MaudOfficialHippoRAGError as exc:
            terminal = _safe_terminal(
                phase="worker_output_invalid",
                returncode=returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_raw=output_raw,
            )
            raise MaudExtractionP2RuntimeError(
                "official worker output failed closed",
                safe_terminal=terminal,
            ) from exc
        try:
            _validate_output_binding(payload, output)
        except MaudExtractionP2RuntimeError as exc:
            terminal = _safe_terminal(
                phase="worker_output_binding_invalid",
                returncode=returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_raw=output_raw,
            )
            raise MaudExtractionP2RuntimeError(
                "official worker output binding failed closed",
                safe_terminal=terminal,
            ) from exc
        terminal = _safe_terminal(
            phase="worker_completed",
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            output_raw=output_raw,
        )
        return WorkerRun(output=output, safe_terminal=terminal)
    finally:
        if os.path.lexists(scratch):
            _destroy_private_scratch(scratch)


def synthetic_contract_payload() -> dict[str, object]:
    """Return the single fixed public fixture; it contains no MAUD row."""

    clauses = (
        "Section 1 defines Aurora Event as a material synthetic change.",
        "Section 2 states the ordinary-course synthetic covenant.",
        "Section 3 creates a synthetic closing condition.",
        "Section 4 provides a synthetic fiduciary exception.",
        "Section 5 describes a synthetic matching right.",
        "Section 6 states a synthetic termination remedy.",
        "Section 7 cross-references Section 3.",
        "Section 8 defines Synthetic Knowledge.",
    )
    documents = [
        {
            "ordinal": ordinal,
            "text": official_worker.canonical_passage_document(
                ordinal=ordinal, text=text
            ),
        }
        for ordinal, text in enumerate(clauses)
    ]
    query_stems = (
        "definition",
        "reference",
        "condition",
        "obligation",
        "exception",
        "remedy",
        "matching right",
        "termination",
        "ordinary course",
        "knowledge",
        "litigation",
        "accuracy",
        "compliance",
        "antitrust",
        "intervening event",
        "superior offer",
        "consideration",
        "no-shop",
        "specific performance",
        "tail period",
        "meeting covenant",
        "acquisition proposal",
    )
    queries = [
        {
            "ordinal": ordinal,
            "text": (
                "Which public synthetic passages concern the "
                f"{stem} clause?"
            ),
            "work_id": f"maud-public-synthetic-query-{ordinal:02d}",
        }
        for ordinal, stem in enumerate(query_stems)
    ]
    return official_worker.input_payload(
        contract_work_id="maud-public-synthetic-contract-v1",
        documents=documents,
        queries=queries,
    )


PUBLIC_SYNTHETIC_FIXTURE_SHA256 = semantic_sha256(
    synthetic_contract_payload()
)


def run_source_free_synthetic_canary(
    *,
    runtime_paths: RuntimePaths,
    runtime_fingerprint_sha256: str,
    scratch_root: str | os.PathLike[str],
    private_custody_root: str | os.PathLike[str],
    launcher: Callable[..., WorkerRun] = production_contract_launcher,
) -> dict[str, object]:
    """Consume the sole public canary capability without a benchmark source."""

    fingerprint = _validated_sha256(
        runtime_fingerprint_sha256, "runtime fingerprint"
    )
    payload = synthetic_contract_payload()
    result = launcher(
        payload=payload,
        runtime_paths=runtime_paths,
        scratch_root=scratch_root,
        private_custody_root=private_custody_root,
        physical_gpu="0",
        timeout_seconds=WORKER_TIMEOUT_SECONDS,
    )
    if not isinstance(result, WorkerRun):
        raise MaudExtractionP2RuntimeError(
            "synthetic canary launcher result drifted"
        )
    output = official_worker.parse_output(
        official_worker.canonical_json_bytes(result.output)
    )
    _validate_output_binding(payload, output)
    if (
        output.get("passage_count") != 8
        or len(output.get("rows", ())) != official_worker.QUERY_COUNT
    ):
        raise MaudExtractionP2RuntimeError(
            "synthetic canary output shape drifted"
        )
    body = {
        "schema": SYNTHETIC_CANARY_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "status": "passed_source_free_public_synthetic_runtime_diagnostic",
        "runtime_fingerprint_sha256": fingerprint,
        "fixture_sha256": PUBLIC_SYNTHETIC_FIXTURE_SHA256,
        "output_sha256": semantic_sha256(output),
        "safe_worker_terminal": result.safe_terminal,
        "safe_worker_terminal_sha256": semantic_sha256(
            result.safe_terminal
        ),
        "shape": {
            "contract_index_count": 1,
            "passage_count": 8,
            "query_count": official_worker.QUERY_COUNT,
            "top_k": official_worker.TOP_K,
        },
        "claim_boundary": {
            "api_or_online_evaluator_call_count": 0,
            "benchmark_source_label_or_score_access_count": 0,
            "formal_model_action_count": 0,
            "public_synthetic_contract_count": 1,
            "retry_replay_resample_count": 0,
        },
    }
    return {**body, "self_sha256": semantic_sha256(body)}


@dataclass(frozen=True)
class ContractLaunchJob:
    payload: Mapping[str, object]
    scratch_root: str
    private_custody_root: str
    physical_gpu: str

    def __post_init__(self) -> None:
        official_worker.validate_input(self.payload)
        _absolute_path(self.scratch_root, "scratch root")
        _absolute_path(self.private_custody_root, "private custody root")
        if self.physical_gpu not in PHYSICAL_GPU_IDS:
            raise MaudExtractionP2RuntimeError("physical GPU lane drifted")


def run_contract_batch(
    jobs: Sequence[ContractLaunchJob],
    *,
    runtime_paths: RuntimePaths,
    launcher: Callable[..., WorkerRun] = production_contract_launcher,
    executor_factory: Callable[..., object] = ThreadPoolExecutor,
) -> tuple[WorkerRun, ...]:
    """Bulk-submit a stage, then join, with two GPU lanes and no retry."""

    if (
        isinstance(jobs, (str, bytes))
        or not isinstance(jobs, Sequence)
        or not jobs
        or not all(isinstance(job, ContractLaunchJob) for job in jobs)
    ):
        raise MaudExtractionP2RuntimeError("contract job batch drifted")
    if not isinstance(runtime_paths, RuntimePaths):
        raise MaudExtractionP2RuntimeError("runtime paths drifted")
    scratch_roots = [job.scratch_root for job in jobs]
    custody_roots = [job.private_custody_root for job in jobs]
    if (
        len(set(scratch_roots)) != len(jobs)
        or len(set(custody_roots)) != len(jobs)
        or set(scratch_roots).intersection(custody_roots)
    ):
        raise MaudExtractionP2RuntimeError(
            "contract job private roots are not disjoint"
        )
    ordered: list[WorkerRun | None] = [None] * len(jobs)
    futures: dict[Future[object], int] = {}
    # One serial executor per physical GPU is stronger than a two-thread
    # shared pool: an uneven job duration can never co-reside two contracts on
    # the same 8 GiB card.  Every job is still submitted before the first join.
    with ExitStack() as stack:
        pools = {
            gpu: stack.enter_context(
                executor_factory(
                    max_workers=1,
                    thread_name_prefix=f"maud-p2-official-gpu-{gpu}",
                )
            )
            for gpu in PHYSICAL_GPU_IDS
        }
        for index, job in enumerate(jobs):
            future = pools[job.physical_gpu].submit(
                launcher,
                payload=job.payload,
                runtime_paths=runtime_paths,
                scratch_root=job.scratch_root,
                private_custody_root=job.private_custody_root,
                physical_gpu=job.physical_gpu,
                timeout_seconds=WORKER_TIMEOUT_SECONDS,
            )
            futures[future] = index
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                raise MaudExtractionP2RuntimeError(
                    "official contract batch failed; no retry permitted"
                ) from exc
            if not isinstance(result, WorkerRun):
                raise MaudExtractionP2RuntimeError(
                    "official contract batch result drifted"
                )
            ordered[futures[future]] = result
    if any(result is None for result in ordered):
        raise MaudExtractionP2RuntimeError(
            "official contract batch is incomplete"
        )
    return tuple(result for result in ordered if result is not None)


__all__ = [
    "ABSOLUTE_MODEL_ARGV_COUNT",
    "ContractLaunchJob",
    "EXPECTED_MODULE_IMPORT_ROOTS",
    "MAX_CPU_WORKERS",
    "MAX_GPU_LANES",
    "OUTER_CGROUP_CPU_QUOTA_PERCENT",
    "OUTER_CGROUP_MEMORY_MAX_BYTES",
    "OUTER_CGROUP_TASKS_MAX",
    "MaudExtractionP2RuntimeError",
    "PUBLIC_SYNTHETIC_FIXTURE_SHA256",
    "RuntimePaths",
    "WorkerRun",
    "build_source_free_runtime_fingerprint",
    "canonical_json_bytes",
    "production_contract_launcher",
    "production_filesystem_inspector",
    "production_hardware_inspector",
    "production_runtime_inspector",
    "production_typed_runtime_inspector",
    "run_contract_batch",
    "run_source_free_synthetic_canary",
    "semantic_sha256",
    "synthetic_contract_payload",
]

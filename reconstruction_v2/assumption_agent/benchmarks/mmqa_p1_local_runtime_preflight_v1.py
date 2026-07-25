"""Public-synthetic, non-scoring local runtime/action preflight for MMQA P1.

The controller accepts only an exact typed Python, the frozen MiniLM and
cross-encoder model roots, an ``nvidia-smi`` executable, and an output path.
There is deliberately no dataset/source, API, evaluator, HippoRAG, retry,
label, score, item-ID, or family-ID input surface.

Two isolated typed-runtime workers exercise the physical GPUs sequentially:

* physical GPU 0 encodes one fixed synthetic query plus eight synthetic
  row/text descriptions twice with MiniLM; and
* physical GPU 1 scores the eight fixed query/description pairs twice with
  the cross encoder.

The controller verifies exact runtime and asset identity before accepting the
worker results.  It then supplies the private in-memory coordinates to the
pure MMQA core to form a typed closure, connected bundles, an E0 selection,
and a small synthetic-only E5 fit.  The durable/public receipt contains only
hashes, shapes, bounded concurrency, zero-access counts, and validation
booleans.  Raw text, coordinates, logits, energies, coefficients, paths, and
synthetic local ordinals are never emitted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import mmqa_p1_typed_proof_e5_core_v1 as core


VERSION = "mmqa_p1_local_runtime_preflight_v1"
RECEIPT_SCHEMA = f"{VERSION}_receipt"
WORKER_SCHEMA = f"{VERSION}_worker"
STUDY_DESIGN_SELF_SHA256 = (
    "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
)
ADDRESS_FAMILY_ISOLATION_CONTRACT = {
    "AF_INET_and_AF_INET6_denial_probe_required": True,
    "private_network_namespace_claimed": False,
    "required_RestrictAddressFamilies": ["AF_UNIX"],
}

EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_TYPED_PYTHON_RESOLVED_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
EXPECTED_TYPED_PYVENV_CFG_SHA256 = (
    "7b20ce176e7bef11f2724ad78c24cfdd77c072b3d5dd28d075d74ed63fed9a42"
)
EXPECTED_RUNTIME_VERSIONS = {
    "huggingface-hub": "1.11.0",
    "numpy": "2.2.6",
    "safetensors": "0.7.0",
    "sentence-transformers": "5.5.1",
    "tokenizers": "0.22.2",
    "torch": "2.8.0+cu128",
    "transformers": "5.10.1",
}
EXPECTED_TORCH_CUDA_VERSION = "12.8"

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

EMBEDDING_DIMENSION = 384
MAXIMUM_SEQUENCE_LENGTH = 256
MINILM_BATCH_SIZE = 32
CE_BATCH_SIZE = 64
CE_MAXIMUM_SEQUENCE_LENGTH = 512
ROW_NORM_ATOL = 1.0e-5
MODEL_WORKER_CONCURRENCY = 1
TOTAL_TYPED_GPU_CONCURRENCY = 2
WORKER_TIMEOUT_SECONDS = 900

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class MmqaP1LocalRuntimePreflightError(RuntimeError):
    """The public-synthetic local preflight failed closed."""


@dataclass(frozen=True)
class AssetFile:
    relative_path: str
    size_bytes: int
    sha256: str


MINILM_REQUIRED_FILES = (
    AssetFile(
        "1_Pooling/config.json",
        190,
        "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    ),
    AssetFile(
        "README.md",
        10502,
        "dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0",
    ),
    AssetFile(
        "config.json",
        612,
        "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    ),
    AssetFile(
        "config_sentence_transformers.json",
        116,
        "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
    ),
    AssetFile(
        "model.safetensors",
        90868376,
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    ),
    AssetFile(
        "modules.json",
        349,
        "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    ),
    AssetFile(
        "sentence_bert_config.json",
        53,
        "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    ),
    AssetFile(
        "special_tokens_map.json",
        112,
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    ),
    AssetFile(
        "tokenizer.json",
        466247,
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    ),
    AssetFile(
        "tokenizer_config.json",
        350,
        "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    ),
    AssetFile(
        "vocab.txt",
        231508,
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
)
MINILM_REQUIRED_TREE_SHA256 = (
    "1514beb65d2d3a2824a93f133a300cc60d5b437ccd6ea1e622eb4cd9881dcfdb"
)

CE_REQUIRED_FILES = (
    AssetFile(
        "config.json",
        794,
        "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    ),
    AssetFile(
        "model.safetensors",
        90870598,
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
    ),
    AssetFile(
        "special_tokens_map.json",
        132,
        "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
    ),
    AssetFile(
        "tokenizer.json",
        711396,
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    AssetFile(
        "tokenizer_config.json",
        1330,
        "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
    ),
    AssetFile(
        "vocab.txt",
        231508,
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
)
CE_REQUIRED_TREE_SHA256 = (
    "923d4371d5fe13534d7431895890c2142a8552a441f09ec7b28d035aaae9120c"
)


@dataclass(frozen=True)
class SyntheticNodeSpec:
    node_type: str
    text: str
    entity_anchor: int
    relation_anchor: int
    numeric_or_temporal_anchor: int


SYNTHETIC_QUERY = (
    "Which linked row and text jointly identify the Aurora device's launch "
    "year and the relation connecting it to the Borealis trial?"
)
SYNTHETIC_NODES = (
    SyntheticNodeSpec(
        core.ROW,
        "Synthetic row A: Aurora device | launch year | 2012.",
        1,
        1,
        1,
    ),
    SyntheticNodeSpec(
        core.ROW,
        "Synthetic row B: Aurora device | evaluated in | Borealis trial.",
        1,
        1,
        0,
    ),
    SyntheticNodeSpec(
        core.ROW,
        "Synthetic row C: Cedar instrument | launch year | 2018.",
        0,
        1,
        1,
    ),
    SyntheticNodeSpec(
        core.ROW,
        "Synthetic row D: Delta sensor | evaluated in | Cirrus trial.",
        0,
        1,
        0,
    ),
    SyntheticNodeSpec(
        core.TEXT,
        "Synthetic text A states that Aurora was introduced in 2012.",
        1,
        0,
        1,
    ),
    SyntheticNodeSpec(
        core.TEXT,
        "Synthetic text B links the Aurora device to the Borealis trial.",
        1,
        1,
        0,
    ),
    SyntheticNodeSpec(
        core.TEXT,
        "Synthetic text C links the Cedar instrument to a different study.",
        0,
        1,
        0,
    ),
    SyntheticNodeSpec(
        core.TEXT,
        "Synthetic text D describes the unrelated Cirrus evaluation.",
        0,
        0,
        0,
    ),
)

# A reciprocal chain supplies three independent synthetic E5 slates while
# keeping every edge within the frozen ROW<->TEXT registry.
_UNDIRECTED_LINKS = (
    (0, 4),
    (1, 4),
    (1, 5),
    (2, 5),
    (2, 6),
    (3, 6),
    (3, 7),
)
_TRAINING_ANCHOR_AND_GOLD = (
    (0, (0, 1, 4)),
    (1, (1, 2, 5)),
    (2, (2, 3, 6)),
)


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MmqaP1LocalRuntimePreflightError(
            "preflight value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    return digest.hexdigest()
                digest.update(block)
    except OSError as exc:
        raise MmqaP1LocalRuntimePreflightError(
            "a frozen runtime file could not be read"
        ) from exc


def _fixture_payload() -> dict[str, object]:
    return {
        "query": SYNTHETIC_QUERY,
        "nodes": [
            {
                "entity_anchor": row.entity_anchor,
                "node_type": row.node_type,
                "numeric_or_temporal_anchor": row.numeric_or_temporal_anchor,
                "relation_anchor": row.relation_anchor,
                "text": row.text,
            }
            for row in SYNTHETIC_NODES
        ],
        "reciprocal_links": [list(row) for row in _UNDIRECTED_LINKS],
        "schema": f"{VERSION}_public_synthetic_fixture",
    }


PUBLIC_SYNTHETIC_FIXTURE_SHA256 = _semantic_hash(_fixture_payload())


def _asset_rows(files: Sequence[AssetFile]) -> list[dict[str, object]]:
    return [
        {
            "path": row.relative_path,
            "sha256": row.sha256,
            "size": row.size_bytes,
        }
        for row in files
    ]


def _validate_static_contract() -> None:
    if _semantic_hash(_asset_rows(MINILM_REQUIRED_FILES)) != (
        MINILM_REQUIRED_TREE_SHA256
    ):
        raise MmqaP1LocalRuntimePreflightError(
            "embedded MiniLM asset contract drifted"
        )
    if _semantic_hash(_asset_rows(CE_REQUIRED_FILES)) != CE_REQUIRED_TREE_SHA256:
        raise MmqaP1LocalRuntimePreflightError(
            "embedded cross-encoder asset contract drifted"
        )
    if len(SYNTHETIC_NODES) != 8 or any(
        row.node_type not in core.NODE_TYPES for row in SYNTHETIC_NODES
    ):
        raise MmqaP1LocalRuntimePreflightError("synthetic fixture drifted")


def _verified_directory(path: str | Path, field: str) -> Path:
    value = Path(path).expanduser().absolute()
    try:
        if value.is_symlink() or not value.is_dir():
            raise MmqaP1LocalRuntimePreflightError(f"{field} is unavailable")
        value.resolve(strict=True)
    except OSError as exc:
        raise MmqaP1LocalRuntimePreflightError(f"{field} is unavailable") from exc
    return value


def _verify_asset_tree(
    model_root: str | Path,
    *,
    files: Sequence[AssetFile],
    expected_tree_sha256: str,
    allowed_top_level_directories: frozenset[str],
    role: str,
) -> dict[str, object]:
    """Verify the normative model files while isolating cache metadata."""

    root = _verified_directory(model_root, f"{role} model root")
    required_top_files = {
        row.relative_path for row in files if "/" not in row.relative_path
    }
    observed_top_files: set[str] = set()
    observed_top_directories: set[str] = set()
    try:
        for child in root.iterdir():
            if child.is_symlink():
                raise MmqaP1LocalRuntimePreflightError(
                    f"{role} model root contains a symlink"
                )
            if child.is_file():
                observed_top_files.add(child.name)
            elif child.is_dir():
                observed_top_directories.add(child.name)
            else:
                raise MmqaP1LocalRuntimePreflightError(
                    f"{role} model root contains a non-regular entry"
                )
    except OSError as exc:
        raise MmqaP1LocalRuntimePreflightError(
            f"{role} model root could not be enumerated"
        ) from exc
    if observed_top_files != required_top_files:
        raise MmqaP1LocalRuntimePreflightError(
            f"{role} top-level model file set drifted"
        )
    required_directories = {
        Path(row.relative_path).parts[0]
        for row in files
        if "/" in row.relative_path
    }
    if not required_directories.issubset(observed_top_directories) or not (
        observed_top_directories
        <= required_directories | allowed_top_level_directories
    ):
        raise MmqaP1LocalRuntimePreflightError(
            f"{role} top-level model directory set drifted"
        )

    observed: list[dict[str, object]] = []
    for expected in files:
        path = root / expected.relative_path
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise MmqaP1LocalRuntimePreflightError(
                f"{role} required model file is unavailable"
            ) from exc
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise MmqaP1LocalRuntimePreflightError(
                f"{role} required model file is not regular"
            )
        digest = _sha256_file(path)
        if metadata.st_size != expected.size_bytes or digest != expected.sha256:
            raise MmqaP1LocalRuntimePreflightError(
                f"{role} required model file drifted"
            )
        observed.append(
            {
                "path": expected.relative_path,
                "sha256": digest,
                "size": metadata.st_size,
            }
        )
    tree = _semantic_hash(observed)
    if tree != expected_tree_sha256:
        raise MmqaP1LocalRuntimePreflightError(
            f"{role} normative model tree drifted"
        )

    auxiliary_rows: list[dict[str, object]] = []
    for directory_name in sorted(observed_top_directories - required_directories):
        directory = root / directory_name
        for current, directories, names in os.walk(directory, followlinks=False):
            base = Path(current)
            for name in sorted(directories):
                child = base / name
                if child.is_symlink() or not child.is_dir():
                    raise MmqaP1LocalRuntimePreflightError(
                        f"{role} auxiliary directory contains a symlink"
                    )
            for name in sorted(names):
                child = base / name
                if child.is_symlink() or not child.is_file():
                    raise MmqaP1LocalRuntimePreflightError(
                        f"{role} auxiliary directory contains a non-file"
                    )
                auxiliary_rows.append(
                    {
                        "path": child.relative_to(root).as_posix(),
                        "sha256": _sha256_file(child),
                        "size": child.stat().st_size,
                    }
                )
    return {
        "auxiliary_file_count": len(auxiliary_rows),
        "auxiliary_tree_sha256": _semantic_hash(auxiliary_rows),
        "required_file_count": len(observed),
        "required_size_bytes": sum(int(row["size"]) for row in observed),
        "required_tree_sha256": tree,
    }


def _verify_minilm_asset(model_root: str | Path) -> dict[str, object]:
    return _verify_asset_tree(
        model_root,
        files=MINILM_REQUIRED_FILES,
        expected_tree_sha256=MINILM_REQUIRED_TREE_SHA256,
        allowed_top_level_directories=frozenset({".cache"}),
        role="MiniLM",
    )


def _verify_ce_asset(model_root: str | Path) -> dict[str, object]:
    return _verify_asset_tree(
        model_root,
        files=CE_REQUIRED_FILES,
        expected_tree_sha256=CE_REQUIRED_TREE_SHA256,
        allowed_top_level_directories=frozenset({".cache"}),
        role="cross-encoder",
    )


def _verify_typed_python(typed_python: str | Path) -> dict[str, object]:
    lexical = Path(typed_python).expanduser().absolute()
    try:
        resolved = lexical.resolve(strict=True)
        if not resolved.is_file():
            raise MmqaP1LocalRuntimePreflightError(
                "typed runtime Python is unavailable"
            )
    except OSError as exc:
        raise MmqaP1LocalRuntimePreflightError(
            "typed runtime Python is unavailable"
        ) from exc
    executable_hash = _sha256_file(resolved)
    if executable_hash != EXPECTED_TYPED_PYTHON_RESOLVED_SHA256:
        raise MmqaP1LocalRuntimePreflightError(
            "typed runtime Python binary identity drifted"
        )
    pyvenv_cfg = lexical.parent.parent / "pyvenv.cfg"
    if pyvenv_cfg.is_symlink() or not pyvenv_cfg.is_file():
        raise MmqaP1LocalRuntimePreflightError("typed runtime pyvenv.cfg is absent")
    cfg_hash = _sha256_file(pyvenv_cfg)
    if cfg_hash != EXPECTED_TYPED_PYVENV_CFG_SHA256:
        raise MmqaP1LocalRuntimePreflightError(
            "typed runtime pyvenv.cfg identity drifted"
        )
    return {
        "executable_resolved_file_sha256": executable_hash,
        "lexical_path_sha256": hashlib.sha256(
            os.fsencode(str(lexical))
        ).hexdigest(),
        "pyvenv_cfg_sha256": cfg_hash,
    }


def _safe_worker_environment(physical_gpu: str) -> dict[str, str]:
    if physical_gpu not in {"0", "1", "0,1"}:
        raise MmqaP1LocalRuntimePreflightError("worker GPU assignment drifted")
    project = Path(__file__).resolve(strict=True).parents[2]
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": physical_gpu,
        "HF_HUB_OFFLINE": "1",
        "HOME": "/nonexistent-mmqa-p1-preflight-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(project),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _execute_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        env=dict(environment),
        timeout=timeout,
    )


def _parse_canonical_worker_output(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MmqaP1LocalRuntimePreflightError(
            "typed worker output is not canonical JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or _canonical_json_bytes(value, newline=True) != raw
        or value.get("schema") != WORKER_SCHEMA
    ):
        raise MmqaP1LocalRuntimePreflightError(
            "typed worker output envelope drifted"
        )
    return value


def _invoke_worker(
    *,
    typed_python: Path,
    mode: str,
    physical_gpu: str,
    model_root: Path | None = None,
) -> dict[str, object]:
    if mode not in {"inventory", "minilm", "cross_encoder"}:
        raise MmqaP1LocalRuntimePreflightError("typed worker mode drifted")
    project = Path(__file__).resolve(strict=True).parents[2]
    command = [
        str(typed_python),
        "-m",
        "assumption_agent.benchmarks.mmqa_p1_local_runtime_preflight_v1",
        "__worker__",
        mode,
    ]
    if model_root is not None:
        command.extend(("--model", str(model_root)))
    try:
        completed = _execute_subprocess(
            command,
            cwd=project,
            environment=_safe_worker_environment(physical_gpu),
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MmqaP1LocalRuntimePreflightError(
            "typed worker failed before producing a receipt"
        ) from exc
    if completed.returncode != 0:
        raise MmqaP1LocalRuntimePreflightError(
            "typed worker failed before producing a receipt"
        )
    return _parse_canonical_worker_output(completed.stdout)


def _inventory_worker() -> dict[str, object]:
    import torch
    import transformers

    versions = {
        name: importlib.metadata.version(name)
        for name in EXPECTED_RUNTIME_VERSIONS
    }
    executable = Path(sys.executable).resolve(strict=True)
    pyvenv_cfg = Path(sys.prefix) / "pyvenv.cfg"
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "memory_total_bytes": int(properties.total_memory),
                "name": str(properties.name),
            }
        )
    return {
        "cuda_device_count": len(devices),
        "cuda_devices": devices,
        "executable_resolved_file_sha256": _sha256_file(executable),
        "mode": "inventory",
        "pyvenv_cfg_sha256": _sha256_file(pyvenv_cfg),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "runtime_versions": versions,
        "schema": WORKER_SCHEMA,
        "torch_cuda_version": str(torch.version.cuda),
        "transformers_import_version": transformers.__version__,
    }


def _load_minilm(model_root: Path):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    model = SentenceTransformer(
        str(model_root),
        device="cuda:0",
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
    model.max_seq_length = MAXIMUM_SEQUENCE_LENGTH
    model.float()
    model.eval()
    parameters = tuple(model.parameters())
    if (
        not parameters
        or model.training
        or model.max_seq_length != MAXIMUM_SEQUENCE_LENGTH
        or any(parameter.device.type != "cuda" for parameter in parameters)
        or any(parameter.dtype != torch.float32 for parameter in parameters)
    ):
        raise MmqaP1LocalRuntimePreflightError(
            "MiniLM worker model contract drifted"
        )
    return model


def _validated_embedding_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value)
    expected_shape = (1 + len(SYNTHETIC_NODES), EMBEDDING_DIMENSION)
    if matrix.dtype != np.dtype(np.float32) or matrix.shape != expected_shape:
        raise MmqaP1LocalRuntimePreflightError(
            "MiniLM worker embedding shape or dtype drifted"
        )
    if not np.isfinite(matrix).all():
        raise MmqaP1LocalRuntimePreflightError(
            "MiniLM worker produced nonfinite coordinates"
        )
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > ROW_NORM_ATOL:
        raise MmqaP1LocalRuntimePreflightError(
            "MiniLM worker normalization drifted"
        )
    if not any(not np.array_equal(matrix[0], row) for row in matrix[1:]):
        raise MmqaP1LocalRuntimePreflightError(
            "MiniLM worker embeddings collapsed"
        )
    return matrix


def _minilm_worker(model_root: Path) -> dict[str, object]:
    model = _load_minilm(model_root)
    texts = [SYNTHETIC_QUERY, *(row.text for row in SYNTHETIC_NODES)]

    def encode() -> np.ndarray:
        values = model.encode(
            texts,
            batch_size=MINILM_BATCH_SIZE,
            convert_to_numpy=True,
            convert_to_tensor=False,
            device="cuda:0",
            normalize_embeddings=True,
            precision="float32",
            show_progress_bar=False,
        )
        return _validated_embedding_matrix(values)

    first = encode()
    second = encode()
    first_bytes = first.astype("<f4", copy=False).tobytes(order="C")
    if not np.array_equal(first, second) or first_bytes != second.astype(
        "<f4", copy=False
    ).tobytes(order="C"):
        raise MmqaP1LocalRuntimePreflightError(
            "MiniLM worker repeat was not exact"
        )
    query = first[0].astype(np.float64)
    coordinates = []
    for row in first[1:]:
        similarity = math.fsum(
            float(left) * float(right) for left, right in zip(query, row)
        )
        coordinates.append(min(1.0, max(0.0, (similarity + 1.0) / 2.0)))
    return {
        "all_finite": True,
        "all_rows_l2_normalized": True,
        "coordinates_float64_hex": [value.hex() for value in coordinates],
        "fixture_sha256": PUBLIC_SYNTHETIC_FIXTURE_SHA256,
        "matrix_little_endian_float32_sha256": hashlib.sha256(
            first_bytes
        ).hexdigest(),
        "matrix_shape": list(first.shape),
        "mode": "minilm",
        "process_concurrency": MODEL_WORKER_CONCURRENCY,
        "repeat_count": 2,
        "repeat_exact": True,
        "schema": WORKER_SCHEMA,
    }


def _load_cross_encoder(model_root: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).eval().to("cuda:0")
    if (
        model.__class__.__name__ != "BertForSequenceClassification"
        or int(model.num_labels) != 1
        or model.training
        or any(parameter.device.type != "cuda" for parameter in model.parameters())
        or any(parameter.dtype != torch.float32 for parameter in model.parameters())
    ):
        raise MmqaP1LocalRuntimePreflightError(
            "cross-encoder worker model contract drifted"
        )
    return tokenizer, model


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _cross_encoder_worker(model_root: Path) -> dict[str, object]:
    import torch

    tokenizer, model = _load_cross_encoder(model_root)
    queries = [SYNTHETIC_QUERY] * len(SYNTHETIC_NODES)
    passages = [row.text for row in SYNTHETIC_NODES]

    def infer() -> np.ndarray:
        encoded = tokenizer(
            queries,
            passages,
            max_length=CE_MAXIMUM_SEQUENCE_LENGTH,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.detach().cpu().numpy()
        matrix = np.asarray(logits)
        if matrix.dtype != np.dtype(np.float32) or matrix.shape != (
            len(SYNTHETIC_NODES),
            1,
        ):
            raise MmqaP1LocalRuntimePreflightError(
                "cross-encoder worker logit shape or dtype drifted"
            )
        if not np.isfinite(matrix).all():
            raise MmqaP1LocalRuntimePreflightError(
                "cross-encoder worker produced nonfinite logits"
            )
        return matrix

    first = infer()
    second = infer()
    first_bytes = first.astype("<f4", copy=False).tobytes(order="C")
    if not np.array_equal(first, second) or first_bytes != second.astype(
        "<f4", copy=False
    ).tobytes(order="C"):
        raise MmqaP1LocalRuntimePreflightError(
            "cross-encoder worker repeat was not exact"
        )
    coordinates = [_stable_sigmoid(float(value)) for value in first[:, 0]]
    return {
        "all_finite": True,
        "coordinates_float64_hex": [value.hex() for value in coordinates],
        "fixture_sha256": PUBLIC_SYNTHETIC_FIXTURE_SHA256,
        "logit_little_endian_float32_sha256": hashlib.sha256(
            first_bytes
        ).hexdigest(),
        "logit_shape": list(first.shape),
        "mode": "cross_encoder",
        "process_concurrency": MODEL_WORKER_CONCURRENCY,
        "repeat_count": 2,
        "repeat_exact": True,
        "schema": WORKER_SCHEMA,
    }


def _validate_inventory_worker(
    payload: Mapping[str, object], typed_binding: Mapping[str, object]
) -> str:
    expected_keys = {
        "cuda_device_count",
        "cuda_devices",
        "executable_resolved_file_sha256",
        "mode",
        "pyvenv_cfg_sha256",
        "python_version",
        "runtime_versions",
        "schema",
        "torch_cuda_version",
        "transformers_import_version",
    }
    if set(payload) != expected_keys or payload.get("mode") != "inventory":
        raise MmqaP1LocalRuntimePreflightError(
            "typed runtime inventory schema drifted"
        )
    if (
        payload.get("python_version") != EXPECTED_PYTHON_VERSION
        or payload.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS
        or payload.get("torch_cuda_version") != EXPECTED_TORCH_CUDA_VERSION
        or payload.get("transformers_import_version")
        != EXPECTED_RUNTIME_VERSIONS["transformers"]
        or payload.get("executable_resolved_file_sha256")
        != typed_binding.get("executable_resolved_file_sha256")
        or payload.get("pyvenv_cfg_sha256")
        != typed_binding.get("pyvenv_cfg_sha256")
    ):
        raise MmqaP1LocalRuntimePreflightError(
            "typed runtime identity or versions drifted"
        )
    devices = payload.get("cuda_devices")
    if (
        payload.get("cuda_device_count") != 2
        or not isinstance(devices, list)
        or len(devices) != 2
    ):
        raise MmqaP1LocalRuntimePreflightError(
            "typed runtime CUDA visibility drifted"
        )
    for position, row in enumerate(devices):
        if (
            not isinstance(row, dict)
            or set(row) != {"index", "memory_total_bytes", "name"}
            or row.get("index") != position
            or row.get("name") != EXPECTED_GPU_ROWS[position]["name"]
            or not isinstance(row.get("memory_total_bytes"), int)
            or int(row["memory_total_bytes"]) < 8_000_000_000
        ):
            raise MmqaP1LocalRuntimePreflightError(
                "typed runtime CUDA device identity drifted"
            )
    return _semantic_hash(payload)


def _parse_coordinate_hex_rows(value: object, role: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(SYNTHETIC_NODES):
        raise MmqaP1LocalRuntimePreflightError(
            f"{role} private coordinate width drifted"
        )
    output: list[float] = []
    for raw in value:
        if not isinstance(raw, str):
            raise MmqaP1LocalRuntimePreflightError(
                f"{role} private coordinate encoding drifted"
            )
        try:
            coordinate = float.fromhex(raw)
        except ValueError as exc:
            raise MmqaP1LocalRuntimePreflightError(
                f"{role} private coordinate encoding drifted"
            ) from exc
        if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
            raise MmqaP1LocalRuntimePreflightError(
                f"{role} private coordinate escaped [0, 1]"
            )
        output.append(0.0 if coordinate == 0.0 else coordinate)
    return tuple(output)


def _validate_model_worker(
    payload: Mapping[str, object], *, role: str
) -> tuple[tuple[float, ...], dict[str, object]]:
    if role == "minilm":
        expected_keys = {
            "all_finite",
            "all_rows_l2_normalized",
            "coordinates_float64_hex",
            "fixture_sha256",
            "matrix_little_endian_float32_sha256",
            "matrix_shape",
            "mode",
            "process_concurrency",
            "repeat_count",
            "repeat_exact",
            "schema",
        }
        hash_key = "matrix_little_endian_float32_sha256"
        shape_key = "matrix_shape"
        expected_shape = [1 + len(SYNTHETIC_NODES), EMBEDDING_DIMENSION]
    elif role == "cross_encoder":
        expected_keys = {
            "all_finite",
            "coordinates_float64_hex",
            "fixture_sha256",
            "logit_little_endian_float32_sha256",
            "logit_shape",
            "mode",
            "process_concurrency",
            "repeat_count",
            "repeat_exact",
            "schema",
        }
        hash_key = "logit_little_endian_float32_sha256"
        shape_key = "logit_shape"
        expected_shape = [len(SYNTHETIC_NODES), 1]
    else:  # pragma: no cover - internal contract
        raise MmqaP1LocalRuntimePreflightError("model worker role drifted")
    if set(payload) != expected_keys or payload.get("mode") != role:
        raise MmqaP1LocalRuntimePreflightError(f"{role} worker schema drifted")
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("fixture_sha256") != PUBLIC_SYNTHETIC_FIXTURE_SHA256
        or payload.get(shape_key) != expected_shape
        or payload.get("process_concurrency") != MODEL_WORKER_CONCURRENCY
        or payload.get("repeat_count") != 2
        or payload.get("repeat_exact") is not True
        or payload.get("all_finite") is not True
        or not isinstance(payload.get(hash_key), str)
        or _HEX64.fullmatch(str(payload[hash_key])) is None
    ):
        raise MmqaP1LocalRuntimePreflightError(
            f"{role} worker validation failed"
        )
    if role == "minilm" and payload.get("all_rows_l2_normalized") is not True:
        raise MmqaP1LocalRuntimePreflightError(
            "MiniLM worker normalization validation failed"
        )
    coordinates = _parse_coordinate_hex_rows(
        payload.get("coordinates_float64_hex"), role
    )
    public_receipt = {
        "all_finite": True,
        "fixture_sha256": PUBLIC_SYNTHETIC_FIXTURE_SHA256,
        "output_sha256": payload[hash_key],
        "output_shape": expected_shape,
        "process_concurrency": MODEL_WORKER_CONCURRENCY,
        "repeat_count": 2,
        "repeat_exact": True,
    }
    if role == "minilm":
        public_receipt["all_rows_l2_normalized"] = True
    return coordinates, public_receipt


def _parse_gpu_probe_output(raw: bytes) -> tuple[dict[str, object], ...]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise MmqaP1LocalRuntimePreflightError(
            "nvidia-smi GPU output is not ASCII"
        ) from exc
    rows: list[dict[str, object]] = []
    for line in lines:
        parts = tuple(part.strip() for part in line.split(","))
        if len(parts) != 4:
            raise MmqaP1LocalRuntimePreflightError(
                "nvidia-smi GPU output shape drifted"
            )
        try:
            index = int(parts[0])
            memory = int(parts[3])
        except ValueError as exc:
            raise MmqaP1LocalRuntimePreflightError(
                "nvidia-smi GPU numeric output drifted"
            ) from exc
        rows.append(
            {
                "index": index,
                "memory_total_mib": memory,
                "name": parts[2],
                "uuid": parts[1],
            }
        )
    result = tuple(sorted(rows, key=lambda row: int(row["index"])))
    if result != EXPECTED_GPU_ROWS:
        raise MmqaP1LocalRuntimePreflightError(
            "311linux physical GPU UUID binding drifted"
        )
    return result


def _probe_gpu_rows(nvidia_smi: str | Path) -> tuple[dict[str, object], ...]:
    executable = Path(nvidia_smi).expanduser().absolute()
    if executable.is_symlink() or not executable.is_file():
        raise MmqaP1LocalRuntimePreflightError("nvidia-smi is unavailable")
    command = [
        str(executable),
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = _execute_subprocess(
            command,
            cwd=Path("/tmp"),
            environment={
                "HOME": "/nonexistent-mmqa-p1-preflight-home",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MmqaP1LocalRuntimePreflightError(
            "nvidia-smi GPU probe failed"
        ) from exc
    if completed.returncode != 0:
        raise MmqaP1LocalRuntimePreflightError("nvidia-smi GPU probe failed")
    return _parse_gpu_probe_output(completed.stdout)


def _typed_edges() -> tuple[core.TypedLinkEdge, ...]:
    edges = []
    for row_ordinal, text_ordinal in _UNDIRECTED_LINKS:
        edges.append(
            core.TypedLinkEdge(row_ordinal, text_ordinal, core.ROW_TO_TEXT)
        )
        edges.append(
            core.TypedLinkEdge(text_ordinal, row_ordinal, core.TEXT_TO_ROW)
        )
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.source_ordinal,
                edge.target_ordinal,
                core.EDGE_TYPES.index(edge.edge_type),
            ),
        )
    )


def _build_core_receipt(
    minilm_coordinates: Sequence[float],
    ce_coordinates: Sequence[float],
) -> dict[str, object]:
    if len(minilm_coordinates) != len(SYNTHETIC_NODES) or len(
        ce_coordinates
    ) != len(SYNTHETIC_NODES):
        raise MmqaP1LocalRuntimePreflightError(
            "synthetic core coordinate width drifted"
        )
    nodes = tuple(
        core.ProofNode(
            ordinal=index,
            node_type=spec.node_type,
            minilm_similarity=float(minilm_coordinates[index]),
            cross_encoder_relevance=float(ce_coordinates[index]),
            entity_anchor=spec.entity_anchor,
            relation_anchor=spec.relation_anchor,
            numeric_or_temporal_anchor=spec.numeric_or_temporal_anchor,
        )
        for index, spec in enumerate(SYNTHETIC_NODES)
    )
    graph = core.ProofGraph(nodes, _typed_edges())
    closures = []
    bundle_registries = []
    e0_rows = []
    training_items = []
    for anchor, gold_ordinals in _TRAINING_ANCHOR_AND_GOLD:
        closure = core.build_query_local_closure(graph, [anchor])
        bundles = core.enumerate_connected_bundles(closure)
        gold = core.ProofBundle(tuple(gold_ordinals))
        if gold not in bundles:
            raise MmqaP1LocalRuntimePreflightError(
                "synthetic gold bundle is absent from the closure"
            )
        e0 = core.select_e0_bundle(closure.graph, bundles)
        training_items.append(
            core.make_e5_training_item(closure.graph, bundles, [gold])
        )
        closures.append(closure)
        bundle_registries.append(bundles)
        e0_rows.append(
            {
                "energy_float64_hex": core.e0_proof_energy(
                    closure.graph, e0
                ).hex(),
                "selection": list(e0.node_ordinals),
            }
        )
    model = core.fit_e5(tuple(training_items))
    e5_rows = [
        list(core.select_e5_bundle(model, closure.graph, bundles).node_ordinals)
        for closure, bundles in zip(closures, bundle_registries, strict=True)
    ]
    private_graph_payload = {
        "edges": [
            [edge.source_ordinal, edge.target_ordinal, edge.edge_type]
            for edge in graph.edges
        ],
        "nodes": [
            {
                "anchors": [
                    node.entity_anchor,
                    node.relation_anchor,
                    node.numeric_or_temporal_anchor,
                ],
                "ce_float64_hex": node.cross_encoder_relevance.hex(),
                "minilm_float64_hex": node.minilm_similarity.hex(),
                "node_type": node.node_type,
                "ordinal": node.ordinal,
            }
            for node in graph.nodes
        ],
    }
    private_closure_payload = [
        {
            "anchors": list(closure.anchor_ordinals),
            "bundles": [list(bundle.node_ordinals) for bundle in bundles],
            "edges": [
                [edge.source_ordinal, edge.target_ordinal, edge.edge_type]
                for edge in closure.graph.edges
            ],
            "nodes": [node.ordinal for node in closure.graph.nodes],
        }
        for closure, bundles in zip(closures, bundle_registries, strict=True)
    ]
    core_output_binding = {
        "closure_registry_sha256": _semantic_hash(private_closure_payload),
        "e0_selection_sha256": _semantic_hash(e0_rows),
        "e5_model_sha256": _semantic_hash(model.payload()),
        "e5_selection_sha256": _semantic_hash(e5_rows),
        "graph_sha256": _semantic_hash(private_graph_payload),
    }
    return {
        "bundle_registry_shapes": [
            [len(bundles), len(core.FEATURE_ORDER)]
            for bundles in bundle_registries
        ],
        "closure_shapes": [
            [len(closure.graph.nodes), len(closure.graph.edges)]
            for closure in closures
        ],
        "core_output_sha256": _semantic_hash(core_output_binding),
        "e0_selection_sha256": core_output_binding["e0_selection_sha256"],
        "e5_model_sha256": core_output_binding["e5_model_sha256"],
        "e5_selection_sha256": core_output_binding["e5_selection_sha256"],
        "feature_shape": [len(core.FEATURE_ORDER)],
        "proof_graph_shape": [len(graph.nodes), len(graph.edges)],
        "training_shape": [
            len(training_items),
            sum(len(row.bundles) for row in training_items),
            len(core.FEATURE_ORDER),
        ],
    }


def production_address_family_isolation_probe(
    *,
    socket_factory: Callable[[int, int], Any] = socket.socket,
) -> dict[str, object]:
    denied: dict[str, int] = {}
    for family, label in (
        (socket.AF_INET, "AF_INET"),
        (socket.AF_INET6, "AF_INET6"),
    ):
        candidate: Any | None = None
        try:
            candidate = socket_factory(family, socket.SOCK_STREAM)
        except OSError as exc:
            if exc.errno != errno.EAFNOSUPPORT:
                raise MmqaP1LocalRuntimePreflightError(
                    "local preflight address-family probe failed closed"
                ) from exc
            denied[label] = exc.errno
        else:
            candidate.close()
            raise MmqaP1LocalRuntimePreflightError(
                "local preflight network isolation is absent"
            )
        finally:
            if candidate is not None:
                candidate.close()
    if denied != {
        "AF_INET": errno.EAFNOSUPPORT,
        "AF_INET6": errno.EAFNOSUPPORT,
    }:
        raise MmqaP1LocalRuntimePreflightError(
            "local preflight address-family probe is incomplete"
        )
    return {
        "AF_INET6_socket_creation_errno": "EAFNOSUPPORT",
        "AF_INET_socket_creation_errno": "EAFNOSUPPORT",
        "address_family_isolation_contract": (
            ADDRESS_FAMILY_ISOLATION_CONTRACT
        ),
        "denied_family_count": 2,
        "probe_count": 2,
        "status": "AF_INET_and_AF_INET6_socket_creation_denied",
    }


def run_preflight(
    *,
    typed_python: str | Path,
    minilm_model: str | Path,
    cross_encoder_model: str | Path,
    nvidia_smi: str | Path = "/usr/bin/nvidia-smi",
) -> dict[str, object]:
    """Run the bounded source-free preflight and return a sanitized receipt."""

    _validate_static_contract()
    address_family_probe = production_address_family_isolation_probe()
    typed_path = Path(typed_python).expanduser().absolute()
    minilm_path = Path(minilm_model).expanduser().absolute()
    ce_path = Path(cross_encoder_model).expanduser().absolute()
    typed_binding = _verify_typed_python(typed_path)
    minilm_asset = _verify_minilm_asset(minilm_path)
    ce_asset = _verify_ce_asset(ce_path)
    gpu_rows = _probe_gpu_rows(nvidia_smi)

    inventory = _invoke_worker(
        typed_python=typed_path,
        mode="inventory",
        physical_gpu="0,1",
    )
    runtime_identity_hash = _validate_inventory_worker(inventory, typed_binding)
    minilm_private = _invoke_worker(
        typed_python=typed_path,
        mode="minilm",
        physical_gpu="0",
        model_root=minilm_path,
    )
    ce_private = _invoke_worker(
        typed_python=typed_path,
        mode="cross_encoder",
        physical_gpu="1",
        model_root=ce_path,
    )
    minilm_coordinates, minilm_receipt = _validate_model_worker(
        minilm_private, role="minilm"
    )
    ce_coordinates, ce_receipt = _validate_model_worker(
        ce_private, role="cross_encoder"
    )
    core_receipt = _build_core_receipt(minilm_coordinates, ce_coordinates)

    body: dict[str, object] = {
        "address_family_isolation_probe": address_family_probe,
        "address_family_isolation_probe_sha256": _semantic_hash(
            address_family_probe
        ),
        "asset_bindings": {
            "cross_encoder_auxiliary_tree_sha256": ce_asset[
                "auxiliary_tree_sha256"
            ],
            "cross_encoder_required_tree_sha256": ce_asset[
                "required_tree_sha256"
            ],
            "minilm_auxiliary_tree_sha256": minilm_asset[
                "auxiliary_tree_sha256"
            ],
            "minilm_required_tree_sha256": minilm_asset[
                "required_tree_sha256"
            ],
        },
        "claim_boundary": {
            "api_or_provider_call_count": 0,
            "formal_HippoRAG_call_count": 0,
            "formal_MMQA_source_or_row_access_count": 0,
            "label_or_score_access_count": 0,
            "online_evaluator_call_count": 0,
            "retry_replay_or_resample_count": 0,
            "synthetic_fixture_count": 1,
        },
        "concurrency": {
            "cross_encoder_physical_gpu_1_process_cap": 1,
            "minilm_physical_gpu_0_process_cap": 1,
            "model_process_co_residency": False,
            "typed_gpu_process_cap": TOTAL_TYPED_GPU_CONCURRENCY,
        },
        "core_action_preflight": core_receipt,
        "gpu_binding": {
            "physical_gpu_count": len(gpu_rows),
            "physical_gpu_uuid_registry_sha256": _semantic_hash(
                [row["uuid"] for row in gpu_rows]
            ),
        },
        "model_canaries": {
            "cross_encoder": ce_receipt,
            "minilm": minilm_receipt,
        },
        "public_synthetic_fixture_sha256": PUBLIC_SYNTHETIC_FIXTURE_SHA256,
        "runtime_binding": {
            "typed_runtime_identity_sha256": runtime_identity_hash,
            "typed_runtime_lexical_path_sha256": typed_binding[
                "lexical_path_sha256"
            ],
        },
        "schema": RECEIPT_SCHEMA,
        "status": "passed_public_synthetic_non_scoring_runtime_action_preflight",
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _write_exclusive(path: str | Path, payload: Mapping[str, object]) -> str:
    destination = Path(path).expanduser().absolute()
    raw = _canonical_json_bytes(payload, newline=True)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            destination,
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
        raise MmqaP1LocalRuntimePreflightError(
            "preflight output already exists or cannot be created"
        ) from exc
    finally:
        if "descriptor" in locals() and descriptor >= 0:
            os.close(descriptor)
    if (
        destination.is_symlink()
        or destination.read_bytes() != raw
        or stat.S_IMODE(destination.stat().st_mode) != 0o600
    ):
        raise MmqaP1LocalRuntimePreflightError(
            "preflight output reopen or mode drifted"
        )
    return hashlib.sha256(raw).hexdigest()


def _controller_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-python", required=True, type=Path)
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument("--cross-encoder-model", required=True, type=Path)
    parser.add_argument(
        "--nvidia-smi", type=Path, default=Path("/usr/bin/nvidia-smi")
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "mode", choices=("inventory", "minilm", "cross_encoder")
    )
    parser.add_argument("--model", type=Path)
    return parser


def _worker_main(argv: Sequence[str]) -> int:
    arguments = _worker_parser().parse_args(argv)
    if arguments.mode == "inventory":
        if arguments.model is not None:
            raise MmqaP1LocalRuntimePreflightError(
                "inventory worker cannot receive a model path"
            )
        payload = _inventory_worker()
    else:
        if arguments.model is None:
            raise MmqaP1LocalRuntimePreflightError(
                "model worker requires an exact local model path"
            )
        root = _verified_directory(arguments.model, "worker model root")
        payload = (
            _minilm_worker(root)
            if arguments.mode == "minilm"
            else _cross_encoder_worker(root)
        )
    sys.stdout.buffer.write(_canonical_json_bytes(payload, newline=True))
    sys.stdout.buffer.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "__worker__":
        return _worker_main(values[1:])
    arguments = _controller_parser().parse_args(values)
    receipt = run_preflight(
        typed_python=arguments.typed_python,
        minilm_model=arguments.minilm_model,
        cross_encoder_model=arguments.cross_encoder_model,
        nvidia_smi=arguments.nvidia_smi,
    )
    _write_exclusive(arguments.output, receipt)
    sys.stdout.buffer.write(_canonical_json_bytes(receipt, newline=True))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CE_REQUIRED_FILES",
    "CE_REQUIRED_TREE_SHA256",
    "EXPECTED_GPU_ROWS",
    "EXPECTED_RUNTIME_VERSIONS",
    "MINILM_REQUIRED_FILES",
    "MINILM_REQUIRED_TREE_SHA256",
    "MmqaP1LocalRuntimePreflightError",
    "PUBLIC_SYNTHETIC_FIXTURE_SHA256",
    "RECEIPT_SCHEMA",
    "STUDY_DESIGN_SELF_SHA256",
    "VERSION",
    "run_preflight",
]

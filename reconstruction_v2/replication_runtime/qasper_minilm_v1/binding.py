"""Immutable, offline MiniLM embedding runtime for the QASPER study.

This module is deliberately independent of every QASPER source row, split,
archive, identifier, answer, and evidence annotation.  Its only persisted
input is the public, content-addressed MiniLM snapshot.  The runtime exposes
normalized embeddings and deterministic integer cosine scores so later study
code never has to depend on platform-specific float serialization.
"""

from __future__ import annotations

import hashlib
from importlib import import_module, metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ASSET_VERSION = "qasper_minilm_runtime_asset_v1"
ASSET_RELATIVE_PATH = Path("manifests/qasper_minilm_runtime_asset_v1.json")
# Filled only from the committed public manifest.  Neither hash depends on a
# QASPER row or archive.
ASSET_SELF_SHA256 = "921d9b1945581130e03c53f448092c3de3b30714431c6cac9b3b32c2ec10abad"
ASSET_FILE_SHA256 = "62b85c7752f2e46932fb9fb13ae2f3aac9eb750a33c8f07102739040feb6cc75"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_ARCHITECTURE = "BertModel"
MODEL_TREE_SHA256 = "1514beb65d2d3a2824a93f133a300cc60d5b437ccd6ea1e622eb4cd9881dcfdb"
WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
EMBEDDING_DIMENSION = 384
MAXIMUM_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
QUANTIZATION_SCALE = 1_000_000
MAXIMUM_TEXTS_PER_CALL = 16_384
MAXIMUM_TEXT_CHARACTERS = 2_000_000
MAXIMUM_TEXT_UTF8_BYTES = 8_000_000

CANARY_SENTENCE_COUNT = 256
CANARY_TEXT_VECTOR_SHA256 = (
    "c122a1e09d2f84ad00a4c0b30abb979e13facdb8c1a5b3b15cb952b51b173249"
)
CANARY_QUANTIZED_EMBEDDING_SHA256 = (
    "f24c3299f365c675cf30a960c23acc8f97e9cd0b4434b9147c80ae95db56ee1b"
)
CANARY_FLOAT32_BYTES_SHA256 = (
    "e76f373bfc7c2b4f16b12d2841dc8d2ec0e0e93f8fe360c04a79062d628c5746"
)

EXPECTED_RUNTIME_VERSIONS = {
    "huggingface_hub": "1.11.0",
    "numpy": "2.2.6",
    "python": "3.10.12",
    "safetensors": "0.7.0",
    "sentence_transformers": "5.5.1",
    "tokenizers": "0.22.2",
    "torch": "2.8.0+cu128",
    "transformers": "5.10.1",
}
EXPECTED_EXECUTION = {
    "backend": "torch",
    "batch_size": BATCH_SIZE,
    "device": "cpu",
    "dtype": "float32",
    "environment": {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    },
    "eval_mode": True,
    "local_files_only": True,
    "maximum_sequence_length": MAXIMUM_SEQUENCE_LENGTH,
    "network_calls": 0,
    "normalize_embeddings": True,
    "output_precision": "float32",
    "torch_deterministic_algorithms": True,
    "torch_manual_seed": 0,
    "torch_num_threads": 1,
    "trust_remote_code": False,
    "use_safetensors": True,
}
EXPECTED_QUANTIZATION = {
    "embedding_component_formula": "int(round(float(component) * 1000000))",
    "embedding_vector_hash_contract": "sha256(canonical_JSON_nested_integer_matrix)",
    "quantization_scale": QUANTIZATION_SCALE,
    "rounding": "Python_round_ties_to_even",
    "similarity_formula": "int(round(math.fsum(float(query[i]) * float(paragraph[i]) for i in range(384)) * 1000000))",
    "similarity_semantics": "cosine_of_L2_normalized_float32_embeddings",
}

_PACKAGE_TO_MODULE = {
    "huggingface_hub": ("huggingface-hub", "huggingface_hub"),
    "numpy": ("numpy", "numpy"),
    "safetensors": ("safetensors", "safetensors"),
    "sentence_transformers": ("sentence-transformers", "sentence_transformers"),
    "tokenizers": ("tokenizers", "tokenizers"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
}


class QasperMiniLMError(RuntimeError):
    """Raised when the frozen offline embedding contract cannot be proven."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QasperMiniLMError("value is not canonical-JSON representable") from exc


def _canonical_hash(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _reject_symlink_components(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise QasperMiniLMError(f"{field} contains a symlink component")
    return absolute


def _load_asset_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = _reject_symlink_components(Path(path), "asset manifest path")
    if not manifest_path.is_file() or manifest_path.stat().st_size > 256 * 1024:
        raise QasperMiniLMError("asset manifest is unavailable or oversized")
    raw = manifest_path.read_bytes()
    if _sha256_bytes(raw) != ASSET_FILE_SHA256:
        raise QasperMiniLMError("committed asset manifest file drifted")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QasperMiniLMError("asset manifest is invalid") from exc
    if not isinstance(value, dict) or value.get("asset_version") != ASSET_VERSION:
        raise QasperMiniLMError("asset manifest version mismatch")
    body = dict(value)
    declared = body.pop("asset_sha256", None)
    if declared != ASSET_SELF_SHA256 or _canonical_hash(body) != declared:
        raise QasperMiniLMError("asset manifest self-hash mismatch")
    return manifest_path, value


def _verify_manifest_contract(asset: Mapping[str, Any]) -> None:
    model = asset.get("model")
    local = asset.get("local_binding")
    canary = asset.get("deterministic_canary")
    scope = asset.get("scope")
    if not all(isinstance(value, Mapping) for value in (model, local, canary, scope)):
        raise QasperMiniLMError("asset manifest binding is incomplete")
    assert isinstance(model, Mapping)
    assert isinstance(local, Mapping)
    assert isinstance(canary, Mapping)
    assert isinstance(scope, Mapping)
    if (
        model.get("model_id") != MODEL_ID
        or model.get("snapshot_revision") != MODEL_REVISION
        or model.get("architecture") != MODEL_ARCHITECTURE
        or model.get("embedding_dimension") != EMBEDDING_DIMENSION
        or model.get("weight_serialization") != "safetensors"
        or model.get("weights_sha256") != WEIGHTS_SHA256
        or asset.get("license") != "Apache-2.0"
        or asset.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS
        or asset.get("execution") != EXPECTED_EXECUTION
        or asset.get("quantization") != EXPECTED_QUANTIZATION
        or local.get("snapshot_tree_sha256") != MODEL_TREE_SHA256
        or canary.get("synthetic_sentence_count") != CANARY_SENTENCE_COUNT
        or canary.get("text_vector_sha256") != CANARY_TEXT_VECTOR_SHA256
        or canary.get("quantized_embedding_matrix_sha256")
        != CANARY_QUANTIZED_EMBEDDING_SHA256
        or canary.get("little_endian_float32_c_order_sha256")
        != CANARY_FLOAT32_BYTES_SHA256
        or canary.get("repeat_count") != 2
        or canary.get("repeat_exact") is not True
        or canary.get("qasper_rows_or_archives_accessed_by_canary") is not False
        or scope.get("qasper_rows_accessed_by_asset_freeze") is not False
        or scope.get("qasper_archives_accessed_by_asset_freeze") is not False
        or scope.get("asset_freeze_only") is not True
    ):
        raise QasperMiniLMError("asset manifest normative contract drifted")


def _verify_model_tree(asset: Mapping[str, Any], model_root: str | Path) -> Path:
    root = _reject_symlink_components(Path(model_root), "model root")
    if not root.is_dir():
        raise QasperMiniLMError("model root is unavailable")
    live_files: list[str] = []
    live_directories: list[str] = []
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in directories:
            path = base / name
            if path.is_symlink():
                raise QasperMiniLMError("model tree contains a symlink")
            live_directories.append(path.relative_to(root).as_posix())
        for name in files:
            path = base / name
            if path.is_symlink():
                raise QasperMiniLMError("model tree contains a symlink")
            live_files.append(path.relative_to(root).as_posix())

    local = asset.get("local_binding")
    if not isinstance(local, Mapping):
        raise QasperMiniLMError("local snapshot binding is missing")
    rows = local.get("snapshot_files")
    if not isinstance(rows, list) or len(rows) != local.get("snapshot_file_count"):
        raise QasperMiniLMError("snapshot file manifest is malformed")
    expected_directories = local.get("snapshot_directories")
    if not isinstance(expected_directories, list) or any(
        not isinstance(value, str) or not value for value in expected_directories
    ):
        raise QasperMiniLMError("snapshot directory manifest is malformed")
    if sorted(live_directories) != sorted(expected_directories):
        raise QasperMiniLMError("snapshot directory set drifted")

    verified_rows: list[dict[str, object]] = []
    expected_paths: list[str] = []
    total_size = 0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size"}:
            raise QasperMiniLMError("snapshot file row is malformed")
        relative_text = row.get("path")
        if not isinstance(relative_text, str) or not relative_text:
            raise QasperMiniLMError("snapshot file path is malformed")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise QasperMiniLMError("snapshot file path is unsafe")
        size = row.get("size")
        digest = row.get("sha256")
        path = root / relative
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise QasperMiniLMError("snapshot file content drifted")
        expected_paths.append(relative_text)
        verified_rows.append({"path": relative_text, "sha256": digest, "size": size})
        total_size += size
    if sorted(live_files) != sorted(expected_paths):
        raise QasperMiniLMError("snapshot file set drifted")
    if total_size != local.get("snapshot_size_bytes"):
        raise QasperMiniLMError("snapshot total size drifted")
    if _canonical_hash(verified_rows) != local.get("snapshot_tree_sha256"):
        raise QasperMiniLMError("snapshot tree hash drifted")
    weight_rows = [row for row in verified_rows if row["path"] == "model.safetensors"]
    if len(weight_rows) != 1 or weight_rows[0]["sha256"] != WEIGHTS_SHA256:
        raise QasperMiniLMError("safetensors weight binding drifted")
    forbidden_weights = {
        path for path in expected_paths if path.endswith((".bin", ".pt", ".pth", ".ckpt"))
    }
    if forbidden_weights:
        raise QasperMiniLMError("snapshot contains a pickle-capable weight file")
    return root


def _verify_package_versions(asset: Mapping[str, Any]) -> dict[str, str]:
    declared = asset.get("runtime_versions")
    if declared != EXPECTED_RUNTIME_VERSIONS:
        raise QasperMiniLMError("declared runtime package versions drifted")
    actual: dict[str, str] = {"python": ".".join(map(str, sys.version_info[:3]))}
    for key, (distribution, module_name) in _PACKAGE_TO_MODULE.items():
        try:
            distribution_version = metadata.version(distribution)
            module_version = str(getattr(import_module(module_name), "__version__"))
        except (ImportError, AttributeError, metadata.PackageNotFoundError) as exc:
            raise QasperMiniLMError(
                f"required runtime package is missing: {distribution}"
            ) from exc
        actual[key] = module_version
        # The torch wheel metadata omits the imported CUDA build tag.
        if key != "torch" and distribution_version != module_version:
            raise QasperMiniLMError("runtime module and distribution versions disagree")
    if actual != declared:
        raise QasperMiniLMError("installed runtime package versions drifted")
    return actual


def verify_runtime_binding(
    *, asset_manifest_path: str | Path, model_root: str | Path
) -> dict[str, object]:
    """Recompute the public manifest, complete snapshot, and runtime binding."""

    manifest_path, asset = _load_asset_manifest(asset_manifest_path)
    _verify_manifest_contract(asset)
    verified_root = _verify_model_tree(asset, model_root)
    versions = _verify_package_versions(asset)
    return {
        "asset_file_sha256": ASSET_FILE_SHA256,
        "asset_manifest_path": str(manifest_path),
        "asset_sha256": ASSET_SELF_SHA256,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "maximum_sequence_length": MAXIMUM_SEQUENCE_LENGTH,
        "model_root": str(verified_root),
        "model_tree_sha256": MODEL_TREE_SHA256,
        "runtime_versions": versions,
        "status": "verified_offline_immutable_qasper_minilm_runtime",
        "weights_sha256": WEIGHTS_SHA256,
    }


def verify_runtime_asset(
    project_root: str | Path, model_path: str | Path | None = None
) -> dict[str, object]:
    """Verify the manifest-bound project-local snapshot, with no row access."""

    root = _reject_symlink_components(Path(project_root), "project root")
    if not root.is_dir():
        raise QasperMiniLMError("project root is unavailable")
    manifest_path, asset = _load_asset_manifest(root / ASSET_RELATIVE_PATH)
    local = asset.get("local_binding")
    if not isinstance(local, Mapping):
        raise QasperMiniLMError("local snapshot binding is missing")
    expected = _reject_symlink_components(
        root / str(local.get("ignored_runtime_directory")), "bound model root"
    )
    actual = expected if model_path is None else _reject_symlink_components(
        Path(model_path), "model root"
    )
    if actual != expected:
        raise QasperMiniLMError("model root is not the manifest-bound local path")
    return verify_runtime_binding(
        asset_manifest_path=manifest_path,
        model_root=actual,
    )


def synthetic_canary_texts() -> tuple[str, ...]:
    """Return the complete public 256-sentence synthetic canary preimage."""

    texts = tuple(
        (
            f"Synthetic evidence sentence {index:03d}: "
            f"entity_{index % 17:02d} relates to method_{(index * 7) % 23:02d} "
            f"under condition_{(index * 11) % 29:02d}."
        )
        for index in range(CANARY_SENTENCE_COUNT)
    )
    if _canonical_hash(list(texts)) != CANARY_TEXT_VECTOR_SHA256:
        raise QasperMiniLMError("synthetic canary generator drifted")
    return texts


def _validate_texts(texts: object) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise QasperMiniLMError("texts must be a sequence")
    if not 1 <= len(texts) <= MAXIMUM_TEXTS_PER_CALL:
        raise QasperMiniLMError("text count is outside the frozen bound")
    normalized: list[str] = []
    for value in texts:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise QasperMiniLMError("each text must be non-empty Unicode without NUL")
        if len(value) > MAXIMUM_TEXT_CHARACTERS:
            raise QasperMiniLMError("text exceeds the frozen character bound")
        try:
            raw = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise QasperMiniLMError("text contains invalid Unicode") from exc
        if len(raw) > MAXIMUM_TEXT_UTF8_BYTES:
            raise QasperMiniLMError("text exceeds the frozen UTF-8 bound")
        normalized.append(value)
    return tuple(normalized)


def quantize_embeddings(embeddings: object) -> tuple[tuple[int, ...], ...]:
    """Apply the frozen Python-round quantizer to a normalized matrix."""

    try:
        matrix = np.asarray(embeddings, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise QasperMiniLMError("embeddings are not a float32 matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
        raise QasperMiniLMError("embedding matrix shape is invalid")
    if not np.isfinite(matrix).all():
        raise QasperMiniLMError("embedding matrix contains a non-finite value")
    return tuple(
        tuple(int(round(float(component) * QUANTIZATION_SCALE)) for component in row)
        for row in matrix
    )


def _quantized_embedding_hash(embeddings: object) -> str:
    return _canonical_hash([list(row) for row in quantize_embeddings(embeddings)])


def quantized_cosine_similarity(query: object, paragraph: object) -> int:
    """Return integer cosine for two frozen normalized float32 embeddings."""

    try:
        left = np.asarray(query, dtype=np.float32)
        right = np.asarray(paragraph, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise QasperMiniLMError("cosine inputs are not float32 vectors") from exc
    if left.shape != (EMBEDDING_DIMENSION,) or right.shape != (EMBEDDING_DIMENSION,):
        raise QasperMiniLMError("cosine input shape is invalid")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise QasperMiniLMError("cosine input contains a non-finite value")
    cosine = math.fsum(
        float(left[index]) * float(right[index])
        for index in range(EMBEDDING_DIMENSION)
    )
    return int(round(cosine * QUANTIZATION_SCALE))


def query_paragraph_similarities(
    encoder: "OfflineMiniLMEncoder",
    query: str,
    paragraphs: Sequence[str],
) -> tuple[int, ...]:
    """Encode one query plus paragraphs and return ordered integer cosines."""

    if not isinstance(query, str):
        raise QasperMiniLMError("query must be text")
    normalized_paragraphs = _validate_texts(paragraphs)
    matrix = encoder.encode((query, *normalized_paragraphs))
    return tuple(
        quantized_cosine_similarity(matrix[0], matrix[index])
        for index in range(1, len(normalized_paragraphs) + 1)
    )


def _configure_offline_environment() -> None:
    for key, value in EXPECTED_EXECUTION["environment"].items():
        os.environ[str(key)] = str(value)


class OfflineMiniLMEncoder:
    """Verified CPU-only float32 encoder with an exact startup canary."""

    def __init__(
        self,
        *,
        asset_manifest_path: str | Path,
        model_root: str | Path,
        run_canary: bool = True,
    ) -> None:
        if run_canary is not True:
            raise QasperMiniLMError("the frozen runtime cannot skip its startup canary")
        self.runtime_receipt = verify_runtime_binding(
            asset_manifest_path=asset_manifest_path,
            model_root=model_root,
        )
        _configure_offline_environment()
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise QasperMiniLMError("offline embedding runtime is missing") from exc
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        try:
            model = SentenceTransformer(
                str(_reject_symlink_components(Path(model_root), "model root")),
                device="cpu",
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
            raise QasperMiniLMError("verified local MiniLM snapshot failed to load") from exc
        model.max_seq_length = MAXIMUM_SEQUENCE_LENGTH
        model.float()
        model.eval()
        parameters = tuple(model.parameters())
        if (
            model.max_seq_length != MAXIMUM_SEQUENCE_LENGTH
            or model.training
            or not parameters
            or any(parameter.device.type != "cpu" for parameter in parameters)
            or any(parameter.dtype != torch.float32 for parameter in parameters)
        ):
            raise QasperMiniLMError("loaded model violates the CPU float32 eval contract")
        self._model = model
        self.canary_receipt = run_synthetic_canary(self)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized = _validate_texts(texts)
        try:
            values = self._model.encode(
                list(normalized),
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                convert_to_tensor=False,
                device="cpu",
                normalize_embeddings=True,
                precision="float32",
                show_progress_bar=False,
            )
        except Exception as exc:
            raise QasperMiniLMError("offline MiniLM encoding failed") from exc
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (len(normalized), EMBEDDING_DIMENSION):
            raise QasperMiniLMError("offline encoder returned the wrong shape")
        if not np.isfinite(matrix).all():
            raise QasperMiniLMError("offline encoder returned a non-finite value")
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
            raise QasperMiniLMError("offline encoder returned an unnormalized embedding")
        return matrix

    def query_paragraph_similarities(
        self, query: str, paragraphs: Sequence[str]
    ) -> tuple[int, ...]:
        return query_paragraph_similarities(self, query, paragraphs)


def run_synthetic_canary(encoder: OfflineMiniLMEncoder) -> dict[str, object]:
    """Repeat the row-free 256-sentence canary and require byte equality."""

    texts = synthetic_canary_texts()
    matrices = tuple(encoder.encode(texts) for _ in range(2))
    quantized_hashes = tuple(_quantized_embedding_hash(matrix) for matrix in matrices)
    float_hashes = tuple(
        _sha256_bytes(matrix.astype("<f4", copy=False).tobytes(order="C"))
        for matrix in matrices
    )
    if (
        quantized_hashes != (CANARY_QUANTIZED_EMBEDDING_SHA256,) * 2
        or float_hashes != (CANARY_FLOAT32_BYTES_SHA256,) * 2
        or not np.array_equal(matrices[0], matrices[1])
    ):
        raise QasperMiniLMError("synthetic MiniLM startup canary drifted")
    return {
        "float32_bytes_sha256": CANARY_FLOAT32_BYTES_SHA256,
        "quantized_embedding_matrix_sha256": CANARY_QUANTIZED_EMBEDDING_SHA256,
        "qasper_rows_or_archives_accessed_by_canary": False,
        "repeat_count": 2,
        "repeat_exact": True,
        "sentence_count": CANARY_SENTENCE_COUNT,
        "status": "passed_exact_row_free_synthetic_canary",
        "text_vector_sha256": CANARY_TEXT_VECTOR_SHA256,
    }

"""Verified offline MiniLM asset with deterministic GPU batching for BRIGHT."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from replication_runtime.qasper_minilm_v1 import binding as frozen_asset


DEVICE = "cuda:0"
DTYPE = "float32"
BATCH_SIZE = 256
MAXIMUM_TEXTS_PER_CALL = 16_384
MAXIMUM_TEXT_CHARACTERS = 24_000
EMBEDDING_DIMENSION = 384
QUANTIZATION_SCALE = 1_000_000


class BrightMiniLMError(RuntimeError):
    """The BRIGHT GPU embedding runtime failed closed."""


def _validate_texts(texts: object) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise BrightMiniLMError("texts are not a sequence")
    if not 1 <= len(texts) <= MAXIMUM_TEXTS_PER_CALL:
        raise BrightMiniLMError("text count is outside the frozen bound")
    rows: list[str] = []
    for value in texts:
        if (
            not isinstance(value, str)
            or not value.strip()
            or "\x00" in value
            or len(value) > MAXIMUM_TEXT_CHARACTERS
        ):
            raise BrightMiniLMError("text is invalid")
        rows.append(value)
    return tuple(rows)


def float32_matrix_sha256(matrix: object) -> str:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != EMBEDDING_DIMENSION:
        raise BrightMiniLMError("embedding matrix shape drifted")
    if not np.isfinite(values).all():
        raise BrightMiniLMError("embedding matrix contains nonfinite values")
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def quantized_scores(matrix: object, query: object) -> np.ndarray:
    documents = np.asarray(matrix, dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32)
    if documents.ndim != 2 or documents.shape[1] != EMBEDDING_DIMENSION:
        raise BrightMiniLMError("document embedding matrix shape drifted")
    if vector.shape != (EMBEDDING_DIMENSION,):
        raise BrightMiniLMError("query embedding shape drifted")
    if not np.isfinite(documents).all() or not np.isfinite(vector).all():
        raise BrightMiniLMError("cosine input contains nonfinite values")
    cosine = np.asarray(documents @ vector, dtype=np.float32)
    return np.rint(cosine.astype(np.float64) * QUANTIZATION_SCALE).astype(np.int32)


class BrightMiniLMEncoder:
    """One verified float32 CUDA model with a repeat-exact synthetic canary."""

    def __init__(self, *, asset_manifest: Path, model_root: Path) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if not torch.cuda.is_available():
            raise BrightMiniLMError("the frozen CUDA device is unavailable")
        self.runtime_receipt = frozen_asset.verify_runtime_binding(
            asset_manifest_path=asset_manifest,
            model_root=model_root,
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            model = SentenceTransformer(
                str(model_root),
                device=DEVICE,
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
            raise BrightMiniLMError("verified MiniLM asset failed to load") from exc
        model.max_seq_length = frozen_asset.MAXIMUM_SEQUENCE_LENGTH
        model.float()
        model.eval()
        parameters = tuple(model.parameters())
        if (
            model.training
            or not parameters
            or any(parameter.device.type != "cuda" for parameter in parameters)
            or any(parameter.dtype != torch.float32 for parameter in parameters)
        ):
            raise BrightMiniLMError("loaded MiniLM violates the frozen GPU contract")
        self._model = model
        canary = frozen_asset.synthetic_canary_texts()
        first = self.encode(canary)
        second = self.encode(canary)
        if not np.array_equal(first, second):
            raise BrightMiniLMError("GPU MiniLM canary is not repeat exact")
        self.canary_receipt = {
            "device": DEVICE,
            "dtype": DTYPE,
            "float32_bytes_sha256": float32_matrix_sha256(first),
            "repeat_count": 2,
            "repeat_exact": True,
            "sentence_count": len(canary),
        }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = _validate_texts(texts)
        try:
            values = self._model.encode(
                list(rows),
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                convert_to_tensor=False,
                device=DEVICE,
                normalize_embeddings=True,
                precision="float32",
                show_progress_bar=False,
            )
        except Exception as exc:
            raise BrightMiniLMError("offline GPU MiniLM encoding failed") from exc
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (len(rows), EMBEDDING_DIMENSION):
            raise BrightMiniLMError("offline GPU MiniLM returned the wrong shape")
        if not np.isfinite(matrix).all():
            raise BrightMiniLMError("offline GPU MiniLM returned nonfinite values")
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
            raise BrightMiniLMError("offline GPU MiniLM returned unnormalized vectors")
        return matrix

"""Portable startup evidence for the exact frozen QASPER MiniLM runtime.

The immutable public manifest, complete model-tree verification, exact package
versions, offline environment, and CPU float32 SentenceTransformer constructor
are inherited from :mod:`replication_runtime.qasper_minilm_v1.binding`.  This
module deliberately does *not* construct ``OfflineMiniLMEncoder`` or invoke its
machine-specific expected-output-hash canary.

Instead, startup acceptance is limited to public structural evidence: the
exact v1 256-sentence preimage is encoded twice, both results must be native
float32 ``[256, 384]`` matrices with finite, normalized, non-collapsed rows,
and the two calls must agree both elementwise and byte-for-byte.  Observed
hashes are diagnostic only and can never be an acceptance oracle.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from replication_runtime.qasper_minilm_v1 import binding as frozen_v1


PORTABLE_CANARY_SCHEMA = "qasper_minilm_portable_startup_canary_v2"
PORTABLE_ROW_L2_NORM_ATOL = 1e-5


class PortableMiniLMError(RuntimeError):
    """Raised when the portable offline MiniLM contract cannot be proven."""


def _load_exact_v1_model(*, model_root: str | Path):
    """Construct the exact v1 CPU float32 model without its hash canary."""

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise PortableMiniLMError("offline embedding runtime is missing") from exc

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    try:
        model = SentenceTransformer(
            str(
                frozen_v1._reject_symlink_components(  # type: ignore[attr-defined]
                    Path(model_root), "model root"
                )
            ),
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
        raise PortableMiniLMError(
            "verified local MiniLM snapshot failed to load"
        ) from exc

    model.max_seq_length = frozen_v1.MAXIMUM_SEQUENCE_LENGTH
    model.float()
    model.eval()
    parameters = tuple(model.parameters())
    if (
        model.max_seq_length != frozen_v1.MAXIMUM_SEQUENCE_LENGTH
        or model.training
        or not parameters
        or any(parameter.device.type != "cpu" for parameter in parameters)
        or any(parameter.dtype != torch.float32 for parameter in parameters)
    ):
        raise PortableMiniLMError(
            "loaded model violates the exact v1 CPU float32 eval contract"
        )
    return model


def _portable_matrix(value: object, *, row_count: int) -> np.ndarray:
    """Validate one native-float32 normalized encoder result."""

    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PortableMiniLMError("portable encoder result is not a matrix") from exc
    if matrix.dtype != np.dtype(np.float32):
        raise PortableMiniLMError("portable encoder result dtype is not float32")
    if matrix.shape != (row_count, frozen_v1.EMBEDDING_DIMENSION):
        raise PortableMiniLMError("portable encoder matrix shape drifted")
    if not np.isfinite(matrix).all():
        raise PortableMiniLMError("portable encoder matrix is non-finite")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    maximum_error = float(np.max(np.abs(norms - 1.0)))
    if maximum_error > PORTABLE_ROW_L2_NORM_ATOL:
        raise PortableMiniLMError("portable encoder row norm drifted")
    return matrix


def _has_at_least_two_distinct_rows(matrix: np.ndarray) -> bool:
    first = matrix[0]
    return any(not np.array_equal(first, row) for row in matrix[1:])


def _observed_output_hashes(matrix: np.ndarray) -> dict[str, object]:
    """Return diagnostic hashes that are explicitly non-normative."""

    float_hash = hashlib.sha256(
        matrix.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    quantized_hash = frozen_v1._quantized_embedding_hash(  # type: ignore[attr-defined]
        matrix
    )
    return {
        "compared_to_expected_or_allowlist": False,
        "float32_little_endian_c_order_sha256": float_hash,
        "normative_acceptance": False,
        "quantized_embedding_matrix_sha256": quantized_hash,
    }


def run_portable_startup_canary(encoder: object) -> dict[str, object]:
    """Run two public-synthetic encodes with structural-only acceptance."""

    texts = frozen_v1.synthetic_canary_texts()
    if (
        len(texts) != frozen_v1.CANARY_SENTENCE_COUNT
        or frozen_v1._canonical_hash(list(texts))  # type: ignore[attr-defined]
        != frozen_v1.CANARY_TEXT_VECTOR_SHA256
    ):
        raise PortableMiniLMError("public canary text vector identity drifted")
    encode = getattr(encoder, "encode", None)
    if not callable(encode):
        raise PortableMiniLMError("portable canary encoder is unavailable")

    first = _portable_matrix(encode(texts), row_count=len(texts))
    second = _portable_matrix(encode(texts), row_count=len(texts))
    elementwise_exact = np.array_equal(first, second)
    byte_exact = first.tobytes(order="C") == second.tobytes(order="C")
    if not elementwise_exact or not byte_exact:
        raise PortableMiniLMError("portable canary repeat is not byte/element exact")
    if not _has_at_least_two_distinct_rows(first):
        raise PortableMiniLMError("portable canary embeddings collapsed to one vector")

    norms = np.linalg.norm(first.astype(np.float64), axis=1)
    maximum_norm_error = float(np.max(np.abs(norms - 1.0)))
    return {
        "all_values_finite": True,
        "at_least_two_distinct_vectors": True,
        "embedding_dtype": "float32",
        "embedding_shape": [
            frozen_v1.CANARY_SENTENCE_COUNT,
            frozen_v1.EMBEDDING_DIMENSION,
        ],
        "external_network_calls": 0,
        "formal_QASPER_source_or_rows_accessed": False,
        "formal_TAT_QA_source_or_rows_accessed": False,
        "maximum_observed_row_l2_norm_error": maximum_norm_error,
        "observed_output_hashes": _observed_output_hashes(first),
        "per_row_l2_norm_maximum_error": PORTABLE_ROW_L2_NORM_ATOL,
        "public_text_vector_identity_exact": True,
        "public_text_vector_sha256": frozen_v1.CANARY_TEXT_VECTOR_SHA256,
        "qasper_rows_or_archives_accessed_by_canary": False,
        "repeat_byte_exact": True,
        "repeat_count": 2,
        "repeat_elementwise_exact": True,
        "schema": PORTABLE_CANARY_SCHEMA,
        "sentence_count": frozen_v1.CANARY_SENTENCE_COUNT,
        "status": "passed_portable_public_synthetic_structural_canary",
        "tatqa_rows_or_archives_accessed_by_canary": False,
    }


class PortableOfflineMiniLMEncoder:
    """Exact v1 offline model binding with portable startup acceptance."""

    def __init__(
        self,
        *,
        asset_manifest_path: str | Path,
        model_root: str | Path,
        run_canary: bool = True,
    ) -> None:
        if run_canary is not True:
            raise PortableMiniLMError("portable startup canary cannot be skipped")
        try:
            self.runtime_receipt = frozen_v1.verify_runtime_binding(
                asset_manifest_path=asset_manifest_path,
                model_root=model_root,
            )
            frozen_v1._configure_offline_environment()  # type: ignore[attr-defined]
        except Exception as exc:
            raise PortableMiniLMError("exact v1 immutable runtime binding failed") from exc
        self._model = _load_exact_v1_model(model_root=model_root)
        self.canary_receipt = run_portable_startup_canary(self)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        try:
            normalized = frozen_v1._validate_texts(texts)  # type: ignore[attr-defined]
            values = self._model.encode(
                list(normalized),
                batch_size=frozen_v1.BATCH_SIZE,
                convert_to_numpy=True,
                convert_to_tensor=False,
                device="cpu",
                normalize_embeddings=True,
                precision="float32",
                show_progress_bar=False,
            )
        except Exception as exc:
            raise PortableMiniLMError("offline MiniLM encoding failed") from exc
        return _portable_matrix(values, row_count=len(normalized))


__all__ = [
    "PORTABLE_CANARY_SCHEMA",
    "PORTABLE_ROW_L2_NORM_ATOL",
    "PortableMiniLMError",
    "PortableOfflineMiniLMEncoder",
    "run_portable_startup_canary",
]

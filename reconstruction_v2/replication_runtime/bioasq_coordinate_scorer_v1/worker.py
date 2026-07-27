"""One-lifecycle GPU1 worker for six source-free BioASQ coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import stat
from typing import Mapping, Protocol, Sequence

import numpy as np

from replication_runtime.bright_minilm_v1 import encoder as bright_minilm

from .contract import (
    BioasqCoordinateScorerError,
    CORPUS_SIZE,
    CUDA_VISIBLE_DEVICES,
    DENSE_SCORE_NAMES,
    LOGICAL_CUDA_DEVICE,
    MAX_SCORE_ABS,
    MODEL_BINDING_SCHEMA,
    PHYSICAL_GPU,
    SCORE_NAMES,
    SCORE_SCALE,
    WORKER_ENVIRONMENT_KEYS,
    WORKER_FIXED_ENVIRONMENT_VALUES,
    canonical_bytes,
    input_projection,
    make_output,
    serialize_passages,
    serialize_query_variants,
    stable_hash,
    validate_input,
    verify_typed_core_binding,
)


class MiniLMEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> object: ...


class CrossEncoderScorer(Protocol):
    def __call__(
        self, pairs: Sequence[tuple[str, str]]
    ) -> Sequence[Real]: ...


_CROSS_ENCODER_REQUIRED_FILES = (
    (
        "config.json",
        794,
        "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    ),
    (
        "model.safetensors",
        90_870_598,
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
    ),
    (
        "special_tokens_map.json",
        132,
        "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
    ),
    (
        "tokenizer.json",
        711_396,
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    (
        "tokenizer_config.json",
        1_330,
        "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
    ),
    (
        "vocab.txt",
        231_508,
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
)
_CROSS_ENCODER_REQUIRED_TREE_SHA256 = (
    "923d4371d5fe13534d7431895890c2142"
    "a8552a441f09ec7b28d035aaae9120c"
)


def _verify_cross_encoder_model_tree(root: Path) -> str:
    """Bind every normative CE byte before constructing the model."""

    if not isinstance(root, Path) or not root.is_absolute():
        raise BioasqCoordinateScorerError(
            "cross-encoder model root must be absolute"
        )
    try:
        metadata = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise BioasqCoordinateScorerError(
                "cross-encoder model root is unavailable"
            )
        children = tuple(root.iterdir())
    except OSError as exc:
        raise BioasqCoordinateScorerError(
            "cross-encoder model root is unavailable"
        ) from exc
    required_names = {
        name for name, _size, _sha256 in _CROSS_ENCODER_REQUIRED_FILES
    }
    observed_files = {
        child.name for child in children if child.is_file()
    }
    observed_directories = {
        child.name for child in children if child.is_dir()
    }
    if (
        observed_files != required_names
        or not observed_directories <= {".cache"}
        or any(child.is_symlink() for child in children)
        or any(
            not child.is_file() and not child.is_dir()
            for child in children
        )
    ):
        raise BioasqCoordinateScorerError(
            "cross-encoder model topology drifted"
        )
    rows: list[dict[str, object]] = []
    for relative, expected_size, expected_sha256 in (
        _CROSS_ENCODER_REQUIRED_FILES
    ):
        path = root / relative
        try:
            file_metadata = path.lstat()
        except OSError as exc:
            raise BioasqCoordinateScorerError(
                "cross-encoder required model file is unavailable"
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(file_metadata.st_mode)
            or file_metadata.st_nlink != 1
            or file_metadata.st_size != expected_size
            or _file_sha256(path, "cross-encoder required model file")
            != expected_sha256
        ):
            raise BioasqCoordinateScorerError(
                "cross-encoder required model file drifted"
            )
        rows.append(
            {
                "path": relative,
                "sha256": expected_sha256,
                "size": expected_size,
            }
        )
    tree_sha256 = stable_hash(rows)
    if tree_sha256 != _CROSS_ENCODER_REQUIRED_TREE_SHA256:
        raise BioasqCoordinateScorerError(
            "cross-encoder normative model tree drifted"
        )
    return tree_sha256


def _validate_effective_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    effective = os.environ if environment is None else environment
    if (
        not isinstance(effective, Mapping)
        or frozenset(effective) != WORKER_ENVIRONMENT_KEYS
        or any(
            effective.get(key) != value
            for key, value in WORKER_FIXED_ENVIRONMENT_VALUES.items()
        )
    ):
        raise BioasqCoordinateScorerError(
            "worker environment contract failed"
        )


def _validate_logical_cuda0(torch_module: object | None = None) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_VISIBLE_DEVICES:
        raise BioasqCoordinateScorerError("physical GPU1 binding drifted")
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise BioasqCoordinateScorerError(
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
        raise BioasqCoordinateScorerError(
            "logical cuda:0 attestation failed"
        ) from exc
    if (
        available is not True
        or type(count) is not int
        or count != 1
        or type(current) is not int
        or current != 0
        or str(getattr(sentinel, "device", None)) != LOGICAL_CUDA_DEVICE
    ):
        raise BioasqCoordinateScorerError(
            "logical cuda:0 attestation failed"
        )


def _validated_model_texts(
    values: Sequence[str], field: str
) -> tuple[str, ...]:
    try:
        return bright_minilm._validate_texts(values)
    except bright_minilm.BrightMiniLMError as exc:
        raise BioasqCoordinateScorerError(
            f"{field} exceeds the frozen MiniLM text contract"
        ) from exc


def _embedding_matrix(
    value: object, *, row_count: int, field: str
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if (
        matrix.shape
        != (row_count, bright_minilm.EMBEDDING_DIMENSION)
        or not np.isfinite(matrix).all()
    ):
        raise BioasqCoordinateScorerError(
            f"{field} embedding matrix drifted"
        )
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
        raise BioasqCoordinateScorerError(
            f"{field} embeddings are not normalized"
        )
    return matrix


def _minilm_vector(matrix: np.ndarray, query: np.ndarray) -> tuple[int, ...]:
    try:
        values = bright_minilm.quantized_scores(matrix, query)
    except bright_minilm.BrightMiniLMError as exc:
        raise BioasqCoordinateScorerError(
            "MiniLM score quantization failed"
        ) from exc
    result = tuple(int(value) for value in values)
    if (
        len(result) != CORPUS_SIZE
        or any(abs(value) > MAX_SCORE_ABS for value in result)
    ):
        raise BioasqCoordinateScorerError(
            "MiniLM bounded integer vector drifted"
        )
    return result


def _cross_values(value: object, *, expected_count: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BioasqCoordinateScorerError(
            "cross-encoder result is not a sequence"
        )
    if len(value) != expected_count:
        raise BioasqCoordinateScorerError(
            "cross-encoder result width drifted"
        )
    raw: list[float] = []
    for score in value:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise BioasqCoordinateScorerError(
                "cross-encoder result is not numeric"
            )
        numeric = float(score)
        if not math.isfinite(numeric):
            raise BioasqCoordinateScorerError(
                "cross-encoder result is not finite"
            )
        raw.append(numeric)
    scaled = np.rint(np.asarray(raw, dtype=np.float64) * SCORE_SCALE)
    if (
        not np.isfinite(scaled).all()
        or np.any(np.abs(scaled) > MAX_SCORE_ABS)
    ):
        raise BioasqCoordinateScorerError(
            "cross-encoder quantized score exceeds the bound"
        )
    return tuple(int(item) for item in scaled)


def _vector_slice(
    flat: Sequence[int], query_ordinal: int
) -> tuple[int, ...]:
    start = query_ordinal * CORPUS_SIZE
    result = tuple(flat[start : start + CORPUS_SIZE])
    if len(result) != CORPUS_SIZE:
        raise BioasqCoordinateScorerError(
            "cross-encoder query slice drifted"
        )
    return result


def score_with_dependencies(
    value: object,
    *,
    minilm_encoder: MiniLMEncoder,
    cross_encoder: CrossEncoderScorer,
    model_binding_sha256: str,
) -> dict[str, object]:
    """Compute six vectors with two CE calls and one MiniLM encode call."""

    scorer_input = validate_input(value)
    encode = getattr(minilm_encoder, "encode", None)
    if not callable(encode) or not callable(cross_encoder):
        raise BioasqCoordinateScorerError(
            "injected scorer dependency is invalid"
        )
    passages = _validated_model_texts(
        serialize_passages(scorer_input.passages),
        "passages",
    )
    bundles = serialize_query_variants(scorer_input.queries)
    dense_queries = _validated_model_texts(
        tuple(
            bundle[name]
            for bundle in bundles
            for name in DENSE_SCORE_NAMES
        ),
        "dense query variants",
    )
    all_minilm_texts = _validated_model_texts(
        passages + dense_queries,
        "batched passage/query variants",
    )
    try:
        all_embeddings = encode(all_minilm_texts)
    except Exception as exc:
        raise BioasqCoordinateScorerError(
            "MiniLM dependency failed"
        ) from exc
    matrix = _embedding_matrix(
        all_embeddings,
        row_count=CORPUS_SIZE + len(dense_queries),
        field="batched passage/query",
    )
    passage_matrix = matrix[:CORPUS_SIZE]
    dense_matrix = matrix[CORPUS_SIZE:]

    pair_count = len(scorer_input.queries) * CORPUS_SIZE
    raw_pairs = tuple(
        (bundle["raw_ce"], passage)
        for bundle in bundles
        for passage in passages
    )
    if len(raw_pairs) != pair_count:
        raise BioasqCoordinateScorerError("raw CE pair slate drifted")
    try:
        raw_flat = _cross_values(
            cross_encoder(raw_pairs),
            expected_count=pair_count,
        )
    except BioasqCoordinateScorerError:
        raise
    except Exception as exc:
        raise BioasqCoordinateScorerError(
            "raw cross-encoder dependency failed"
        ) from exc
    del raw_pairs

    focus_pairs = tuple(
        (bundle["focus_ce"], passage)
        for bundle in bundles
        for passage in passages
    )
    if len(focus_pairs) != pair_count:
        raise BioasqCoordinateScorerError("focus CE pair slate drifted")
    try:
        focus_flat = _cross_values(
            cross_encoder(focus_pairs),
            expected_count=pair_count,
        )
    except BioasqCoordinateScorerError:
        raise
    except Exception as exc:
        raise BioasqCoordinateScorerError(
            "focus cross-encoder dependency failed"
        ) from exc

    score_rows: list[dict[str, tuple[int, ...]]] = []
    for query_ordinal in range(len(scorer_input.queries)):
        dense_start = query_ordinal * len(DENSE_SCORE_NAMES)
        row: dict[str, tuple[int, ...]] = {
            "raw_ce": _vector_slice(raw_flat, query_ordinal),
            "focus_ce": _vector_slice(focus_flat, query_ordinal),
        }
        for offset, name in enumerate(DENSE_SCORE_NAMES):
            row[name] = _minilm_vector(
                passage_matrix,
                dense_matrix[dense_start + offset],
            )
        if tuple(row) != SCORE_NAMES:
            raise BioasqCoordinateScorerError(
                "score coordinate order drifted"
            )
        score_rows.append(row)
    return make_output(
        scorer_input=scorer_input,
        score_rows=score_rows,
        model_binding_sha256=model_binding_sha256,
    )


def _load_json(path: Path, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise BioasqCoordinateScorerError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BioasqCoordinateScorerError(f"{field} is invalid") from exc
    if raw != canonical_bytes(value) or not isinstance(value, dict):
        raise BioasqCoordinateScorerError(
            f"{field} is not canonical JSON"
        )
    return value


def _write_exclusive(path: Path, value: object) -> None:
    raw = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.is_symlink() or (path.stat().st_mode & 0o777) != 0o600:
        raise BioasqCoordinateScorerError(
            "private output permissions drifted"
        )


def _file_sha256(path: Path, field: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise BioasqCoordinateScorerError(f"{field} is unavailable")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_production_dependencies(
    *,
    minilm_asset_manifest: Path,
    minilm_model_root: Path,
    cross_encoder_model_root: Path,
) -> tuple[object, object, str]:
    """Load each frozen local model exactly once on physical GPU1."""

    _validate_logical_cuda0()
    cross_encoder_tree_sha256 = _verify_cross_encoder_model_tree(
        cross_encoder_model_root
    )
    from assumption_agent.benchmarks.hitab_p1_runtime_v1 import (
        BrightCrossEncoderProductionScorer,
    )
    from replication_runtime.bright_minilm_v1.encoder import BrightMiniLMEncoder

    try:
        minilm = BrightMiniLMEncoder(
            asset_manifest=minilm_asset_manifest,
            model_root=minilm_model_root,
        )
        cross = BrightCrossEncoderProductionScorer(
            cross_encoder_model_root,
            physical_gpu=PHYSICAL_GPU,
        )
    except Exception as exc:
        raise BioasqCoordinateScorerError(
            "frozen local scorer dependency failed to load"
        ) from exc
    runtime_receipt = getattr(minilm, "runtime_receipt", None)
    canary_receipt = getattr(minilm, "canary_receipt", None)
    if not isinstance(runtime_receipt, Mapping) or not isinstance(
        canary_receipt, Mapping
    ):
        raise BioasqCoordinateScorerError(
            "MiniLM binding receipts are unavailable"
        )
    if (
        canary_receipt.get("repeat_count") != 2
        or canary_receipt.get("repeat_exact") is not True
    ):
        raise BioasqCoordinateScorerError(
            "MiniLM constructor canary count drifted"
        )
    model_binding = {
        "cross_encoder_class": (
            "hitab_p1_runtime_v1.BrightCrossEncoderProductionScorer"
        ),
        "cross_encoder_model_root": str(cross_encoder_model_root),
        "cross_encoder_required_tree_sha256": (
            cross_encoder_tree_sha256
        ),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "logical_cuda_device": LOGICAL_CUDA_DEVICE,
        "minilm_asset_manifest_sha256": _file_sha256(
            minilm_asset_manifest, "MiniLM asset manifest"
        ),
        "minilm_canary_receipt_sha256": stable_hash(dict(canary_receipt)),
        "minilm_class": "bright_minilm_v1.BrightMiniLMEncoder",
        "minilm_model_root": str(minilm_model_root),
        "minilm_runtime_receipt_sha256": stable_hash(dict(runtime_receipt)),
        "physical_gpu": PHYSICAL_GPU,
        "schema": MODEL_BINDING_SCHEMA,
    }
    return minilm, cross, stable_hash(model_binding)


def main(argv: Sequence[str] | None = None) -> int:
    _validate_effective_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--minilm-asset-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model-root", required=True, type=Path)
    parser.add_argument("--cross-encoder-model-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    verify_typed_core_binding(arguments.project_root)
    scorer_input = validate_input(_load_json(arguments.input, "private input"))
    minilm, cross, model_binding_sha256 = _build_production_dependencies(
        minilm_asset_manifest=arguments.minilm_asset_manifest,
        minilm_model_root=arguments.minilm_model_root,
        cross_encoder_model_root=arguments.cross_encoder_model_root,
    )
    output = score_with_dependencies(
        input_projection(scorer_input),
        minilm_encoder=minilm,
        cross_encoder=cross,
        model_binding_sha256=model_binding_sha256,
    )
    _write_exclusive(arguments.output, output)
    print(
        json.dumps(
            {
                "corpus_count": CORPUS_SIZE,
                "model_binding_sha256": model_binding_sha256,
                "output_self_sha256": output["self_sha256"],
                "query_count": len(scorer_input.queries),
                "stage": "coordinate_score",
                "status": "passed",
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_build_production_dependencies",
    "_validate_effective_environment",
    "_validate_logical_cuda0",
    "score_with_dependencies",
]

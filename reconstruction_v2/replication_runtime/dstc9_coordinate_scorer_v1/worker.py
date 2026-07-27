"""One-lifecycle GPU1 worker for six source-free DSTC9 score coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from replication_runtime.bright_minilm_v1 import encoder as bright_minilm

from .contract import (
    CORPUS_SIZE,
    CUDA_VISIBLE_DEVICES,
    ENTITY_NONE_SERIALIZATION,
    LOGICAL_CUDA_DEVICE,
    MAX_SCORE_ABS,
    MODEL_BINDING_SCHEMA,
    PHYSICAL_GPU,
    SCORE_NAMES,
    SCORE_SCALE,
    WORKER_ENVIRONMENT_KEYS,
    WORKER_FIXED_ENVIRONMENT_VALUES,
    Dstc9CoordinateScorerError,
    ScorerInput,
    canonical_bytes,
    input_projection,
    make_output,
    serialize_entity_fields,
    serialize_model_queries,
    serialize_passages,
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
        raise Dstc9CoordinateScorerError(
            "worker environment contract failed"
        )


def _validate_logical_cuda0(torch_module: object | None = None) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_VISIBLE_DEVICES:
        raise Dstc9CoordinateScorerError("physical GPU1 binding drifted")
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise Dstc9CoordinateScorerError(
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
        raise Dstc9CoordinateScorerError(
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
        raise Dstc9CoordinateScorerError(
            "logical cuda:0 attestation failed"
        )


def _validated_model_texts(
    values: Sequence[str], field: str
) -> tuple[str, ...]:
    try:
        return bright_minilm._validate_texts(values)
    except bright_minilm.BrightMiniLMError as exc:
        raise Dstc9CoordinateScorerError(
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
        raise Dstc9CoordinateScorerError(
            f"{field} embedding matrix drifted"
        )
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
        raise Dstc9CoordinateScorerError(
            f"{field} embeddings are not normalized"
        )
    return matrix


def _minilm_vector(matrix: np.ndarray, query: np.ndarray) -> tuple[int, ...]:
    try:
        values = bright_minilm.quantized_scores(matrix, query)
    except bright_minilm.BrightMiniLMError as exc:
        raise Dstc9CoordinateScorerError(
            "MiniLM score quantization failed"
        ) from exc
    result = tuple(int(value) for value in values)
    if (
        len(result) != CORPUS_SIZE
        or any(abs(value) > MAX_SCORE_ABS for value in result)
    ):
        raise Dstc9CoordinateScorerError(
            "MiniLM bounded integer vector drifted"
        )
    return result


def _cross_vector(
    value: object,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Dstc9CoordinateScorerError(
            "cross-encoder result is not a sequence"
        )
    if len(value) != CORPUS_SIZE:
        raise Dstc9CoordinateScorerError(
            "cross-encoder result width drifted"
        )
    raw: list[float] = []
    for score in value:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise Dstc9CoordinateScorerError(
                "cross-encoder result is not numeric"
            )
        numeric = float(score)
        if not math.isfinite(numeric):
            raise Dstc9CoordinateScorerError(
                "cross-encoder result is not finite"
            )
        raw.append(numeric)
    scaled = np.rint(np.asarray(raw, dtype=np.float64) * SCORE_SCALE)
    if (
        not np.isfinite(scaled).all()
        or np.any(np.abs(scaled) > MAX_SCORE_ABS)
    ):
        raise Dstc9CoordinateScorerError(
            "cross-encoder quantized score exceeds the bound"
        )
    return tuple(int(value) for value in scaled)


def score_with_dependencies(
    value: object,
    *,
    minilm_encoder: MiniLMEncoder,
    cross_encoder: CrossEncoderScorer,
    model_binding_sha256: str,
) -> dict[str, object]:
    """Compute all six vectors with exactly one pair of loaded dependencies."""

    scorer_input = validate_input(value)
    encode = getattr(minilm_encoder, "encode", None)
    if not callable(encode) or not callable(cross_encoder):
        raise Dstc9CoordinateScorerError(
            "injected scorer dependency is invalid"
        )
    queries = _validated_model_texts(
        serialize_model_queries(scorer_input.histories),
        "model queries",
    )
    passages = _validated_model_texts(
        serialize_passages(scorer_input.snippets),
        "full passages",
    )
    entity_fields = _validated_model_texts(
        serialize_entity_fields(scorer_input.snippets),
        "entity fields",
    )
    title_fields = _validated_model_texts(
        tuple(row.title for row in scorer_input.snippets),
        "title fields",
    )
    body_fields = _validated_model_texts(
        tuple(row.body for row in scorer_input.snippets),
        "body fields",
    )
    if ENTITY_NONE_SERIALIZATION not in entity_fields and any(
        row.entity_name is None for row in scorer_input.snippets
    ):
        raise Dstc9CoordinateScorerError(
            "missing entity serialization drifted"
        )

    query_matrix = _embedding_matrix(
        encode(queries), row_count=len(queries), field="query"
    )
    passage_matrix = _embedding_matrix(
        encode(passages), row_count=CORPUS_SIZE, field="passage"
    )
    entity_matrix = _embedding_matrix(
        encode(entity_fields), row_count=CORPUS_SIZE, field="entity"
    )
    title_matrix = _embedding_matrix(
        encode(title_fields), row_count=CORPUS_SIZE, field="title"
    )
    body_matrix = _embedding_matrix(
        encode(body_fields), row_count=CORPUS_SIZE, field="body"
    )

    score_rows: list[dict[str, tuple[int, ...]]] = []
    for ordinal, (history, query_text) in enumerate(
        zip(scorer_input.histories, queries)
    ):
        last_user_turn = history.turns[-1].text
        global_pairs = tuple((query_text, passage) for passage in passages)
        last_turn_pairs = tuple(
            (last_user_turn, passage) for passage in passages
        )
        try:
            global_ce = _cross_vector(cross_encoder(global_pairs))
            last_turn_ce = _cross_vector(cross_encoder(last_turn_pairs))
        except Dstc9CoordinateScorerError:
            raise
        except Exception as exc:
            raise Dstc9CoordinateScorerError(
                f"cross-encoder dependency failed for query {ordinal}"
            ) from exc
        query_vector = query_matrix[ordinal]
        score_rows.append(
            {
                "global_ce": global_ce,
                "last_turn_ce": last_turn_ce,
                "minilm": _minilm_vector(passage_matrix, query_vector),
                "entity": _minilm_vector(entity_matrix, query_vector),
                "title": _minilm_vector(title_matrix, query_vector),
                "body": _minilm_vector(body_matrix, query_vector),
            }
        )
    if tuple(score_rows[0]) != SCORE_NAMES:
        raise Dstc9CoordinateScorerError(
            "score coordinate order drifted"
        )
    return make_output(
        scorer_input=scorer_input,
        score_rows=score_rows,
        model_binding_sha256=model_binding_sha256,
    )


def _load_json(path: Path, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Dstc9CoordinateScorerError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9CoordinateScorerError(f"{field} is invalid") from exc
    if raw != canonical_bytes(value) or not isinstance(value, dict):
        raise Dstc9CoordinateScorerError(
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
        raise Dstc9CoordinateScorerError(
            "private output permissions drifted"
        )


def _file_sha256(path: Path, field: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise Dstc9CoordinateScorerError(f"{field} is unavailable")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_production_dependencies(
    *,
    minilm_asset_manifest: Path,
    minilm_model_root: Path,
    cross_encoder_model_root: Path,
) -> tuple[object, object, str]:
    """Load the frozen MiniLM and cross encoder once on physical GPU1."""

    _validate_logical_cuda0()
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
        raise Dstc9CoordinateScorerError(
            "frozen local scorer dependency failed to load"
        ) from exc
    runtime_receipt = getattr(minilm, "runtime_receipt", None)
    canary_receipt = getattr(minilm, "canary_receipt", None)
    if not isinstance(runtime_receipt, Mapping) or not isinstance(
        canary_receipt, Mapping
    ):
        raise Dstc9CoordinateScorerError(
            "MiniLM binding receipts are unavailable"
        )
    model_binding = {
        "cross_encoder_class": (
            "hitab_p1_runtime_v1.BrightCrossEncoderProductionScorer"
        ),
        "cross_encoder_model_root": str(cross_encoder_model_root),
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
                "query_count": len(scorer_input.histories),
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

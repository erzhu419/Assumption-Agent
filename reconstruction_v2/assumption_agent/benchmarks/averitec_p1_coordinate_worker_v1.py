"""Single-GPU MiniLM coordinate worker for AVeriTeC P1.

The worker embeds one block corpus and every frozen typed query variant once.
It emits quantized cosine coordinates keyed only by opaque item IDs.  It has no
source loader, qrel, family, verdict, evaluator, score, API, or network surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Callable, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks.averitec_p1_typed_core_v1 import (
    QUERY_VARIANT_IDS,
    SCALE,
    STUDY_ID,
    typed_query_variants,
)


VERSION = "averitec_p1_coordinate_worker_v1"
INPUT_SCHEMA = f"{VERSION}_private_input_v1"
OUTPUT_SCHEMA = f"{VERSION}_private_output_v1"
MAX_DOCUMENT_COUNT = 4_096
MAX_QUERY_COUNT = 512
MAX_TEXT_CHARACTERS = 1_000_000
BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 512
NATIVE_THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class AveritecP1CoordinateError(RuntimeError):
    """The coordinate input, model, CUDA runtime, or output drifted."""


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AveritecP1CoordinateError(
            "coordinate value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > MAX_TEXT_CHARACTERS
    ):
        raise AveritecP1CoordinateError(f"{field} is invalid")
    return value


def _opaque(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise AveritecP1CoordinateError(f"{field} is not opaque")
    return value


def validate_input(
    value: object,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"documents", "queries", "schema", "study_id"}
        or value.get("schema") != INPUT_SCHEMA
        or value.get("study_id") != STUDY_ID
    ):
        raise AveritecP1CoordinateError("coordinate input envelope drifted")
    raw_documents = value.get("documents")
    raw_queries = value.get("queries")
    if (
        not isinstance(raw_documents, list)
        or not 5 <= len(raw_documents) <= MAX_DOCUMENT_COUNT
        or not isinstance(raw_queries, list)
        or not 1 <= len(raw_queries) <= MAX_QUERY_COUNT
    ):
        raise AveritecP1CoordinateError("coordinate input cardinality drifted")
    documents: list[str] = []
    for ordinal, row in enumerate(raw_documents):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"ordinal", "text"}
            or row.get("ordinal") != ordinal
        ):
            raise AveritecP1CoordinateError("coordinate document shape drifted")
        documents.append(_text(row.get("text"), "document text"))
    if len(set(documents)) != len(documents):
        raise AveritecP1CoordinateError("coordinate documents are duplicated")
    queries: list[tuple[str, str]] = []
    seen_items: set[str] = set()
    for ordinal, row in enumerate(raw_queries):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"item_id", "ordinal", "text"}
            or row.get("ordinal") != ordinal
        ):
            raise AveritecP1CoordinateError("coordinate query shape drifted")
        item_id = _opaque(row.get("item_id"), "item_id")
        if item_id in seen_items:
            raise AveritecP1CoordinateError("coordinate item_id is duplicated")
        seen_items.add(item_id)
        queries.append((item_id, _text(row.get("text"), "query text")))
    return tuple(documents), tuple(queries)


def private_input_payload(
    *, documents: Sequence[str], queries: Sequence[tuple[str, str]]
) -> dict[str, object]:
    body = {
        "documents": [
            {"ordinal": ordinal, "text": text}
            for ordinal, text in enumerate(documents)
        ],
        "queries": [
            {"item_id": item_id, "ordinal": ordinal, "text": text}
            for ordinal, (item_id, text) in enumerate(queries)
        ],
        "schema": INPUT_SCHEMA,
        "study_id": STUDY_ID,
    }
    validate_input(body)
    return body


def _matrix(value: object, expected_rows: int) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AveritecP1CoordinateError("embedding matrix is invalid") from exc
    if (
        matrix.ndim != 2
        or matrix.shape[0] != expected_rows
        or matrix.shape[1] <= 0
        or not np.isfinite(matrix).all()
    ):
        raise AveritecP1CoordinateError("embedding matrix shape drifted")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
        raise AveritecP1CoordinateError("embedding rows are not normalized")
    return matrix


def coordinate_output(
    *,
    private_input: Mapping[str, object],
    encode: Callable[[Sequence[str]], object],
    runtime_receipt: Mapping[str, object],
) -> dict[str, object]:
    documents, queries = validate_input(private_input)
    variant_rows: list[tuple[str, str]] = []
    variant_texts: list[str] = []
    for item_id, query in queries:
        variants = typed_query_variants(query)
        for variant in QUERY_VARIANT_IDS:
            variant_rows.append((item_id, variant))
            variant_texts.append(variants[variant])
    document_matrix = _matrix(encode(documents), len(documents))
    query_matrix = _matrix(encode(variant_texts), len(variant_texts))
    cosine = query_matrix @ document_matrix.T
    if not np.isfinite(cosine).all() or np.any(cosine < -1.00001) or np.any(
        cosine > 1.00001
    ):
        raise AveritecP1CoordinateError("cosine coordinate drifted")
    unit = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)
    quantized = np.rint(unit * SCALE).astype(np.int64)
    rows: dict[str, dict[str, list[int]]] = {
        item_id: {} for item_id, _query in queries
    }
    for row_index, (item_id, variant) in enumerate(variant_rows):
        rows[item_id][variant] = [
            int(value) for value in quantized[row_index].tolist()
        ]
    ordered_rows = []
    for item_id, _query in queries:
        if tuple(rows[item_id]) != QUERY_VARIANT_IDS:
            raise AveritecP1CoordinateError("variant output order drifted")
        ordered_rows.append(
            {"item_id": item_id, "variant_scores": rows[item_id]}
        )
    body = {
        "document_count": len(documents),
        "input_sha256": stable_hash(private_input),
        "query_count": len(queries),
        "rows": ordered_rows,
        "runtime_receipt": dict(runtime_receipt),
        "schema": OUTPUT_SCHEMA,
        "study_id": STUDY_ID,
    }
    body["self_sha256"] = stable_hash(body)
    return body


def validate_output(
    value: object,
    *,
    expected_input: Mapping[str, object],
) -> dict[str, object]:
    """Validate a worker result without admitting text or a wider payload."""

    documents, queries = validate_input(expected_input)
    if not isinstance(value, Mapping) or set(value) != {
        "document_count",
        "input_sha256",
        "query_count",
        "rows",
        "runtime_receipt",
        "schema",
        "self_sha256",
        "study_id",
    }:
        raise AveritecP1CoordinateError("coordinate output envelope drifted")
    normalized = dict(value)
    self_sha256 = normalized.pop("self_sha256", None)
    if (
        not isinstance(self_sha256, str)
        or _HEX64.fullmatch(self_sha256) is None
        or self_sha256 != stable_hash(normalized)
    ):
        raise AveritecP1CoordinateError("coordinate output self hash drifted")
    if (
        normalized.get("schema") != OUTPUT_SCHEMA
        or normalized.get("study_id") != STUDY_ID
        or normalized.get("input_sha256") != stable_hash(expected_input)
        or normalized.get("document_count") != len(documents)
        or normalized.get("query_count") != len(queries)
    ):
        raise AveritecP1CoordinateError("coordinate output binding drifted")
    runtime = normalized.get("runtime_receipt")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "cuda_allocate_and_synchronize",
        "cuda_device_count",
        "deterministic_algorithms_enabled",
        "cuda_logical_device",
        "minilm_all_parameters_cuda0",
        "minilm_parameter_count",
        "native_and_torch_thread_count",
        "torch_manual_seed",
    }:
        raise AveritecP1CoordinateError("coordinate runtime receipt drifted")
    if (
        runtime.get("cuda_allocate_and_synchronize") is not True
        or runtime.get("cuda_device_count") != 1
        or runtime.get("deterministic_algorithms_enabled") is not True
        or runtime.get("cuda_logical_device") != 0
        or runtime.get("minilm_all_parameters_cuda0") is not True
        or type(runtime.get("minilm_parameter_count")) is not int
        or int(runtime["minilm_parameter_count"]) <= 0
        or runtime.get("native_and_torch_thread_count") != 1
        or runtime.get("torch_manual_seed") != 0
    ):
        raise AveritecP1CoordinateError("coordinate CUDA attestation drifted")
    rows = normalized.get("rows")
    if not isinstance(rows, list) or len(rows) != len(queries):
        raise AveritecP1CoordinateError("coordinate result row count drifted")
    expected_items = [item_id for item_id, _text_value in queries]
    for expected_item, row in zip(expected_items, rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"item_id", "variant_scores"}
            or row.get("item_id") != expected_item
        ):
            raise AveritecP1CoordinateError("coordinate result item drifted")
        scores = row.get("variant_scores")
        if (
            not isinstance(scores, Mapping)
            or set(scores) != set(QUERY_VARIANT_IDS)
        ):
            raise AveritecP1CoordinateError("coordinate result variants drifted")
        for variant in QUERY_VARIANT_IDS:
            values = scores.get(variant)
            if (
                not isinstance(values, list)
                or len(values) != len(documents)
                or any(
                    type(coordinate) is not int
                    or not 0 <= coordinate <= SCALE
                    for coordinate in values
                )
            ):
                raise AveritecP1CoordinateError(
                    "coordinate result row width or value drifted"
                )
    normalized["self_sha256"] = self_sha256
    return normalized


def _require_native_thread_environment() -> None:
    if any(
        os.environ.get(key) != "1"
        for key in NATIVE_THREAD_ENVIRONMENT_KEYS
    ) or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise AveritecP1CoordinateError(
            "native thread environment drifted"
        )


def _production_encoder(model_root: Path):
    _require_native_thread_environment()
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise AveritecP1CoordinateError("torch thread state drifted")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise AveritecP1CoordinateError(
            "torch deterministic algorithm state drifted"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AveritecP1CoordinateError(
            "worker does not have exactly one visible CUDA device"
        )
    if torch.cuda.current_device() != 0:
        raise AveritecP1CoordinateError("logical CUDA device drifted")
    sentinel = torch.tensor([17.0], device="cuda:0")
    if float(sentinel.item()) != 17.0:
        raise AveritecP1CoordinateError("CUDA sentinel drifted")
    torch.cuda.synchronize(0)
    model = SentenceTransformer(str(model_root), device="cuda")
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    parameters = list(model.parameters())
    if not parameters or any(
        parameter.device.type != "cuda" or parameter.device.index != 0
        for parameter in parameters
    ):
        raise AveritecP1CoordinateError(
            "MiniLM parameters are not entirely on logical cuda:0"
        )

    def encode(texts: Sequence[str]) -> object:
        return model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    receipt = {
        "cuda_allocate_and_synchronize": True,
        "cuda_device_count": torch.cuda.device_count(),
        "deterministic_algorithms_enabled": True,
        "cuda_logical_device": 0,
        "minilm_all_parameters_cuda0": True,
        "minilm_parameter_count": len(parameters),
        "native_and_torch_thread_count": 1,
        "torch_manual_seed": 0,
    }
    return encode, receipt


def _load_private(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AveritecP1CoordinateError("coordinate input is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AveritecP1CoordinateError("coordinate input cannot be read") from exc
    if (
        not isinstance(value, dict)
        or raw != canonical_bytes(value)
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise AveritecP1CoordinateError(
            "coordinate input is not canonical private JSON"
        )
    return value


def read_private_output(
    path: Path,
    *,
    expected_input: Mapping[str, object],
) -> dict[str, object]:
    value = _load_private(path)
    return validate_output(value, expected_input=expected_input)


def _write_private(path: Path, value: Mapping[str, object]) -> None:
    raw = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AveritecP1CoordinateError(
            "coordinate output could not be written once"
        ) from exc
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise AveritecP1CoordinateError("coordinate output mode drifted")


def run_worker(*, input_path: Path, output_path: Path, model_root: Path) -> None:
    private_input = _load_private(input_path)
    encode, runtime_receipt = _production_encoder(model_root)
    output = coordinate_output(
        private_input=private_input,
        encode=encode,
        runtime_receipt=runtime_receipt,
    )
    _write_private(output_path, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    run_worker(
        input_path=arguments.input,
        output_path=arguments.output,
        model_root=arguments.model_root,
    )
    print(
        json.dumps(
            {"schema": OUTPUT_SCHEMA, "status": "completed"},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

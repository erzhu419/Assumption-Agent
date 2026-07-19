"""Offline feature runtime for the EntailmentBank G1/E1 study."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import entailmentbank_proof_retrieval_core_v1 as core
from replication_runtime.qasc_nli_v1 import binding as qasc_nli_binding
from replication_runtime.qasc_nli_v1.contract import NLIPair, encode_request
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


VERSION = "entailmentbank_proof_retrieval_runtime_v1"
NLI_WORKER_COUNT = 2
NLI_MODEL_RELATIVE_PATH = Path("artifacts/qasc_nli_runtime_v3/model")
NLI_ASSET_RELATIVE_PATH = Path("manifests/qasc_nli_runtime_asset_v1.json")
MINILM_MODEL_RELATIVE_PATH = Path("artifacts/qasper_minilm_runtime_v1/model")
MINILM_ASSET_RELATIVE_PATH = Path("manifests/qasper_minilm_runtime_asset_v1.json")


class EntailmentBankRuntimeError(RuntimeError):
    """An offline model, private pack, feature, or tensor seal drifted."""


class MiniLMEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class NLIItemScorer(Protocol):
    def score_items(
        self, items: Sequence[tuple[str, Sequence[NLIPair]]]
    ) -> Mapping[str, tuple[int, ...]]: ...


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EntailmentBankRuntimeError(f"{field} is not an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EntailmentBankRuntimeError(f"{field} is not a sequence")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EntailmentBankRuntimeError(f"{field} is not an integer")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise EntailmentBankRuntimeError(f"{field} is not exact text")
    return value


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntailmentBankRuntimeError(f"{label} is unreadable") from exc
    if acquisition.canonical_json_bytes(value) + b"\n" != raw:
        raise EntailmentBankRuntimeError(f"{label} is not canonical JSON")
    return _mapping(value, label)


def decode_view_pack(value: Mapping[str, Any], *, block: str) -> tuple[core.LabelFreeItem, ...]:
    if block not in acquisition.BLOCK_ORDER:
        raise EntailmentBankRuntimeError("view block is invalid")
    expected_keys = {
        "schema",
        "block",
        "source_split",
        "item_count",
        "items",
        "excluded_fields",
        "pack_sha256",
    }
    if set(value) != expected_keys:
        raise EntailmentBankRuntimeError("view pack keys drifted")
    try:
        acquisition.verify_self_hash(value, "pack_sha256")
    except acquisition.EntailmentBankAcquisitionError as exc:
        raise EntailmentBankRuntimeError("view pack self-hash drifted") from exc
    source_split = "dev" if block == "M_search" else "train"
    rows = _sequence(value.get("items"), "view items")
    excluded = [
        "proof",
        "meta.distractors",
        "meta.intermediate_conclusions",
        "gold_leaf_IDs",
        "family",
        "source_item_ID",
    ]
    if (
        value.get("schema") != f"{acquisition.VERSION}_block_view"
        or value.get("block") != block
        or value.get("source_split") != source_split
        or value.get("item_count") != acquisition.BLOCK_COUNTS[block]
        or len(rows) != acquisition.BLOCK_COUNTS[block]
        or value.get("excluded_fields") != excluded
    ):
        raise EntailmentBankRuntimeError("view pack envelope drifted")
    items: list[core.LabelFreeItem] = []
    for ordinal, raw_row in enumerate(rows):
        row = _mapping(raw_row, "view item")
        if set(row) != {
            "ordinal",
            "item_commitment_sha256",
            "question",
            "answer",
            "hypothesis",
            "node_texts",
        } or row.get("ordinal") != ordinal:
            raise EntailmentBankRuntimeError("view item keys or ordinal drifted")
        nodes = tuple(_text(text, "node text") for text in _sequence(row.get("node_texts"), "nodes"))
        try:
            items.append(
                core.LabelFreeItem(
                    _text(row.get("item_commitment_sha256"), "item commitment"),
                    _text(row.get("question"), "question"),
                    _text(row.get("answer"), "answer"),
                    _text(row.get("hypothesis"), "hypothesis"),
                    nodes,
                )
            )
        except core.EntailmentBankCoreError as exc:
            raise EntailmentBankRuntimeError("view item is invalid") from exc
    if len({item.item_commitment_sha256 for item in items}) != len(items):
        raise EntailmentBankRuntimeError("view commitments are duplicated")
    return tuple(items)


def decode_label_pack(value: Mapping[str, Any], *, block: str) -> tuple[core.ItemLabel, ...]:
    if block not in acquisition.BLOCK_ORDER or block == "F_search":
        raise EntailmentBankRuntimeError("label block is invalid")
    if set(value) != {
        "schema",
        "block",
        "source_split",
        "item_count",
        "items",
        "pack_sha256",
    }:
        raise EntailmentBankRuntimeError("label pack keys drifted")
    try:
        acquisition.verify_self_hash(value, "pack_sha256")
    except acquisition.EntailmentBankAcquisitionError as exc:
        raise EntailmentBankRuntimeError("label pack self-hash drifted") from exc
    source_split = "dev" if block == "M_search" else "train"
    rows = _sequence(value.get("items"), "label items")
    if (
        value.get("schema") != f"{acquisition.VERSION}_block_labels"
        or value.get("block") != block
        or value.get("source_split") != source_split
        or value.get("item_count") != acquisition.BLOCK_COUNTS[block]
        or len(rows) != acquisition.BLOCK_COUNTS[block]
    ):
        raise EntailmentBankRuntimeError("label pack envelope drifted")
    labels: list[core.ItemLabel] = []
    for ordinal, raw_row in enumerate(rows):
        row = _mapping(raw_row, "label item")
        if set(row) != {
            "ordinal",
            "item_commitment_sha256",
            "family",
            "gold_ordinals",
        } or row.get("ordinal") != ordinal:
            raise EntailmentBankRuntimeError("label item keys or ordinal drifted")
        try:
            labels.append(
                core.ItemLabel(
                    _text(row.get("item_commitment_sha256"), "label commitment"),
                    _text(row.get("family"), "family"),
                    tuple(
                        _integer(value, "gold ordinal")
                        for value in _sequence(row.get("gold_ordinals"), "gold ordinals")
                    ),
                )
            )
        except core.EntailmentBankCoreError as exc:
            raise EntailmentBankRuntimeError("label item is invalid") from exc
    if len({label.item_commitment_sha256 for label in labels}) != len(labels):
        raise EntailmentBankRuntimeError("label commitments are duplicated")
    return tuple(labels)


def load_view_pack(path: Path, *, block: str) -> tuple[core.LabelFreeItem, ...]:
    return decode_view_pack(_read_json(path, label=f"{block} view"), block=block)


def load_label_pack(path: Path, *, block: str) -> tuple[core.ItemLabel, ...]:
    return decode_label_pack(_read_json(path, label=f"{block} labels"), block=block)


class LocalTwoWorkerNLIPool:
    """Exactly two persistent offline NLI worker processes, with no QASC design check."""

    def __init__(
        self,
        *,
        project_root: Path,
        runtime_python: str | Path = sys.executable,
        workers: int = NLI_WORKER_COUNT,
    ) -> None:
        if isinstance(workers, bool) or workers != NLI_WORKER_COUNT:
            raise EntailmentBankRuntimeError("EntailmentBank requires exactly two NLI workers")
        root = project_root.resolve()
        manifest = root / NLI_ASSET_RELATIVE_PATH
        model = root / NLI_MODEL_RELATIVE_PATH
        live_workers = []
        try:
            self.runtime_binding = qasc_nli_binding.verify_runtime_binding(
                asset_manifest_path=manifest, model_root=model
            )
            for _ in range(NLI_WORKER_COUNT):
                live_workers.append(
                    qasc_nli_binding._PersistentWorker(  # type: ignore[attr-defined]
                    runtime_python=runtime_python,
                    asset_manifest_path=manifest,
                    model_root=model,
                    project_root=root,
                )
                )
            self._workers = tuple(live_workers)
        except Exception as exc:
            for worker in live_workers:
                worker.close()
            raise EntailmentBankRuntimeError("offline NLI workers failed to initialize") from exc
        self._closed = False

    def score_items(
        self, items: Sequence[tuple[str, Sequence[NLIPair]]]
    ) -> Mapping[str, tuple[int, ...]]:
        if self._closed or not items:
            raise EntailmentBankRuntimeError("NLI item batch is empty or pool is closed")
        seen: set[str] = set()
        normalized: list[tuple[str, tuple[NLIPair, ...]]] = []
        for key, pairs in items:
            if not isinstance(key, str) or not key or key in seen:
                raise EntailmentBankRuntimeError("NLI item keys are invalid")
            validated_raw = encode_request(pairs)
            from replication_runtime.qasc_nli_v1.contract import decode_request

            validated = decode_request(validated_raw)
            normalized.append((key, validated))
            seen.add(key)
        partitions = (normalized[::2], normalized[1::2])

        def run_partition(worker_index: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
            partition = partitions[worker_index]
            if not partition:
                return ()
            flat = tuple(pair for _key, pairs in partition for pair in pairs)
            request = encode_request(flat)
            scores = self._workers[worker_index].score(request, expected_count=len(flat))
            output = []
            cursor = 0
            for key, pairs in partition:
                output.append((key, tuple(scores[cursor : cursor + len(pairs)])))
                cursor += len(pairs)
            if cursor != len(scores):
                raise EntailmentBankRuntimeError("NLI partition reconstruction drifted")
            return tuple(output)

        try:
            with ThreadPoolExecutor(max_workers=NLI_WORKER_COUNT) as executor:
                futures = [executor.submit(run_partition, index) for index in range(NLI_WORKER_COUNT)]
                rows = tuple(row for future in futures for row in future.result())
        except Exception as exc:
            self.close()
            raise EntailmentBankRuntimeError("offline NLI scoring failed closed") from exc
        result = dict(rows)
        if set(result) != seen:
            raise EntailmentBankRuntimeError("NLI result key registry drifted")
        return result

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for worker in getattr(self, "_workers", ()):
            worker.close()

    def __enter__(self) -> "LocalTwoWorkerNLIPool":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def create_minilm_encoder(project_root: Path) -> minilm_binding.OfflineMiniLMEncoder:
    root = project_root.resolve()
    return minilm_binding.OfflineMiniLMEncoder(
        asset_manifest_path=root / MINILM_ASSET_RELATIVE_PATH,
        model_root=root / MINILM_MODEL_RELATIVE_PATH,
    )


def build_item_tensors(
    items: Sequence[core.LabelFreeItem],
    *,
    minilm_encoder: MiniLMEncoder,
    nli_scorer: NLIItemScorer,
) -> tuple[core.ItemTensor, ...]:
    rows = tuple(items)
    if not rows or len({item.item_commitment_sha256 for item in rows}) != len(rows):
        raise EntailmentBankRuntimeError("feature item registry is empty or duplicated")
    embedding_texts = tuple(
        text
        for item in rows
        for text in (item.hypothesis, item.answer_query, *item.node_texts)
    )
    try:
        embeddings = np.asarray(minilm_encoder.encode(embedding_texts), dtype=np.float32)
    except Exception as exc:
        raise EntailmentBankRuntimeError("MiniLM block encoding failed") from exc
    expected_shape = (27 * len(rows), minilm_binding.EMBEDDING_DIMENSION)
    if embeddings.shape != expected_shape or not np.isfinite(embeddings).all():
        raise EntailmentBankRuntimeError("MiniLM block matrix shape drifted")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
        raise EntailmentBankRuntimeError("MiniLM block matrix is not normalized")
    nli_items = tuple(
        (
            item.item_commitment_sha256,
            tuple(
                NLIPair(node, item.hypothesis) for node in item.node_texts
            )
            + tuple(NLIPair(node, item.answer_query) for node in item.node_texts),
        )
        for item in rows
    )
    try:
        nli_by_item = nli_scorer.score_items(nli_items)
    except Exception as exc:
        raise EntailmentBankRuntimeError("NLI block scoring failed") from exc
    if set(nli_by_item) != {item.item_commitment_sha256 for item in rows}:
        raise EntailmentBankRuntimeError("NLI item registry drifted")
    tensors: list[core.ItemTensor] = []
    for item_index, item in enumerate(rows):
        offset = item_index * 27
        hypothesis_embedding = embeddings[offset]
        answer_embedding = embeddings[offset + 1]
        node_embeddings = embeddings[offset + 2 : offset + 27]
        nli_scores = tuple(nli_by_item[item.item_commitment_sha256])
        if len(nli_scores) != 50 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in nli_scores
        ):
            raise EntailmentBankRuntimeError("NLI score vector drifted")
        pair = core.build_pair_token_f1(item.node_texts)
        node_features = []
        for ordinal, node_text in enumerate(item.node_texts):
            try:
                minilm_hypothesis = minilm_binding.quantized_cosine_similarity(
                    node_embeddings[ordinal], hypothesis_embedding
                )
                minilm_answer = minilm_binding.quantized_cosine_similarity(
                    node_embeddings[ordinal], answer_embedding
                )
            except Exception as exc:
                raise EntailmentBankRuntimeError("MiniLM cosine failed") from exc
            mean_pair = int(
                round(
                    math.fsum(
                        pair[ordinal][other] for other in range(25) if other != ordinal
                    )
                    / 24
                )
            )
            node_features.append(
                (
                    nli_scores[ordinal],
                    nli_scores[25 + ordinal],
                    minilm_hypothesis,
                    minilm_answer,
                    core.token_f1(node_text, item.hypothesis),
                    core.token_f1(node_text, item.question),
                    core.token_f1(node_text, item.answer),
                    mean_pair,
                )
            )
        try:
            tensors.append(
                core.ItemTensor(
                    item.item_commitment_sha256,
                    tuple(node_features),
                    pair,
                )
            )
        except core.EntailmentBankCoreError as exc:
            raise EntailmentBankRuntimeError("item tensor validation failed") from exc
    return tuple(tensors)


def tensor_pack(block: str, tensors: Sequence[core.ItemTensor]) -> Mapping[str, Any]:
    if block not in acquisition.BLOCK_ORDER or len(tensors) != acquisition.BLOCK_COUNTS[block]:
        raise EntailmentBankRuntimeError("tensor block shape drifted")
    if len({tensor.item_commitment_sha256 for tensor in tensors}) != len(tensors):
        raise EntailmentBankRuntimeError("tensor commitments are duplicated")
    body = {
        "schema": f"{VERSION}_tensor_pack",
        "block": block,
        "item_count": len(tensors),
        "node_feature_count": core.NODE_FEATURE_COUNT,
        "items": [
            {
                "ordinal": ordinal,
                "item_commitment_sha256": tensor.item_commitment_sha256,
                "node_features": [list(row) for row in tensor.node_features],
                "pair_token_f1": [list(row) for row in tensor.pair_token_f1],
            }
            for ordinal, tensor in enumerate(tensors)
        ],
    }
    return acquisition.self_hashed(body, "tensor_pack_sha256")


def decode_tensor_pack(value: Mapping[str, Any], *, block: str) -> tuple[core.ItemTensor, ...]:
    if set(value) != {
        "schema",
        "block",
        "item_count",
        "node_feature_count",
        "items",
        "tensor_pack_sha256",
    }:
        raise EntailmentBankRuntimeError("tensor pack keys drifted")
    try:
        acquisition.verify_self_hash(value, "tensor_pack_sha256")
    except acquisition.EntailmentBankAcquisitionError as exc:
        raise EntailmentBankRuntimeError("tensor pack hash drifted") from exc
    rows = _sequence(value.get("items"), "tensor items")
    if (
        value.get("schema") != f"{VERSION}_tensor_pack"
        or value.get("block") != block
        or value.get("item_count") != acquisition.BLOCK_COUNTS.get(block)
        or value.get("node_feature_count") != core.NODE_FEATURE_COUNT
        or len(rows) != acquisition.BLOCK_COUNTS.get(block)
    ):
        raise EntailmentBankRuntimeError("tensor pack envelope drifted")
    tensors = []
    for ordinal, raw_row in enumerate(rows):
        row = _mapping(raw_row, "tensor item")
        if set(row) != {
            "ordinal",
            "item_commitment_sha256",
            "node_features",
            "pair_token_f1",
        } or row.get("ordinal") != ordinal:
            raise EntailmentBankRuntimeError("tensor item keys drifted")
        try:
            tensors.append(
                core.ItemTensor(
                    _text(row.get("item_commitment_sha256"), "tensor commitment"),
                    tuple(
                        tuple(_integer(value, "node feature") for value in _sequence(feature_row, "feature row"))
                        for feature_row in _sequence(row.get("node_features"), "node features")
                    ),
                    tuple(
                        tuple(_integer(value, "pair value") for value in _sequence(pair_row, "pair row"))
                        for pair_row in _sequence(row.get("pair_token_f1"), "pair matrix")
                    ),
                )
            )
        except core.EntailmentBankCoreError as exc:
            raise EntailmentBankRuntimeError("tensor item is invalid") from exc
    return tuple(tensors)


def write_private_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise EntailmentBankRuntimeError("private output already exists")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    raw = acquisition.canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "EntailmentBankRuntimeError",
    "LocalTwoWorkerNLIPool",
    "NLI_WORKER_COUNT",
    "build_item_tensors",
    "create_minilm_encoder",
    "decode_label_pack",
    "decode_tensor_pack",
    "decode_view_pack",
    "load_label_pack",
    "load_view_pack",
    "tensor_pack",
    "write_private_json",
]

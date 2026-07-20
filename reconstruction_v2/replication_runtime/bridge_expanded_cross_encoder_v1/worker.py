"""Deterministic GPU worker for variable-size P10 candidate pools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .contract import (
    SCORE_SCALE,
    BridgeExpandedCrossEncoderError,
    canonical_json_bytes,
    output_item,
    output_payload,
    parse_input,
)


BATCH_SIZE = 64
MAXIMUM_SEQUENCE_LENGTH = 512
EXPECTED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}


def _validate_environment() -> None:
    for key, expected in EXPECTED_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise BridgeExpandedCrossEncoderError(f"{key} drifted")


def _model_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise BridgeExpandedCrossEncoderError("model root is unavailable")
    required = {
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
    if {entry.name for entry in path.iterdir() if entry.is_file()} != required:
        raise BridgeExpandedCrossEncoderError("model file set drifted")
    if any(entry.is_symlink() for entry in path.iterdir()):
        raise BridgeExpandedCrossEncoderError("model root contains a symlink")
    return path


def _write_output(path: Path, payload: Mapping[str, object]) -> None:
    raw = canonical_json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def run(*, input_path: Path, output_path: Path, model_root: Path) -> dict[str, object]:
    _validate_environment()
    if input_path.is_symlink() or not input_path.is_file():
        raise BridgeExpandedCrossEncoderError("input is unavailable")
    if output_path.exists() or output_path.is_symlink():
        raise BridgeExpandedCrossEncoderError("output already exists")
    items = parse_input(input_path.read_bytes())
    root = _model_root(model_root)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise BridgeExpandedCrossEncoderError("CUDA is unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        root,
        local_files_only=True,
        use_safetensors=True,
    ).eval().cuda()
    if model.__class__.__name__ != "BertForSequenceClassification" or model.num_labels != 1:
        raise BridgeExpandedCrossEncoderError("model architecture drifted")

    queries: list[str] = []
    passages: list[str] = []
    item_slices: list[tuple[int, int, int]] = []
    for item in items:
        start = len(queries)
        queries.extend([item.relation_query] * len(item.documents))
        passages.extend(document.content for document in item.documents)
        middle = len(queries)
        queries.extend([item.mechanism_query] * len(item.documents))
        passages.extend(document.content for document in item.documents)
        item_slices.append((start, middle, len(queries)))

    raw_scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(queries), BATCH_SIZE):
            encoded = tokenizer(
                queries[start : start + BATCH_SIZE],
                passages[start : start + BATCH_SIZE],
                max_length=MAXIMUM_SEQUENCE_LENGTH,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
            encoded = {key: value.cuda() for key, value in encoded.items()}
            logits = model(**encoded).logits
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise BridgeExpandedCrossEncoderError("model output shape drifted")
            raw_scores.extend(float(value) for value in logits[:, 0].detach().cpu())
    if len(raw_scores) != len(queries):
        raise BridgeExpandedCrossEncoderError("score count drifted")
    rows: list[dict[str, object]] = []
    for item, (start, middle, end) in zip(items, item_slices):
        relation = [int(round(value * SCORE_SCALE)) for value in raw_scores[start:middle]]
        mechanism = [int(round(value * SCORE_SCALE)) for value in raw_scores[middle:end]]
        rows.append(
            output_item(
                ordinal=item.ordinal,
                relation_scores_quantized=relation,
                mechanism_scores_quantized=mechanism,
            )
        )
    payload = output_payload(rows)
    _write_output(output_path, payload)
    return {
        "batch_size": BATCH_SIZE,
        "input_item_count": len(items),
        "pair_count": len(raw_scores),
        "status": "passed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run(
        input_path=arguments.input,
        output_path=arguments.output,
        model_root=arguments.model,
    )
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent offline worker for deterministic integer NLI margins."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence

from .binding import (
    ASSET_SELF_SHA256,
    EXPECTED_EXECUTION,
    EXPECTED_LABELS,
    MODEL_ARCHITECTURE,
    verify_runtime_binding,
)
from .contract import (
    BATCH_SIZE,
    MAXIMUM_REQUEST_BYTES,
    MAXIMUM_SEQUENCE_LENGTH,
    NLIPair,
    QASCNLIError,
    decode_request,
    encode_response,
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def canonical_canary_pairs() -> tuple[NLIPair, ...]:
    hypotheses = (
        "Specimen {i} is matter.",
        "Specimen {i} is not matter.",
        "Specimen {i} is a living organism.",
        "Matter includes specimen {i}.",
    )
    return tuple(
        NLIPair(
            premise=f"Specimen {index} is a mineral. Every mineral is matter.",
            hypothesis=hypotheses[index % 4].format(i=index),
        )
        for index in range(256)
    )


class DeterministicNLIScorer:
    """One-process scorer; instantiate only inside an independent worker."""

    def __init__(self, *, asset_manifest_path: str | Path, model_root: str | Path) -> None:
        os.environ.update(EXPECTED_EXECUTION["environment"])
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.runtime_binding = verify_runtime_binding(
            asset_manifest_path=asset_manifest_path,
            model_root=model_root,
        )
        model_path = Path(model_root).expanduser().absolute()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise QASCNLIError("frozen NLI runtime packages are unavailable") from exc
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        if torch.get_num_threads() != EXPECTED_EXECUTION["torch_num_threads"]:
            torch.set_num_threads(EXPECTED_EXECUTION["torch_num_threads"])
        if torch.get_num_interop_threads() != EXPECTED_EXECUTION["torch_interop_threads"]:
            try:
                torch.set_num_interop_threads(EXPECTED_EXECUTION["torch_interop_threads"])
            except RuntimeError as exc:
                raise QASCNLIError("Torch interop thread contract cannot be applied") from exc
        torch.manual_seed(EXPECTED_EXECUTION["torch_manual_seed"])
        torch.use_deterministic_algorithms(True)
        if (
            torch.get_num_threads() != 4
            or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()
        ):
            raise QASCNLIError("Torch deterministic execution contract drifted")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True,
                use_safetensors=True,
                dtype=torch.float32,
            ).to("cpu")
        except Exception as exc:
            raise QASCNLIError("offline safetensors model load failed") from exc
        self._model.eval()
        self._torch = torch
        if self._model.__class__.__name__ != MODEL_ARCHITECTURE:
            raise QASCNLIError("loaded model architecture drifted")
        actual_labels = {
            str(index): str(label).casefold()
            for index, label in self._model.config.id2label.items()
        }
        if actual_labels != EXPECTED_LABELS:
            raise QASCNLIError("loaded model label mapping drifted")
        if any(parameter.device.type != "cpu" for parameter in self._model.parameters()):
            raise QASCNLIError("loaded model escaped the CPU device contract")
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in self._model.parameters()
        ):
            raise QASCNLIError("loaded model dtype drifted")
        self.canary_receipt = self._verify_canary(asset_manifest_path)

    def _score_without_validation(self, pairs: Sequence[NLIPair]) -> tuple[int, ...]:
        scores: list[int] = []
        torch = self._torch
        with torch.inference_mode():
            for offset in range(0, len(pairs), BATCH_SIZE):
                batch = pairs[offset : offset + BATCH_SIZE]
                encoded = self._tokenizer(
                    [pair.premise for pair in batch],
                    [pair.hypothesis for pair in batch],
                    padding=True,
                    truncation=True,
                    max_length=MAXIMUM_SEQUENCE_LENGTH,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                if tuple(logits.shape) != (len(batch), 3) or not torch.isfinite(logits).all():
                    raise QASCNLIError("model returned malformed or non-finite logits")
                for row in logits:
                    contradiction = float(row[0])
                    entailment = float(row[1])
                    neutral = float(row[2])
                    margin = entailment - max(contradiction, neutral)
                    if not math.isfinite(margin):
                        raise QASCNLIError("model returned a non-finite NLI margin")
                    scores.append(int(round(margin * 1_000_000)))
        return tuple(scores)

    def _verify_canary(self, asset_manifest_path: str | Path) -> dict[str, object]:
        try:
            asset = json.loads(Path(asset_manifest_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QASCNLIError("canary asset manifest is unreadable") from exc
        if asset.get("asset_sha256") != ASSET_SELF_SHA256:
            raise QASCNLIError("canary asset binding drifted")
        canary = asset.get("deterministic_canary", {}).get("canonical_probe")
        if not isinstance(canary, dict):
            raise QASCNLIError("deterministic canary contract is unavailable")
        pairs = canonical_canary_pairs()
        pair_payload = [pair.as_payload() for pair in pairs]
        if (
            len(pairs) != canary.get("input_pair_count")
            or _canonical_hash(pair_payload) != canary.get("input_pairs_sha256")
        ):
            raise QASCNLIError("deterministic canary input contract drifted")
        first = self._score_without_validation(pairs)
        second = self._score_without_validation(pairs)
        vector_hash = _canonical_hash(list(first))
        if (
            first != second
            or canary.get("repeat_count") != 2
            or canary.get("repeat_exact") is not True
            or len(first) != canary.get("integer_score_vector_length")
            or vector_hash != canary.get("integer_score_vector_sha256")
            or min(first) != canary.get("observed_integer_score_minimum")
            or max(first) != canary.get("observed_integer_score_maximum")
        ):
            raise QASCNLIError("deterministic NLI canary drifted")
        return {
            "asset_sha256": ASSET_SELF_SHA256,
            "input_pairs_sha256": canary["input_pairs_sha256"],
            "integer_score_vector_sha256": vector_hash,
            "repeat_exact": True,
            "status": "passed",
        }

    def score(self, pairs: Sequence[NLIPair]) -> tuple[int, ...]:
        if not pairs:
            raise QASCNLIError("NLI pair batch is empty")
        return self._score_without_validation(pairs)


def _read_request(path: Path) -> tuple[NLIPair, ...]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAXIMUM_REQUEST_BYTES:
        raise QASCNLIError("worker input is unavailable or oversized")
    return decode_request(path.read_bytes())


def _write_response(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise QASCNLIError("worker output target is not fresh")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _serve_jsonl(scorer: DeterministicNLIScorer) -> int:
    for raw in sys.stdin.buffer:
        if len(raw) > MAXIMUM_REQUEST_BYTES:
            raise QASCNLIError("streaming NLI request is oversized")
        pairs = decode_request(raw)
        sys.stdout.buffer.write(encode_response(scorer.score(pairs)))
        sys.stdout.buffer.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--serve-jsonl", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    one_shot = arguments.input is not None or arguments.output is not None
    if arguments.serve_jsonl == one_shot:
        parser.error("choose exactly one of --serve-jsonl or --input/--output")
    if one_shot and (arguments.input is None or arguments.output is None):
        parser.error("--input and --output are required together")
    scorer = DeterministicNLIScorer(
        asset_manifest_path=arguments.asset_manifest,
        model_root=arguments.model_root,
    )
    if arguments.serve_jsonl:
        return _serve_jsonl(scorer)
    assert arguments.input is not None and arguments.output is not None
    pairs = _read_request(arguments.input)
    _write_response(arguments.output, encode_response(scorer.score(pairs)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QASCNLIError as exc:
        print(f"qasc_nli_v1 failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

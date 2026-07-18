"""Independent CPU-only worker for deterministic typed-entity extraction."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any, Iterator, Mapping, Sequence

from .binding import (
    ASSET_VERSION,
    EXPECTED_EXECUTION,
    EXPECTED_LABELS,
    INFERENCE_WINDOW_BATCH_SIZE,
    MAXIMUM_SEQUENCE_LENGTH,
    MODEL_ARCHITECTURE,
    MODEL_FILES,
    WINDOW_OVERLAP,
    _canonical_hash,
    _load_asset_manifest,
    _verify_package_versions,
    configure_offline_environment,
    verify_runtime_binding,
)
from .contract import (
    MAXIMUM_REQUEST_BYTES,
    CanonicalText,
    EntitySpan,
    MultiHopRAGNERError,
    decode_request,
    encode_response,
    synthetic_canary_inputs,
    validate_inputs,
)


def _blocked_network(*_args: object, **_kwargs: object) -> None:
    raise MultiHopRAGNERError("network access is forbidden in the offline NER runtime")


@contextmanager
def network_disabled() -> Iterator[None]:
    """Fail closed if Python code attempts network transport."""

    original_socket = socket.socket
    original_connection = socket.create_connection
    socket.socket = _blocked_network  # type: ignore[assignment]
    socket.create_connection = _blocked_network  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_connection  # type: ignore[assignment]


def _normalized_label_map(value: Mapping[object, object]) -> dict[str, str]:
    try:
        return {str(int(key)): str(label) for key, label in value.items()}
    except (TypeError, ValueError) as exc:
        raise MultiHopRAGNERError("loaded model label mapping is malformed") from exc


def _entity_label(label: str) -> tuple[str, str] | None:
    if label == "O":
        return None
    if len(label) < 3 or label[1] != "-":
        raise MultiHopRAGNERError("model emitted a malformed BIO label")
    prefix, entity_type = label[0], label[2:]
    if prefix not in {"B", "I"} or entity_type not in {"PER", "ORG", "LOC", "MISC"}:
        raise MultiHopRAGNERError("model emitted a label outside the frozen ontology")
    return prefix, entity_type


def merge_window_logits(
    *,
    text: str,
    window_offsets: Sequence[Sequence[Sequence[int]]],
    window_logits: Sequence[Sequence[Sequence[float]]],
    id2label: Mapping[str, str] = EXPECTED_LABELS,
) -> tuple[EntitySpan, ...]:
    """Pool overlaps per character, then apply the frozen deterministic BIO merge.

    Every character covered by a WordPiece receives the largest raw float32
    logit among all labels and all overlapping windows.  Exact ties choose the
    smaller label id, then the earlier window and token.  BIO merging operates
    on the resulting character winners, repairing an orphan ``I-X`` by starting
    a new ``X`` span and allowing a matching ``I-X`` to cross only uncovered
    whitespace.
    """

    if not isinstance(text, str) or not text:
        raise MultiHopRAGNERError("canonical text must be non-empty")
    if len(window_offsets) != len(window_logits) or not window_offsets:
        raise MultiHopRAGNERError("offset/logit window count mismatch")
    if id2label != EXPECTED_LABELS:
        raise MultiHopRAGNERError("NER label mapping drifted")

    # Winner tuple is (score, -label, -window, -token, label, start, end,
    # window, token).  The first four fields implement the frozen max order.
    winners: list[tuple[float, int, int, int, int, int, int, int, int] | None] = [
        None
    ] * len(text)
    for window_index, (offset_rows, logit_rows) in enumerate(
        zip(window_offsets, window_logits)
    ):
        if len(offset_rows) != len(logit_rows):
            raise MultiHopRAGNERError("offset/logit token count mismatch")
        for token_index, (raw_offset, raw_scores) in enumerate(
            zip(offset_rows, logit_rows)
        ):
            if (
                isinstance(raw_offset, (str, bytes))
                or not isinstance(raw_offset, Sequence)
                or len(raw_offset) != 2
            ):
                raise MultiHopRAGNERError("token offset is malformed")
            start, end = raw_offset
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
            ):
                raise MultiHopRAGNERError("token offset is not integral")
            if (start, end) == (0, 0):
                continue
            if not 0 <= start < end <= len(text):
                raise MultiHopRAGNERError("token offset is outside canonical text")
            if (
                isinstance(raw_scores, (str, bytes))
                or not isinstance(raw_scores, Sequence)
                or len(raw_scores) != len(EXPECTED_LABELS)
            ):
                raise MultiHopRAGNERError("token logit vector has the wrong shape")
            token_best: tuple[float, int] | None = None
            for label_id, raw_score in enumerate(raw_scores):
                if isinstance(raw_score, bool):
                    raise MultiHopRAGNERError("token logit is not numeric")
                try:
                    score = float(raw_score)
                except (TypeError, ValueError) as exc:
                    raise MultiHopRAGNERError("token logit is not numeric") from exc
                if not math.isfinite(score):
                    raise MultiHopRAGNERError("token logit is non-finite")
                candidate = (score, -label_id)
                if token_best is None or candidate > token_best:
                    token_best = candidate
            assert token_best is not None
            score, negative_label = token_best
            label_id = -negative_label
            candidate = (
                score,
                -label_id,
                -window_index,
                -token_index,
                label_id,
                start,
                end,
                window_index,
                token_index,
            )
            for character in range(start, end):
                current = winners[character]
                if current is None or candidate[:4] > current[:4]:
                    winners[character] = candidate

    # Collapse characters won by the same canonical token interval and label.
    # Window/token provenance remains in the key so an overlap boundary is
    # explicit even when its predicted label happens to be equal.
    runs: list[
        tuple[
            int,
            int,
            tuple[float, int, int, int, int, int, int, int, int] | None,
        ]
    ] = []
    run_start = 0
    run_value = winners[0]
    for position in range(1, len(winners) + 1):
        value = None if position == len(winners) else winners[position]
        if position == len(winners) or value != run_value:
            runs.append((run_start, position, run_value))
            if position < len(winners):
                run_start = position
                run_value = value

    spans: list[EntitySpan] = []
    active_type: str | None = None
    active_start = -1
    active_end = -1

    def flush() -> None:
        nonlocal active_type, active_start, active_end
        if active_type is not None:
            span_text = text[active_start:active_end]
            if not span_text.strip():
                raise MultiHopRAGNERError("BIO merge produced an empty entity")
            spans.append(
                EntitySpan(
                    entity_type=active_type,
                    start=active_start,
                    end=active_end,
                    text=span_text,
                )
            )
        active_type = None
        active_start = active_end = -1

    for start, end, winner in runs:
        label = "O" if winner is None else id2label[str(winner[4])]
        typed = _entity_label(label)
        if typed is None:
            # Only whitespace not covered by any WordPiece is provisionally
            # bridgeable.  A token-covered O run is an explicit BIO boundary.
            if winner is not None or (text[start:end] and not text[start:end].isspace()):
                flush()
            continue
        prefix, entity_type = typed
        if (
            prefix == "I"
            and active_type == entity_type
            and (active_end == start or text[active_end:start].isspace())
        ):
            active_end = end
            continue
        # B always starts.  A mismatched/orphan I deterministically starts too.
        flush()
        active_type = entity_type
        active_start = start
        active_end = end
    flush()
    if any(left.end > right.start for left, right in zip(spans, spans[1:])):
        raise MultiHopRAGNERError("BIO merge produced overlapping spans")
    return tuple(spans)


def _nested_rows(value: object, field: str) -> list[list[Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise MultiHopRAGNERError(f"tokenizer {field} is malformed")
    rows = list(value)
    if rows and isinstance(rows[0], int) and not isinstance(rows[0], bool):
        rows = [rows]
    normalized: list[list[Any]] = []
    for row in rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise MultiHopRAGNERError(f"tokenizer {field} row is malformed")
        normalized.append(list(row))
    return normalized


def tokenize_windows(
    tokenizer: object, text: str
) -> tuple[dict[str, list[list[int]]], list[list[list[int]]]]:
    """Apply the exact 512 WordPiece / 64 overlap tokenizer contract."""

    call = getattr(tokenizer, "__call__", None)
    if not callable(call):
        raise MultiHopRAGNERError("tokenizer is not callable")
    encoded = call(
        text,
        add_special_tokens=True,
        max_length=MAXIMUM_SEQUENCE_LENGTH,
        padding="max_length",
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        stride=WINDOW_OVERLAP,
        truncation=True,
    )
    if not isinstance(encoded, Mapping):
        raise MultiHopRAGNERError("tokenizer output is malformed")
    offsets_raw = _nested_rows(encoded.get("offset_mapping"), "offset_mapping")
    input_ids_raw = _nested_rows(encoded.get("input_ids"), "input_ids")
    masks_raw = _nested_rows(encoded.get("attention_mask"), "attention_mask")
    if not len(offsets_raw) == len(input_ids_raw) == len(masks_raw):
        raise MultiHopRAGNERError("tokenizer window counts disagree")
    model_inputs: dict[str, list[list[int]]] = {
        "attention_mask": [],
        "input_ids": [],
    }
    offsets: list[list[list[int]]] = []
    for raw_ids, raw_mask, raw_offsets in zip(input_ids_raw, masks_raw, offsets_raw):
        if not len(raw_ids) == len(raw_mask) == len(raw_offsets) == MAXIMUM_SEQUENCE_LENGTH:
            raise MultiHopRAGNERError("tokenizer did not emit exact 512-token windows")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids):
            raise MultiHopRAGNERError("tokenizer input_ids are malformed")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in (0, 1)
            for value in raw_mask
        ):
            raise MultiHopRAGNERError("tokenizer attention_mask is malformed")
        parsed_offsets: list[list[int]] = []
        for raw_offset in raw_offsets:
            if (
                isinstance(raw_offset, (str, bytes))
                or not isinstance(raw_offset, Sequence)
                or len(raw_offset) != 2
            ):
                raise MultiHopRAGNERError("tokenizer offset is malformed")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_offset
            ):
                raise MultiHopRAGNERError("tokenizer offset is not integral")
            parsed_offsets.append([raw_offset[0], raw_offset[1]])
        model_inputs["input_ids"].append([int(value) for value in raw_ids])
        model_inputs["attention_mask"].append([int(value) for value in raw_mask])
        offsets.append(parsed_offsets)
    token_types = encoded.get("token_type_ids")
    if token_types is not None:
        token_type_rows = _nested_rows(token_types, "token_type_ids")
        if len(token_type_rows) != len(offsets) or any(
            len(row) != MAXIMUM_SEQUENCE_LENGTH for row in token_type_rows
        ):
            raise MultiHopRAGNERError("tokenizer token_type_ids are malformed")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for row in token_type_rows
            for value in row
        ):
            raise MultiHopRAGNERError("tokenizer token_type_ids are malformed")
        model_inputs["token_type_ids"] = [
            [int(value) for value in row] for row in token_type_rows
        ]
    return model_inputs, offsets


class FrozenNERExtractor:
    """Verified local dslim BERT-NER extractor with a startup canary."""

    def __init__(self, *, asset_manifest_path: str | Path, model_root: str | Path) -> None:
        configure_offline_environment()
        self.runtime_binding = verify_runtime_binding(
            asset_manifest_path=asset_manifest_path, model_root=model_root
        )
        with network_disabled():
            self._load_backend(Path(model_root))
            self.canary_receipt = verify_synthetic_canary(
                self, asset_manifest_path=asset_manifest_path
            )

    def _load_backend(self, model_root: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise MultiHopRAGNERError("frozen NER runtime packages are unavailable") from exc
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        torch.set_num_threads(EXPECTED_EXECUTION["torch_num_threads"])
        try:
            torch.set_num_interop_threads(EXPECTED_EXECUTION["torch_interop_threads"])
        except RuntimeError:
            if torch.get_num_interop_threads() != EXPECTED_EXECUTION["torch_interop_threads"]:
                raise MultiHopRAGNERError(
                    "Torch interop thread contract cannot be applied"
                ) from None
        torch.manual_seed(EXPECTED_EXECUTION["torch_manual_seed"])
        torch.use_deterministic_algorithms(True)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_root),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
            self._model = AutoModelForTokenClassification.from_pretrained(
                str(model_root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
            ).to("cpu")
        except Exception as exc:
            raise MultiHopRAGNERError("offline safetensors NER model load failed") from exc
        self._model.eval()
        self._torch = torch
        if getattr(self._tokenizer, "is_fast", False) is not True:
            raise MultiHopRAGNERError("loaded tokenizer is not the required fast tokenizer")
        if self._model.__class__.__name__ != MODEL_ARCHITECTURE:
            raise MultiHopRAGNERError("loaded model architecture drifted")
        if _normalized_label_map(self._model.config.id2label) != EXPECTED_LABELS:
            raise MultiHopRAGNERError("loaded model label mapping drifted")
        if any(parameter.device.type != "cpu" for parameter in self._model.parameters()):
            raise MultiHopRAGNERError("loaded model escaped the CPU contract")
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in self._model.parameters()
        ):
            raise MultiHopRAGNERError("loaded model dtype drifted")
        if (
            torch.get_num_threads() != 1
            or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()
        ):
            raise MultiHopRAGNERError("Torch deterministic execution contract drifted")

    def _extract_text(self, text: str) -> tuple[EntitySpan, ...]:
        inputs, offsets = tokenize_windows(self._tokenizer, text)
        logits: list[list[list[float]]] = []
        torch = self._torch
        with network_disabled(), torch.inference_mode():
            for start in range(0, len(offsets), INFERENCE_WINDOW_BATCH_SIZE):
                end = min(start + INFERENCE_WINDOW_BATCH_SIZE, len(offsets))
                batch = {
                    key: torch.tensor(rows[start:end], dtype=torch.long, device="cpu")
                    for key, rows in inputs.items()
                }
                values = self._model(**batch).logits.detach().to(
                    device="cpu", dtype=torch.float32
                )
                if tuple(values.shape) != (
                    end - start,
                    MAXIMUM_SEQUENCE_LENGTH,
                    len(EXPECTED_LABELS),
                ) or not torch.isfinite(values).all():
                    raise MultiHopRAGNERError("model returned malformed logits")
                logits.extend(values.tolist())
        return merge_window_logits(
            text=text,
            window_offsets=offsets,
            window_logits=logits,
        )

    def extract_canonical(self, values: Sequence[CanonicalText]) -> tuple[tuple[EntitySpan, ...], ...]:
        if not values:
            raise MultiHopRAGNERError("canonical NER batch is empty")
        return tuple(self._extract_text(value.text) for value in values)

    def extract_inputs(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]:
        return self.extract_canonical(validate_inputs(values))


def compute_synthetic_canary(extractor: object) -> dict[str, object]:
    """Compute the manifest-ready row-free canary without requiring a manifest."""

    method = getattr(extractor, "extract_inputs", None)
    if not callable(method):
        raise MultiHopRAGNERError("canary extractor is malformed")
    inputs = synthetic_canary_inputs()
    first = method(inputs)
    second = method(inputs)
    if first != second:
        raise MultiHopRAGNERError("synthetic NER canary is not repeat-exact")
    payload = [[span.as_payload() for span in row] for row in first]
    return {
        "generator_version": "multihoprag_ner_synthetic_16_v1",
        "input_count": len(inputs),
        "input_sha256": _canonical_hash(list(inputs)),
        "multihoprag_rows_or_archives_accessed": False,
        "output_sha256": _canonical_hash(payload),
        "repeat_count": 2,
        "repeat_exact": True,
    }


def compute_preasset_canary(model_root: str | Path) -> dict[str, object]:
    """Load the exact local six-file tree to form the first asset canary.

    This entry point exists only for asset formation, before the public
    manifest can bind its own output hash.  It still freezes package versions,
    exact tree shape, CPU/determinism, local-only loading, and disabled sockets.
    """

    configure_offline_environment()
    _verify_package_versions()
    root = Path(model_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise MultiHopRAGNERError("preasset model root is unavailable or a symlink")
    entries = list(root.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or sorted(entry.name for entry in entries) != sorted(MODEL_FILES)
    ):
        raise MultiHopRAGNERError("preasset model tree is not the exact six-file tree")
    extractor = object.__new__(FrozenNERExtractor)
    with network_disabled():
        extractor._load_backend(root)
        receipt = compute_synthetic_canary(extractor)
    return {
        **receipt,
        "asset_version": ASSET_VERSION,
        "model_root": str(root),
        "network_calls": 0,
        "status": "passed_row_free_preasset_canary",
    }


def verify_synthetic_canary(
    extractor: FrozenNERExtractor, *, asset_manifest_path: str | Path
) -> dict[str, object]:
    _, asset = _load_asset_manifest(asset_manifest_path)
    expected = asset.get("deterministic_canary")
    if not isinstance(expected, Mapping):
        raise MultiHopRAGNERError("asset canary binding is unavailable")
    actual = compute_synthetic_canary(extractor)
    if actual != dict(expected):
        raise MultiHopRAGNERError("synthetic NER startup canary drifted")
    return {**actual, "status": "passed_exact_row_free_synthetic_canary"}


def _read_request(path: Path) -> tuple[CanonicalText, ...]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAXIMUM_REQUEST_BYTES:
        raise MultiHopRAGNERError("worker input is unavailable or oversized")
    return decode_request(path.read_bytes())


def _write_response(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise MultiHopRAGNERError("worker output target is not fresh")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _serve_jsonl(extractor: FrozenNERExtractor) -> int:
    for raw in sys.stdin.buffer:
        if len(raw) > MAXIMUM_REQUEST_BYTES:
            raise MultiHopRAGNERError("streaming NER request is oversized")
        values = decode_request(raw)
        sys.stdout.buffer.write(encode_response(extractor.extract_canonical(values)))
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
    extractor = FrozenNERExtractor(
        asset_manifest_path=arguments.asset_manifest, model_root=arguments.model_root
    )
    if arguments.serve_jsonl:
        return _serve_jsonl(extractor)
    assert arguments.input is not None and arguments.output is not None
    values = _read_request(arguments.input)
    _write_response(arguments.output, encode_response(extractor.extract_canonical(values)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MultiHopRAGNERError as exc:
        print(f"multihoprag_ner_v1 failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

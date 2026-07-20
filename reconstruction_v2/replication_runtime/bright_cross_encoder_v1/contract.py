"""Canonical label-free contract for BRIGHT relation/mechanism cross-encoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence


VERSION = "bright_cross_encoder_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
CANDIDATE_COUNT = 32
MAXIMUM_ITEM_COUNT = 64
MAXIMUM_QUERY_CHARACTERS = 1_000
MAXIMUM_DOCUMENT_CHARACTERS = 3_000
SCORE_SCALE = 1_000_000
_INPUT_KEYS = frozenset({"items", "schema"})
_INPUT_ITEM_KEYS = frozenset(
    {"documents", "mechanism_query", "ordinal", "relation_query"}
)
_DOCUMENT_KEYS = frozenset({"content", "ordinal"})
_OUTPUT_KEYS = frozenset({"items", "schema"})
_OUTPUT_ITEM_KEYS = frozenset(
    {"mean_logit_quantized", "ordinal", "ranked_ordinals"}
)


class BrightCrossEncoderError(RuntimeError):
    """The isolated cross-encoder contract failed closed."""


@dataclass(frozen=True)
class CandidateDocument:
    ordinal: int
    content: str


@dataclass(frozen=True)
class CrossEncoderItem:
    ordinal: int
    relation_query: str
    mechanism_query: str
    documents: tuple[CandidateDocument, ...]


def canonical_json_bytes(value: Any) -> bytes:
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
        raise BrightCrossEncoderError("value is not canonical JSON") from exc


def _text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BrightCrossEncoderError(f"{field} is invalid")
    return value


def validate_items(value: object) -> tuple[CrossEncoderItem, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BrightCrossEncoderError("items are not a sequence")
    if not 1 <= len(value) <= MAXIMUM_ITEM_COUNT:
        raise BrightCrossEncoderError("item count is outside the frozen bound")
    output: list[CrossEncoderItem] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _INPUT_ITEM_KEYS:
            raise BrightCrossEncoderError("input item shape drifted")
        ordinal = raw.get("ordinal")
        documents = raw.get("documents")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != position
            or isinstance(documents, (str, bytes))
            or not isinstance(documents, Sequence)
            or len(documents) != CANDIDATE_COUNT
        ):
            raise BrightCrossEncoderError("input item identity drifted")
        checked_documents: list[CandidateDocument] = []
        for document_position, document in enumerate(documents):
            if not isinstance(document, Mapping) or set(document) != _DOCUMENT_KEYS:
                raise BrightCrossEncoderError("document shape drifted")
            document_ordinal = document.get("ordinal")
            if (
                isinstance(document_ordinal, bool)
                or not isinstance(document_ordinal, int)
                or document_ordinal != document_position
            ):
                raise BrightCrossEncoderError("document ordinal drifted")
            checked_documents.append(
                CandidateDocument(
                    ordinal=document_ordinal,
                    content=_text(
                        document.get("content"),
                        f"items[{position}].documents[{document_position}].content",
                        MAXIMUM_DOCUMENT_CHARACTERS,
                    ),
                )
            )
        output.append(
            CrossEncoderItem(
                ordinal=ordinal,
                relation_query=_text(
                    raw.get("relation_query"),
                    f"items[{position}].relation_query",
                    MAXIMUM_QUERY_CHARACTERS,
                ),
                mechanism_query=_text(
                    raw.get("mechanism_query"),
                    f"items[{position}].mechanism_query",
                    MAXIMUM_QUERY_CHARACTERS,
                ),
                documents=tuple(checked_documents),
            )
        )
    return tuple(output)


def input_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    validate_items(rows)
    return {"items": rows, "schema": INPUT_SCHEMA}


def parse_input(raw: bytes) -> tuple[CrossEncoderItem, ...]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightCrossEncoderError("input is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != _INPUT_KEYS
        or value.get("schema") != INPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise BrightCrossEncoderError("input envelope drifted")
    return validate_items(value.get("items"))


def output_item(*, ordinal: int, mean_logit_quantized: Sequence[int]) -> dict[str, Any]:
    scores = tuple(mean_logit_quantized)
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or len(scores) != CANDIDATE_COUNT
        or any(isinstance(value, bool) or not isinstance(value, int) for value in scores)
    ):
        raise BrightCrossEncoderError("output scores are invalid")
    ranking = tuple(sorted(range(CANDIDATE_COUNT), key=lambda index: (-scores[index], index)))
    return {
        "mean_logit_quantized": list(scores),
        "ordinal": ordinal,
        "ranked_ordinals": list(ranking),
    }


def output_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    if not 1 <= len(rows) <= MAXIMUM_ITEM_COUNT:
        raise BrightCrossEncoderError("output item count is outside the frozen bound")
    checked: list[dict[str, Any]] = []
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _OUTPUT_ITEM_KEYS:
            raise BrightCrossEncoderError("output item shape drifted")
        if raw.get("ordinal") != position:
            raise BrightCrossEncoderError("output ordinal drifted")
        rebuilt = output_item(
            ordinal=position,
            mean_logit_quantized=raw.get("mean_logit_quantized", ()),
        )
        if rebuilt != dict(raw):
            raise BrightCrossEncoderError("output ranking drifted")
        checked.append(rebuilt)
    return {"items": checked, "schema": OUTPUT_SCHEMA}


def parse_output(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightCrossEncoderError("output is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != _OUTPUT_KEYS
        or value.get("schema") != OUTPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise BrightCrossEncoderError("output envelope drifted")
    return output_payload(value.get("items", ()))


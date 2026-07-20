"""Canonical label-free contract for P10 expanded-pool cross-encoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence


VERSION = "bridge_expanded_cross_encoder_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
MINIMUM_DOCUMENT_COUNT = 32
MAXIMUM_DOCUMENT_COUNT = 288
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
    {
        "document_count",
        "mechanism_scores_quantized",
        "ordinal",
        "relation_scores_quantized",
    }
)


class BridgeExpandedCrossEncoderError(RuntimeError):
    """The expanded cross-encoder contract failed closed."""


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
        raise BridgeExpandedCrossEncoderError("value is not canonical JSON") from exc


def _text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BridgeExpandedCrossEncoderError(f"{field} is invalid")
    return value


def validate_items(value: object) -> tuple[CrossEncoderItem, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BridgeExpandedCrossEncoderError("items are not a sequence")
    if not 1 <= len(value) <= MAXIMUM_ITEM_COUNT:
        raise BridgeExpandedCrossEncoderError("item count is outside the frozen bound")
    output: list[CrossEncoderItem] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _INPUT_ITEM_KEYS:
            raise BridgeExpandedCrossEncoderError("input item shape drifted")
        ordinal = raw.get("ordinal")
        documents = raw.get("documents")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != position
            or isinstance(documents, (str, bytes))
            or not isinstance(documents, Sequence)
            or not MINIMUM_DOCUMENT_COUNT <= len(documents) <= MAXIMUM_DOCUMENT_COUNT
        ):
            raise BridgeExpandedCrossEncoderError("input item identity drifted")
        checked_documents: list[CandidateDocument] = []
        for document_position, document in enumerate(documents):
            if not isinstance(document, Mapping) or set(document) != _DOCUMENT_KEYS:
                raise BridgeExpandedCrossEncoderError("document shape drifted")
            document_ordinal = document.get("ordinal")
            if (
                isinstance(document_ordinal, bool)
                or not isinstance(document_ordinal, int)
                or document_ordinal != document_position
            ):
                raise BridgeExpandedCrossEncoderError("document ordinal drifted")
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
        raise BridgeExpandedCrossEncoderError("input is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != _INPUT_KEYS
        or value.get("schema") != INPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise BridgeExpandedCrossEncoderError("input envelope drifted")
    return validate_items(value.get("items"))


def output_item(
    *,
    ordinal: int,
    relation_scores_quantized: Sequence[int],
    mechanism_scores_quantized: Sequence[int],
) -> dict[str, Any]:
    relation = tuple(relation_scores_quantized)
    mechanism = tuple(mechanism_scores_quantized)
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or len(relation) != len(mechanism)
        or not MINIMUM_DOCUMENT_COUNT <= len(relation) <= MAXIMUM_DOCUMENT_COUNT
        or any(
            isinstance(score, bool) or not isinstance(score, int)
            for score in (*relation, *mechanism)
        )
    ):
        raise BridgeExpandedCrossEncoderError("output scores are invalid")
    return {
        "document_count": len(relation),
        "mechanism_scores_quantized": list(mechanism),
        "ordinal": ordinal,
        "relation_scores_quantized": list(relation),
    }


def output_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    if not 1 <= len(rows) <= MAXIMUM_ITEM_COUNT:
        raise BridgeExpandedCrossEncoderError("output item count is outside the frozen bound")
    checked: list[dict[str, Any]] = []
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _OUTPUT_ITEM_KEYS:
            raise BridgeExpandedCrossEncoderError("output item shape drifted")
        if raw.get("ordinal") != position:
            raise BridgeExpandedCrossEncoderError("output ordinal drifted")
        rebuilt = output_item(
            ordinal=position,
            relation_scores_quantized=raw.get("relation_scores_quantized", ()),
            mechanism_scores_quantized=raw.get("mechanism_scores_quantized", ()),
        )
        if rebuilt != dict(raw) or rebuilt["document_count"] != raw.get("document_count"):
            raise BridgeExpandedCrossEncoderError("output values drifted")
        checked.append(rebuilt)
    return {"items": checked, "schema": OUTPUT_SCHEMA}


def parse_output(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeExpandedCrossEncoderError("output is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != _OUTPUT_KEYS
        or value.get("schema") != OUTPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise BridgeExpandedCrossEncoderError("output envelope drifted")
    return output_payload(value.get("items", ()))

"""Exact row-minimal wire contract for the frozen MultiHopRAG NER worker.

Only an article's title/body or a query string can cross this boundary.  Source
identifiers, URLs, dates, relation-family labels, answers, evidence annotations,
retrieval ranks, and evaluator outcomes are intentionally not representable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "multihoprag_ner_request_v1"
RESPONSE_SCHEMA = "multihoprag_ner_response_v1"
REQUEST_KEYS = frozenset({"schema", "texts"})
RESPONSE_KEYS = frozenset({"entities", "schema"})
ARTICLE_KEYS = frozenset({"body", "kind", "title"})
QUERY_KEYS = frozenset({"kind", "query"})
SPAN_KEYS = frozenset({"end", "entity_type", "start", "text"})
ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")

MAXIMUM_TEXTS_PER_REQUEST = 4_096
MAXIMUM_TITLE_CHARACTERS = 32_768
MAXIMUM_BODY_CHARACTERS = 2_000_000
MAXIMUM_QUERY_CHARACTERS = 131_072
MAXIMUM_TEXT_UTF8_BYTES = 8_000_000
MAXIMUM_REQUEST_BYTES = 64 * 1024 * 1024
MAXIMUM_RESPONSE_BYTES = 64 * 1024 * 1024


class MultiHopRAGNERError(RuntimeError):
    """Raised when the frozen offline NER boundary cannot be proven."""


@dataclass(frozen=True)
class CanonicalText:
    kind: str
    text: str


@dataclass(frozen=True)
class EntitySpan:
    entity_type: str
    start: int
    end: int
    text: str

    def as_payload(self) -> dict[str, object]:
        return {
            "end": self.end,
            "entity_type": self.entity_type,
            "start": self.start,
            "text": self.text,
        }


def canonical_json_line(value: object) -> bytes:
    """Return the only accepted JSON wire representation."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MultiHopRAGNERError("value is not canonical-JSON representable") from exc


def _reject_json_constant(value: str) -> None:
    raise MultiHopRAGNERError(f"non-finite JSON constant is forbidden: {value}")


def _decode_canonical_json_line(raw: bytes, *, maximum: int, field: str) -> Any:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        raise MultiHopRAGNERError(f"{field} byte size is outside the frozen bound")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise MultiHopRAGNERError(f"{field} is not one canonical JSON line")
    try:
        value = json.loads(raw.decode("ascii"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiHopRAGNERError(f"{field} is not valid canonical JSON") from exc
    if canonical_json_line(value) != raw:
        raise MultiHopRAGNERError(f"{field} is not canonical JSON")
    return value


def _required_text(
    value: object, *, maximum: int, field: str, allow_empty: bool = False
) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value)
        or "\x00" in value
    ):
        requirement = "NUL-free text" if allow_empty else "non-empty NUL-free text"
        raise MultiHopRAGNERError(f"{field} must be {requirement}")
    if len(value) > maximum:
        raise MultiHopRAGNERError(f"{field} exceeds the frozen character bound")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise MultiHopRAGNERError(f"{field} contains invalid Unicode") from exc
    if len(encoded) > MAXIMUM_TEXT_UTF8_BYTES:
        raise MultiHopRAGNERError(f"{field} exceeds the frozen UTF-8 bound")
    return value


def validate_inputs(values: object) -> tuple[CanonicalText, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MultiHopRAGNERError("texts must be a sequence")
    if not 1 <= len(values) <= MAXIMUM_TEXTS_PER_REQUEST:
        raise MultiHopRAGNERError("text count is outside the frozen request bound")
    normalized: list[CanonicalText] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise MultiHopRAGNERError(f"texts[{index}] must be an object")
        kind = raw.get("kind")
        if kind == "article" and set(raw) == ARTICLE_KEYS:
            title = _required_text(
                raw.get("title"), maximum=MAXIMUM_TITLE_CHARACTERS,
                field=f"texts[{index}].title",
            )
            body = _required_text(
                raw.get("body"), maximum=MAXIMUM_BODY_CHARACTERS,
                field=f"texts[{index}].body",
                allow_empty=True,
            )
            # This exact separator is a frozen part of the representation.
            text = title + "\n\n" + body
        elif kind == "query" and set(raw) == QUERY_KEYS:
            text = _required_text(
                raw.get("query"), maximum=MAXIMUM_QUERY_CHARACTERS,
                field=f"texts[{index}].query",
            )
        else:
            raise MultiHopRAGNERError(
                "each text must be exactly article{title,body} or query{query}"
            )
        normalized.append(CanonicalText(kind=str(kind), text=text))
    return tuple(normalized)


def request_payload(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    # Validation proves the exact input schema but the original fields are
    # serialized so the article boundary remains title + two LF + body.
    validate_inputs(values)
    return {"schema": REQUEST_SCHEMA, "texts": [dict(value) for value in values]}


def encode_request(values: Sequence[Mapping[str, object]]) -> bytes:
    return canonical_json_line(request_payload(values))


def decode_request(raw: bytes) -> tuple[CanonicalText, ...]:
    value = _decode_canonical_json_line(
        raw, maximum=MAXIMUM_REQUEST_BYTES, field="NER request"
    )
    if not isinstance(value, Mapping) or set(value) != REQUEST_KEYS:
        raise MultiHopRAGNERError("NER request envelope is not exact")
    if value.get("schema") != REQUEST_SCHEMA:
        raise MultiHopRAGNERError("NER request schema mismatch")
    return validate_inputs(value.get("texts"))


def validate_entity_rows(
    rows: object, *, canonical_texts: Sequence[str]
) -> tuple[tuple[EntitySpan, ...], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise MultiHopRAGNERError("entity rows must be a sequence")
    if len(rows) != len(canonical_texts):
        raise MultiHopRAGNERError("NER response text count mismatch")
    result: list[tuple[EntitySpan, ...]] = []
    for row_index, (raw_spans, source) in enumerate(zip(rows, canonical_texts)):
        if isinstance(raw_spans, (str, bytes)) or not isinstance(raw_spans, Sequence):
            raise MultiHopRAGNERError("each entity row must be a sequence")
        spans: list[EntitySpan] = []
        previous_end = -1
        for span_index, raw in enumerate(raw_spans):
            if not isinstance(raw, Mapping) or set(raw) != SPAN_KEYS:
                raise MultiHopRAGNERError("each entity span must have the exact key set")
            entity_type = raw.get("entity_type")
            start = raw.get("start")
            end = raw.get("end")
            text = raw.get("text")
            if entity_type not in ENTITY_TYPES:
                raise MultiHopRAGNERError("entity span type is outside the frozen ontology")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(source)
                or start < previous_end
            ):
                raise MultiHopRAGNERError("entity span offsets are malformed or overlap")
            if not isinstance(text, str) or text != source[start:end] or not text.strip():
                raise MultiHopRAGNERError("entity span text does not match exact offsets")
            spans.append(
                EntitySpan(
                    entity_type=str(entity_type), start=start, end=end, text=text
                )
            )
            previous_end = end
        result.append(tuple(spans))
    return tuple(result)


def encode_response(rows: Sequence[Sequence[EntitySpan]]) -> bytes:
    return canonical_json_line(
        {
            "entities": [[span.as_payload() for span in row] for row in rows],
            "schema": RESPONSE_SCHEMA,
        }
    )


def decode_response(
    raw: bytes, *, canonical_texts: Sequence[str]
) -> tuple[tuple[EntitySpan, ...], ...]:
    value = _decode_canonical_json_line(
        raw, maximum=MAXIMUM_RESPONSE_BYTES, field="NER response"
    )
    if not isinstance(value, Mapping) or set(value) != RESPONSE_KEYS:
        raise MultiHopRAGNERError("NER response envelope is not exact")
    if value.get("schema") != RESPONSE_SCHEMA:
        raise MultiHopRAGNERError("NER response schema mismatch")
    return validate_entity_rows(value.get("entities"), canonical_texts=canonical_texts)


def synthetic_canary_inputs() -> tuple[dict[str, str], ...]:
    """Return a fixed, public, benchmark-row-free canary preimage."""

    rows: list[dict[str, str]] = []
    for index in range(16):
        if index % 2 == 0:
            rows.append(
                {
                    "body": (
                        f"Researcher Ada_{index:02d} visited Northport_{index % 5:02d} "
                        f"for Synthetic_Lab_{(index * 3) % 7:02d}."
                    ),
                    "kind": "article",
                    "title": f"Synthetic bulletin {index:02d}",
                }
            )
        else:
            rows.append(
                {
                    "kind": "query",
                    "query": (
                        f"Which organization hosted Person_{index:02d} in "
                        f"Test_City_{(index * 5) % 9:02d}?"
                    ),
                }
            )
    # Keep generator changes visible even before the asset manifest exists.
    validate_inputs(rows)
    return tuple(rows)

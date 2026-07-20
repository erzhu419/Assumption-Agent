"""Canonical label-free contract for offline BRIGHT query expansion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


VERSION = "bright_query_generator_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
MAXIMUM_ITEM_COUNT = 64
MAXIMUM_QUERY_CHARACTERS = 24_000
MAXIMUM_EXPANSION_CHARACTERS = 1_000
MAXIMUM_COMPLETION_TOKENS = 192
EXPANSION_KEYS = (
    "entity_query",
    "relation_query",
    "mechanism_query",
    "constraint_query",
)
_INPUT_KEYS = frozenset({"items", "schema"})
_INPUT_ITEM_KEYS = frozenset({"ordinal", "query"})
_OUTPUT_KEYS = frozenset({"items", "schema"})
_OUTPUT_ITEM_KEYS = frozenset(
    {
        "completion_sha256",
        "completion_token_count",
        "expansions",
        "generation_valid",
        "ordinal",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class BrightQueryGeneratorError(RuntimeError):
    """The offline query-generator contract failed closed."""


@dataclass(frozen=True)
class QueryItem:
    ordinal: int
    query: str


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
        raise BrightQueryGeneratorError("value is not canonical JSON") from exc


def _text(value: object, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BrightQueryGeneratorError(f"{field} is invalid")
    return value


def validate_items(value: object) -> tuple[QueryItem, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BrightQueryGeneratorError("items are not a sequence")
    if not 1 <= len(value) <= MAXIMUM_ITEM_COUNT:
        raise BrightQueryGeneratorError("item count is outside the frozen bound")
    rows: list[QueryItem] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _INPUT_ITEM_KEYS:
            raise BrightQueryGeneratorError("input item shape drifted")
        ordinal = raw.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal != position:
            raise BrightQueryGeneratorError("input ordinals are not canonical")
        rows.append(
            QueryItem(
                ordinal=ordinal,
                query=_text(
                    raw.get("query"),
                    field=f"items[{position}].query",
                    maximum=MAXIMUM_QUERY_CHARACTERS,
                ),
            )
        )
    return tuple(rows)


def parse_input(raw: bytes) -> tuple[QueryItem, ...]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightQueryGeneratorError("input is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != _INPUT_KEYS
        or value.get("schema") != INPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise BrightQueryGeneratorError("input envelope drifted")
    return validate_items(value.get("items"))


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def parse_completion(completion: str, *, original_query: str) -> tuple[str, ...]:
    text = completion.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n") : -len("\n```")].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BrightQueryGeneratorError("completion is not the frozen JSON object") from exc
    if not isinstance(value, Mapping) or tuple(value.keys()) != EXPANSION_KEYS:
        raise BrightQueryGeneratorError("completion keys or order drifted")
    expansions = tuple(
        _text(value.get(key), field=key, maximum=MAXIMUM_EXPANSION_CHARACTERS)
        for key in EXPANSION_KEYS
    )
    normalized = [_normalize(original_query), *(_normalize(value) for value in expansions)]
    if len(set(normalized)) != len(normalized):
        raise BrightQueryGeneratorError("completion queries are not distinct")
    return expansions


def build_output_item(
    *, ordinal: int, completion: str, completion_token_count: int, query: str
) -> dict[str, Any]:
    if (
        isinstance(completion_token_count, bool)
        or not isinstance(completion_token_count, int)
        or not 0 <= completion_token_count <= MAXIMUM_COMPLETION_TOKENS
    ):
        raise BrightQueryGeneratorError("completion token count drifted")
    digest = hashlib.sha256(completion.encode("utf-8")).hexdigest()
    try:
        expansions = list(parse_completion(completion, original_query=query))
        valid = True
    except BrightQueryGeneratorError:
        expansions = []
        valid = False
    return {
        "completion_sha256": digest,
        "completion_token_count": completion_token_count,
        "expansions": expansions,
        "generation_valid": valid,
        "ordinal": ordinal,
    }


def output_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    if not 1 <= len(rows) <= MAXIMUM_ITEM_COUNT:
        raise BrightQueryGeneratorError("output item count drifted")
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _OUTPUT_ITEM_KEYS:
            raise BrightQueryGeneratorError("output item shape drifted")
        ordinal = row.get("ordinal")
        token_count = row.get("completion_token_count")
        digest = row.get("completion_sha256")
        valid = row.get("generation_valid")
        expansions = row.get("expansions")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != position
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or not 0 <= token_count <= MAXIMUM_COMPLETION_TOKENS
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or not isinstance(valid, bool)
            or not isinstance(expansions, list)
            or (valid and len(expansions) != len(EXPANSION_KEYS))
            or (not valid and expansions != [])
        ):
            raise BrightQueryGeneratorError("output item values drifted")
        if valid:
            checked = [
                _text(
                    value,
                    field=f"items[{position}].expansions",
                    maximum=MAXIMUM_EXPANSION_CHARACTERS,
                )
                for value in expansions
            ]
            if len({_normalize(value) for value in checked}) != len(checked):
                raise BrightQueryGeneratorError("output expansions are duplicated")
    return {"items": rows, "schema": OUTPUT_SCHEMA}


def parse_output(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightQueryGeneratorError("output is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != _OUTPUT_KEYS
        or value.get("schema") != OUTPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise BrightQueryGeneratorError("output envelope drifted")
    return output_payload(value.get("items", ()))

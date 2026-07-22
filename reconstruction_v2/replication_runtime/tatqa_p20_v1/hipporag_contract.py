"""Frozen item-local official-HippoRAG retrieval contract for TAT-QA P20."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence


VERSION = "tatqa_p20_official_hipporag_item_retrieve_only_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
DOCUMENT_SCHEMA = f"{VERSION}_document"
TOP_K = 5
MINIMUM_UNIT_COUNT = TOP_K
MAXIMUM_UNIT_COUNT = 96
MAXIMUM_QUERY_CHARACTERS = 24_000
MAXIMUM_UNIT_CHARACTERS = 24_000
_UNIT_ID = re.compile(r"(?:T:(?:0|[1-9][0-9]*)|P:[1-9][0-9]*)\Z")
_UNIT_KEYS = frozenset({"ordinal", "text", "unit_id"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TatqaP20OfficialHippoRAGError(RuntimeError):
    """The frozen item-local official-core retrieval contract failed closed."""


@dataclass(frozen=True)
class CandidateUnit:
    ordinal: int
    unit_id: str
    text: str


def _unit_key(unit_id: str) -> tuple[int, int]:
    prefix, ordinal = unit_id.split(":", 1)
    return (0 if prefix == "T" else 1, int(ordinal))


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
        raise TatqaP20OfficialHippoRAGError("value is not canonical JSON") from exc


def _text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise TatqaP20OfficialHippoRAGError(f"{field} is invalid")
    return value


def validate_input(
    query: object, units: object
) -> tuple[str, tuple[CandidateUnit, ...]]:
    query_text = _text(query, "query", MAXIMUM_QUERY_CHARACTERS)
    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise TatqaP20OfficialHippoRAGError("candidate units are not a sequence")
    if not MINIMUM_UNIT_COUNT <= len(units) <= MAXIMUM_UNIT_COUNT:
        raise TatqaP20OfficialHippoRAGError("candidate unit count drifted")
    rows: list[CandidateUnit] = []
    identifiers: set[str] = set()
    for position, raw in enumerate(units):
        if not isinstance(raw, Mapping) or set(raw) != _UNIT_KEYS:
            raise TatqaP20OfficialHippoRAGError("candidate unit shape drifted")
        ordinal = raw.get("ordinal")
        unit_id = raw.get("unit_id")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != position
            or not isinstance(unit_id, str)
            or _UNIT_ID.fullmatch(unit_id) is None
            or unit_id in identifiers
        ):
            raise TatqaP20OfficialHippoRAGError("candidate unit identity drifted")
        identifiers.add(unit_id)
        rows.append(
            CandidateUnit(
                ordinal=ordinal,
                unit_id=unit_id,
                text=_text(raw.get("text"), "unit text", MAXIMUM_UNIT_CHARACTERS),
            )
        )
    serialized = serialize_units(rows)
    if len(set(serialized)) != len(rows):
        raise TatqaP20OfficialHippoRAGError("serialized candidate units duplicated")
    identifiers_in_order = tuple(row.unit_id for row in rows)
    if identifiers_in_order[0] != "T:0" or identifiers_in_order != tuple(
        sorted(identifiers_in_order, key=_unit_key)
    ):
        raise TatqaP20OfficialHippoRAGError(
            "candidate units lack T:0 or are not in canonical order"
        )
    return query_text, tuple(rows)


def serialize_unit(row: CandidateUnit) -> str:
    return json.dumps(
        {
            "content": row.text,
            "document_ordinal": row.ordinal,
            "schema": DOCUMENT_SCHEMA,
            "unit_id": row.unit_id,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def serialize_units(rows: Sequence[CandidateUnit]) -> tuple[str, ...]:
    return tuple(serialize_unit(row) for row in rows)


def input_binding_sha256(query: str, units: Sequence[CandidateUnit]) -> str:
    body = {
        "query": query,
        "schema": INPUT_SCHEMA,
        "units": [
            {"ordinal": row.ordinal, "text": row.text, "unit_id": row.unit_id}
            for row in units
        ],
    }
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def input_payload(*, query: object, units: object) -> dict[str, Any]:
    checked_query, checked_units = validate_input(query, units)
    body = {
        "query": checked_query,
        "schema": INPUT_SCHEMA,
        "units": [
            {"ordinal": row.ordinal, "text": row.text, "unit_id": row.unit_id}
            for row in checked_units
        ],
    }
    return {
        **body,
        "input_sha256": input_binding_sha256(checked_query, checked_units),
    }


def parse_input(raw: bytes) -> tuple[str, tuple[CandidateUnit, ...]]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP20OfficialHippoRAGError("worker input is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"input_sha256", "query", "schema", "units"}
        or value.get("schema") != INPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise TatqaP20OfficialHippoRAGError("worker input envelope drifted")
    query, units = validate_input(value.get("query"), value.get("units"))
    if value.get("input_sha256") != input_binding_sha256(query, units):
        raise TatqaP20OfficialHippoRAGError("worker input self binding drifted")
    return query, units


def stable_top_k(
    *,
    retrieved_documents: object,
    retrieved_scores: object,
    document_to_unit: Mapping[str, CandidateUnit],
) -> tuple[str, ...]:
    if isinstance(retrieved_documents, (str, bytes)) or isinstance(
        retrieved_scores, (str, bytes)
    ):
        raise TatqaP20OfficialHippoRAGError("official result is malformed")
    try:
        documents = list(retrieved_documents)  # type: ignore[arg-type]
        scores = list(retrieved_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TatqaP20OfficialHippoRAGError("official result is not iterable") from exc
    if len(documents) != len(document_to_unit) or len(scores) != len(document_to_unit):
        raise TatqaP20OfficialHippoRAGError("official result omitted candidates")
    ranked: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if (
            not isinstance(document, str)
            or document not in document_to_unit
            or document in seen
            or isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            raise TatqaP20OfficialHippoRAGError("official result row drifted")
        seen.add(document)
        unit = document_to_unit[document]
        ranked.append((float(score), unit.ordinal, unit.unit_id))
    if seen != set(document_to_unit):
        raise TatqaP20OfficialHippoRAGError("official result candidate set drifted")
    ranked.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(row[2] for row in ranked[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise TatqaP20OfficialHippoRAGError("official top five drifted")
    return result


def output_payload(
    *,
    top_unit_ids: Sequence[str],
    graph_nodes: int,
    graph_edges: int,
    unit_count: int,
    input_sha256: str,
) -> dict[str, Any]:
    values = tuple(top_unit_ids)
    if (
        not MINIMUM_UNIT_COUNT <= unit_count <= MAXIMUM_UNIT_COUNT
        or len(values) != TOP_K
        or len(set(values)) != TOP_K
        or any(not isinstance(value, str) or _UNIT_ID.fullmatch(value) is None for value in values)
        or not isinstance(input_sha256, str)
        or _SHA256.fullmatch(input_sha256) is None
    ):
        raise TatqaP20OfficialHippoRAGError("official output identities drifted")
    for value, field in ((graph_nodes, "graph nodes"), (graph_edges, "graph edges")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TatqaP20OfficialHippoRAGError(f"{field} drifted")
    return {
        "graph_edge_count": graph_edges,
        "graph_node_count": graph_nodes,
        "input_sha256": input_sha256,
        "schema": OUTPUT_SCHEMA,
        "top_unit_ids": list(values),
        "unit_count": unit_count,
    }


def parse_output(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP20OfficialHippoRAGError("worker output is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "graph_edge_count",
            "graph_node_count",
            "input_sha256",
            "schema",
            "top_unit_ids",
            "unit_count",
        }
        or value.get("schema") != OUTPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise TatqaP20OfficialHippoRAGError("worker output envelope drifted")
    return output_payload(
        top_unit_ids=value.get("top_unit_ids", ()),
        graph_nodes=value.get("graph_node_count"),  # type: ignore[arg-type]
        graph_edges=value.get("graph_edge_count"),  # type: ignore[arg-type]
        unit_count=value.get("unit_count"),  # type: ignore[arg-type]
        input_sha256=value.get("input_sha256"),  # type: ignore[arg-type]
    )


__all__ = [
    "CandidateUnit",
    "INPUT_SCHEMA",
    "MAXIMUM_UNIT_COUNT",
    "OUTPUT_SCHEMA",
    "TOP_K",
    "TatqaP20OfficialHippoRAGError",
    "VERSION",
    "canonical_json_bytes",
    "input_binding_sha256",
    "input_payload",
    "output_payload",
    "parse_input",
    "parse_output",
    "serialize_units",
    "stable_top_k",
    "validate_input",
]

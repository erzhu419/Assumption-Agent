"""Label-free contract for candidate-restricted BRIGHT HippoRAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any, Mapping, Sequence


VERSION = "bright_official_hipporag_candidate_retrieval_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
DOCUMENT_SCHEMA = f"{VERSION}_document"
CANDIDATE_COUNT = 32
TOP_K = 10
MAX_QUERY_CHARACTERS = 24_000
MAX_DOCUMENT_CHARACTERS = 3_000
DOCUMENT_KEYS = frozenset({"ordinal", "content"})


class BrightOfficialHippoRAGError(RuntimeError):
    """The isolated official-core retrieval contract failed closed."""


@dataclass(frozen=True)
class CandidateDocument:
    ordinal: int
    content: str


def canonical_json_bytes(value: Any) -> bytes:
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
        raise BrightOfficialHippoRAGError("value is not canonical JSON") from exc


def _text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BrightOfficialHippoRAGError(f"{field} is invalid")
    return value


def validate_input(
    query: object, documents: object
) -> tuple[str, tuple[CandidateDocument, ...]]:
    query_text = _text(query, "query", MAX_QUERY_CHARACTERS)
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise BrightOfficialHippoRAGError("documents are not a sequence")
    if len(documents) != CANDIDATE_COUNT:
        raise BrightOfficialHippoRAGError("candidate count drifted")
    rows: list[CandidateDocument] = []
    for position, raw in enumerate(documents):
        if not isinstance(raw, Mapping) or set(raw) != DOCUMENT_KEYS:
            raise BrightOfficialHippoRAGError("candidate document shape drifted")
        ordinal = raw.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal != position:
            raise BrightOfficialHippoRAGError("candidate ordinals are not canonical")
        rows.append(
            CandidateDocument(
                ordinal=ordinal,
                content=_text(
                    raw.get("content"),
                    f"documents[{position}].content",
                    MAX_DOCUMENT_CHARACTERS,
                ),
            )
        )
    serialized = serialize_documents(rows)
    if len(set(serialized)) != CANDIDATE_COUNT:
        raise BrightOfficialHippoRAGError("serialized candidates are duplicated")
    return query_text, tuple(rows)


def serialize_document(row: CandidateDocument) -> str:
    return json.dumps(
        {
            "content": row.content,
            "document_ordinal": row.ordinal,
            "schema": DOCUMENT_SCHEMA,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def serialize_documents(rows: Sequence[CandidateDocument]) -> tuple[str, ...]:
    return tuple(serialize_document(row) for row in rows)


def stable_top_k(
    *,
    retrieved_documents: object,
    retrieved_scores: object,
    document_to_ordinal: Mapping[str, int],
) -> tuple[int, ...]:
    if isinstance(retrieved_documents, (str, bytes)) or isinstance(
        retrieved_scores, (str, bytes)
    ):
        raise BrightOfficialHippoRAGError("official result is malformed")
    try:
        documents = list(retrieved_documents)  # type: ignore[arg-type]
        scores = list(retrieved_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BrightOfficialHippoRAGError("official result is not iterable") from exc
    if len(documents) != CANDIDATE_COUNT or len(scores) != CANDIDATE_COUNT:
        raise BrightOfficialHippoRAGError("official result did not return all candidates")
    ranked: list[tuple[float, int]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if not isinstance(document, str) or document not in document_to_ordinal:
            raise BrightOfficialHippoRAGError("official result contains an unknown document")
        if document in seen:
            raise BrightOfficialHippoRAGError("official result contains a duplicate document")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise BrightOfficialHippoRAGError("official result score is not numeric")
        numeric = float(score)
        if not math.isfinite(numeric):
            raise BrightOfficialHippoRAGError("official result score is not finite")
        seen.add(document)
        ranked.append((numeric, document_to_ordinal[document]))
    if seen != set(document_to_ordinal):
        raise BrightOfficialHippoRAGError("official result omitted a candidate")
    ranked.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(ordinal for _score, ordinal in ranked[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise BrightOfficialHippoRAGError("official top-k output drifted")
    return result


def output_payload(
    *, top_ordinals: Sequence[int], graph_nodes: int, graph_edges: int
) -> dict[str, Any]:
    values = tuple(top_ordinals)
    if (
        len(values) != TOP_K
        or len(set(values)) != TOP_K
        or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < CANDIDATE_COUNT for value in values)
    ):
        raise BrightOfficialHippoRAGError("top ordinals are invalid")
    for value, field in ((graph_nodes, "graph nodes"), (graph_edges, "graph edges")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BrightOfficialHippoRAGError(f"{field} is invalid")
    return {
        "graph_edge_count": graph_edges,
        "graph_node_count": graph_nodes,
        "schema": OUTPUT_SCHEMA,
        "top_ordinals": list(values),
    }


def parse_output(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightOfficialHippoRAGError("worker output is invalid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "graph_edge_count",
        "graph_node_count",
        "schema",
        "top_ordinals",
    }:
        raise BrightOfficialHippoRAGError("worker output shape drifted")
    if canonical_json_bytes(value) != raw or value.get("schema") != OUTPUT_SCHEMA:
        raise BrightOfficialHippoRAGError("worker output is not canonical")
    return output_payload(
        top_ordinals=value.get("top_ordinals", ()),
        graph_nodes=value.get("graph_node_count"),  # type: ignore[arg-type]
        graph_edges=value.get("graph_edge_count"),  # type: ignore[arg-type]
    )

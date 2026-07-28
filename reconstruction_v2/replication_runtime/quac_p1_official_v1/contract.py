"""Strict source-free contract for one QuAC official-HippoRAG block.

The adapter sees one complete, fixed evidence-window corpus and one eager
batch of full question-only queries.  Its input surface intentionally has no
source split, relation family, qrel, answer, gold, candidate graph, evaluator,
or score field.  HippoRAG must rank the complete block corpus for every query;
the adapter then applies the frozen score-descending, opaque-unit-ID tie break
and returns only top-five unit IDs plus commitments.

Raw unit text is allowed to repeat.  Each unit is addressed inside HippoRAG by
a canonical ASCII JSON document containing its opaque ID, so equal text never
collides in the official result-to-unit join.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence


VERSION = "quac_p1_official_hipporag_block_v1"
INPUT_SCHEMA = f"{VERSION}_private_input"
OUTPUT_SCHEMA = f"{VERSION}_private_output"
RUNTIME_SCHEMA = f"{VERSION}_runtime_receipt"
TOP_K = 5
MIN_UNIT_COUNT = TOP_K
MAX_UNIT_COUNT = 8_192
MAX_QUERY_COUNT = 512
MAX_UNIT_TEXT_CHARACTERS = 32_000
MAX_QUERY_TEXT_CHARACTERS = 32_000
OFFICIAL_HIPPORAG_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_INPUT_KEYS = frozenset(
    {
        "block_id",
        "corpus",
        "corpus_sha256",
        "queries",
        "queries_sha256",
        "schema",
    }
)
_UNIT_KEYS = frozenset({"text", "unit_id"})
_QUERY_KEYS = frozenset({"query_id", "text"})
_OUTPUT_KEYS = frozenset(
    {
        "block_id",
        "corpus_sha256",
        "full_rankings_sha256",
        "input_sha256",
        "queries_sha256",
        "rows",
        "runtime",
        "schema",
        "self_sha256",
    }
)
_OUTPUT_ROW_KEYS = frozenset({"query_id", "top5_unit_ids"})
_RUNTIME_KEYS = frozenset(
    {
        "complete_ranking_count",
        "corpus_count",
        "graph_edge_count",
        "graph_node_count",
        "index_call_count",
        "offline_required",
        "official_hipporag_commit",
        "query_count",
        "retrieve_call_count",
        "single_gpu_required",
        "single_thread_required",
    }
)


class QuacP1OfficialHippoRAGError(RuntimeError):
    """The private block contract or official result failed closed."""


@dataclass(frozen=True)
class UnitRow:
    """One opaque, gold-independent evidence-window unit."""

    unit_id: str
    text: str


@dataclass(frozen=True)
class QueryRow:
    """One opaque full question-only query."""

    query_id: str
    text: str


@dataclass(frozen=True)
class BlockInput:
    """Validated complete block corpus and eager query batch."""

    block_id: str
    units: tuple[UnitRow, ...]
    queries: tuple[QueryRow, ...]
    corpus_sha256: str
    queries_sha256: str


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    """Encode one exact ASCII private IPC value."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1OfficialHippoRAGError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    """Hash the exact canonical semantic value without a trailing newline."""

    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def _opaque_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise QuacP1OfficialHippoRAGError(
            f"{field} must be an opaque lowercase SHA-256 ID"
        )
    return value


def _exact_text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise QuacP1OfficialHippoRAGError(f"{field} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise QuacP1OfficialHippoRAGError(
            f"{field} contains invalid Unicode"
        ) from exc
    return value


def _projection_units(units: Sequence[UnitRow]) -> list[dict[str, str]]:
    return [
        {"text": row.text, "unit_id": row.unit_id}
        for row in units
    ]


def _projection_queries(
    queries: Sequence[QueryRow],
) -> list[dict[str, str]]:
    return [
        {"query_id": row.query_id, "text": row.text}
        for row in queries
    ]


def _parse_units(value: object) -> tuple[UnitRow, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QuacP1OfficialHippoRAGError("corpus must be an array")
    if not MIN_UNIT_COUNT <= len(value) <= MAX_UNIT_COUNT:
        raise QuacP1OfficialHippoRAGError(
            "corpus count is outside the frozen bounds"
        )
    rows: list[UnitRow] = []
    for ordinal, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _UNIT_KEYS:
            raise QuacP1OfficialHippoRAGError(
                f"corpus[{ordinal}] shape drifted"
            )
        rows.append(
            UnitRow(
                unit_id=_opaque_id(
                    raw.get("unit_id"), f"corpus[{ordinal}].unit_id"
                ),
                text=_exact_text(
                    raw.get("text"),
                    f"corpus[{ordinal}].text",
                    MAX_UNIT_TEXT_CHARACTERS,
                ),
            )
        )
    result = tuple(rows)
    unit_ids = tuple(row.unit_id for row in result)
    if unit_ids != tuple(sorted(unit_ids)) or len(set(unit_ids)) != len(
        unit_ids
    ):
        raise QuacP1OfficialHippoRAGError(
            "opaque unit IDs must be unique and strictly sorted"
        )
    return result


def _parse_queries(value: object) -> tuple[QueryRow, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QuacP1OfficialHippoRAGError("queries must be an array")
    if not 1 <= len(value) <= MAX_QUERY_COUNT:
        raise QuacP1OfficialHippoRAGError(
            "query count is outside the frozen bounds"
        )
    rows: list[QueryRow] = []
    for ordinal, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _QUERY_KEYS:
            raise QuacP1OfficialHippoRAGError(
                f"queries[{ordinal}] shape drifted"
            )
        rows.append(
            QueryRow(
                query_id=_opaque_id(
                    raw.get("query_id"), f"queries[{ordinal}].query_id"
                ),
                text=_exact_text(
                    raw.get("text"),
                    f"queries[{ordinal}].text",
                    MAX_QUERY_TEXT_CHARACTERS,
                ),
            )
        )
    result = tuple(rows)
    query_ids = tuple(row.query_id for row in result)
    if query_ids != tuple(sorted(query_ids)) or len(set(query_ids)) != len(
        query_ids
    ):
        raise QuacP1OfficialHippoRAGError(
            "opaque query IDs must be unique and strictly sorted"
        )
    return result


def validate_input(value: object) -> BlockInput:
    """Validate the complete label-free block input and commitments."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _INPUT_KEYS
        or value.get("schema") != INPUT_SCHEMA
    ):
        raise QuacP1OfficialHippoRAGError("input envelope drifted")
    block_id = _opaque_id(value.get("block_id"), "block_id")
    units = _parse_units(value.get("corpus"))
    queries = _parse_queries(value.get("queries"))
    claimed_corpus = _opaque_id(
        value.get("corpus_sha256"), "corpus_sha256"
    )
    claimed_queries = _opaque_id(
        value.get("queries_sha256"), "queries_sha256"
    )
    expected_corpus = stable_hash(_projection_units(units))
    expected_queries = stable_hash(_projection_queries(queries))
    if claimed_corpus != expected_corpus:
        raise QuacP1OfficialHippoRAGError(
            "corpus commitment mismatched"
        )
    if claimed_queries != expected_queries:
        raise QuacP1OfficialHippoRAGError(
            "query commitment mismatched"
        )
    return BlockInput(
        block_id=block_id,
        units=units,
        queries=queries,
        corpus_sha256=claimed_corpus,
        queries_sha256=claimed_queries,
    )


def build_input(
    *,
    block_id: str,
    units: Sequence[Mapping[str, object]],
    queries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build one canonical input after sorting only by opaque IDs."""

    unit_rows = sorted(
        (
            {
                "text": row.get("text"),
                "unit_id": row.get("unit_id"),
            }
            if isinstance(row, Mapping) and set(row) == _UNIT_KEYS
            else dict(row)
            for row in units
        ),
        key=lambda row: str(row.get("unit_id")),
    )
    query_rows = sorted(
        (
            {
                "query_id": row.get("query_id"),
                "text": row.get("text"),
            }
            if isinstance(row, Mapping) and set(row) == _QUERY_KEYS
            else dict(row)
            for row in queries
        ),
        key=lambda row: str(row.get("query_id")),
    )
    parsed_units = _parse_units(unit_rows)
    parsed_queries = _parse_queries(query_rows)
    value = {
        "block_id": _opaque_id(block_id, "block_id"),
        "corpus": _projection_units(parsed_units),
        "corpus_sha256": stable_hash(_projection_units(parsed_units)),
        "queries": _projection_queries(parsed_queries),
        "queries_sha256": stable_hash(
            _projection_queries(parsed_queries)
        ),
        "schema": INPUT_SCHEMA,
    }
    validate_input(value)
    return value


def canonical_unit_document(unit: UnitRow) -> str:
    """Return a unique ASCII HippoRAG document even for duplicate raw text."""

    if not isinstance(unit, UnitRow):
        raise QuacP1OfficialHippoRAGError("unit row is invalid")
    _opaque_id(unit.unit_id, "unit_id")
    _exact_text(unit.text, "unit text", MAX_UNIT_TEXT_CHARACTERS)
    return canonical_bytes(
        {
            "text": unit.text,
            "title": f"QUAC_EVIDENCE_UNIT_{unit.unit_id}",
        }
    ).decode("ascii")


def serialize_corpus(units: Sequence[UnitRow]) -> tuple[str, ...]:
    """Serialize the complete fixed corpus in canonical opaque-ID order."""

    rows = tuple(units)
    if (
        not MIN_UNIT_COUNT <= len(rows) <= MAX_UNIT_COUNT
        or any(not isinstance(row, UnitRow) for row in rows)
        or tuple(row.unit_id for row in rows)
        != tuple(sorted(row.unit_id for row in rows))
    ):
        raise QuacP1OfficialHippoRAGError(
            "validated corpus rows are required"
        )
    documents = tuple(canonical_unit_document(row) for row in rows)
    if len(set(documents)) != len(documents):
        raise QuacP1OfficialHippoRAGError(
            "canonical unit documents collided"
        )
    return documents


def stable_complete_ranking(
    *,
    retrieved_documents: object,
    retrieved_scores: object,
    document_to_unit_id: Mapping[str, str],
) -> tuple[str, ...]:
    """Validate a complete official permutation and apply the frozen order."""

    if (
        isinstance(retrieved_documents, (str, bytes))
        or isinstance(retrieved_scores, (str, bytes))
        or not isinstance(document_to_unit_id, Mapping)
        or len(document_to_unit_id) < TOP_K
    ):
        raise QuacP1OfficialHippoRAGError(
            "official complete-ranking input is malformed"
        )
    try:
        documents = list(retrieved_documents)  # type: ignore[arg-type]
        scores = list(retrieved_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise QuacP1OfficialHippoRAGError(
            "official complete ranking is not iterable"
        ) from exc
    expected_documents = set(document_to_unit_id)
    expected_unit_ids = tuple(document_to_unit_id.values())
    if (
        len(expected_documents) != len(document_to_unit_id)
        or len(set(expected_unit_ids)) != len(expected_unit_ids)
        or any(
            not isinstance(value, str) or _HEX64.fullmatch(value) is None
            for value in expected_unit_ids
        )
        or len(documents) != len(document_to_unit_id)
        or len(scores) != len(document_to_unit_id)
    ):
        raise QuacP1OfficialHippoRAGError(
            "official result is not a complete corpus permutation"
        )
    ranked: list[tuple[float, str]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if (
            not isinstance(document, str)
            or document not in document_to_unit_id
            or document in seen
            or isinstance(score, bool)
            or not isinstance(score, Real)
        ):
            raise QuacP1OfficialHippoRAGError(
                "official complete ranking row drifted"
            )
        numeric = float(score)
        if not math.isfinite(numeric):
            raise QuacP1OfficialHippoRAGError(
                "official ranking score is nonfinite"
            )
        seen.add(document)
        ranked.append((numeric, document_to_unit_id[document]))
    if seen != expected_documents:
        raise QuacP1OfficialHippoRAGError(
            "official result is not a complete corpus permutation"
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(unit_id for _score, unit_id in ranked)
    if set(result) != set(expected_unit_ids):
        raise QuacP1OfficialHippoRAGError(
            "stable complete ranking drifted"
        )
    return result


def _nonnegative_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise QuacP1OfficialHippoRAGError(f"{field} is invalid")
    return value


def build_output(
    *,
    input_value: Mapping[str, object],
    full_rankings: Sequence[Sequence[str]],
    graph_node_count: int,
    graph_edge_count: int,
) -> dict[str, object]:
    """Build the private top-five output and full-ranking commitment."""

    block = validate_input(input_value)
    rankings = tuple(tuple(row) for row in full_rankings)
    expected_units = {row.unit_id for row in block.units}
    if len(rankings) != len(block.queries):
        raise QuacP1OfficialHippoRAGError(
            "full-ranking query count drifted"
        )
    for ranking in rankings:
        if (
            len(ranking) != len(block.units)
            or len(set(ranking)) != len(ranking)
            or set(ranking) != expected_units
        ):
            raise QuacP1OfficialHippoRAGError(
                "full ranking is not a unit permutation"
            )
    node_count = _nonnegative_integer(
        graph_node_count, "graph_node_count"
    )
    edge_count = _nonnegative_integer(
        graph_edge_count, "graph_edge_count"
    )
    ranking_projection = [
        {
            "query_id": query.query_id,
            "unit_ids": list(ranking),
        }
        for query, ranking in zip(block.queries, rankings)
    ]
    body: dict[str, object] = {
        "block_id": block.block_id,
        "corpus_sha256": block.corpus_sha256,
        "full_rankings_sha256": stable_hash(ranking_projection),
        "input_sha256": stable_hash(input_value),
        "queries_sha256": block.queries_sha256,
        "rows": [
            {
                "query_id": query.query_id,
                "top5_unit_ids": list(ranking[:TOP_K]),
            }
            for query, ranking in zip(block.queries, rankings)
        ],
        "runtime": {
            "complete_ranking_count": len(rankings),
            "corpus_count": len(block.units),
            "graph_edge_count": edge_count,
            "graph_node_count": node_count,
            "index_call_count": 1,
            "offline_required": True,
            "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
            "query_count": len(block.queries),
            "retrieve_call_count": 1,
            "single_gpu_required": True,
            "single_thread_required": True,
        },
        "schema": OUTPUT_SCHEMA,
    }
    body["self_sha256"] = stable_hash(body)
    validate_output(body, expected_input=input_value)
    return body


def validate_output(
    value: object,
    *,
    expected_input: Mapping[str, object],
) -> dict[str, object]:
    """Validate top-five rows and all input/output commitments."""

    block = validate_input(expected_input)
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
        raise QuacP1OfficialHippoRAGError("output envelope drifted")
    normalized = dict(value)
    claimed_self = normalized.pop("self_sha256", None)
    if (
        not isinstance(claimed_self, str)
        or _HEX64.fullmatch(claimed_self) is None
        or claimed_self != stable_hash(normalized)
        or normalized.get("schema") != OUTPUT_SCHEMA
        or normalized.get("block_id") != block.block_id
        or normalized.get("corpus_sha256") != block.corpus_sha256
        or normalized.get("queries_sha256") != block.queries_sha256
        or normalized.get("input_sha256") != stable_hash(expected_input)
        or not isinstance(normalized.get("full_rankings_sha256"), str)
        or _HEX64.fullmatch(
            str(normalized.get("full_rankings_sha256"))
        )
        is None
    ):
        raise QuacP1OfficialHippoRAGError(
            "output commitment drifted"
        )
    runtime = normalized.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_KEYS:
        raise QuacP1OfficialHippoRAGError(
            "runtime receipt shape drifted"
        )
    for field, expected in (
        ("complete_ranking_count", len(block.queries)),
        ("corpus_count", len(block.units)),
        ("index_call_count", 1),
        ("query_count", len(block.queries)),
        ("retrieve_call_count", 1),
    ):
        if runtime.get(field) != expected:
            raise QuacP1OfficialHippoRAGError(
                f"runtime {field} drifted"
            )
    if (
        runtime.get("offline_required") is not True
        or runtime.get("single_gpu_required") is not True
        or runtime.get("single_thread_required") is not True
        or runtime.get("official_hipporag_commit")
        != OFFICIAL_HIPPORAG_COMMIT
    ):
        raise QuacP1OfficialHippoRAGError(
            "runtime execution contract drifted"
        )
    _nonnegative_integer(
        runtime.get("graph_node_count"), "graph_node_count"
    )
    _nonnegative_integer(
        runtime.get("graph_edge_count"), "graph_edge_count"
    )
    rows = normalized.get("rows")
    if not isinstance(rows, list) or len(rows) != len(block.queries):
        raise QuacP1OfficialHippoRAGError("output rows drifted")
    valid_units = {row.unit_id for row in block.units}
    for query, row in zip(block.queries, rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != _OUTPUT_ROW_KEYS
            or row.get("query_id") != query.query_id
        ):
            raise QuacP1OfficialHippoRAGError(
                "output query row drifted"
            )
        top5 = row.get("top5_unit_ids")
        if (
            not isinstance(top5, list)
            or len(top5) != TOP_K
            or len(set(top5)) != TOP_K
            or any(unit_id not in valid_units for unit_id in top5)
        ):
            raise QuacP1OfficialHippoRAGError(
                "output top-five drifted"
            )
    normalized["self_sha256"] = claimed_self
    return normalized


__all__ = [
    "INPUT_SCHEMA",
    "MAX_QUERY_COUNT",
    "MAX_UNIT_COUNT",
    "MIN_UNIT_COUNT",
    "OFFICIAL_HIPPORAG_COMMIT",
    "OUTPUT_SCHEMA",
    "RUNTIME_SCHEMA",
    "TOP_K",
    "BlockInput",
    "QueryRow",
    "QuacP1OfficialHippoRAGError",
    "UnitRow",
    "build_input",
    "build_output",
    "canonical_bytes",
    "canonical_unit_document",
    "serialize_corpus",
    "stable_complete_ranking",
    "stable_hash",
    "validate_input",
    "validate_output",
]

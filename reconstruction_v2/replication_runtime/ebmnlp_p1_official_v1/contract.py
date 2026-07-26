"""Label-free one-abstract, three-query official HippoRAG contract.

The contract has no PMID, split, P/I/O label, gold span, utility, evaluator,
or online surface.  It receives an opaque abstract work identifier, every
gold-independent evidence window in canonical order, and the three frozen
role-definition queries.  The official core returns a complete deterministic
window permutation for each query; top-k truncation remains a controller
operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence


VERSION = "ebmnlp_p1_official_hipporag_contract_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
DOCUMENT_SCHEMA = f"{VERSION}_document"
ROLE_ORDER = ("PARTICIPANT", "INTERVENTION", "OUTCOME")
ROLE_QUERIES = {
    "PARTICIPANT": (
        "Which text describes the participants or patient population "
        "in this clinical trial?"
    ),
    "INTERVENTION": (
        "Which text describes the intervention or treatment "
        "in this clinical trial?"
    ),
    "OUTCOME": (
        "Which text describes the outcomes or endpoints measured "
        "in this clinical trial?"
    ),
}
MIN_DOCUMENT_COUNT = 1
MAX_DOCUMENT_COUNT = 8_192
MAX_DOCUMENT_CHARACTERS = 20_000
MAX_OPAQUE_ID_CHARACTERS = 1_024
MAX_QUERY_CHARACTERS = 1_024
DOCUMENT_KEYS = frozenset({"ordinal", "text", "window_id"})
QUERY_KEYS = frozenset({"ordinal", "role", "text", "work_id"})
INPUT_KEYS = frozenset(
    {
        "abstract_work_id",
        "corpus_sha256",
        "documents",
        "queries",
        "schema",
    }
)
OUTPUT_ROW_KEYS = frozenset(
    {"query_ordinal", "rank_window_ordinals", "role", "work_id"}
)
OUTPUT_KEYS = frozenset(
    {
        "abstract_work_id",
        "corpus_sha256",
        "document_count",
        "graph_edge_count",
        "graph_node_count",
        "rows",
        "schema",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WINDOW_ID = re.compile(r"W:[0-9]{8}:[0-9]{8}\Z")


class EBMNLPOfficialHippoRAGError(RuntimeError):
    """The isolated official retrieve-only contract failed closed."""


@dataclass(frozen=True)
class WindowDocument:
    ordinal: int
    window_id: str
    text: str


@dataclass(frozen=True)
class RoleQuery:
    ordinal: int
    role: str
    work_id: str
    text: str


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EBMNLPOfficialHippoRAGError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def _bounded_text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise EBMNLPOfficialHippoRAGError(f"{field} is invalid")
    return value


def _opaque_id(value: object, field: str) -> str:
    return _bounded_text(value, field, MAX_OPAQUE_ID_CHARACTERS)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EBMNLPOfficialHippoRAGError(f"{field} is invalid")
    return value


def _validate_documents(value: object) -> tuple[WindowDocument, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EBMNLPOfficialHippoRAGError("documents are not a sequence")
    if not MIN_DOCUMENT_COUNT <= len(value) <= MAX_DOCUMENT_COUNT:
        raise EBMNLPOfficialHippoRAGError(
            "document count is outside frozen bounds"
        )
    rows: list[WindowDocument] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != DOCUMENT_KEYS:
            raise EBMNLPOfficialHippoRAGError("document shape drifted")
        ordinal = raw.get("ordinal")
        window_id = raw.get("window_id")
        if type(ordinal) is not int or ordinal != position:
            raise EBMNLPOfficialHippoRAGError(
                "document ordinals are not canonical"
            )
        if (
            not isinstance(window_id, str)
            or _WINDOW_ID.fullmatch(window_id) is None
        ):
            raise EBMNLPOfficialHippoRAGError("window ID drifted")
        rows.append(
            WindowDocument(
                ordinal=ordinal,
                window_id=window_id,
                text=_bounded_text(
                    raw.get("text"),
                    f"documents[{position}].text",
                    MAX_DOCUMENT_CHARACTERS,
                ),
            )
        )
    documents = tuple(rows)
    if len({row.window_id for row in documents}) != len(documents):
        raise EBMNLPOfficialHippoRAGError("window IDs are duplicated")
    return documents


def _validate_queries(value: object) -> tuple[RoleQuery, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EBMNLPOfficialHippoRAGError("queries are not a sequence")
    if len(value) != len(ROLE_ORDER):
        raise EBMNLPOfficialHippoRAGError("query count drifted")
    rows: list[RoleQuery] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != QUERY_KEYS:
            raise EBMNLPOfficialHippoRAGError("query shape drifted")
        role = ROLE_ORDER[position]
        if (
            raw.get("ordinal") != position
            or raw.get("role") != role
            or raw.get("text") != ROLE_QUERIES[role]
        ):
            raise EBMNLPOfficialHippoRAGError(
                "role query registry drifted"
            )
        rows.append(
            RoleQuery(
                ordinal=position,
                role=role,
                work_id=_opaque_id(
                    raw.get("work_id"), f"queries[{position}].work_id"
                ),
                text=_bounded_text(
                    raw.get("text"),
                    f"queries[{position}].text",
                    MAX_QUERY_CHARACTERS,
                ),
            )
        )
    queries = tuple(rows)
    if len({row.work_id for row in queries}) != len(queries):
        raise EBMNLPOfficialHippoRAGError("query work IDs are duplicated")
    return queries


def _document_projection(
    documents: Sequence[WindowDocument],
) -> list[dict[str, object]]:
    return [
        {
            "ordinal": row.ordinal,
            "text": row.text,
            "window_id": row.window_id,
        }
        for row in documents
    ]


def canonical_index_document(document: WindowDocument) -> str:
    if not isinstance(document, WindowDocument):
        raise EBMNLPOfficialHippoRAGError("window document is malformed")
    return canonical_json_bytes(
        {
            "schema": DOCUMENT_SCHEMA,
            "text": document.text,
            "window_id": document.window_id,
        },
        newline=False,
    ).decode("ascii")


def corpus_sha256(documents: object) -> str:
    rows = _validate_documents(documents)
    return hashlib.sha256(
        canonical_json_bytes(_document_projection(rows), newline=False)
    ).hexdigest()


def input_payload(
    *,
    abstract_work_id: object,
    documents: object,
    queries: object,
) -> dict[str, object]:
    document_rows = _validate_documents(documents)
    query_rows = _validate_queries(queries)
    payload = {
        "abstract_work_id": _opaque_id(
            abstract_work_id, "abstract work ID"
        ),
        "corpus_sha256": hashlib.sha256(
            canonical_json_bytes(
                _document_projection(document_rows), newline=False
            )
        ).hexdigest(),
        "documents": _document_projection(document_rows),
        "queries": [
            {
                "ordinal": row.ordinal,
                "role": row.role,
                "text": row.text,
                "work_id": row.work_id,
            }
            for row in query_rows
        ],
        "schema": INPUT_SCHEMA,
    }
    validate_input(payload)
    return payload


def validate_input(
    value: object,
) -> tuple[
    str,
    str,
    tuple[WindowDocument, ...],
    tuple[RoleQuery, ...],
]:
    if (
        not isinstance(value, Mapping)
        or set(value) != INPUT_KEYS
        or value.get("schema") != INPUT_SCHEMA
    ):
        raise EBMNLPOfficialHippoRAGError("input envelope drifted")
    abstract_work_id = _opaque_id(
        value.get("abstract_work_id"), "abstract work ID"
    )
    documents = _validate_documents(value.get("documents"))
    queries = _validate_queries(value.get("queries"))
    claimed = _sha256(value.get("corpus_sha256"), "corpus SHA-256")
    expected = hashlib.sha256(
        canonical_json_bytes(
            _document_projection(documents), newline=False
        )
    ).hexdigest()
    if claimed != expected:
        raise EBMNLPOfficialHippoRAGError("corpus SHA-256 mismatched")
    return abstract_work_id, claimed, documents, queries


def _stable_permutation(
    *,
    retrieved_documents: object,
    retrieved_scores: object,
    document_to_ordinal: Mapping[str, int],
) -> tuple[int, ...]:
    if isinstance(retrieved_documents, (str, bytes)) or isinstance(
        retrieved_scores, (str, bytes)
    ):
        raise EBMNLPOfficialHippoRAGError("official result is malformed")
    try:
        documents = list(retrieved_documents)  # type: ignore[arg-type]
        scores = list(retrieved_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise EBMNLPOfficialHippoRAGError(
            "official result is not iterable"
        ) from exc
    count = len(document_to_ordinal)
    if len(documents) != count or len(scores) != count:
        raise EBMNLPOfficialHippoRAGError(
            "official result did not return the complete corpus"
        )
    ranked: list[tuple[float, int]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if (
            not isinstance(document, str)
            or document not in document_to_ordinal
            or document in seen
            or isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            raise EBMNLPOfficialHippoRAGError(
                "official result row drifted"
            )
        seen.add(document)
        ranked.append((float(score), document_to_ordinal[document]))
    if seen != set(document_to_ordinal):
        raise EBMNLPOfficialHippoRAGError(
            "official result omitted a document"
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return tuple(ordinal for _score, ordinal in ranked)


def _output_payload(
    *,
    abstract_work_id: str,
    corpus_hash: str,
    document_count: int,
    queries: Sequence[RoleQuery],
    rank_rows: Sequence[Sequence[int]],
    graph_nodes: object,
    graph_edges: object,
) -> dict[str, object]:
    if (
        type(document_count) is not int
        or not MIN_DOCUMENT_COUNT <= document_count <= MAX_DOCUMENT_COUNT
        or len(queries) != len(ROLE_ORDER)
        or len(rank_rows) != len(ROLE_ORDER)
    ):
        raise EBMNLPOfficialHippoRAGError("output dimensions drifted")
    rows: list[dict[str, object]] = []
    expected = set(range(document_count))
    for query, raw_rank in zip(queries, rank_rows):
        rank = tuple(raw_rank)
        if (
            len(rank) != document_count
            or set(rank) != expected
            or any(type(value) is not int for value in rank)
        ):
            raise EBMNLPOfficialHippoRAGError(
                "complete rank permutation drifted"
            )
        rows.append(
            {
                "query_ordinal": query.ordinal,
                "rank_window_ordinals": list(rank),
                "role": query.role,
                "work_id": query.work_id,
            }
        )
    for value, field in (
        (graph_nodes, "graph nodes"),
        (graph_edges, "graph edges"),
    ):
        if type(value) is not int or value < 0:
            raise EBMNLPOfficialHippoRAGError(f"{field} is invalid")
    return {
        "abstract_work_id": _opaque_id(
            abstract_work_id, "abstract work ID"
        ),
        "corpus_sha256": _sha256(corpus_hash, "corpus SHA-256"),
        "document_count": document_count,
        "graph_edge_count": graph_edges,
        "graph_node_count": graph_nodes,
        "rows": rows,
        "schema": OUTPUT_SCHEMA,
    }


def retrieve_abstract_with_core(
    *, core: object, payload: Mapping[str, object]
) -> dict[str, object]:
    abstract_work_id, corpus_hash, documents, queries = validate_input(
        payload
    )
    serialized = [canonical_index_document(row) for row in documents]
    mapping = {
        text: row.ordinal for text, row in zip(serialized, documents)
    }
    if len(mapping) != len(documents):
        raise EBMNLPOfficialHippoRAGError(
            "canonical document mapping collided"
        )
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise EBMNLPOfficialHippoRAGError(
            "official core methods are unavailable"
        )
    index(serialized)
    solutions = retrieve(
        [row.text for row in queries],
        num_to_retrieve=len(documents),
    )
    if not isinstance(solutions, list) or len(solutions) != len(queries):
        raise EBMNLPOfficialHippoRAGError(
            "official query batch result drifted"
        )
    rank_rows = [
        _stable_permutation(
            retrieved_documents=getattr(solution, "docs", None),
            retrieved_scores=getattr(solution, "doc_scores", None),
            document_to_ordinal=mapping,
        )
        for solution in solutions
    ]
    graph = getattr(core, "graph", None)
    vcount = getattr(graph, "vcount", None)
    ecount = getattr(graph, "ecount", None)
    if not callable(vcount) or not callable(ecount):
        raise EBMNLPOfficialHippoRAGError(
            "official graph counters are unavailable"
        )
    return _output_payload(
        abstract_work_id=abstract_work_id,
        corpus_hash=corpus_hash,
        document_count=len(documents),
        queries=queries,
        rank_rows=rank_rows,
        graph_nodes=vcount(),
        graph_edges=ecount(),
    )


def parse_output(raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise EBMNLPOfficialHippoRAGError("worker output is not bytes")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EBMNLPOfficialHippoRAGError(
            "worker output is invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != OUTPUT_KEYS
        or value.get("schema") != OUTPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise EBMNLPOfficialHippoRAGError(
            "worker output envelope drifted"
        )
    rows = value.get("rows")
    count = value.get("document_count")
    if (
        type(count) is not int
        or isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or len(rows) != len(ROLE_ORDER)
    ):
        raise EBMNLPOfficialHippoRAGError("worker output rows drifted")
    query_rows: list[RoleQuery] = []
    rank_rows: list[Sequence[int]] = []
    for position, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != OUTPUT_ROW_KEYS
            or row.get("query_ordinal") != position
            or row.get("role") != ROLE_ORDER[position]
        ):
            raise EBMNLPOfficialHippoRAGError(
                "worker output row drifted"
            )
        query_rows.append(
            RoleQuery(
                ordinal=position,
                role=ROLE_ORDER[position],
                work_id=_opaque_id(row.get("work_id"), "query work ID"),
                text=ROLE_QUERIES[ROLE_ORDER[position]],
            )
        )
        rank = row.get("rank_window_ordinals")
        if isinstance(rank, (str, bytes)) or not isinstance(rank, Sequence):
            raise EBMNLPOfficialHippoRAGError(
                "worker output rank drifted"
            )
        rank_rows.append(rank)  # type: ignore[arg-type]
    return _output_payload(
        abstract_work_id=str(value.get("abstract_work_id")),
        corpus_hash=str(value.get("corpus_sha256")),
        document_count=count,
        queries=query_rows,
        rank_rows=rank_rows,
        graph_nodes=value.get("graph_node_count"),
        graph_edges=value.get("graph_edge_count"),
    )

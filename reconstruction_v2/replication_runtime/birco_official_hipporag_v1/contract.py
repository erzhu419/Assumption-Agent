"""Frozen label-free contract for query-local BIRCO HippoRAG retrieval.

Only an opaque controller work identifier, the frozen task objective, the
original query text, and controller-frozen common candidate projections cross
the private input boundary.  Source identifiers, family/block membership, and
qrel values have no representation in this contract.  The public result is a
complete ordinal permutation, the common-projection hash, and content-free
graph counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence


VERSION = "birco_official_hipporag_candidate_retrieval_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
DOCUMENT_SCHEMA = f"{VERSION}_document"
OFFICIAL_HIPPORAG_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"

MIN_CANDIDATE_COUNT = 10
MAX_CANDIDATE_COUNT = 256
# Pool aliases make the dynamic nature of this adapter explicit to controllers.
MIN_POOL_SIZE = MIN_CANDIDATE_COUNT
MAX_POOL_SIZE = MAX_CANDIDATE_COUNT
MAX_WORK_ID_CHARACTERS = 1_024
MAX_OBJECTIVE_CHARACTERS = 8_192
MAX_QUERY_CHARACTERS = 250_000
MAX_DOCUMENT_CHARACTERS = 2_000_000
DOCUMENT_KEYS = frozenset({"ordinal", "text"})
INPUT_KEYS = frozenset(
    {
        "common_projection_sha256",
        "documents",
        "objective",
        "query",
        "schema",
        "work_id",
    }
)
OUTPUT_KEYS = frozenset(
    {
        "candidate_count",
        "common_projection_sha256",
        "graph_edge_count",
        "graph_node_count",
        "rank_ordinals",
        "schema",
        "work_id",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BircoOfficialHippoRAGError(RuntimeError):
    """The isolated official-core retrieval contract failed closed."""


@dataclass(frozen=True)
class CandidateDocument:
    """One controller-frozen common projection with identities removed."""

    ordinal: int
    text: str


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    """Encode exact ASCII JSON for private IPC and content-free output."""

    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BircoOfficialHippoRAGError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def _bounded_text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BircoOfficialHippoRAGError(f"{field} is invalid")
    return value


def _validate_documents(documents: object) -> tuple[CandidateDocument, ...]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise BircoOfficialHippoRAGError("documents are not a sequence")
    candidate_count = len(documents)
    if not MIN_CANDIDATE_COUNT <= candidate_count <= MAX_CANDIDATE_COUNT:
        raise BircoOfficialHippoRAGError("candidate count is outside frozen bounds")

    rows: list[CandidateDocument] = []
    for position, raw in enumerate(documents):
        if not isinstance(raw, Mapping) or set(raw) != DOCUMENT_KEYS:
            raise BircoOfficialHippoRAGError("candidate document shape drifted")
        ordinal = raw.get("ordinal")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != position
        ):
            raise BircoOfficialHippoRAGError(
                "candidate ordinals are not canonical"
            )
        rows.append(
            CandidateDocument(
                ordinal=ordinal,
                text=_bounded_text(
                    raw.get("text"),
                    f"documents[{position}].text",
                    MAX_DOCUMENT_CHARACTERS,
                ),
            )
        )

    validated = tuple(rows)
    serialize_documents(validated)
    return validated


def _common_projection_payload(
    *, objective: str, query: str, documents: Sequence[CandidateDocument]
) -> dict[str, Any]:
    return {
        "documents": [
            {"ordinal": row.ordinal, "text": row.text} for row in documents
        ],
        "objective": objective,
        "query": query,
    }


def common_projection_sha256(
    *, objective: object, query: object, documents: object
) -> str:
    """Hash the exact controller-common objective/query/document projection."""

    objective_text = _bounded_text(
        objective, "objective", MAX_OBJECTIVE_CHARACTERS
    )
    query_text = _bounded_text(query, "query", MAX_QUERY_CHARACTERS)
    rows = _validate_documents(documents)
    return hashlib.sha256(
        canonical_json_bytes(
            _common_projection_payload(
                objective=objective_text, query=query_text, documents=rows
            ),
            newline=False,
        )
    ).hexdigest()


def core_query_text(*, objective: object, query: object) -> str:
    """Return the official core's canonical objective/query query string."""

    objective_text = _bounded_text(
        objective, "objective", MAX_OBJECTIVE_CHARACTERS
    )
    query_text = _bounded_text(query, "query", MAX_QUERY_CHARACTERS)
    return canonical_json_bytes(
        {"objective": objective_text, "query": query_text}, newline=False
    ).decode("ascii")


def _validated_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BircoOfficialHippoRAGError(f"{field} is invalid")
    return value


def validate_input(
    work_id: object,
    objective: object,
    query: object,
    documents: object,
    common_projection_sha256: object,
) -> tuple[str, str, str, tuple[CandidateDocument, ...], str]:
    """Validate one exact, label-free, controller-common candidate pool."""

    opaque_work_id = _bounded_text(
        work_id, "work_id", MAX_WORK_ID_CHARACTERS
    )
    objective_text = _bounded_text(
        objective, "objective", MAX_OBJECTIVE_CHARACTERS
    )
    query_text = _bounded_text(query, "query", MAX_QUERY_CHARACTERS)
    rows = _validate_documents(documents)
    claimed_hash = _validated_sha256(
        common_projection_sha256, "common projection SHA-256"
    )
    expected_hash = hashlib.sha256(
        canonical_json_bytes(
            _common_projection_payload(
                objective=objective_text, query=query_text, documents=rows
            ),
            newline=False,
        )
    ).hexdigest()
    if claimed_hash != expected_hash:
        raise BircoOfficialHippoRAGError("common projection SHA-256 mismatched")
    return opaque_work_id, objective_text, query_text, rows, claimed_hash


def serialize_document(row: CandidateDocument) -> str:
    """Make the source-free ordinal part of official content addressing."""

    if (
        not isinstance(row, CandidateDocument)
        or isinstance(row.ordinal, bool)
        or not isinstance(row.ordinal, int)
        or not 0 <= row.ordinal < MAX_CANDIDATE_COUNT
    ):
        raise BircoOfficialHippoRAGError("candidate document is malformed")
    text = _bounded_text(row.text, "candidate document text", MAX_DOCUMENT_CHARACTERS)
    return json.dumps(
        {
            "document_ordinal": row.ordinal,
            "schema": DOCUMENT_SCHEMA,
            "text": text,
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def serialize_documents(
    rows: Sequence[CandidateDocument],
) -> tuple[str, ...]:
    """Serialize a pool and reject any content-addressing collision."""

    documents = tuple(serialize_document(row) for row in rows)
    if len(set(documents)) != len(documents):
        raise BircoOfficialHippoRAGError("serialized candidates are duplicated")
    return documents


def _validate_document_mapping(
    document_to_ordinal: Mapping[str, int],
) -> int:
    if not isinstance(document_to_ordinal, Mapping):
        raise BircoOfficialHippoRAGError("candidate mapping is malformed")
    candidate_count = len(document_to_ordinal)
    if not MIN_CANDIDATE_COUNT <= candidate_count <= MAX_CANDIDATE_COUNT:
        raise BircoOfficialHippoRAGError("candidate mapping count drifted")
    ordinals = tuple(document_to_ordinal.values())
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in ordinals
        )
        or set(ordinals) != set(range(candidate_count))
        or any(not isinstance(document, str) for document in document_to_ordinal)
    ):
        raise BircoOfficialHippoRAGError("candidate mapping is not canonical")
    return candidate_count


def stable_permutation(
    *,
    retrieved_documents: object,
    retrieved_scores: object,
    document_to_ordinal: Mapping[str, int],
) -> tuple[int, ...]:
    """Return the full pool ordered by score descending, then ordinal."""

    candidate_count = _validate_document_mapping(document_to_ordinal)
    if isinstance(retrieved_documents, (str, bytes)) or isinstance(
        retrieved_scores, (str, bytes)
    ):
        raise BircoOfficialHippoRAGError("official result is malformed")
    try:
        documents = list(retrieved_documents)  # type: ignore[arg-type]
        scores = list(retrieved_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BircoOfficialHippoRAGError(
            "official result is not iterable"
        ) from exc
    if len(documents) != candidate_count or len(scores) != candidate_count:
        raise BircoOfficialHippoRAGError(
            "official result did not return all candidates"
        )

    ranked: list[tuple[float, int]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if not isinstance(document, str) or document not in document_to_ordinal:
            raise BircoOfficialHippoRAGError(
                "official result contains an unknown document"
            )
        if document in seen:
            raise BircoOfficialHippoRAGError(
                "official result contains a duplicate document"
            )
        if isinstance(score, bool) or not isinstance(score, Real):
            raise BircoOfficialHippoRAGError(
                "official result score is not numeric"
            )
        try:
            numeric = float(score)
        except (OverflowError, TypeError, ValueError) as exc:
            raise BircoOfficialHippoRAGError(
                "official result score is not finite"
            ) from exc
        if not math.isfinite(numeric):
            raise BircoOfficialHippoRAGError(
                "official result score is not finite"
            )
        seen.add(document)
        ranked.append((numeric, document_to_ordinal[document]))

    if seen != set(document_to_ordinal):
        raise BircoOfficialHippoRAGError(
            "official result omitted a candidate"
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    permutation = tuple(ordinal for _score, ordinal in ranked)
    if set(permutation) != set(range(candidate_count)):
        raise BircoOfficialHippoRAGError(
            "official full-pool permutation drifted"
        )
    return permutation


def output_payload(
    *,
    work_id: object,
    common_projection_sha256: object,
    candidate_count: object,
    rank_ordinals: Sequence[int],
    graph_nodes: object,
    graph_edges: object,
) -> dict[str, Any]:
    """Build the complete content-free worker result."""

    opaque_work_id = _bounded_text(
        work_id, "work_id", MAX_WORK_ID_CHARACTERS
    )
    projection_hash = _validated_sha256(
        common_projection_sha256, "common projection SHA-256"
    )
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or not MIN_CANDIDATE_COUNT
        <= candidate_count
        <= MAX_CANDIDATE_COUNT
    ):
        raise BircoOfficialHippoRAGError("candidate count is invalid")
    if isinstance(rank_ordinals, (str, bytes)) or not isinstance(
        rank_ordinals, Sequence
    ):
        raise BircoOfficialHippoRAGError("rank ordinals are malformed")
    values = tuple(rank_ordinals)
    if (
        len(values) != candidate_count
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        )
        or set(values) != set(range(candidate_count))
    ):
        raise BircoOfficialHippoRAGError(
            "rank ordinals are not a complete permutation"
        )
    for value, field in (
        (graph_nodes, "graph nodes"),
        (graph_edges, "graph edges"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BircoOfficialHippoRAGError(f"{field} is invalid")
    return {
        "candidate_count": candidate_count,
        "common_projection_sha256": projection_hash,
        "graph_edge_count": graph_edges,
        "graph_node_count": graph_nodes,
        "rank_ordinals": list(values),
        "schema": OUTPUT_SCHEMA,
        "work_id": opaque_work_id,
    }


def parse_output(raw: bytes) -> dict[str, Any]:
    """Parse only the exact canonical, source-text-free output schema."""

    if not isinstance(raw, bytes):
        raise BircoOfficialHippoRAGError("worker output is not bytes")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BircoOfficialHippoRAGError(
            "worker output is invalid JSON"
        ) from exc
    if not isinstance(value, Mapping) or set(value) != OUTPUT_KEYS:
        raise BircoOfficialHippoRAGError("worker output shape drifted")
    if canonical_json_bytes(value) != raw or value.get("schema") != OUTPUT_SCHEMA:
        raise BircoOfficialHippoRAGError("worker output is not canonical")
    return output_payload(
        work_id=value.get("work_id"),
        common_projection_sha256=value.get("common_projection_sha256"),
        candidate_count=value.get("candidate_count"),
        rank_ordinals=value.get("rank_ordinals", ()),  # type: ignore[arg-type]
        graph_nodes=value.get("graph_node_count"),
        graph_edges=value.get("graph_edge_count"),
    )

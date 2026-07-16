"""Gold-free MuSiQue paragraph and retrieval-output contract.

This module deliberately accepts one question and one candidate corpus at a
time.  Item identifiers, answers, aliases, and support labels are outside the
type and therefore cannot be forwarded to the official retrieval runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any, Mapping, Sequence


ADAPTER_VERSION = "musique_official_hipporag_retrieve_only_v1"
INPUT_SCHEMA = "musique_official_hipporag_single_item_input_v1"
DOCUMENT_SCHEMA = "musique_candidate_paragraph_v1"
TOP_K = 5
MAX_PARAGRAPHS = 128
MAX_TEXT_CHARACTERS = 250_000
PARAGRAPH_KEYS = frozenset({"idx", "title", "paragraph_text"})

FROZEN_CORE_CONFIG: dict[str, Any] = {
    "config_class": "hipporag.utils.config_utils.BaseConfig",
    "core_class": "hipporag.HippoRAG",
    "llm_backend": "Transformers/local_asset",
    "embedding_backend": "Transformers/local_asset",
    "openie_mode": "online",
    "max_new_tokens": 4,
    "retrieval_top_k": TOP_K,
    "qa_top_k": TOP_K,
    "force_index_from_scratch": True,
    "save_openie": True,
    "network_namespace": "isolated_no_transport",
    "official_retrieve_num_to_retrieve": "all_item_candidates",
    "adapter_top_k_selection": "negative_official_score_then_paragraph_idx_v1",
}


class MuSiQueOfficialHippoRAGError(RuntimeError):
    """Raised when the retrieve-only contract cannot be proven."""


@dataclass(frozen=True)
class CandidateParagraph:
    idx: int
    title: str
    paragraph_text: str


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MuSiQueOfficialHippoRAGError(f"{field} must be non-empty text")
    if "\x00" in value:
        raise MuSiQueOfficialHippoRAGError(f"{field} contains a NUL character")
    if len(value) > MAX_TEXT_CHARACTERS:
        raise MuSiQueOfficialHippoRAGError(f"{field} exceeds the frozen size bound")
    return value


def validate_single_item(
    question: str,
    paragraphs: Sequence[Mapping[str, object]],
) -> tuple[str, tuple[CandidateParagraph, ...]]:
    """Validate an exact, label-free, one-item candidate corpus."""

    normalized_question = _required_text(question, "question")
    if isinstance(paragraphs, (str, bytes)) or not isinstance(paragraphs, Sequence):
        raise MuSiQueOfficialHippoRAGError("paragraphs must be a sequence")
    if not TOP_K <= len(paragraphs) <= MAX_PARAGRAPHS:
        raise MuSiQueOfficialHippoRAGError("candidate corpus size is outside the frozen bounds")

    rows: list[CandidateParagraph] = []
    for position, raw in enumerate(paragraphs):
        if not isinstance(raw, Mapping) or set(raw) != PARAGRAPH_KEYS:
            raise MuSiQueOfficialHippoRAGError(
                "paragraph must contain only idx, title, and paragraph_text"
            )
        idx = raw.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx != position:
            raise MuSiQueOfficialHippoRAGError(
                "paragraph idx must be canonical contiguous zero-based order"
            )
        rows.append(
            CandidateParagraph(
                idx=idx,
                title=_required_text(raw.get("title"), f"paragraphs[{position}].title"),
                paragraph_text=_required_text(
                    raw.get("paragraph_text"),
                    f"paragraphs[{position}].paragraph_text",
                ),
            )
        )
    return normalized_question, tuple(rows)


def serialize_paragraph(paragraph: CandidateParagraph) -> str:
    """Serialize a candidate uniquely while preserving the official idx."""

    return json.dumps(
        {
            "paragraph_idx": paragraph.idx,
            "paragraph_text": paragraph.paragraph_text,
            "schema": DOCUMENT_SCHEMA,
            "title": paragraph.title,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def serialize_candidate_corpus(
    paragraphs: Sequence[CandidateParagraph],
) -> tuple[str, ...]:
    documents = tuple(serialize_paragraph(row) for row in paragraphs)
    if len(set(documents)) != len(documents):
        raise MuSiQueOfficialHippoRAGError("serialized candidates are not unique")
    return documents


def stable_top_five_from_official_result(
    *,
    retrieved_documents: Sequence[object],
    retrieved_scores: Sequence[object],
    document_to_idx: Mapping[str, int],
) -> tuple[int, ...]:
    """Map the complete official result to stable, exact top-five idx values."""

    if isinstance(retrieved_documents, (str, bytes)):
        raise MuSiQueOfficialHippoRAGError("official documents are malformed")
    if isinstance(retrieved_scores, (str, bytes)):
        raise MuSiQueOfficialHippoRAGError("official scores are malformed")
    try:
        document_rows = list(retrieved_documents)
        score_rows = list(retrieved_scores)
    except TypeError as exc:
        raise MuSiQueOfficialHippoRAGError("official result rows are not iterable") from exc
    if len(document_rows) != len(document_to_idx) or len(score_rows) != len(document_to_idx):
        raise MuSiQueOfficialHippoRAGError(
            "official retrieve must return every item-local candidate exactly once"
        )

    ranked: list[tuple[float, int]] = []
    seen_documents: set[str] = set()
    for document, score in zip(document_rows, score_rows):
        if not isinstance(document, str) or document not in document_to_idx:
            raise MuSiQueOfficialHippoRAGError(
                "official result contains a cross-corpus or unknown document"
            )
        if document in seen_documents:
            raise MuSiQueOfficialHippoRAGError("official result contains a duplicate document")
        seen_documents.add(document)
        if isinstance(score, bool) or not isinstance(score, Real):
            raise MuSiQueOfficialHippoRAGError("official result score is not numeric")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise MuSiQueOfficialHippoRAGError("official result score is not finite")
        ranked.append((numeric_score, document_to_idx[document]))

    if seen_documents != set(document_to_idx):
        raise MuSiQueOfficialHippoRAGError("official result omitted an item-local candidate")
    ranked.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(idx for _, idx in ranked[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise MuSiQueOfficialHippoRAGError("adapter did not produce five unique idx values")
    return result


def parse_idx_only_output(raw: bytes, *, candidate_count: int) -> tuple[int, ...]:
    """Parse the worker's entire output file as a JSON array of five idx values."""

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError("worker output is not canonical JSON") from exc
    if raw != (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8"):
        raise MuSiQueOfficialHippoRAGError("worker output is not canonical JSON")
    if not isinstance(value, list) or len(value) != TOP_K:
        raise MuSiQueOfficialHippoRAGError("worker output must contain exactly five idx values")
    result: list[int] = []
    for idx in value:
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < candidate_count:
            raise MuSiQueOfficialHippoRAGError("worker output idx is outside the item corpus")
        result.append(idx)
    if len(set(result)) != TOP_K:
        raise MuSiQueOfficialHippoRAGError("worker output contains duplicate idx values")
    return tuple(result)

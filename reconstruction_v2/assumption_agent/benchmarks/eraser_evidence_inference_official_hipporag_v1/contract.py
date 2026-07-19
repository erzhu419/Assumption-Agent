"""Frozen item-local sentence and ordinal-only HippoRAG contract.

Logical sentences are addressed by their zero-based position.  The official
core receives each unique *exact sentence string* once: no ordinal, title,
JSON envelope, or other prefix is added to an indexed document.  An official
ranked exact-text quotient is expanded rank by rank, with the logical
ordinals belonging to each quotient member emitted in ascending order, until
exactly five ordinals have been produced.

There is deliberately no 128-sentence cap and no truncation path.  Item
isolation is enforced by the adapter/worker's fresh, ephemeral index root.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from replication_runtime.musique_official_hipporag_v1.contract import (
    FROZEN_CORE_CONFIG as _MUSIQUE_FROZEN_CORE_CONFIG,
    TOP_K,
)


ADAPTER_VERSION = "eraser_evidence_inference_item_local_official_hipporag_v1"
INPUT_SCHEMA = "eraser_evidence_inference_hipporag_single_item_input_v1"
OUTPUT_SCHEMA = "canonical_json_array_of_five_logical_sentence_ordinals_v1"
DOCUMENT_SERIALIZATION = "exact_sentence_text_without_prefix_or_truncation_v1"
EXACT_TEXT_QUOTIENT_POLICY = (
    "first_logical_occurrence_exact_text_quotient_v1"
)
DUPLICATE_EXPANSION_POLICY = (
    "official_quotient_rank_then_each_text_logical_ordinals_ascending_to_five_v1"
)

FROZEN_CORE_CONFIG: dict[str, Any] = {
    **_MUSIQUE_FROZEN_CORE_CONFIG,
    "official_retrieve_num_to_retrieve": "all_item_local_unique_exact_texts",
    "official_document_serialization": DOCUMENT_SERIALIZATION,
    "logical_duplicate_expansion": DUPLICATE_EXPANSION_POLICY,
    "candidate_sentence_count_upper_bound": None,
    "candidate_sentence_truncation": False,
    "index_lifecycle": "fresh_force_from_scratch_then_destroy_per_item",
    "adapter_top_k_selection": DUPLICATE_EXPANSION_POLICY,
}


class EraserEvidenceInferenceOfficialHippoRAGError(RuntimeError):
    """The exact-text, item-local, ordinal-only contract cannot be proven."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            f"{field} must be non-empty text"
        )
    if "\x00" in value:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            f"{field} contains a NUL character"
        )
    return value


def validate_single_item(
    query: str,
    sentence_texts: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    """Validate one query and every exact sentence without a count ceiling."""

    validated_query = _required_text(query, "query")
    if isinstance(sentence_texts, (str, bytes)) or not isinstance(
        sentence_texts, Sequence
    ):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "sentence_texts must be a sequence"
        )
    if len(sentence_texts) < TOP_K:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "item must contain at least five logical sentences"
        )
    validated = tuple(
        _required_text(text, f"sentence_texts[{ordinal}]")
        for ordinal, text in enumerate(sentence_texts)
    )
    return validated_query, validated


def exact_text_quotient(
    sentence_texts: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    """Return first-occurrence quotient texts and ascending logical ordinals."""

    _query, validated = validate_single_item("quotient validation", sentence_texts)
    ordinal_lists: dict[str, list[int]] = {}
    content_hash_to_text: dict[str, str] = {}
    for ordinal, text in enumerate(validated):
        # HippoRAG's pinned passage store is addressed by the MD5 of exact text.
        # Prove that a distinct-string collision cannot silently merge entries.
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        prior = content_hash_to_text.setdefault(content_hash, text)
        if prior != text:
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "distinct exact sentence texts collide under the official content hash"
            )
        ordinal_lists.setdefault(text, []).append(ordinal)
    quotient = tuple(ordinal_lists)
    mapping = {text: tuple(ordinals) for text, ordinals in ordinal_lists.items()}
    if not quotient or any(quotient[position] != text for position, text in enumerate(mapping)):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "exact-text quotient order drifted"
        )
    return quotient, mapping


def expand_ranked_quotient_to_top_five(
    *,
    retrieved_documents: Sequence[object],
    document_to_ordinals: Mapping[str, Sequence[int]],
    logical_sentence_count: int,
) -> tuple[int, ...]:
    """Expand the complete official quotient in its returned rank order."""

    if isinstance(logical_sentence_count, bool) or not isinstance(
        logical_sentence_count, int
    ) or logical_sentence_count < TOP_K:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "logical sentence count is invalid"
        )
    if isinstance(retrieved_documents, (str, bytes)):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "official ranked quotient is malformed"
        )
    try:
        ranked_documents = list(retrieved_documents)
    except TypeError as exc:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "official ranked quotient is not iterable"
        ) from exc

    canonical_mapping: dict[str, tuple[int, ...]] = {}
    all_ordinals: list[int] = []
    for document, raw_ordinals in document_to_ordinals.items():
        if not isinstance(document, str) or not document:
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "exact-text quotient mapping contains an invalid document"
            )
        if isinstance(raw_ordinals, (str, bytes)):
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "exact-text quotient mapping contains malformed ordinals"
            )
        try:
            ordinals = tuple(raw_ordinals)
        except TypeError as exc:
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "exact-text quotient mapping contains malformed ordinals"
            ) from exc
        if (
            not ordinals
            or any(type(ordinal) is not int for ordinal in ordinals)
            or tuple(sorted(set(ordinals))) != ordinals
        ):
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "logical ordinals are not unique ascending integers"
            )
        canonical_mapping[document] = ordinals
        all_ordinals.extend(ordinals)
    if sorted(all_ordinals) != list(range(logical_sentence_count)):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "exact-text quotient does not partition the item-local ordinals"
        )
    if len(ranked_documents) != len(canonical_mapping):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "official retrieve omitted or added an exact-text quotient member"
        )

    seen: set[str] = set()
    for document in ranked_documents:
        if not isinstance(document, str) or document not in canonical_mapping:
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "official result contains an unknown or cross-item sentence text"
            )
        if document in seen:
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "official result contains a duplicate quotient member"
            )
        seen.add(document)
    if seen != set(canonical_mapping):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "official result is not the complete item-local exact-text quotient"
        )

    expanded: list[int] = []
    for document in ranked_documents:
        for ordinal in canonical_mapping[document]:
            expanded.append(ordinal)
            if len(expanded) == TOP_K:
                break
        if len(expanded) == TOP_K:
            break
    result = tuple(expanded)
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "rank expansion did not yield five unique logical ordinals"
        )
    return result


def parse_ordinals_only_output(
    raw: bytes,
    *,
    logical_sentence_count: int,
) -> tuple[int, ...]:
    """Parse the worker's entire output as five canonical JSON ordinals."""

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker output is not canonical JSON"
        ) from exc
    if raw != canonical_json_bytes(value):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker output is not canonical JSON"
        )
    if not isinstance(value, list) or len(value) != TOP_K:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker output must contain exactly five ordinals"
        )
    result: list[int] = []
    for ordinal in value:
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < logical_sentence_count
        ):
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "worker output ordinal is outside the item-local corpus"
            )
        result.append(ordinal)
    if len(set(result)) != TOP_K:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker output contains duplicate ordinals"
        )
    return tuple(result)


__all__ = [
    "ADAPTER_VERSION",
    "DOCUMENT_SERIALIZATION",
    "DUPLICATE_EXPANSION_POLICY",
    "EXACT_TEXT_QUOTIENT_POLICY",
    "EraserEvidenceInferenceOfficialHippoRAGError",
    "FROZEN_CORE_CONFIG",
    "INPUT_SCHEMA",
    "OUTPUT_SCHEMA",
    "TOP_K",
    "canonical_json_bytes",
    "exact_text_quotient",
    "expand_ranked_quotient_to_top_five",
    "parse_ordinals_only_output",
    "validate_single_item",
]

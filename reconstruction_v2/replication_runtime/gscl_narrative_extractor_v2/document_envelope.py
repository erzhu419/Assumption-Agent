"""Bounded byte-total orchestration over the frozen GSCL v2 leaf ABI.

The v2 extractor and its independent parser remain unchanged.  This module
partitions a root document into exact UTF-8 byte cores, delegates only
17--175 lexical-token cores to the existing ``select_story`` leaf surface,
and records an explicit outcome for every byte of the root document.

``byte-total`` is deliberately narrow: every admitted byte belongs to one
``EXTRACTED``, ``NO_RELATION``, ``CONTEXT_ONLY_SHORT_SENTENCE`` or
``TYPED_FAILURE`` segment.  It does *not* claim total relation recall or
invent endpoints for a sentence with fewer than three lexical tokens.  The
document result is a new envelope ABI and is never represented as the old
single-source ``NarrativeExtraction``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping, Protocol

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
)
from replication_runtime.gscl_narrative_extractor_v1.contract import (
    canonical_json_bytes,
    semantic_sha256,
)

from .closed_choice import (
    CLAIM_SCOPE as LEAF_CLAIM_SCOPE,
    MAXIMUM_CANDIDATES_PER_RELATION,
    MAXIMUM_RELATIONS_PER_EPISODE,
    PROMPT_CLOSURE_SHA256 as LEAF_PROMPT_CLOSURE_SHA256,
    RECEIPT_SCHEMA as LEAF_RECEIPT_SCHEMA,
    SCORING_BATCH_SIZE,
    VERSION as LEAF_VERSION,
    ClosedChoiceV2Decision,
)
from .contract import (
    ERROR_TAXONOMY,
    ClosedChoiceV2Abstention,
    ClosedChoiceV2Error,
)


VERSION = "gscl_narrative_document_envelope_v1"
RECEIPT_SCHEMA = f"{VERSION}.safe_receipt.v1"
SEGMENTATION_POLICY_VERSION = (
    "gscl_narrative_exact_byte_partition_ascii_cjk_v1"
)

MINIMUM_LEAF_LEXICAL_TOKENS = 17
MAXIMUM_LEAF_LEXICAL_TOKENS = 175
MAXIMUM_ROOT_LEXICAL_TOKENS = 1_024
MAXIMUM_ROOT_BYTES = 131_072
MAXIMUM_SEGMENTS = 128
MAXIMUM_EXTRACTABLE_SEGMENTS = 32
MAXIMUM_PROJECTED_RELATIONS = (
    MAXIMUM_EXTRACTABLE_SEGMENTS * MAXIMUM_RELATIONS_PER_EPISODE
)
MAXIMUM_PROJECTED_MENTIONS = 3 * MAXIMUM_PROJECTED_RELATIONS
MAXIMUM_CANDIDATES_PER_SINGLE_SENTENCE_LEAF = (
    2 * MAXIMUM_CANDIDATES_PER_RELATION + 11
)
MAXIMUM_FORWARD_BATCH_CALLS_PER_SINGLE_SENTENCE_LEAF = (
    2 * 16 + 3
)
MAXIMUM_DECLARED_CANDIDATES = (
    MAXIMUM_EXTRACTABLE_SEGMENTS
    * MAXIMUM_CANDIDATES_PER_SINGLE_SENTENCE_LEAF
)
MAXIMUM_DECLARED_FORWARD_BATCH_CALLS = (
    MAXIMUM_EXTRACTABLE_SEGMENTS
    * MAXIMUM_FORWARD_BATCH_CALLS_PER_SINGLE_SENTENCE_LEAF
)

_LEXICAL_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_BOUNDARY_CHARACTERS = frozenset(".?!\n。！？")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ENGINE_MARKER = object()
_DOCUMENT_TYPED_FAILURE_CODES = frozenset(
    {
        *(
            code
            for code in ERROR_TAXONOMY
            if code != "V2_PLAN_NO_RELATION_SELECTED"
        ),
        "DOCUMENT_ABORTED_AFTER_TYPED_FAILURE",
        "DOCUMENT_LEAF_RUNTIME_FAILED",
    }
)


SEGMENTATION_POLICY_SHA256 = semantic_sha256(
    {
        "boundary_characters": sorted(_BOUNDARY_CHARACTERS),
        "lexical_token_pattern": _LEXICAL_TOKEN.pattern,
        "maximum_extractable_segments": MAXIMUM_EXTRACTABLE_SEGMENTS,
        "maximum_leaf_lexical_tokens": MAXIMUM_LEAF_LEXICAL_TOKENS,
        "maximum_projected_mentions": MAXIMUM_PROJECTED_MENTIONS,
        "maximum_projected_relations": MAXIMUM_PROJECTED_RELATIONS,
        "maximum_root_bytes": MAXIMUM_ROOT_BYTES,
        "maximum_root_lexical_tokens": MAXIMUM_ROOT_LEXICAL_TOKENS,
        "maximum_segments": MAXIMUM_SEGMENTS,
        "minimum_leaf_lexical_tokens": MINIMUM_LEAF_LEXICAL_TOKENS,
        "leaf_prompt_closure_sha256": LEAF_PROMPT_CLOSURE_SHA256,
        "leaf_receipt_schema": LEAF_RECEIPT_SCHEMA,
        "leaf_version": LEAF_VERSION,
        "short_segment_policy": "context_only_no_synthetic_endpoint",
        "split_policy": "balanced_lexical_chunks_exact_byte_partition",
        "version": SEGMENTATION_POLICY_VERSION,
    }
)


class DocumentEnvelopeError(RuntimeError):
    """Stable document-envelope failure without source content."""

    def __init__(self, issue_id: str) -> None:
        if issue_id not in {
            "DOCUMENT_AUTHORITY_INVALID",
            "DOCUMENT_EXTRACTABLE_SEGMENT_CAPACITY_UNSUPPORTED",
            "DOCUMENT_GLOBAL_GROUNDING_INVALID",
            "DOCUMENT_LEAF_DECISION_INVALID",
            "DOCUMENT_OWNERSHIP_INVALID",
            "DOCUMENT_RECEIPT_INVALID",
            "DOCUMENT_RESOURCE_BOUND_EXCEEDED",
            "DOCUMENT_ROOT_INVALID",
            "DOCUMENT_SEGMENT_CAPACITY_UNSUPPORTED",
            "DOCUMENT_SEGMENT_TOPOLOGY_INVALID",
            "DOCUMENT_TOKEN_CAPACITY_UNSUPPORTED",
        }:
            raise ValueError("document_envelope_issue_id_unknown")
        self.issue_id = issue_id
        super().__init__(issue_id)


class SegmentDisposition(str, Enum):
    EXTRACTED = "EXTRACTED"
    NO_RELATION = "NO_RELATION"
    CONTEXT_ONLY_SHORT_SENTENCE = "CONTEXT_ONLY_SHORT_SENTENCE"
    TYPED_FAILURE = "TYPED_FAILURE"


class LeafSelector(Protocol):
    def select_story(self, story_text: str) -> ClosedChoiceV2Decision: ...


@dataclass(frozen=True, slots=True)
class _Token:
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    segment_id: str
    parent_sentence_id: str
    parent_start_byte: int
    parent_end_byte: int
    core_start_byte: int
    core_end_byte: int
    lexical_token_count: int
    chunk_index: int
    chunk_count: int
    leaf_eligible: bool


@dataclass(frozen=True, slots=True)
class ProjectedMention:
    mention_id: str
    segment_id: str
    parent_sentence_id: str
    kind: str
    quote: str = field(repr=False)
    occurrence: int
    start_byte: int
    end_byte: int
    quote_sha256: str
    leaf_mention_id: str


@dataclass(frozen=True, slots=True)
class ProjectedRelation:
    relation_id: str
    segment_id: str
    parent_sentence_id: str
    anchor_mention_id: str
    slot_mention_ids: tuple[str, str]
    generator_kind: str
    polarity: str
    temporal_orientation: str
    causal_orientation: str
    leaf_generator_id: str


@dataclass(frozen=True, slots=True)
class SegmentOutcome:
    plan: SegmentPlan
    disposition: SegmentDisposition
    leaf_called: bool
    error_code: str | None
    leaf_source_sha256: str | None
    leaf_decision_sha256: str | None
    leaf_receipt_sha256: str | None
    leaf_parser_provenance_hash: str | None
    reported_candidate_count: int
    reported_forward_batch_count: int
    mention_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    leaf_decision: ClosedChoiceV2Decision | None = field(
        repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class NarrativeDocumentEnvelopeV1:
    """Consistency-validated envelope; safe payload omits source text."""

    source_text: str = field(repr=False)
    segments: tuple[SegmentOutcome, ...]
    mentions: tuple[ProjectedMention, ...] = field(repr=False)
    relations: tuple[ProjectedRelation, ...] = field(repr=False)
    receipt_bytes: bytes

    def __post_init__(self) -> None:
        _validate_envelope(self)

    @property
    def receipt(self) -> Mapping[str, object]:
        try:
            value = json.loads(self.receipt_bytes.decode("ascii"))
        except Exception as exc:
            raise DocumentEnvelopeError(
                "DOCUMENT_RECEIPT_INVALID"
            ) from exc
        if type(value) is not dict:
            raise DocumentEnvelopeError("DOCUMENT_RECEIPT_INVALID")
        return MappingProxyType(value)

    @property
    def downstream_eligible(self) -> bool:
        return bool(self.receipt["downstream_eligible"])

    @property
    def partial_projection_available(self) -> bool:
        return bool(self.receipt["partial_projection_available"])

    def safe_payload(self) -> Mapping[str, object]:
        return self.receipt


def _balanced_sizes(total: int, count: int) -> tuple[int, ...]:
    base, remainder = divmod(total, count)
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(count)
    )


def _character_to_byte_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8", errors="strict"))
        offsets.append(total)
    return tuple(offsets)


def _lexical_tokens(
    text: str, character_to_byte: tuple[int, ...]
) -> tuple[_Token, ...]:
    return tuple(
        _Token(
            start_byte=character_to_byte[match.start()],
            end_byte=character_to_byte[match.end()],
        )
        for match in _LEXICAL_TOKEN.finditer(text)
    )


def _parent_character_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return an exact character partition using program-owned boundaries."""

    if not text:
        return ()
    spans: list[tuple[int, int]] = []
    start = 0
    lexical_since_start = False
    for index, character in enumerate(text):
        if _LEXICAL_TOKEN.fullmatch(character) is not None:
            lexical_since_start = True
        if character in _BOUNDARY_CHARACTERS and lexical_since_start:
            spans.append((start, index + 1))
            start = index + 1
            lexical_since_start = False
    if start < len(text):
        if lexical_since_start or not spans:
            spans.append((start, len(text)))
        else:
            left, _ = spans[-1]
            spans[-1] = (left, len(text))
    return tuple(spans)


def plan_document_segments(story_text: str) -> tuple[SegmentPlan, ...]:
    """Plan a deterministic, byte-exact root partition before model access."""

    if not isinstance(story_text, str) or not story_text:
        raise DocumentEnvelopeError("DOCUMENT_ROOT_INVALID")
    try:
        raw = story_text.encode("utf-8", errors="strict")
        character_to_byte = _character_to_byte_offsets(story_text)
    except UnicodeError as exc:
        raise DocumentEnvelopeError("DOCUMENT_ROOT_INVALID") from exc
    if len(raw) > MAXIMUM_ROOT_BYTES:
        raise DocumentEnvelopeError("DOCUMENT_ROOT_INVALID")
    tokens = _lexical_tokens(story_text, character_to_byte)
    if len(tokens) > MAXIMUM_ROOT_LEXICAL_TOKENS:
        raise DocumentEnvelopeError(
            "DOCUMENT_TOKEN_CAPACITY_UNSUPPORTED"
        )
    parent_character_spans = _parent_character_spans(story_text)
    if not parent_character_spans:
        raise DocumentEnvelopeError("DOCUMENT_ROOT_INVALID")

    specs: list[
        tuple[int, int, int, int, int, int, int, int]
    ] = []
    # parent index, parent left/right, core left/right, token count,
    # chunk index/count
    for parent_index, (left_character, right_character) in enumerate(
        parent_character_spans
    ):
        parent_left = character_to_byte[left_character]
        parent_right = character_to_byte[right_character]
        parent_tokens = tuple(
            token
            for token in tokens
            if parent_left <= token.start_byte
            and token.end_byte <= parent_right
        )
        token_count = len(parent_tokens)
        if token_count <= MAXIMUM_LEAF_LEXICAL_TOKENS:
            specs.append(
                (
                    parent_index,
                    parent_left,
                    parent_right,
                    parent_left,
                    parent_right,
                    token_count,
                    0,
                    1,
                )
            )
            continue
        chunk_count = math.ceil(
            token_count / MAXIMUM_LEAF_LEXICAL_TOKENS
        )
        sizes = _balanced_sizes(token_count, chunk_count)
        if any(
            not MINIMUM_LEAF_LEXICAL_TOKENS
            <= size
            <= MAXIMUM_LEAF_LEXICAL_TOKENS
            for size in sizes
        ):
            raise DocumentEnvelopeError(
                "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"
            )
        cursor = 0
        core_left = parent_left
        for chunk_index, size in enumerate(sizes):
            cursor += size
            core_right = (
                parent_right
                if chunk_index == chunk_count - 1
                else parent_tokens[cursor].start_byte
            )
            specs.append(
                (
                    parent_index,
                    parent_left,
                    parent_right,
                    core_left,
                    core_right,
                    size,
                    chunk_index,
                    chunk_count,
                )
            )
            core_left = core_right

    if not 1 <= len(specs) <= MAXIMUM_SEGMENTS:
        raise DocumentEnvelopeError(
            "DOCUMENT_SEGMENT_CAPACITY_UNSUPPORTED"
        )
    eligible_count = sum(
        MINIMUM_LEAF_LEXICAL_TOKENS <= row[5]
        for row in specs
    )
    if eligible_count > MAXIMUM_EXTRACTABLE_SEGMENTS:
        raise DocumentEnvelopeError(
            "DOCUMENT_EXTRACTABLE_SEGMENT_CAPACITY_UNSUPPORTED"
        )
    plans = tuple(
        SegmentPlan(
            segment_id=f"seg{index:03d}",
            parent_sentence_id=f"sent{parent_index:03d}",
            parent_start_byte=parent_left,
            parent_end_byte=parent_right,
            core_start_byte=core_left,
            core_end_byte=core_right,
            lexical_token_count=token_count,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            leaf_eligible=(
                MINIMUM_LEAF_LEXICAL_TOKENS
                <= token_count
                <= MAXIMUM_LEAF_LEXICAL_TOKENS
            ),
        )
        for index, (
            parent_index,
            parent_left,
            parent_right,
            core_left,
            core_right,
            token_count,
            chunk_index,
            chunk_count,
        ) in enumerate(specs)
    )
    _validate_plan(raw, plans)
    return plans


def _validate_plan(raw: bytes, plans: tuple[SegmentPlan, ...]) -> None:
    previous_end = 0
    parent_chunks: dict[str, list[SegmentPlan]] = {}
    for index, plan in enumerate(plans):
        if (
            type(plan) is not SegmentPlan
            or not isinstance(plan.segment_id, str)
            or not isinstance(plan.parent_sentence_id, str)
            or any(
                type(value) is not int
                for value in (
                    plan.parent_start_byte,
                    plan.parent_end_byte,
                    plan.core_start_byte,
                    plan.core_end_byte,
                    plan.lexical_token_count,
                    plan.chunk_index,
                    plan.chunk_count,
                )
            )
            or type(plan.leaf_eligible) is not bool
            or plan.segment_id != f"seg{index:03d}"
            or plan.core_start_byte != previous_end
            or not 0 <= plan.parent_start_byte <= plan.core_start_byte
            < plan.core_end_byte <= plan.parent_end_byte <= len(raw)
            or plan.chunk_count < 1
            or not 0 <= plan.chunk_index < plan.chunk_count
            or plan.leaf_eligible
            != (
                MINIMUM_LEAF_LEXICAL_TOKENS
                <= plan.lexical_token_count
                <= MAXIMUM_LEAF_LEXICAL_TOKENS
            )
        ):
            raise DocumentEnvelopeError(
                "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"
            )
        previous_end = plan.core_end_byte
        parent_chunks.setdefault(plan.parent_sentence_id, []).append(plan)
    if previous_end != len(raw):
        raise DocumentEnvelopeError(
            "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"
        )
    for parent_id, chunks in parent_chunks.items():
        first = chunks[0]
        if (
            any(row.parent_sentence_id != parent_id for row in chunks)
            or any(
                row.parent_start_byte != first.parent_start_byte
                or row.parent_end_byte != first.parent_end_byte
                or row.chunk_count != len(chunks)
                or row.chunk_index != index
                for index, row in enumerate(chunks)
            )
            or chunks[0].core_start_byte != first.parent_start_byte
            or chunks[-1].core_end_byte != first.parent_end_byte
        ):
            raise DocumentEnvelopeError(
                "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"
            )


def _quote_byte_positions(
    root_text: str,
    quote: str,
    character_to_byte: tuple[int, ...],
) -> tuple[int, ...]:
    positions: list[int] = []
    offset = 0
    while offset <= len(root_text):
        position = root_text.find(quote, offset)
        if position < 0:
            break
        positions.append(character_to_byte[position])
        offset = position + 1
    return tuple(positions)


def _validate_leaf_decision_receipt(
    leaf_text: str, decision: ClosedChoiceV2Decision
) -> None:
    """Validate the complete safe leaf receipt against its private decision."""

    try:
        receipt = dict(decision.receipt)
        self_sha256 = receipt.pop("self_sha256")
        wire_bytes = decision.wire_completion.encode(
            "ascii", errors="strict"
        )
        canonical_bytes = decision.canonical_completion.encode(
            "utf-8", errors="strict"
        )
    except (KeyError, TypeError, UnicodeError) as exc:
        raise DocumentEnvelopeError(
            "DOCUMENT_LEAF_DECISION_INVALID"
        ) from exc
    required = {
        "canonical_completion_commitment",
        "catalog_commitment",
        "claim_scope",
        "consumer_binding",
        "endpoint_selection_receipt_commitments",
        "exclusive_endpoint_ownership",
        "free_form_generation_count",
        "model_runtime_commitment",
        "prompt_closure_sha256",
        "resource_summary",
        "schema",
        "selected_answer_token_count",
        "slot_binding_semantics",
        "steps_commitment",
        "story_commitment",
        "version",
        "wire_commitment",
    }
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    resource = receipt.get("resource_summary")
    consumers = receipt.get("consumer_binding")
    endpoints = receipt.get(
        "endpoint_selection_receipt_commitments"
    )
    consumers_valid = (
        type(consumers) is dict
        and set(consumers)
        == {
            "flat_label_no_verifier",
            "full",
            "legacy_keyword",
            "semantic_only",
        }
        and all(value == canonical_sha256 for value in consumers.values())
    )
    resource_keys = {
        "candidate_count",
        "episode_count",
        "forward_batch_count",
        "maximum_candidates_in_one_batch",
        "maximum_span_lexical_width",
        "relation_count",
        "sentence_count",
    }
    resource_valid = (
        type(resource) is dict
        and set(resource) == resource_keys
        and all(type(value) is int for value in resource.values())
    )
    relation_count = (
        resource["relation_count"] if resource_valid else 0
    )
    endpoints_valid = (
        type(endpoints) is dict
        and set(endpoints)
        == {f"r{index:02d}" for index in range(relation_count)}
        and all(
            type(rows) is dict
            and set(rows) == {"anchor", "object0", "object1"}
            and all(
                isinstance(value, str)
                and _SHA256.fullmatch(value) is not None
                for value in rows.values()
            )
            for rows in endpoints.values()
        )
    )
    if (
        set(receipt) != required
        or not isinstance(self_sha256, str)
        or self_sha256 != semantic_sha256(receipt)
        or receipt.get("schema") != LEAF_RECEIPT_SCHEMA
        or receipt.get("version") != LEAF_VERSION
        or receipt.get("claim_scope") != LEAF_CLAIM_SCOPE
        or receipt.get("prompt_closure_sha256")
        != LEAF_PROMPT_CLOSURE_SHA256
        or receipt.get("exclusive_endpoint_ownership") is not True
        or receipt.get("free_form_generation_count") != 0
        or receipt.get("story_commitment")
        != hashlib.sha256(leaf_text.encode("utf-8")).hexdigest()
        or receipt.get("wire_commitment")
        != hashlib.sha256(wire_bytes).hexdigest()
        or receipt.get("canonical_completion_commitment")
        != canonical_sha256
        or decision.extraction.completion_sha256 != canonical_sha256
        or receipt.get("selected_answer_token_count")
        != decision.selected_answer_token_count
        or any(
            not isinstance(receipt.get(key), str)
            or _SHA256.fullmatch(receipt[key]) is None
            for key in (
                "catalog_commitment",
                "model_runtime_commitment",
                "steps_commitment",
            )
        )
        or not consumers_valid
        or not resource_valid
        or resource.get("maximum_candidates_in_one_batch")
        != SCORING_BATCH_SIZE
        or resource.get("maximum_span_lexical_width") != 4
        or resource.get("relation_count")
        != len(decision.extraction.generators)
        or resource.get("relation_count")
        != len(decision.extraction.mentions) // 3
        or not 1 <= resource.get("sentence_count", 0) <= 1
        or not 1 <= resource.get("episode_count", 0) <= 8
        or not 1
        <= resource.get("relation_count", 0)
        <= MAXIMUM_RELATIONS_PER_EPISODE
        or not 1
        <= resource.get("candidate_count", 0)
        <= MAXIMUM_CANDIDATES_PER_SINGLE_SENTENCE_LEAF
        or not 1
        <= resource.get("forward_batch_count", 0)
        <= MAXIMUM_FORWARD_BATCH_CALLS_PER_SINGLE_SENTENCE_LEAF
        or not endpoints_valid
    ):
        raise DocumentEnvelopeError(
            "DOCUMENT_LEAF_DECISION_INVALID"
        )


def _project_decision(
    *,
    root_text: str,
    root_bytes: bytes,
    plan: SegmentPlan,
    leaf_text: str,
    decision: ClosedChoiceV2Decision,
    occurrence_cache: dict[str, tuple[int, ...]],
    character_to_byte: tuple[int, ...],
) -> tuple[
    tuple[ProjectedMention, ...],
    tuple[ProjectedRelation, ...],
]:
    if type(decision) is not ClosedChoiceV2Decision:
        raise DocumentEnvelopeError("DOCUMENT_LEAF_DECISION_INVALID")
    extraction = decision.extraction
    if (
        type(extraction) is not NarrativeExtraction
        or extraction.source.text != leaf_text
        or not extraction.mentions
        or not extraction.generators
    ):
        raise DocumentEnvelopeError("DOCUMENT_LEAF_DECISION_INVALID")
    _validate_leaf_decision_receipt(leaf_text, decision)
    by_leaf_id: dict[str, ProjectedMention] = {}
    projected_mentions: list[ProjectedMention] = []
    for mention in extraction.mentions:
        start = plan.core_start_byte + mention.start_byte
        end = plan.core_start_byte + mention.end_byte
        try:
            quote_bytes = mention.quote.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise DocumentEnvelopeError(
                "DOCUMENT_GLOBAL_GROUNDING_INVALID"
            ) from exc
        if (
            not plan.core_start_byte <= start < end <= plan.core_end_byte
            or root_bytes[start:end] != quote_bytes
            or hashlib.sha256(quote_bytes).hexdigest()
            != mention.quote_sha256
        ):
            raise DocumentEnvelopeError(
                "DOCUMENT_GLOBAL_GROUNDING_INVALID"
            )
        positions = occurrence_cache.get(mention.quote)
        if positions is None:
            positions = _quote_byte_positions(
                root_text, mention.quote, character_to_byte
            )
            occurrence_cache[mention.quote] = positions
        try:
            occurrence = positions.index(start)
        except ValueError as exc:
            raise DocumentEnvelopeError(
                "DOCUMENT_GLOBAL_GROUNDING_INVALID"
            ) from exc
        identifier_payload = (
            mention.mention_id + ":" + str(start)
        ).encode("ascii")
        identifier = (
            f"{plan.segment_id}."
            f"{hashlib.sha256(identifier_payload).hexdigest()[:24]}"
        )
        projected = ProjectedMention(
            mention_id=identifier,
            segment_id=plan.segment_id,
            parent_sentence_id=plan.parent_sentence_id,
            kind=mention.kind.value,
            quote=mention.quote,
            occurrence=occurrence,
            start_byte=start,
            end_byte=end,
            quote_sha256=mention.quote_sha256,
            leaf_mention_id=mention.mention_id,
        )
        by_leaf_id[mention.mention_id] = projected
        projected_mentions.append(projected)

    projected_relations: list[ProjectedRelation] = []
    for generator in extraction.generators:
        if (
            generator.anchor_mention_id not in by_leaf_id
            or any(row not in by_leaf_id for row in generator.slot_mention_ids)
            or len(generator.slot_mention_ids) != 2
        ):
            raise DocumentEnvelopeError("DOCUMENT_OWNERSHIP_INVALID")
        anchor = by_leaf_id[generator.anchor_mention_id]
        slots = tuple(
            by_leaf_id[row].mention_id
            for row in generator.slot_mention_ids
        )
        relation_id = (
            f"{plan.segment_id}."
            f"{hashlib.sha256(generator.generator_id.encode('ascii')).hexdigest()[:24]}"
        )
        projected_relations.append(
            ProjectedRelation(
                relation_id=relation_id,
                segment_id=plan.segment_id,
                parent_sentence_id=plan.parent_sentence_id,
                anchor_mention_id=anchor.mention_id,
                slot_mention_ids=(slots[0], slots[1]),
                generator_kind=generator.generator_kind.value,
                polarity=generator.polarity.value,
                temporal_orientation=generator.temporal_orientation.value,
                causal_orientation=generator.causal_orientation.value,
                leaf_generator_id=generator.generator_id,
            )
        )
    return (
        tuple(
            sorted(
                projected_mentions,
                key=lambda row: (
                    row.start_byte,
                    row.end_byte,
                    row.mention_id,
                ),
            )
        ),
        tuple(
            sorted(
                projected_relations,
                key=lambda row: by_leaf_id[
                    next(
                        leaf_id
                        for leaf_id, projected in by_leaf_id.items()
                        if projected.mention_id == row.anchor_mention_id
                    )
                ].start_byte,
            )
        ),
    )


def _decision_sha256(decision: ClosedChoiceV2Decision) -> str:
    return semantic_sha256(
        {
            "canonical_completion_sha256": hashlib.sha256(
                decision.canonical_completion.encode("utf-8")
            ).hexdigest(),
            "receipt_sha256": hashlib.sha256(
                decision.receipt_bytes
            ).hexdigest(),
            "wire_completion_sha256": hashlib.sha256(
                decision.wire_completion.encode("ascii")
            ).hexdigest(),
        }
    )


def _segment_commitment(outcomes: tuple[SegmentOutcome, ...]) -> str:
    return semantic_sha256(
        [
            {
                "chunk_count": row.plan.chunk_count,
                "chunk_index": row.plan.chunk_index,
                "core_end_byte": row.plan.core_end_byte,
                "core_start_byte": row.plan.core_start_byte,
                "disposition": row.disposition.value,
                "error_code": row.error_code,
                "leaf_called": row.leaf_called,
                "leaf_decision_sha256": row.leaf_decision_sha256,
                "leaf_parser_provenance_hash": (
                    row.leaf_parser_provenance_hash
                ),
                "leaf_receipt_sha256": row.leaf_receipt_sha256,
                "leaf_source_sha256": row.leaf_source_sha256,
                "lexical_token_count": row.plan.lexical_token_count,
                "mention_count": len(row.mention_ids),
                "parent_end_byte": row.plan.parent_end_byte,
                "parent_sentence_id": row.plan.parent_sentence_id,
                "parent_start_byte": row.plan.parent_start_byte,
                "relation_count": len(row.relation_ids),
                "segment_id": row.plan.segment_id,
            }
            for row in outcomes
        ]
    )


def _projection_commitment(
    mentions: tuple[ProjectedMention, ...],
    relations: tuple[ProjectedRelation, ...],
) -> str:
    return semantic_sha256(
        {
            "mentions": [
                {
                    "end_byte": row.end_byte,
                    "kind": row.kind,
                    "mention_id": row.mention_id,
                    "occurrence": row.occurrence,
                    "parent_sentence_id": row.parent_sentence_id,
                    "quote_sha256": row.quote_sha256,
                    "segment_id": row.segment_id,
                    "start_byte": row.start_byte,
                }
                for row in mentions
            ],
            "relations": [
                {
                    "anchor_mention_id": row.anchor_mention_id,
                    "causal_orientation": row.causal_orientation,
                    "generator_kind": row.generator_kind,
                    "parent_sentence_id": row.parent_sentence_id,
                    "polarity": row.polarity,
                    "relation_id": row.relation_id,
                    "segment_id": row.segment_id,
                    "slot_mention_ids": list(row.slot_mention_ids),
                    "temporal_orientation": row.temporal_orientation,
                }
                for row in relations
            ],
        }
    )


def _receipt_body(
    *,
    source_text: str,
    outcomes: tuple[SegmentOutcome, ...],
    mentions: tuple[ProjectedMention, ...],
    relations: tuple[ProjectedRelation, ...],
) -> dict[str, object]:
    raw = source_text.encode("utf-8", errors="strict")
    disposition_counts = {
        disposition.value: sum(
            row.disposition is disposition for row in outcomes
        )
        for disposition in SegmentDisposition
    }
    leaf_calls = sum(row.leaf_called for row in outcomes)
    reported_candidates = sum(
        row.reported_candidate_count for row in outcomes
    )
    reported_forwards = sum(
        row.reported_forward_batch_count for row in outcomes
    )
    typed_failures = disposition_counts[
        SegmentDisposition.TYPED_FAILURE.value
    ]
    return {
        "byte_outcome_coverage_complete": True,
        "claim_scope": (
            "caller_bound_document_orchestration_consistency_only"
        ),
        "disposition_counts": disposition_counts,
        "downstream_eligible": False,
        "downstream_policy": (
            "blocked_until_bounded_set_level_envelope_consumer_is_qualified"
        ),
        "free_form_generation_count": 0,
        "leaf_abi": "gscl_narrative_hierarchical_closed_choice_v2",
        "leaf_prompt_closure_sha256": LEAF_PROMPT_CLOSURE_SHA256,
        "leaf_receipt_schema": LEAF_RECEIPT_SCHEMA,
        "leaf_version": LEAF_VERSION,
        "formal_leaf_authority_established": False,
        "projection_commitment": _projection_commitment(
            mentions, relations
        ),
        "partial_projection_available": (
            typed_failures == 0 and bool(relations)
        ),
        "private_leaf_evidence_required_for_validation": True,
        "relation_recall_total": False,
        "resource_summary": {
            "declared_candidate_bound": (
                leaf_calls
                * MAXIMUM_CANDIDATES_PER_SINGLE_SENTENCE_LEAF
            ),
            "declared_forward_batch_call_bound": (
                leaf_calls
                * MAXIMUM_FORWARD_BATCH_CALLS_PER_SINGLE_SENTENCE_LEAF
            ),
            "declared_leaf_batch_capacity": SCORING_BATCH_SIZE,
            "leaf_call_count": leaf_calls,
            "projected_mention_count": len(mentions),
            "projected_relation_count": len(relations),
            "reported_success_candidate_count": reported_candidates,
            "reported_success_forward_batch_count": reported_forwards,
            "root_byte_count": len(raw),
            "root_lexical_token_count": len(
                _LEXICAL_TOKEN.findall(source_text)
            ),
            "segment_count": len(outcomes),
        },
        "root_source_sha256": hashlib.sha256(raw).hexdigest(),
        "schema": RECEIPT_SCHEMA,
        "segmentation_policy_sha256": SEGMENTATION_POLICY_SHA256,
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "segments_commitment": _segment_commitment(outcomes),
        "semantic_short_segment_coverage_complete": (
            disposition_counts[
                SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE.value
            ]
            == 0
        ),
        "typed_failure_count": typed_failures,
        "version": VERSION,
    }


def _receipt_bytes(
    *,
    source_text: str,
    outcomes: tuple[SegmentOutcome, ...],
    mentions: tuple[ProjectedMention, ...],
    relations: tuple[ProjectedRelation, ...],
) -> bytes:
    body = _receipt_body(
        source_text=source_text,
        outcomes=outcomes,
        mentions=mentions,
        relations=relations,
    )
    return canonical_json_bytes(
        {**body, "self_sha256": semantic_sha256(body)}
    )


def _validate_envelope(envelope: NarrativeDocumentEnvelopeV1) -> None:
    try:
        raw = envelope.source_text.encode("utf-8", errors="strict")
    except (AttributeError, UnicodeError) as exc:
        raise DocumentEnvelopeError("DOCUMENT_ROOT_INVALID") from exc
    if (
        not raw
        or len(raw) > MAXIMUM_ROOT_BYTES
        or not isinstance(envelope.segments, tuple)
        or not isinstance(envelope.mentions, tuple)
        or not isinstance(envelope.relations, tuple)
    ):
        raise DocumentEnvelopeError("DOCUMENT_ROOT_INVALID")
    if len(_LEXICAL_TOKEN.findall(envelope.source_text)) > (
        MAXIMUM_ROOT_LEXICAL_TOKENS
    ):
        raise DocumentEnvelopeError(
            "DOCUMENT_TOKEN_CAPACITY_UNSUPPORTED"
        )
    if (
        not 1 <= len(envelope.segments) <= MAXIMUM_SEGMENTS
        or any(
            not isinstance(row, SegmentOutcome)
            for row in envelope.segments
        )
        or sum(row.plan.leaf_eligible for row in envelope.segments)
        > MAXIMUM_EXTRACTABLE_SEGMENTS
    ):
        raise DocumentEnvelopeError(
            "DOCUMENT_SEGMENT_CAPACITY_UNSUPPORTED"
        )
    plans = tuple(row.plan for row in envelope.segments)
    if plans != plan_document_segments(envelope.source_text):
        raise DocumentEnvelopeError(
            "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"
        )
    _validate_plan(raw, plans)
    plan_by_id = {row.segment_id: row for row in plans}
    if len(envelope.mentions) > MAXIMUM_PROJECTED_MENTIONS or len(
        envelope.relations
    ) > MAXIMUM_PROJECTED_RELATIONS:
        raise DocumentEnvelopeError("DOCUMENT_RESOURCE_BOUND_EXCEEDED")

    mention_by_id: dict[str, ProjectedMention] = {}
    occurrence_cache: dict[str, tuple[int, ...]] = {}
    character_to_byte = _character_to_byte_offsets(envelope.source_text)
    previous_interval: tuple[int, int] | None = None
    for mention in envelope.mentions:
        try:
            quote_bytes = mention.quote.encode(
                "utf-8", errors="strict"
            )
        except (AttributeError, UnicodeError) as exc:
            raise DocumentEnvelopeError(
                "DOCUMENT_GLOBAL_GROUNDING_INVALID"
            ) from exc
        if (
            type(mention) is not ProjectedMention
            or not isinstance(mention.mention_id, str)
            or not isinstance(mention.segment_id, str)
            or not isinstance(mention.parent_sentence_id, str)
            or not isinstance(mention.kind, str)
            or not isinstance(mention.quote, str)
            or not isinstance(mention.quote_sha256, str)
            or not isinstance(mention.leaf_mention_id, str)
            or type(mention.occurrence) is not int
            or mention.occurrence < 0
            or type(mention.start_byte) is not int
            or type(mention.end_byte) is not int
            or mention.mention_id in mention_by_id
            or mention.segment_id not in plan_by_id
            or not 0 <= mention.start_byte < mention.end_byte <= len(raw)
            or raw[mention.start_byte : mention.end_byte]
            != quote_bytes
            or hashlib.sha256(
                raw[mention.start_byte : mention.end_byte]
            ).hexdigest()
            != mention.quote_sha256
        ):
            raise DocumentEnvelopeError(
                "DOCUMENT_GLOBAL_GROUNDING_INVALID"
            )
        plan = plan_by_id[mention.segment_id]
        if (
            mention.parent_sentence_id != plan.parent_sentence_id
            or not plan.core_start_byte
            <= mention.start_byte
            < mention.end_byte
            <= plan.core_end_byte
            or mention.kind not in {"generator", "object"}
        ):
            raise DocumentEnvelopeError("DOCUMENT_OWNERSHIP_INVALID")
        if previous_interval is not None and (
            mention.start_byte < previous_interval[1]
        ):
            raise DocumentEnvelopeError("DOCUMENT_OWNERSHIP_INVALID")
        previous_interval = (mention.start_byte, mention.end_byte)
        mention_by_id[mention.mention_id] = mention
        positions = occurrence_cache.get(mention.quote)
        if positions is None:
            positions = _quote_byte_positions(
                envelope.source_text,
                mention.quote,
                character_to_byte,
            )
            occurrence_cache[mention.quote] = positions
        if (
            mention.occurrence >= len(positions)
            or positions[mention.occurrence] != mention.start_byte
        ):
            raise DocumentEnvelopeError(
                "DOCUMENT_GLOBAL_GROUNDING_INVALID"
            )

    used: set[str] = set()
    previous_anchor = -1
    relation_by_id: set[str] = set()
    for relation in envelope.relations:
        refs = (
            relation.anchor_mention_id,
            *relation.slot_mention_ids,
        )
        if (
            type(relation) is not ProjectedRelation
            or not isinstance(relation.relation_id, str)
            or not isinstance(relation.segment_id, str)
            or not isinstance(relation.parent_sentence_id, str)
            or not isinstance(relation.anchor_mention_id, str)
            or type(relation.slot_mention_ids) is not tuple
            or any(
                not isinstance(value, str)
                for value in relation.slot_mention_ids
            )
            or not isinstance(relation.generator_kind, str)
            or not isinstance(relation.polarity, str)
            or not isinstance(relation.temporal_orientation, str)
            or not isinstance(relation.causal_orientation, str)
            or not isinstance(relation.leaf_generator_id, str)
            or relation.relation_id in relation_by_id
            or len(set(refs)) != 3
            or any(ref not in mention_by_id for ref in refs)
            or any(ref in used for ref in refs)
            or relation.generator_kind
            not in {"relation", "state_change", "temporal", "causal"}
            or relation.polarity
            not in {"positive", "negative", "neutral"}
            or relation.temporal_orientation
            not in {"none", "forward", "reverse"}
            or relation.causal_orientation
            not in {"none", "forward", "reverse"}
        ):
            raise DocumentEnvelopeError("DOCUMENT_OWNERSHIP_INVALID")
        rows = tuple(mention_by_id[ref] for ref in refs)
        anchor = rows[0]
        if (
            any(row.segment_id != relation.segment_id for row in rows)
            or any(
                row.parent_sentence_id != relation.parent_sentence_id
                for row in rows
            )
            or anchor.start_byte < previous_anchor
            or anchor.kind != "generator"
            or any(row.kind != "object" for row in rows[1:])
        ):
            raise DocumentEnvelopeError("DOCUMENT_OWNERSHIP_INVALID")
        previous_anchor = anchor.start_byte
        used.update(refs)
        relation_by_id.add(relation.relation_id)
    if used != set(mention_by_id):
        raise DocumentEnvelopeError("DOCUMENT_OWNERSHIP_INVALID")

    mention_ids = set(mention_by_id)
    relation_ids = relation_by_id
    for outcome in envelope.segments:
        plan = outcome.plan
        segment_mentions = tuple(
            row
            for row in envelope.mentions
            if row.segment_id == plan.segment_id
        )
        segment_relations = tuple(
            row
            for row in envelope.relations
            if row.segment_id == plan.segment_id
        )
        expected_leaf_sha = hashlib.sha256(
            raw[plan.core_start_byte : plan.core_end_byte]
        ).hexdigest()
        if (
            not isinstance(outcome.leaf_called, bool)
            or isinstance(outcome.reported_candidate_count, bool)
            or not isinstance(outcome.reported_candidate_count, int)
            or outcome.reported_candidate_count < 0
            or isinstance(outcome.reported_forward_batch_count, bool)
            or not isinstance(
                outcome.reported_forward_batch_count, int
            )
            or outcome.reported_forward_batch_count < 0
            or not isinstance(outcome.mention_ids, tuple)
            or not isinstance(outcome.relation_ids, tuple)
            or len(set(outcome.mention_ids)) != len(outcome.mention_ids)
            or len(set(outcome.relation_ids)) != len(outcome.relation_ids)
            or outcome.mention_ids
            != tuple(row.mention_id for row in segment_mentions)
            or outcome.relation_ids
            != tuple(row.relation_id for row in segment_relations)
            or any(row not in mention_ids for row in outcome.mention_ids)
            or any(row not in relation_ids for row in outcome.relation_ids)
            or any(
                mention_by_id[row].segment_id != plan.segment_id
                for row in outcome.mention_ids
            )
            or any(
                next(
                    relation
                    for relation in envelope.relations
                    if relation.relation_id == row
                ).segment_id
                != plan.segment_id
                for row in outcome.relation_ids
            )
        ):
            raise DocumentEnvelopeError("DOCUMENT_OWNERSHIP_INVALID")
        if outcome.disposition is SegmentDisposition.EXTRACTED:
            decision = outcome.leaf_decision
            if (
                not outcome.leaf_called
                or not plan.leaf_eligible
                or type(decision) is not ClosedChoiceV2Decision
                or not outcome.mention_ids
                or not outcome.relation_ids
                or outcome.error_code is not None
                or outcome.leaf_source_sha256 != expected_leaf_sha
                or len(outcome.relation_ids)
                > MAXIMUM_RELATIONS_PER_EPISODE
                or len(outcome.mention_ids)
                != 3 * len(outcome.relation_ids)
                or outcome.reported_candidate_count
                > MAXIMUM_CANDIDATES_PER_SINGLE_SENTENCE_LEAF
                or outcome.reported_forward_batch_count
                > MAXIMUM_FORWARD_BATCH_CALLS_PER_SINGLE_SENTENCE_LEAF
                or any(
                    value is None
                    or _SHA256.fullmatch(value) is None
                    for value in (
                        outcome.leaf_source_sha256,
                        outcome.leaf_decision_sha256,
                        outcome.leaf_receipt_sha256,
                        outcome.leaf_parser_provenance_hash,
                    )
                )
            ):
                raise DocumentEnvelopeError(
                    "DOCUMENT_LEAF_DECISION_INVALID"
                )
            leaf_text = raw[
                plan.core_start_byte : plan.core_end_byte
            ].decode("utf-8", errors="strict")
            expected_mentions, expected_relations = _project_decision(
                root_text=envelope.source_text,
                root_bytes=raw,
                plan=plan,
                leaf_text=leaf_text,
                decision=decision,
                occurrence_cache=occurrence_cache,
                character_to_byte=character_to_byte,
            )
            resource = decision.receipt.get("resource_summary")
            if (
                expected_mentions != segment_mentions
                or expected_relations != segment_relations
                or outcome.leaf_decision_sha256
                != _decision_sha256(decision)
                or outcome.leaf_receipt_sha256
                != hashlib.sha256(decision.receipt_bytes).hexdigest()
                or outcome.leaf_parser_provenance_hash
                != decision.extraction.provenance_hash
                or type(resource) is not dict
                or resource.get("candidate_count")
                != outcome.reported_candidate_count
                or resource.get("forward_batch_count")
                != outcome.reported_forward_batch_count
            ):
                raise DocumentEnvelopeError(
                    "DOCUMENT_LEAF_DECISION_INVALID"
                )
        elif outcome.mention_ids or outcome.relation_ids:
            raise DocumentEnvelopeError("DOCUMENT_OWNERSHIP_INVALID")
        elif outcome.disposition is SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE:
            if (
                outcome.leaf_called
                or plan.leaf_eligible
                or outcome.leaf_decision is not None
                or outcome.reported_candidate_count != 0
                or outcome.reported_forward_batch_count != 0
                or outcome.leaf_source_sha256 is not None
                or any(
                    value is not None
                    for value in (
                        outcome.leaf_decision_sha256,
                        outcome.leaf_receipt_sha256,
                        outcome.leaf_parser_provenance_hash,
                    )
                )
            ):
                raise DocumentEnvelopeError(
                    "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"
                )
        elif outcome.disposition is SegmentDisposition.NO_RELATION:
            if (
                not outcome.leaf_called
                or not plan.leaf_eligible
                or outcome.leaf_decision is not None
                or outcome.error_code != "V2_PLAN_NO_RELATION_SELECTED"
                or outcome.reported_candidate_count != 0
                or outcome.reported_forward_batch_count != 0
                or outcome.leaf_source_sha256 != expected_leaf_sha
                or any(
                    value is not None
                    for value in (
                        outcome.leaf_decision_sha256,
                        outcome.leaf_receipt_sha256,
                        outcome.leaf_parser_provenance_hash,
                    )
                )
            ):
                raise DocumentEnvelopeError(
                    "DOCUMENT_LEAF_DECISION_INVALID"
                )
        elif outcome.disposition is SegmentDisposition.TYPED_FAILURE:
            if (
                not plan.leaf_eligible
                or not outcome.error_code
                or outcome.error_code not in _DOCUMENT_TYPED_FAILURE_CODES
                or outcome.leaf_decision is not None
                or outcome.reported_candidate_count != 0
                or outcome.reported_forward_batch_count != 0
                or outcome.leaf_source_sha256 != expected_leaf_sha
                or any(
                    value is not None
                    for value in (
                        outcome.leaf_decision_sha256,
                        outcome.leaf_receipt_sha256,
                        outcome.leaf_parser_provenance_hash,
                    )
                )
            ):
                raise DocumentEnvelopeError(
                    "DOCUMENT_LEAF_DECISION_INVALID"
                )
            if (
                outcome.error_code
                == "DOCUMENT_ABORTED_AFTER_TYPED_FAILURE"
            ) != (not outcome.leaf_called):
                raise DocumentEnvelopeError(
                    "DOCUMENT_LEAF_DECISION_INVALID"
                )
        else:
            raise DocumentEnvelopeError("DOCUMENT_LEAF_DECISION_INVALID")

    expected = _receipt_bytes(
        source_text=envelope.source_text,
        outcomes=envelope.segments,
        mentions=envelope.mentions,
        relations=envelope.relations,
    )
    if envelope.receipt_bytes != expected:
        raise DocumentEnvelopeError("DOCUMENT_RECEIPT_INVALID")
    receipt = json.loads(expected.decode("ascii"))
    resource = receipt["resource_summary"]
    if (
        resource["declared_candidate_bound"]
        > MAXIMUM_DECLARED_CANDIDATES
        or resource["declared_forward_batch_call_bound"]
        > MAXIMUM_DECLARED_FORWARD_BATCH_CALLS
        or resource["reported_success_candidate_count"]
        > resource["declared_candidate_bound"]
        or resource["reported_success_forward_batch_count"]
        > resource["declared_forward_batch_call_bound"]
        or (
            receipt["typed_failure_count"] > 0
            and receipt["downstream_eligible"]
        )
    ):
        raise DocumentEnvelopeError("DOCUMENT_RESOURCE_BOUND_EXCEEDED")


class _DocumentEnvelopeEngine:
    __slots__ = ("_marker",)

    def __init__(self, marker: object) -> None:
        if marker is not _ENGINE_MARKER:
            raise DocumentEnvelopeError("DOCUMENT_AUTHORITY_INVALID")
        self._marker = marker

    def select(
        self, story_text: str, *, leaf_selector: LeafSelector
    ) -> NarrativeDocumentEnvelopeV1:
        if (
            type(self) is not _DocumentEnvelopeEngine
            or self._marker is not _ENGINE_MARKER
            or not callable(getattr(leaf_selector, "select_story", None))
        ):
            raise DocumentEnvelopeError("DOCUMENT_AUTHORITY_INVALID")
        plans = plan_document_segments(story_text)
        root_bytes = story_text.encode("utf-8", errors="strict")
        character_to_byte = _character_to_byte_offsets(story_text)
        occurrence_cache: dict[str, tuple[int, ...]] = {}
        outcomes: list[SegmentOutcome] = []
        mentions: list[ProjectedMention] = []
        relations: list[ProjectedRelation] = []
        abort_after_failure = False
        for plan in plans:
            leaf_bytes = root_bytes[
                plan.core_start_byte : plan.core_end_byte
            ]
            try:
                leaf_text = leaf_bytes.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise DocumentEnvelopeError(
                    "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"
                ) from exc
            leaf_sha = hashlib.sha256(leaf_bytes).hexdigest()
            if not plan.leaf_eligible:
                outcomes.append(
                    SegmentOutcome(
                        plan=plan,
                        disposition=(
                            SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE
                        ),
                        leaf_called=False,
                        error_code=None,
                        leaf_source_sha256=None,
                        leaf_decision_sha256=None,
                        leaf_receipt_sha256=None,
                        leaf_parser_provenance_hash=None,
                        reported_candidate_count=0,
                        reported_forward_batch_count=0,
                        mention_ids=(),
                        relation_ids=(),
                        leaf_decision=None,
                    )
                )
                continue
            if abort_after_failure:
                outcomes.append(
                    SegmentOutcome(
                        plan=plan,
                        disposition=SegmentDisposition.TYPED_FAILURE,
                        leaf_called=False,
                        error_code="DOCUMENT_ABORTED_AFTER_TYPED_FAILURE",
                        leaf_source_sha256=leaf_sha,
                        leaf_decision_sha256=None,
                        leaf_receipt_sha256=None,
                        leaf_parser_provenance_hash=None,
                        reported_candidate_count=0,
                        reported_forward_batch_count=0,
                        mention_ids=(),
                        relation_ids=(),
                        leaf_decision=None,
                    )
                )
                continue
            try:
                decision = leaf_selector.select_story(leaf_text)
            except ClosedChoiceV2Abstention as exc:
                if exc.issue_id == "V2_PLAN_NO_RELATION_SELECTED":
                    outcomes.append(
                        SegmentOutcome(
                            plan=plan,
                            disposition=SegmentDisposition.NO_RELATION,
                            leaf_called=True,
                            error_code=exc.issue_id,
                            leaf_source_sha256=leaf_sha,
                            leaf_decision_sha256=None,
                            leaf_receipt_sha256=None,
                            leaf_parser_provenance_hash=None,
                            reported_candidate_count=0,
                            reported_forward_batch_count=0,
                            mention_ids=(),
                            relation_ids=(),
                            leaf_decision=None,
                        )
                    )
                    continue
                outcomes.append(
                    SegmentOutcome(
                        plan=plan,
                        disposition=SegmentDisposition.TYPED_FAILURE,
                        leaf_called=True,
                        error_code=exc.issue_id,
                        leaf_source_sha256=leaf_sha,
                        leaf_decision_sha256=None,
                        leaf_receipt_sha256=None,
                        leaf_parser_provenance_hash=None,
                        reported_candidate_count=0,
                        reported_forward_batch_count=0,
                        mention_ids=(),
                        relation_ids=(),
                        leaf_decision=None,
                    )
                )
                abort_after_failure = True
                continue
            except ClosedChoiceV2Error as exc:
                outcomes.append(
                    SegmentOutcome(
                        plan=plan,
                        disposition=SegmentDisposition.TYPED_FAILURE,
                        leaf_called=True,
                        error_code=exc.issue_id,
                        leaf_source_sha256=leaf_sha,
                        leaf_decision_sha256=None,
                        leaf_receipt_sha256=None,
                        leaf_parser_provenance_hash=None,
                        reported_candidate_count=0,
                        reported_forward_batch_count=0,
                        mention_ids=(),
                        relation_ids=(),
                        leaf_decision=None,
                    )
                )
                abort_after_failure = True
                continue
            except Exception:
                outcomes.append(
                    SegmentOutcome(
                        plan=plan,
                        disposition=SegmentDisposition.TYPED_FAILURE,
                        leaf_called=True,
                        error_code="DOCUMENT_LEAF_RUNTIME_FAILED",
                        leaf_source_sha256=leaf_sha,
                        leaf_decision_sha256=None,
                        leaf_receipt_sha256=None,
                        leaf_parser_provenance_hash=None,
                        reported_candidate_count=0,
                        reported_forward_batch_count=0,
                        mention_ids=(),
                        relation_ids=(),
                        leaf_decision=None,
                    )
                )
                abort_after_failure = True
                continue

            projected_mentions, projected_relations = _project_decision(
                root_text=story_text,
                root_bytes=root_bytes,
                plan=plan,
                leaf_text=leaf_text,
                decision=decision,
                occurrence_cache=occurrence_cache,
                character_to_byte=character_to_byte,
            )
            resource = decision.receipt.get("resource_summary")
            if type(resource) is not dict:
                raise DocumentEnvelopeError(
                    "DOCUMENT_LEAF_DECISION_INVALID"
                )
            candidate_count = resource.get("candidate_count")
            forward_count = resource.get("forward_batch_count")
            if (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count < 0
                or isinstance(forward_count, bool)
                or not isinstance(forward_count, int)
                or forward_count < 0
            ):
                raise DocumentEnvelopeError(
                    "DOCUMENT_LEAF_DECISION_INVALID"
                )
            mentions.extend(projected_mentions)
            relations.extend(projected_relations)
            outcomes.append(
                SegmentOutcome(
                    plan=plan,
                    disposition=SegmentDisposition.EXTRACTED,
                    leaf_called=True,
                    error_code=None,
                    leaf_source_sha256=leaf_sha,
                    leaf_decision_sha256=_decision_sha256(decision),
                    leaf_receipt_sha256=hashlib.sha256(
                        decision.receipt_bytes
                    ).hexdigest(),
                    leaf_parser_provenance_hash=(
                        decision.extraction.provenance_hash
                    ),
                    reported_candidate_count=candidate_count,
                    reported_forward_batch_count=forward_count,
                    mention_ids=tuple(
                        row.mention_id for row in projected_mentions
                    ),
                    relation_ids=tuple(
                        row.relation_id for row in projected_relations
                    ),
                    leaf_decision=decision,
                )
            )

        mentions_tuple = tuple(
            sorted(
                mentions,
                key=lambda row: (
                    row.start_byte,
                    row.end_byte,
                    row.mention_id,
                ),
            )
        )
        mention_by_id = {
            row.mention_id: row for row in mentions_tuple
        }
        relations_tuple = tuple(
            sorted(
                relations,
                key=lambda row: (
                    mention_by_id[row.anchor_mention_id].start_byte,
                    row.relation_id,
                ),
            )
        )
        outcomes_tuple = tuple(outcomes)
        if (
            len(mentions_tuple) > MAXIMUM_PROJECTED_MENTIONS
            or len(relations_tuple) > MAXIMUM_PROJECTED_RELATIONS
        ):
            raise DocumentEnvelopeError(
                "DOCUMENT_RESOURCE_BOUND_EXCEEDED"
            )
        receipt = _receipt_bytes(
            source_text=story_text,
            outcomes=outcomes_tuple,
            mentions=mentions_tuple,
            relations=relations_tuple,
        )
        return NarrativeDocumentEnvelopeV1(
            source_text=story_text,
            segments=outcomes_tuple,
            mentions=mentions_tuple,
            relations=relations_tuple,
            receipt_bytes=receipt,
        )


def select_document_qualification_only(
    story_text: str, *, leaf_selector: LeafSelector
) -> NarrativeDocumentEnvelopeV1:
    """Source-free qualification surface with an explicit fake leaf."""

    return _DocumentEnvelopeEngine(_ENGINE_MARKER).select(
        story_text, leaf_selector=leaf_selector
    )


def select_document_runtime_only(
    story_text: str, *, runtime: object
) -> NarrativeDocumentEnvelopeV1:
    """Type-bound runtime surface; a formal wrapper must bind exact assets.

    ``MemorySafeQwenRuntime`` also has an explicit qualification-fake
    constructor, so exact CUDA/asset custody cannot honestly be inferred from
    Python type alone.  A later fixed formal wrapper must establish that
    independent binding before calling this narrow surface.
    """

    from .memory_safe_qwen import MemorySafeQwenRuntime

    if type(runtime) is not MemorySafeQwenRuntime:
        raise DocumentEnvelopeError("DOCUMENT_AUTHORITY_INVALID")
    return _DocumentEnvelopeEngine(_ENGINE_MARKER).select(
        story_text, leaf_selector=runtime
    )


__all__ = [
    "MAXIMUM_EXTRACTABLE_SEGMENTS",
    "MAXIMUM_LEAF_LEXICAL_TOKENS",
    "MAXIMUM_PROJECTED_MENTIONS",
    "MAXIMUM_PROJECTED_RELATIONS",
    "MAXIMUM_ROOT_BYTES",
    "MAXIMUM_ROOT_LEXICAL_TOKENS",
    "MAXIMUM_SEGMENTS",
    "MINIMUM_LEAF_LEXICAL_TOKENS",
    "NarrativeDocumentEnvelopeV1",
    "DocumentEnvelopeError",
    "ProjectedMention",
    "ProjectedRelation",
    "RECEIPT_SCHEMA",
    "SEGMENTATION_POLICY_SHA256",
    "SEGMENTATION_POLICY_VERSION",
    "SegmentDisposition",
    "SegmentOutcome",
    "SegmentPlan",
    "VERSION",
    "plan_document_segments",
    "select_document_runtime_only",
    "select_document_qualification_only",
]

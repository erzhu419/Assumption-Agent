"""Grounded, source-free narrative proposal consistency.

A generator completion is untrusted.  The trusted boundary is deliberately
narrow: strict JSON, exact quote/occurrence grounding, deterministic
canonical identifiers, bounded injective search, and a score-free structural
checker.

The resulting :class:`CorrespondenceCertificate` certifies only
*proposal-internal structural consistency*.  It does not establish that a
proposed relation is narratively true, semantically correct, or the intended
interpretation of the source.  Deterministic sentence and relation coverage
checks reduce trivial cherry-picking, but are not a semantic proof.

Semantic scores form bounded domains and rank the flat envelope.  They are
absent from :class:`StructuralMapping`, the checker input, and the certificate
objective.  Mapping results bind the complete proposal set, score table, and
search configuration.  The full arm adds only the fixed checker; neither a
checker callback nor a motif hook is exposed.

This module performs no model or network call and opens no benchmark source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "gscl.narrative.extraction.v1"
CORE_VERSION = "gscl.narrative.proposal_consistency.v2"
PARSER_VERSION = "gscl.narrative.strict_parser.v2"

MAX_SOURCE_BYTES = 131_072
MAX_COMPLETION_BYTES = 65_536
MAX_JSON_DEPTH = 8
MAX_JSON_NODES = 2_048
MAX_MENTIONS = 64
MAX_GENERATORS = 64
MAX_QUOTE_BYTES = 4_096
MAX_SLOTS = 4
MAX_JSON_INTEGER_ABS = 1_000_000_000
MAX_SCORE_ABS = 1_000_000_000
MAX_AGGREGATE_SCORE_ABS = MAX_SCORE_ABS * (
    MAX_MENTIONS + MAX_GENERATORS
)
MAX_SCORE_ROWS = MAX_MENTIONS * MAX_MENTIONS
MAX_TOP_K = 16
MAX_SEARCH_ASSIGNMENTS = 100_000
MAX_OPERATORS = 16

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{1,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_KEY_TOKEN = re.compile(
    r"(?:^|_)(?:answer|answers|choice|choices|correct|gold|label|labels|"
    r"law|laws|law_id|motif|motifs|verdict|solution)(?:_|$)"
)
_PROMPT_INJECTION = re.compile(
    r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
    r"system\s+prompt|developer\s+message|"
    r"(?:correct\s+)?answer\s*(?:is|:|=)|"
    r"(?:correct\s+)?choice\s*(?:is|:|=))"
)
_ALPHANUMERIC = re.compile(r"\w", re.UNICODE)


class NarrativeContractError(ValueError):
    """Stable fail-closed error that never embeds private source text."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def _strict_json_check(value: Any) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if isinstance(value, list):
        for child in value:
            _strict_json_check(child)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("safe payload key must be a string")
        for child in value.values():
            _strict_json_check(child)
        return
    raise TypeError(f"non-strict safe payload type {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    _strict_json_check(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_identifier(value: object, issue_id: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise NarrativeContractError(issue_id)
    return value


def _require_hash(value: object, issue_id: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise NarrativeContractError(issue_id)
    return value


def _strict_int(value: object, issue_id: str, *, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or abs(value) > MAX_JSON_INTEGER_ABS
    ):
        raise NarrativeContractError(issue_id)
    return value


class MentionKind(str, Enum):
    OBJECT = "object"
    GENERATOR = "generator"


class GeneratorKind(str, Enum):
    RELATION = "relation"
    STATE_CHANGE = "state_change"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


class SignedState(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"

    @property
    def sign(self) -> int:
        return {
            SignedState.NEGATIVE: -1,
            SignedState.NEUTRAL: 0,
            SignedState.POSITIVE: 1,
        }[self]


class Orientation(str, Enum):
    REVERSE = "reverse"
    NONE = "none"
    FORWARD = "forward"

    @property
    def sign(self) -> int:
        return {
            Orientation.REVERSE: -1,
            Orientation.NONE: 0,
            Orientation.FORWARD: 1,
        }[self]


class OrientationMode(str, Enum):
    """Global orientation behavior; no categorical claim is implied."""

    PRESERVING = "orientation_preserving"
    INVERTING = "orientation_inverting"


class SlotPermutation(str, Enum):
    IDENTITY = "identity"
    REVERSE = "reverse"
    ROTATE_LEFT = "rotate_left"


class ArmName(str, Enum):
    FLAT = "flat"
    FULL = "full"


class ChoiceDisposition(str, Enum):
    SELECTED = "selected"
    PROPOSAL_STRUCTURALLY_CONTRADICTED = (
        "proposal_structurally_contradicted"
    )
    ABSTAIN = "abstain"


class CertificateDisposition(str, Enum):
    PROPOSAL_INTERNALLY_CONSISTENT = "proposal_internally_consistent"
    PROPOSAL_STRUCTURALLY_CONTRADICTED = (
        "proposal_structurally_contradicted"
    )


@dataclass(frozen=True)
class NarrativeSource:
    source_id: str
    text: str

    def __post_init__(self) -> None:
        _require_identifier(self.source_id, "source_id_invalid")
        if not isinstance(self.text, str) or not self.text or "\x00" in self.text:
            raise NarrativeContractError("source_text_invalid")
        try:
            encoded = self.text.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise NarrativeContractError("source_utf8_invalid") from exc
        if len(encoded) > MAX_SOURCE_BYTES:
            raise NarrativeContractError("source_too_large")

    @classmethod
    def from_utf8_bytes(
        cls, source_id: str, payload: bytes
    ) -> "NarrativeSource":
        if not isinstance(payload, bytes):
            raise NarrativeContractError("source_bytes_invalid")
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise NarrativeContractError("source_utf8_invalid") from exc
        return cls(source_id=source_id, text=text)

    @property
    def utf8_bytes(self) -> bytes:
        return self.text.encode("utf-8")

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.utf8_bytes).hexdigest()

    def private_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "byte_length": len(self.utf8_bytes),
        }


def _quote_positions(text: str, quote: str) -> tuple[int, ...]:
    positions: list[int] = []
    offset = 0
    while offset <= len(text):
        position = text.find(quote, offset)
        if position < 0:
            break
        positions.append(position)
        offset = position + 1
    return tuple(positions)


def _canonical_mention_id(
    kind: MentionKind,
    start_byte: int,
    end_byte: int,
    quote_sha256: str,
) -> str:
    commitment = _content_hash(
        {
            "kind": kind.value,
            "start_byte": start_byte,
            "end_byte": end_byte,
            "quote_sha256": quote_sha256,
        }
    )
    prefix = "object" if kind is MentionKind.OBJECT else "anchor"
    return f"{prefix}.{commitment[:24]}"


@dataclass(frozen=True)
class Mention:
    mention_id: str
    kind: MentionKind
    quote: str
    occurrence: int
    start_byte: int
    end_byte: int
    quote_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.mention_id, "mention_id_invalid")
        if not isinstance(self.kind, MentionKind):
            raise NarrativeContractError("mention_kind_invalid")
        if not isinstance(self.quote, str) or not self.quote:
            raise NarrativeContractError("mention_quote_invalid")
        _strict_int(self.occurrence, "mention_occurrence_invalid")
        _strict_int(self.start_byte, "mention_start_invalid")
        if (
            not isinstance(self.end_byte, int)
            or isinstance(self.end_byte, bool)
            or self.end_byte <= self.start_byte
            or self.end_byte > MAX_SOURCE_BYTES
        ):
            raise NarrativeContractError("mention_end_invalid")
        _require_hash(self.quote_sha256, "mention_quote_hash_invalid")
        if self.mention_id != _canonical_mention_id(
            self.kind,
            self.start_byte,
            self.end_byte,
            self.quote_sha256,
        ):
            raise NarrativeContractError("mention_id_not_canonical")

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "kind": self.kind.value,
            "occurrence": self.occurrence,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "quote_sha256": self.quote_sha256,
        }


def _canonical_generator_id(
    *,
    anchor_mention_id: str,
    slot_mention_ids: tuple[str, ...],
    generator_kind: GeneratorKind,
    polarity: SignedState,
    temporal_orientation: Orientation,
    causal_orientation: Orientation,
) -> str:
    commitment = _content_hash(
        {
            "anchor_mention_id": anchor_mention_id,
            "slot_mention_ids": list(slot_mention_ids),
            "generator_kind": generator_kind.value,
            "polarity": polarity.value,
            "temporal_orientation": temporal_orientation.value,
            "causal_orientation": causal_orientation.value,
        }
    )
    return f"generator.{commitment[:24]}"


@dataclass(frozen=True)
class GeneratorProposal:
    generator_id: str
    anchor_mention_id: str
    slot_mention_ids: tuple[str, ...]
    generator_kind: GeneratorKind
    polarity: SignedState
    temporal_orientation: Orientation
    causal_orientation: Orientation

    def __post_init__(self) -> None:
        _require_identifier(self.generator_id, "generator_id_invalid")
        _require_identifier(
            self.anchor_mention_id, "generator_anchor_invalid"
        )
        if (
            not isinstance(self.slot_mention_ids, tuple)
            or not 2 <= len(self.slot_mention_ids) <= MAX_SLOTS
            or len(set(self.slot_mention_ids)) != len(self.slot_mention_ids)
        ):
            raise NarrativeContractError("generator_slots_invalid")
        for slot in self.slot_mention_ids:
            _require_identifier(slot, "generator_slot_ref_invalid")
        if not isinstance(self.generator_kind, GeneratorKind):
            raise NarrativeContractError("generator_kind_invalid")
        if not isinstance(self.polarity, SignedState):
            raise NarrativeContractError("generator_polarity_invalid")
        if not isinstance(self.temporal_orientation, Orientation):
            raise NarrativeContractError("generator_temporal_invalid")
        if not isinstance(self.causal_orientation, Orientation):
            raise NarrativeContractError("generator_causal_invalid")
        expected = _canonical_generator_id(
            anchor_mention_id=self.anchor_mention_id,
            slot_mention_ids=self.slot_mention_ids,
            generator_kind=self.generator_kind,
            polarity=self.polarity,
            temporal_orientation=self.temporal_orientation,
            causal_orientation=self.causal_orientation,
        )
        if self.generator_id != expected:
            raise NarrativeContractError("generator_id_not_canonical")

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "generator_id": self.generator_id,
            "anchor_mention_id": self.anchor_mention_id,
            "slot_mention_ids": list(self.slot_mention_ids),
            "generator_kind": self.generator_kind.value,
            "polarity": self.polarity.value,
            "temporal_orientation": self.temporal_orientation.value,
            "causal_orientation": self.causal_orientation.value,
        }


@dataclass(frozen=True)
class FiniteTypedHypergraph:
    object_mention_ids: tuple[str, ...]
    generators: tuple[GeneratorProposal, ...]

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "object_mention_ids": list(self.object_mention_ids),
            "generators": [
                generator.semantic_payload() for generator in self.generators
            ],
        }

    @property
    def graph_hash(self) -> str:
        return _content_hash(self.semantic_payload())


def _sentence_byte_spans(text: str) -> tuple[tuple[int, int], ...]:
    character_spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(text):
        if character in ".?!\n":
            segment = text[start : index + 1]
            if _ALPHANUMERIC.search(segment):
                character_spans.append((start, index + 1))
            start = index + 1
    if start < len(text) and _ALPHANUMERIC.search(text[start:]):
        character_spans.append((start, len(text)))
    return tuple(
        (
            len(text[:left].encode("utf-8")),
            len(text[:right].encode("utf-8")),
        )
        for left, right in character_spans
    )


def _semantic_extraction_payload(
    source: NarrativeSource,
    mentions: tuple[Mention, ...],
    generators: tuple[GeneratorProposal, ...],
) -> dict[str, Any]:
    objects = tuple(
        mention.mention_id
        for mention in mentions
        if mention.kind is MentionKind.OBJECT
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "core_version": CORE_VERSION,
        # source_id is deliberately excluded from semantic identity.
        "source_commitment": {
            "source_sha256": source.source_sha256,
            "byte_length": len(source.utf8_bytes),
        },
        "mentions": [mention.semantic_payload() for mention in mentions],
        "hypergraph": FiniteTypedHypergraph(
            object_mention_ids=objects,
            generators=generators,
        ).semantic_payload(),
    }


def _parser_binding_hash(
    *,
    semantic_hash: str,
    completion_sha256: str,
) -> str:
    return _content_hash(
        {
            "parser_version": PARSER_VERSION,
            "semantic_hash": semantic_hash,
            "completion_sha256": completion_sha256,
        }
    )


class _InternalExtractionSeal:
    __slots__ = ("binding_hash",)

    def __init__(self, binding_hash: str) -> None:
        self.binding_hash = binding_hash


@dataclass(frozen=True)
class NarrativeExtraction:
    source: NarrativeSource
    mentions: tuple[Mention, ...]
    generators: tuple[GeneratorProposal, ...]
    completion_sha256: str
    parser_binding_hash: str
    _parser_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self._parser_seal, _InternalExtractionSeal)
            or self._parser_seal.binding_hash != self.parser_binding_hash
        ):
            raise NarrativeContractError("extraction_not_parser_sealed")
        if not isinstance(self.source, NarrativeSource):
            raise NarrativeContractError("extraction_source_invalid")
        if (
            not isinstance(self.mentions, tuple)
            or not 1 <= len(self.mentions) <= MAX_MENTIONS
            or any(not isinstance(item, Mention) for item in self.mentions)
            or len({item.mention_id for item in self.mentions})
            != len(self.mentions)
            or self.mentions
            != tuple(
                sorted(
                    self.mentions,
                    key=lambda item: (
                        item.start_byte,
                        item.end_byte,
                        item.kind.value,
                        item.mention_id,
                    ),
                )
            )
        ):
            raise NarrativeContractError("extraction_mentions_invalid")
        if (
            not isinstance(self.generators, tuple)
            or not 1 <= len(self.generators) <= MAX_GENERATORS
            or any(
                not isinstance(item, GeneratorProposal)
                for item in self.generators
            )
            or len({item.generator_id for item in self.generators})
            != len(self.generators)
        ):
            raise NarrativeContractError("extraction_generators_invalid")
        _require_hash(self.completion_sha256, "completion_hash_invalid")
        _require_hash(self.parser_binding_hash, "parser_binding_hash_invalid")

        source_bytes = self.source.utf8_bytes
        occupied: set[tuple[int, int]] = set()
        mention_by_id: dict[str, Mention] = {}
        for mention in self.mentions:
            if mention.end_byte > len(source_bytes):
                raise NarrativeContractError("mention_span_out_of_bounds")
            span = (mention.start_byte, mention.end_byte)
            if span in occupied:
                raise NarrativeContractError("mention_span_nonunique")
            occupied.add(span)
            quote_bytes = source_bytes[mention.start_byte:mention.end_byte]
            try:
                quote = quote_bytes.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise NarrativeContractError(
                    "mention_span_utf8_invalid"
                ) from exc
            if quote != mention.quote:
                raise NarrativeContractError("mention_quote_span_mismatch")
            if hashlib.sha256(quote_bytes).hexdigest() != mention.quote_sha256:
                raise NarrativeContractError("mention_quote_hash_mismatch")
            positions = _quote_positions(self.source.text, mention.quote)
            if (
                mention.occurrence >= len(positions)
                or len(
                    self.source.text[:positions[mention.occurrence]].encode(
                        "utf-8"
                    )
                )
                != mention.start_byte
            ):
                raise NarrativeContractError(
                    "mention_occurrence_span_mismatch"
                )
            mention_by_id[mention.mention_id] = mention

        object_ids = {
            item.mention_id
            for item in self.mentions
            if item.kind is MentionKind.OBJECT
        }
        anchor_ids = {
            item.mention_id
            for item in self.mentions
            if item.kind is MentionKind.GENERATOR
        }
        used_anchors: set[str] = set()
        used_objects: set[str] = set()
        anchor_positions: list[tuple[int, str]] = []
        for generator in self.generators:
            if (
                generator.anchor_mention_id not in anchor_ids
                or any(slot not in object_ids for slot in generator.slot_mention_ids)
            ):
                raise NarrativeContractError("generator_refs_invalid")
            if generator.anchor_mention_id in used_anchors:
                raise NarrativeContractError("generator_anchor_nonunique")
            used_anchors.add(generator.anchor_mention_id)
            used_objects.update(generator.slot_mention_ids)
            anchor_positions.append(
                (
                    mention_by_id[generator.anchor_mention_id].start_byte,
                    generator.generator_id,
                )
            )
        if used_anchors != anchor_ids:
            raise NarrativeContractError("generator_anchor_coverage_invalid")
        if used_objects != object_ids:
            raise NarrativeContractError("object_relation_coverage_invalid")
        expected_generator_order = tuple(
            generator_id
            for _, generator_id in sorted(anchor_positions)
        )
        if tuple(item.generator_id for item in self.generators) != (
            expected_generator_order
        ):
            raise NarrativeContractError("generator_order_not_canonical")

        sentence_spans = _sentence_byte_spans(self.source.text)
        if not sentence_spans:
            raise NarrativeContractError("sentence_coverage_empty")
        for left, right in sentence_spans:
            in_sentence = tuple(
                mention
                for mention in self.mentions
                if left <= mention.start_byte and mention.end_byte <= right
            )
            if not any(
                item.kind is MentionKind.GENERATOR for item in in_sentence
            ):
                raise NarrativeContractError(
                    "sentence_generator_coverage_incomplete"
                )
            if not any(item.kind is MentionKind.OBJECT for item in in_sentence):
                raise NarrativeContractError(
                    "sentence_object_coverage_incomplete"
                )
        for generator in self.generators:
            anchor = mention_by_id[generator.anchor_mention_id]
            if not any(
                left <= anchor.start_byte
                and anchor.end_byte <= right
                and any(
                    left <= mention_by_id[slot].start_byte
                    and mention_by_id[slot].end_byte <= right
                    for slot in generator.slot_mention_ids
                )
                for left, right in sentence_spans
            ):
                raise NarrativeContractError(
                    "generator_sentence_grounding_incomplete"
                )

        expected_binding = _parser_binding_hash(
            semantic_hash=self.semantic_hash,
            completion_sha256=self.completion_sha256,
        )
        if self.parser_binding_hash != expected_binding:
            raise NarrativeContractError("parser_binding_mismatch")

    @property
    def hypergraph(self) -> FiniteTypedHypergraph:
        return FiniteTypedHypergraph(
            object_mention_ids=tuple(
                item.mention_id
                for item in self.mentions
                if item.kind is MentionKind.OBJECT
            ),
            generators=self.generators,
        )

    def semantic_payload(self) -> dict[str, Any]:
        return _semantic_extraction_payload(
            self.source, self.mentions, self.generators
        )

    @property
    def semantic_hash(self) -> str:
        return _content_hash(self.semantic_payload())

    @property
    def extraction_hash(self) -> str:
        """Compatibility alias for the score-independent semantic hash."""

        return self.semantic_hash

    def provenance_payload(self) -> dict[str, Any]:
        return {
            "parser_version": PARSER_VERSION,
            "semantic_hash": self.semantic_hash,
            "completion_sha256": self.completion_sha256,
            "parser_binding_hash": self.parser_binding_hash,
        }

    @property
    def provenance_hash(self) -> str:
        return _content_hash(self.provenance_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "semantic": self.semantic_payload(),
            "semantic_hash": self.semantic_hash,
            "provenance": self.provenance_payload(),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class GlobalOperator:
    orientation_mode: OrientationMode = OrientationMode.PRESERVING
    invert_polarity: bool = False
    slot_permutation: SlotPermutation = SlotPermutation.IDENTITY

    def __post_init__(self) -> None:
        if not isinstance(self.orientation_mode, OrientationMode):
            raise NarrativeContractError("operator_orientation_mode_invalid")
        if not isinstance(self.invert_polarity, bool):
            raise NarrativeContractError("operator_polarity_invalid")
        if not isinstance(self.slot_permutation, SlotPermutation):
            raise NarrativeContractError("operator_permutation_invalid")

    def safe_payload(self) -> dict[str, Any]:
        return {
            "orientation_mode": self.orientation_mode.value,
            "invert_polarity": self.invert_polarity,
            "slot_permutation": self.slot_permutation.value,
        }

    @property
    def complexity(self) -> int:
        return (
            int(self.orientation_mode is OrientationMode.INVERTING)
            + int(self.invert_polarity)
            + int(self.slot_permutation is not SlotPermutation.IDENTITY)
        )


def _default_operators() -> tuple[GlobalOperator, ...]:
    return tuple(
        GlobalOperator(mode, invert, permutation)
        for mode in (
            OrientationMode.PRESERVING,
            OrientationMode.INVERTING,
        )
        for invert in (False, True)
        for permutation in (
            SlotPermutation.IDENTITY,
            SlotPermutation.REVERSE,
        )
    )


@dataclass(frozen=True)
class MappingSearchConfig:
    object_top_k: int = 2
    generator_top_k: int = 2
    minimum_score_micros: int = -MAX_SCORE_ABS
    max_assignments: int = 4_096
    operators: tuple[GlobalOperator, ...] = field(
        default_factory=_default_operators
    )

    def __post_init__(self) -> None:
        for value, issue in (
            (self.object_top_k, "object_top_k_invalid"),
            (self.generator_top_k, "generator_top_k_invalid"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= MAX_TOP_K
            ):
                raise NarrativeContractError(issue)
        if (
            not isinstance(self.max_assignments, int)
            or isinstance(self.max_assignments, bool)
            or not 1 <= self.max_assignments <= MAX_SEARCH_ASSIGNMENTS
        ):
            raise NarrativeContractError("max_assignments_invalid")
        if (
            not isinstance(self.minimum_score_micros, int)
            or isinstance(self.minimum_score_micros, bool)
            or abs(self.minimum_score_micros) > MAX_SCORE_ABS
        ):
            raise NarrativeContractError("minimum_score_invalid")
        if (
            not isinstance(self.operators, tuple)
            or not 1 <= len(self.operators) <= MAX_OPERATORS
            or any(
                not isinstance(operator, GlobalOperator)
                for operator in self.operators
            )
        ):
            raise NarrativeContractError("operators_invalid")
        payloads = tuple(
            _canonical_bytes(item.safe_payload()) for item in self.operators
        )
        if len(payloads) != len(set(payloads)):
            raise NarrativeContractError("operators_duplicate")

    def safe_payload(self) -> dict[str, Any]:
        return {
            "object_top_k": self.object_top_k,
            "generator_top_k": self.generator_top_k,
            "minimum_score_micros": self.minimum_score_micros,
            "max_assignments": self.max_assignments,
            "operators": [
                operator.safe_payload() for operator in self.operators
            ],
        }

    @property
    def config_hash(self) -> str:
        return _content_hash(self.safe_payload())


@dataclass(frozen=True)
class SemanticScoreTable:
    """Exact semantic envelope; never accepted by the structural checker."""

    object_scores: tuple[tuple[str, str, int], ...]
    generator_scores: tuple[tuple[str, str, int], ...]

    def __post_init__(self) -> None:
        for rows, prefix in (
            (self.object_scores, "object"),
            (self.generator_scores, "generator"),
        ):
            if (
                not isinstance(rows, tuple)
                or len(rows) > MAX_SCORE_ROWS
                or rows != tuple(sorted(rows))
            ):
                raise NarrativeContractError(f"{prefix}_scores_invalid")
            pairs: set[tuple[str, str]] = set()
            for row in rows:
                if (
                    not isinstance(row, tuple)
                    or len(row) != 3
                    or not isinstance(row[0], str)
                    or not isinstance(row[1], str)
                    or not isinstance(row[2], int)
                    or isinstance(row[2], bool)
                    or abs(row[2]) > MAX_SCORE_ABS
                ):
                    raise NarrativeContractError(
                        f"{prefix}_score_row_invalid"
                    )
                pair = (row[0], row[1])
                if pair in pairs:
                    raise NarrativeContractError(
                        f"{prefix}_score_pair_duplicate"
                    )
                pairs.add(pair)

    @classmethod
    def from_mappings(
        cls,
        *,
        object_scores: Mapping[tuple[str, str], int],
        generator_scores: Mapping[tuple[str, str], int],
    ) -> "SemanticScoreTable":
        def rows(
            values: Mapping[tuple[str, str], int],
        ) -> tuple[tuple[str, str, int], ...]:
            if not isinstance(values, Mapping) or len(values) > MAX_SCORE_ROWS:
                raise NarrativeContractError("score_mapping_invalid")
            normalized: list[tuple[str, str, int]] = []
            for key, score in values.items():
                if (
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or not all(isinstance(item, str) for item in key)
                ):
                    raise NarrativeContractError("score_mapping_key_invalid")
                normalized.append((key[0], key[1], score))
            return tuple(
                sorted(normalized)
            )

        return cls(
            object_scores=rows(object_scores),
            generator_scores=rows(generator_scores),
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "object_scores": [list(row) for row in self.object_scores],
            "generator_scores": [list(row) for row in self.generator_scores],
        }

    @property
    def score_table_hash(self) -> str:
        return _content_hash(self.safe_payload())


@dataclass(frozen=True)
class StructuralMapping:
    """Score-free mapping checked for proposal-internal consistency."""

    source_semantic_hash: str
    target_semantic_hash: str
    object_mapping: tuple[tuple[str, str], ...]
    generator_mapping: tuple[tuple[str, str], ...]
    operator: GlobalOperator

    def __post_init__(self) -> None:
        _require_hash(
            self.source_semantic_hash, "source_semantic_hash_invalid"
        )
        _require_hash(
            self.target_semantic_hash, "target_semantic_hash_invalid"
        )
        for values, prefix in (
            (self.object_mapping, "object"),
            (self.generator_mapping, "generator"),
        ):
            if (
                not isinstance(values, tuple)
                or any(
                    not isinstance(row, tuple)
                    or len(row) != 2
                    or not all(isinstance(item, str) for item in row)
                    for row in values
                )
                or len({row[0] for row in values}) != len(values)
                or len({row[1] for row in values}) != len(values)
            ):
                raise NarrativeContractError(f"{prefix}_mapping_invalid")
        if not isinstance(self.operator, GlobalOperator):
            raise NarrativeContractError("mapping_operator_invalid")

    def safe_payload(self) -> dict[str, Any]:
        return {
            "source_semantic_hash": self.source_semantic_hash,
            "target_semantic_hash": self.target_semantic_hash,
            "object_mapping": [list(row) for row in self.object_mapping],
            "generator_mapping": [list(row) for row in self.generator_mapping],
            "operator": self.operator.safe_payload(),
        }

    @property
    def mapping_hash(self) -> str:
        return _content_hash(self.safe_payload())


@dataclass(frozen=True)
class PairMappingProposal:
    """Semantic ranking envelope around one score-free mapping."""

    mapping: StructuralMapping
    semantic_score_micros: int

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, StructuralMapping):
            raise NarrativeContractError("proposal_mapping_invalid")
        if (
            not isinstance(self.semantic_score_micros, int)
            or isinstance(self.semantic_score_micros, bool)
            or abs(self.semantic_score_micros) > MAX_AGGREGATE_SCORE_ABS
        ):
            raise NarrativeContractError("proposal_semantic_score_invalid")

    def safe_payload(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping.safe_payload(),
            "semantic_score_micros": self.semantic_score_micros,
        }

    @property
    def proposal_hash(self) -> str:
        return _content_hash(self.safe_payload())


class _InternalResultSeal:
    """Per-result seal; dataclass replacement cannot rebind proposal content."""

    __slots__ = ("binding_hash",)

    def __init__(self, binding_hash: str) -> None:
        self.binding_hash = binding_hash


def _result_binding_payload(
    *,
    source_semantic_hash: str,
    target_semantic_hash: str,
    score_table_hash: str,
    config_hash: str,
    proposals: tuple[PairMappingProposal, ...],
    assignments_explored: int,
    budget: int,
    budget_exhausted: bool,
    reason_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "source_semantic_hash": source_semantic_hash,
        "target_semantic_hash": target_semantic_hash,
        "score_table_hash": score_table_hash,
        "config_hash": config_hash,
        "proposals": [
            proposal.safe_payload()
            for proposal in sorted(
                proposals, key=lambda item: item.proposal_hash
            )
        ],
        "assignments_explored": assignments_explored,
        "budget": budget,
        "budget_exhausted": budget_exhausted,
        "reason_ids": list(reason_ids),
    }


@dataclass(frozen=True)
class MappingSearchResult:
    source_semantic_hash: str
    target_semantic_hash: str
    score_table_hash: str
    config_hash: str
    proposals: tuple[PairMappingProposal, ...]
    assignments_explored: int
    budget: int
    budget_exhausted: bool
    reason_ids: tuple[str, ...]
    result_binding_hash: str
    _internal_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self._internal_seal, _InternalResultSeal)
            or self._internal_seal.binding_hash != self.result_binding_hash
        ):
            raise NarrativeContractError("search_result_not_internal")
        for value, issue in (
            (self.source_semantic_hash, "search_source_hash_invalid"),
            (self.target_semantic_hash, "search_target_hash_invalid"),
            (self.score_table_hash, "search_score_hash_invalid"),
            (self.config_hash, "search_config_hash_invalid"),
            (self.result_binding_hash, "search_binding_hash_invalid"),
        ):
            _require_hash(value, issue)
        if (
            not isinstance(self.proposals, tuple)
            or len({item.proposal_hash for item in self.proposals})
            != len(self.proposals)
            or any(
                not isinstance(item, PairMappingProposal)
                or item.mapping.source_semantic_hash
                != self.source_semantic_hash
                or item.mapping.target_semantic_hash
                != self.target_semantic_hash
                for item in self.proposals
            )
        ):
            raise NarrativeContractError("search_proposals_invalid")
        if (
            not isinstance(self.assignments_explored, int)
            or isinstance(self.assignments_explored, bool)
            or not 0 <= self.assignments_explored <= self.budget
            or not isinstance(self.budget, int)
            or isinstance(self.budget, bool)
            or not 1 <= self.budget <= MAX_SEARCH_ASSIGNMENTS
            or not isinstance(self.budget_exhausted, bool)
            or (self.budget_exhausted and self.proposals)
        ):
            raise NarrativeContractError("search_budget_fields_invalid")
        if (
            not isinstance(self.reason_ids, tuple)
            or any(not isinstance(item, str) or not item for item in self.reason_ids)
        ):
            raise NarrativeContractError("search_reasons_invalid")
        expected = _content_hash(
            _result_binding_payload(
                source_semantic_hash=self.source_semantic_hash,
                target_semantic_hash=self.target_semantic_hash,
                score_table_hash=self.score_table_hash,
                config_hash=self.config_hash,
                proposals=self.proposals,
                assignments_explored=self.assignments_explored,
                budget=self.budget,
                budget_exhausted=self.budget_exhausted,
                reason_ids=self.reason_ids,
            )
        )
        if expected != self.result_binding_hash:
            raise NarrativeContractError("search_result_binding_mismatch")

    @property
    def pair_input_hash(self) -> str:
        return _content_hash(
            {
                "source_semantic_hash": self.source_semantic_hash,
                "target_semantic_hash": self.target_semantic_hash,
                "score_table_hash": self.score_table_hash,
                "config_hash": self.config_hash,
            }
        )

    @property
    def proposal_set_hash(self) -> str:
        return _content_hash(
            [
                proposal.safe_payload()
                for proposal in sorted(
                    self.proposals, key=lambda item: item.proposal_hash
                )
            ]
        )

    def validate_internal(self) -> None:
        self.__post_init__()


def _make_search_result(
    *,
    source: NarrativeExtraction,
    target: NarrativeExtraction,
    scores: SemanticScoreTable,
    config: MappingSearchConfig,
    proposals: tuple[PairMappingProposal, ...],
    assignments_explored: int,
    budget_exhausted: bool,
    reason_ids: tuple[str, ...],
) -> MappingSearchResult:
    payload = _result_binding_payload(
        source_semantic_hash=source.semantic_hash,
        target_semantic_hash=target.semantic_hash,
        score_table_hash=scores.score_table_hash,
        config_hash=config.config_hash,
        proposals=proposals,
        assignments_explored=assignments_explored,
        budget=config.max_assignments,
        budget_exhausted=budget_exhausted,
        reason_ids=reason_ids,
    )
    binding_hash = _content_hash(payload)
    return MappingSearchResult(
        source_semantic_hash=source.semantic_hash,
        target_semantic_hash=target.semantic_hash,
        score_table_hash=scores.score_table_hash,
        config_hash=config.config_hash,
        proposals=proposals,
        assignments_explored=assignments_explored,
        budget=config.max_assignments,
        budget_exhausted=budget_exhausted,
        reason_ids=reason_ids,
        result_binding_hash=binding_hash,
        _internal_seal=_InternalResultSeal(binding_hash),
    )


@dataclass(frozen=True)
class CorrespondenceCertificate:
    """Score-free certificate of proposal-internal consistency only."""

    mapping_hash: str
    disposition: CertificateDisposition
    incidence_contradictions: int
    polarity_contradictions: int
    temporal_contradictions: int
    causal_contradictions: int
    preserved_connected_paths: int
    covered_objects: int
    covered_generators: int
    grounded_sentences: int
    complexity: int
    lexicographic_score: tuple[int, ...]
    reason_ids: tuple[str, ...]

    def safe_payload(self) -> dict[str, Any]:
        return {
            "claim_scope": "grounded_proposal_internal_consistency_only",
            "mapping_hash": self.mapping_hash,
            "disposition": self.disposition.value,
            "contradictions": {
                "incidence": self.incidence_contradictions,
                "polarity": self.polarity_contradictions,
                "temporal": self.temporal_contradictions,
                "causal": self.causal_contradictions,
            },
            "preserved_connected_paths": self.preserved_connected_paths,
            "covered_objects": self.covered_objects,
            "covered_generators": self.covered_generators,
            "grounded_sentences": self.grounded_sentences,
            "complexity": self.complexity,
            "lexicographic_score": list(self.lexicographic_score),
            "reason_ids": list(self.reason_ids),
        }

    @property
    def certificate_hash(self) -> str:
        return _content_hash(self.safe_payload())


@dataclass(frozen=True)
class ArmChoice:
    arm: ArmName
    disposition: ChoiceDisposition
    pair_input_hash: str
    proposal_set_hash: str
    search_result_binding_hash: str
    selected_proposal_hash: str | None
    checker_called: bool
    reason_ids: tuple[str, ...]
    certificate: CorrespondenceCertificate | None = None

    def safe_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "disposition": self.disposition.value,
            "pair_input_hash": self.pair_input_hash,
            "proposal_set_hash": self.proposal_set_hash,
            "search_result_binding_hash": self.search_result_binding_hash,
            "selected_proposal_hash": self.selected_proposal_hash,
            "checker_called": self.checker_called,
            "reason_ids": list(self.reason_ids),
            "certificate": (
                None
                if self.certificate is None
                else self.certificate.safe_payload()
            ),
        }

    @property
    def choice_hash(self) -> str:
        return _content_hash(self.safe_payload())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NarrativeContractError("json_duplicate_key")
        result[key] = value
    return result


def _bounded_parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 10:
        raise NarrativeContractError("json_integer_out_of_bounds")
    parsed = int(value)
    if abs(parsed) > MAX_JSON_INTEGER_ABS:
        raise NarrativeContractError("json_integer_out_of_bounds")
    return parsed


def _reject_float(_: str) -> float:
    raise NarrativeContractError("json_float_forbidden")


def _reject_constant(_: str) -> None:
    raise NarrativeContractError("json_constant_forbidden")


def _check_json_shape(value: Any) -> None:
    nodes = 0

    def walk(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise NarrativeContractError("json_node_budget_exceeded")
        if depth > MAX_JSON_DEPTH:
            raise NarrativeContractError("json_depth_exceeded")
        if item is None or type(item) in {bool, int, str}:
            return
        if isinstance(item, list):
            for child in item:
                walk(child, depth + 1)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = re.sub(
                    r"[^a-z0-9]+", "_", key.lower()
                ).strip("_")
                if _FORBIDDEN_KEY_TOKEN.search(normalized):
                    raise NarrativeContractError(
                        "forbidden_decision_or_relation_family_key"
                    )
                walk(child, depth + 1)
            return
        raise NarrativeContractError("json_type_forbidden")

    walk(value, 0)


def _require_exact_keys(
    value: object, expected: set[str], issue_id: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise NarrativeContractError(issue_id)
    return value


def _enum_value(enum_type: type[Enum], value: object, issue_id: str) -> Any:
    if not isinstance(value, str):
        raise NarrativeContractError(issue_id)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise NarrativeContractError(issue_id) from exc


def _ground_raw_mention(
    source: NarrativeSource, raw: object
) -> tuple[str, Mention]:
    record = _require_exact_keys(
        raw,
        {"mention_id", "kind", "quote", "occurrence"},
        "mention_fields_invalid",
    )
    raw_id = _require_identifier(record["mention_id"], "mention_id_invalid")
    kind = _enum_value(MentionKind, record["kind"], "mention_kind_invalid")
    quote = record["quote"]
    if not isinstance(quote, str) or not quote or "\x00" in quote:
        raise NarrativeContractError("mention_quote_invalid")
    try:
        quote_bytes = quote.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise NarrativeContractError("mention_quote_utf8_invalid") from exc
    if len(quote_bytes) > MAX_QUOTE_BYTES:
        raise NarrativeContractError("mention_quote_invalid")
    if _PROMPT_INJECTION.search(quote):
        raise NarrativeContractError("mention_quote_tainted")
    occurrence = _strict_int(
        record["occurrence"], "mention_occurrence_invalid"
    )
    positions = _quote_positions(source.text, quote)
    if not positions:
        raise NarrativeContractError("mention_quote_hallucinated")
    if occurrence >= len(positions):
        raise NarrativeContractError("mention_occurrence_out_of_range")
    character_start = positions[occurrence]
    character_end = character_start + len(quote)
    start_byte = len(source.text[:character_start].encode("utf-8"))
    end_byte = len(source.text[:character_end].encode("utf-8"))
    grounded = source.utf8_bytes[start_byte:end_byte]
    if grounded != quote_bytes:
        raise NarrativeContractError("mention_span_roundtrip_failed")
    quote_hash = hashlib.sha256(grounded).hexdigest()
    return (
        raw_id,
        Mention(
            mention_id=_canonical_mention_id(
                kind, start_byte, end_byte, quote_hash
            ),
            kind=kind,
            quote=quote,
            occurrence=occurrence,
            start_byte=start_byte,
            end_byte=end_byte,
            quote_sha256=quote_hash,
        ),
    )


def parse_untrusted_generator_completion(
    source: NarrativeSource,
    completion: str | bytes,
) -> NarrativeExtraction:
    if isinstance(completion, str):
        try:
            completion_bytes = completion.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise NarrativeContractError("completion_utf8_invalid") from exc
    elif isinstance(completion, bytes):
        completion_bytes = completion
        try:
            completion = completion.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise NarrativeContractError("completion_utf8_invalid") from exc
    else:
        raise NarrativeContractError("completion_type_invalid")
    if not completion_bytes or len(completion_bytes) > MAX_COMPLETION_BYTES:
        raise NarrativeContractError("completion_size_invalid")
    try:
        payload = json.loads(
            completion,
            object_pairs_hook=_unique_object,
            parse_int=_bounded_parse_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except NarrativeContractError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise NarrativeContractError("json_invalid") from exc
    _check_json_shape(payload)
    root = _require_exact_keys(
        payload,
        {"schema_version", "mentions", "generators"},
        "root_fields_invalid",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise NarrativeContractError("schema_version_invalid")
    raw_mentions = root["mentions"]
    raw_generators = root["generators"]
    if (
        not isinstance(raw_mentions, list)
        or not 1 <= len(raw_mentions) <= MAX_MENTIONS
    ):
        raise NarrativeContractError("mention_count_invalid")
    if (
        not isinstance(raw_generators, list)
        or not 1 <= len(raw_generators) <= MAX_GENERATORS
    ):
        raise NarrativeContractError("generator_count_invalid")

    raw_to_canonical: dict[str, str] = {}
    grounded_mentions: list[Mention] = []
    occupied: set[tuple[int, int]] = set()
    for raw in raw_mentions:
        raw_id, mention = _ground_raw_mention(source, raw)
        if raw_id in raw_to_canonical:
            raise NarrativeContractError("mention_id_duplicate")
        span = (mention.start_byte, mention.end_byte)
        if span in occupied:
            raise NarrativeContractError("mention_span_nonunique")
        occupied.add(span)
        raw_to_canonical[raw_id] = mention.mention_id
        grounded_mentions.append(mention)
    mentions = tuple(
        sorted(
            grounded_mentions,
            key=lambda item: (
                item.start_byte,
                item.end_byte,
                item.kind.value,
                item.mention_id,
            ),
        )
    )
    mention_by_canonical = {item.mention_id: item for item in mentions}

    generators: list[GeneratorProposal] = []
    raw_generator_ids: set[str] = set()
    used_anchors: set[str] = set()
    for raw in raw_generators:
        record = _require_exact_keys(
            raw,
            {
                "generator_id",
                "anchor_mention_id",
                "slot_mention_ids",
                "generator_kind",
                "polarity",
                "temporal_orientation",
                "causal_orientation",
            },
            "generator_fields_invalid",
        )
        raw_generator_id = _require_identifier(
            record["generator_id"], "generator_id_invalid"
        )
        if raw_generator_id in raw_generator_ids:
            raise NarrativeContractError("generator_id_duplicate")
        raw_generator_ids.add(raw_generator_id)
        raw_anchor = _require_identifier(
            record["anchor_mention_id"], "generator_anchor_invalid"
        )
        if raw_anchor not in raw_to_canonical:
            raise NarrativeContractError("generator_anchor_ref_invalid")
        anchor = raw_to_canonical[raw_anchor]
        if (
            mention_by_canonical[anchor].kind is not MentionKind.GENERATOR
            or anchor in used_anchors
        ):
            raise NarrativeContractError("generator_anchor_ref_invalid")
        used_anchors.add(anchor)
        raw_slots = record["slot_mention_ids"]
        if (
            not isinstance(raw_slots, list)
            or not 2 <= len(raw_slots) <= MAX_SLOTS
            or any(not isinstance(item, str) for item in raw_slots)
            or len(set(raw_slots)) != len(raw_slots)
            or any(item not in raw_to_canonical for item in raw_slots)
        ):
            raise NarrativeContractError("generator_slots_invalid")
        slots = tuple(raw_to_canonical[item] for item in raw_slots)
        if any(
            mention_by_canonical[item].kind is not MentionKind.OBJECT
            for item in slots
        ):
            raise NarrativeContractError("generator_slot_ref_invalid")
        kind = _enum_value(
            GeneratorKind,
            record["generator_kind"],
            "generator_kind_invalid",
        )
        polarity = _enum_value(
            SignedState,
            record["polarity"],
            "generator_polarity_invalid",
        )
        temporal = _enum_value(
            Orientation,
            record["temporal_orientation"],
            "generator_temporal_invalid",
        )
        causal = _enum_value(
            Orientation,
            record["causal_orientation"],
            "generator_causal_invalid",
        )
        generators.append(
            GeneratorProposal(
                generator_id=_canonical_generator_id(
                    anchor_mention_id=anchor,
                    slot_mention_ids=slots,
                    generator_kind=kind,
                    polarity=polarity,
                    temporal_orientation=temporal,
                    causal_orientation=causal,
                ),
                anchor_mention_id=anchor,
                slot_mention_ids=slots,
                generator_kind=kind,
                polarity=polarity,
                temporal_orientation=temporal,
                causal_orientation=causal,
            )
        )
    generators_tuple = tuple(
        sorted(
            generators,
            key=lambda item: mention_by_canonical[
                item.anchor_mention_id
            ].start_byte,
        )
    )
    completion_hash = hashlib.sha256(completion_bytes).hexdigest()
    semantic_hash = _content_hash(
        _semantic_extraction_payload(source, mentions, generators_tuple)
    )
    binding_hash = _parser_binding_hash(
        semantic_hash=semantic_hash,
        completion_sha256=completion_hash,
    )
    return NarrativeExtraction(
        source=source,
        mentions=mentions,
        generators=generators_tuple,
        completion_sha256=completion_hash,
        parser_binding_hash=binding_hash,
        _parser_seal=_InternalExtractionSeal(binding_hash),
    )


def _score_lookup(
    rows: tuple[tuple[str, str, int], ...],
    *,
    source_ids: set[str],
    target_ids: set[str],
    prefix: str,
) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for source_id, target_id, score in rows:
        if source_id not in source_ids or target_id not in target_ids:
            raise NarrativeContractError(f"{prefix}_score_ref_invalid")
        result[(source_id, target_id)] = score
    return result


def _domains(
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    scores: Mapping[tuple[str, str], int],
    *,
    top_k: int,
    minimum: int,
) -> dict[str, tuple[tuple[str, int], ...]]:
    result: dict[str, tuple[tuple[str, int], ...]] = {}
    for source_id in source_ids:
        ranked = sorted(
            (
                (target_id, scores[(source_id, target_id)])
                for target_id in target_ids
                if (source_id, target_id) in scores
                and scores[(source_id, target_id)] >= minimum
            ),
            key=lambda row: (-row[1], row[0]),
        )
        result[source_id] = tuple(ranked[:top_k])
    return result


def generate_pair_mapping_proposals(
    source: NarrativeExtraction,
    target: NarrativeExtraction,
    scores: SemanticScoreTable,
    *,
    config: MappingSearchConfig | None = None,
) -> MappingSearchResult:
    config = config or MappingSearchConfig()
    if (
        not isinstance(source, NarrativeExtraction)
        or not isinstance(target, NarrativeExtraction)
        or not isinstance(scores, SemanticScoreTable)
        or not isinstance(config, MappingSearchConfig)
    ):
        raise NarrativeContractError("mapping_inputs_invalid")
    # Re-run deep validation at the search boundary.
    source.__post_init__()
    target.__post_init__()
    source_objects = source.hypergraph.object_mention_ids
    target_objects = target.hypergraph.object_mention_ids
    source_generators = tuple(
        item.generator_id for item in source.generators
    )
    target_generators = tuple(
        item.generator_id for item in target.generators
    )
    reasons: list[str] = []
    if len(source_objects) > len(target_objects):
        reasons.append("object_injection_impossible")
    if len(source_generators) > len(target_generators):
        reasons.append("generator_injection_impossible")
    if reasons:
        return _make_search_result(
            source=source,
            target=target,
            scores=scores,
            config=config,
            proposals=(),
            assignments_explored=0,
            budget_exhausted=False,
            reason_ids=tuple(sorted(reasons)),
        )
    object_scores = _score_lookup(
        scores.object_scores,
        source_ids=set(source_objects),
        target_ids=set(target_objects),
        prefix="object",
    )
    generator_scores = _score_lookup(
        scores.generator_scores,
        source_ids=set(source_generators),
        target_ids=set(target_generators),
        prefix="generator",
    )
    object_domains = _domains(
        source_objects,
        target_objects,
        object_scores,
        top_k=config.object_top_k,
        minimum=config.minimum_score_micros,
    )
    generator_domains = _domains(
        source_generators,
        target_generators,
        generator_scores,
        top_k=config.generator_top_k,
        minimum=config.minimum_score_micros,
    )
    if any(not value for value in object_domains.values()):
        reasons.append("object_domain_empty")
    if any(not value for value in generator_domains.values()):
        reasons.append("generator_domain_empty")
    if reasons:
        return _make_search_result(
            source=source,
            target=target,
            scores=scores,
            config=config,
            proposals=(),
            assignments_explored=0,
            budget_exhausted=False,
            reason_ids=tuple(sorted(reasons)),
        )

    variables = tuple(
        ("object", source_id, object_domains[source_id])
        for source_id in source_objects
    ) + tuple(
        ("generator", source_id, generator_domains[source_id])
        for source_id in source_generators
    )
    object_rows: list[tuple[str, str]] = []
    generator_rows: list[tuple[str, str]] = []
    used_objects: set[str] = set()
    used_generators: set[str] = set()
    proposals: list[PairMappingProposal] = []
    explored = 0
    exhausted = False

    def spend_budget() -> bool:
        nonlocal explored, exhausted
        if explored >= config.max_assignments:
            exhausted = True
            return False
        explored += 1
        return True

    def visit(index: int, score_total: int) -> None:
        nonlocal exhausted
        if exhausted:
            return
        if index < len(variables):
            kind, source_id, domain = variables[index]
            used = used_objects if kind == "object" else used_generators
            rows = object_rows if kind == "object" else generator_rows
            for target_id, score in domain:
                if target_id in used:
                    continue
                if not spend_budget():
                    return
                used.add(target_id)
                rows.append((source_id, target_id))
                visit(index + 1, score_total + score)
                rows.pop()
                used.remove(target_id)
                if exhausted:
                    return
            return
        for operator in config.operators:
            if not spend_budget():
                return
            proposals.append(
                PairMappingProposal(
                    mapping=StructuralMapping(
                        source_semantic_hash=source.semantic_hash,
                        target_semantic_hash=target.semantic_hash,
                        object_mapping=tuple(object_rows),
                        generator_mapping=tuple(generator_rows),
                        operator=operator,
                    ),
                    semantic_score_micros=score_total,
                )
            )

    visit(0, 0)
    if exhausted:
        return _make_search_result(
            source=source,
            target=target,
            scores=scores,
            config=config,
            proposals=(),
            assignments_explored=explored,
            budget_exhausted=True,
            reason_ids=("mapping_budget_exhausted",),
        )
    if not proposals:
        return _make_search_result(
            source=source,
            target=target,
            scores=scores,
            config=config,
            proposals=(),
            assignments_explored=explored,
            budget_exhausted=False,
            reason_ids=("injective_assignment_empty",),
        )
    return _make_search_result(
        source=source,
        target=target,
        scores=scores,
        config=config,
        proposals=tuple(
            sorted(proposals, key=lambda item: item.proposal_hash)
        ),
        assignments_explored=explored,
        budget_exhausted=False,
        reason_ids=(),
    )


def _permuted(
    values: tuple[str, ...], permutation: SlotPermutation
) -> tuple[str, ...]:
    if permutation is SlotPermutation.IDENTITY:
        return values
    if permutation is SlotPermutation.REVERSE:
        return tuple(reversed(values))
    if permutation is SlotPermutation.ROTATE_LEFT:
        return values[1:] + values[:1]
    raise AssertionError("unreachable permutation")


def verify_correspondence(
    mapping: StructuralMapping,
    source: NarrativeExtraction,
    target: NarrativeExtraction,
) -> CorrespondenceCertificate:
    """Check only grounded proposal-internal structural consistency."""

    if not isinstance(mapping, StructuralMapping):
        raise NarrativeContractError("checker_mapping_invalid")
    source.__post_init__()
    target.__post_init__()
    incidence = polarity = temporal = causal = 0
    reasons: set[str] = set()
    source_objects = set(source.hypergraph.object_mention_ids)
    target_objects = set(target.hypergraph.object_mention_ids)
    source_generators = {
        item.generator_id: item for item in source.generators
    }
    target_generators = {
        item.generator_id: item for item in target.generators
    }
    if (
        mapping.source_semantic_hash != source.semantic_hash
        or mapping.target_semantic_hash != target.semantic_hash
    ):
        incidence += 1
        reasons.add("proposal_internal_semantic_commitment_mismatch")
    object_mapping = dict(mapping.object_mapping)
    generator_mapping = dict(mapping.generator_mapping)
    object_valid = (
        set(object_mapping) == source_objects
        and len(object_mapping) == len(mapping.object_mapping)
        and len(set(object_mapping.values())) == len(object_mapping)
        and set(object_mapping.values()).issubset(target_objects)
    )
    generator_valid = (
        set(generator_mapping) == set(source_generators)
        and len(generator_mapping) == len(mapping.generator_mapping)
        and len(set(generator_mapping.values())) == len(generator_mapping)
        and set(generator_mapping.values()).issubset(target_generators)
    )
    if not object_valid:
        incidence += 1
        reasons.add("proposal_internal_object_mapping_invalid")
    if not generator_valid:
        incidence += 1
        reasons.add("proposal_internal_generator_mapping_invalid")

    clean: set[str] = set()
    if object_valid and generator_valid:
        for source_id in sorted(source_generators):
            source_generator = source_generators[source_id]
            target_generator = target_generators[generator_mapping[source_id]]
            expected_slots = _permuted(
                tuple(
                    object_mapping[slot]
                    for slot in source_generator.slot_mention_ids
                ),
                mapping.operator.slot_permutation,
            )
            local_incidence = int(
                source_generator.generator_kind
                is not target_generator.generator_kind
                or expected_slots != target_generator.slot_mention_ids
            )
            expected_polarity = source_generator.polarity.sign * (
                -1 if mapping.operator.invert_polarity else 1
            )
            local_polarity = int(
                expected_polarity != target_generator.polarity.sign
            )
            orientation_sign = (
                -1
                if mapping.operator.orientation_mode
                is OrientationMode.INVERTING
                else 1
            )
            local_temporal = int(
                source_generator.temporal_orientation.sign
                * orientation_sign
                != target_generator.temporal_orientation.sign
            )
            local_causal = int(
                source_generator.causal_orientation.sign
                * orientation_sign
                != target_generator.causal_orientation.sign
            )
            incidence += local_incidence
            polarity += local_polarity
            temporal += local_temporal
            causal += local_causal
            if local_incidence:
                reasons.add("proposal_internal_incidence_contradiction")
            if local_polarity:
                reasons.add("proposal_internal_polarity_contradiction")
            if local_temporal:
                reasons.add("proposal_internal_temporal_contradiction")
            if local_causal:
                reasons.add("proposal_internal_causal_contradiction")
            if not (
                local_incidence
                or local_polarity
                or local_temporal
                or local_causal
            ):
                clean.add(source_id)

    preserved_paths = len(clean)
    clean_ordered = sorted(clean)
    for index, left_id in enumerate(clean_ordered):
        left = source_generators[left_id]
        for right_id in clean_ordered[index + 1 :]:
            right = source_generators[right_id]
            shared = set(left.slot_mention_ids).intersection(
                right.slot_mention_ids
            )
            if not shared:
                continue
            target_left = target_generators[generator_mapping[left_id]]
            target_right = target_generators[generator_mapping[right_id]]
            mapped_shared = {object_mapping[item] for item in shared}
            if mapped_shared.issubset(
                set(target_left.slot_mention_ids).intersection(
                    target_right.slot_mention_ids
                )
            ):
                preserved_paths += 1
    covered_objects = len(set(object_mapping).intersection(source_objects))
    covered_generators = len(
        set(generator_mapping).intersection(source_generators)
    )
    grounded_sentences = len(_sentence_byte_spans(source.source.text))
    total = incidence + polarity + temporal + causal
    score = (
        total,
        incidence,
        polarity,
        temporal,
        causal,
        -preserved_paths,
        -covered_generators,
        -covered_objects,
        -grounded_sentences,
        mapping.operator.complexity,
    )
    disposition = (
        CertificateDisposition.PROPOSAL_INTERNALLY_CONSISTENT
        if total == 0
        else CertificateDisposition.PROPOSAL_STRUCTURALLY_CONTRADICTED
    )
    return CorrespondenceCertificate(
        mapping_hash=mapping.mapping_hash,
        disposition=disposition,
        incidence_contradictions=incidence,
        polarity_contradictions=polarity,
        temporal_contradictions=temporal,
        causal_contradictions=causal,
        preserved_connected_paths=preserved_paths,
        covered_objects=covered_objects,
        covered_generators=covered_generators,
        grounded_sentences=grounded_sentences,
        complexity=mapping.operator.complexity,
        lexicographic_score=score,
        reason_ids=tuple(sorted(reasons)),
    )


def _abstain_choice(
    arm: ArmName,
    result: MappingSearchResult,
    reason_id: str,
    *,
    checker_called: bool,
) -> ArmChoice:
    return ArmChoice(
        arm=arm,
        disposition=ChoiceDisposition.ABSTAIN,
        pair_input_hash=result.pair_input_hash,
        proposal_set_hash=result.proposal_set_hash,
        search_result_binding_hash=result.result_binding_hash,
        selected_proposal_hash=None,
        checker_called=checker_called,
        reason_ids=(reason_id,),
        certificate=None,
    )


def choose_flat_arm(result: MappingSearchResult) -> ArmChoice:
    if not isinstance(result, MappingSearchResult):
        raise NarrativeContractError("flat_search_result_invalid")
    result.validate_internal()
    if result.budget_exhausted:
        return _abstain_choice(
            ArmName.FLAT,
            result,
            "mapping_budget_exhausted",
            checker_called=False,
        )
    if not result.proposals:
        return _abstain_choice(
            ArmName.FLAT,
            result,
            "mapping_proposals_empty",
            checker_called=False,
        )
    best_key = min(
        (
            -item.semantic_score_micros,
            item.mapping.operator.complexity,
        )
        for item in result.proposals
    )
    best = tuple(
        item
        for item in result.proposals
        if (
            -item.semantic_score_micros,
            item.mapping.operator.complexity,
        )
        == best_key
    )
    if len(best) != 1:
        return _abstain_choice(
            ArmName.FLAT,
            result,
            "flat_exact_tie",
            checker_called=False,
        )
    return ArmChoice(
        arm=ArmName.FLAT,
        disposition=ChoiceDisposition.SELECTED,
        pair_input_hash=result.pair_input_hash,
        proposal_set_hash=result.proposal_set_hash,
        search_result_binding_hash=result.result_binding_hash,
        selected_proposal_hash=best[0].proposal_hash,
        checker_called=False,
        reason_ids=(),
        certificate=None,
    )


def choose_full_arm(
    source: NarrativeExtraction,
    target: NarrativeExtraction,
    result: MappingSearchResult,
) -> ArmChoice:
    if not isinstance(result, MappingSearchResult):
        raise NarrativeContractError("full_search_result_invalid")
    result.validate_internal()
    source.__post_init__()
    target.__post_init__()
    if (
        result.source_semantic_hash != source.semantic_hash
        or result.target_semantic_hash != target.semantic_hash
    ):
        return _abstain_choice(
            ArmName.FULL,
            result,
            "search_semantic_commitment_mismatch",
            checker_called=False,
        )
    if result.budget_exhausted:
        return _abstain_choice(
            ArmName.FULL,
            result,
            "mapping_budget_exhausted",
            checker_called=False,
        )
    if not result.proposals:
        return _abstain_choice(
            ArmName.FULL,
            result,
            "mapping_proposals_empty",
            checker_called=False,
        )
    certificates = tuple(
        verify_correspondence(item.mapping, source, target)
        for item in result.proposals
    )
    best_score = min(item.lexicographic_score for item in certificates)
    best_indices = tuple(
        index
        for index, item in enumerate(certificates)
        if item.lexicographic_score == best_score
    )
    # Complexity is part of the exact objective.  Only a genuine full-tuple
    # tie abstains.
    if len(best_indices) != 1:
        return _abstain_choice(
            ArmName.FULL,
            result,
            "checker_exact_lexicographic_tie",
            checker_called=True,
        )
    index = best_indices[0]
    certificate = certificates[index]
    proposal = result.proposals[index]
    return ArmChoice(
        arm=ArmName.FULL,
        disposition=(
            ChoiceDisposition.SELECTED
            if certificate.disposition
            is CertificateDisposition.PROPOSAL_INTERNALLY_CONSISTENT
            else ChoiceDisposition.PROPOSAL_STRUCTURALLY_CONTRADICTED
        ),
        pair_input_hash=result.pair_input_hash,
        proposal_set_hash=result.proposal_set_hash,
        search_result_binding_hash=result.result_binding_hash,
        selected_proposal_hash=proposal.proposal_hash,
        checker_called=True,
        reason_ids=certificate.reason_ids,
        certificate=certificate,
    )


__all__ = [
    "ArmChoice",
    "ArmName",
    "CertificateDisposition",
    "ChoiceDisposition",
    "CorrespondenceCertificate",
    "CORE_VERSION",
    "FiniteTypedHypergraph",
    "GeneratorKind",
    "GeneratorProposal",
    "GlobalOperator",
    "MappingSearchConfig",
    "MappingSearchResult",
    "Mention",
    "MentionKind",
    "NarrativeContractError",
    "NarrativeExtraction",
    "NarrativeSource",
    "Orientation",
    "OrientationMode",
    "PairMappingProposal",
    "SCHEMA_VERSION",
    "SemanticScoreTable",
    "SignedState",
    "SlotPermutation",
    "StructuralMapping",
    "choose_flat_arm",
    "choose_full_arm",
    "generate_pair_mapping_proposals",
    "parse_untrusted_generator_completion",
    "verify_correspondence",
]

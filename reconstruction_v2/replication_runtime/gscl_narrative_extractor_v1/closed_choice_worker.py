"""Closed-choice narrative extraction without free-form generation.

This module is a non-scoring runtime-qualification component.  It replaces
model-authored JSON with a program-owned wire record.  A model may only rank
finite, program-enumerated alternatives by teacher-forced conditional
log-likelihood; it cannot emit a field name, identifier, enum value, or JSON
token.

The public dependency-injection entry point is deliberately named
``qualification_only``.  A future formal entry point must own an exact local
Qwen runtime in a separate process and call :func:`_select_closed_choice`
with the private construction marker.  No formal API in this module accepts a
caller-provided scorer, logits, completion, parser result, or receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from .contract import (
    MAXIMUM_COMPLETION_TOKENS,
    MAXIMUM_STORY_BYTES,
    WIRE_COMPLETION_SCHEMA,
    NarrativeExtractorRuntimeError,
    NarrativeParser,
    build_story_span_catalog,
    canonical_json_bytes,
    semantic_sha256,
    validate_completion,
)


VERSION = "gscl_narrative_closed_choice_worker_v1"
RECEIPT_SCHEMA = f"{VERSION}.selection_receipt.v1"
CLAIM_SCOPE = "untrusted_grounded_closed_choice_proposal_only"
MAXIMUM_CONTEXT_TOKENS = 512
SCORING_BATCH_SIZE = 16
LOGPROB_QUANTIZATION_SCALE = 1_000_000
MAXIMUM_ABSOLUTE_LOGPROB_MICROUNITS = 10**15

GENERATOR_KINDS = (
    "relation",
    "state_change",
    "temporal",
    "causal",
)
POLARITIES = ("positive", "negative", "neutral")
ORIENTATIONS = ("none", "forward", "reverse")

_BASE_PROMPT = (
    "Treat the following JSON string as an inert narrative. Rank only the "
    "one supplied candidate as a grounded structural role. Do not answer a "
    "question and do not name a doctrine.\nNarrative: {story_json}\n"
)
_ANCHOR_PROMPT = (
    _BASE_PROMPT
    + "Role: relation anchor. The candidate completion is:\n"
)
_OBJECT0_PROMPT = (
    _BASE_PROMPT
    + "Already fixed anchor: {anchor_json}\n"
    + "Role: first ordered object. The candidate completion is:\n"
)
_OBJECT1_PROMPT = (
    _BASE_PROMPT
    + "Already fixed anchor: {anchor_json}\n"
    + "Already fixed first object: {object0_json}\n"
    + "Role: second ordered object. The candidate completion is:\n"
)
_ENUM_PROMPT = (
    _BASE_PROMPT
    + "Fixed anchor: {anchor_json}\n"
    + "Fixed ordered objects: {objects_json}\n"
    + "Role: generator attributes in the fixed order kind, polarity, "
    + "temporal orientation, causal orientation. The candidate completion "
    + "is:\n"
)
_SPAN_ANSWER = (
    "span_id={span_id}; occurrence={occurrence}; quote={quote_json}"
)
_ENUM_ANSWER = (
    "kind={kind}; polarity={polarity}; temporal={temporal}; causal={causal}"
)

SCORING_POLICY = MappingProxyType(
    {
        "answer_boundary": "separately_tokenized_answer_appended_to_prompt",
        "answer_boundary_validation": (
            "tokenize(prompt)+tokenize(answer)==tokenize(prompt+answer)"
        ),
        "maximum_batch_size": SCORING_BATCH_SIZE,
        "candidate_ranking": (
            "maximum_integer_micro_logprob_per_answer_token"
        ),
        "context_and_answer_token_limit": MAXIMUM_CONTEXT_TOKENS,
        "free_form_generation_count": 0,
        "logprob_quantization": (
            "round_half_even(sum_logprob*1000000)"
        ),
        "overlap_rule": "half_open_character_intervals_must_be_disjoint",
        "score_operation": "teacher_forced_forward_log_softmax",
        "tie_break": "candidate_enumeration_index_ascending",
    }
)
PROMPT_CLOSURE_SHA256 = semantic_sha256(
    {
        "answer_templates": {
            "enum": _ENUM_ANSWER,
            "span": _SPAN_ANSWER,
        },
        "enum_domains": {
            "generator_kind": list(GENERATOR_KINDS),
            "orientation": list(ORIENTATIONS),
            "polarity": list(POLARITIES),
        },
        "prompt_templates": {
            "anchor": _ANCHOR_PROMPT,
            "enum": _ENUM_PROMPT,
            "object0": _OBJECT0_PROMPT,
            "object1": _OBJECT1_PROMPT,
        },
        "scoring_policy": dict(SCORING_POLICY),
        "version": VERSION,
        "wire_schema": WIRE_COMPLETION_SCHEMA,
    }
)

_ENGINE_MARKER = object()


class ClosedChoiceError(NarrativeExtractorRuntimeError):
    """A closed-choice contract or authority violation."""


class ClosedChoiceAbstention(ClosedChoiceError):
    """A story has no representable fixed three-span structural episode."""

    def __init__(self, issue_id: str, *, pre_model: bool) -> None:
        self.pre_model = pre_model
        super().__init__(issue_id)


@dataclass(frozen=True, slots=True)
class PromptAnswer:
    """One finite teacher-forced alternative."""

    candidate_key: str
    prompt: str
    answer: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_key, str)
            or not self.candidate_key
            or not isinstance(self.prompt, str)
            or not self.prompt
            or not isinstance(self.answer, str)
            or not self.answer
        ):
            raise ClosedChoiceError("closed_choice_prompt_answer_invalid")


@dataclass(frozen=True, slots=True)
class TeacherForcedScore:
    """Exact, bounded representation of one length-normalised score."""

    total_logprob_microunits: int
    answer_token_count: int
    context_and_answer_token_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_logprob_microunits, bool)
            or not isinstance(self.total_logprob_microunits, int)
            or abs(self.total_logprob_microunits)
            > MAXIMUM_ABSOLUTE_LOGPROB_MICROUNITS
            or isinstance(self.answer_token_count, bool)
            or not isinstance(self.answer_token_count, int)
            or not 1 <= self.answer_token_count
            <= MAXIMUM_CONTEXT_TOKENS
            or isinstance(self.context_and_answer_token_count, bool)
            or not isinstance(
                self.context_and_answer_token_count, int
            )
            or not self.answer_token_count
            <= self.context_and_answer_token_count
            <= MAXIMUM_CONTEXT_TOKENS
        ):
            raise ClosedChoiceError(
                "closed_choice_teacher_forced_score_invalid"
            )

    @property
    def normalised(self) -> Fraction:
        return Fraction(
            self.total_logprob_microunits,
            self.answer_token_count,
        )


class QualificationScoreBackend(Protocol):
    """Qualification-only scorer interface; never a formal authority."""

    @property
    def runtime_commitment(self) -> str:
        """Return a SHA-256 commitment to the fake/actual runtime closure."""

    def score_batch(
        self, pairs: tuple[PromptAnswer, ...]
    ) -> tuple[TeacherForcedScore, ...]:
        """Teacher-force one non-empty, bounded candidate batch."""

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int:
        """Tokenize the assembled wire JSON without generating any token."""


@dataclass(frozen=True, slots=True)
class ClosedChoiceDecision:
    """Private completion plus a content-free safe receipt."""

    completion: str
    canonical_completion: str
    selected_answer_token_count: int
    wire_completion_token_count: int
    receipt_bytes: bytes

    def __post_init__(self) -> None:
        if (
            not isinstance(self.completion, str)
            or not self.completion
            or not isinstance(self.canonical_completion, str)
            or not self.canonical_completion
            or isinstance(self.selected_answer_token_count, bool)
            or not isinstance(self.selected_answer_token_count, int)
            or not 1 <= self.selected_answer_token_count
            < MAXIMUM_COMPLETION_TOKENS
            or isinstance(self.wire_completion_token_count, bool)
            or not isinstance(self.wire_completion_token_count, int)
            or not 1 <= self.wire_completion_token_count
            < MAXIMUM_COMPLETION_TOKENS
            or not isinstance(self.receipt_bytes, bytes)
            or not self.receipt_bytes
        ):
            raise ClosedChoiceError("closed_choice_decision_invalid")

    @property
    def receipt(self) -> Mapping[str, object]:
        value = json.loads(self.receipt_bytes.decode("ascii"))
        if type(value) is not dict:
            raise ClosedChoiceError("closed_choice_receipt_invalid")
        return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class _Span:
    span_id: str
    quote: str
    occurrence: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    key: str
    payload: object
    pair: PromptAnswer


@dataclass(frozen=True, slots=True)
class _Ranked:
    selected: _Candidate
    selected_score: TeacherForcedScore
    step_receipt: Mapping[str, object]


def _json_string(value: str) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _occurrence_start(
    story_text: str, quote: str, occurrence: int
) -> int:
    positions: list[int] = []
    offset = 0
    while True:
        position = story_text.find(quote, offset)
        if position < 0:
            break
        positions.append(position)
        offset = position + 1
    if not 0 <= occurrence < len(positions):
        raise ClosedChoiceError("closed_choice_span_grounding_invalid")
    return positions[occurrence]


def _catalog_spans(story_text: str) -> tuple[_Span, ...]:
    rows = build_story_span_catalog(story_text)
    spans: list[_Span] = []
    for expected, row in enumerate(rows):
        span_id = row.get("span_id")
        quote = row.get("quote")
        occurrence = row.get("occurrence")
        if (
            span_id != f"s{expected:03d}"
            or not isinstance(quote, str)
            or not quote
            or isinstance(occurrence, bool)
            or not isinstance(occurrence, int)
            or occurrence < 0
        ):
            raise ClosedChoiceError(
                "closed_choice_span_catalog_invalid"
            )
        start = _occurrence_start(story_text, quote, occurrence)
        spans.append(
            _Span(
                span_id=span_id,
                quote=quote,
                occurrence=occurrence,
                start=start,
                end=start + len(quote),
            )
        )
    return tuple(spans)


def _disjoint(left: _Span, right: _Span) -> bool:
    return left.end <= right.start or right.end <= left.start


def _span_answer(span: _Span) -> str:
    return _SPAN_ANSWER.format(
        occurrence=span.occurrence,
        quote_json=_json_string(span.quote),
        span_id=span.span_id,
    )


def _span_summary(span: _Span) -> str:
    return _span_answer(span)


def _candidate_commitment(candidate: _Candidate) -> dict[str, object]:
    return {
        "answer_sha256": hashlib.sha256(
            candidate.pair.answer.encode("utf-8")
        ).hexdigest(),
        "candidate_key_sha256": hashlib.sha256(
            candidate.key.encode("utf-8")
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            candidate.pair.prompt.encode("utf-8")
        ).hexdigest(),
    }


def _rank_candidates(
    *,
    step_name: str,
    candidates: Sequence[_Candidate],
    backend: QualificationScoreBackend,
    scoring_batch_size: int,
) -> _Ranked:
    if (
        not isinstance(step_name, str)
        or not step_name
        or not candidates
        or isinstance(scoring_batch_size, bool)
        or not isinstance(scoring_batch_size, int)
        or not 1 <= scoring_batch_size <= SCORING_BATCH_SIZE
    ):
        raise ClosedChoiceError("closed_choice_ranking_request_invalid")
    if len({candidate.key for candidate in candidates}) != len(candidates):
        raise ClosedChoiceError("closed_choice_candidate_key_duplicate")

    scores: list[TeacherForcedScore] = []
    for offset in range(0, len(candidates), scoring_batch_size):
        batch = tuple(
            candidate.pair
            for candidate in candidates[
                offset : offset + scoring_batch_size
            ]
        )
        try:
            observed = backend.score_batch(batch)
        except ClosedChoiceError:
            raise
        except Exception as exc:
            raise ClosedChoiceError(
                "closed_choice_score_backend_failed"
            ) from exc
        if (
            type(observed) is not tuple
            or len(observed) != len(batch)
            or any(type(score) is not TeacherForcedScore for score in observed)
        ):
            raise ClosedChoiceError(
                "closed_choice_score_batch_invalid"
            )
        scores.extend(observed)

    selected_index = 0
    selected_score = scores[0]
    for index, score in enumerate(scores[1:], start=1):
        # Enumeration index is the explicit and only tie-break.
        if score.normalised > selected_score.normalised:
            selected_index = index
            selected_score = score

    committed_candidates = [
        _candidate_commitment(candidate)
        for candidate in candidates
    ]
    score_rows = [
        {
            "answer_token_count": score.answer_token_count,
            "candidate_commitment": semantic_sha256(
                committed_candidates[index]
            ),
            "context_and_answer_token_count": (
                score.context_and_answer_token_count
            ),
            "total_logprob_microunits": (
                score.total_logprob_microunits
            ),
        }
        for index, score in enumerate(scores)
    ]
    step_receipt = MappingProxyType(
        {
            "candidate_count": len(candidates),
            "candidate_set_commitment": semantic_sha256(
                committed_candidates
            ),
            "score_summary_commitment": semantic_sha256(score_rows),
            "selected_candidate_commitment": semantic_sha256(
                committed_candidates[selected_index]
            ),
            "step": step_name,
        }
    )
    return _Ranked(
        selected=candidates[selected_index],
        selected_score=selected_score,
        step_receipt=step_receipt,
    )


def _feasible_anchors(spans: Sequence[_Span]) -> tuple[_Span, ...]:
    feasible: list[_Span] = []
    for anchor in spans:
        remaining = tuple(
            span for span in spans if _disjoint(anchor, span)
        )
        if any(
            _disjoint(left, right)
            for left, right in itertools.combinations(remaining, 2)
        ):
            feasible.append(anchor)
    return tuple(feasible)


def _feasible_object0(
    spans: Sequence[_Span], anchor: _Span
) -> tuple[_Span, ...]:
    return tuple(
        candidate
        for candidate in spans
        if _disjoint(anchor, candidate)
        and any(
            other is not candidate
            and _disjoint(anchor, other)
            and _disjoint(candidate, other)
            for other in spans
        )
    )


def _feasible_object1(
    spans: Sequence[_Span], anchor: _Span, object0: _Span
) -> tuple[_Span, ...]:
    return tuple(
        candidate
        for candidate in spans
        if candidate is not object0
        and _disjoint(anchor, candidate)
        and _disjoint(object0, candidate)
    )


def _span_candidates(
    *,
    role: str,
    prompt: str,
    spans: Sequence[_Span],
) -> tuple[_Candidate, ...]:
    return tuple(
        _Candidate(
            key=f"{role}:{span.span_id}",
            payload=span,
            pair=PromptAnswer(
                candidate_key=f"{role}:{span.span_id}",
                prompt=prompt,
                answer=_span_answer(span),
            ),
        )
        for span in spans
    )


def _enum_candidates(prompt: str) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    for index, values in enumerate(
        itertools.product(
            GENERATOR_KINDS,
            POLARITIES,
            ORIENTATIONS,
            ORIENTATIONS,
        )
    ):
        kind, polarity, temporal, causal = values
        key = f"enum:{index:03d}"
        candidates.append(
            _Candidate(
                key=key,
                payload=values,
                pair=PromptAnswer(
                    candidate_key=key,
                    prompt=prompt,
                    answer=_ENUM_ANSWER.format(
                        causal=causal,
                        kind=kind,
                        polarity=polarity,
                        temporal=temporal,
                    ),
                ),
            )
        )
    return tuple(candidates)


def _wire_completion(
    *,
    anchor: _Span,
    object0: _Span,
    object1: _Span,
    enum_values: tuple[str, str, str, str],
) -> str:
    kind, polarity, temporal, causal = enum_values
    # Every key and every primitive below is program-owned.  Model output
    # cannot make a required field absent or an enum value unknown.
    return canonical_json_bytes(
        {
            "generators": [
                {
                    "anchor_span_id": anchor.span_id,
                    "causal_orientation": causal,
                    "generator_id": "g0",
                    "generator_kind": kind,
                    "polarity": polarity,
                    "slot_object_ids": ["o0", "o1"],
                    "temporal_orientation": temporal,
                }
            ],
            "objects": [
                {"object_id": "o0", "span_id": object0.span_id},
                {"object_id": "o1", "span_id": object1.span_id},
            ],
            "schema_version": WIRE_COMPLETION_SCHEMA,
        },
        newline=False,
    ).decode("ascii")


class _ClosedChoiceEngine:
    __slots__ = ("_marker",)

    def __init__(self, marker: object) -> None:
        if marker is not _ENGINE_MARKER:
            raise ClosedChoiceError(
                "closed_choice_engine_authority_invalid"
            )
        self._marker = marker

    def select(
        self,
        story_text: str,
        *,
        backend: QualificationScoreBackend,
        narrative_parser: NarrativeParser,
        scoring_batch_size: int,
    ) -> ClosedChoiceDecision:
        if (
            type(self) is not _ClosedChoiceEngine
            or self._marker is not _ENGINE_MARKER
        ):
            raise ClosedChoiceError(
                "closed_choice_engine_authority_invalid"
            )
        if (
            not isinstance(story_text, str)
            or not story_text
            or len(story_text.encode("utf-8", errors="strict"))
            > MAXIMUM_STORY_BYTES
            or not callable(narrative_parser)
            or not isinstance(
                getattr(backend, "runtime_commitment", None), str
            )
            or len(backend.runtime_commitment) != 64
            or any(
                character not in "0123456789abcdef"
                for character in backend.runtime_commitment
            )
        ):
            raise ClosedChoiceError(
                "closed_choice_selection_request_invalid"
            )

        spans = _catalog_spans(story_text)
        anchors = _feasible_anchors(spans)
        if not anchors:
            raise ClosedChoiceAbstention(
                "closed_choice_nonoverlapping_triple_unavailable",
                pre_model=True,
            )
        story_json = _json_string(story_text)
        anchor_prompt = _ANCHOR_PROMPT.format(
            story_json=story_json
        )
        anchor_ranked = _rank_candidates(
            step_name="anchor_span",
            candidates=_span_candidates(
                role="anchor",
                prompt=anchor_prompt,
                spans=anchors,
            ),
            backend=backend,
            scoring_batch_size=scoring_batch_size,
        )
        anchor = anchor_ranked.selected.payload
        if not isinstance(anchor, _Span):
            raise ClosedChoiceError(
                "closed_choice_internal_span_invalid"
            )

        object0_spans = _feasible_object0(spans, anchor)
        if not object0_spans:
            raise ClosedChoiceError(
                "closed_choice_feasibility_drifted"
            )
        object0_prompt = _OBJECT0_PROMPT.format(
            anchor_json=_json_string(_span_summary(anchor)),
            story_json=story_json,
        )
        object0_ranked = _rank_candidates(
            step_name="object0_span",
            candidates=_span_candidates(
                role="object0",
                prompt=object0_prompt,
                spans=object0_spans,
            ),
            backend=backend,
            scoring_batch_size=scoring_batch_size,
        )
        object0 = object0_ranked.selected.payload
        if not isinstance(object0, _Span):
            raise ClosedChoiceError(
                "closed_choice_internal_span_invalid"
            )

        object1_spans = _feasible_object1(spans, anchor, object0)
        if not object1_spans:
            raise ClosedChoiceError(
                "closed_choice_feasibility_drifted"
            )
        object1_prompt = _OBJECT1_PROMPT.format(
            anchor_json=_json_string(_span_summary(anchor)),
            object0_json=_json_string(_span_summary(object0)),
            story_json=story_json,
        )
        object1_ranked = _rank_candidates(
            step_name="object1_span",
            candidates=_span_candidates(
                role="object1",
                prompt=object1_prompt,
                spans=object1_spans,
            ),
            backend=backend,
            scoring_batch_size=scoring_batch_size,
        )
        object1 = object1_ranked.selected.payload
        if not isinstance(object1, _Span):
            raise ClosedChoiceError(
                "closed_choice_internal_span_invalid"
            )

        enum_prompt = _ENUM_PROMPT.format(
            anchor_json=_json_string(_span_summary(anchor)),
            objects_json=_json_string(
                f"{_span_summary(object0)} | {_span_summary(object1)}"
            ),
            story_json=story_json,
        )
        enum_ranked = _rank_candidates(
            step_name="generator_attributes",
            candidates=_enum_candidates(enum_prompt),
            backend=backend,
            scoring_batch_size=scoring_batch_size,
        )
        enum_values = enum_ranked.selected.payload
        if (
            not isinstance(enum_values, tuple)
            or len(enum_values) != 4
        ):
            raise ClosedChoiceError(
                "closed_choice_internal_enum_invalid"
            )

        completion = _wire_completion(
            anchor=anchor,
            object0=object0,
            object1=object1,
            enum_values=enum_values,
        )
        canonical = validate_completion(
            story_text,
            completion,
            narrative_parser=narrative_parser,
        )
        selected_token_count = sum(
            ranked.selected_score.answer_token_count
            for ranked in (
                anchor_ranked,
                object0_ranked,
                object1_ranked,
                enum_ranked,
            )
        )
        if not 1 <= selected_token_count < MAXIMUM_COMPLETION_TOKENS:
            raise ClosedChoiceError(
                "closed_choice_selected_answer_token_count_invalid"
            )
        try:
            wire_token_count = (
                backend.count_program_owned_completion_tokens(
                    completion
                )
            )
        except ClosedChoiceError:
            raise
        except Exception as exc:
            raise ClosedChoiceError(
                "closed_choice_wire_tokenizer_failed"
            ) from exc
        if (
            isinstance(wire_token_count, bool)
            or not isinstance(wire_token_count, int)
            or not 1 <= wire_token_count < MAXIMUM_COMPLETION_TOKENS
        ):
            raise ClosedChoiceError(
                "closed_choice_wire_token_count_invalid"
            )
        step_receipts = [
            dict(ranked.step_receipt)
            for ranked in (
                anchor_ranked,
                object0_ranked,
                object1_ranked,
                enum_ranked,
            )
        ]
        body: dict[str, object] = {
            "canonical_completion_commitment": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
            "claim_scope": CLAIM_SCOPE,
            "free_form_generation_count": 0,
            "model_runtime_commitment": backend.runtime_commitment,
            "parser_abi": "validate_completion_then_narrative_parser",
            "prompt_closure_sha256": PROMPT_CLOSURE_SHA256,
            "schema": RECEIPT_SCHEMA,
            "scoring_policy_sha256": semantic_sha256(
                dict(SCORING_POLICY)
            ),
            "selection_commitment": hashlib.sha256(
                completion.encode("utf-8")
            ).hexdigest(),
            "selected_answer_token_count": selected_token_count,
            "span_catalog_commitment": semantic_sha256(
                [
                    {
                        "occurrence": span.occurrence,
                        "quote": span.quote,
                        "span_id": span.span_id,
                    }
                    for span in spans
                ]
            ),
            "steps": step_receipts,
            "story_commitment": hashlib.sha256(
                story_text.encode("utf-8")
            ).hexdigest(),
            "version": VERSION,
            "wire_shape": {
                "generator_count": 1,
                "object_count": 2,
                "slot_count": 2,
            },
        }
        receipt = {
            **body,
            "self_sha256": semantic_sha256(body),
        }
        return ClosedChoiceDecision(
            completion=completion,
            canonical_completion=canonical,
            selected_answer_token_count=selected_token_count,
            wire_completion_token_count=wire_token_count,
            receipt_bytes=canonical_json_bytes(receipt),
        )


def select_closed_choice_qualification_only(
    story_text: str,
    *,
    backend: QualificationScoreBackend,
    narrative_parser: NarrativeParser,
    scoring_batch_size: int = SCORING_BATCH_SIZE,
) -> ClosedChoiceDecision:
    """Run the source-free/fake-logit harness, never a formal measurement."""

    engine = _ClosedChoiceEngine(_ENGINE_MARKER)
    return engine.select(
        story_text,
        backend=backend,
        narrative_parser=narrative_parser,
        scoring_batch_size=scoring_batch_size,
    )


__all__ = [
    "CLAIM_SCOPE",
    "ClosedChoiceAbstention",
    "ClosedChoiceDecision",
    "ClosedChoiceError",
    "GENERATOR_KINDS",
    "LOGPROB_QUANTIZATION_SCALE",
    "MAXIMUM_CONTEXT_TOKENS",
    "ORIENTATIONS",
    "POLARITIES",
    "PROMPT_CLOSURE_SHA256",
    "PromptAnswer",
    "QualificationScoreBackend",
    "RECEIPT_SCHEMA",
    "SCORING_BATCH_SIZE",
    "SCORING_POLICY",
    "TeacherForcedScore",
    "VERSION",
    "select_closed_choice_qualification_only",
]

"""Hierarchical, program-owned closed-choice narrative extractor v2.

The document is partitioned deterministically into parser-aligned sentences
and balanced local episodes.  For each sentence, the model first ranks a
typed NO/ONE/TWO relation plan and then one eligible local episode.  For each
selected relation it ranks only finite alternatives:

* a coarse group of at most six atomic grounded heads;
* one atomic head inside the selected group;
* one admissible forward boundary width from one to four lexical tokens;
* one value from each frozen enum domain.

It never authors JSON, field names, identifiers, references, or primitives.
The program assembles a complete typed wire, binds both independently ranked
object endpoints directly, normalises it to the existing independent parser
ABI, and records bounded resource receipts.  This is a qualification-only
dependency-injection surface; formal execution must own an exact private
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import re
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from replication_runtime.gscl_narrative_extractor_v1.closed_choice_worker import (
    MAXIMUM_ABSOLUTE_LOGPROB_MICROUNITS,
    PromptAnswer,
)
from replication_runtime.gscl_narrative_extractor_v1.contract import (
    COMPLETION_SCHEMA,
    canonical_json_bytes,
    semantic_sha256,
)

from .contract import (
    HIERARCHICAL_WIRE_SCHEMA,
    ClosedChoiceV2Abstention,
    ClosedChoiceV2Error,
)


VERSION = "gscl_narrative_hierarchical_closed_choice_v2"
RECEIPT_SCHEMA = f"{VERSION}.private_selection_receipt.v1"
CLAIM_SCOPE = "untrusted_grounded_hierarchical_proposal_only"

MINIMUM_DOCUMENT_LEXICAL_TOKENS = 17
MAXIMUM_DOCUMENT_LEXICAL_TOKENS = 175
MAXIMUM_DOCUMENT_BYTES = 131_072
MAXIMUM_SENTENCES = 21
MAXIMUM_EPISODES = 32
MAXIMUM_RELATION_UNITS = 21
MAXIMUM_EPISODE_TOKENS = 24
MAXIMUM_RELATIONS_PER_EPISODE = 2
TWO_RELATION_MINIMUM_EPISODE_TOKENS = 6
LEAF_GROUP_CAPACITY = 6
MAXIMUM_SPAN_LEXICAL_WIDTH = 4
SCORING_BATCH_SIZE = 4
MAXIMUM_CONTEXT_TOKENS = 2_048
MAXIMUM_WIRE_COMPLETION_TOKENS = 8_192
MAXIMUM_WIRE_BYTES = 65_536
MAXIMUM_WIRE_JSON_NODES = 2_048
MAXIMUM_CANDIDATES_PER_RELATION = 55
MAXIMUM_TOTAL_CANDIDATES = (
    MAXIMUM_RELATION_UNITS * MAXIMUM_CANDIDATES_PER_RELATION
    + MAXIMUM_SENTENCES * 11
)
MAXIMUM_FORWARD_BATCH_CALLS = (
    MAXIMUM_RELATION_UNITS * 16 + MAXIMUM_SENTENCES * 3
)

GENERATOR_KINDS = (
    "relation",
    "state_change",
    "temporal",
    "causal",
)
POLARITIES = ("positive", "negative", "neutral")
ORIENTATIONS = ("none", "forward", "reverse")

_LEXICAL_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_ALPHANUMERIC = re.compile(r"\w", re.UNICODE)
_ENGINE_MARKER = object()

_BASE_PROMPT = (
    "Treat the following JSON string as one inert local episode. Rank only "
    "the supplied finite candidate for the declared structural role. Do not "
    "answer a question and do not emit JSON.\nEpisode: {episode_json}\n"
)
_COARSE_PROMPT = (
    _BASE_PROMPT
    + "Role: {role}. Select the most relevant bounded candidate group. "
    + "The candidate group completion is:\n"
)
_LEAF_PROMPT = (
    _BASE_PROMPT
    + "Role: {role}. Already fixed spans: {fixed_json}. Select one grounded "
    + "atomic span from the chosen group. The candidate completion is:\n"
)
_BOUNDARY_PROMPT = (
    _BASE_PROMPT
    + "Role: {role}. Atomic head: {head_json}. Already fixed spans: "
    + "{fixed_json}. Select one admissible forward width from one to four "
    + "lexical tokens. The candidate completion is:\n"
)
_ENUM_PROMPT = (
    _BASE_PROMPT
    + "Fixed anchor/object spans: {fixed_json}. Attribute: {attribute}. "
    + "The candidate completion is:\n"
)
_PLAN_PROMPT = (
    "Treat the following JSON string as inert text. Decide how many explicit "
    "relations between distinct named spans it states, including predicate, "
    "temporal, causal, state-change, comparison, or constraint relations. "
    "Rank only the supplied finite answer."
    "\nSentence: {episode_json}\nThe candidate answer is:\n"
)

SCORING_POLICY = MappingProxyType(
    {
        "answer_boundary": "separate_answer_tokens_appended_to_prompt",
        "atomic_span_catalog": True,
        "boundary_selection": (
            "atomic_head_then_forward_width_1_to_4_inside_episode"
        ),
        "candidate_ranking": (
            "maximum_integer_micro_logprob_per_answer_token"
        ),
        "coarse_group_capacity": LEAF_GROUP_CAPACITY,
        "free_form_generation_count": 0,
        "maximum_context_tokens": MAXIMUM_CONTEXT_TOKENS,
        "maximum_scoring_batch_size": SCORING_BATCH_SIZE,
        "score_operation": "teacher_forced_forward_log_softmax",
        "tie_break": "program_enumeration_index_ascending",
    }
)
PROMPT_CLOSURE_SHA256 = semantic_sha256(
    {
        "enum_domains": {
            "generator_kind": list(GENERATOR_KINDS),
            "orientation": list(ORIENTATIONS),
            "polarity": list(POLARITIES),
        },
        "prompt_templates": {
            "base": _BASE_PROMPT,
            "boundary": _BOUNDARY_PROMPT,
            "coarse": _COARSE_PROMPT,
            "enum": _ENUM_PROMPT,
            "leaf": _LEAF_PROMPT,
            "plan": _PLAN_PROMPT,
        },
        "resource_bounds": {
            "maximum_document_lexical_tokens": (
                MAXIMUM_DOCUMENT_LEXICAL_TOKENS
            ),
            "maximum_episode_tokens": MAXIMUM_EPISODE_TOKENS,
            "maximum_episodes": MAXIMUM_EPISODES,
            "maximum_relation_units": MAXIMUM_RELATION_UNITS,
            "maximum_sentences": MAXIMUM_SENTENCES,
            "maximum_span_lexical_width": (
                MAXIMUM_SPAN_LEXICAL_WIDTH
            ),
        },
        "scoring_policy": dict(SCORING_POLICY),
        "version": VERSION,
        "wire_schema": HIERARCHICAL_WIRE_SCHEMA,
    }
)


@dataclass(frozen=True, slots=True)
class TeacherForcedScore:
    """V2 score record with the v2 context bound, not v1's 512 ceiling."""

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
            or isinstance(
                self.context_and_answer_token_count, bool
            )
            or not isinstance(
                self.context_and_answer_token_count, int
            )
            or not 1
            <= self.answer_token_count
            <= self.context_and_answer_token_count
            <= MAXIMUM_CONTEXT_TOKENS
        ):
            raise ClosedChoiceV2Error(
                "V2_MODEL_SCORE_BATCH_INVALID"
            )

    @property
    def normalised(self) -> Fraction:
        return Fraction(
            self.total_logprob_microunits,
            self.answer_token_count,
        )


class QualificationScoreBackendV2(Protocol):
    @property
    def runtime_commitment(self) -> str: ...

    def score_batch(
        self, pairs: tuple[PromptAnswer, ...]
    ) -> tuple[TeacherForcedScore, ...]: ...

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int: ...


NarrativeParser = Callable[[str, str], object]


@dataclass(frozen=True, slots=True)
class GroundedAtom:
    span_id: str
    quote: str
    occurrence: int
    start: int
    end: int
    sentence_index: int
    episode_index: int
    lexical_index: int


@dataclass(frozen=True, slots=True)
class GroundedSpan:
    span_id: str
    quote: str
    occurrence: int
    start: int
    end: int
    sentence_index: int
    episode_index: int
    lexical_index: int
    lexical_width: int


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    sentence_id: str
    text: str
    atoms: tuple[GroundedAtom, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    key: str
    payload: object
    pair: PromptAnswer


@dataclass(frozen=True, slots=True)
class _Ranked:
    selected: _Candidate
    score: TeacherForcedScore
    receipt: Mapping[str, object]
    candidate_count: int
    forward_batch_count: int


@dataclass(frozen=True, slots=True)
class _RelationSelection:
    relation_id: str
    episode_id: str
    sentence_id: str
    anchor: GroundedSpan
    object0: GroundedSpan
    object1: GroundedSpan
    generator_kind: str
    polarity: str
    temporal_orientation: str
    causal_orientation: str


@dataclass(frozen=True, slots=True)
class ClosedChoiceV2Decision:
    wire_completion: str
    canonical_completion: str
    extraction: object
    selected_answer_token_count: int
    wire_completion_token_count: int
    receipt_bytes: bytes

    @property
    def receipt(self) -> Mapping[str, object]:
        value = json.loads(self.receipt_bytes.decode("ascii"))
        if type(value) is not dict:
            raise ClosedChoiceV2Error("V2_VERIFIER_REJECTED")
        return MappingProxyType(value)


def _sentence_character_spans(
    story_text: str,
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(story_text):
        if character in ".?!\n":
            segment = story_text[start : index + 1]
            if _ALPHANUMERIC.search(segment):
                spans.append((start, index + 1))
            start = index + 1
    if (
        start < len(story_text)
        and _ALPHANUMERIC.search(story_text[start:])
    ):
        spans.append((start, len(story_text)))
    return tuple(spans)


def _occurrence(story_text: str, quote: str, start: int) -> int:
    positions: list[int] = []
    offset = 0
    while True:
        position = story_text.find(quote, offset)
        if position < 0:
            break
        positions.append(position)
        offset = position + 1
    try:
        return positions.index(start)
    except ValueError as exc:
        raise ClosedChoiceV2Abstention(
            "V2_CATALOG_SPAN_GROUNDING_INVALID",
            before_model_forward=True,
        ) from exc


def _balanced_sizes(total: int, count: int) -> tuple[int, ...]:
    base, remainder = divmod(total, count)
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(count)
    )


def build_hierarchical_episodes(
    story_text: str,
) -> tuple[Episode, ...]:
    """Build atomic local catalogs with no global n-gram enumeration."""

    if not isinstance(story_text, str) or not story_text:
        raise ClosedChoiceV2Abstention(
            "V2_CATALOG_DOCUMENT_TOKEN_COUNT_UNSUPPORTED",
            before_model_forward=True,
        )
    try:
        story_bytes = story_text.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ClosedChoiceV2Abstention(
            "V2_CATALOG_SPAN_GROUNDING_INVALID",
            before_model_forward=True,
        ) from exc
    if len(story_bytes) > MAXIMUM_DOCUMENT_BYTES:
        raise ClosedChoiceV2Abstention(
            "V2_CATALOG_DOCUMENT_TOKEN_COUNT_UNSUPPORTED",
            before_model_forward=True,
        )
    sentence_spans = _sentence_character_spans(story_text)
    if not 1 <= len(sentence_spans) <= MAXIMUM_SENTENCES:
        raise ClosedChoiceV2Abstention(
            "V2_CATALOG_SENTENCE_CAPACITY_UNSUPPORTED",
            before_model_forward=True,
        )
    sentence_tokens: list[
        tuple[tuple[int, int, str], ...]
    ] = []
    document_token_count = 0
    for left, right in sentence_spans:
        tokens = tuple(
            (
                left + match.start(),
                left + match.end(),
                story_text[
                    left + match.start() : left + match.end()
                ],
            )
            for match in _LEXICAL_TOKEN.finditer(
                story_text[left:right]
            )
        )
        if len(tokens) < 3:
            raise ClosedChoiceV2Abstention(
                "V2_CATALOG_SENTENCE_TOKEN_COUNT_UNSUPPORTED",
                before_model_forward=True,
            )
        document_token_count += len(tokens)
        sentence_tokens.append(tokens)
    if not MINIMUM_DOCUMENT_LEXICAL_TOKENS <= document_token_count <= (
        MAXIMUM_DOCUMENT_LEXICAL_TOKENS
    ):
        raise ClosedChoiceV2Abstention(
            "V2_CATALOG_DOCUMENT_TOKEN_COUNT_UNSUPPORTED",
            before_model_forward=True,
        )

    episode_specs: list[
        tuple[int, tuple[tuple[int, int, str], ...]]
    ] = []
    for sentence_index, tokens in enumerate(sentence_tokens):
        episode_count = math.ceil(
            len(tokens) / MAXIMUM_EPISODE_TOKENS
        )
        offset = 0
        for size in _balanced_sizes(len(tokens), episode_count):
            episode_specs.append(
                (sentence_index, tokens[offset : offset + size])
            )
            offset += size
    if not 1 <= len(episode_specs) <= MAXIMUM_EPISODES:
        raise ClosedChoiceV2Abstention(
            "V2_CATALOG_EPISODE_CAPACITY_UNSUPPORTED",
            before_model_forward=True,
        )

    episodes: list[Episode] = []
    for episode_index, (sentence_index, tokens) in enumerate(
        episode_specs
    ):
        atoms = tuple(
            GroundedAtom(
                span_id=f"e{episode_index:02d}.t{local_index:02d}",
                quote=quote,
                occurrence=_occurrence(
                    story_text, quote, start
                ),
                start=start,
                end=end,
                sentence_index=sentence_index,
                episode_index=episode_index,
                lexical_index=local_index,
            )
            for local_index, (start, end, quote) in enumerate(
                tokens
            )
        )
        if any(
            len(atom.quote.encode("utf-8", errors="strict")) > 4_096
            for atom in atoms
        ):
            raise ClosedChoiceV2Abstention(
                "V2_CATALOG_SPAN_GROUNDING_INVALID",
                before_model_forward=True,
            )
        episode_start = tokens[0][0]
        episode_end = tokens[-1][1]
        episodes.append(
            Episode(
                episode_id=f"e{episode_index:02d}",
                sentence_id=f"s{sentence_index:02d}",
                text=story_text[episode_start:episode_end],
                atoms=atoms,
            )
        )
    return tuple(episodes)


def _json_string(value: str) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":")
    )


def _span_answer(span: GroundedAtom | GroundedSpan) -> str:
    return (
        f"span_id={span.span_id}; occurrence={span.occurrence}; "
        f"quote={_json_string(span.quote)}"
    )


def _group_answer(group: Sequence[GroundedAtom]) -> str:
    return (
        f"first={group[0].span_id}; last={group[-1].span_id}; "
        f"bounded_text={_json_string(' '.join(row.quote for row in group))}"
    )


def _intervals_overlap(
    left: GroundedSpan, right: GroundedSpan
) -> bool:
    return left.start < right.end and right.start < left.end


def _grounded_span_from_head(
    *,
    story_text: str,
    episode: Episode,
    lexical_index: int,
    lexical_width: int,
) -> GroundedSpan:
    if (
        isinstance(lexical_index, bool)
        or not isinstance(lexical_index, int)
        or isinstance(lexical_width, bool)
        or not isinstance(lexical_width, int)
        or not 0 <= lexical_index < len(episode.atoms)
        or not 1 <= lexical_width <= MAXIMUM_SPAN_LEXICAL_WIDTH
        or lexical_index + lexical_width > len(episode.atoms)
    ):
        raise ClosedChoiceV2Error("V2_WIRE_ENDPOINT_REF_INVALID")
    first = episode.atoms[lexical_index]
    last = episode.atoms[lexical_index + lexical_width - 1]
    if (
        first.episode_index != last.episode_index
        or first.sentence_index != last.sentence_index
    ):
        raise ClosedChoiceV2Error("V2_WIRE_ENDPOINT_REF_INVALID")
    quote = story_text[first.start : last.end]
    if (
        not quote
        or len(quote.encode("utf-8", errors="strict")) > 4_096
    ):
        raise ClosedChoiceV2Error("V2_WIRE_ENDPOINT_REF_INVALID")
    return GroundedSpan(
        span_id=(
            f"{episode.episode_id}.t{lexical_index:02d}."
            f"w{lexical_width}"
        ),
        quote=quote,
        occurrence=_occurrence(story_text, quote, first.start),
        start=first.start,
        end=last.end,
        sentence_index=first.sentence_index,
        episode_index=first.episode_index,
        lexical_index=lexical_index,
        lexical_width=lexical_width,
    )


_SPAN_REFERENCE = re.compile(
    r"(e[0-9]{2})\.t([0-9]{2})\.w([1-4])\Z"
)


def _resolve_span_reference(
    *,
    story_text: str,
    episode: Episode,
    span_id: object,
) -> GroundedSpan:
    if not isinstance(span_id, str):
        raise ClosedChoiceV2Error("V2_WIRE_ENDPOINT_REF_INVALID")
    match = _SPAN_REFERENCE.fullmatch(span_id)
    if match is None or match.group(1) != episode.episode_id:
        raise ClosedChoiceV2Error("V2_WIRE_ENDPOINT_REF_INVALID")
    span = _grounded_span_from_head(
        story_text=story_text,
        episode=episode,
        lexical_index=int(match.group(2)),
        lexical_width=int(match.group(3)),
    )
    if span.span_id != span_id:
        raise ClosedChoiceV2Error("V2_WIRE_ENDPOINT_REF_INVALID")
    return span


def _candidate_commitment(candidate: _Candidate) -> dict[str, str]:
    return {
        "answer_sha256": hashlib.sha256(
            candidate.pair.answer.encode("utf-8")
        ).hexdigest(),
        "candidate_key_sha256": hashlib.sha256(
            candidate.key.encode("ascii")
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            candidate.pair.prompt.encode("utf-8")
        ).hexdigest(),
    }


def _rank(
    *,
    step: str,
    candidates: Sequence[_Candidate],
    backend: QualificationScoreBackendV2,
) -> _Ranked:
    if not candidates or len(candidates) > 8:
        raise ClosedChoiceV2Error(
            "V2_MODEL_SCORE_BATCH_INVALID"
        )
    scores: list[TeacherForcedScore] = []
    calls = 0
    for offset in range(0, len(candidates), SCORING_BATCH_SIZE):
        batch = tuple(
            candidate.pair
            for candidate in candidates[
                offset : offset + SCORING_BATCH_SIZE
            ]
        )
        calls += 1
        try:
            observed = backend.score_batch(batch)
        except ClosedChoiceV2Error:
            raise
        except Exception as exc:
            raise ClosedChoiceV2Error(
                "V2_MODEL_FORWARD_FAILED"
            ) from exc
        if (
            type(observed) is not tuple
            or len(observed) != len(batch)
            or any(type(row) is not TeacherForcedScore for row in observed)
            or any(
                isinstance(row.answer_token_count, bool)
                or not isinstance(row.answer_token_count, int)
                or row.answer_token_count < 1
                or isinstance(
                    row.context_and_answer_token_count, bool
                )
                or not isinstance(
                    row.context_and_answer_token_count, int
                )
                or row.context_and_answer_token_count
                < row.answer_token_count
                or isinstance(
                    row.total_logprob_microunits, bool
                )
                or not isinstance(
                    row.total_logprob_microunits, int
                )
                for row in observed
            )
        ):
            raise ClosedChoiceV2Error(
                "V2_MODEL_SCORE_BATCH_INVALID"
            )
        scores.extend(observed)
    selected_index = 0
    best = Fraction(
        scores[0].total_logprob_microunits,
        scores[0].answer_token_count,
    )
    for index, score in enumerate(scores[1:], start=1):
        value = Fraction(
            score.total_logprob_microunits,
            score.answer_token_count,
        )
        if value > best:
            selected_index = index
            best = value
    candidate_rows = [
        _candidate_commitment(candidate)
        for candidate in candidates
    ]
    score_rows = [
        {
            "answer_token_count": score.answer_token_count,
            "candidate_commitment": semantic_sha256(
                candidate_rows[index]
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
    return _Ranked(
        selected=candidates[selected_index],
        score=scores[selected_index],
        receipt=MappingProxyType(
            {
                "candidate_count": len(candidates),
                "candidate_set_commitment": semantic_sha256(
                    candidate_rows
                ),
                "score_commitment": semantic_sha256(score_rows),
                "selected_candidate_commitment": semantic_sha256(
                    candidate_rows[selected_index]
                ),
                "step": step,
            }
        ),
        candidate_count=len(candidates),
        forward_batch_count=calls,
    )


def _admissible_widths(
    *,
    story_text: str,
    episode: Episode,
    head: GroundedAtom,
    fixed: Sequence[GroundedSpan],
    reserve_atomic_tokens: int,
) -> tuple[GroundedSpan, ...]:
    rows: list[GroundedSpan] = []
    for width in range(1, MAXIMUM_SPAN_LEXICAL_WIDTH + 1):
        if head.lexical_index + width > len(episode.atoms):
            break
        candidate = _grounded_span_from_head(
            story_text=story_text,
            episode=episode,
            lexical_index=head.lexical_index,
            lexical_width=width,
        )
        if any(
            _intervals_overlap(candidate, prior) for prior in fixed
        ):
            continue
        remaining = sum(
            1
            for atom in episode.atoms
            if not any(
                atom.start < occupied.end
                and occupied.start < atom.end
                for occupied in (*fixed, candidate)
            )
        )
        if remaining >= reserve_atomic_tokens:
            rows.append(candidate)
    return tuple(rows)


def _select_span(
    *,
    story_text: str,
    episode: Episode,
    role: str,
    fixed: Sequence[GroundedSpan],
    reserve_atomic_tokens: int,
    backend: QualificationScoreBackendV2,
) -> tuple[GroundedSpan, tuple[_Ranked, _Ranked, _Ranked]]:
    if (
        isinstance(reserve_atomic_tokens, bool)
        or not isinstance(reserve_atomic_tokens, int)
        or reserve_atomic_tokens < 0
    ):
        raise ClosedChoiceV2Error(
            "V2_PLAN_CANDIDATE_SET_INVALID"
        )
    available = tuple(
        atom
        for atom in episode.atoms
        if _admissible_widths(
            story_text=story_text,
            episode=episode,
            head=atom,
            fixed=fixed,
            reserve_atomic_tokens=reserve_atomic_tokens,
        )
    )
    groups = tuple(
        tuple(
            available[
                offset : offset + LEAF_GROUP_CAPACITY
            ]
        )
        for offset in range(
            0, len(available), LEAF_GROUP_CAPACITY
        )
    )
    if not groups or len(groups) > 4:
        raise ClosedChoiceV2Abstention(
            "V2_CATALOG_EPISODE_CAPACITY_UNSUPPORTED",
            before_model_forward=True,
        )
    coarse_prompt = _COARSE_PROMPT.format(
        episode_json=_json_string(episode.text),
        role=role,
    )
    coarse_candidates = tuple(
        _Candidate(
            key=f"{role}.group.{index:02d}",
            payload=group,
            pair=PromptAnswer(
                candidate_key=f"{role}.group.{index:02d}",
                prompt=coarse_prompt,
                answer=_group_answer(group),
            ),
        )
        for index, group in enumerate(groups)
    )
    coarse = _rank(
        step=f"{role}.coarse",
        candidates=coarse_candidates,
        backend=backend,
    )
    selected_group = coarse.selected.payload
    if not isinstance(selected_group, tuple):
        raise ClosedChoiceV2Error(
            "V2_VERIFIER_REJECTED"
        )
    leaf_prompt = _LEAF_PROMPT.format(
        episode_json=_json_string(episode.text),
        fixed_json=_json_string(
            " | ".join(_span_answer(row) for row in fixed)
        ),
        role=role,
    )
    leaf_candidates = tuple(
        _Candidate(
            key=f"{role}.leaf.{atom.span_id}",
            payload=atom,
            pair=PromptAnswer(
                candidate_key=f"{role}.leaf.{atom.span_id}",
                prompt=leaf_prompt,
                answer=_span_answer(atom),
            ),
        )
        for atom in selected_group
    )
    leaf = _rank(
        step=f"{role}.leaf",
        candidates=leaf_candidates,
        backend=backend,
    )
    atom = leaf.selected.payload
    if not isinstance(atom, GroundedAtom):
        raise ClosedChoiceV2Error("V2_VERIFIER_REJECTED")
    widths = _admissible_widths(
        story_text=story_text,
        episode=episode,
        head=atom,
        fixed=fixed,
        reserve_atomic_tokens=reserve_atomic_tokens,
    )
    if not widths:
        raise ClosedChoiceV2Error(
            "V2_PLAN_CANDIDATE_SET_INVALID"
        )
    boundary_prompt = _BOUNDARY_PROMPT.format(
        episode_json=_json_string(episode.text),
        fixed_json=_json_string(
            " | ".join(_span_answer(row) for row in fixed)
        ),
        head_json=_json_string(_span_answer(atom)),
        role=role,
    )
    boundary_candidates = tuple(
        _Candidate(
            key=f"{role}.width.{span.lexical_width}",
            payload=span,
            pair=PromptAnswer(
                candidate_key=(
                    f"{role}.width.{span.lexical_width}"
                ),
                prompt=boundary_prompt,
                answer=_span_answer(span),
            ),
        )
        for span in widths
    )
    boundary = _rank(
        step=f"{role}.boundary",
        candidates=boundary_candidates,
        backend=backend,
    )
    span = boundary.selected.payload
    if not isinstance(span, GroundedSpan):
        raise ClosedChoiceV2Error("V2_VERIFIER_REJECTED")
    return span, (coarse, leaf, boundary)


def _select_enum(
    *,
    episode: Episode,
    relation_id: str,
    attribute: str,
    domain: Sequence[str],
    fixed: Sequence[GroundedSpan],
    backend: QualificationScoreBackendV2,
) -> tuple[str, _Ranked]:
    prompt = _ENUM_PROMPT.format(
        attribute=attribute,
        episode_json=_json_string(episode.text),
        fixed_json=_json_string(
            " | ".join(_span_answer(row) for row in fixed)
        ),
    )
    candidates = tuple(
        _Candidate(
            key=f"{relation_id}.{attribute}.{index:02d}",
            payload=value,
            pair=PromptAnswer(
                candidate_key=(
                    f"{relation_id}.{attribute}.{index:02d}"
                ),
                prompt=prompt,
                answer=f"{attribute}={value}",
            ),
        )
        for index, value in enumerate(domain)
    )
    ranked = _rank(
        step=f"{relation_id}.{attribute}",
        candidates=candidates,
        backend=backend,
    )
    value = ranked.selected.payload
    if not isinstance(value, str) or value not in domain:
        raise ClosedChoiceV2Error("V2_VERIFIER_REJECTED")
    return value, ranked


def _select_sentence_plan(
    *,
    sentence_id: str,
    episodes: Sequence[Episode],
    backend: QualificationScoreBackendV2,
) -> tuple[Episode, int, tuple[_Ranked, ...]]:
    if (
        not episodes
        or any(row.sentence_id != sentence_id for row in episodes)
        or len(episodes) > 8
    ):
        raise ClosedChoiceV2Error(
            "V2_PLAN_CANDIDATE_SET_INVALID"
        )
    sentence_excerpt = " ".join(row.text for row in episodes)
    plan_prompt = _PLAN_PROMPT.format(
        episode_json=_json_string(sentence_excerpt),
    )
    count_domain: list[tuple[str, int, str]] = [
        ("ONE_RELATION", 1, "exactly one explicit relation"),
        ("NO_RELATION", 0, "no explicit relation"),
    ]
    if any(
        len(episode.atoms)
        >= TWO_RELATION_MINIMUM_EPISODE_TOKENS
        for episode in episodes
    ):
        count_domain.append(
            (
                "TWO_RELATIONS",
                2,
                "at least two explicit relations",
            )
        )
    plan_candidates = tuple(
        _Candidate(
            key=f"{sentence_id}.plan.{name.lower()}",
            payload=count,
            pair=PromptAnswer(
                candidate_key=(
                    f"{sentence_id}.plan.{name.lower()}"
                ),
                prompt=plan_prompt,
                answer=answer,
            ),
        )
        for name, count, answer in count_domain
    )
    plan_ranked = _rank(
        step=f"{sentence_id}.sentence_plan",
        candidates=plan_candidates,
        backend=backend,
    )
    relation_count = plan_ranked.selected.payload
    if relation_count == 0:
        raise ClosedChoiceV2Abstention(
            "V2_PLAN_NO_RELATION_SELECTED",
            before_model_forward=False,
        )
    if relation_count not in {1, 2}:
        raise ClosedChoiceV2Error(
            "V2_PLAN_CANDIDATE_SET_INVALID"
        )
    eligible = tuple(
        episode
        for episode in episodes
        if relation_count == 1
        or len(episode.atoms)
        >= TWO_RELATION_MINIMUM_EPISODE_TOKENS
    )
    if not eligible:
        raise ClosedChoiceV2Error(
            "V2_PLAN_CANDIDATE_SET_INVALID"
        )
    episode_prompt = _COARSE_PROMPT.format(
        episode_json=_json_string(sentence_excerpt),
        role=f"{sentence_id}.relation_episode",
    )
    episode_candidates = tuple(
        _Candidate(
            key=f"{sentence_id}.episode.{episode.episode_id}",
            payload=episode,
            pair=PromptAnswer(
                candidate_key=(
                    f"{sentence_id}.episode.{episode.episode_id}"
                ),
                prompt=episode_prompt,
                answer=(
                    f"episode_id={episode.episode_id}; "
                    f"bounded_text={_json_string(episode.text)}"
                ),
            ),
        )
        for episode in eligible
    )
    episode_ranked = _rank(
        step=f"{sentence_id}.relation_episode",
        candidates=episode_candidates,
        backend=backend,
    )
    selected_episode = episode_ranked.selected.payload
    if not isinstance(selected_episode, Episode):
        raise ClosedChoiceV2Error(
            "V2_PLAN_CANDIDATE_SET_INVALID"
        )
    return (
        selected_episode,
        relation_count,
        (plan_ranked, episode_ranked),
    )


def _build_wire(
    relations: Sequence[_RelationSelection],
) -> str:
    by_episode: dict[str, list[_RelationSelection]] = {}
    sentence_ids: dict[str, str] = {}
    for relation in relations:
        by_episode.setdefault(relation.episode_id, []).append(
            relation
        )
        sentence_ids[relation.episode_id] = relation.sentence_id
    return canonical_json_bytes(
        {
            "episodes": [
                {
                    "episode_id": episode_id,
                    "relations": [
                        {
                            "anchor_span_id": row.anchor.span_id,
                            "causal_orientation": (
                                row.causal_orientation
                            ),
                            "generator_kind": row.generator_kind,
                            "object0_span_id": row.object0.span_id,
                            "object1_span_id": row.object1.span_id,
                            "polarity": row.polarity,
                            "relation_id": row.relation_id,
                            "temporal_orientation": (
                                row.temporal_orientation
                            ),
                        }
                        for row in rows
                    ],
                    "sentence_id": sentence_ids[episode_id],
                }
                for episode_id, rows in by_episode.items()
            ],
            "schema_version": HIERARCHICAL_WIRE_SCHEMA,
        },
        newline=False,
    ).decode("ascii")


def _json_node_count(value: object) -> int:
    count = 1
    if isinstance(value, list):
        count += sum(_json_node_count(row) for row in value)
    elif isinstance(value, dict):
        count += sum(
            1 + _json_node_count(row)
            for row in value.values()
        )
    return count


def _exact_dict(
    value: object, keys: set[str]
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ClosedChoiceV2Error("V2_WIRE_FIELDS_INVALID")
    return dict(value)


def validate_hierarchical_wire(
    story_text: str,
    wire_completion: str,
    *,
    episodes: Sequence[Episode],
    expected_selections: Sequence[_RelationSelection],
    narrative_parser: NarrativeParser,
) -> tuple[str, object]:
    """Parse the exact v2 wire and derive the sole canonical parser input."""

    if (
        not isinstance(wire_completion, str)
        or not wire_completion
        or len(wire_completion.encode("utf-8", errors="strict"))
        > MAXIMUM_WIRE_BYTES
        or not callable(narrative_parser)
    ):
        raise ClosedChoiceV2Error("V2_WIRE_FIELDS_INVALID")
    try:
        value = json.loads(
            wire_completion,
            parse_float=lambda _: (_ for _ in ()).throw(
                ClosedChoiceV2Error("V2_WIRE_FIELDS_INVALID")
            ),
            parse_constant=lambda _: (_ for _ in ()).throw(
                ClosedChoiceV2Error("V2_WIRE_FIELDS_INVALID")
            ),
        )
    except ClosedChoiceV2Error:
        raise
    except Exception as exc:
        raise ClosedChoiceV2Error(
            "V2_WIRE_FIELDS_INVALID"
        ) from exc
    if (
        _json_node_count(value) > MAXIMUM_WIRE_JSON_NODES
        or canonical_json_bytes(value, newline=False).decode(
            "ascii"
        )
        != wire_completion
    ):
        raise ClosedChoiceV2Error("V2_WIRE_FIELDS_INVALID")
    root = _exact_dict(
        value,
        {
            "episodes",
            "schema_version",
        },
    )
    if root["schema_version"] != HIERARCHICAL_WIRE_SCHEMA:
        raise ClosedChoiceV2Error("V2_WIRE_FIELDS_INVALID")
    raw_episodes = root["episodes"]
    expected_sentence_ids = {
        episode.sentence_id for episode in episodes
    }
    if (
        type(raw_episodes) is not list
        or len(raw_episodes) != len(expected_sentence_ids)
    ):
        raise ClosedChoiceV2Error(
            "V2_WIRE_SENTENCE_COVERAGE_INVALID"
        )

    episode_by_id = {
        episode.episode_id: episode for episode in episodes
    }
    if (
        not expected_selections
        or len(expected_selections) > MAXIMUM_RELATION_UNITS
        or any(
            not isinstance(row, _RelationSelection)
            for row in expected_selections
        )
        or len(
            {row.relation_id for row in expected_selections}
        )
        != len(expected_selections)
    ):
        raise ClosedChoiceV2Error(
            "V2_WIRE_ENDPOINT_SELECTION_MISSING"
        )
    expected_by_relation = {
        row.relation_id: row for row in expected_selections
    }
    expected_episode_for_sentence: dict[str, str] = {}
    for row in expected_selections:
        previous = expected_episode_for_sentence.setdefault(
            row.sentence_id, row.episode_id
        )
        if (
            previous != row.episode_id
            or row.episode_id not in episode_by_id
            or episode_by_id[row.episode_id].sentence_id
            != row.sentence_id
        ):
            raise ClosedChoiceV2Error(
                "V2_WIRE_ENDPOINT_SELECTION_MISSING"
            )
    if set(expected_episode_for_sentence) != expected_sentence_ids:
        raise ClosedChoiceV2Error(
            "V2_WIRE_SENTENCE_COVERAGE_INVALID"
        )

    parsed_relations: list[_RelationSelection] = []
    relation_ids: set[str] = set()
    seen_episode_ids: set[str] = set()
    seen_sentence_ids: set[str] = set()
    selected_spans: list[GroundedSpan] = []
    previous_episode_index = -1
    for raw_episode in raw_episodes:
        episode_row = _exact_dict(
            raw_episode,
            {"episode_id", "relations", "sentence_id"},
        )
        episode_id = episode_row["episode_id"]
        if (
            not isinstance(episode_id, str)
            or episode_id not in episode_by_id
            or episode_id in seen_episode_ids
        ):
            raise ClosedChoiceV2Error(
                "V2_WIRE_REFERENCE_INVALID"
            )
        expected_episode = episode_by_id[episode_id]
        if (
            episode_row["sentence_id"]
            != expected_episode.sentence_id
            or expected_episode.sentence_id in seen_sentence_ids
            or expected_episode_for_sentence.get(
                expected_episode.sentence_id
            )
            != episode_id
        ):
            raise ClosedChoiceV2Error(
                "V2_WIRE_SENTENCE_COVERAGE_INVALID"
            )
        episode_index = int(expected_episode.episode_id[1:])
        if episode_index <= previous_episode_index:
            raise ClosedChoiceV2Error("V2_WIRE_ORDER_INVALID")
        previous_episode_index = episode_index
        seen_episode_ids.add(episode_id)
        seen_sentence_ids.add(expected_episode.sentence_id)
        raw_relations = episode_row["relations"]
        if (
            type(raw_relations) is not list
            or not 1 <= len(raw_relations)
            <= MAXIMUM_RELATIONS_PER_EPISODE
        ):
            raise ClosedChoiceV2Error(
                "V2_WIRE_COVERAGE_INVALID"
            )
        for raw_relation in raw_relations:
            row = _exact_dict(
                raw_relation,
                {
                    "anchor_span_id",
                    "causal_orientation",
                    "generator_kind",
                    "object0_span_id",
                    "object1_span_id",
                    "polarity",
                    "relation_id",
                    "temporal_orientation",
                },
            )
            relation_id = row["relation_id"]
            anchor_id = row["anchor_span_id"]
            object0_id = row["object0_span_id"]
            object1_id = row["object1_span_id"]
            if (
                not isinstance(relation_id, str)
                or re.fullmatch(r"r[0-9]{2}", relation_id)
                is None
                or relation_id in relation_ids
            ):
                raise ClosedChoiceV2Error(
                    "V2_WIRE_REFERENCE_INVALID"
                )
            relation_ids.add(relation_id)
            if (
                row["generator_kind"] not in GENERATOR_KINDS
                or row["polarity"] not in POLARITIES
                or row["temporal_orientation"]
                not in ORIENTATIONS
                or row["causal_orientation"] not in ORIENTATIONS
            ):
                raise ClosedChoiceV2Error(
                    "V2_WIRE_REFERENCE_INVALID"
                )
            anchor = _resolve_span_reference(
                story_text=story_text,
                episode=expected_episode,
                span_id=anchor_id,
            )
            object0 = _resolve_span_reference(
                story_text=story_text,
                episode=expected_episode,
                span_id=object0_id,
            )
            object1 = _resolve_span_reference(
                story_text=story_text,
                episode=expected_episode,
                span_id=object1_id,
            )
            relation_spans = (anchor, object0, object1)
            if len({row.span_id for row in relation_spans}) != 3 or any(
                _intervals_overlap(left, right)
                for left, right in itertools.combinations(
                    relation_spans, 2
                )
            ):
                raise ClosedChoiceV2Error(
                    "V2_WIRE_ENDPOINT_OVERLAP"
                )
            parsed = _RelationSelection(
                relation_id=relation_id,
                episode_id=expected_episode.episode_id,
                sentence_id=expected_episode.sentence_id,
                anchor=anchor,
                object0=object0,
                object1=object1,
                generator_kind=str(row["generator_kind"]),
                polarity=str(row["polarity"]),
                temporal_orientation=str(
                    row["temporal_orientation"]
                ),
                causal_orientation=str(
                    row["causal_orientation"]
                ),
            )
            expected = expected_by_relation.get(relation_id)
            if expected is None or parsed != expected:
                raise ClosedChoiceV2Error(
                    "V2_WIRE_ENDPOINT_SELECTION_MISSING"
                )
            selected_spans.extend(relation_spans)
            parsed_relations.append(parsed)
    if seen_sentence_ids != expected_sentence_ids:
        raise ClosedChoiceV2Error(
            "V2_WIRE_SENTENCE_COVERAGE_INVALID"
        )
    if relation_ids != {
        f"r{index:02d}" for index in range(len(parsed_relations))
    } or relation_ids != set(expected_by_relation):
        raise ClosedChoiceV2Error("V2_WIRE_COVERAGE_INVALID")
    expected_order = sorted(
        parsed_relations,
        key=lambda row: (
            row.anchor.start,
            row.anchor.end,
            row.relation_id,
        ),
    )
    if parsed_relations != expected_order:
        raise ClosedChoiceV2Error("V2_WIRE_ORDER_INVALID")

    ordered_intervals = sorted(
        (span.start, span.end, span.span_id)
        for span in selected_spans
    )
    if len({row[2] for row in ordered_intervals}) != len(
        ordered_intervals
    ) or any(
        left[1] > right[0]
        for left, right in zip(
            ordered_intervals,
            ordered_intervals[1:],
            strict=False,
        )
    ):
        raise ClosedChoiceV2Error(
            "V2_WIRE_ENDPOINT_OVERLAP"
        )
    canonical = _canonical_completion(
        parsed_relations
    )
    try:
        extraction = narrative_parser(story_text, canonical)
    except ClosedChoiceV2Error:
        raise
    except Exception as exc:
        raise ClosedChoiceV2Error("V2_PARSER_REJECTED") from exc
    try:
        generators = tuple(extraction.generators)
        object_ids = tuple(
            extraction.hypergraph.object_mention_ids
        )
        slot_ids = tuple(
            slot
            for generator in generators
            for slot in generator.slot_mention_ids
        )
    except Exception as exc:
        raise ClosedChoiceV2Error(
            "V2_WIRE_CANONICAL_MISMATCH"
        ) from exc
    if (
        len(generators) != len(parsed_relations)
        or len(object_ids) != 2 * len(parsed_relations)
        or len(slot_ids) != len(object_ids)
        or len(set(slot_ids)) != len(slot_ids)
        or set(slot_ids) != set(object_ids)
        or any(
            len(generator.slot_mention_ids) != 2
            for generator in generators
        )
    ):
        raise ClosedChoiceV2Error(
            "V2_WIRE_OBJECT_OWNERSHIP_INVALID"
        )
    return canonical, extraction


def _canonical_completion(
    relations: Sequence[_RelationSelection],
) -> str:
    span_by_identifier: dict[str, GroundedSpan] = {}
    selected_span_ids: set[str] = set()
    mentions: list[dict[str, object]] = []
    generator_rows: list[dict[str, object]] = []
    for index, relation in enumerate(relations):
        anchor_id = f"a{index:03d}"
        object0_id = f"o{2 * index:03d}"
        object1_id = f"o{2 * index + 1:03d}"
        for identifier, kind, span in (
            (anchor_id, "generator", relation.anchor),
            (object0_id, "object", relation.object0),
            (object1_id, "object", relation.object1),
        ):
            if span.span_id in selected_span_ids:
                raise ClosedChoiceV2Error(
                    "V2_WIRE_OBJECT_OWNERSHIP_INVALID"
                )
            selected_span_ids.add(span.span_id)
            span_by_identifier[identifier] = span
            mentions.append(
                {
                    "kind": kind,
                    "mention_id": identifier,
                    "occurrence": span.occurrence,
                    "quote": span.quote,
                }
            )
        generator_rows.append(
            {
                "anchor_mention_id": anchor_id,
                "causal_orientation": relation.causal_orientation,
                "generator_id": f"g{index:03d}",
                "generator_kind": relation.generator_kind,
                "polarity": relation.polarity,
                "slot_mention_ids": [
                    object0_id,
                    object1_id,
                ],
                "temporal_orientation": (
                    relation.temporal_orientation
                ),
            }
        )
    mentions.sort(
        key=lambda row: (
            span_by_identifier[str(row["mention_id"])].start,
            span_by_identifier[str(row["mention_id"])].end,
            str(row["mention_id"]),
        )
    )
    return canonical_json_bytes(
        {
            "generators": generator_rows,
            "mentions": mentions,
            "schema_version": COMPLETION_SCHEMA,
        },
        newline=False,
    ).decode("ascii")


class _HierarchicalEngine:
    __slots__ = ("_marker",)

    def __init__(self, marker: object) -> None:
        if marker is not _ENGINE_MARKER:
            raise ClosedChoiceV2Error("V2_AUTHORITY_INVALID")
        self._marker = marker

    def select(
        self,
        story_text: str,
        *,
        backend: QualificationScoreBackendV2,
        narrative_parser: NarrativeParser,
    ) -> ClosedChoiceV2Decision:
        if (
            type(self) is not _HierarchicalEngine
            or self._marker is not _ENGINE_MARKER
            or not callable(narrative_parser)
        ):
            raise ClosedChoiceV2Error("V2_AUTHORITY_INVALID")
        try:
            runtime_commitment = backend.runtime_commitment
        except Exception as exc:
            raise ClosedChoiceV2Error(
                "V2_AUTHORITY_INVALID"
            ) from exc
        if (
            not isinstance(runtime_commitment, str)
            or re.fullmatch(
                r"[0-9a-f]{64}", runtime_commitment
            ) is None
        ):
            raise ClosedChoiceV2Error("V2_AUTHORITY_INVALID")
        episodes = build_hierarchical_episodes(story_text)
        relations: list[_RelationSelection] = []
        step_receipts: list[dict[str, object]] = []
        selected_answer_tokens = 0
        candidate_count = 0
        forward_batch_count = 0
        endpoint_receipt_commitments: dict[
            str, dict[str, str]
        ] = {}
        episodes_by_sentence: dict[str, list[Episode]] = {}
        for episode in episodes:
            episodes_by_sentence.setdefault(
                episode.sentence_id, []
            ).append(episode)
        selected_plans: list[tuple[Episode, int]] = []
        for sentence_id, sentence_episodes in (
            episodes_by_sentence.items()
        ):
            selected_episode, planned_count, rankings = (
                _select_sentence_plan(
                    sentence_id=sentence_id,
                    episodes=tuple(sentence_episodes),
                    backend=backend,
                )
            )
            selected_plans.append(
                (selected_episode, planned_count)
            )
            for ranked in rankings:
                step_receipts.append(dict(ranked.receipt))
                selected_answer_tokens += (
                    ranked.score.answer_token_count
                )
                candidate_count += ranked.candidate_count
                forward_batch_count += ranked.forward_batch_count
        if sum(count for _, count in selected_plans) > (
            MAXIMUM_RELATION_UNITS
        ):
            raise ClosedChoiceV2Abstention(
                "V2_PLAN_RELATION_CAPACITY_EXCEEDED",
                before_model_forward=False,
            )

        relation_index = 0
        for episode, planned_count in selected_plans:
            selected_spans: list[GroundedSpan] = []
            total_endpoint_count = 3 * planned_count
            selected_endpoint_count = 0
            for _ in range(planned_count):
                relation_id = f"r{relation_index:02d}"
                anchor, anchor_rankings = _select_span(
                    story_text=story_text,
                    episode=episode,
                    role=f"{relation_id}.anchor",
                    fixed=tuple(selected_spans),
                    reserve_atomic_tokens=(
                        total_endpoint_count
                        - selected_endpoint_count
                        - 1
                    ),
                    backend=backend,
                )
                selected_spans.append(anchor)
                selected_endpoint_count += 1
                object0, object0_rankings = _select_span(
                    story_text=story_text,
                    episode=episode,
                    role=f"{relation_id}.object0",
                    fixed=tuple(selected_spans),
                    reserve_atomic_tokens=(
                        total_endpoint_count
                        - selected_endpoint_count
                        - 1
                    ),
                    backend=backend,
                )
                selected_spans.append(object0)
                selected_endpoint_count += 1
                object1, object1_rankings = _select_span(
                    story_text=story_text,
                    episode=episode,
                    role=f"{relation_id}.object1",
                    fixed=tuple(selected_spans),
                    reserve_atomic_tokens=(
                        total_endpoint_count
                        - selected_endpoint_count
                        - 1
                    ),
                    backend=backend,
                )
                selected_spans.append(object1)
                selected_endpoint_count += 1
                fixed = (anchor, object0, object1)
                kind, kind_ranked = _select_enum(
                    episode=episode,
                    relation_id=relation_id,
                    attribute="generator_kind",
                    domain=GENERATOR_KINDS,
                    fixed=fixed,
                    backend=backend,
                )
                polarity, polarity_ranked = _select_enum(
                    episode=episode,
                    relation_id=relation_id,
                    attribute="polarity",
                    domain=POLARITIES,
                    fixed=fixed,
                    backend=backend,
                )
                temporal, temporal_ranked = _select_enum(
                    episode=episode,
                    relation_id=relation_id,
                    attribute="temporal_orientation",
                    domain=ORIENTATIONS,
                    fixed=fixed,
                    backend=backend,
                )
                causal, causal_ranked = _select_enum(
                    episode=episode,
                    relation_id=relation_id,
                    attribute="causal_orientation",
                    domain=ORIENTATIONS,
                    fixed=fixed,
                    backend=backend,
                )
                ranked_steps = (
                    *anchor_rankings,
                    *object0_rankings,
                    *object1_rankings,
                    kind_ranked,
                    polarity_ranked,
                    temporal_ranked,
                    causal_ranked,
                )
                endpoint_receipt_commitments[relation_id] = {
                    "anchor": semantic_sha256(
                        [dict(row.receipt) for row in anchor_rankings]
                    ),
                    "object0": semantic_sha256(
                        [dict(row.receipt) for row in object0_rankings]
                    ),
                    "object1": semantic_sha256(
                        [dict(row.receipt) for row in object1_rankings]
                    ),
                }
                for ranked in ranked_steps:
                    step_receipts.append(dict(ranked.receipt))
                    selected_answer_tokens += (
                        ranked.score.answer_token_count
                    )
                    candidate_count += ranked.candidate_count
                    forward_batch_count += (
                        ranked.forward_batch_count
                    )
                relations.append(
                    _RelationSelection(
                        relation_id=relation_id,
                        episode_id=episode.episode_id,
                        sentence_id=episode.sentence_id,
                        anchor=anchor,
                        object0=object0,
                        object1=object1,
                        generator_kind=kind,
                        polarity=polarity,
                        temporal_orientation=temporal,
                        causal_orientation=causal,
                    )
                )
                relation_index += 1

        relations.sort(
            key=lambda row: (
                row.anchor.start,
                row.anchor.end,
                row.relation_id,
            )
        )
        if (
            not 1 <= len(relations) <= MAXIMUM_RELATION_UNITS
            or candidate_count > MAXIMUM_TOTAL_CANDIDATES
            or forward_batch_count > MAXIMUM_FORWARD_BATCH_CALLS
        ):
            raise ClosedChoiceV2Error("V2_VERIFIER_REJECTED")
        wire = _build_wire(relations)
        canonical, extraction = validate_hierarchical_wire(
            story_text,
            wire,
            episodes=episodes,
            expected_selections=relations,
            narrative_parser=narrative_parser,
        )
        try:
            wire_token_count = (
                backend.count_program_owned_completion_tokens(wire)
            )
        except ClosedChoiceV2Error:
            raise
        except Exception as exc:
            raise ClosedChoiceV2Error(
                "V2_TOKEN_BOUNDARY_INVALID"
            ) from exc
        if (
            isinstance(wire_token_count, bool)
            or not isinstance(wire_token_count, int)
            or not 1 <= wire_token_count
            < MAXIMUM_WIRE_COMPLETION_TOKENS
        ):
            raise ClosedChoiceV2Error(
                "V2_TOKEN_BOUNDARY_INVALID"
            )
        catalog_commitment = semantic_sha256(
            [
                {
                    "episode_id": episode.episode_id,
                    "sentence_id": episode.sentence_id,
                    "spans": [
                        {
                            "occurrence": atom.occurrence,
                            "quote": atom.quote,
                            "span_id": atom.span_id,
                        }
                        for atom in episode.atoms
                    ],
                }
                for episode in episodes
            ]
        )
        body: dict[str, object] = {
            "canonical_completion_commitment": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
            "catalog_commitment": catalog_commitment,
            "claim_scope": CLAIM_SCOPE,
            "consumer_binding": {
                "flat_label_no_verifier": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
                "full": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
                "legacy_keyword": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
                "semantic_only": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
            },
            "endpoint_selection_receipt_commitments": (
                endpoint_receipt_commitments
            ),
            "exclusive_endpoint_ownership": True,
            "free_form_generation_count": 0,
            "model_runtime_commitment": (
                runtime_commitment
            ),
            "prompt_closure_sha256": PROMPT_CLOSURE_SHA256,
            "resource_summary": {
                "candidate_count": candidate_count,
                "episode_count": len(episodes),
                "forward_batch_count": forward_batch_count,
                "maximum_candidates_in_one_batch": (
                    SCORING_BATCH_SIZE
                ),
                "maximum_span_lexical_width": (
                    MAXIMUM_SPAN_LEXICAL_WIDTH
                ),
                "relation_count": len(relations),
                "sentence_count": len(
                    {row.sentence_id for row in episodes}
                ),
            },
            "schema": RECEIPT_SCHEMA,
            "selected_answer_token_count": (
                selected_answer_tokens
            ),
            "steps_commitment": semantic_sha256(step_receipts),
            "story_commitment": hashlib.sha256(
                story_text.encode("utf-8")
            ).hexdigest(),
            "slot_binding_semantics": (
                "slot0_and_slot1_are_independently_model_ranked_distinct_"
                "grounded_endpoints; each_endpoint_is_owned_by_exactly_one_"
                "generator; object_degree_equals_one; shared_slot_bonus_is_"
                "structurally_zero"
            ),
            "version": VERSION,
            "wire_commitment": hashlib.sha256(
                wire.encode("ascii")
            ).hexdigest(),
        }
        receipt = {
            **body,
            "self_sha256": semantic_sha256(body),
        }
        return ClosedChoiceV2Decision(
            wire_completion=wire,
            canonical_completion=canonical,
            extraction=extraction,
            selected_answer_token_count=(
                selected_answer_tokens
            ),
            wire_completion_token_count=wire_token_count,
            receipt_bytes=canonical_json_bytes(receipt),
        )


def select_hierarchical_qualification_only(
    story_text: str,
    *,
    backend: QualificationScoreBackendV2,
    narrative_parser: NarrativeParser,
) -> ClosedChoiceV2Decision:
    """Run the fake/source-free qualification surface only."""

    return _HierarchicalEngine(_ENGINE_MARKER).select(
        story_text,
        backend=backend,
        narrative_parser=narrative_parser,
    )


__all__ = [
    "CLAIM_SCOPE",
    "ClosedChoiceV2Decision",
    "Episode",
    "GENERATOR_KINDS",
    "GroundedAtom",
    "GroundedSpan",
    "LEAF_GROUP_CAPACITY",
    "MAXIMUM_CANDIDATES_PER_RELATION",
    "MAXIMUM_DOCUMENT_LEXICAL_TOKENS",
    "MAXIMUM_EPISODE_TOKENS",
    "MAXIMUM_EPISODES",
    "MAXIMUM_FORWARD_BATCH_CALLS",
    "MAXIMUM_RELATION_UNITS",
    "MAXIMUM_SENTENCES",
    "MAXIMUM_SPAN_LEXICAL_WIDTH",
    "MAXIMUM_TOTAL_CANDIDATES",
    "MAXIMUM_WIRE_COMPLETION_TOKENS",
    "ORIENTATIONS",
    "POLARITIES",
    "PROMPT_CLOSURE_SHA256",
    "QualificationScoreBackendV2",
    "RECEIPT_SCHEMA",
    "SCORING_BATCH_SIZE",
    "SCORING_POLICY",
    "MINIMUM_DOCUMENT_LEXICAL_TOKENS",
    "VERSION",
    "build_hierarchical_episodes",
    "select_hierarchical_qualification_only",
    "validate_hierarchical_wire",
]

"""Source-agnostic typed turn retrieval and evaluator core for LoCoMo P1.

The action former accepts only a question and public fields from turns in the
same conversation.  It has no loader, file, network, model, benchmark-family,
conversation-identity, answer, or relevance-label entrypoint.  Exact offline
utilities may enter only through :class:`AFormExample`, after all six action
slates and their structural signatures have been sealed.

RAW is ordinary local BM25 over the complete, unchanged question and turn
texts.  Five label-free typed recipes share that same BM25 coordinate and add
only deterministic structure derived from speaker, session ordinal, date,
turn ordinal, and text.  E0 is the frozen typed cascade.  E1 is a conservative
pairwise signature table fitted from A_form only; an unseen, tied, or
contradicted signature falls back to E0.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import json
import math
import re
import unicodedata
from typing import Mapping, Sequence


VERSION = "locomo_p1_typed_core_v1"
STUDY_ID = "LOCOMO_P1_TYPED_TURN_RETRIEVAL_V1"
TOP_K = 5
SCALE = 1_000_000
BM25_K1 = 1.2
BM25_B = 0.75
MIN_SIGNATURE_SUPPORT = 3

R0_RAW_BM25 = "R0_RAW_BM25"
R1_ENTITY_FOCUS = "R1_ENTITY_FOCUS"
R2_TEMPORAL_FOCUS = "R2_TEMPORAL_FOCUS"
R3_SPEAKER_EVENT = "R3_SPEAKER_EVENT"
R4_MULTI_SEED_MARGINAL = "R4_MULTI_SEED_MARGINAL"
R5_TYPED_CASCADE = "R5_TYPED_CASCADE"
RECIPE_IDS = (
    R0_RAW_BM25,
    R1_ENTITY_FOCUS,
    R2_TEMPORAL_FOCUS,
    R3_SPEAKER_EVENT,
    R4_MULTI_SEED_MARGINAL,
    R5_TYPED_CASCADE,
)
E0_RECIPE_ID = R5_TYPED_CASCADE
POLICY_STAGES = ("F_search", "A_hold", "M_search")
PUBLIC_TURN_FIELDS = (
    "ordinal",
    "session_ordinal",
    "speaker",
    "date",
    "text",
)

FEATURE_NAMES = (
    "raw_overlap_count",
    "entity_turn_count",
    "temporal_turn_count",
    "speaker_turn_count",
    "event_turn_count",
    "adjacent_pair_count",
    "session_coverage_count",
    "speaker_coverage_count",
    "mean_raw_bm25_micros",
    "minimum_raw_bm25_micros",
    "lexical_diversity_micros",
    "ordinal_span_micros",
)
SIGNATURE_FEATURE_NAMES = FEATURE_NAMES

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}\Z")
_DATE_NUMBER_RE = re.compile(r"\d{1,4}\Z")
_MONTHS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }
)
_TEMPORAL_CUES = frozenset(
    {
        "after",
        "before",
        "during",
        "earlier",
        "earliest",
        "first",
        "last",
        "later",
        "latest",
        "next",
        "previous",
        "recent",
        "recently",
        "then",
        "today",
        "tomorrow",
        "when",
        "while",
        "yesterday",
    }
)
_LATE_CUES = frozenset({"after", "last", "later", "latest", "next", "recent", "recently"})
_EARLY_CUES = frozenset({"before", "earlier", "earliest", "first", "previous"})
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
    }
)


class LocomoP1TypedCoreError(ValueError):
    """A public projection, frozen action, or A_form evaluator drifted."""


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LocomoP1TypedCoreError("value is not canonical JSON") from exc
    return encoded + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def normalize_text(value: str, *, field: str = "text") -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise LocomoP1TypedCoreError(f"{field} is not valid text")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized or len(normalized) > 20_000:
        raise LocomoP1TypedCoreError(f"{field} is empty or too long")
    return normalized


def lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(value)))


@dataclass(frozen=True, slots=True)
class Turn:
    ordinal: int
    session_ordinal: int
    speaker: str
    date: str
    text: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise LocomoP1TypedCoreError("turn ordinal is invalid")
        if type(self.session_ordinal) is not int or self.session_ordinal < 0:
            raise LocomoP1TypedCoreError("session ordinal is invalid")
        if not isinstance(self.speaker, str) or len(self.speaker) > 256:
            raise LocomoP1TypedCoreError("speaker is too long")
        if not isinstance(self.date, str) or len(self.date) > 256:
            raise LocomoP1TypedCoreError("date is too long")
        if not isinstance(self.text, str):
            raise LocomoP1TypedCoreError("turn text is not a string")
        normalize_text(self.speaker, field="speaker")
        normalize_text(self.date, field="date")
        normalize_text(self.text, field="turn text")


def turn_from_public_fields(value: object) -> Turn:
    if not isinstance(value, Mapping) or set(value) != set(PUBLIC_TURN_FIELDS):
        raise LocomoP1TypedCoreError("turn projection is not the exact public field set")
    try:
        return Turn(
            ordinal=value["ordinal"],  # type: ignore[arg-type]
            session_ordinal=value["session_ordinal"],  # type: ignore[arg-type]
            speaker=value["speaker"],  # type: ignore[arg-type]
            date=value["date"],  # type: ignore[arg-type]
            text=value["text"],  # type: ignore[arg-type]
        )
    except KeyError as exc:
        raise LocomoP1TypedCoreError("turn projection is incomplete") from exc


def turn_public_payload(turn: Turn) -> dict[str, object]:
    if not isinstance(turn, Turn):
        raise LocomoP1TypedCoreError("turn is not a public Turn")
    return {field.name: getattr(turn, field.name) for field in fields(Turn)}


def _checked_turns(turns: Sequence[Turn]) -> tuple[Turn, ...]:
    if isinstance(turns, (str, bytes)) or len(turns) < TOP_K:
        raise LocomoP1TypedCoreError("conversation has fewer than five turns")
    checked = tuple(turns)
    if any(not isinstance(turn, Turn) for turn in checked):
        raise LocomoP1TypedCoreError("conversation contains a non-Turn value")
    if len({turn.ordinal for turn in checked}) != len(checked):
        raise LocomoP1TypedCoreError("turn ordinals are not unique")
    return tuple(sorted(checked, key=lambda turn: turn.ordinal))


def bm25_scores(question: str, document_texts: Sequence[str]) -> tuple[int, ...]:
    """Return quantized Okapi BM25 for the complete, unchanged question."""

    query_terms = lexical_tokens(question)
    if not query_terms:
        raise LocomoP1TypedCoreError("question tokenization is empty")
    if (
        isinstance(document_texts, (str, bytes))
        or len(document_texts) < TOP_K
        or any(not isinstance(text, str) for text in document_texts)
    ):
        raise LocomoP1TypedCoreError("BM25 corpus is malformed")
    documents = [lexical_tokens(text) for text in document_texts]
    if any(not terms for terms in documents):
        raise LocomoP1TypedCoreError("BM25 document tokenization is empty")
    document_count = len(documents)
    average_length = sum(map(len, documents)) / document_count
    document_frequency: Counter[str] = Counter()
    for terms in documents:
        document_frequency.update(set(terms))
    query_frequency = Counter(query_terms)
    scores: list[int] = []
    for terms in documents:
        term_frequency = Counter(terms)
        score = 0.0
        for term, query_count in query_frequency.items():
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * len(terms) / average_length
            )
            score += (
                query_count
                * inverse_document_frequency
                * frequency
                * (BM25_K1 + 1.0)
                / denominator
            )
        scores.append(int(round(score * SCALE)))
    return tuple(scores)


def _query_capitalized_tokens(question: str) -> frozenset[str]:
    raw_tokens = _TOKEN_RE.findall(unicodedata.normalize("NFKC", question))
    return frozenset(
        token.casefold()
        for token in raw_tokens
        if (
            any(character.isupper() for character in token)
            and token.casefold() not in _STOPWORDS
        )
        or any(character.isdigit() for character in token)
    )


def _temporal_tokens(value: str) -> frozenset[str]:
    tokens = tuple(re.findall(r"[^\W_]+", normalize_text(value), re.UNICODE))
    return frozenset(
        token
        for token in tokens
        if token in _MONTHS
        or token in _TEMPORAL_CUES
        or _YEAR_RE.fullmatch(token) is not None
        or (
            _DATE_NUMBER_RE.fullmatch(token) is not None
            and any(character.isdigit() for character in token)
        )
    )


@dataclass(frozen=True, slots=True)
class QueryStructure:
    entity_anchors: tuple[str, ...]
    temporal_anchors: tuple[str, ...]
    speaker_anchors: tuple[str, ...]
    event_anchors: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "entity_anchors",
            "temporal_anchors",
            "speaker_anchors",
            "event_anchors",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise LocomoP1TypedCoreError(f"{name} is not a sorted set")

    def payload(self) -> dict[str, object]:
        return {
            "entity_anchors": list(self.entity_anchors),
            "event_anchors": list(self.event_anchors),
            "speaker_anchors": list(self.speaker_anchors),
            "temporal_anchors": list(self.temporal_anchors),
        }


def query_structure(question: str, turns: Sequence[Turn]) -> QueryStructure:
    checked = _checked_turns(turns)
    question_tokens = lexical_tokens(question)
    question_set = set(question_tokens)
    speaker_vocabulary = {
        token
        for turn in checked
        for token in lexical_tokens(turn.speaker)
    }
    speaker_anchors = question_set & speaker_vocabulary
    temporal_anchors = set(_temporal_tokens(question))

    document_frequency: Counter[str] = Counter()
    for turn in checked:
        document_frequency.update(
            set(lexical_tokens(f"{turn.speaker} {turn.text}"))
        )
    rare_content = {
        token
        for token in question_set
        if len(token) >= 4
        and token not in _STOPWORDS
        and token not in temporal_anchors
        and document_frequency.get(token, 0) <= max(1, len(checked) // 3)
    }
    entity_anchors = (
        set(_query_capitalized_tokens(question))
        | rare_content
        | speaker_anchors
    ) - temporal_anchors
    event_anchors = {
        token
        for token in question_set
        if len(token) >= 3
        and token not in _STOPWORDS
        and token not in entity_anchors
        and token not in temporal_anchors
        and token not in speaker_anchors
    }
    return QueryStructure(
        entity_anchors=tuple(sorted(entity_anchors)),
        temporal_anchors=tuple(sorted(temporal_anchors)),
        speaker_anchors=tuple(sorted(speaker_anchors)),
        event_anchors=tuple(sorted(event_anchors)),
    )


@dataclass(frozen=True, slots=True)
class TurnStructure:
    turn_ordinal: int
    raw_bm25: int
    entity_hits: int
    temporal_hits: int
    speaker_hits: int
    event_hits: int

    def __post_init__(self) -> None:
        if type(self.turn_ordinal) is not int or self.turn_ordinal < 0:
            raise LocomoP1TypedCoreError("turn feature ordinal is invalid")
        if type(self.raw_bm25) is not int or self.raw_bm25 < 0:
            raise LocomoP1TypedCoreError("turn BM25 coordinate is invalid")
        for name in ("entity_hits", "temporal_hits", "speaker_hits", "event_hits"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise LocomoP1TypedCoreError(f"{name} is invalid")

    def payload(self) -> dict[str, int]:
        return {
            "entity_hits": self.entity_hits,
            "event_hits": self.event_hits,
            "raw_bm25": self.raw_bm25,
            "speaker_hits": self.speaker_hits,
            "temporal_hits": self.temporal_hits,
            "turn_ordinal": self.turn_ordinal,
        }


def _turn_structures(
    question: str,
    turns: Sequence[Turn],
    structure: QueryStructure,
) -> tuple[TurnStructure, ...]:
    raw = bm25_scores(question, [turn.text for turn in turns])
    entity = set(structure.entity_anchors)
    temporal = set(structure.temporal_anchors)
    speaker = set(structure.speaker_anchors)
    event = set(structure.event_anchors)
    rows: list[TurnStructure] = []
    for index, turn in enumerate(turns):
        text_tokens = set(lexical_tokens(turn.text))
        speaker_tokens = set(lexical_tokens(turn.speaker))
        entity_tokens = text_tokens | speaker_tokens
        temporal_tokens = set(_temporal_tokens(f"{turn.date} {turn.text}"))
        rows.append(
            TurnStructure(
                turn_ordinal=turn.ordinal,
                raw_bm25=raw[index],
                entity_hits=len(entity & entity_tokens),
                temporal_hits=len(temporal & temporal_tokens),
                speaker_hits=len(speaker & speaker_tokens),
                event_hits=len(event & text_tokens),
            )
        )
    return tuple(rows)


def _rank(scores: Sequence[int], turns: Sequence[Turn]) -> tuple[int, ...]:
    if len(scores) != len(turns):
        raise LocomoP1TypedCoreError("score vector width drifted")
    return tuple(
        sorted(
            range(len(turns)),
            key=lambda index: (-scores[index], turns[index].ordinal),
        )
    )


def _directional_temporal_bonus(
    question: str, turns: Sequence[Turn]
) -> tuple[int, ...]:
    question_terms = set(lexical_tokens(question))
    maximum_session = max(turn.session_ordinal for turn in turns) or 1
    if question_terms & _LATE_CUES:
        return tuple(
            turn.session_ordinal * 200_000 // maximum_session for turn in turns
        )
    if question_terms & _EARLY_CUES:
        return tuple(
            (maximum_session - turn.session_ordinal) * 200_000 // maximum_session
            for turn in turns
        )
    return (0,) * len(turns)


def _jaccard_distance(left: set[str], right: set[str]) -> int:
    union = left | right
    if not union:
        return 0
    return len(union - (left & right)) * SCALE // len(union)


def _multi_seed_selection(
    *,
    turns: Sequence[Turn],
    raw_scores: Sequence[int],
    entity_scores: Sequence[int],
    temporal_scores: Sequence[int],
    speaker_event_scores: Sequence[int],
    structures: Sequence[TurnStructure],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected: list[int] = []
    selected_scores: list[int] = []
    for scores in (
        entity_scores,
        temporal_scores,
        speaker_event_scores,
        raw_scores,
    ):
        candidate = _rank(scores, turns)[0]
        if candidate not in selected:
            selected.append(candidate)
            selected_scores.append(scores[candidate])
        if len(selected) == TOP_K:
            return tuple(selected), tuple(selected_scores)

    token_sets = [set(lexical_tokens(turn.text)) for turn in turns]
    while len(selected) < TOP_K:
        used_speakers = {normalize_text(turns[index].speaker) for index in selected}
        used_sessions = {turns[index].session_ordinal for index in selected}
        candidates: list[tuple[int, int, int]] = []
        for index, turn in enumerate(turns):
            if index in selected:
                continue
            base = max(
                entity_scores[index],
                temporal_scores[index],
                speaker_event_scores[index],
            )
            adjacency = any(
                turns[other].session_ordinal == turn.session_ordinal
                and abs(turns[other].ordinal - turn.ordinal) == 1
                for other in selected
            )
            novelty = min(
                _jaccard_distance(token_sets[index], token_sets[other])
                for other in selected
            )
            marginal = (
                base
                + raw_scores[index]
                + (700_000 if adjacency else 0)
                + (
                    250_000
                    if normalize_text(turn.speaker) not in used_speakers
                    else 0
                )
                + (150_000 if turn.session_ordinal not in used_sessions else 0)
                + novelty // 4
                + 100_000
                * (
                    structures[index].entity_hits
                    + structures[index].temporal_hits
                    + structures[index].speaker_hits
                )
            )
            candidates.append((-marginal, turn.ordinal, index))
        negative, _ordinal, chosen = min(candidates)
        selected.append(chosen)
        selected_scores.append(-negative)
    return tuple(selected), tuple(selected_scores)


def _typed_cascade_selection(
    *,
    turns: Sequence[Turn],
    composite_scores: Sequence[int],
    entity_scores: Sequence[int],
    temporal_scores: Sequence[int],
    speaker_event_scores: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected: list[int] = []
    selected_scores: list[int] = []

    def add(index: int, score: int) -> None:
        if index not in selected and len(selected) < TOP_K:
            selected.append(index)
            selected_scores.append(score)

    seed = _rank(composite_scores, turns)[0]
    add(seed, composite_scores[seed])
    neighbors = sorted(
        (
            index
            for index, turn in enumerate(turns)
            if index != seed
            and turn.session_ordinal == turns[seed].session_ordinal
            and abs(turn.ordinal - turns[seed].ordinal) == 1
        ),
        key=lambda index: (
            abs(turns[index].ordinal - turns[seed].ordinal),
            -composite_scores[index],
            turns[index].ordinal,
        ),
    )
    for index in neighbors:
        add(index, composite_scores[index] + 800_000)
    for scores in (entity_scores, temporal_scores, speaker_event_scores):
        for index in _rank(scores, turns):
            if index not in selected:
                add(index, scores[index])
                break
    for index in _rank(composite_scores, turns):
        add(index, composite_scores[index])
    if len(selected) != TOP_K:
        raise LocomoP1TypedCoreError("typed cascade did not totalize top5")
    return tuple(selected), tuple(selected_scores)


@dataclass(frozen=True, slots=True)
class RecipeAction:
    recipe_id: str
    top5_turn_ordinals: tuple[int, ...]
    selected_scores: tuple[int, ...]
    raw_top5_turn_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise LocomoP1TypedCoreError("recipe is not frozen")
        if (
            len(self.top5_turn_ordinals) != TOP_K
            or len(set(self.top5_turn_ordinals)) != TOP_K
            or any(
                type(value) is not int or value < 0
                for value in self.top5_turn_ordinals
            )
        ):
            raise LocomoP1TypedCoreError("recipe top5 is malformed")
        if (
            len(self.selected_scores) != TOP_K
            or any(type(value) is not int or value < 0 for value in self.selected_scores)
        ):
            raise LocomoP1TypedCoreError("recipe scores are malformed")
        if (
            len(self.raw_top5_turn_ordinals) != TOP_K
            or len(set(self.raw_top5_turn_ordinals)) != TOP_K
        ):
            raise LocomoP1TypedCoreError("RAW top5 is malformed")

    def payload(self) -> dict[str, object]:
        return {
            "raw_top5_turn_ordinals": list(self.raw_top5_turn_ordinals),
            "recipe_id": self.recipe_id,
            "selected_scores": list(self.selected_scores),
            "top5_turn_ordinals": list(self.top5_turn_ordinals),
        }


@dataclass(frozen=True, slots=True)
class ActionFeatures:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(FEATURE_NAMES) or any(
            type(value) is not int or not 0 <= value <= SCALE
            for value in self.values
        ):
            raise LocomoP1TypedCoreError("action structural features drifted")

    def payload(self) -> dict[str, int]:
        return dict(zip(FEATURE_NAMES, self.values, strict=True))


def _action_features(
    *,
    action: RecipeAction,
    turns: Sequence[Turn],
    structures: Sequence[TurnStructure],
    raw_maximum: int,
) -> ActionFeatures:
    by_ordinal = {turn.ordinal: index for index, turn in enumerate(turns)}
    indices = tuple(by_ordinal[ordinal] for ordinal in action.top5_turn_ordinals)
    raw_values = [structures[index].raw_bm25 for index in indices]
    normalizer = raw_maximum or 1
    normalized_raw = [
        min(SCALE, value * SCALE // normalizer) for value in raw_values
    ]
    token_sets = [set(lexical_tokens(turns[index].text)) for index in indices]
    distances = [
        _jaccard_distance(token_sets[left], token_sets[right])
        for left in range(TOP_K)
        for right in range(left + 1, TOP_K)
    ]
    adjacent_pairs = sum(
        turns[indices[left]].session_ordinal
        == turns[indices[right]].session_ordinal
        and abs(
            turns[indices[left]].ordinal - turns[indices[right]].ordinal
        )
        == 1
        for left in range(TOP_K)
        for right in range(left + 1, TOP_K)
    )
    corpus_span = max(turn.ordinal for turn in turns) - min(
        turn.ordinal for turn in turns
    )
    selected_span = max(action.top5_turn_ordinals) - min(
        action.top5_turn_ordinals
    )
    values = (
        len(set(action.top5_turn_ordinals) & set(action.raw_top5_turn_ordinals)),
        sum(structures[index].entity_hits > 0 for index in indices),
        sum(structures[index].temporal_hits > 0 for index in indices),
        sum(structures[index].speaker_hits > 0 for index in indices),
        sum(structures[index].event_hits > 0 for index in indices),
        adjacent_pairs,
        len({turns[index].session_ordinal for index in indices}),
        len({normalize_text(turns[index].speaker) for index in indices}),
        sum(normalized_raw) // TOP_K,
        min(normalized_raw),
        sum(distances) // len(distances),
        selected_span * SCALE // (corpus_span or 1),
    )
    return ActionFeatures(values)


@dataclass(frozen=True, slots=True)
class ActionSlate:
    question_sha256: str
    public_turn_projection_sha256: str
    query: QueryStructure
    turn_structures: tuple[TurnStructure, ...]
    actions: tuple[RecipeAction, ...]
    features: tuple[ActionFeatures, ...]

    def __post_init__(self) -> None:
        if (
            not re.fullmatch(r"[0-9a-f]{64}", self.question_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", self.public_turn_projection_sha256)
            or tuple(action.recipe_id for action in self.actions) != RECIPE_IDS
            or len(self.features) != len(RECIPE_IDS)
            or len({row.turn_ordinal for row in self.turn_structures})
            != len(self.turn_structures)
        ):
            raise LocomoP1TypedCoreError("action slate drifted")

    def action(self, recipe_id: str) -> RecipeAction:
        if recipe_id not in RECIPE_IDS:
            raise LocomoP1TypedCoreError("requested recipe is not frozen")
        return self.actions[RECIPE_IDS.index(recipe_id)]

    def feature_row(self, recipe_id: str) -> ActionFeatures:
        if recipe_id not in RECIPE_IDS:
            raise LocomoP1TypedCoreError("requested recipe is not frozen")
        return self.features[RECIPE_IDS.index(recipe_id)]

    def audit_payload(self) -> dict[str, object]:
        body = {
            "actions": [action.payload() for action in self.actions],
            "feature_names": list(FEATURE_NAMES),
            "features": [feature.payload() for feature in self.features],
            "public_turn_projection_sha256": self.public_turn_projection_sha256,
            "query_structure": self.query.payload(),
            "question_sha256": self.question_sha256,
            "recipe_ids": list(RECIPE_IDS),
            "schema": f"{VERSION}_action_slate",
            "turn_structures": [row.payload() for row in self.turn_structures],
        }
        body["self_sha256"] = stable_hash(body)
        return body


def build_action_slate(question: str, turns: Sequence[Turn]) -> ActionSlate:
    """Materialize all six label-free actions from one public conversation."""

    normalized_question = normalize_text(question, field="question")
    checked_turns = _checked_turns(turns)
    structure = query_structure(question, checked_turns)
    rows = _turn_structures(question, checked_turns, structure)
    raw_scores = tuple(row.raw_bm25 for row in rows)
    directional = _directional_temporal_bonus(question, checked_turns)
    entity_scores = tuple(
        row.raw_bm25
        + 8_000_000 * row.entity_hits
        + 4_000_000 * row.speaker_hits
        for row in rows
    )
    temporal_scores = tuple(
        row.raw_bm25
        + 8_000_000 * row.temporal_hits
        + directional[index]
        for index, row in enumerate(rows)
    )
    speaker_event_scores = tuple(
        row.raw_bm25
        + 8_000_000 * row.speaker_hits
        + 2_000_000 * row.event_hits
        + 1_000_000 * row.entity_hits
        for row in rows
    )
    composite_scores = tuple(
        row.raw_bm25
        + 6_000_000 * row.entity_hits
        + 6_000_000 * row.temporal_hits
        + 5_000_000 * row.speaker_hits
        + 1_500_000 * row.event_hits
        + directional[index]
        for index, row in enumerate(rows)
    )
    raw_indices = _rank(raw_scores, checked_turns)[:TOP_K]
    raw_ordinals = tuple(checked_turns[index].ordinal for index in raw_indices)

    indices_and_scores: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for recipe_id, scores in (
        (R0_RAW_BM25, raw_scores),
        (R1_ENTITY_FOCUS, entity_scores),
        (R2_TEMPORAL_FOCUS, temporal_scores),
        (R3_SPEAKER_EVENT, speaker_event_scores),
    ):
        indices = _rank(scores, checked_turns)[:TOP_K]
        indices_and_scores[recipe_id] = (
            indices,
            tuple(scores[index] for index in indices),
        )
    indices_and_scores[R4_MULTI_SEED_MARGINAL] = _multi_seed_selection(
        turns=checked_turns,
        raw_scores=raw_scores,
        entity_scores=entity_scores,
        temporal_scores=temporal_scores,
        speaker_event_scores=speaker_event_scores,
        structures=rows,
    )
    indices_and_scores[R5_TYPED_CASCADE] = _typed_cascade_selection(
        turns=checked_turns,
        composite_scores=composite_scores,
        entity_scores=entity_scores,
        temporal_scores=temporal_scores,
        speaker_event_scores=speaker_event_scores,
    )
    actions = tuple(
        RecipeAction(
            recipe_id=recipe_id,
            top5_turn_ordinals=tuple(
                checked_turns[index].ordinal
                for index in indices_and_scores[recipe_id][0]
            ),
            selected_scores=indices_and_scores[recipe_id][1],
            raw_top5_turn_ordinals=raw_ordinals,
        )
        for recipe_id in RECIPE_IDS
    )
    raw_maximum = max(raw_scores)
    action_features = tuple(
        _action_features(
            action=action,
            turns=checked_turns,
            structures=rows,
            raw_maximum=raw_maximum,
        )
        for action in actions
    )
    public_payload = [turn_public_payload(turn) for turn in checked_turns]
    return ActionSlate(
        question_sha256=hashlib.sha256(normalized_question.encode("utf-8")).hexdigest(),
        public_turn_projection_sha256=stable_hash(public_payload),
        query=structure,
        turn_structures=rows,
        actions=actions,
        features=action_features,
    )


def pairwise_signature(
    candidate: ActionFeatures, reference: ActionFeatures
) -> tuple[int, ...]:
    if not isinstance(candidate, ActionFeatures) or not isinstance(
        reference, ActionFeatures
    ):
        raise LocomoP1TypedCoreError("pairwise signature inputs drifted")
    return tuple(
        (left > right) - (left < right)
        for left, right in zip(candidate.values, reference.values, strict=True)
    )


@dataclass(frozen=True, slots=True)
class AFormExample:
    features: tuple[ActionFeatures, ...]
    utilities: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.features) != len(RECIPE_IDS) or len(self.utilities) != len(
            RECIPE_IDS
        ):
            raise LocomoP1TypedCoreError("A_form example width drifted")
        if any(not isinstance(value, Fraction) for value in self.utilities):
            raise LocomoP1TypedCoreError("A_form utility is not exact")
        if any(value < 0 or value > 1 for value in self.utilities):
            raise LocomoP1TypedCoreError("A_form utility is outside [0,1]")


def make_aform_example(
    slate: ActionSlate, utility_by_recipe: Mapping[str, Fraction]
) -> AFormExample:
    if not isinstance(slate, ActionSlate) or set(utility_by_recipe) != set(
        RECIPE_IDS
    ):
        raise LocomoP1TypedCoreError("A_form utility projection drifted")
    return AFormExample(
        features=slate.features,
        utilities=tuple(utility_by_recipe[recipe] for recipe in RECIPE_IDS),
    )


@dataclass(frozen=True, slots=True)
class SignatureRule:
    recipe_id: str
    signature: tuple[int, ...]
    support_count: int
    positive_count: int
    negative_count: int
    tie_count: int
    net_utility: Fraction
    minimum_delta: Fraction
    one_sided_reference_tail: Fraction
    qualified: bool

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS or self.recipe_id == E0_RECIPE_ID:
            raise LocomoP1TypedCoreError("signature rule recipe drifted")
        if (
            len(self.signature) != len(SIGNATURE_FEATURE_NAMES)
            or any(value not in (-1, 0, 1) for value in self.signature)
            or self.support_count <= 0
            or self.positive_count + self.negative_count + self.tie_count
            != self.support_count
            or self.minimum_delta < -1
            or self.minimum_delta > 1
        ):
            raise LocomoP1TypedCoreError("signature rule statistics drifted")
        expected = (
            self.support_count >= MIN_SIGNATURE_SUPPORT
            and self.positive_count == self.support_count
            and self.negative_count == 0
            and self.tie_count == 0
            and self.minimum_delta > 0
            and self.one_sided_reference_tail
            <= Fraction(1, 2**MIN_SIGNATURE_SUPPORT)
        )
        if self.qualified is not expected:
            raise LocomoP1TypedCoreError("signature qualification drifted")

    def payload(self) -> dict[str, object]:
        return {
            "minimum_delta": [
                self.minimum_delta.numerator,
                self.minimum_delta.denominator,
            ],
            "negative_count": self.negative_count,
            "net_utility": [
                self.net_utility.numerator,
                self.net_utility.denominator,
            ],
            "one_sided_reference_tail": [
                self.one_sided_reference_tail.numerator,
                self.one_sided_reference_tail.denominator,
            ],
            "positive_count": self.positive_count,
            "qualified": self.qualified,
            "recipe_id": self.recipe_id,
            "signature": list(self.signature),
            "support_count": self.support_count,
            "tie_count": self.tie_count,
        }


@dataclass(frozen=True, slots=True)
class E1Model:
    rules: tuple[SignatureRule, ...]
    training_item_count: int
    pair_observation_count: int
    training_stage: str = "A_form"
    reference_recipe_id: str = E0_RECIPE_ID

    def __post_init__(self) -> None:
        if (
            self.training_item_count <= 0
            or self.pair_observation_count
            != self.training_item_count * (len(RECIPE_IDS) - 1)
            or self.training_stage != "A_form"
            or self.reference_recipe_id != E0_RECIPE_ID
        ):
            raise LocomoP1TypedCoreError("E1 model identity drifted")
        keys = [(rule.recipe_id, rule.signature) for rule in self.rules]
        if keys != sorted(
            keys, key=lambda row: (RECIPE_IDS.index(row[0]), row[1])
        ) or len(set(keys)) != len(keys):
            raise LocomoP1TypedCoreError("E1 signature registry drifted")

    def payload(self) -> dict[str, object]:
        body = {
            "forbidden_model_inputs": [
                "benchmark_family",
                "conversation_identity",
                "split_identity_as_feature",
            ],
            "minimum_signature_support": MIN_SIGNATURE_SUPPORT,
            "pair_observation_count": self.pair_observation_count,
            "reference_recipe_id": self.reference_recipe_id,
            "rules": [rule.payload() for rule in self.rules],
            "schema": f"{VERSION}_E1_pairwise_signature_model",
            "signature_feature_names": list(SIGNATURE_FEATURE_NAMES),
            "study_id": STUDY_ID,
            "training_item_count": self.training_item_count,
            "training_stage": self.training_stage,
        }
        body["self_sha256"] = stable_hash(body)
        return body


def fit_e1(examples: Sequence[AFormExample]) -> E1Model:
    """Fit exact candidate-vs-E0 signature rules from sealed A_form only."""

    if isinstance(examples, (str, bytes)) or not examples:
        raise LocomoP1TypedCoreError("A_form examples are empty")
    checked = tuple(examples)
    if any(not isinstance(example, AFormExample) for example in checked):
        raise LocomoP1TypedCoreError("A_form contains an invalid example")
    reference_index = RECIPE_IDS.index(E0_RECIPE_ID)
    grouped: defaultdict[tuple[str, tuple[int, ...]], list[Fraction]] = defaultdict(
        list
    )
    for example in checked:
        reference_features = example.features[reference_index]
        reference_utility = example.utilities[reference_index]
        for recipe_index, recipe_id in enumerate(RECIPE_IDS):
            if recipe_id == E0_RECIPE_ID:
                continue
            signature = pairwise_signature(
                example.features[recipe_index], reference_features
            )
            grouped[(recipe_id, signature)].append(
                example.utilities[recipe_index] - reference_utility
            )
    rules: list[SignatureRule] = []
    for recipe_id, signature in sorted(
        grouped,
        key=lambda row: (RECIPE_IDS.index(row[0]), row[1]),
    ):
        deltas = grouped[(recipe_id, signature)]
        positive = sum(delta > 0 for delta in deltas)
        negative = sum(delta < 0 for delta in deltas)
        ties = len(deltas) - positive - negative
        nonzero_count = positive + negative
        tail = (
            Fraction(1, 1)
            if nonzero_count == 0
            else Fraction(
                sum(
                    math.comb(nonzero_count, successes)
                    for successes in range(positive, nonzero_count + 1)
                ),
                2**nonzero_count,
            )
        )
        minimum = min(deltas)
        qualified = (
            len(deltas) >= MIN_SIGNATURE_SUPPORT
            and positive == len(deltas)
            and negative == 0
            and ties == 0
            and minimum > 0
            and tail <= Fraction(1, 2**MIN_SIGNATURE_SUPPORT)
        )
        rules.append(
            SignatureRule(
                recipe_id=recipe_id,
                signature=signature,
                support_count=len(deltas),
                positive_count=positive,
                negative_count=negative,
                tie_count=ties,
                net_utility=sum(deltas, Fraction(0, 1)),
                minimum_delta=minimum,
                one_sided_reference_tail=tail,
                qualified=qualified,
            )
        )
    return E1Model(
        rules=tuple(rules),
        training_item_count=len(checked),
        pair_observation_count=len(checked) * (len(RECIPE_IDS) - 1),
    )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    stage: str
    evaluator_id: str
    selected_recipe_id: str
    e0_recipe_id: str
    top5_turn_ordinals: tuple[int, ...]
    fallback_to_e0: bool
    matched_signature: tuple[int, ...] | None
    conservative_minimum_delta: Fraction

    def __post_init__(self) -> None:
        if (
            self.stage not in POLICY_STAGES
            or self.evaluator_id not in {"E0", "E1"}
            or self.selected_recipe_id not in RECIPE_IDS
            or self.e0_recipe_id != E0_RECIPE_ID
            or len(self.top5_turn_ordinals) != TOP_K
            or len(set(self.top5_turn_ordinals)) != TOP_K
        ):
            raise LocomoP1TypedCoreError("policy decision drifted")
        if self.evaluator_id == "E0" and (
            self.selected_recipe_id != E0_RECIPE_ID
            or not self.fallback_to_e0
            or self.matched_signature is not None
            or self.conservative_minimum_delta != 0
        ):
            raise LocomoP1TypedCoreError("E0 decision drifted")


def apply_e0(slate: ActionSlate, *, stage: str) -> PolicyDecision:
    if not isinstance(slate, ActionSlate) or stage not in POLICY_STAGES:
        raise LocomoP1TypedCoreError("E0 application stage drifted")
    action = slate.action(E0_RECIPE_ID)
    return PolicyDecision(
        stage=stage,
        evaluator_id="E0",
        selected_recipe_id=E0_RECIPE_ID,
        e0_recipe_id=E0_RECIPE_ID,
        top5_turn_ordinals=action.top5_turn_ordinals,
        fallback_to_e0=True,
        matched_signature=None,
        conservative_minimum_delta=Fraction(0, 1),
    )


def apply_e1(
    model: E1Model, slate: ActionSlate, *, stage: str
) -> PolicyDecision:
    """Apply one frozen A_form model identically on F, A_hold, or M."""

    if (
        not isinstance(model, E1Model)
        or not isinstance(slate, ActionSlate)
        or stage not in POLICY_STAGES
    ):
        raise LocomoP1TypedCoreError("E1 application inputs drifted")
    reference = slate.feature_row(E0_RECIPE_ID)
    candidates: list[tuple[Fraction, Fraction, int, int, SignatureRule]] = []
    rules = {
        (rule.recipe_id, rule.signature): rule
        for rule in model.rules
        if rule.qualified
    }
    for recipe_id in RECIPE_IDS:
        if recipe_id == E0_RECIPE_ID:
            continue
        signature = pairwise_signature(slate.feature_row(recipe_id), reference)
        rule = rules.get((recipe_id, signature))
        if rule is None:
            continue
        candidates.append(
            (
                rule.minimum_delta,
                rule.net_utility,
                rule.support_count,
                -RECIPE_IDS.index(recipe_id),
                rule,
            )
        )
    if not candidates:
        baseline = apply_e0(slate, stage=stage)
        return PolicyDecision(
            stage=stage,
            evaluator_id="E1",
            selected_recipe_id=baseline.selected_recipe_id,
            e0_recipe_id=E0_RECIPE_ID,
            top5_turn_ordinals=baseline.top5_turn_ordinals,
            fallback_to_e0=True,
            matched_signature=None,
            conservative_minimum_delta=Fraction(0, 1),
        )
    selected_rule = max(candidates, key=lambda row: row[:4])[4]
    action = slate.action(selected_rule.recipe_id)
    return PolicyDecision(
        stage=stage,
        evaluator_id="E1",
        selected_recipe_id=selected_rule.recipe_id,
        e0_recipe_id=E0_RECIPE_ID,
        top5_turn_ordinals=action.top5_turn_ordinals,
        fallback_to_e0=False,
        matched_signature=selected_rule.signature,
        conservative_minimum_delta=selected_rule.minimum_delta,
    )


__all__ = [
    "ActionFeatures",
    "ActionSlate",
    "AFormExample",
    "E0_RECIPE_ID",
    "E1Model",
    "FEATURE_NAMES",
    "LocomoP1TypedCoreError",
    "MIN_SIGNATURE_SUPPORT",
    "POLICY_STAGES",
    "PUBLIC_TURN_FIELDS",
    "PolicyDecision",
    "QueryStructure",
    "R0_RAW_BM25",
    "R1_ENTITY_FOCUS",
    "R2_TEMPORAL_FOCUS",
    "R3_SPEAKER_EVENT",
    "R4_MULTI_SEED_MARGINAL",
    "R5_TYPED_CASCADE",
    "RECIPE_IDS",
    "RecipeAction",
    "SCALE",
    "SIGNATURE_FEATURE_NAMES",
    "SignatureRule",
    "TOP_K",
    "Turn",
    "TurnStructure",
    "apply_e0",
    "apply_e1",
    "bm25_scores",
    "build_action_slate",
    "canonical_bytes",
    "fit_e1",
    "lexical_tokens",
    "make_aform_example",
    "normalize_text",
    "pairwise_signature",
    "query_structure",
    "stable_hash",
    "turn_from_public_fields",
    "turn_public_payload",
]

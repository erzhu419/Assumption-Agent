"""Source-free typed passage selection for MultiDoc2Dial P1.

The module deliberately has no dataset loader and no label-bearing input.  An
action is a pure function of:

* the chronological public dialogue through the current user turn;
* canonical public passage coordinates ``ordinal/title/section/path/text``;
* frozen, integer-valued RAW dense and cross-encoder score arrays.

The source-native dialogue-act, domain, family, answer, document identity,
reference, qrel, split, and outcome never enter action formation.  Official
HippoRAG scores are handled by a separate baseline-only function and are not
accepted by :func:`build_action_slate`.

Four typed operators change top-five set coverage: history referent expansion,
topic/title/path expansion, complementary current/context coverage, and
section-neighbour closure.  E0 is a fifth, fixed conservative cascade over
those operators.  E1 can be fitted only from an already-computed A_form
integer utility vector.  It uses label-free structural signatures, requires
repeated non-negative gains, and penalizes each slot changed from E0.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import json
import re
import unicodedata
from typing import Mapping, Sequence


VERSION = "multidoc2dial_p1_typed_core_v1"
STUDY_ID = "MULTIDOC2DIAL_P1_TYPED_PASSAGE_RETRIEVAL_V1"
TOP_K = 5
SCALE = 1_000_000

MAX_HISTORY_TURNS = 128
MAX_TURN_CHARACTERS = 100_000
MAX_DIALOGUE_CHARACTERS = 1_000_000
MAX_PASSAGE_FIELD_CHARACTERS = 1_000_000
MAX_PATH_DEPTH = 64
MAX_SCORE_ABS = 10**15

MIN_SIGNATURE_SUPPORT = 4
MIN_POSITIVE_FRACTION_NUMERATOR = 3
MIN_POSITIVE_FRACTION_DENOMINATOR = 4
SET_CHANGE_PENALTY_MICROS = 10_000
POLICY_STAGES = ("F_search", "A_hold", "M_search")

R0_RAW_DENSE = "R0_RAW_DENSE"
R1_RAW_CE = "R1_RAW_CE"
R2_HISTORY_REFERENT = "R2_HISTORY_REFERENT"
R3_TOPIC_PATH_EXPANSION = "R3_TOPIC_PATH_EXPANSION"
R4_CONDITION_SOLUTION_COVERAGE = "R4_CONDITION_SOLUTION_COVERAGE"
R5_SECTION_NEIGHBOR_CLOSURE = "R5_SECTION_NEIGHBOR_CLOSURE"
R6_CONSERVATIVE_TYPED_CASCADE = "R6_CONSERVATIVE_TYPED_CASCADE"

BASELINE_RECIPE_IDS = (R0_RAW_DENSE, R1_RAW_CE)
AGENT_RECIPE_IDS = (
    R2_HISTORY_REFERENT,
    R3_TOPIC_PATH_EXPANSION,
    R4_CONDITION_SOLUTION_COVERAGE,
    R5_SECTION_NEIGHBOR_CLOSURE,
    R6_CONSERVATIVE_TYPED_CASCADE,
)
RECIPE_IDS = BASELINE_RECIPE_IDS + AGENT_RECIPE_IDS
E0_RECIPE_ID = R6_CONSERVATIVE_TYPED_CASCADE
OFFICIAL_HIPPORAG_BASELINE_ID = "B2_OFFICIAL_HIPPORAG"

PUBLIC_TURN_FIELDS = ("role", "text")
PUBLIC_PASSAGE_FIELDS = ("ordinal", "title", "section", "path", "text")
ROLE_VALUES = ("user", "agent")

FEATURE_NAMES = (
    "history_turn_bucket",
    "current_token_bucket",
    "context_token_bucket",
    "current_context_overlap_bucket",
    "raw_rank_agreement_bucket",
    "structural_path_bucket",
    "typed_set_diversity_bucket",
)

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "me",
        "my",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


class MultiDoc2DialP1TypedCoreError(ValueError):
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
        raise MultiDoc2DialP1TypedCoreError(
            "value is not canonical JSON"
        ) from exc
    return encoded + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _checked_text(
    value: object,
    *,
    field: str,
    maximum_length: int,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > maximum_length
        or (not allow_empty and not value.strip())
    ):
        raise MultiDoc2DialP1TypedCoreError(f"{field} is invalid")
    return value


def normalize_text(
    value: str,
    *,
    field: str = "text",
    allow_empty: bool = False,
) -> str:
    _checked_text(
        value,
        field=field,
        maximum_length=MAX_PASSAGE_FIELD_CHARACTERS,
        allow_empty=allow_empty,
    )
    normalized = " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    )
    if not allow_empty and not normalized:
        raise MultiDoc2DialP1TypedCoreError(f"{field} is empty")
    return normalized


def lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(value, allow_empty=True)))


def _content_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in lexical_tokens(value)
        if len(token) >= 2 and token not in _STOPWORDS
    )


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_VALUES:
            raise MultiDoc2DialP1TypedCoreError("turn role is invalid")
        _checked_text(
            self.text,
            field="turn text",
            maximum_length=MAX_TURN_CHARACTERS,
        )


def turn_from_public_fields(value: object) -> DialogueTurn:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(PUBLIC_TURN_FIELDS)
    ):
        raise MultiDoc2DialP1TypedCoreError(
            "turn projection is not the exact public field set"
        )
    return DialogueTurn(
        role=value["role"],  # type: ignore[arg-type]
        text=value["text"],  # type: ignore[arg-type]
    )


def turn_public_payload(turn: DialogueTurn) -> dict[str, str]:
    if not isinstance(turn, DialogueTurn):
        raise MultiDoc2DialP1TypedCoreError("turn is not a DialogueTurn")
    return {"role": turn.role, "text": turn.text}


def _checked_history(
    history: Sequence[DialogueTurn],
) -> tuple[DialogueTurn, ...]:
    if (
        isinstance(history, (str, bytes))
        or not 1 <= len(history) <= MAX_HISTORY_TURNS
    ):
        raise MultiDoc2DialP1TypedCoreError("dialogue history is invalid")
    checked = tuple(history)
    if any(not isinstance(turn, DialogueTurn) for turn in checked):
        raise MultiDoc2DialP1TypedCoreError(
            "dialogue history contains a non-turn"
        )
    if checked[0].role != "user" or checked[-1].role != "user":
        raise MultiDoc2DialP1TypedCoreError(
            "history must start and end at a user turn"
        )
    if sum(len(turn.text) for turn in checked) > MAX_DIALOGUE_CHARACTERS:
        raise MultiDoc2DialP1TypedCoreError("dialogue is too large")
    return checked


def normalized_query_payload(
    history: Sequence[DialogueTurn],
) -> dict[str, object]:
    """Return the exact label-free collision identity used by P0/formal."""

    checked = _checked_history(history)
    return {
        "turns": [
            {
                "role": turn.role,
                "text": normalize_text(
                    turn.text,
                    field="turn text",
                ),
            }
            for turn in checked
        ]
    }


def normalized_query_sha256(
    history: Sequence[DialogueTurn],
) -> str:
    return stable_hash(normalized_query_payload(history))


@dataclass(frozen=True, slots=True)
class Passage:
    ordinal: int
    title: str
    section: str
    path: tuple[str, ...]
    text: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise MultiDoc2DialP1TypedCoreError(
                "passage ordinal is invalid"
            )
        _checked_text(
            self.title,
            field="passage title",
            maximum_length=MAX_PASSAGE_FIELD_CHARACTERS,
            allow_empty=True,
        )
        _checked_text(
            self.section,
            field="passage section",
            maximum_length=MAX_PASSAGE_FIELD_CHARACTERS,
            allow_empty=True,
        )
        if (
            not isinstance(self.path, tuple)
            or len(self.path) > MAX_PATH_DEPTH
            or any(
                not isinstance(component, str)
                or "\x00" in component
                or len(component) > MAX_PASSAGE_FIELD_CHARACTERS
                for component in self.path
            )
        ):
            raise MultiDoc2DialP1TypedCoreError("passage path is invalid")
        _checked_text(
            self.text,
            field="passage text",
            maximum_length=MAX_PASSAGE_FIELD_CHARACTERS,
        )
        if not lexical_tokens(
            "\n".join((self.title, self.section, *self.path, self.text))
        ):
            raise MultiDoc2DialP1TypedCoreError(
                "passage has no lexical token"
            )


def passage_from_public_fields(value: object) -> Passage:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(PUBLIC_PASSAGE_FIELDS)
    ):
        raise MultiDoc2DialP1TypedCoreError(
            "passage projection is not the exact public field set"
        )
    path = value["path"]
    if isinstance(path, list):
        path = tuple(path)
    return Passage(
        ordinal=value["ordinal"],  # type: ignore[arg-type]
        title=value["title"],  # type: ignore[arg-type]
        section=value["section"],  # type: ignore[arg-type]
        path=path,  # type: ignore[arg-type]
        text=value["text"],  # type: ignore[arg-type]
    )


def passage_public_payload(passage: Passage) -> dict[str, object]:
    if not isinstance(passage, Passage):
        raise MultiDoc2DialP1TypedCoreError("passage is not a Passage")
    return {
        "ordinal": passage.ordinal,
        "path": list(passage.path),
        "section": passage.section,
        "text": passage.text,
        "title": passage.title,
    }


def serialize_passage_text(passage: Passage) -> str:
    """Canonical label-free text shared by typed passage operators."""

    if not isinstance(passage, Passage):
        raise MultiDoc2DialP1TypedCoreError("passage is not a Passage")
    return "\n".join(
        (passage.title, *passage.path, passage.section, passage.text)
    )


def _checked_inputs(
    passages: Sequence[Passage],
    raw_dense_scores: Sequence[int],
    raw_ce_scores: Sequence[int],
) -> tuple[tuple[Passage, ...], tuple[int, ...], tuple[int, ...]]:
    if isinstance(passages, (str, bytes)) or len(passages) < TOP_K:
        raise MultiDoc2DialP1TypedCoreError(
            "candidate set has fewer than five passages"
        )
    checked_passages = tuple(passages)
    if any(not isinstance(passage, Passage) for passage in checked_passages):
        raise MultiDoc2DialP1TypedCoreError(
            "candidate set contains a non-passage"
        )
    if (
        len({passage.ordinal for passage in checked_passages})
        != len(checked_passages)
    ):
        raise MultiDoc2DialP1TypedCoreError(
            "passage ordinals are not unique"
        )
    if (
        isinstance(raw_dense_scores, (str, bytes))
        or isinstance(raw_ce_scores, (str, bytes))
        or len(raw_dense_scores) != len(checked_passages)
        or len(raw_ce_scores) != len(checked_passages)
    ):
        raise MultiDoc2DialP1TypedCoreError(
            "base score vector width drifted"
        )
    dense = tuple(raw_dense_scores)
    ce = tuple(raw_ce_scores)
    if any(
        type(score) is not int or abs(score) > MAX_SCORE_ABS
        for score in dense + ce
    ):
        raise MultiDoc2DialP1TypedCoreError(
            "base scores are not frozen bounded integers"
        )
    rows = sorted(
        zip(checked_passages, dense, ce),
        key=lambda row: row[0].ordinal,
    )
    return (
        tuple(row[0] for row in rows),
        tuple(row[1] for row in rows),
        tuple(row[2] for row in rows),
    )


def _rank(
    scores: Sequence[int],
    passages: Sequence[Passage],
) -> tuple[int, ...]:
    if len(scores) != len(passages):
        raise MultiDoc2DialP1TypedCoreError("rank width drifted")
    return tuple(
        sorted(
            range(len(passages)),
            key=lambda index: (-scores[index], passages[index].ordinal),
        )
    )


def _rank_points(
    scores: Sequence[int],
    passages: Sequence[Passage],
) -> tuple[int, ...]:
    order = _rank(scores, passages)
    points = [0] * len(passages)
    for rank, index in enumerate(order):
        points[index] = (
            (len(passages) - rank) * SCALE // len(passages)
        )
    return tuple(points)


def _coverage(tokens: set[str], anchors: set[str]) -> int:
    if not anchors:
        return 0
    return len(tokens & anchors) * SCALE // len(anchors)


def _jaccard(left: set[str], right: set[str]) -> int:
    union = left | right
    if not union:
        return 0
    return len(left & right) * SCALE // len(union)


def _bucket(value: int, cuts: tuple[int, ...]) -> int:
    return sum(value >= cut for cut in cuts)


@dataclass(frozen=True, slots=True)
class QueryStructure:
    current_tokens: tuple[str, ...]
    prior_user_tokens: tuple[str, ...]
    prior_agent_tokens: tuple[str, ...]
    recency_expansion_tokens: tuple[str, ...]
    normalized_query_sha256: str
    history_turn_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "current_tokens",
            "prior_user_tokens",
            "prior_agent_tokens",
            "recency_expansion_tokens",
        ):
            value = getattr(self, field_name)
            if value != tuple(sorted(set(value))):
                raise MultiDoc2DialP1TypedCoreError(
                    f"{field_name} is not a sorted set"
                )
        if re.fullmatch(r"[0-9a-f]{64}", self.normalized_query_sha256) is None:
            raise MultiDoc2DialP1TypedCoreError(
                "normalized query hash is invalid"
            )

    def payload(self) -> dict[str, object]:
        return {
            "current_tokens": list(self.current_tokens),
            "history_turn_count": self.history_turn_count,
            "normalized_query_sha256": self.normalized_query_sha256,
            "prior_agent_tokens": list(self.prior_agent_tokens),
            "prior_user_tokens": list(self.prior_user_tokens),
            "recency_expansion_tokens": list(
                self.recency_expansion_tokens
            ),
        }


def query_structure(
    history: Sequence[DialogueTurn],
) -> QueryStructure:
    checked = _checked_history(history)
    current = set(_content_tokens(checked[-1].text))
    prior_user: set[str] = set()
    prior_agent: set[str] = set()
    weighted: Counter[str] = Counter()
    for recency, turn in enumerate(checked[:-1], start=1):
        tokens = set(_content_tokens(turn.text))
        if turn.role == "user":
            prior_user.update(tokens)
        else:
            prior_agent.update(tokens)
        weight = recency * (2 if turn.role == "agent" else 1)
        weighted.update({token: weight for token in tokens})
    ranked_expansion = [
        token
        for token, _weight in sorted(
            weighted.items(),
            key=lambda row: (-row[1], row[0]),
        )
        if token not in current
    ][:32]
    expansion = tuple(sorted(ranked_expansion))
    return QueryStructure(
        current_tokens=tuple(sorted(current)),
        prior_user_tokens=tuple(sorted(prior_user)),
        prior_agent_tokens=tuple(sorted(prior_agent)),
        recency_expansion_tokens=expansion,
        normalized_query_sha256=normalized_query_sha256(checked),
        history_turn_count=len(checked),
    )


@dataclass(frozen=True, slots=True)
class PassageStructure:
    passage_ordinal: int
    serialized_sha256: str
    current_coverage: int
    context_coverage: int
    history_coverage: int
    structural_token_count: int

    def payload(self) -> dict[str, object]:
        return {
            "context_coverage": self.context_coverage,
            "current_coverage": self.current_coverage,
            "history_coverage": self.history_coverage,
            "passage_ordinal": self.passage_ordinal,
            "serialized_sha256": self.serialized_sha256,
            "structural_token_count": self.structural_token_count,
        }


@dataclass(frozen=True, slots=True)
class _PassageCoordinates:
    all_tokens: frozenset[str]
    structural_tokens: frozenset[str]
    sentence_tokens: tuple[frozenset[str], ...]


def _passage_coordinates(
    passage: Passage,
) -> _PassageCoordinates:
    structural = _content_tokens(
        "\n".join((passage.title, *passage.path, passage.section))
    )
    all_tokens = _content_tokens(serialize_passage_text(passage))
    sentences = tuple(
        tokens
        for part in _SENTENCE_RE.split(passage.text)
        if (tokens := _content_tokens(part))
    )
    if not sentences:
        sentences = (all_tokens,)
    return _PassageCoordinates(
        all_tokens=all_tokens,
        structural_tokens=structural,
        sentence_tokens=sentences,
    )


def _passage_structures(
    passages: Sequence[Passage],
    coordinates: Sequence[_PassageCoordinates],
    structure: QueryStructure,
) -> tuple[PassageStructure, ...]:
    current = set(structure.current_tokens)
    context = set(structure.prior_user_tokens) | set(
        structure.prior_agent_tokens
    )
    expansion = set(structure.recency_expansion_tokens)
    return tuple(
        PassageStructure(
            passage_ordinal=passage.ordinal,
            serialized_sha256=hashlib.sha256(
                serialize_passage_text(passage).encode("utf-8")
            ).hexdigest(),
            current_coverage=_coverage(
                set(coordinate.all_tokens),
                current,
            ),
            context_coverage=_coverage(
                set(coordinate.all_tokens),
                context,
            ),
            history_coverage=_coverage(
                set(coordinate.all_tokens),
                current | expansion,
            ),
            structural_token_count=len(coordinate.structural_tokens),
        )
        for passage, coordinate in zip(passages, coordinates)
    )


def _top5(order: Sequence[int]) -> tuple[int, ...]:
    result = tuple(order[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise MultiDoc2DialP1TypedCoreError(
            "operator did not totalize a unique top5"
        )
    return result


def _history_referent_order(
    *,
    passages: Sequence[Passage],
    coordinates: Sequence[_PassageCoordinates],
    structure: QueryStructure,
    dense_points: Sequence[int],
    ce_points: Sequence[int],
) -> tuple[int, ...]:
    current = set(structure.current_tokens)
    context = set(structure.recency_expansion_tokens)
    scores = tuple(
        3 * ce_points[index]
        + dense_points[index]
        + 3
        * _coverage(set(coordinate.all_tokens), current)
        + 12
        * _coverage(set(coordinate.all_tokens), context)
        for index, coordinate in enumerate(coordinates)
    )
    return _rank(scores, passages)


def _topic_path_order(
    *,
    passages: Sequence[Passage],
    coordinates: Sequence[_PassageCoordinates],
    dense_points: Sequence[int],
    ce_points: Sequence[int],
    seed_order: Sequence[int],
) -> tuple[int, ...]:
    seed_indices = tuple(seed_order[:1])
    seed_structure = set().union(
        *(set(coordinates[index].structural_tokens) for index in seed_indices)
    )
    scores = tuple(
        2 * ce_points[index]
        + dense_points[index]
        + 5
        * _jaccard(
            set(coordinate.structural_tokens),
            seed_structure,
        )
        for index, coordinate in enumerate(coordinates)
    )
    ranked = _rank(scores, passages)
    selected = [seed_indices[0]]
    selected.extend(index for index in ranked if index not in selected)
    return tuple(selected)


def _complementary_coverage_order(
    *,
    passages: Sequence[Passage],
    coordinates: Sequence[_PassageCoordinates],
    structure: QueryStructure,
    dense_points: Sequence[int],
    ce_points: Sequence[int],
) -> tuple[int, ...]:
    current_uncovered = set(structure.current_tokens)
    context_uncovered = set(structure.prior_user_tokens) | set(
        structure.prior_agent_tokens
    )
    selected: list[int] = []
    while len(selected) < len(passages):
        candidates: list[tuple[int, int, int]] = []
        for index, passage in enumerate(passages):
            if index in selected:
                continue
            tokens = set(coordinates[index].all_tokens)
            current_gain = _coverage(tokens, current_uncovered)
            context_gain = _coverage(tokens, context_uncovered)
            sentence_bridge = max(
                (
                    min(
                        _coverage(set(sentence), current_uncovered),
                        _coverage(set(sentence), context_uncovered),
                    )
                    for sentence in coordinates[index].sentence_tokens
                ),
                default=0,
            )
            base = ce_points[index] + dense_points[index]
            score = (
                base
                + 4 * current_gain
                + 4 * context_gain
                + 2 * sentence_bridge
            )
            candidates.append((-score, passage.ordinal, index))
        _negative, _ordinal, chosen = min(candidates)
        selected.append(chosen)
        chosen_tokens = set(coordinates[chosen].all_tokens)
        current_uncovered.difference_update(chosen_tokens)
        context_uncovered.difference_update(chosen_tokens)
    return tuple(selected)


def _neighbour_candidates(
    seed: int,
    *,
    passages: Sequence[Passage],
) -> tuple[int, ...]:
    seed_passage = passages[seed]
    same_title = [
        index
        for index, passage in enumerate(passages)
        if normalize_text(
            passage.title,
            allow_empty=True,
        )
        == normalize_text(seed_passage.title, allow_empty=True)
    ]
    same_parent = [
        index
        for index in same_title
        if passages[index].path[:-1] == seed_passage.path[:-1]
    ]
    group = same_parent if len(same_parent) >= 2 else same_title
    ordered = sorted(group, key=lambda index: passages[index].ordinal)
    if seed not in ordered:
        return ()
    location = ordered.index(seed)
    result: list[int] = []
    for offset in (1, -1, 2, -2):
        neighbour = location + offset
        if 0 <= neighbour < len(ordered):
            result.append(ordered[neighbour])
    return tuple(result)


def _section_neighbour_order(
    *,
    passages: Sequence[Passage],
    ce_order: Sequence[int],
) -> tuple[int, ...]:
    seeds: list[int] = []
    seen_groups: set[tuple[str, tuple[str, ...]]] = set()
    for index in ce_order:
        passage = passages[index]
        group = (
            normalize_text(passage.title, allow_empty=True),
            passage.path[:-1],
        )
        if group not in seen_groups:
            seen_groups.add(group)
            seeds.append(index)
        if len(seeds) == 3:
            break
    selected: list[int] = []
    for seed in seeds:
        if seed not in selected:
            selected.append(seed)
    neighbour_lists = [
        _neighbour_candidates(seed, passages=passages)
        for seed in seeds
    ]
    for neighbours in neighbour_lists:
        if neighbours:
            neighbour = neighbours[0]
            if neighbour not in selected:
                selected.append(neighbour)
    for neighbours in neighbour_lists:
        for neighbour in neighbours[1:]:
            if neighbour not in selected:
                selected.append(neighbour)
    selected.extend(index for index in ce_order if index not in selected)
    return tuple(selected)


def _conservative_cascade_order(
    *,
    ce_order: Sequence[int],
    dense_order: Sequence[int],
    typed_orders: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    selected: list[int] = [ce_order[0]]
    for order in typed_orders:
        for index in order:
            if index not in selected:
                selected.append(index)
                break
    for order in (ce_order, dense_order):
        selected.extend(index for index in order if index not in selected)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class RecipeAction:
    recipe_id: str
    top5_passage_ordinals: tuple[int, ...]
    selection_trace: tuple[str, ...]
    behavior_digest: str

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise MultiDoc2DialP1TypedCoreError("recipe is not frozen")
        if (
            len(self.top5_passage_ordinals) != TOP_K
            or len(set(self.top5_passage_ordinals)) != TOP_K
            or any(
                type(value) is not int or value < 0
                for value in self.top5_passage_ordinals
            )
        ):
            raise MultiDoc2DialP1TypedCoreError(
                "recipe top5 is malformed"
            )
        if (
            len(self.selection_trace) != TOP_K
            or any(not isinstance(value, str) for value in self.selection_trace)
        ):
            raise MultiDoc2DialP1TypedCoreError(
                "recipe trace is malformed"
            )
        if re.fullmatch(r"[0-9a-f]{64}", self.behavior_digest) is None:
            raise MultiDoc2DialP1TypedCoreError(
                "recipe behavior digest is invalid"
            )

    def payload(self) -> dict[str, object]:
        return {
            "behavior_digest": self.behavior_digest,
            "recipe_id": self.recipe_id,
            "selection_trace": list(self.selection_trace),
            "top5_passage_ordinals": list(self.top5_passage_ordinals),
        }


def _make_action(
    *,
    recipe_id: str,
    order: Sequence[int],
    passages: Sequence[Passage],
    query_sha256: str,
    passage_sha256: str,
    base_score_sha256: str,
    trace_prefix: str,
) -> RecipeAction:
    selected = _top5(order)
    ordinals = tuple(passages[index].ordinal for index in selected)
    trace = tuple(
        f"{trace_prefix}:{rank}:{passages[index].ordinal}"
        for rank, index in enumerate(selected)
    )
    behavior = stable_hash(
        {
            "base_score_sha256": base_score_sha256,
            "passage_projection_sha256": passage_sha256,
            "query_sha256": query_sha256,
            "recipe_id": recipe_id,
            "selection_trace": list(trace),
            "top5_passage_ordinals": list(ordinals),
            "version": VERSION,
        }
    )
    return RecipeAction(
        recipe_id=recipe_id,
        top5_passage_ordinals=ordinals,
        selection_trace=trace,
        behavior_digest=behavior,
    )


@dataclass(frozen=True, slots=True)
class ActionFeatures:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            len(self.values) != len(FEATURE_NAMES)
            or any(type(value) is not int or value < 0 for value in self.values)
        ):
            raise MultiDoc2DialP1TypedCoreError(
                "action feature vector is invalid"
            )

    def payload(self) -> dict[str, int]:
        return dict(zip(FEATURE_NAMES, self.values))


@dataclass(frozen=True, slots=True)
class ActionSlate:
    query: QueryStructure
    passage_structures: tuple[PassageStructure, ...]
    actions: tuple[RecipeAction, ...]
    features: ActionFeatures
    passage_projection_sha256: str
    base_score_sha256: str

    def __post_init__(self) -> None:
        if tuple(action.recipe_id for action in self.actions) != RECIPE_IDS:
            raise MultiDoc2DialP1TypedCoreError(
                "action slate recipe order drifted"
            )
        if len({action.behavior_digest for action in self.actions}) != len(
            self.actions
        ):
            raise MultiDoc2DialP1TypedCoreError(
                "action behavior digests collided"
            )

    def action(self, recipe_id: str) -> RecipeAction:
        if recipe_id not in RECIPE_IDS:
            raise MultiDoc2DialP1TypedCoreError("recipe is not frozen")
        return self.actions[RECIPE_IDS.index(recipe_id)]

    def audit_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "actions": [action.payload() for action in self.actions],
            "agent_recipe_ids": list(AGENT_RECIPE_IDS),
            "base_score_sha256": self.base_score_sha256,
            "baseline_recipe_ids": list(BASELINE_RECIPE_IDS),
            "e0_recipe_id": E0_RECIPE_ID,
            "feature_names": list(FEATURE_NAMES),
            "features": self.features.payload(),
            "hipporag_is_agent_input": False,
            "passage_projection_sha256": (
                self.passage_projection_sha256
            ),
            "passage_structures": [
                row.payload() for row in self.passage_structures
            ],
            "policy_stages": list(POLICY_STAGES),
            "public_passage_fields": list(PUBLIC_PASSAGE_FIELDS),
            "public_turn_fields": list(PUBLIC_TURN_FIELDS),
            "query": self.query.payload(),
            "study_id": STUDY_ID,
            "version": VERSION,
        }
        return {**payload, "self_sha256": stable_hash(payload)}


def _feature_vector(
    *,
    structure: QueryStructure,
    passages: Sequence[Passage],
    dense_order: Sequence[int],
    ce_order: Sequence[int],
    typed_top5: Sequence[tuple[int, ...]],
) -> ActionFeatures:
    current = set(structure.current_tokens)
    context = set(structure.prior_user_tokens) | set(
        structure.prior_agent_tokens
    )
    overlap = _coverage(current, context)
    rank_agreement = len(set(dense_order[:TOP_K]) & set(ce_order[:TOP_K]))
    structural = sum(bool(passage.path or passage.section) for passage in passages)
    unique_sets = len(set(typed_top5))
    return ActionFeatures(
        (
            _bucket(structure.history_turn_count, (3, 7, 15)),
            _bucket(len(current), (2, 5, 10)),
            _bucket(len(context), (5, 15, 40)),
            _bucket(overlap, (1, SCALE // 4, SCALE // 2)),
            rank_agreement,
            _bucket(structural, (TOP_K, 2 * TOP_K, 4 * TOP_K)),
            unique_sets,
        )
    )


def build_action_slate(
    history: Sequence[DialogueTurn],
    passages: Sequence[Passage],
    raw_dense_scores: Sequence[int],
    raw_ce_scores: Sequence[int],
) -> ActionSlate:
    """Build all frozen actions without accepting any label or Hippo score."""

    checked_history = _checked_history(history)
    checked_passages, dense, ce = _checked_inputs(
        passages,
        raw_dense_scores,
        raw_ce_scores,
    )
    structure = query_structure(checked_history)
    coordinates = tuple(
        _passage_coordinates(passage) for passage in checked_passages
    )
    rows = _passage_structures(
        checked_passages,
        coordinates,
        structure,
    )
    dense_order = _rank(dense, checked_passages)
    ce_order = _rank(ce, checked_passages)
    dense_points = _rank_points(dense, checked_passages)
    ce_points = _rank_points(ce, checked_passages)
    history_order = _history_referent_order(
        passages=checked_passages,
        coordinates=coordinates,
        structure=structure,
        dense_points=dense_points,
        ce_points=ce_points,
    )
    topic_order = _topic_path_order(
        passages=checked_passages,
        coordinates=coordinates,
        dense_points=dense_points,
        ce_points=ce_points,
        seed_order=history_order,
    )
    complementary_order = _complementary_coverage_order(
        passages=checked_passages,
        coordinates=coordinates,
        structure=structure,
        dense_points=dense_points,
        ce_points=ce_points,
    )
    neighbour_order = _section_neighbour_order(
        passages=checked_passages,
        ce_order=ce_order,
    )
    cascade_order = _conservative_cascade_order(
        ce_order=ce_order,
        dense_order=dense_order,
        typed_orders=(
            history_order,
            topic_order,
            complementary_order,
            neighbour_order,
        ),
    )
    order_by_recipe = {
        R0_RAW_DENSE: dense_order,
        R1_RAW_CE: ce_order,
        R2_HISTORY_REFERENT: history_order,
        R3_TOPIC_PATH_EXPANSION: topic_order,
        R4_CONDITION_SOLUTION_COVERAGE: complementary_order,
        R5_SECTION_NEIGHBOR_CLOSURE: neighbour_order,
        R6_CONSERVATIVE_TYPED_CASCADE: cascade_order,
    }
    projection_payload = [
        passage_public_payload(passage) for passage in checked_passages
    ]
    passage_sha = stable_hash(projection_payload)
    base_sha = stable_hash(
        {
            "ordinals": [
                passage.ordinal for passage in checked_passages
            ],
            "raw_ce_scores": list(ce),
            "raw_dense_scores": list(dense),
        }
    )
    actions = tuple(
        _make_action(
            recipe_id=recipe_id,
            order=order_by_recipe[recipe_id],
            passages=checked_passages,
            query_sha256=structure.normalized_query_sha256,
            passage_sha256=passage_sha,
            base_score_sha256=base_sha,
            trace_prefix=recipe_id.casefold(),
        )
        for recipe_id in RECIPE_IDS
    )
    typed_top5 = tuple(
        action.top5_passage_ordinals
        for action in actions
        if action.recipe_id in AGENT_RECIPE_IDS
    )
    return ActionSlate(
        query=structure,
        passage_structures=rows,
        actions=actions,
        features=_feature_vector(
            structure=structure,
            passages=checked_passages,
            dense_order=dense_order,
            ce_order=ce_order,
            typed_top5=typed_top5,
        ),
        passage_projection_sha256=passage_sha,
        base_score_sha256=base_sha,
    )


@dataclass(frozen=True, slots=True)
class ExternalBaselineAction:
    baseline_id: str
    top5_passage_ordinals: tuple[int, ...]
    behavior_digest: str

    def __post_init__(self) -> None:
        if self.baseline_id != OFFICIAL_HIPPORAG_BASELINE_ID:
            raise MultiDoc2DialP1TypedCoreError(
                "external baseline is not frozen"
            )
        if (
            len(self.top5_passage_ordinals) != TOP_K
            or len(set(self.top5_passage_ordinals)) != TOP_K
        ):
            raise MultiDoc2DialP1TypedCoreError(
                "external baseline top5 is malformed"
            )


def build_official_hipporag_baseline(
    passages: Sequence[Passage],
    hipporag_scores: Sequence[int],
) -> ExternalBaselineAction:
    """Rank optional Hippo scores in a baseline-only, isolated API."""

    dummy = tuple(0 for _ in passages)
    checked, hippo, _dummy = _checked_inputs(
        passages,
        hipporag_scores,
        dummy,
    )
    order = _rank(hippo, checked)
    top5 = tuple(checked[index].ordinal for index in order[:TOP_K])
    return ExternalBaselineAction(
        baseline_id=OFFICIAL_HIPPORAG_BASELINE_ID,
        top5_passage_ordinals=top5,
        behavior_digest=stable_hash(
            {
                "baseline_id": OFFICIAL_HIPPORAG_BASELINE_ID,
                "ordinals": [passage.ordinal for passage in checked],
                "scores": list(hippo),
                "top5_passage_ordinals": list(top5),
                "version": VERSION,
            }
        ),
    )


def policy_signature(slate: ActionSlate) -> tuple[int, ...]:
    if not isinstance(slate, ActionSlate):
        raise MultiDoc2DialP1TypedCoreError("slate is invalid")
    return slate.features.values


@dataclass(frozen=True, slots=True)
class AFormExample:
    signature: tuple[int, ...]
    outcome_vector: tuple[int, ...]
    changed_slots_from_e0: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.signature) != len(FEATURE_NAMES):
            raise MultiDoc2DialP1TypedCoreError(
                "A_form signature width drifted"
            )
        if (
            len(self.outcome_vector) != len(AGENT_RECIPE_IDS)
            or any(
                type(value) is not int or not 0 <= value <= SCALE
                for value in self.outcome_vector
            )
        ):
            raise MultiDoc2DialP1TypedCoreError(
                "A_form outcome vector is invalid"
            )
        if (
            len(self.changed_slots_from_e0) != len(AGENT_RECIPE_IDS)
            or any(
                type(value) is not int or not 0 <= value <= TOP_K
                for value in self.changed_slots_from_e0
            )
        ):
            raise MultiDoc2DialP1TypedCoreError(
                "A_form change vector is invalid"
            )


def make_aform_example(
    slate: ActionSlate,
    outcome_vector: Sequence[int],
) -> AFormExample:
    """Bind an externally scored, fixed-order utility vector to a slate."""

    if not isinstance(slate, ActionSlate):
        raise MultiDoc2DialP1TypedCoreError("slate is invalid")
    outcomes = tuple(outcome_vector)
    if len(outcomes) != len(AGENT_RECIPE_IDS):
        raise MultiDoc2DialP1TypedCoreError(
            "outcome vector must follow AGENT_RECIPE_IDS"
        )
    e0 = set(slate.action(E0_RECIPE_ID).top5_passage_ordinals)
    changes = tuple(
        TOP_K
        - len(
            e0
            & set(
                slate.action(recipe_id).top5_passage_ordinals
            )
        )
        for recipe_id in AGENT_RECIPE_IDS
    )
    return AFormExample(
        signature=policy_signature(slate),
        outcome_vector=outcomes,
        changed_slots_from_e0=changes,
    )


@dataclass(frozen=True, slots=True)
class SignatureRule:
    signature: tuple[int, ...]
    recipe_id: str
    support_count: int
    positive_count: int
    minimum_delta: int
    mean_delta: Fraction
    mean_changed_slots: Fraction
    regularized_mean_delta: Fraction
    qualified: bool

    def __post_init__(self) -> None:
        if self.recipe_id not in AGENT_RECIPE_IDS[:-1]:
            raise MultiDoc2DialP1TypedCoreError(
                "E1 rule may select only a non-E0 Agent recipe"
            )
        expected = (
            self.support_count >= MIN_SIGNATURE_SUPPORT
            and self.positive_count
            * MIN_POSITIVE_FRACTION_DENOMINATOR
            >= self.support_count * MIN_POSITIVE_FRACTION_NUMERATOR
            and self.minimum_delta >= 0
            and self.regularized_mean_delta > 0
        )
        if self.qualified != expected:
            raise MultiDoc2DialP1TypedCoreError(
                "E1 qualification rule drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "mean_changed_slots": [
                self.mean_changed_slots.numerator,
                self.mean_changed_slots.denominator,
            ],
            "mean_delta": [
                self.mean_delta.numerator,
                self.mean_delta.denominator,
            ],
            "minimum_delta": self.minimum_delta,
            "positive_count": self.positive_count,
            "qualified": self.qualified,
            "recipe_id": self.recipe_id,
            "regularized_mean_delta": [
                self.regularized_mean_delta.numerator,
                self.regularized_mean_delta.denominator,
            ],
            "signature": list(self.signature),
            "support_count": self.support_count,
        }


@dataclass(frozen=True, slots=True)
class E1Model:
    rules: tuple[SignatureRule, ...]
    training_item_count: int
    training_stage: str = "A_form"

    def __post_init__(self) -> None:
        if (
            type(self.training_item_count) is not int
            or self.training_item_count < 0
            or self.training_stage != "A_form"
        ):
            raise MultiDoc2DialP1TypedCoreError("E1 model is invalid")
        identities = tuple(
            (rule.signature, rule.recipe_id) for rule in self.rules
        )
        if identities != tuple(sorted(identities)):
            raise MultiDoc2DialP1TypedCoreError(
                "E1 rules are not canonical"
            )
        if len(set(identities)) != len(identities):
            raise MultiDoc2DialP1TypedCoreError(
                "E1 rules are duplicated"
            )

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "agent_recipe_ids": list(AGENT_RECIPE_IDS),
            "e0_recipe_id": E0_RECIPE_ID,
            "feature_names": list(FEATURE_NAMES),
            "minimum_signature_support": MIN_SIGNATURE_SUPPORT,
            "minimum_positive_fraction": [
                MIN_POSITIVE_FRACTION_NUMERATOR,
                MIN_POSITIVE_FRACTION_DENOMINATOR,
            ],
            "rules": [rule.payload() for rule in self.rules],
            "set_change_penalty_micros": SET_CHANGE_PENALTY_MICROS,
            "training_item_count": self.training_item_count,
            "training_stage": self.training_stage,
            "version": VERSION,
        }
        return {**payload, "self_sha256": stable_hash(payload)}


def fit_e1(examples: Sequence[AFormExample]) -> E1Model:
    """Fit the conservative E1 table from A_form outcome vectors only."""

    if isinstance(examples, (str, bytes)):
        raise MultiDoc2DialP1TypedCoreError("A_form examples are invalid")
    checked = tuple(examples)
    if any(not isinstance(example, AFormExample) for example in checked):
        raise MultiDoc2DialP1TypedCoreError(
            "A_form contains a non-example"
        )
    grouped: defaultdict[
        tuple[int, ...],
        list[AFormExample],
    ] = defaultdict(list)
    for example in checked:
        grouped[example.signature].append(example)
    e0_index = AGENT_RECIPE_IDS.index(E0_RECIPE_ID)
    rules: list[SignatureRule] = []
    for signature in sorted(grouped):
        rows = grouped[signature]
        for recipe_index, recipe_id in enumerate(AGENT_RECIPE_IDS[:-1]):
            deltas = [
                row.outcome_vector[recipe_index]
                - row.outcome_vector[e0_index]
                for row in rows
            ]
            changes = [
                row.changed_slots_from_e0[recipe_index]
                for row in rows
            ]
            support = len(rows)
            positive = sum(delta > 0 for delta in deltas)
            minimum = min(deltas)
            mean_delta = Fraction(sum(deltas), support)
            mean_changes = Fraction(sum(changes), support)
            regularized = (
                mean_delta
                - SET_CHANGE_PENALTY_MICROS * mean_changes
            )
            qualified = (
                support >= MIN_SIGNATURE_SUPPORT
                and positive * MIN_POSITIVE_FRACTION_DENOMINATOR
                >= support * MIN_POSITIVE_FRACTION_NUMERATOR
                and minimum >= 0
                and regularized > 0
            )
            rules.append(
                SignatureRule(
                    signature=signature,
                    recipe_id=recipe_id,
                    support_count=support,
                    positive_count=positive,
                    minimum_delta=minimum,
                    mean_delta=mean_delta,
                    mean_changed_slots=mean_changes,
                    regularized_mean_delta=regularized,
                    qualified=qualified,
                )
            )
    return E1Model(
        rules=tuple(
            sorted(rules, key=lambda rule: (rule.signature, rule.recipe_id))
        ),
        training_item_count=len(checked),
    )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    evaluator_id: str
    stage: str
    selected_recipe_id: str
    top5_passage_ordinals: tuple[int, ...]
    behavior_digest: str
    fallback_to_e0: bool

    def __post_init__(self) -> None:
        if self.evaluator_id not in ("E0", "E1"):
            raise MultiDoc2DialP1TypedCoreError("evaluator is invalid")
        if self.stage not in POLICY_STAGES:
            raise MultiDoc2DialP1TypedCoreError("policy stage is invalid")
        if self.selected_recipe_id not in AGENT_RECIPE_IDS:
            raise MultiDoc2DialP1TypedCoreError(
                "policy selected a non-Agent recipe"
            )


def apply_e0(slate: ActionSlate, *, stage: str) -> PolicyDecision:
    if stage not in POLICY_STAGES:
        raise MultiDoc2DialP1TypedCoreError("policy stage is invalid")
    action = slate.action(E0_RECIPE_ID)
    return PolicyDecision(
        evaluator_id="E0",
        stage=stage,
        selected_recipe_id=E0_RECIPE_ID,
        top5_passage_ordinals=action.top5_passage_ordinals,
        behavior_digest=action.behavior_digest,
        fallback_to_e0=False,
    )


def apply_e1(
    model: E1Model,
    slate: ActionSlate,
    *,
    stage: str,
) -> PolicyDecision:
    if not isinstance(model, E1Model):
        raise MultiDoc2DialP1TypedCoreError("E1 model is invalid")
    if stage not in POLICY_STAGES:
        raise MultiDoc2DialP1TypedCoreError("policy stage is invalid")
    signature = policy_signature(slate)
    candidates = [
        rule
        for rule in model.rules
        if rule.signature == signature and rule.qualified
    ]
    if not candidates:
        e0 = apply_e0(slate, stage=stage)
        return PolicyDecision(
            evaluator_id="E1",
            stage=stage,
            selected_recipe_id=e0.selected_recipe_id,
            top5_passage_ordinals=e0.top5_passage_ordinals,
            behavior_digest=e0.behavior_digest,
            fallback_to_e0=True,
        )
    selected = max(
        candidates,
        key=lambda rule: (
            rule.regularized_mean_delta,
            rule.minimum_delta,
            -rule.mean_changed_slots,
            -AGENT_RECIPE_IDS.index(rule.recipe_id),
        ),
    )
    action = slate.action(selected.recipe_id)
    return PolicyDecision(
        evaluator_id="E1",
        stage=stage,
        selected_recipe_id=selected.recipe_id,
        top5_passage_ordinals=action.top5_passage_ordinals,
        behavior_digest=action.behavior_digest,
        fallback_to_e0=False,
    )


__all__ = [
    "AGENT_RECIPE_IDS",
    "ActionFeatures",
    "ActionSlate",
    "AFormExample",
    "BASELINE_RECIPE_IDS",
    "DialogueTurn",
    "E0_RECIPE_ID",
    "E1Model",
    "ExternalBaselineAction",
    "FEATURE_NAMES",
    "MIN_SIGNATURE_SUPPORT",
    "MIN_POSITIVE_FRACTION_DENOMINATOR",
    "MIN_POSITIVE_FRACTION_NUMERATOR",
    "MultiDoc2DialP1TypedCoreError",
    "OFFICIAL_HIPPORAG_BASELINE_ID",
    "POLICY_STAGES",
    "PUBLIC_PASSAGE_FIELDS",
    "PUBLIC_TURN_FIELDS",
    "Passage",
    "PolicyDecision",
    "QueryStructure",
    "RECIPE_IDS",
    "RecipeAction",
    "SET_CHANGE_PENALTY_MICROS",
    "STUDY_ID",
    "SignatureRule",
    "TOP_K",
    "VERSION",
    "apply_e0",
    "apply_e1",
    "build_action_slate",
    "build_official_hipporag_baseline",
    "canonical_bytes",
    "fit_e1",
    "lexical_tokens",
    "make_aform_example",
    "normalize_text",
    "normalized_query_payload",
    "normalized_query_sha256",
    "passage_from_public_fields",
    "passage_public_payload",
    "policy_signature",
    "query_structure",
    "serialize_passage_text",
    "stable_hash",
    "turn_from_public_fields",
    "turn_public_payload",
]

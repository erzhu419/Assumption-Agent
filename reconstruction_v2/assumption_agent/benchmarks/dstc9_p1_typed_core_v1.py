"""Source-free typed knowledge selection for DSTC9 Track 1 P1.

This module deliberately has no dataset loader, file, network, model, split,
dialogue-identity, domain, entity-id, document-id, family, response, qrel, or
outcome entrypoint.  Action formation accepts only:

* the complete public dialogue history through the target user turn;
* the exact public ``ordinal/entity_name/title/body`` knowledge projection;
* six already-quantized, integer, label-free score vectors; and
* a label-free predicted bucket.

Five fixed typed combiners produce complete candidate rankings.  E0 is a
sixth, frozen hierarchical cascade over those five rankings.  E1 may be
formed only from fixed-order integer utility vectors on A_form.  It learns a
small predicted-bucket-to-recipe program with zero-prior shrinkage; negative,
under-supported, non-positive, or tied evidence falls back to E0.  The frozen
program can be applied unchanged on A_hold and M_search, and its aggregate
behavior can be sealed without exposing item contents or utility values.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
import unicodedata
from typing import Mapping, Sequence


VERSION = "dstc9_p1_typed_core_v1"
STUDY_ID = "DSTC9_P1_HIERARCHICAL_KNOWLEDGE_EVALUATOR_L5_V1"
TOP_K = 5
SCALE = 1_000_000

MAX_HISTORY_TURNS = 256
MAX_TURN_CHARACTERS = 100_000
MAX_DIALOGUE_CHARACTERS = 1_000_000
MAX_ENTITY_NAME_CHARACTERS = 100_000
MAX_TITLE_CHARACTERS = 100_000
MAX_BODY_CHARACTERS = 2_000_000
MAX_CORPUS_SIZE = 100_000
MAX_ORDINAL = 10**12
MAX_SCORE_ABS = 10**15

PREDICTED_BUCKETS = (0, 1, 2, 3)
POLICY_STAGES = ("A_hold", "M_search")

MIN_BUCKET_SUPPORT = 3
MIN_POSITIVE_FRACTION_NUMERATOR = 3
MIN_POSITIVE_FRACTION_DENOMINATOR = 4
SHRINKAGE_PSEUDOCOUNT = 2

R1_GLOBAL_CONTEXT = "R1_GLOBAL_CONTEXT"
R2_LAST_TURN_ENTITY = "R2_LAST_TURN_ENTITY"
R3_TITLE_ANCHOR = "R3_TITLE_ANCHOR"
R4_BODY_EVIDENCE = "R4_BODY_EVIDENCE"
R5_MULTI_VIEW_DIVERSITY = "R5_MULTI_VIEW_DIVERSITY"
R6_HIERARCHICAL_CASCADE = "R6_HIERARCHICAL_CASCADE"

TYPED_RECIPE_IDS = (
    R1_GLOBAL_CONTEXT,
    R2_LAST_TURN_ENTITY,
    R3_TITLE_ANCHOR,
    R4_BODY_EVIDENCE,
    R5_MULTI_VIEW_DIVERSITY,
)
E0_RECIPE_ID = R6_HIERARCHICAL_CASCADE
RECIPE_IDS = TYPED_RECIPE_IDS + (E0_RECIPE_ID,)

PUBLIC_TURN_FIELDS = ("speaker", "text")
PUBLIC_SNIPPET_FIELDS = (
    "ordinal",
    "entity_name",
    "title",
    "body",
)
SPEAKERS = ("U", "S")

SCORE_NAMES = (
    "global_ce",
    "last_turn_ce",
    "minilm",
    "entity",
    "title",
    "body",
)

_RECIPE_WEIGHTS: dict[str, tuple[tuple[str, int], ...]] = {
    R1_GLOBAL_CONTEXT: (
        ("global_ce", 12),
        ("minilm", 4),
        ("body", 2),
        ("last_turn_ce", 1),
    ),
    R2_LAST_TURN_ENTITY: (
        ("last_turn_ce", 12),
        ("entity", 5),
        ("title", 2),
        ("global_ce", 1),
    ),
    R3_TITLE_ANCHOR: (
        ("title", 12),
        ("entity", 4),
        ("last_turn_ce", 2),
        ("minilm", 1),
    ),
    R4_BODY_EVIDENCE: (
        ("body", 12),
        ("minilm", 4),
        ("global_ce", 2),
        ("title", 1),
    ),
}

_E0_PRIMARY_BY_BUCKET = (
    R1_GLOBAL_CONTEXT,
    R2_LAST_TURN_ENTITY,
    R3_TITLE_ANCHOR,
    R4_BODY_EVIDENCE,
)

_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class Dstc9P1TypedCoreError(ValueError):
    """A public projection, frozen action, or evaluator contract drifted."""


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    """Encode strict, deterministic JSON for hashes and sealed artifacts."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Dstc9P1TypedCoreError("value is not canonical JSON") from exc
    return encoded + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _validate_characters(value: str, *, field: str) -> None:
    for character in value:
        category = unicodedata.category(character)
        if category.startswith("C") and not character.isspace():
            raise Dstc9P1TypedCoreError(
                f"{field} contains a forbidden control character"
            )


def normalize_text(
    value: object,
    *,
    field: str = "text",
    maximum_length: int = MAX_BODY_CHARACTERS,
    allow_empty: bool = False,
) -> str:
    """Apply NFKC and collapse every Unicode whitespace run to ASCII space."""

    if (
        type(maximum_length) is not int
        or maximum_length < 0
        or not isinstance(value, str)
        or len(value) > maximum_length
    ):
        raise Dstc9P1TypedCoreError(f"{field} is invalid")
    _validate_characters(value, field=field)
    normalized = unicodedata.normalize("NFKC", value)
    _validate_characters(normalized, field=field)
    normalized = " ".join(normalized.split())
    if len(normalized) > maximum_length:
        raise Dstc9P1TypedCoreError(f"{field} is too long after normalization")
    if not allow_empty and not normalized:
        raise Dstc9P1TypedCoreError(f"{field} is empty")
    return normalized


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    speaker: str
    text: str

    def __post_init__(self) -> None:
        if self.speaker not in SPEAKERS:
            raise Dstc9P1TypedCoreError("turn speaker must be U or S")
        canonical = normalize_text(
            self.text,
            field="turn text",
            maximum_length=MAX_TURN_CHARACTERS,
        )
        object.__setattr__(self, "text", canonical)


def turn_from_public_fields(value: object) -> DialogueTurn:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(PUBLIC_TURN_FIELDS)
    ):
        raise Dstc9P1TypedCoreError(
            "turn projection is not the exact public field set"
        )
    return DialogueTurn(
        speaker=value["speaker"],  # type: ignore[arg-type]
        text=value["text"],  # type: ignore[arg-type]
    )


def turn_public_payload(turn: DialogueTurn) -> dict[str, str]:
    if not isinstance(turn, DialogueTurn):
        raise Dstc9P1TypedCoreError("turn is not a DialogueTurn")
    return {"speaker": turn.speaker, "text": turn.text}


def _checked_history(
    history: Sequence[DialogueTurn],
) -> tuple[DialogueTurn, ...]:
    if (
        isinstance(history, (str, bytes))
        or not 1 <= len(history) <= MAX_HISTORY_TURNS
    ):
        raise Dstc9P1TypedCoreError("dialogue history is invalid")
    checked = tuple(history)
    if any(not isinstance(turn, DialogueTurn) for turn in checked):
        raise Dstc9P1TypedCoreError(
            "dialogue history contains a non-turn"
        )
    # Consecutive equal speakers are intentionally valid.  DSTC9's public
    # schema supplies speaker labels but does not promise strict alternation.
    if checked[0].speaker != "U" or checked[-1].speaker != "U":
        raise Dstc9P1TypedCoreError(
            "history must begin and end with a user turn"
        )
    if sum(len(turn.text) for turn in checked) > MAX_DIALOGUE_CHARACTERS:
        raise Dstc9P1TypedCoreError("dialogue history is too long")
    return checked


def serialize_model_query(history: Sequence[DialogueTurn]) -> str:
    """Serialize every public turn, with no truncation or label-bearing data."""

    checked = _checked_history(history)
    return "\n".join(
        f"{turn.speaker}: {turn.text}" for turn in checked
    )


def normalized_query_payload(
    history: Sequence[DialogueTurn],
) -> dict[str, object]:
    checked = _checked_history(history)
    return {
        "model_query": serialize_model_query(checked),
        "schema": f"{VERSION}_normalized_query",
        "turns": [turn_public_payload(turn) for turn in checked],
    }


def normalized_query_sha256(
    history: Sequence[DialogueTurn],
) -> str:
    return stable_hash(normalized_query_payload(history))


@dataclass(frozen=True, slots=True)
class KnowledgeSnippet:
    ordinal: int
    entity_name: str | None
    title: str
    body: str

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or not 0 <= self.ordinal <= MAX_ORDINAL
        ):
            raise Dstc9P1TypedCoreError("snippet ordinal is invalid")
        if self.entity_name is not None:
            entity_name = normalize_text(
                self.entity_name,
                field="entity name",
                maximum_length=MAX_ENTITY_NAME_CHARACTERS,
            )
            object.__setattr__(self, "entity_name", entity_name)
        title = normalize_text(
            self.title,
            field="snippet title",
            maximum_length=MAX_TITLE_CHARACTERS,
        )
        body = normalize_text(
            self.body,
            field="snippet body",
            maximum_length=MAX_BODY_CHARACTERS,
        )
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "body", body)


def snippet_from_public_fields(value: object) -> KnowledgeSnippet:
    if not isinstance(value, Mapping):
        raise Dstc9P1TypedCoreError("snippet projection is not an object")
    keys = set(value)
    required = {"ordinal", "title", "body"}
    allowed = set(PUBLIC_SNIPPET_FIELDS)
    if not required <= keys <= allowed:
        raise Dstc9P1TypedCoreError(
            "snippet projection is not the exact public field set"
        )
    return KnowledgeSnippet(
        ordinal=value["ordinal"],  # type: ignore[arg-type]
        entity_name=value.get("entity_name"),  # type: ignore[arg-type]
        title=value["title"],  # type: ignore[arg-type]
        body=value["body"],  # type: ignore[arg-type]
    )


def snippet_public_payload(
    snippet: KnowledgeSnippet,
) -> dict[str, object]:
    if not isinstance(snippet, KnowledgeSnippet):
        raise Dstc9P1TypedCoreError(
            "snippet is not a KnowledgeSnippet"
        )
    return {
        "body": snippet.body,
        "entity_name": snippet.entity_name,
        "ordinal": snippet.ordinal,
        "title": snippet.title,
    }


def serialize_passage(snippet: KnowledgeSnippet) -> str:
    """Return the frozen label-free passage text shared by every model arm."""

    if not isinstance(snippet, KnowledgeSnippet):
        raise Dstc9P1TypedCoreError(
            "snippet is not a KnowledgeSnippet"
        )
    fields_to_serialize = []
    if snippet.entity_name is not None:
        fields_to_serialize.append(f"ENTITY: {snippet.entity_name}")
    fields_to_serialize.extend(
        (f"TITLE: {snippet.title}", f"BODY: {snippet.body}")
    )
    return "\n".join(fields_to_serialize)


def serialize_passage_bytes(snippet: KnowledgeSnippet) -> bytes:
    return serialize_passage(snippet).encode("utf-8")


def _checked_predicted_bucket(value: object) -> int:
    if type(value) is not int or value not in PREDICTED_BUCKETS:
        raise Dstc9P1TypedCoreError("predicted bucket is invalid")
    return value


def _checked_action_inputs(
    snippets: Sequence[KnowledgeSnippet],
    score_vectors: Mapping[str, Sequence[int]],
) -> tuple[
    tuple[KnowledgeSnippet, ...],
    dict[str, tuple[int, ...]],
]:
    if (
        isinstance(snippets, (str, bytes))
        or not TOP_K <= len(snippets) <= MAX_CORPUS_SIZE
    ):
        raise Dstc9P1TypedCoreError("snippet corpus size is invalid")
    checked_snippets = tuple(snippets)
    if any(
        not isinstance(snippet, KnowledgeSnippet)
        for snippet in checked_snippets
    ):
        raise Dstc9P1TypedCoreError(
            "snippet corpus contains a non-snippet"
        )
    if (
        len({snippet.ordinal for snippet in checked_snippets})
        != len(checked_snippets)
    ):
        raise Dstc9P1TypedCoreError("snippet ordinals are duplicated")
    if set(score_vectors) != set(SCORE_NAMES):
        raise Dstc9P1TypedCoreError("score vector registry drifted")
    checked_vectors: dict[str, tuple[int, ...]] = {}
    for name in SCORE_NAMES:
        vector = score_vectors[name]
        if (
            isinstance(vector, (str, bytes))
            or len(vector) != len(checked_snippets)
        ):
            raise Dstc9P1TypedCoreError(
                f"{name} score vector width drifted"
            )
        values = tuple(vector)
        if any(
            type(score) is not int or abs(score) > MAX_SCORE_ABS
            for score in values
        ):
            raise Dstc9P1TypedCoreError(
                f"{name} scores are not bounded integers"
            )
        checked_vectors[name] = values

    permutation = tuple(
        sorted(
            range(len(checked_snippets)),
            key=lambda index: checked_snippets[index].ordinal,
        )
    )
    sorted_snippets = tuple(
        checked_snippets[index] for index in permutation
    )
    sorted_vectors = {
        name: tuple(checked_vectors[name][index] for index in permutation)
        for name in SCORE_NAMES
    }
    return sorted_snippets, sorted_vectors


def _rank(
    scores: Sequence[int],
    snippets: Sequence[KnowledgeSnippet],
) -> tuple[int, ...]:
    if len(scores) != len(snippets):
        raise Dstc9P1TypedCoreError("rank width drifted")
    return tuple(
        sorted(
            range(len(snippets)),
            key=lambda index: (-scores[index], snippets[index].ordinal),
        )
    )


def _rank_points(
    scores: Sequence[int],
    snippets: Sequence[KnowledgeSnippet],
) -> tuple[int, ...]:
    order = _rank(scores, snippets)
    result = [0] * len(snippets)
    for rank, index in enumerate(order):
        result[index] = (
            (len(snippets) - rank) * SCALE // len(snippets)
        )
    return tuple(result)


def _weighted_order(
    *,
    snippets: Sequence[KnowledgeSnippet],
    points: Mapping[str, Sequence[int]],
    weights: Sequence[tuple[str, int]],
) -> tuple[int, ...]:
    if any(
        name not in SCORE_NAMES or type(weight) is not int or weight <= 0
        for name, weight in weights
    ):
        raise Dstc9P1TypedCoreError("recipe weight registry drifted")
    fused = tuple(
        sum(weight * points[name][index] for name, weight in weights)
        for index in range(len(snippets))
    )
    return _rank(fused, snippets)


def _multi_view_order(
    *,
    view_orders: Mapping[str, Sequence[int]],
    predicted_bucket: int,
) -> tuple[int, ...]:
    view_names = list(SCORE_NAMES)
    shift = predicted_bucket % len(view_names)
    view_names = view_names[shift:] + view_names[:shift]
    selected: list[int] = []
    seen: set[int] = set()
    corpus_size = len(next(iter(view_orders.values())))
    for depth in range(corpus_size):
        for name in view_names:
            index = view_orders[name][depth]
            if index not in seen:
                selected.append(index)
                seen.add(index)
    if len(selected) != corpus_size:
        raise Dstc9P1TypedCoreError(
            "multi-view recipe did not totalize the corpus"
        )
    return tuple(selected)


def _take_next_unseen(
    order: Sequence[int],
    *,
    cursor: int,
    seen: set[int],
) -> tuple[int | None, int]:
    while cursor < len(order):
        index = order[cursor]
        cursor += 1
        if index not in seen:
            return index, cursor
    return None, cursor


def _hierarchical_cascade(
    *,
    typed_orders: Mapping[str, Sequence[int]],
    predicted_bucket: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    primary = _E0_PRIMARY_BY_BUCKET[predicted_bucket]
    remaining = [
        recipe_id
        for recipe_id in TYPED_RECIPE_IDS
        if recipe_id not in {primary, R5_MULTI_VIEW_DIVERSITY}
    ]
    shift = predicted_bucket % len(remaining)
    remaining = remaining[shift:] + remaining[:shift]
    tiers = [R5_MULTI_VIEW_DIVERSITY, *remaining]
    cursors = {recipe_id: 0 for recipe_id in TYPED_RECIPE_IDS}
    selected: list[int] = []
    traces: list[str] = []
    seen: set[int] = set()

    for primary_slot in range(2):
        chosen, cursors[primary] = _take_next_unseen(
            typed_orders[primary],
            cursor=cursors[primary],
            seen=seen,
        )
        if chosen is None:
            raise Dstc9P1TypedCoreError("E0 primary tier was exhausted")
        selected.append(chosen)
        seen.add(chosen)
        traces.append(
            f"primary:{primary}:slot{primary_slot}"
        )

    for recipe_id in tiers:
        chosen, cursors[recipe_id] = _take_next_unseen(
            typed_orders[recipe_id],
            cursor=cursors[recipe_id],
            seen=seen,
        )
        if chosen is not None:
            selected.append(chosen)
            seen.add(chosen)
            traces.append(f"tier:{recipe_id}")

    cycle = [primary, *tiers]
    while len(selected) < len(typed_orders[primary]):
        added = False
        for recipe_id in cycle:
            chosen, cursors[recipe_id] = _take_next_unseen(
                typed_orders[recipe_id],
                cursor=cursors[recipe_id],
                seen=seen,
            )
            if chosen is None:
                continue
            selected.append(chosen)
            seen.add(chosen)
            traces.append(f"fill:{recipe_id}")
            added = True
            if len(selected) == len(typed_orders[primary]):
                break
        if not added:
            raise Dstc9P1TypedCoreError(
                "E0 cascade did not totalize the corpus"
            )
    return tuple(selected), tuple(traces)


@dataclass(frozen=True, slots=True)
class RecipeAction:
    recipe_id: str
    ranked_ordinals: tuple[int, ...]
    top5_trace: tuple[str, ...]
    behavior_digest: str

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise Dstc9P1TypedCoreError("recipe id is not frozen")
        if (
            len(self.ranked_ordinals) < TOP_K
            or len(set(self.ranked_ordinals)) != len(self.ranked_ordinals)
            or any(
                type(ordinal) is not int or ordinal < 0
                for ordinal in self.ranked_ordinals
            )
        ):
            raise Dstc9P1TypedCoreError("recipe ranking is malformed")
        if (
            len(self.top5_trace) != TOP_K
            or any(
                not isinstance(value, str) or not value
                for value in self.top5_trace
            )
        ):
            raise Dstc9P1TypedCoreError("recipe trace is malformed")
        if _HEX_SHA256_RE.fullmatch(self.behavior_digest) is None:
            raise Dstc9P1TypedCoreError(
                "recipe behavior digest is malformed"
            )

    @property
    def top5_ordinals(self) -> tuple[int, ...]:
        return self.ranked_ordinals[:TOP_K]

    def payload(self) -> dict[str, object]:
        return {
            "behavior_digest": self.behavior_digest,
            "ranking_sha256": stable_hash(list(self.ranked_ordinals)),
            "recipe_id": self.recipe_id,
            "top5_ordinals": list(self.top5_ordinals),
            "top5_trace": list(self.top5_trace),
        }


def _make_action(
    *,
    recipe_id: str,
    order: Sequence[int],
    traces: Sequence[str],
    snippets: Sequence[KnowledgeSnippet],
    query_sha256: str,
    snippet_sha256: str,
    score_sha256: str,
    predicted_bucket: int,
) -> RecipeAction:
    if (
        len(order) != len(snippets)
        or len(set(order)) != len(order)
        or set(order) != set(range(len(snippets)))
        or len(traces) != len(order)
    ):
        raise Dstc9P1TypedCoreError(
            "recipe did not return a full candidate permutation"
        )
    ordinals = tuple(snippets[index].ordinal for index in order)
    top5_trace = tuple(traces[:TOP_K])
    behavior = stable_hash(
        {
            "predicted_bucket": predicted_bucket,
            "query_sha256": query_sha256,
            "ranked_ordinals": list(ordinals),
            "recipe_id": recipe_id,
            "score_sha256": score_sha256,
            "snippet_sha256": snippet_sha256,
            "top5_trace": list(top5_trace),
            "version": VERSION,
        }
    )
    return RecipeAction(
        recipe_id=recipe_id,
        ranked_ordinals=ordinals,
        top5_trace=top5_trace,
        behavior_digest=behavior,
    )


@dataclass(frozen=True, slots=True)
class ActionSlate:
    predicted_bucket: int
    normalized_query_sha256: str
    model_query_sha256: str
    snippet_projection_sha256: str
    passage_serialization_sha256: str
    score_bundle_sha256: str
    actions: tuple[RecipeAction, ...]

    def __post_init__(self) -> None:
        _checked_predicted_bucket(self.predicted_bucket)
        for digest in (
            self.normalized_query_sha256,
            self.model_query_sha256,
            self.snippet_projection_sha256,
            self.passage_serialization_sha256,
            self.score_bundle_sha256,
        ):
            if _HEX_SHA256_RE.fullmatch(digest) is None:
                raise Dstc9P1TypedCoreError("slate digest is malformed")
        if tuple(action.recipe_id for action in self.actions) != RECIPE_IDS:
            raise Dstc9P1TypedCoreError(
                "slate recipe order drifted"
            )
        universes = {
            frozenset(action.ranked_ordinals) for action in self.actions
        }
        widths = {len(action.ranked_ordinals) for action in self.actions}
        if len(universes) != 1 or len(widths) != 1:
            raise Dstc9P1TypedCoreError(
                "recipe candidate universes drifted"
            )

    def action(self, recipe_id: str) -> RecipeAction:
        if recipe_id not in RECIPE_IDS:
            raise Dstc9P1TypedCoreError("recipe id is not frozen")
        return self.actions[RECIPE_IDS.index(recipe_id)]

    def audit_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "actions": [action.payload() for action in self.actions],
            "e0_recipe_id": E0_RECIPE_ID,
            "label_bearing_action_inputs": False,
            "model_query_sha256": self.model_query_sha256,
            "normalized_query_sha256": self.normalized_query_sha256,
            "passage_serialization_sha256": (
                self.passage_serialization_sha256
            ),
            "policy_stages": list(POLICY_STAGES),
            "predicted_bucket": self.predicted_bucket,
            "public_snippet_fields": list(PUBLIC_SNIPPET_FIELDS),
            "public_turn_fields": list(PUBLIC_TURN_FIELDS),
            "recipe_ids": list(RECIPE_IDS),
            "score_bundle_sha256": self.score_bundle_sha256,
            "score_names": list(SCORE_NAMES),
            "snippet_projection_sha256": (
                self.snippet_projection_sha256
            ),
            "study_id": STUDY_ID,
            "typed_recipe_ids": list(TYPED_RECIPE_IDS),
            "version": VERSION,
        }
        return {**payload, "self_sha256": stable_hash(payload)}


def build_action_slate(
    history: Sequence[DialogueTurn],
    snippets: Sequence[KnowledgeSnippet],
    global_ce_scores: Sequence[int],
    last_turn_ce_scores: Sequence[int],
    minilm_scores: Sequence[int],
    entity_scores: Sequence[int],
    title_scores: Sequence[int],
    body_scores: Sequence[int],
    predicted_bucket: int,
) -> ActionSlate:
    """Build all frozen rankings without accepting any gold information."""

    checked_history = _checked_history(history)
    bucket = _checked_predicted_bucket(predicted_bucket)
    checked_snippets, score_vectors = _checked_action_inputs(
        snippets,
        {
            "body": body_scores,
            "entity": entity_scores,
            "global_ce": global_ce_scores,
            "last_turn_ce": last_turn_ce_scores,
            "minilm": minilm_scores,
            "title": title_scores,
        },
    )
    query_payload = normalized_query_payload(checked_history)
    query_sha = stable_hash(query_payload)
    model_query = serialize_model_query(checked_history)
    model_query_sha = hashlib.sha256(
        model_query.encode("utf-8")
    ).hexdigest()
    snippet_payload = [
        snippet_public_payload(snippet)
        for snippet in checked_snippets
    ]
    snippet_sha = stable_hash(snippet_payload)
    passage_sha = stable_hash(
        [
            hashlib.sha256(serialize_passage_bytes(snippet)).hexdigest()
            for snippet in checked_snippets
        ]
    )
    score_sha = stable_hash(
        {
            "ordinals": [
                snippet.ordinal for snippet in checked_snippets
            ],
            "scores": {
                name: list(score_vectors[name]) for name in SCORE_NAMES
            },
        }
    )

    view_orders = {
        name: _rank(score_vectors[name], checked_snippets)
        for name in SCORE_NAMES
    }
    points = {
        name: _rank_points(score_vectors[name], checked_snippets)
        for name in SCORE_NAMES
    }
    typed_orders: dict[str, tuple[int, ...]] = {
        recipe_id: _weighted_order(
            snippets=checked_snippets,
            points=points,
            weights=_RECIPE_WEIGHTS[recipe_id],
        )
        for recipe_id in _RECIPE_WEIGHTS
    }
    typed_orders[R5_MULTI_VIEW_DIVERSITY] = _multi_view_order(
        view_orders=view_orders,
        predicted_bucket=bucket,
    )
    if set(typed_orders) != set(TYPED_RECIPE_IDS):
        raise Dstc9P1TypedCoreError("typed recipe registry drifted")
    e0_order, e0_traces = _hierarchical_cascade(
        typed_orders=typed_orders,
        predicted_bucket=bucket,
    )

    orders: dict[str, tuple[int, ...]] = {
        **typed_orders,
        E0_RECIPE_ID: e0_order,
    }
    traces: dict[str, tuple[str, ...]] = {
        recipe_id: tuple(
            f"fused:{recipe_id}:rank{rank}"
            for rank in range(len(order))
        )
        for recipe_id, order in typed_orders.items()
    }
    traces[E0_RECIPE_ID] = e0_traces
    actions = tuple(
        _make_action(
            recipe_id=recipe_id,
            order=orders[recipe_id],
            traces=traces[recipe_id],
            snippets=checked_snippets,
            query_sha256=query_sha,
            snippet_sha256=snippet_sha,
            score_sha256=score_sha,
            predicted_bucket=bucket,
        )
        for recipe_id in RECIPE_IDS
    )
    return ActionSlate(
        predicted_bucket=bucket,
        normalized_query_sha256=query_sha,
        model_query_sha256=model_query_sha,
        snippet_projection_sha256=snippet_sha,
        passage_serialization_sha256=passage_sha,
        score_bundle_sha256=score_sha,
        actions=actions,
    )


@dataclass(frozen=True, slots=True)
class AFormExample:
    predicted_bucket: int
    utility_vector: tuple[int, ...]

    def __post_init__(self) -> None:
        _checked_predicted_bucket(self.predicted_bucket)
        if (
            len(self.utility_vector) != len(RECIPE_IDS)
            or any(
                type(value) is not int or not 0 <= value <= SCALE
                for value in self.utility_vector
            )
        ):
            raise Dstc9P1TypedCoreError(
                "A_form utility vector is invalid"
            )


def make_aform_example(
    slate: ActionSlate,
    utility_vector: Sequence[int],
) -> AFormExample:
    """Bind a fixed-order A_form utility vector after the slate is sealed."""

    if not isinstance(slate, ActionSlate):
        raise Dstc9P1TypedCoreError("slate is invalid")
    return AFormExample(
        predicted_bucket=slate.predicted_bucket,
        utility_vector=tuple(utility_vector),
    )


@dataclass(frozen=True, slots=True)
class RecipeEvidence:
    predicted_bucket: int
    recipe_id: str
    support_count: int
    positive_count: int
    minimum_delta: int
    total_delta: int
    shrunken_mean_delta: Fraction
    qualified: bool

    def __post_init__(self) -> None:
        _checked_predicted_bucket(self.predicted_bucket)
        if self.recipe_id not in TYPED_RECIPE_IDS:
            raise Dstc9P1TypedCoreError(
                "E1 evidence contains a non-typed recipe"
            )
        if (
            type(self.support_count) is not int
            or self.support_count < 0
            or type(self.positive_count) is not int
            or not 0 <= self.positive_count <= self.support_count
            or type(self.minimum_delta) is not int
            or type(self.total_delta) is not int
            or not isinstance(self.shrunken_mean_delta, Fraction)
        ):
            raise Dstc9P1TypedCoreError("E1 evidence is malformed")
        expected_shrinkage = Fraction(
            self.total_delta,
            self.support_count + SHRINKAGE_PSEUDOCOUNT,
        )
        if self.shrunken_mean_delta != expected_shrinkage:
            raise Dstc9P1TypedCoreError(
                "E1 shrinkage formula drifted"
            )
        expected_qualified = (
            self.support_count >= MIN_BUCKET_SUPPORT
            and self.positive_count
            * MIN_POSITIVE_FRACTION_DENOMINATOR
            >= self.support_count
            * MIN_POSITIVE_FRACTION_NUMERATOR
            and self.minimum_delta >= 0
            and self.shrunken_mean_delta > 0
        )
        if self.qualified != expected_qualified:
            raise Dstc9P1TypedCoreError(
                "E1 qualification rule drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "minimum_delta": self.minimum_delta,
            "positive_count": self.positive_count,
            "predicted_bucket": self.predicted_bucket,
            "qualified": self.qualified,
            "recipe_id": self.recipe_id,
            "shrunken_mean_delta": [
                self.shrunken_mean_delta.numerator,
                self.shrunken_mean_delta.denominator,
            ],
            "support_count": self.support_count,
            "total_delta": self.total_delta,
        }


@dataclass(frozen=True, slots=True)
class BucketRule:
    predicted_bucket: int
    selected_recipe_id: str
    fallback_reason: str
    evidence: tuple[RecipeEvidence, ...]

    def __post_init__(self) -> None:
        _checked_predicted_bucket(self.predicted_bucket)
        expected_identities = tuple(
            (self.predicted_bucket, recipe_id)
            for recipe_id in TYPED_RECIPE_IDS
        )
        identities = tuple(
            (row.predicted_bucket, row.recipe_id)
            for row in self.evidence
        )
        if identities != expected_identities:
            raise Dstc9P1TypedCoreError(
                "bucket evidence registry drifted"
            )
        qualified = [row for row in self.evidence if row.qualified]
        if not qualified:
            expected_recipe = E0_RECIPE_ID
            expected_reason = "no_qualified_recipe"
        else:
            maximum = max(
                row.shrunken_mean_delta for row in qualified
            )
            winners = [
                row for row in qualified
                if row.shrunken_mean_delta == maximum
            ]
            if len(winners) != 1:
                expected_recipe = E0_RECIPE_ID
                expected_reason = "tie_to_e0"
            else:
                expected_recipe = winners[0].recipe_id
                expected_reason = "selected"
        if (
            self.selected_recipe_id != expected_recipe
            or self.fallback_reason != expected_reason
        ):
            raise Dstc9P1TypedCoreError(
                "bucket selection rule drifted"
            )

    @property
    def support_count(self) -> int:
        return self.evidence[0].support_count

    def payload(self) -> dict[str, object]:
        return {
            "evidence": [row.payload() for row in self.evidence],
            "fallback_reason": self.fallback_reason,
            "predicted_bucket": self.predicted_bucket,
            "selected_recipe_id": self.selected_recipe_id,
            "support_count": self.support_count,
        }


@dataclass(frozen=True, slots=True)
class E1Program:
    rules: tuple[BucketRule, ...]
    training_item_count: int
    training_stage: str = "A_form"

    def __post_init__(self) -> None:
        if tuple(rule.predicted_bucket for rule in self.rules) != (
            PREDICTED_BUCKETS
        ):
            raise Dstc9P1TypedCoreError(
                "E1 program bucket registry drifted"
            )
        if (
            type(self.training_item_count) is not int
            or self.training_item_count < 0
            or self.training_stage != "A_form"
        ):
            raise Dstc9P1TypedCoreError("E1 program is malformed")
        if sum(rule.support_count for rule in self.rules) != (
            self.training_item_count
        ):
            raise Dstc9P1TypedCoreError(
                "E1 program support total drifted"
            )

    def rule(self, predicted_bucket: int) -> BucketRule:
        bucket = _checked_predicted_bucket(predicted_bucket)
        return self.rules[PREDICTED_BUCKETS.index(bucket)]

    def body_payload(self) -> dict[str, object]:
        return {
            "e0_recipe_id": E0_RECIPE_ID,
            "minimum_bucket_support": MIN_BUCKET_SUPPORT,
            "minimum_positive_fraction": [
                MIN_POSITIVE_FRACTION_NUMERATOR,
                MIN_POSITIVE_FRACTION_DENOMINATOR,
            ],
            "predicted_buckets": list(PREDICTED_BUCKETS),
            "recipe_ids": list(RECIPE_IDS),
            "rules": [rule.payload() for rule in self.rules],
            "schema": f"{VERSION}_E1_bucket_program",
            "shrinkage_pseudocount": SHRINKAGE_PSEUDOCOUNT,
            "training_item_count": self.training_item_count,
            "training_stage": self.training_stage,
            "typed_recipe_ids": list(TYPED_RECIPE_IDS),
            "version": VERSION,
        }

    @property
    def program_sha256(self) -> str:
        return stable_hash(self.body_payload())

    def payload(self) -> dict[str, object]:
        body = self.body_payload()
        return {**body, "self_sha256": stable_hash(body)}


def fit_e1(examples: Sequence[AFormExample]) -> E1Program:
    """Fit the immutable bucket program from A_form utilities only."""

    if isinstance(examples, (str, bytes)):
        raise Dstc9P1TypedCoreError("A_form examples are invalid")
    checked = tuple(examples)
    if any(not isinstance(row, AFormExample) for row in checked):
        raise Dstc9P1TypedCoreError(
            "A_form contains a non-example"
        )
    grouped: defaultdict[int, list[AFormExample]] = defaultdict(list)
    for row in checked:
        grouped[row.predicted_bucket].append(row)
    e0_index = RECIPE_IDS.index(E0_RECIPE_ID)
    rules: list[BucketRule] = []
    for bucket in PREDICTED_BUCKETS:
        rows = grouped[bucket]
        evidence: list[RecipeEvidence] = []
        for recipe_id in TYPED_RECIPE_IDS:
            recipe_index = RECIPE_IDS.index(recipe_id)
            deltas = [
                row.utility_vector[recipe_index]
                - row.utility_vector[e0_index]
                for row in rows
            ]
            support = len(rows)
            positive = sum(delta > 0 for delta in deltas)
            minimum = min(deltas) if deltas else 0
            total = sum(deltas)
            shrunken = Fraction(
                total,
                support + SHRINKAGE_PSEUDOCOUNT,
            )
            qualified = (
                support >= MIN_BUCKET_SUPPORT
                and positive * MIN_POSITIVE_FRACTION_DENOMINATOR
                >= support * MIN_POSITIVE_FRACTION_NUMERATOR
                and minimum >= 0
                and shrunken > 0
            )
            evidence.append(
                RecipeEvidence(
                    predicted_bucket=bucket,
                    recipe_id=recipe_id,
                    support_count=support,
                    positive_count=positive,
                    minimum_delta=minimum,
                    total_delta=total,
                    shrunken_mean_delta=shrunken,
                    qualified=qualified,
                )
            )
        qualified_rows = [row for row in evidence if row.qualified]
        if not qualified_rows:
            selected_recipe = E0_RECIPE_ID
            fallback_reason = "no_qualified_recipe"
        else:
            maximum = max(
                row.shrunken_mean_delta for row in qualified_rows
            )
            winners = [
                row for row in qualified_rows
                if row.shrunken_mean_delta == maximum
            ]
            if len(winners) == 1:
                selected_recipe = winners[0].recipe_id
                fallback_reason = "selected"
            else:
                selected_recipe = E0_RECIPE_ID
                fallback_reason = "tie_to_e0"
        rules.append(
            BucketRule(
                predicted_bucket=bucket,
                selected_recipe_id=selected_recipe,
                fallback_reason=fallback_reason,
                evidence=tuple(evidence),
            )
        )
    return E1Program(
        rules=tuple(rules),
        training_item_count=len(checked),
    )


_E0_PROGRAM_SHA256 = stable_hash(
    {
        "e0_primary_by_bucket": list(_E0_PRIMARY_BY_BUCKET),
        "e0_recipe_id": E0_RECIPE_ID,
        "predicted_buckets": list(PREDICTED_BUCKETS),
        "version": VERSION,
    }
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    evaluator_id: str
    stage: str
    predicted_bucket: int
    selected_recipe_id: str
    top5_ordinals: tuple[int, ...]
    fallback_to_e0: bool
    program_sha256: str
    action_behavior_digest: str
    decision_digest: str

    def __post_init__(self) -> None:
        if self.evaluator_id not in {"E0", "E1"}:
            raise Dstc9P1TypedCoreError("evaluator id is invalid")
        if self.stage not in POLICY_STAGES:
            raise Dstc9P1TypedCoreError("policy stage is invalid")
        _checked_predicted_bucket(self.predicted_bucket)
        if self.selected_recipe_id not in RECIPE_IDS:
            raise Dstc9P1TypedCoreError(
                "policy selected an unknown recipe"
            )
        if self.evaluator_id == "E0":
            if (
                self.selected_recipe_id != E0_RECIPE_ID
                or self.fallback_to_e0
                or self.program_sha256 != _E0_PROGRAM_SHA256
            ):
                raise Dstc9P1TypedCoreError(
                    "E0 policy decision drifted"
                )
        elif self.fallback_to_e0 != (
            self.selected_recipe_id == E0_RECIPE_ID
        ):
            raise Dstc9P1TypedCoreError(
                "E1 fallback flag drifted"
            )
        if (
            len(self.top5_ordinals) != TOP_K
            or len(set(self.top5_ordinals)) != TOP_K
        ):
            raise Dstc9P1TypedCoreError(
                "policy decision top5 is malformed"
            )
        for digest in (
            self.program_sha256,
            self.action_behavior_digest,
            self.decision_digest,
        ):
            if _HEX_SHA256_RE.fullmatch(digest) is None:
                raise Dstc9P1TypedCoreError(
                    "policy decision digest is malformed"
                )
        expected = stable_hash(
            {
                "action_behavior_digest": self.action_behavior_digest,
                "evaluator_id": self.evaluator_id,
                "fallback_to_e0": self.fallback_to_e0,
                "predicted_bucket": self.predicted_bucket,
                "program_sha256": self.program_sha256,
                "selected_recipe_id": self.selected_recipe_id,
                "stage": self.stage,
                "top5_ordinals": list(self.top5_ordinals),
                "version": VERSION,
            }
        )
        if self.decision_digest != expected:
            raise Dstc9P1TypedCoreError(
                "policy decision digest drifted"
            )


def _make_decision(
    *,
    evaluator_id: str,
    stage: str,
    slate: ActionSlate,
    selected_recipe_id: str,
    fallback_to_e0: bool,
    program_sha256: str,
) -> PolicyDecision:
    if stage not in POLICY_STAGES:
        raise Dstc9P1TypedCoreError("policy stage is invalid")
    action = slate.action(selected_recipe_id)
    payload = {
        "action_behavior_digest": action.behavior_digest,
        "evaluator_id": evaluator_id,
        "fallback_to_e0": fallback_to_e0,
        "predicted_bucket": slate.predicted_bucket,
        "program_sha256": program_sha256,
        "selected_recipe_id": selected_recipe_id,
        "stage": stage,
        "top5_ordinals": list(action.top5_ordinals),
        "version": VERSION,
    }
    return PolicyDecision(
        evaluator_id=evaluator_id,
        stage=stage,
        predicted_bucket=slate.predicted_bucket,
        selected_recipe_id=selected_recipe_id,
        top5_ordinals=action.top5_ordinals,
        fallback_to_e0=fallback_to_e0,
        program_sha256=program_sha256,
        action_behavior_digest=action.behavior_digest,
        decision_digest=stable_hash(payload),
    )


def apply_e0(slate: ActionSlate, *, stage: str) -> PolicyDecision:
    if not isinstance(slate, ActionSlate):
        raise Dstc9P1TypedCoreError("slate is invalid")
    return _make_decision(
        evaluator_id="E0",
        stage=stage,
        slate=slate,
        selected_recipe_id=E0_RECIPE_ID,
        fallback_to_e0=False,
        program_sha256=_E0_PROGRAM_SHA256,
    )


def apply_e1(
    program: E1Program,
    slate: ActionSlate,
    *,
    stage: str,
) -> PolicyDecision:
    if not isinstance(program, E1Program) or not isinstance(
        slate, ActionSlate
    ):
        raise Dstc9P1TypedCoreError("E1 application input is invalid")
    rule = program.rule(slate.predicted_bucket)
    return _make_decision(
        evaluator_id="E1",
        stage=stage,
        slate=slate,
        selected_recipe_id=rule.selected_recipe_id,
        fallback_to_e0=(
            rule.selected_recipe_id == E0_RECIPE_ID
        ),
        program_sha256=program.program_sha256,
    )


@dataclass(frozen=True, slots=True)
class BehaviorSummary:
    evaluator_id: str
    stage: str
    program_sha256: str
    item_count: int
    fallback_count: int
    bucket_recipe_counts: tuple[tuple[int, str, int], ...]
    decision_set_sha256: str

    def __post_init__(self) -> None:
        if self.evaluator_id != "E1" or self.stage not in POLICY_STAGES:
            raise Dstc9P1TypedCoreError(
                "behavior summary identity drifted"
            )
        if (
            _HEX_SHA256_RE.fullmatch(self.program_sha256) is None
            or _HEX_SHA256_RE.fullmatch(
                self.decision_set_sha256
            )
            is None
        ):
            raise Dstc9P1TypedCoreError(
                "behavior summary digest is malformed"
            )
        if (
            type(self.item_count) is not int
            or self.item_count <= 0
            or type(self.fallback_count) is not int
            or not 0 <= self.fallback_count <= self.item_count
        ):
            raise Dstc9P1TypedCoreError(
                "behavior summary counts are invalid"
            )
        identities = tuple(
            (bucket, recipe_id)
            for bucket, recipe_id, _count
            in self.bucket_recipe_counts
        )
        if identities != tuple(sorted(identities)):
            raise Dstc9P1TypedCoreError(
                "behavior count registry is not canonical"
            )
        if (
            any(
                bucket not in PREDICTED_BUCKETS
                or recipe_id not in RECIPE_IDS
                or type(count) is not int
                or count <= 0
                for bucket, recipe_id, count
                in self.bucket_recipe_counts
            )
            or sum(
                count for _bucket, _recipe_id, count
                in self.bucket_recipe_counts
            )
            != self.item_count
        ):
            raise Dstc9P1TypedCoreError(
                "behavior count values are invalid"
            )

    def body_payload(self) -> dict[str, object]:
        return {
            "bucket_recipe_counts": [
                {
                    "count": count,
                    "predicted_bucket": bucket,
                    "recipe_id": recipe_id,
                }
                for bucket, recipe_id, count
                in self.bucket_recipe_counts
            ],
            "decision_set_sha256": self.decision_set_sha256,
            "evaluator_id": self.evaluator_id,
            "fallback_count": self.fallback_count,
            "item_count": self.item_count,
            "program_sha256": self.program_sha256,
            "schema": f"{VERSION}_behavior_summary",
            "stage": self.stage,
            "version": VERSION,
        }

    def payload(self) -> dict[str, object]:
        body = self.body_payload()
        return {**body, "self_sha256": stable_hash(body)}


def summarize_e1_behavior(
    program: E1Program,
    decisions: Sequence[PolicyDecision],
    *,
    stage: str,
) -> BehaviorSummary:
    if not isinstance(program, E1Program) or stage not in POLICY_STAGES:
        raise Dstc9P1TypedCoreError(
            "behavior summary input is invalid"
        )
    if isinstance(decisions, (str, bytes)) or not decisions:
        raise Dstc9P1TypedCoreError("behavior decisions are invalid")
    checked = tuple(decisions)
    if any(
        not isinstance(decision, PolicyDecision)
        or decision.evaluator_id != "E1"
        or decision.stage != stage
        or decision.program_sha256 != program.program_sha256
        for decision in checked
    ):
        raise Dstc9P1TypedCoreError(
            "behavior decisions do not share the frozen program"
        )
    counts = Counter(
        (
            decision.predicted_bucket,
            decision.selected_recipe_id,
        )
        for decision in checked
    )
    count_rows = tuple(
        (bucket, recipe_id, count)
        for (bucket, recipe_id), count in sorted(counts.items())
    )
    return BehaviorSummary(
        evaluator_id="E1",
        stage=stage,
        program_sha256=program.program_sha256,
        item_count=len(checked),
        fallback_count=sum(
            decision.fallback_to_e0 for decision in checked
        ),
        bucket_recipe_counts=count_rows,
        decision_set_sha256=stable_hash(
            sorted(decision.decision_digest for decision in checked)
        ),
    )


__all__ = [
    "ActionSlate",
    "AFormExample",
    "BehaviorSummary",
    "BucketRule",
    "DialogueTurn",
    "Dstc9P1TypedCoreError",
    "E0_RECIPE_ID",
    "E1Program",
    "KnowledgeSnippet",
    "MAX_BODY_CHARACTERS",
    "MAX_DIALOGUE_CHARACTERS",
    "MAX_ENTITY_NAME_CHARACTERS",
    "MAX_HISTORY_TURNS",
    "MAX_TITLE_CHARACTERS",
    "MAX_TURN_CHARACTERS",
    "MIN_BUCKET_SUPPORT",
    "MIN_POSITIVE_FRACTION_DENOMINATOR",
    "MIN_POSITIVE_FRACTION_NUMERATOR",
    "POLICY_STAGES",
    "PREDICTED_BUCKETS",
    "PUBLIC_SNIPPET_FIELDS",
    "PUBLIC_TURN_FIELDS",
    "PolicyDecision",
    "R1_GLOBAL_CONTEXT",
    "R2_LAST_TURN_ENTITY",
    "R3_TITLE_ANCHOR",
    "R4_BODY_EVIDENCE",
    "R5_MULTI_VIEW_DIVERSITY",
    "R6_HIERARCHICAL_CASCADE",
    "RECIPE_IDS",
    "RecipeAction",
    "RecipeEvidence",
    "SCALE",
    "SCORE_NAMES",
    "SHRINKAGE_PSEUDOCOUNT",
    "STUDY_ID",
    "TOP_K",
    "TYPED_RECIPE_IDS",
    "VERSION",
    "apply_e0",
    "apply_e1",
    "build_action_slate",
    "canonical_bytes",
    "fit_e1",
    "make_aform_example",
    "normalize_text",
    "normalized_query_payload",
    "normalized_query_sha256",
    "serialize_model_query",
    "serialize_passage",
    "serialize_passage_bytes",
    "snippet_from_public_fields",
    "snippet_public_payload",
    "stable_hash",
    "summarize_e1_behavior",
    "turn_from_public_fields",
    "turn_public_payload",
]

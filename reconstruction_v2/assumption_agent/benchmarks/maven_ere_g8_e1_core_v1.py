"""Pure offline G8/E1 core for the derived MAVEN-ERE context study.

The action path accepts only a validated label-free item.  The late top-level
relation family is representable only by ``LabelledItem`` and the fitting or
utility functions.  No filesystem, source-reader, model-loading, network, or
private-capability code lives here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import struct
import unicodedata
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


VERSION = "maven_ere_g8_e1_core_v1"
DESIGN_COMMIT = "5b6232a927205e85ae78c9726d692819661ad3c2"
DESIGN_SELF_SHA256 = (
    "314a9804d32a3c3fb848e0100bc62bc693a468e8e3ac09c9baf018c7cfeee417"
)
Q6_SCALE = 1_000_000
TOP_K = 3
FRONTIER_SIZE = 16
QUERY_ATOM_CAP = 8
TERMINAL_CAP = 4
PATH_CAP = 4
RIDGE_LAMBDA = 1.0

FAMILY_ORDER = ("CAUSAL", "SUBEVENT", "TEMPORAL")
FAMILY_INDEX = {family: index for index, family in enumerate(FAMILY_ORDER)}

AUTHORITY_KIND_ORDER = (
    "QUERY_ANCHOR",
    "HEAD_TERMINAL",
    "TAIL_TERMINAL",
    "DIRECT_MENTION",
    "ENDPOINT_COREFERENCE",
    "CONTEXT_WINDOW",
    "GENERIC_RELATION_NEIGHBOR",
    "GENERIC_TWO_EDGE_PATH",
)
_AUTHORITY_KIND_RANK = {
    value: index for index, value in enumerate(AUTHORITY_KIND_ORDER)
}
_DELETABLE_KINDS = frozenset(
    {
        "DIRECT_MENTION",
        "ENDPOINT_COREFERENCE",
        "CONTEXT_WINDOW",
        "GENERIC_RELATION_NEIGHBOR",
        "GENERIC_TWO_EDGE_PATH",
    }
)

G8_FEATURE_ORDER = (
    "mean_neutral_query_sentence_similarity",
    "minimum_neutral_query_sentence_similarity",
    "head_mention_terminal_fraction",
    "tail_mention_terminal_fraction",
    "direct_head_tail_mention_fraction",
    "head_and_tail_set_coverage_indicator",
    "endpoint_coreference_terminal_fraction",
    "context_window_terminal_fraction",
    "generic_relation_neighbor_terminal_fraction",
    "generic_two_edge_path_terminal_fraction",
    "query_anchor_terminal_fraction",
    "authorization_kind_coverage_fraction",
    "classifier_top1_minus_top2_margin",
    "classifier_argmax_consensus_fraction",
    "mean_selected_max_family_NLI_score_in_millions",
    "negative_maximum_selected_pair_sentence_redundancy",
)

E1_FEATURE_ORDER = (
    "G8_generator_energy",
    "classifier_top1_minus_top2_margin",
    "minimum_selected_terminal_singleton_sufficiency",
    "mean_delete_one_label_free_coverage_drop",
    "minimum_delete_one_label_free_coverage_drop",
    "best_same_authority_substitute_one_label_free_coverage_drop",
    "head_and_tail_set_coverage_indicator",
    "generic_two_edge_path_indicator",
    "negative_maximum_selected_pair_sentence_redundancy",
)


class MavenEreG8E1Error(ValueError):
    """Fail-closed validation or deterministic core execution error."""


def _strict_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise MavenEreG8E1Error(f"{field} must be nonempty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MavenEreG8E1Error(f"{field} must be valid UTF-8 text") from exc
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise MavenEreG8E1Error(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MavenEreG8E1Error(f"{field} must be finite")
    return result


def _validated_vector(values: Sequence[object], *, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MavenEreG8E1Error(f"{field} must be a vector")
    result = tuple(
        _finite_float(value, field=f"{field}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise MavenEreG8E1Error(f"{field} must be nonempty")
    norm_squared = math.fsum(value * value for value in result)
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise MavenEreG8E1Error(f"{field} must have positive norm")
    return result


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MavenEreG8E1Error("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _float64_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype="<f8", order="C")
    if not np.isfinite(array).all():
        raise MavenEreG8E1Error("nonfinite float64 array")
    return array.tobytes(order="C")


def _normal_equation_hash(matrix: np.ndarray, target: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", int(matrix.shape[0]), int(matrix.shape[1])))
    digest.update(_float64_bytes(matrix))
    digest.update(struct.pack("<I", int(target.shape[0])))
    digest.update(_float64_bytes(target))
    return digest.hexdigest()


def _alias_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def canonical_aliases(values: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        value = _strict_text(raw, field=f"alias {index}")
        key = _alias_key(value)
        if not key:
            raise MavenEreG8E1Error("empty normalized alias")
        if key not in seen:
            result.append(value)
            seen.add(key)
    if not result:
        raise MavenEreG8E1Error("event must have an alias")
    return tuple(result)


@dataclass(frozen=True)
class Mention:
    surface: str
    sentence_ordinal: int


@dataclass(frozen=True)
class Event:
    event_id: int
    event_type: str
    mentions: tuple[Mention, ...]


@dataclass(frozen=True)
class GenericRelation:
    left_event: int
    right_event: int


@dataclass(frozen=True)
class ValidatedActionItem:
    sentences: tuple[str, ...]
    sentence_embeddings: tuple[tuple[float, ...], ...]
    events: tuple[Event, ...]
    head_event: int
    tail_event: int
    generic_relations: tuple[GenericRelation, ...]
    common_query: str
    query_embedding: tuple[float, ...]
    sentence_family_nli_scores: tuple[tuple[int, int, int], ...]

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


@dataclass(frozen=True)
class LabelledItem:
    item: ValidatedActionItem
    family: str


@dataclass(frozen=True)
class Witness:
    kind: str
    event_ordinals: tuple[int, ...]
    sentence_ordinals: tuple[int, ...]

    def sort_key(self) -> tuple[object, ...]:
        return (
            _AUTHORITY_KIND_RANK[self.kind],
            self.event_ordinals,
            self.sentence_ordinals,
        )


@dataclass(frozen=True)
class TerminalAuthorization:
    ordinal: int
    kinds: tuple[str, ...]
    witnesses: tuple[Witness, ...]


@dataclass(frozen=True)
class TypedActionSpace:
    item: ValidatedActionItem
    authorized_ordinals: tuple[int, ...]
    authorizations: tuple[TerminalAuthorization, ...]
    query_q6: tuple[int, ...]
    pair_similarity_q6: Mapping[tuple[int, int], int]
    deleted_witnesses: tuple[Witness, ...]

    def authorization_map(self) -> dict[int, TerminalAuthorization]:
        return {row.ordinal: row for row in self.authorizations}


@dataclass(frozen=True)
class G8ItemSufficientStatistics:
    set_count: int
    centered_xx: tuple[tuple[float, ...], ...]
    centered_xy: tuple[float, ...]
    centered_target_sha256: str


@dataclass(frozen=True)
class G8Model:
    weights: tuple[float, ...]
    normal_equation_sha256: str
    observation_weight_sha256: str
    centered_target_sha256: str
    coefficient_sha256: str
    fit_sha256: str
    item_count: int = 96
    set_observation_count: int = 0

    def __post_init__(self) -> None:
        if len(self.weights) != len(G8_FEATURE_ORDER):
            raise MavenEreG8E1Error("G8 weight dimension mismatch")
        if not all(math.isfinite(value) for value in self.weights):
            raise MavenEreG8E1Error("G8 weights must be finite")


@dataclass(frozen=True)
class FrontierEntry:
    ordinals: tuple[int, int, int]
    phi: tuple[float, ...]
    generator_energy: float


@dataclass(frozen=True)
class G8Frontier:
    entries: tuple[FrontierEntry, ...]

    @property
    def e0(self) -> FrontierEntry:
        return self.entries[0]


@dataclass(frozen=True)
class E1Model:
    weights: tuple[float, ...]
    feature_stds: tuple[float, ...]
    normal_equation_sha256: str
    observation_weight_sha256: str
    target_sha256: str
    coefficient_sha256: str
    fit_sha256: str
    item_count: int = 48
    oriented_pair_count: int = 11_520

    def __post_init__(self) -> None:
        if len(self.weights) != len(E1_FEATURE_ORDER):
            raise MavenEreG8E1Error("E1 weight dimension mismatch")
        if len(self.feature_stds) != len(E1_FEATURE_ORDER):
            raise MavenEreG8E1Error("E1 scale dimension mismatch")
        if not all(math.isfinite(value) for value in self.weights):
            raise MavenEreG8E1Error("E1 weights must be finite")
        if not all(math.isfinite(value) and value >= 0 for value in self.feature_stds):
            raise MavenEreG8E1Error("E1 scales must be finite nonnegative")


@dataclass(frozen=True)
class E1Selection:
    entry: FrontierEntry
    psi: tuple[float, ...]
    score: float


@dataclass(frozen=True)
class SignFlipResult:
    observed_sum: int
    nonzero_pair_count: int
    tail_count: int
    assignment_count: int
    p_value: Fraction


@dataclass(frozen=True)
class EdgeDeletionReceipt:
    witness: Witness
    e0_before: tuple[int, int, int]
    e0_after: tuple[int, int, int]
    e0_changed: bool
    e1_before: tuple[int, int, int] | None
    e1_after: tuple[int, int, int] | None
    e1_changed: bool | None


def serialize_common_query(head: Event, tail: Event) -> str:
    heads = canonical_aliases(tuple(mention.surface for mention in head.mentions))
    tails = canonical_aliases(tuple(mention.surface for mention in tail.mentions))
    head_type = _strict_text(head.event_type, field="head event type")
    tail_type = _strict_text(tail.event_type, field="tail event type")
    return (
        f"EVENT_A aliases: {' | '.join(heads)}\n"
        f"EVENT_A type: {head_type}\n"
        f"EVENT_B aliases: {' | '.join(tails)}\n"
        f"EVENT_B type: {tail_type}\n"
        "Question: What is the relationship between event A and event B?"
    )


def validate_action_item(
    *,
    sentences: Sequence[object],
    sentence_embeddings: Sequence[Sequence[object]],
    events: Sequence[Event],
    head_event: int,
    tail_event: int,
    generic_relations: Sequence[GenericRelation],
    common_query: str,
    query_embedding: Sequence[object],
    sentence_family_nli_scores: Sequence[Sequence[object]],
) -> ValidatedActionItem:
    """Construct the exact label-free item; no gold/family argument exists."""

    if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence):
        raise MavenEreG8E1Error("sentences must be a sequence")
    sentence_rows = tuple(
        _strict_text(value, field=f"sentence {index}")
        for index, value in enumerate(sentences)
    )
    if len(sentence_rows) < 6:
        raise MavenEreG8E1Error("item must contain at least six sentences")
    if len(sentence_embeddings) != len(sentence_rows):
        raise MavenEreG8E1Error("sentence embedding count mismatch")
    embedding_rows = tuple(
        _validated_vector(value, field=f"sentence embedding {index}")
        for index, value in enumerate(sentence_embeddings)
    )
    query_vector = _validated_vector(query_embedding, field="query embedding")
    dimension = len(query_vector)
    if any(len(row) != dimension for row in embedding_rows):
        raise MavenEreG8E1Error("embedding dimension mismatch")

    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise MavenEreG8E1Error("events must be a sequence")
    event_rows: list[Event] = []
    for expected, event in enumerate(events):
        if not isinstance(event, Event) or event.event_id != expected:
            raise MavenEreG8E1Error("event IDs must be contiguous source-order integers")
        event_type = _strict_text(event.event_type, field=f"event {expected} type")
        if not event.mentions:
            raise MavenEreG8E1Error("event must have mentions")
        mentions: list[Mention] = []
        for mention_index, mention in enumerate(event.mentions):
            if not isinstance(mention, Mention):
                raise MavenEreG8E1Error("invalid mention")
            surface = _strict_text(
                mention.surface,
                field=f"event {expected} mention {mention_index} surface",
            )
            ordinal = mention.sentence_ordinal
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise MavenEreG8E1Error("mention sentence ordinal must be integer")
            if not 0 <= ordinal < len(sentence_rows):
                raise MavenEreG8E1Error("mention sentence ordinal out of range")
            mentions.append(Mention(surface, ordinal))
        event_rows.append(Event(expected, event_type, tuple(mentions)))
    if (
        isinstance(head_event, bool)
        or not isinstance(head_event, int)
        or isinstance(tail_event, bool)
        or not isinstance(tail_event, int)
        or head_event == tail_event
        or not 0 <= head_event < len(event_rows)
        or not 0 <= tail_event < len(event_rows)
    ):
        raise MavenEreG8E1Error("query endpoint events are invalid")

    relation_rows: list[GenericRelation] = []
    seen_relations: set[tuple[int, int]] = set()
    query_pair = tuple(sorted((head_event, tail_event)))
    for relation in generic_relations:
        if not isinstance(relation, GenericRelation):
            raise MavenEreG8E1Error("invalid generic relation")
        left, right = relation.left_event, relation.right_event
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or left == right
            or not 0 <= left < len(event_rows)
            or not 0 <= right < len(event_rows)
        ):
            raise MavenEreG8E1Error("generic relation endpoint is invalid")
        canonical = tuple(sorted((left, right)))
        if canonical == query_pair:
            raise MavenEreG8E1Error("query endpoint relation leaked into action item")
        if canonical in seen_relations or (left, right) != canonical:
            raise MavenEreG8E1Error("generic relations must be unique canonical pairs")
        seen_relations.add(canonical)
        relation_rows.append(GenericRelation(*canonical))
    relation_rows.sort(key=lambda row: (row.left_event, row.right_event))

    query = _strict_text(common_query, field="common query")
    expected_query = serialize_common_query(event_rows[head_event], event_rows[tail_event])
    if query != expected_query:
        raise MavenEreG8E1Error("common query is not the frozen neutral projection")

    if len(sentence_family_nli_scores) != len(sentence_rows):
        raise MavenEreG8E1Error("NLI sentence row count mismatch")
    score_rows: list[tuple[int, int, int]] = []
    for row_index, raw_row in enumerate(sentence_family_nli_scores):
        if isinstance(raw_row, (str, bytes)) or len(raw_row) != len(FAMILY_ORDER):
            raise MavenEreG8E1Error("NLI family score row shape mismatch")
        parsed: list[int] = []
        for column, value in enumerate(raw_row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise MavenEreG8E1Error(
                    f"NLI score {row_index}/{column} must be integer"
                )
            if not -(2**63) <= value <= 2**63 - 1:
                raise MavenEreG8E1Error("NLI score outside int64")
            parsed.append(value)
        score_rows.append(tuple(parsed))  # type: ignore[arg-type]

    return ValidatedActionItem(
        sentences=sentence_rows,
        sentence_embeddings=embedding_rows,
        events=tuple(event_rows),
        head_event=head_event,
        tail_event=tail_event,
        generic_relations=tuple(relation_rows),
        common_query=query,
        query_embedding=query_vector,
        sentence_family_nli_scores=tuple(score_rows),
    )


def labelled_item(item: ValidatedActionItem, family: str) -> LabelledItem:
    if not isinstance(item, ValidatedActionItem):
        raise MavenEreG8E1Error("validated action item required")
    if family not in FAMILY_ORDER:
        raise MavenEreG8E1Error("unknown frozen family")
    return LabelledItem(item, family)


def q6_cosine(left: Sequence[object], right: Sequence[object]) -> int:
    a = _validated_vector(left, field="left cosine vector")
    b = _validated_vector(right, field="right cosine vector")
    if len(a) != len(b):
        raise MavenEreG8E1Error("cosine dimension mismatch")
    numerator = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(math.fsum(x * x for x in a) * math.fsum(x * x for x in b))
    value = max(-1.0, min(1.0, numerator / norm))
    return int(round(value * Q6_SCALE))


def _event_sentence_sets(item: ValidatedActionItem) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(mention.sentence_ordinal for mention in event.mentions)
        for event in item.events
    )


def _ranked_cap(values: Iterable[int], scores: Sequence[int], cap: int) -> tuple[int, ...]:
    unique = set(values)
    return tuple(sorted(unique, key=lambda ordinal: (-scores[ordinal], ordinal))[:cap])


def _witness(
    kind: str,
    event_ordinals: Sequence[int],
    sentence_ordinals: Sequence[int],
) -> Witness:
    if kind not in _DELETABLE_KINDS:
        raise MavenEreG8E1Error("invalid deletable witness kind")
    return Witness(kind, tuple(event_ordinals), tuple(sentence_ordinals))


def build_action_space(
    item: ValidatedActionItem,
    *,
    deleted_witnesses: Iterable[Witness] = (),
) -> TypedActionSpace:
    """Build the closed typed terminal grammar without labels or baselines."""

    if not isinstance(item, ValidatedActionItem):
        raise MavenEreG8E1Error("validated action item required")
    deleted = frozenset(deleted_witnesses)
    if any(row.kind not in _DELETABLE_KINDS for row in deleted):
        raise MavenEreG8E1Error("only typed witnesses can be deleted")
    query_q6 = tuple(
        q6_cosine(item.query_embedding, row) for row in item.sentence_embeddings
    )
    event_sentences = _event_sentence_sets(item)
    head_all = event_sentences[item.head_event]
    tail_all = event_sentences[item.tail_event]
    head = _ranked_cap(head_all, query_q6, TERMINAL_CAP)
    tail = _ranked_cap(tail_all, query_q6, TERMINAL_CAP)
    query_atoms = _ranked_cap(
        range(item.sentence_count), query_q6, min(QUERY_ATOM_CAP, item.sentence_count)
    )

    kinds: list[set[str]] = [set() for _ in item.sentences]
    witnesses: list[set[Witness]] = [set() for _ in item.sentences]

    def authorize(ordinal: int, kind: str, witness: Witness | None = None) -> None:
        if witness is not None and witness in deleted:
            return
        kinds[ordinal].add(kind)
        if witness is not None:
            witnesses[ordinal].add(witness)

    for ordinal in query_atoms:
        authorize(ordinal, "QUERY_ANCHOR")
    for ordinal in head:
        authorize(ordinal, "HEAD_TERMINAL")
    for ordinal in tail:
        authorize(ordinal, "TAIL_TERMINAL")
    for ordinal in sorted(set(head) & set(tail)):
        row = _witness(
            "DIRECT_MENTION",
            (item.head_event, item.tail_event),
            (ordinal,),
        )
        authorize(ordinal, "DIRECT_MENTION", row)

    for event_id, terminals in ((item.head_event, head), (item.tail_event, tail)):
        for left, right in itertools.combinations(sorted(terminals), 2):
            row = _witness("ENDPOINT_COREFERENCE", (event_id,), (left, right))
            authorize(left, "ENDPOINT_COREFERENCE", row)
            authorize(right, "ENDPOINT_COREFERENCE", row)

    for endpoint, terminals in ((item.head_event, head[:2]), (item.tail_event, tail[:2])):
        for terminal in terminals:
            for neighbor in (terminal - 1, terminal + 1):
                if not 0 <= neighbor < item.sentence_count:
                    continue
                row = _witness("CONTEXT_WINDOW", (endpoint,), (terminal, neighbor))
                authorize(neighbor, "CONTEXT_WINDOW", row)

    adjacency: list[set[int]] = [set() for _ in item.events]
    for relation in item.generic_relations:
        adjacency[relation.left_event].add(relation.right_event)
        adjacency[relation.right_event].add(relation.left_event)
    one_edge_events = (adjacency[item.head_event] | adjacency[item.tail_event]) - {
        item.head_event,
        item.tail_event,
    }
    one_edge_sentences = _ranked_cap(
        (
            sentence
            for event_id in sorted(one_edge_events)
            for sentence in event_sentences[event_id]
        ),
        query_q6,
        PATH_CAP,
    )
    for ordinal in one_edge_sentences:
        owners = tuple(
            event_id
            for event_id in sorted(one_edge_events)
            if ordinal in event_sentences[event_id]
        )
        for owner in owners:
            endpoint = (
                item.head_event
                if owner in adjacency[item.head_event]
                else item.tail_event
            )
            row = _witness(
                "GENERIC_RELATION_NEIGHBOR",
                (endpoint, owner),
                (ordinal,),
            )
            authorize(ordinal, "GENERIC_RELATION_NEIGHBOR", row)

    intermediates = (
        adjacency[item.head_event] & adjacency[item.tail_event]
    ) - {item.head_event, item.tail_event}
    bridge_sentences = _ranked_cap(
        (
            sentence
            for event_id in sorted(intermediates)
            for sentence in event_sentences[event_id]
        ),
        query_q6,
        PATH_CAP,
    )
    for ordinal in bridge_sentences:
        for event_id in sorted(intermediates):
            if ordinal not in event_sentences[event_id]:
                continue
            row = _witness(
                "GENERIC_TWO_EDGE_PATH",
                (item.head_event, event_id, item.tail_event),
                (ordinal,),
            )
            authorize(ordinal, "GENERIC_TWO_EDGE_PATH", row)

    authorized = tuple(index for index, row in enumerate(kinds) if row)
    if len(authorized) < QUERY_ATOM_CAP and item.sentence_count >= QUERY_ATOM_CAP:
        raise MavenEreG8E1Error("query anchor authorization drifted")
    if len(authorized) < TOP_K:
        raise MavenEreG8E1Error("fewer than three authorized terminals")
    rows = tuple(
        TerminalAuthorization(
            ordinal=ordinal,
            kinds=tuple(sorted(kinds[ordinal], key=_AUTHORITY_KIND_RANK.__getitem__)),
            witnesses=tuple(sorted(witnesses[ordinal], key=Witness.sort_key)),
        )
        for ordinal in authorized
    )
    pair_similarity_q6 = {
        (left, right): q6_cosine(
            item.sentence_embeddings[left], item.sentence_embeddings[right]
        )
        for left, right in itertools.combinations(authorized, 2)
    }
    return TypedActionSpace(
        item=item,
        authorized_ordinals=authorized,
        authorizations=rows,
        query_q6=query_q6,
        pair_similarity_q6=pair_similarity_q6,
        deleted_witnesses=tuple(sorted(deleted, key=Witness.sort_key)),
    )


def iter_authorized_set3(space: TypedActionSpace) -> Iterator[tuple[int, int, int]]:
    yield from itertools.combinations(space.authorized_ordinals, TOP_K)


def _validated_set3(
    space: TypedActionSpace, values: Sequence[object]
) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or len(values) != TOP_K:
        raise MavenEreG8E1Error("Set3 requires exactly three ordinals")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise MavenEreG8E1Error("Set3 ordinal must be integer")
        parsed.append(value)
    result = tuple(sorted(parsed))
    if len(set(result)) != TOP_K:
        raise MavenEreG8E1Error("Set3 ordinals must be unique")
    if any(value not in space.authorized_ordinals for value in result):
        raise MavenEreG8E1Error("Set3 contains unauthorized ordinal")
    return result  # type: ignore[return-value]


def _subset_family_scores(
    item: ValidatedActionItem, values: Sequence[object]
) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or not 1 <= len(values) <= TOP_K:
        raise MavenEreG8E1Error("classifier subset requires one through three ordinals")
    parsed: list[int] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < item.sentence_count
        ):
            raise MavenEreG8E1Error("classifier ordinal is invalid")
        parsed.append(value)
    if len(set(parsed)) != len(parsed):
        raise MavenEreG8E1Error("classifier ordinals must be unique")
    scores: list[int] = []
    for family_index in range(len(FAMILY_ORDER)):
        rows = [
            item.sentence_family_nli_scores[ordinal][family_index]
            for ordinal in parsed
        ]
        scores.append(4 * max(rows) + sum(rows))
    return tuple(scores)  # type: ignore[return-value]


def selected_set_family_scores(
    item: ValidatedActionItem, values: Sequence[object]
) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or len(values) != TOP_K:
        raise MavenEreG8E1Error("classifier requires exactly three ordinals")
    return _subset_family_scores(item, values)


def predict_family(item: ValidatedActionItem, values: Sequence[object]) -> str:
    scores = selected_set_family_scores(item, values)
    return FAMILY_ORDER[max(range(len(scores)), key=lambda index: (scores[index], -index))]


def utility(values: Sequence[object], family: str, item: ValidatedActionItem) -> int:
    if family not in FAMILY_ORDER:
        raise MavenEreG8E1Error("unknown frozen family")
    return int(predict_family(item, values) == family)


def _classifier_margin(item: ValidatedActionItem, selected: Sequence[int]) -> float:
    scores = sorted(_subset_family_scores(item, selected), reverse=True)
    return (scores[0] - scores[1]) / Q6_SCALE


def _sentence_argmax_family(row: Sequence[int]) -> int:
    return max(range(len(FAMILY_ORDER)), key=lambda index: (row[index], -index))


def phi_features(
    space: TypedActionSpace, values: Sequence[object]
) -> tuple[float, ...]:
    selected = _validated_set3(space, values)
    item = space.item
    authorization = space.authorization_map()
    event_sentences = _event_sentence_sets(item)
    head = event_sentences[item.head_event]
    tail = event_sentences[item.tail_event]
    query_scores = [space.query_q6[index] / Q6_SCALE for index in selected]
    union_kinds = {
        kind for ordinal in selected for kind in authorization[ordinal].kinds
    }
    pair_redundancies = [
        space.pair_similarity_q6[(left, right)] / Q6_SCALE
        for left, right in itertools.combinations(selected, 2)
    ]
    argmaxes = [
        _sentence_argmax_family(item.sentence_family_nli_scores[ordinal])
        for ordinal in selected
    ]
    consensus = max(Counter(argmaxes).values()) / 3.0
    mean_max_nli = math.fsum(
        max(item.sentence_family_nli_scores[ordinal]) / Q6_SCALE
        for ordinal in selected
    ) / 3.0
    result = (
        math.fsum(query_scores) / 3.0,
        min(query_scores),
        sum(ordinal in head for ordinal in selected) / 3.0,
        sum(ordinal in tail for ordinal in selected) / 3.0,
        sum(ordinal in head and ordinal in tail for ordinal in selected) / 3.0,
        float(
            any(ordinal in head for ordinal in selected)
            and any(ordinal in tail for ordinal in selected)
        ),
        sum(
            "ENDPOINT_COREFERENCE" in authorization[ordinal].kinds
            for ordinal in selected
        )
        / 3.0,
        sum("CONTEXT_WINDOW" in authorization[ordinal].kinds for ordinal in selected)
        / 3.0,
        sum(
            "GENERIC_RELATION_NEIGHBOR" in authorization[ordinal].kinds
            for ordinal in selected
        )
        / 3.0,
        sum(
            "GENERIC_TWO_EDGE_PATH" in authorization[ordinal].kinds
            for ordinal in selected
        )
        / 3.0,
        sum("QUERY_ANCHOR" in authorization[ordinal].kinds for ordinal in selected)
        / 3.0,
        len(union_kinds) / len(AUTHORITY_KIND_ORDER),
        _classifier_margin(item, selected),
        consensus,
        mean_max_nli,
        -max(pair_redundancies),
    )
    if len(result) != len(G8_FEATURE_ORDER) or not all(
        math.isfinite(value) for value in result
    ):
        raise MavenEreG8E1Error("invalid G8 feature vector")
    return result


def action_item_commitment(item: ValidatedActionItem) -> str:
    return stable_hash(
        {
            "query_sha256": hashlib.sha256(item.common_query.encode("utf-8")).hexdigest(),
            "sentence_sha256": [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in item.sentences
            ],
            "event_sentence_rows": [
                [mention.sentence_ordinal for mention in event.mentions]
                for event in item.events
            ],
            "head_event": item.head_event,
            "tail_event": item.tail_event,
            "generic_relations": [
                [row.left_event, row.right_event] for row in item.generic_relations
            ],
            "nli_score_sha256": stable_hash(item.sentence_family_nli_scores),
        }
    )


def _ordered_labelled_items(
    examples: Sequence[LabelledItem], *, per_family: int
) -> tuple[LabelledItem, ...]:
    if len(examples) != per_family * len(FAMILY_ORDER):
        raise MavenEreG8E1Error("formal labelled item count mismatch")
    by_family: dict[str, list[tuple[str, LabelledItem]]] = {
        family: [] for family in FAMILY_ORDER
    }
    seen: set[str] = set()
    for example in examples:
        if not isinstance(example, LabelledItem) or example.family not in by_family:
            raise MavenEreG8E1Error("invalid labelled item")
        commitment = action_item_commitment(example.item)
        if commitment in seen:
            raise MavenEreG8E1Error("duplicate action item commitment")
        seen.add(commitment)
        by_family[example.family].append((commitment, example))
    ordered: list[LabelledItem] = []
    for family in FAMILY_ORDER:
        rows = sorted(by_family[family], key=lambda row: row[0])
        if len(rows) != per_family:
            raise MavenEreG8E1Error("formal family count mismatch")
        ordered.extend(example for _, example in rows)
    return tuple(ordered)


def g8_item_sufficient_statistics(example: LabelledItem) -> G8ItemSufficientStatistics:
    space = build_action_space(example.item)
    dimension = len(G8_FEATURE_ORDER)
    feature_sum = np.zeros(dimension, dtype=np.float64)
    target_sum = 0.0
    set_count = 0
    for selected in iter_authorized_set3(space):
        feature_sum += np.asarray(phi_features(space, selected), dtype=np.float64)
        target_sum += utility(selected, example.family, example.item)
        set_count += 1
    if set_count < FRONTIER_SIZE:
        raise MavenEreG8E1Error("complete Set3 space has fewer than sixteen sets")
    mean_phi = feature_sum / float(set_count)
    mean_target = target_sum / float(set_count)
    centered_xx = np.zeros((dimension, dimension), dtype=np.float64)
    centered_xy = np.zeros(dimension, dtype=np.float64)
    target_digest = hashlib.sha256()
    for selected in iter_authorized_set3(space):
        centered_phi = np.asarray(phi_features(space, selected), dtype=np.float64) - mean_phi
        centered_target = (
            utility(selected, example.family, example.item) - mean_target
        )
        centered_xx += np.outer(centered_phi, centered_phi)
        centered_xy += centered_phi * centered_target
        target_digest.update(struct.pack("<d", centered_target))
    return G8ItemSufficientStatistics(
        set_count=set_count,
        centered_xx=tuple(tuple(float(value) for value in row) for row in centered_xx),
        centered_xy=tuple(float(value) for value in centered_xy),
        centered_target_sha256=target_digest.hexdigest(),
    )


def fit_g8(examples: Sequence[LabelledItem]) -> G8Model:
    ordered = _ordered_labelled_items(examples, per_family=32)
    dimension = len(G8_FEATURE_ORDER)
    matrix = np.eye(dimension, dtype=np.float64) * RIDGE_LAMBDA
    target = np.zeros(dimension, dtype=np.float64)
    weight_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    total_observations = 0
    for example in ordered:
        stats = g8_item_sufficient_statistics(example)
        set_weight = (1.0 / 96.0) / float(stats.set_count)
        matrix += set_weight * np.asarray(stats.centered_xx, dtype=np.float64)
        target += set_weight * np.asarray(stats.centered_xy, dtype=np.float64)
        commitment = action_item_commitment(example.item)
        weight_rows.append(
            {
                "item": commitment,
                "set_count": stats.set_count,
                "set_weight_hex": set_weight.hex(),
            }
        )
        target_rows.append(
            {"item": commitment, "target_sha256": stats.centered_target_sha256}
        )
        total_observations += stats.set_count
    try:
        coefficients = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError as exc:
        raise MavenEreG8E1Error("G8 normal equation solve failed") from exc
    if not np.isfinite(coefficients).all():
        raise MavenEreG8E1Error("nonfinite G8 coefficients")
    normal_hash = _normal_equation_hash(matrix, target)
    coefficient_hash = hashlib.sha256(_float64_bytes(coefficients)).hexdigest()
    weight_hash = stable_hash(weight_rows)
    target_hash = stable_hash(target_rows)
    fit_hash = stable_hash(
        {
            "coefficient_sha256": coefficient_hash,
            "normal_equation_sha256": normal_hash,
            "observation_weight_sha256": weight_hash,
            "centered_target_sha256": target_hash,
            "feature_order": G8_FEATURE_ORDER,
            "lambda": RIDGE_LAMBDA,
        }
    )
    return G8Model(
        weights=tuple(float(value) for value in coefficients),
        normal_equation_sha256=normal_hash,
        observation_weight_sha256=weight_hash,
        centered_target_sha256=target_hash,
        coefficient_sha256=coefficient_hash,
        fit_sha256=fit_hash,
        set_observation_count=total_observations,
    )


def g8_energy(model: G8Model, phi: Sequence[object]) -> float:
    if len(phi) != len(G8_FEATURE_ORDER):
        raise MavenEreG8E1Error("G8 feature dimension mismatch")
    result = math.fsum(
        weight * _finite_float(value, field=f"G8 feature {index}")
        for index, (weight, value) in enumerate(zip(model.weights, phi, strict=True))
    )
    if not math.isfinite(result):
        raise MavenEreG8E1Error("nonfinite G8 energy")
    return result


def _frontier_key(row: FrontierEntry) -> tuple[float, tuple[int, int, int]]:
    return (-row.generator_energy, row.ordinals)


def g8_frontier(
    item: ValidatedActionItem,
    model: G8Model,
    *,
    space: TypedActionSpace | None = None,
) -> G8Frontier:
    action_space = build_action_space(item) if space is None else space
    retained: list[FrontierEntry] = []
    for selected in iter_authorized_set3(action_space):
        phi = phi_features(action_space, selected)
        row = FrontierEntry(selected, phi, g8_energy(model, phi))
        if len(retained) < FRONTIER_SIZE:
            retained.append(row)
            continue
        worst = max(range(len(retained)), key=lambda index: _frontier_key(retained[index]))
        if _frontier_key(row) < _frontier_key(retained[worst]):
            retained[worst] = row
    retained.sort(key=_frontier_key)
    if len(retained) != FRONTIER_SIZE:
        raise MavenEreG8E1Error("fewer than sixteen frontier sets")
    return G8Frontier(tuple(retained))


def _label_free_coverage(space: TypedActionSpace, selected: Sequence[int]) -> float:
    if not selected:
        raise MavenEreG8E1Error("coverage requires a nonempty set")
    item = space.item
    event_sentences = _event_sentence_sets(item)
    head = event_sentences[item.head_event]
    tail = event_sentences[item.tail_event]
    authorization = space.authorization_map()
    query_max = max((space.query_q6[value] / Q6_SCALE + 1.0) / 2.0 for value in selected)
    return (
        query_max
        + float(any(value in head for value in selected))
        + float(any(value in tail for value in selected))
        + float(
            any(
                "GENERIC_TWO_EDGE_PATH" in authorization[value].kinds
                for value in selected
            )
        )
        + min(1.0, _classifier_margin(item, selected) / 8.0)
    ) / 5.0


def psi_features(space: TypedActionSpace, entry: FrontierEntry) -> tuple[float, ...]:
    selected = _validated_set3(space, entry.ordinals)
    authorization = space.authorization_map()
    item = space.item
    event_sentences = _event_sentence_sets(item)
    head = event_sentences[item.head_event]
    tail = event_sentences[item.tail_event]
    singleton: list[float] = []
    for ordinal in selected:
        kinds = authorization[ordinal].kinds
        singleton.append(
            (
                (space.query_q6[ordinal] / Q6_SCALE + 1.0) / 2.0
                + float(ordinal in head)
                + float(ordinal in tail)
                + float("QUERY_ANCHOR" in kinds)
                + float("GENERIC_TWO_EDGE_PATH" in kinds)
            )
            / 5.0
        )
    full = _label_free_coverage(space, selected)
    deletion_drops = [
        full - _label_free_coverage(space, tuple(value for value in selected if value != removed))
        for removed in selected
    ]
    selected_set = set(selected)
    substitute_coverages: list[float] = []
    for removed in selected:
        kinds = set(authorization[removed].kinds)
        for candidate in space.authorized_ordinals:
            if candidate in selected_set:
                continue
            if not kinds.intersection(authorization[candidate].kinds):
                continue
            replacement = tuple(sorted((selected_set - {removed}) | {candidate}))
            substitute_coverages.append(_label_free_coverage(space, replacement))
    substitute_drop = full - max(substitute_coverages) if substitute_coverages else 0.0
    phi = entry.phi
    result = (
        entry.generator_energy,
        _classifier_margin(item, selected),
        min(singleton),
        math.fsum(deletion_drops) / 3.0,
        min(deletion_drops),
        substitute_drop,
        phi[G8_FEATURE_ORDER.index("head_and_tail_set_coverage_indicator")],
        float(
            phi[G8_FEATURE_ORDER.index("generic_two_edge_path_terminal_fraction")] > 0
        ),
        phi[
            G8_FEATURE_ORDER.index(
                "negative_maximum_selected_pair_sentence_redundancy"
            )
        ],
    )
    if len(result) != len(E1_FEATURE_ORDER) or not all(
        math.isfinite(value) for value in result
    ):
        raise MavenEreG8E1Error("invalid E1 feature vector")
    return result


def _solve_pairwise_ridge(
    differences: Sequence[Sequence[float]], targets: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...], str, str, str, str]:
    if len(differences) != len(targets) or not differences:
        raise MavenEreG8E1Error("pairwise rows are invalid")
    rows = np.asarray(differences, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if rows.shape != (len(differences), len(E1_FEATURE_ORDER)):
        raise MavenEreG8E1Error("pairwise feature shape mismatch")
    weight = 1.0 / len(rows)
    means = rows.sum(axis=0, dtype=np.float64) * weight
    if not np.all(np.abs(means) <= 1e-15):
        raise MavenEreG8E1Error("oriented pair rows are not antisymmetric")
    stds = np.sqrt((rows * rows).sum(axis=0, dtype=np.float64) * weight)
    standardized = np.zeros_like(rows)
    nonzero = stds > 0
    standardized[:, nonzero] = rows[:, nonzero] / stds[nonzero]
    matrix = np.eye(len(E1_FEATURE_ORDER), dtype=np.float64) * RIDGE_LAMBDA
    target = np.zeros(len(E1_FEATURE_ORDER), dtype=np.float64)
    for row, value in zip(standardized, y, strict=True):
        matrix += weight * np.outer(row, row)
        target += weight * row * value
    try:
        coefficients = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError as exc:
        raise MavenEreG8E1Error("E1 normal equation solve failed") from exc
    return (
        tuple(float(value) for value in coefficients),
        tuple(float(value) for value in stds),
        _normal_equation_hash(matrix, target),
        stable_hash({"row_count": len(rows), "row_weight_hex": weight.hex()}),
        hashlib.sha256(_float64_bytes(y)).hexdigest(),
        hashlib.sha256(_float64_bytes(coefficients)).hexdigest(),
    )


def fit_e1(examples: Sequence[LabelledItem], g8_model: G8Model) -> E1Model:
    ordered = _ordered_labelled_items(examples, per_family=16)
    differences: list[tuple[float, ...]] = []
    targets: list[float] = []
    frontier_rows: list[dict[str, object]] = []
    for example in ordered:
        space = build_action_space(example.item)
        frontier = g8_frontier(example.item, g8_model, space=space)
        psi_rows = tuple(psi_features(space, row) for row in frontier.entries)
        utility_rows = tuple(
            utility(row.ordinals, example.family, example.item)
            for row in frontier.entries
        )
        for left in range(FRONTIER_SIZE):
            for right in range(left + 1, FRONTIER_SIZE):
                difference = tuple(
                    psi_rows[left][index] - psi_rows[right][index]
                    for index in range(len(E1_FEATURE_ORDER))
                )
                target = float(utility_rows[left] - utility_rows[right])
                differences.extend((difference, tuple(-value for value in difference)))
                targets.extend((target, -target))
        frontier_rows.append(
            {
                "item": action_item_commitment(example.item),
                "frontier": [row.ordinals for row in frontier.entries],
                "psi_sha256": hashlib.sha256(
                    _float64_bytes(np.asarray(psi_rows, dtype=np.float64))
                ).hexdigest(),
            }
        )
    if len(differences) != 11_520:
        raise MavenEreG8E1Error("E1 oriented pair count mismatch")
    weights, stds, normal, observation, target_hash, coefficient = _solve_pairwise_ridge(
        differences, targets
    )
    fit_hash = stable_hash(
        {
            "coefficient_sha256": coefficient,
            "normal_equation_sha256": normal,
            "observation_weight_sha256": observation,
            "target_sha256": target_hash,
            "feature_stds_hex": [value.hex() for value in stds],
            "feature_order": E1_FEATURE_ORDER,
            "frontiers": frontier_rows,
            "lambda": RIDGE_LAMBDA,
        }
    )
    return E1Model(
        weights=weights,
        feature_stds=stds,
        normal_equation_sha256=normal,
        observation_weight_sha256=observation,
        target_sha256=target_hash,
        coefficient_sha256=coefficient,
        fit_sha256=fit_hash,
    )


def e1_score(model: E1Model, psi: Sequence[object]) -> float:
    if len(psi) != len(E1_FEATURE_ORDER):
        raise MavenEreG8E1Error("E1 feature dimension mismatch")
    standardized = [
        0.0
        if model.feature_stds[index] == 0
        else _finite_float(value, field=f"E1 feature {index}")
        / model.feature_stds[index]
        for index, value in enumerate(psi)
    ]
    result = math.fsum(
        weight * value
        for weight, value in zip(model.weights, standardized, strict=True)
    )
    if not math.isfinite(result):
        raise MavenEreG8E1Error("nonfinite E1 score")
    return result


def e1_select(space: TypedActionSpace, frontier: G8Frontier, model: E1Model) -> E1Selection:
    rows = [
        E1Selection(entry, psi_features(space, entry), e1_score(model, psi_features(space, entry)))
        for entry in frontier.entries
    ]
    rows.sort(
        key=lambda row: (-row.score, -row.entry.generator_energy, row.entry.ordinals)
    )
    return rows[0]


def raw3(item: ValidatedActionItem) -> tuple[int, int, int]:
    rows = [
        (q6_cosine(item.query_embedding, embedding), ordinal)
        for ordinal, embedding in enumerate(item.sentence_embeddings)
    ]
    rows.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(ordinal for _, ordinal in rows[:TOP_K])
    return result  # type: ignore[return-value]


def exact_sign_flip(deltas: Sequence[object]) -> SignFlipResult:
    parsed: list[int] = []
    for value in deltas:
        if isinstance(value, bool) or not isinstance(value, int) or value not in {-1, 0, 1}:
            raise MavenEreG8E1Error("binary utility delta must be -1, 0, or 1")
        parsed.append(value)
    observed = sum(parsed)
    magnitudes = [1 for value in parsed if value != 0]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        next_distribution: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            next_distribution[subtotal + magnitude] += count
            next_distribution[subtotal - magnitude] += count
        distribution = next_distribution
    assignment_count = 1 << len(magnitudes)
    tail_count = sum(count for total, count in distribution.items() if total >= observed)
    return SignFlipResult(
        observed_sum=observed,
        nonzero_pair_count=len(magnitudes),
        tail_count=tail_count,
        assignment_count=assignment_count,
        p_value=Fraction(tail_count, assignment_count),
    )


def behavior_hash(
    item: ValidatedActionItem,
    space: TypedActionSpace,
    frontier: G8Frontier,
    selected: Sequence[object],
) -> str:
    selected_set = _validated_set3(space, selected)
    witnesses = sorted(
        {
            witness
            for authorization in space.authorizations
            for witness in authorization.witnesses
        },
        key=Witness.sort_key,
    )
    return stable_hash(
        {
            "item": action_item_commitment(item),
            "authorized": space.authorized_ordinals,
            "witnesses": [
                {
                    "kind": row.kind,
                    "event_ordinals": row.event_ordinals,
                    "sentence_ordinals": row.sentence_ordinals,
                }
                for row in witnesses
            ],
            "frontier": [
                {"ordinals": row.ordinals, "energy_hex": row.generator_energy.hex()}
                for row in frontier.entries
            ],
            "selected": selected_set,
        }
    )


def edge_deletion_redecode(
    item: ValidatedActionItem,
    g8_model: G8Model,
    *,
    e1_model: E1Model | None = None,
) -> tuple[EdgeDeletionReceipt, ...]:
    base_space = build_action_space(item)
    base_frontier = g8_frontier(item, g8_model, space=base_space)
    e0_before = base_frontier.e0.ordinals
    e1_before = (
        e1_select(base_space, base_frontier, e1_model).entry.ordinals
        if e1_model is not None
        else None
    )
    selected = set(e0_before)
    if e1_before is not None:
        selected.update(e1_before)
    witnesses = sorted(
        {
            witness
            for authorization in base_space.authorizations
            if authorization.ordinal in selected
            for witness in authorization.witnesses
        },
        key=Witness.sort_key,
    )
    receipts: list[EdgeDeletionReceipt] = []
    for witness in witnesses:
        changed_space = build_action_space(item, deleted_witnesses=(witness,))
        changed_frontier = g8_frontier(item, g8_model, space=changed_space)
        e0_after = changed_frontier.e0.ordinals
        e1_after = (
            e1_select(changed_space, changed_frontier, e1_model).entry.ordinals
            if e1_model is not None
            else None
        )
        receipts.append(
            EdgeDeletionReceipt(
                witness=witness,
                e0_before=e0_before,
                e0_after=e0_after,
                e0_changed=e0_before != e0_after,
                e1_before=e1_before,
                e1_after=e1_after,
                e1_changed=(e1_before != e1_after) if e1_model is not None else None,
            )
        )
    return tuple(receipts)


def g8_model_payload(model: G8Model) -> dict[str, object]:
    return {
        "schema": "maven_ere_G8_model_v1",
        "weights_hex": [value.hex() for value in model.weights],
        "normal_equation_sha256": model.normal_equation_sha256,
        "observation_weight_sha256": model.observation_weight_sha256,
        "centered_target_sha256": model.centered_target_sha256,
        "coefficient_sha256": model.coefficient_sha256,
        "fit_sha256": model.fit_sha256,
        "item_count": model.item_count,
        "set_observation_count": model.set_observation_count,
    }


def e1_model_payload(model: E1Model) -> dict[str, object]:
    return {
        "schema": "maven_ere_E1_model_v1",
        "weights_hex": [value.hex() for value in model.weights],
        "feature_stds_hex": [value.hex() for value in model.feature_stds],
        "normal_equation_sha256": model.normal_equation_sha256,
        "observation_weight_sha256": model.observation_weight_sha256,
        "target_sha256": model.target_sha256,
        "coefficient_sha256": model.coefficient_sha256,
        "fit_sha256": model.fit_sha256,
        "item_count": model.item_count,
        "oriented_pair_count": model.oriented_pair_count,
    }


__all__ = [
    "AUTHORITY_KIND_ORDER",
    "DESIGN_COMMIT",
    "DESIGN_SELF_SHA256",
    "E1Model",
    "E1Selection",
    "E1_FEATURE_ORDER",
    "EdgeDeletionReceipt",
    "Event",
    "FAMILY_ORDER",
    "FRONTIER_SIZE",
    "FrontierEntry",
    "G8Frontier",
    "G8Model",
    "G8_FEATURE_ORDER",
    "GenericRelation",
    "LabelledItem",
    "MavenEreG8E1Error",
    "Mention",
    "SignFlipResult",
    "TOP_K",
    "TerminalAuthorization",
    "TypedActionSpace",
    "ValidatedActionItem",
    "action_item_commitment",
    "behavior_hash",
    "build_action_space",
    "canonical_aliases",
    "e1_model_payload",
    "e1_score",
    "e1_select",
    "edge_deletion_redecode",
    "exact_sign_flip",
    "fit_e1",
    "fit_g8",
    "g8_energy",
    "g8_frontier",
    "g8_model_payload",
    "iter_authorized_set3",
    "labelled_item",
    "phi_features",
    "predict_family",
    "psi_features",
    "q6_cosine",
    "raw3",
    "selected_set_family_scores",
    "serialize_common_query",
    "stable_hash",
    "utility",
    "validate_action_item",
]

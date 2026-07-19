"""Pure, offline DocRED G8/E1 structured-set decoder core.

This module deliberately has no filesystem, network, model-loading, source-reader,
or private-capability code.  Action construction accepts only a validated,
label-free item.  Gold evidence enters only the explicitly labelled fitting and
utility APIs.

The implementation follows the design frozen at commit 8fda3578 (design self
hash 67bbba4dc0cf62928e28f97f96cd757249400f95abddec3b3ec2f753053f3345).
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


VERSION = "docred_structured_set_decoder_g8_e1_v1"
DESIGN_COMMIT = "8fda35782ecf10d2c0f0045049d9944abf0c8c32"
DESIGN_SELF_SHA256 = (
    "67bbba4dc0cf62928e28f97f96cd757249400f95abddec3b3ec2f753053f3345"
)
Q6_SCALE = 1_000_000
TOP_K = 3
FRONTIER_SIZE = 16
QUERY_ATOM_COUNT = 8
RIDGE_LAMBDA = 1.0
E1_DEPLOYMENT_FORMULA = (
    "sum_beta_std_times_Psi_div_sigma_pair_zero_variance_zero_no_intercept"
)

FAMILY_ORDER = (
    "GEO_SOVEREIGNTY",
    "MEMBERSHIP_STRUCTURE",
    "PERSON_CREATIVE_LIFE",
)

AUTHORITY_KIND_ORDER = (
    "QUERY",
    "HEAD",
    "TAIL",
    "DIRECT",
    "COREFERENCE",
    "BRIDGE",
)
_AUTHORITY_KIND_RANK = {
    value: index for index, value in enumerate(AUTHORITY_KIND_ORDER)
}

G8_FEATURE_ORDER = (
    "mean_full_query_sentence_similarity",
    "minimum_full_query_sentence_similarity",
    "mean_relation_description_sentence_similarity",
    "head_mention_terminal_fraction",
    "tail_mention_terminal_fraction",
    "direct_head_tail_terminal_fraction",
    "head_and_tail_set_coverage_indicator",
    "one_bridge_witness_pair_fraction",
    "shared_entity_connected_pair_fraction",
    "query_atom_terminal_fraction",
    "authorization_kind_coverage_fraction",
    "negative_maximum_selected_pair_sentence_redundancy",
)

E1_FEATURE_ORDER = (
    "G8_generator_energy",
    "minimum_selected_terminal_singleton_sufficiency",
    "mean_delete_one_query_coverage_drop",
    "minimum_delete_one_query_coverage_drop",
    "best_same_authority_substitute_one_query_coverage_drop",
    "head_and_tail_set_coverage_indicator",
    "one_bridge_witness_indicator",
    "negative_maximum_selected_pair_sentence_redundancy",
)


class DocredStructuredSetDecoderError(ValueError):
    """Fail-closed validation or deterministic execution error."""


def _strict_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise DocredStructuredSetDecoderError(f"{field} must be nonempty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise DocredStructuredSetDecoderError(
            f"{field} must be valid UTF-8 text"
        ) from exc
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise DocredStructuredSetDecoderError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DocredStructuredSetDecoderError(f"{field} must be finite")
    return result


def _validated_vector(values: Sequence[object], *, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DocredStructuredSetDecoderError(f"{field} must be a vector")
    result = tuple(
        _finite_float(value, field=f"{field}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise DocredStructuredSetDecoderError(f"{field} must be nonempty")
    norm_squared = math.fsum(value * value for value in result)
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise DocredStructuredSetDecoderError(f"{field} must have positive norm")
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _float64_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype="<f8", order="C")
    if not np.isfinite(array).all():
        raise DocredStructuredSetDecoderError("nonfinite float64 array")
    return array.tobytes(order="C")


def _normal_equation_hash(matrix: np.ndarray, target: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", int(matrix.shape[0]), int(matrix.shape[1])))
    digest.update(_float64_bytes(matrix))
    digest.update(struct.pack("<I", int(target.shape[0])))
    digest.update(_float64_bytes(target))
    return digest.hexdigest()


@dataclass(frozen=True)
class Mention:
    name: str
    sentence_ordinal: int
    entity_type: str


@dataclass(frozen=True)
class Entity:
    entity_id: int
    mentions: tuple[Mention, ...]


@dataclass(frozen=True)
class ValidatedActionItem:
    """Label-free item accepted by every action-generation function."""

    sentences: tuple[str, ...]
    sentence_embeddings: tuple[tuple[float, ...], ...]
    entities: tuple[Entity, ...]
    head_entity: int
    tail_entity: int
    relation_description: str
    common_query: str
    full_query_embedding: tuple[float, ...]
    relation_description_embedding: tuple[float, ...]

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @property
    def embedding_dimension(self) -> int:
        return len(self.full_query_embedding)


@dataclass(frozen=True)
class GoldEvidence:
    ordinals: tuple[int, ...]


@dataclass(frozen=True)
class LabelledItem:
    item: ValidatedActionItem
    gold: GoldEvidence
    family: str


@dataclass(frozen=True)
class Witness:
    kind: str
    entity_id: int | None
    left_sentence: int
    right_sentence: int

    def sort_key(self) -> tuple[int, int, int, int]:
        return (
            _AUTHORITY_KIND_RANK[self.kind],
            -1 if self.entity_id is None else self.entity_id,
            self.left_sentence,
            self.right_sentence,
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
    full_query_q6: tuple[int, ...]
    relation_description_q6: tuple[int, ...]
    deleted_witnesses: tuple[Witness, ...]

    def authorization_map(self) -> dict[int, TerminalAuthorization]:
        return {row.ordinal: row for row in self.authorizations}


@dataclass(frozen=True)
class G8ItemSufficientStatistics:
    set_count: int
    mean_phi: tuple[float, ...]
    mean_target: float
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
            raise DocredStructuredSetDecoderError("G8 weight dimension mismatch")
        if not all(math.isfinite(value) for value in self.weights):
            raise DocredStructuredSetDecoderError("G8 weights must be finite")


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
class PairwiseRidgeSolution:
    weights: tuple[float, ...]
    feature_stds: tuple[float, ...]
    normal_equation_sha256: str
    observation_weight_sha256: str
    target_sha256: str
    coefficient_sha256: str


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
            raise DocredStructuredSetDecoderError("E1 weight dimension mismatch")
        if len(self.feature_stds) != len(E1_FEATURE_ORDER):
            raise DocredStructuredSetDecoderError("E1 scale dimension mismatch")
        if not all(math.isfinite(value) for value in self.weights):
            raise DocredStructuredSetDecoderError("E1 weights must be finite")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.feature_stds):
            raise DocredStructuredSetDecoderError("E1 scales must be finite nonnegative")


@dataclass(frozen=True)
class E1Selection:
    entry: FrontierEntry
    psi: tuple[float, ...]
    score: float


@dataclass(frozen=True)
class SignFlipResult:
    observed_sum_x6: int
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


def render_sentence(tokens: Sequence[object]) -> str:
    """Render one official token sequence using exactly one ASCII space."""

    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise DocredStructuredSetDecoderError("sentence tokens must be a sequence")
    rendered: list[str] = []
    for index, token in enumerate(tokens):
        value = _strict_text(token, field=f"sentence token {index}")
        rendered.append(value)
    if not rendered:
        raise DocredStructuredSetDecoderError("sentence must contain a token")
    return " ".join(rendered)


def _alias_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    collapsed = " ".join(normalized.split())
    return collapsed.casefold()


def canonical_aliases(mention_names: Sequence[object]) -> tuple[str, ...]:
    """Deduplicate by NFKC/whitespace/casefold while retaining first surfaces."""

    result: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(mention_names):
        value = _strict_text(raw, field=f"mention name {index}")
        key = _alias_key(value)
        if not key:
            raise DocredStructuredSetDecoderError("empty normalized alias")
        if key not in seen:
            result.append(value)
            seen.add(key)
    if not result:
        raise DocredStructuredSetDecoderError("entity must have an alias")
    return tuple(result)


def serialize_common_query(
    head_aliases: Sequence[str],
    relation_description: str,
    tail_aliases: Sequence[str],
) -> str:
    relation = _strict_text(relation_description, field="relation description")
    heads = canonical_aliases(tuple(head_aliases))
    tails = canonical_aliases(tuple(tail_aliases))
    return (
        f"HEAD: {' | '.join(heads)}\n"
        f"RELATION: {relation}\n"
        f"TAIL: {' | '.join(tails)}"
    )


def validate_action_item(
    *,
    sentence_tokens: Sequence[Sequence[object]],
    sentence_embeddings: Sequence[Sequence[object]],
    entities: Sequence[Entity],
    head_entity: int,
    tail_entity: int,
    relation_description: str,
    full_query_embedding: Sequence[object],
    relation_description_embedding: Sequence[object],
) -> ValidatedActionItem:
    """Validate and construct an item without accepting any gold argument."""

    if isinstance(sentence_tokens, (str, bytes)) or not isinstance(
        sentence_tokens, Sequence
    ):
        raise DocredStructuredSetDecoderError("sentence_tokens must be a sequence")
    sentences = tuple(render_sentence(tokens) for tokens in sentence_tokens)
    if len(sentences) < 10:
        raise DocredStructuredSetDecoderError("item must have at least ten sentences")
    if len(sentence_embeddings) != len(sentences):
        raise DocredStructuredSetDecoderError("sentence embedding count mismatch")
    embedded = tuple(
        _validated_vector(vector, field=f"sentence embedding {index}")
        for index, vector in enumerate(sentence_embeddings)
    )
    full_vector = _validated_vector(full_query_embedding, field="full query embedding")
    relation_vector = _validated_vector(
        relation_description_embedding, field="relation description embedding"
    )
    dimension = len(full_vector)
    if len(relation_vector) != dimension or any(
        len(vector) != dimension for vector in embedded
    ):
        raise DocredStructuredSetDecoderError("embedding dimension mismatch")

    if isinstance(entities, (str, bytes)) or not isinstance(entities, Sequence):
        raise DocredStructuredSetDecoderError("entities must be a sequence")
    validated_entities: list[Entity] = []
    for expected_id, entity in enumerate(entities):
        if not isinstance(entity, Entity) or entity.entity_id != expected_id:
            raise DocredStructuredSetDecoderError(
                "entity IDs must be contiguous source-order integers"
            )
        if not entity.mentions:
            raise DocredStructuredSetDecoderError("entity must have mentions")
        mentions: list[Mention] = []
        for mention_index, mention in enumerate(entity.mentions):
            if not isinstance(mention, Mention):
                raise DocredStructuredSetDecoderError("invalid mention type")
            name = _strict_text(
                mention.name,
                field=f"entity {expected_id} mention {mention_index} name",
            )
            entity_type = _strict_text(
                mention.entity_type,
                field=f"entity {expected_id} mention {mention_index} type",
            )
            ordinal = mention.sentence_ordinal
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise DocredStructuredSetDecoderError("mention ordinal must be integer")
            if ordinal < 0 or ordinal >= len(sentences):
                raise DocredStructuredSetDecoderError("mention ordinal out of range")
            mentions.append(Mention(name, ordinal, entity_type))
        validated_entities.append(Entity(expected_id, tuple(mentions)))

    if isinstance(head_entity, bool) or not isinstance(head_entity, int):
        raise DocredStructuredSetDecoderError("head entity must be integer")
    if isinstance(tail_entity, bool) or not isinstance(tail_entity, int):
        raise DocredStructuredSetDecoderError("tail entity must be integer")
    if head_entity == tail_entity:
        raise DocredStructuredSetDecoderError("head and tail entities must differ")
    if not (0 <= head_entity < len(validated_entities)) or not (
        0 <= tail_entity < len(validated_entities)
    ):
        raise DocredStructuredSetDecoderError("head or tail entity out of range")

    relation = _strict_text(relation_description, field="relation description")
    head_aliases = canonical_aliases(
        tuple(mention.name for mention in validated_entities[head_entity].mentions)
    )
    tail_aliases = canonical_aliases(
        tuple(mention.name for mention in validated_entities[tail_entity].mentions)
    )
    query = serialize_common_query(head_aliases, relation, tail_aliases)
    return ValidatedActionItem(
        sentences=sentences,
        sentence_embeddings=embedded,
        entities=tuple(validated_entities),
        head_entity=head_entity,
        tail_entity=tail_entity,
        relation_description=relation,
        common_query=query,
        full_query_embedding=full_vector,
        relation_description_embedding=relation_vector,
    )


def validate_gold(item: ValidatedActionItem, ordinals: Sequence[object]) -> GoldEvidence:
    """Label-dependent validation kept separate from action-item validation."""

    if isinstance(ordinals, (str, bytes)) or not isinstance(ordinals, Sequence):
        raise DocredStructuredSetDecoderError("gold ordinals must be a sequence")
    parsed: list[int] = []
    for value in ordinals:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DocredStructuredSetDecoderError("gold ordinal must be integer")
        if value < 0 or value >= item.sentence_count:
            raise DocredStructuredSetDecoderError("gold ordinal out of range")
        parsed.append(value)
    unique = tuple(sorted(set(parsed)))
    if len(unique) != len(parsed):
        raise DocredStructuredSetDecoderError("gold ordinals must be unique")
    if not 1 <= len(unique) <= 3:
        raise DocredStructuredSetDecoderError("gold size must be one through three")
    return GoldEvidence(unique)


def labelled_item(
    item: ValidatedActionItem,
    gold_ordinals: Sequence[object],
    family: str,
) -> LabelledItem:
    if family not in FAMILY_ORDER:
        raise DocredStructuredSetDecoderError("unknown frozen family")
    return LabelledItem(item, validate_gold(item, gold_ordinals), family)


def quantize_cosine_value(value: object) -> int:
    cosine = _finite_float(value, field="cosine")
    if cosine < -1.000000000001 or cosine > 1.000000000001:
        raise DocredStructuredSetDecoderError("cosine outside [-1, 1]")
    clipped = min(1.0, max(-1.0, cosine))
    return int(round(clipped * Q6_SCALE))


def q6_cosine(left: Sequence[object], right: Sequence[object]) -> int:
    a = _validated_vector(left, field="left cosine vector")
    b = _validated_vector(right, field="right cosine vector")
    if len(a) != len(b):
        raise DocredStructuredSetDecoderError("cosine dimension mismatch")
    numerator = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    return quantize_cosine_value(numerator / (norm_a * norm_b))


def _entity_sentence_sets(item: ValidatedActionItem) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(mention.sentence_ordinal for mention in entity.mentions)
        for entity in item.entities
    )


def _sentence_entity_sets(item: ValidatedActionItem) -> tuple[frozenset[int], ...]:
    rows: list[set[int]] = [set() for _ in item.sentences]
    for entity in item.entities:
        for mention in entity.mentions:
            rows[mention.sentence_ordinal].add(entity.entity_id)
    return tuple(frozenset(row) for row in rows)


def _witness(
    kind: str,
    entity_id: int | None,
    left_sentence: int,
    right_sentence: int,
) -> Witness:
    if kind not in {"DIRECT", "COREFERENCE", "BRIDGE"}:
        raise DocredStructuredSetDecoderError("invalid deletable witness kind")
    return Witness(kind, entity_id, left_sentence, right_sentence)


def build_action_space(
    item: ValidatedActionItem,
    *,
    deleted_witnesses: Iterable[Witness] = (),
) -> TypedActionSpace:
    """Build the closed typed grammar without accepting labels or baselines."""

    if not isinstance(item, ValidatedActionItem):
        raise DocredStructuredSetDecoderError("validated action item required")
    deleted = frozenset(deleted_witnesses)
    if any(witness.kind not in {"DIRECT", "COREFERENCE", "BRIDGE"} for witness in deleted):
        raise DocredStructuredSetDecoderError("only typed edges can be deleted")

    full_q6 = tuple(
        q6_cosine(item.full_query_embedding, embedding)
        for embedding in item.sentence_embeddings
    )
    relation_q6 = tuple(
        q6_cosine(item.relation_description_embedding, embedding)
        for embedding in item.sentence_embeddings
    )
    query_atoms = frozenset(
        sorted(
            range(item.sentence_count),
            key=lambda ordinal: (-relation_q6[ordinal], ordinal),
        )[:QUERY_ATOM_COUNT]
    )
    entity_sentences = _entity_sentence_sets(item)
    head_sentences = entity_sentences[item.head_entity]
    tail_sentences = entity_sentences[item.tail_entity]

    kinds: list[set[str]] = [set() for _ in item.sentences]
    witnesses: list[set[Witness]] = [set() for _ in item.sentences]
    for ordinal in query_atoms:
        kinds[ordinal].add("QUERY")
    for ordinal in head_sentences:
        kinds[ordinal].add("HEAD")
    for ordinal in tail_sentences:
        kinds[ordinal].add("TAIL")

    for ordinal in sorted(head_sentences & tail_sentences):
        row = _witness("DIRECT", None, ordinal, ordinal)
        if row not in deleted:
            kinds[ordinal].add("DIRECT")
            witnesses[ordinal].add(row)

    for entity_id, sentence_set in enumerate(entity_sentences):
        for left, right in itertools.combinations(sorted(sentence_set), 2):
            row = _witness("COREFERENCE", entity_id, left, right)
            if row in deleted:
                continue
            kinds[left].add("COREFERENCE")
            kinds[right].add("COREFERENCE")
            witnesses[left].add(row)
            witnesses[right].add(row)

    for entity_id, sentence_set in enumerate(entity_sentences):
        if entity_id in {item.head_entity, item.tail_entity}:
            continue
        head_sides = sorted(head_sentences & sentence_set)
        tail_sides = sorted(tail_sentences & sentence_set)
        for head_sentence in head_sides:
            for tail_sentence in tail_sides:
                if head_sentence == tail_sentence:
                    continue
                row = _witness(
                    "BRIDGE", entity_id, head_sentence, tail_sentence
                )
                if row in deleted:
                    continue
                kinds[head_sentence].add("BRIDGE")
                kinds[tail_sentence].add("BRIDGE")
                witnesses[head_sentence].add(row)
                witnesses[tail_sentence].add(row)

    authorized = tuple(index for index, row in enumerate(kinds) if row)
    if len(authorized) < TOP_K:
        raise DocredStructuredSetDecoderError("fewer than three authorized terminals")
    authorizations = tuple(
        TerminalAuthorization(
            ordinal=ordinal,
            kinds=tuple(
                sorted(kinds[ordinal], key=_AUTHORITY_KIND_RANK.__getitem__)
            ),
            witnesses=tuple(sorted(witnesses[ordinal], key=Witness.sort_key)),
        )
        for ordinal in authorized
    )
    return TypedActionSpace(
        item=item,
        authorized_ordinals=authorized,
        authorizations=authorizations,
        full_query_q6=full_q6,
        relation_description_q6=relation_q6,
        deleted_witnesses=tuple(sorted(deleted, key=Witness.sort_key)),
    )


def iter_authorized_set3(space: TypedActionSpace) -> Iterator[tuple[int, int, int]]:
    """Yield the complete authorized set space in lexicographic order."""

    yield from itertools.combinations(space.authorized_ordinals, TOP_K)


def _validated_set3(
    space: TypedActionSpace, ordinals: Sequence[object]
) -> tuple[int, int, int]:
    if isinstance(ordinals, (str, bytes)) or len(ordinals) != TOP_K:
        raise DocredStructuredSetDecoderError("Set3 requires exactly three ordinals")
    parsed: list[int] = []
    for value in ordinals:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DocredStructuredSetDecoderError("Set3 ordinal must be integer")
        parsed.append(value)
    canonical = tuple(sorted(parsed))
    if len(set(canonical)) != TOP_K:
        raise DocredStructuredSetDecoderError("Set3 ordinals must be unique")
    if any(value not in space.authorized_ordinals for value in canonical):
        raise DocredStructuredSetDecoderError("Set3 contains unauthorized ordinal")
    return canonical  # type: ignore[return-value]


def _active_bridge_pairs(space: TypedActionSpace) -> frozenset[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for authorization in space.authorizations:
        for witness in authorization.witnesses:
            if witness.kind == "BRIDGE":
                result.add(tuple(sorted((witness.left_sentence, witness.right_sentence))))
    return frozenset(result)


def phi_features(
    space: TypedActionSpace, ordinals: Sequence[object]
) -> tuple[float, ...]:
    selected = _validated_set3(space, ordinals)
    item = space.item
    authorization = space.authorization_map()
    head_sentences = _entity_sentence_sets(item)[item.head_entity]
    tail_sentences = _entity_sentence_sets(item)[item.tail_entity]
    sentence_entities = _sentence_entity_sets(item)
    full_scores = [space.full_query_q6[index] / Q6_SCALE for index in selected]
    relation_scores = [
        space.relation_description_q6[index] / Q6_SCALE for index in selected
    ]
    selected_pairs = tuple(itertools.combinations(selected, 2))
    bridge_pairs = _active_bridge_pairs(space)
    pair_redundancies = [
        q6_cosine(
            item.sentence_embeddings[left], item.sentence_embeddings[right]
        )
        / Q6_SCALE
        for left, right in selected_pairs
    ]
    union_kinds = {
        kind for ordinal in selected for kind in authorization[ordinal].kinds
    }
    result = (
        math.fsum(full_scores) / 3.0,
        min(full_scores),
        math.fsum(relation_scores) / 3.0,
        sum(ordinal in head_sentences for ordinal in selected) / 3.0,
        sum(ordinal in tail_sentences for ordinal in selected) / 3.0,
        sum(
            ordinal in head_sentences and ordinal in tail_sentences
            for ordinal in selected
        )
        / 3.0,
        float(
            any(ordinal in head_sentences for ordinal in selected)
            and any(ordinal in tail_sentences for ordinal in selected)
        ),
        sum(tuple(sorted(pair)) in bridge_pairs for pair in selected_pairs) / 3.0,
        sum(
            bool(sentence_entities[left] & sentence_entities[right])
            for left, right in selected_pairs
        )
        / 3.0,
        sum("QUERY" in authorization[ordinal].kinds for ordinal in selected) / 3.0,
        len(union_kinds) / 6.0,
        -max(pair_redundancies),
    )
    if len(result) != len(G8_FEATURE_ORDER) or not all(
        math.isfinite(value) for value in result
    ):
        raise DocredStructuredSetDecoderError("invalid G8 feature vector")
    return result


def utility_x6(ordinals: Sequence[object], gold: GoldEvidence) -> int:
    if isinstance(ordinals, (str, bytes)) or len(ordinals) != TOP_K:
        raise DocredStructuredSetDecoderError("utility requires exactly three outputs")
    selected: list[int] = []
    for value in ordinals:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DocredStructuredSetDecoderError("selected ordinal must be integer")
        selected.append(value)
    if len(set(selected)) != TOP_K:
        raise DocredStructuredSetDecoderError("selected ordinals must be unique")
    gold_set = set(gold.ordinals)
    hits = len(set(selected) & gold_set)
    recall_x6 = hits * (6 // len(gold_set))
    complete_x6 = 6 if gold_set.issubset(selected) else 0
    return recall_x6 + complete_x6


def utility(ordinals: Sequence[object], gold: GoldEvidence) -> Fraction:
    return Fraction(utility_x6(ordinals, gold), 6)


def g8_item_sufficient_statistics(
    item: ValidatedActionItem,
    gold: GoldEvidence,
    *,
    space: TypedActionSpace | None = None,
) -> G8ItemSufficientStatistics:
    """Two-pass, constant-set-memory item-centred sufficient statistics."""

    action_space = build_action_space(item) if space is None else space
    feature_sum = np.zeros(len(G8_FEATURE_ORDER), dtype=np.float64)
    target_sum = 0.0
    set_count = 0
    for selected in iter_authorized_set3(action_space):
        feature_sum += np.asarray(phi_features(action_space, selected), dtype=np.float64)
        target_sum += utility_x6(selected, gold) / 6.0
        set_count += 1
    if set_count < FRONTIER_SIZE:
        raise DocredStructuredSetDecoderError("complete set space has fewer than 16 sets")
    mean_phi = feature_sum / float(set_count)
    mean_target = target_sum / float(set_count)

    centered_xx = np.zeros(
        (len(G8_FEATURE_ORDER), len(G8_FEATURE_ORDER)), dtype=np.float64
    )
    centered_xy = np.zeros(len(G8_FEATURE_ORDER), dtype=np.float64)
    target_digest = hashlib.sha256()
    for selected in iter_authorized_set3(action_space):
        centered_phi = (
            np.asarray(phi_features(action_space, selected), dtype=np.float64)
            - mean_phi
        )
        centered_target = utility_x6(selected, gold) / 6.0 - mean_target
        centered_xx += np.outer(centered_phi, centered_phi)
        centered_xy += centered_phi * centered_target
        target_digest.update(struct.pack("<d", centered_target))
    if not np.isfinite(centered_xx).all() or not np.isfinite(centered_xy).all():
        raise DocredStructuredSetDecoderError("nonfinite G8 sufficient statistics")
    return G8ItemSufficientStatistics(
        set_count=set_count,
        mean_phi=tuple(float(value) for value in mean_phi),
        mean_target=float(mean_target),
        centered_xx=tuple(
            tuple(float(value) for value in row) for row in centered_xx
        ),
        centered_xy=tuple(float(value) for value in centered_xy),
        centered_target_sha256=target_digest.hexdigest(),
    )


def action_item_commitment(item: ValidatedActionItem) -> str:
    """Private deterministic ordering commitment; never an action feature."""

    return _stable_hash(
        {
            "common_query_sha256": hashlib.sha256(
                item.common_query.encode("utf-8")
            ).hexdigest(),
            "relation_description_sha256": hashlib.sha256(
                item.relation_description.encode("utf-8")
            ).hexdigest(),
            "sentence_sha256": [
                hashlib.sha256(sentence.encode("utf-8")).hexdigest()
                for sentence in item.sentences
            ],
            "entity_sentence_rows": [
                [mention.sentence_ordinal for mention in entity.mentions]
                for entity in item.entities
            ],
            "head_entity": item.head_entity,
            "tail_entity": item.tail_entity,
        }
    )


def _ordered_labelled_items(
    examples: Sequence[LabelledItem], *, per_family: int
) -> tuple[LabelledItem, ...]:
    if len(examples) != per_family * len(FAMILY_ORDER):
        raise DocredStructuredSetDecoderError("formal labelled item count mismatch")
    by_family: dict[str, list[tuple[str, LabelledItem]]] = {
        family: [] for family in FAMILY_ORDER
    }
    seen: set[str] = set()
    for example in examples:
        if not isinstance(example, LabelledItem) or example.family not in by_family:
            raise DocredStructuredSetDecoderError("invalid labelled item")
        commitment = action_item_commitment(example.item)
        if commitment in seen:
            raise DocredStructuredSetDecoderError("duplicate action item commitment")
        seen.add(commitment)
        by_family[example.family].append((commitment, example))
    ordered: list[LabelledItem] = []
    for family in FAMILY_ORDER:
        rows = sorted(by_family[family], key=lambda row: row[0])
        if len(rows) != per_family:
            raise DocredStructuredSetDecoderError("formal family count mismatch")
        ordered.extend(example for _, example in rows)
    return tuple(ordered)


def fit_g8(examples: Sequence[LabelledItem]) -> G8Model:
    """Fit the frozen 12-dimensional G8 ridge on exactly 96 G_form items."""

    ordered = _ordered_labelled_items(examples, per_family=32)
    dimension = len(G8_FEATURE_ORDER)
    matrix = np.eye(dimension, dtype=np.float64) * RIDGE_LAMBDA
    target = np.zeros(dimension, dtype=np.float64)
    weight_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    total_observations = 0
    for example in ordered:
        stats = g8_item_sufficient_statistics(example.item, example.gold)
        item_weight = 1.0 / 96.0
        set_weight = item_weight / float(stats.set_count)
        xx = np.asarray(stats.centered_xx, dtype=np.float64)
        xy = np.asarray(stats.centered_xy, dtype=np.float64)
        matrix += set_weight * xx
        target += set_weight * xy
        commitment = action_item_commitment(example.item)
        weight_rows.append(
            {
                "item": commitment,
                "set_count": stats.set_count,
                "set_weight_hex": set_weight.hex(),
            }
        )
        target_rows.append(
            {
                "item": commitment,
                "centered_target_sha256": stats.centered_target_sha256,
            }
        )
        total_observations += stats.set_count
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise DocredStructuredSetDecoderError("nonfinite G8 normal equation")
    try:
        coefficients = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError as exc:
        raise DocredStructuredSetDecoderError("G8 normal equation solve failed") from exc
    if not np.isfinite(coefficients).all():
        raise DocredStructuredSetDecoderError("nonfinite G8 coefficients")
    normal_hash = _normal_equation_hash(matrix, target)
    weight_hash = _stable_hash(weight_rows)
    target_hash = _stable_hash(target_rows)
    coefficient_hash = hashlib.sha256(_float64_bytes(coefficients)).hexdigest()
    fit_hash = _stable_hash(
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
        raise DocredStructuredSetDecoderError("G8 feature dimension mismatch")
    values = tuple(
        _finite_float(value, field=f"G8 feature {index}")
        for index, value in enumerate(phi)
    )
    result = math.fsum(
        weight * value for weight, value in zip(model.weights, values, strict=True)
    )
    if not math.isfinite(result):
        raise DocredStructuredSetDecoderError("nonfinite G8 energy")
    return result


def _frontier_key(entry: FrontierEntry) -> tuple[float, tuple[int, int, int]]:
    return (-entry.generator_energy, entry.ordinals)


def g8_frontier(
    item: ValidatedActionItem,
    model: G8Model,
    *,
    space: TypedActionSpace | None = None,
) -> G8Frontier:
    """Stream the complete set space and retain the exact best sixteen."""

    action_space = build_action_space(item) if space is None else space
    retained: list[FrontierEntry] = []
    for selected in iter_authorized_set3(action_space):
        phi = phi_features(action_space, selected)
        entry = FrontierEntry(selected, phi, g8_energy(model, phi))
        if len(retained) < FRONTIER_SIZE:
            retained.append(entry)
            continue
        worst_index = max(range(len(retained)), key=lambda index: _frontier_key(retained[index]))
        if _frontier_key(entry) < _frontier_key(retained[worst_index]):
            retained[worst_index] = entry
    retained.sort(key=_frontier_key)
    if len(retained) != FRONTIER_SIZE:
        raise DocredStructuredSetDecoderError("fewer than sixteen frontier sets")
    return G8Frontier(tuple(retained))


def _set_coverage(space: TypedActionSpace, selected: Sequence[int]) -> float:
    if not selected:
        raise DocredStructuredSetDecoderError("coverage requires a nonempty set")
    item = space.item
    head_sentences = _entity_sentence_sets(item)[item.head_entity]
    tail_sentences = _entity_sentence_sets(item)[item.tail_entity]
    full_max = max(space.full_query_q6[index] / Q6_SCALE for index in selected)
    relation_max = max(
        space.relation_description_q6[index] / Q6_SCALE for index in selected
    )
    return (
        full_max
        + relation_max
        + float(any(index in head_sentences for index in selected))
        + float(any(index in tail_sentences for index in selected))
    ) / 4.0


def psi_features(
    space: TypedActionSpace,
    entry: FrontierEntry,
) -> tuple[float, ...]:
    selected = _validated_set3(space, entry.ordinals)
    authorization = space.authorization_map()
    singleton_scores: list[float] = []
    for ordinal in selected:
        kinds = set(authorization[ordinal].kinds)
        singleton_scores.append(
            (
                space.full_query_q6[ordinal] / Q6_SCALE
                + space.relation_description_q6[ordinal] / Q6_SCALE
                + float("HEAD" in kinds)
                + float("TAIL" in kinds)
                + float("DIRECT" in kinds)
                + float("QUERY" in kinds)
            )
            / 6.0
        )
    full_coverage = _set_coverage(space, selected)
    deletion_drops = [
        full_coverage - _set_coverage(space, tuple(value for value in selected if value != removed))
        for removed in selected
    ]
    substitute_coverages: list[float] = []
    selected_set = set(selected)
    for removed in selected:
        removed_kinds = set(authorization[removed].kinds)
        for candidate in space.authorized_ordinals:
            if candidate in selected_set:
                continue
            if not removed_kinds.intersection(authorization[candidate].kinds):
                continue
            replacement = tuple(sorted((selected_set - {removed}) | {candidate}))
            substitute_coverages.append(_set_coverage(space, replacement))
    substitute_drop = (
        full_coverage - max(substitute_coverages) if substitute_coverages else 0.0
    )
    phi = entry.phi
    result = (
        entry.generator_energy,
        min(singleton_scores),
        math.fsum(deletion_drops) / 3.0,
        min(deletion_drops),
        substitute_drop,
        phi[G8_FEATURE_ORDER.index("head_and_tail_set_coverage_indicator")],
        float(phi[G8_FEATURE_ORDER.index("one_bridge_witness_pair_fraction")] > 0.0),
        phi[G8_FEATURE_ORDER.index("negative_maximum_selected_pair_sentence_redundancy")],
    )
    if len(result) != len(E1_FEATURE_ORDER) or not all(
        math.isfinite(value) for value in result
    ):
        raise DocredStructuredSetDecoderError("invalid E1 feature vector")
    return result


def solve_standardized_pairwise_ridge(
    pair_differences: Sequence[Sequence[object]],
    targets: Sequence[object],
    *,
    row_weight: float,
) -> PairwiseRidgeSolution:
    """Deterministic no-intercept lambda-one pairwise ridge solver."""

    if len(pair_differences) != len(targets) or not pair_differences:
        raise DocredStructuredSetDecoderError("pairwise row count mismatch")
    weight = _finite_float(row_weight, field="pairwise row weight")
    if weight <= 0.0 or not math.isclose(
        weight * len(pair_differences), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise DocredStructuredSetDecoderError("pairwise weights must sum to one")
    dimension = len(pair_differences[0])
    if dimension <= 0:
        raise DocredStructuredSetDecoderError("empty pairwise feature dimension")
    rows = np.empty((len(pair_differences), dimension), dtype=np.float64)
    y = np.empty(len(targets), dtype=np.float64)
    for row_index, row in enumerate(pair_differences):
        if len(row) != dimension:
            raise DocredStructuredSetDecoderError("pairwise feature dimension mismatch")
        rows[row_index] = [
            _finite_float(value, field=f"pair row {row_index} feature {index}")
            for index, value in enumerate(row)
        ]
        y[row_index] = _finite_float(targets[row_index], field="pair target")
    weighted_mean = rows.sum(axis=0, dtype=np.float64) * weight
    if not np.array_equal(weighted_mean, np.zeros(dimension, dtype=np.float64)):
        if not np.all(np.abs(weighted_mean) <= 1e-15):
            raise DocredStructuredSetDecoderError(
                "oriented pair differences are not antisymmetric"
            )
    variances = (rows * rows).sum(axis=0, dtype=np.float64) * weight
    stds = np.sqrt(variances)
    standardized = np.zeros_like(rows)
    nonzero = stds > 0.0
    standardized[:, nonzero] = rows[:, nonzero] / stds[nonzero]
    matrix = np.eye(dimension, dtype=np.float64) * RIDGE_LAMBDA
    target = np.zeros(dimension, dtype=np.float64)
    for row_index in range(len(rows)):
        row = standardized[row_index]
        matrix += weight * np.outer(row, row)
        target += weight * row * y[row_index]
    try:
        coefficients = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError as exc:
        raise DocredStructuredSetDecoderError("E1 normal equation solve failed") from exc
    if not np.isfinite(coefficients).all():
        raise DocredStructuredSetDecoderError("nonfinite E1 coefficients")
    normal_hash = _normal_equation_hash(matrix, target)
    weight_hash = _stable_hash(
        {"row_count": len(rows), "row_weight_hex": weight.hex()}
    )
    target_hash = hashlib.sha256(_float64_bytes(y)).hexdigest()
    coefficient_hash = hashlib.sha256(_float64_bytes(coefficients)).hexdigest()
    return PairwiseRidgeSolution(
        weights=tuple(float(value) for value in coefficients),
        feature_stds=tuple(float(value) for value in stds),
        normal_equation_sha256=normal_hash,
        observation_weight_sha256=weight_hash,
        target_sha256=target_hash,
        coefficient_sha256=coefficient_hash,
    )


def fit_e1(examples: Sequence[LabelledItem], g8_model: G8Model) -> E1Model:
    """Fit the frozen eight-dimensional E1 ridge on exactly 48 A_form items."""

    ordered = _ordered_labelled_items(examples, per_family=16)
    differences: list[tuple[float, ...]] = []
    targets: list[float] = []
    frontier_hash_rows: list[dict[str, object]] = []
    for example in ordered:
        space = build_action_space(example.item)
        frontier = g8_frontier(example.item, g8_model, space=space)
        psi_rows = tuple(psi_features(space, entry) for entry in frontier.entries)
        utility_rows = tuple(
            utility_x6(entry.ordinals, example.gold) / 6.0
            for entry in frontier.entries
        )
        for left in range(FRONTIER_SIZE):
            for right in range(left + 1, FRONTIER_SIZE):
                difference = tuple(
                    psi_rows[left][index] - psi_rows[right][index]
                    for index in range(len(E1_FEATURE_ORDER))
                )
                target = utility_rows[left] - utility_rows[right]
                differences.append(difference)
                targets.append(target)
                differences.append(tuple(-value for value in difference))
                targets.append(-target)
        frontier_hash_rows.append(
            {
                "item": action_item_commitment(example.item),
                "frontier": [entry.ordinals for entry in frontier.entries],
                "psi_sha256": hashlib.sha256(
                    _float64_bytes(np.asarray(psi_rows, dtype=np.float64))
                ).hexdigest(),
            }
        )
    if len(differences) != 11_520:
        raise DocredStructuredSetDecoderError("E1 oriented pair count mismatch")
    solution = solve_standardized_pairwise_ridge(
        differences,
        targets,
        row_weight=1.0 / 11_520.0,
    )
    fit_hash = _stable_hash(
        {
            "coefficient_sha256": solution.coefficient_sha256,
            "normal_equation_sha256": solution.normal_equation_sha256,
            "observation_weight_sha256": solution.observation_weight_sha256,
            "target_sha256": solution.target_sha256,
            "feature_stds_hex": [value.hex() for value in solution.feature_stds],
            "feature_order": E1_FEATURE_ORDER,
            "deployment_formula": E1_DEPLOYMENT_FORMULA,
            "frontiers": frontier_hash_rows,
            "lambda": RIDGE_LAMBDA,
        }
    )
    return E1Model(
        weights=solution.weights,
        feature_stds=solution.feature_stds,
        normal_equation_sha256=solution.normal_equation_sha256,
        observation_weight_sha256=solution.observation_weight_sha256,
        target_sha256=solution.target_sha256,
        coefficient_sha256=solution.coefficient_sha256,
        fit_sha256=fit_hash,
    )


def e1_score(model: E1Model, psi: Sequence[object]) -> float:
    if len(psi) != len(E1_FEATURE_ORDER):
        raise DocredStructuredSetDecoderError("E1 feature dimension mismatch")
    standardized: list[float] = []
    for index, value in enumerate(psi):
        parsed = _finite_float(value, field=f"E1 feature {index}")
        std = model.feature_stds[index]
        standardized.append(0.0 if std == 0.0 else parsed / std)
    result = math.fsum(
        weight * value
        for weight, value in zip(model.weights, standardized, strict=True)
    )
    if not math.isfinite(result):
        raise DocredStructuredSetDecoderError("nonfinite E1 score")
    return result


def e1_select(
    space: TypedActionSpace,
    frontier: G8Frontier,
    model: E1Model,
) -> E1Selection:
    selections: list[E1Selection] = []
    for entry in frontier.entries:
        psi = psi_features(space, entry)
        selections.append(E1Selection(entry, psi, e1_score(model, psi)))
    selections.sort(
        key=lambda row: (
            -row.score,
            -row.entry.generator_energy,
            row.entry.ordinals,
        )
    )
    return selections[0]


def raw3(item: ValidatedActionItem) -> tuple[int, int, int]:
    """Exact dense baseline over every document sentence, with no candidate gate."""

    rows = [
        (q6_cosine(item.full_query_embedding, embedding), ordinal)
        for ordinal, embedding in enumerate(item.sentence_embeddings)
    ]
    rows.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(ordinal for _, ordinal in rows[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise DocredStructuredSetDecoderError("RAW3 output invalid")
    return result  # type: ignore[return-value]


def exact_sign_flip_x6(deltas_x6: Sequence[object]) -> SignFlipResult:
    parsed: list[int] = []
    for value in deltas_x6:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DocredStructuredSetDecoderError("x6 delta must be integer")
        parsed.append(value)
    observed = sum(parsed)
    magnitudes = [abs(value) for value in parsed if value != 0]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        next_distribution: Counter[int] = Counter()
        for subtotal, count in sorted(distribution.items()):
            next_distribution[subtotal + magnitude] += count
            next_distribution[subtotal - magnitude] += count
        distribution = next_distribution
    assignment_count = 1 << len(magnitudes)
    tail_count = sum(
        count for signed_sum, count in distribution.items() if signed_sum >= observed
    )
    return SignFlipResult(
        observed_sum_x6=observed,
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
    witness_rows = sorted(
        {
            witness
            for authorization in space.authorizations
            for witness in authorization.witnesses
        },
        key=Witness.sort_key,
    )
    return _stable_hash(
        {
            "item_commitment": action_item_commitment(item),
            "query_commitment": hashlib.sha256(
                item.common_query.encode("utf-8")
            ).hexdigest(),
            "authorized_ordinals": space.authorized_ordinals,
            "typed_edges": [
                {
                    "kind": row.kind,
                    "entity_id": row.entity_id,
                    "left": row.left_sentence,
                    "right": row.right_sentence,
                }
                for row in witness_rows
            ],
            "frontier": [
                {
                    "ordinals": row.ordinals,
                    "energy_hex": row.generator_energy.hex(),
                }
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
    """Delete each used typed edge once and causally re-decode without labels."""

    base_space = build_action_space(item)
    base_frontier = g8_frontier(item, g8_model, space=base_space)
    e0_before = base_frontier.e0.ordinals
    e1_before = (
        e1_select(base_space, base_frontier, e1_model).entry.ordinals
        if e1_model is not None
        else None
    )
    selected_terminals = set(e0_before)
    if e1_before is not None:
        selected_terminals.update(e1_before)
    used_witnesses = sorted(
        {
            witness
            for authorization in base_space.authorizations
            if authorization.ordinal in selected_terminals
            for witness in authorization.witnesses
            if witness.kind in {"DIRECT", "COREFERENCE", "BRIDGE"}
        },
        key=Witness.sort_key,
    )
    receipts: list[EdgeDeletionReceipt] = []
    for witness in used_witnesses:
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


def edge_deletion_action_change_count(
    receipts_by_item: Iterable[Sequence[EdgeDeletionReceipt]],
) -> int:
    return sum(
        any(row.e0_changed or row.e1_changed is True for row in receipts)
        for receipts in receipts_by_item
    )


def outside_raw3_count(
    agent_outputs: Sequence[Sequence[object]],
    raw_outputs: Sequence[Sequence[object]],
) -> int:
    if len(agent_outputs) != len(raw_outputs):
        raise DocredStructuredSetDecoderError("Agent/RAW item count mismatch")
    total = 0
    for agent, raw in zip(agent_outputs, raw_outputs, strict=True):
        if len(agent) != TOP_K or len(raw) != TOP_K:
            raise DocredStructuredSetDecoderError("Agent/RAW output must be top3")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (*agent, *raw)):
            raise DocredStructuredSetDecoderError("Agent/RAW ordinals must be integers")
        if len(set(agent)) != TOP_K or len(set(raw)) != TOP_K:
            raise DocredStructuredSetDecoderError("Agent/RAW ordinals must be unique")
        total += len(set(agent) - set(raw))
    return total


__all__ = [
    "AUTHORITY_KIND_ORDER",
    "DESIGN_COMMIT",
    "DESIGN_SELF_SHA256",
    "DocredStructuredSetDecoderError",
    "E1_DEPLOYMENT_FORMULA",
    "E1Model",
    "E1Selection",
    "E1_FEATURE_ORDER",
    "EdgeDeletionReceipt",
    "Entity",
    "FAMILY_ORDER",
    "FRONTIER_SIZE",
    "FrontierEntry",
    "G8Frontier",
    "G8ItemSufficientStatistics",
    "G8Model",
    "G8_FEATURE_ORDER",
    "GoldEvidence",
    "LabelledItem",
    "Mention",
    "PairwiseRidgeSolution",
    "SignFlipResult",
    "TerminalAuthorization",
    "TypedActionSpace",
    "ValidatedActionItem",
    "VERSION",
    "Witness",
    "action_item_commitment",
    "behavior_hash",
    "build_action_space",
    "canonical_aliases",
    "e1_score",
    "e1_select",
    "edge_deletion_action_change_count",
    "edge_deletion_redecode",
    "exact_sign_flip_x6",
    "fit_e1",
    "fit_g8",
    "g8_energy",
    "g8_frontier",
    "g8_item_sufficient_statistics",
    "iter_authorized_set3",
    "labelled_item",
    "outside_raw3_count",
    "phi_features",
    "psi_features",
    "q6_cosine",
    "quantize_cosine_value",
    "raw3",
    "render_sentence",
    "serialize_common_query",
    "solve_standardized_pairwise_ridge",
    "utility",
    "utility_x6",
    "validate_action_item",
    "validate_gold",
]

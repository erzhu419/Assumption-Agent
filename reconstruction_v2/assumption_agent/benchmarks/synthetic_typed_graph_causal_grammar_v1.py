"""Deterministic, stdlib-only grammar for a typed-graph causal stress test.

This module is a prospective data-generating process, not a benchmark loader.
It imports no model, embedding, retriever, scorer, graph implementation, or
external data package.  A caller supplies exactly 32 secret bytes; the module
then compiles every preregistered slot directly.  There is no candidate pool,
row filter, retry, replacement, or outcome-dependent branch.

The generated worlds are deliberately synthetic.  They can test whether the
frozen typed-edge/action/evaluator mechanism behaves causally under a public
symbolic grammar, but cannot support a claim about any natural population or
real benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any, Mapping, Sequence


VERSION = "synthetic_typed_graph_causal_grammar_v1"
DOMAIN = "synthetic_typed_graph_causal_v1"

MENTIONS_DEFINITION = "MENTIONS_DEFINITION"
EXCEPTION_SCOPE = "EXCEPTION_SCOPE"
LIST_SIBLING = "LIST_SIBLING"
EXPLICIT_CROSS_REFERENCE = "EXPLICIT_CROSS_REFERENCE"
EDGE_FAMILIES = (
    MENTIONS_DEFINITION,
    EXCEPTION_SCOPE,
    LIST_SIBLING,
    EXPLICIT_CROSS_REFERENCE,
)
EDGE_CODE = {
    MENTIONS_DEFINITION: "DEF",
    EXCEPTION_SCOPE: "EXC",
    LIST_SIBLING: "LST",
    EXPLICIT_CROSS_REFERENCE: "XRF",
}

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
BLOCK_SIZE = 64
NODE_COUNT = 32
MIN_GOLD = 1
MAX_GOLD = 3
TOP_K = 5
OFFICIAL_CONCURRENCY_CAP = 8
LOCAL_CONCURRENCY_CAP = 64
E00_CONTROL_EVALUATOR_ID = "E_UNIFORM_L025"

TRAIN_POSITIVE_1 = "train_positive_1"
TRAIN_POSITIVE_2 = "train_positive_2"
TRAIN_NEGATIVE_1 = "train_negative_1"
TRAIN_NEGATIVE_2 = "train_negative_2"
FAMILYOUT_POSITIVE = "familyout_positive"
FAMILYOUT_NEGATIVE = "familyout_negative"
FAMILY_ROLES = (
    TRAIN_POSITIVE_1,
    TRAIN_POSITIVE_2,
    TRAIN_NEGATIVE_1,
    TRAIN_NEGATIVE_2,
    FAMILYOUT_POSITIVE,
    FAMILYOUT_NEGATIVE,
)

POSITIVE = "positive"
NEGATIVE = "negative"
TRAIN_SPLIT = "train_templates"
FAMILYOUT_SPLIT = "familyout_templates"
EDGE_PRESENT_NONCAUSAL = "edge_present_but_noncausal"
EDGE_PRESENT_INDEPENDENT_DIRECT_CUE_VARIANT_2 = (
    "edge_present_but_query_and_gold_are_independent_direct_cue"
)

FULL_GRAPH = "full"
DROP_DESIGNATED = "drop_designated"
WRONG_TYPE = "wrong_type"
ENDPOINT_PERMUTED = "endpoint_permuted"
ABLATION_MODES = (
    FULL_GRAPH,
    DROP_DESIGNATED,
    WRONG_TYPE,
    ENDPOINT_PERMUTED,
)

_SYLLABLES = (
    "ba",
    "ce",
    "di",
    "fo",
    "gu",
    "ha",
    "ji",
    "ko",
    "lu",
    "me",
    "ni",
    "pa",
    "qu",
    "ri",
    "so",
    "tu",
)
_SYMBOL_ROLES = (
    "actor",
    "concept",
    "term",
    "payload",
    "direct_cue",
    "target_0",
    "target_1",
    "target_2",
    "decoy_0",
    "decoy_1",
    "decoy_2",
)
_NODE_ROLES = (
    "anchor",
    "causal_target_0",
    "causal_target_1",
    "causal_target_2",
    "permutation_decoy_0",
    "permutation_decoy_1",
    "permutation_decoy_2",
    "direct_gold_0",
    "direct_gold_1",
    "direct_gold_2",
    *(f"distractor_{index:02d}" for index in range(22)),
)
_DISTRACTOR_STYLES = (
    "catalog",
    "ledger",
    "register",
    "archive",
)
_WRONG_EDGE = {
    MENTIONS_DEFINITION: EXCEPTION_SCOPE,
    EXCEPTION_SCOPE: LIST_SIBLING,
    LIST_SIBLING: EXPLICIT_CROSS_REFERENCE,
    EXPLICIT_CROSS_REFERENCE: MENTIONS_DEFINITION,
}


class SyntheticCausalGrammarError(ValueError):
    """An input or generated invariant violates the frozen grammar."""


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    edge_family: str
    family_role: str
    template_split: str
    polarity: str
    surface_variant: int
    match_group: str
    matched_family_id: str
    negative_kind: str | None

    @property
    def held_out(self) -> bool:
        return self.template_split == FAMILYOUT_SPLIT


@dataclass(frozen=True)
class LatentWorld:
    schema: str
    block: str
    block_ordinal: int
    family_slot: int
    world_serial: int
    family: FamilySpec
    gold_count: int
    symbols: tuple[tuple[str, str], ...]
    role_order: tuple[str, ...]
    distractor_styles: tuple[str, ...]
    proof_roles: tuple[str, ...]
    pair_key: str
    structural_draw_sha256: str

    def symbol(self, role: str) -> str:
        values = dict(self.symbols)
        try:
            return values[role]
        except KeyError as exc:
            raise SyntheticCausalGrammarError("latent symbol role is absent") from exc


@dataclass(frozen=True)
class SyntheticNode:
    span_i: int
    start: int
    end: int
    identity_text: str
    latent_role: str


@dataclass(frozen=True, order=True)
class SyntheticEdge:
    edge_family: str
    left_span_i: int
    right_span_i: int

    def __post_init__(self) -> None:
        if self.edge_family not in EDGE_FAMILIES:
            raise SyntheticCausalGrammarError("edge family is not frozen")
        if not (
            type(self.left_span_i) is int
            and type(self.right_span_i) is int
            and 0 <= self.left_span_i < self.right_span_i < NODE_COUNT
        ):
            raise SyntheticCausalGrammarError("synthetic edge endpoints are invalid")


@dataclass(frozen=True)
class CompiledItem:
    schema: str
    block: str
    block_ordinal: int
    family_slot: int
    family_id: str
    family_role: str
    template_split: str
    polarity: str
    negative_kind: str | None
    edge_family: str
    pair_key: str
    item_commitment_sha256: str
    label_free_commitment_sha256: str
    matching_signature_sha256: str
    question: str
    context: str
    nodes: tuple[SyntheticNode, ...]
    gold_node_indices: tuple[int, ...]
    designated_edges: tuple[SyntheticEdge, ...]
    endpoint_permutation: tuple[tuple[int, int], ...]
    structural_draw_sha256: str


def _family_registry() -> tuple[FamilySpec, ...]:
    result: list[FamilySpec] = []
    role_rows = (
        (TRAIN_POSITIVE_1, TRAIN_SPLIT, POSITIVE, 1, TRAIN_NEGATIVE_1, None),
        (TRAIN_POSITIVE_2, TRAIN_SPLIT, POSITIVE, 2, TRAIN_NEGATIVE_2, None),
        (
            TRAIN_NEGATIVE_1,
            TRAIN_SPLIT,
            NEGATIVE,
            1,
            TRAIN_POSITIVE_1,
            EDGE_PRESENT_NONCAUSAL,
        ),
        (
            TRAIN_NEGATIVE_2,
            TRAIN_SPLIT,
            NEGATIVE,
            2,
            TRAIN_POSITIVE_2,
            EDGE_PRESENT_INDEPENDENT_DIRECT_CUE_VARIANT_2,
        ),
        (
            FAMILYOUT_POSITIVE,
            FAMILYOUT_SPLIT,
            POSITIVE,
            3,
            FAMILYOUT_NEGATIVE,
            None,
        ),
        (
            FAMILYOUT_NEGATIVE,
            FAMILYOUT_SPLIT,
            NEGATIVE,
            3,
            FAMILYOUT_POSITIVE,
            EDGE_PRESENT_NONCAUSAL,
        ),
    )
    role_code = {
        TRAIN_POSITIVE_1: "TP1",
        TRAIN_POSITIVE_2: "TP2",
        TRAIN_NEGATIVE_1: "TN1",
        TRAIN_NEGATIVE_2: "TN2",
        FAMILYOUT_POSITIVE: "FOP",
        FAMILYOUT_NEGATIVE: "FON",
    }
    for edge_family in EDGE_FAMILIES:
        edge_code = EDGE_CODE[edge_family]
        for role, split, polarity, variant, matched_role, negative_kind in role_rows:
            result.append(
                FamilySpec(
                    family_id=f"{edge_code}_{role_code[role]}",
                    edge_family=edge_family,
                    family_role=role,
                    template_split=split,
                    polarity=polarity,
                    surface_variant=variant,
                    match_group=f"{edge_code}_V{variant}",
                    matched_family_id=f"{edge_code}_{role_code[matched_role]}",
                    negative_kind=negative_kind,
                )
            )
    return tuple(result)


FAMILY_REGISTRY = _family_registry()
FAMILY_BY_ID = {family.family_id: family for family in FAMILY_REGISTRY}


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SyntheticCausalGrammarError("value is not canonical JSON") from exc


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_seed(seed: bytes) -> bytes:
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise SyntheticCausalGrammarError("seed must be exactly 32 raw bytes")
    return seed


def field_digest(
    seed: bytes,
    *,
    block: str,
    family_key: str,
    slot: int,
    field: str,
    counter: int = 0,
) -> bytes:
    """One canonical HMAC field draw; no mutable PRNG state exists."""

    _validate_seed(seed)
    if block not in BLOCK_ORDER or not family_key or not field:
        raise SyntheticCausalGrammarError("HMAC field domain is invalid")
    if type(slot) is not int or slot < 0 or type(counter) is not int or counter < 0:
        raise SyntheticCausalGrammarError("HMAC field coordinate is invalid")
    message = json.dumps(
        [DOMAIN, block, family_key, slot, field, counter],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(seed, message, hashlib.sha256).digest()


def field_integer(
    seed: bytes,
    *,
    block: str,
    family_key: str,
    slot: int,
    field: str,
    modulus: int,
    counter: int = 0,
) -> int:
    if type(modulus) is not int or modulus <= 0:
        raise SyntheticCausalGrammarError("HMAC modulus must be positive")
    return int.from_bytes(
        field_digest(
            seed,
            block=block,
            family_key=family_key,
            slot=slot,
            field=field,
            counter=counter,
        ),
        "big",
    ) % modulus


def public_lexeme(serial: int) -> str:
    """Injectively map a bounded integer to a fixed-width synthetic token."""

    if type(serial) is not int or not 0 <= serial < len(_SYLLABLES) ** 8:
        raise SyntheticCausalGrammarError("synthetic lexeme serial is out of range")
    digits: list[str] = []
    value = serial
    for _ in range(8):
        digits.append(_SYLLABLES[value % len(_SYLLABLES)])
        value //= len(_SYLLABLES)
    return "-".join(reversed(digits))


def family_quota(block: str) -> tuple[tuple[str, int], ...]:
    if block not in BLOCK_ORDER:
        raise SyntheticCausalGrammarError("block is not frozen")
    split = FAMILYOUT_SPLIT if block == "M_search" else TRAIN_SPLIT
    count = 8 if block == "M_search" else 4
    return tuple(
        (family.family_id, count)
        for family in FAMILY_REGISTRY
        if family.template_split == split
    )


def family_registry_rows() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "family_id": family.family_id,
            "edge_family": family.edge_family,
            "family_role": family.family_role,
            "template_split": family.template_split,
            "polarity": family.polarity,
            "surface_variant": family.surface_variant,
            "match_group": family.match_group,
            "matched_family_id": family.matched_family_id,
            "negative_kind": family.negative_kind,
        }
        for family in FAMILY_REGISTRY
    )


def quota_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        block: [
            {"family_id": family_id, "count": count}
            for family_id, count in family_quota(block)
        ]
        for block in BLOCK_ORDER
    }


def _hmac_order(
    values: Sequence[str],
    *,
    seed: bytes,
    block: str,
    family_key: str,
    slot: int,
    field: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                field_digest(
                    seed,
                    block=block,
                    family_key=family_key,
                    slot=slot,
                    field=field,
                    counter=int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big"),
                ),
                value,
            ),
        )
    )


def _gold_schedule(
    *, seed: bytes, block: str, match_group: str, quota: int
) -> tuple[int, ...]:
    if quota == 4:
        values = (1, 1, 2, 3)
    elif quota == 8:
        values = (1, 1, 1, 2, 2, 3, 3, 3)
    else:
        raise SyntheticCausalGrammarError("family quota has no frozen gold schedule")
    decorated = [
        (
            field_digest(
                seed,
                block=block,
                family_key=match_group,
                slot=index,
                field="gold_schedule",
            ),
            index,
            value,
        )
        for index, value in enumerate(values)
    ]
    return tuple(value for _digest, _index, value in sorted(decorated))


def _role_order(
    *, seed: bytes, block: str, family: FamilySpec, family_slot: int
) -> tuple[str, ...]:
    ordered = _hmac_order(
        _NODE_ROLES,
        seed=seed,
        block=block,
        family_key=family.match_group,
        slot=family_slot,
        field="node_order",
    )
    if family.edge_family not in {EXCEPTION_SCOPE, LIST_SIBLING}:
        return ordered
    causal_run = (
        "anchor",
        "causal_target_0",
        "causal_target_1",
        "causal_target_2",
    )
    remaining = tuple(role for role in ordered if role not in causal_run)
    start = field_integer(
        seed,
        block=block,
        family_key=family.match_group,
        slot=family_slot,
        field="causal_run_start",
        modulus=NODE_COUNT - len(causal_run) + 1,
    )
    return (*remaining[:start], *causal_run, *remaining[start:])


def _symbol_table(
    *,
    seed: bytes,
    block: str,
    family: FamilySpec,
    family_slot: int,
    world_serial: int,
) -> tuple[tuple[str, str], ...]:
    role_order = _hmac_order(
        _SYMBOL_ROLES,
        seed=seed,
        block=block,
        family_key=family.family_id,
        slot=family_slot,
        field="symbol_role_bijection",
    )
    offset_by_role = {role: offset for offset, role in enumerate(role_order)}
    return tuple(
        (
            role,
            public_lexeme(world_serial * 64 + offset_by_role[role]),
        )
        for role in _SYMBOL_ROLES
    )


def build_latent_world(
    *,
    seed: bytes,
    block: str,
    block_ordinal: int,
    family_id: str,
    family_slot: int,
) -> LatentWorld:
    _validate_seed(seed)
    if block not in BLOCK_ORDER or family_id not in FAMILY_BY_ID:
        raise SyntheticCausalGrammarError("world coordinate is not frozen")
    if type(block_ordinal) is not int or not 0 <= block_ordinal < BLOCK_SIZE:
        raise SyntheticCausalGrammarError("block ordinal is out of range")
    family = FAMILY_BY_ID[family_id]
    quota = dict(family_quota(block)).get(family_id)
    if quota is None or type(family_slot) is not int or not 0 <= family_slot < quota:
        raise SyntheticCausalGrammarError("family slot is out of range")
    world_serial = BLOCK_ORDER.index(block) * BLOCK_SIZE + block_ordinal
    schedule = _gold_schedule(
        seed=seed,
        block=block,
        match_group=family.match_group,
        quota=quota,
    )
    gold_count = schedule[family_slot]
    role_order = _role_order(
        seed=seed,
        block=block,
        family=family,
        family_slot=family_slot,
    )
    distractor_styles = tuple(
        _DISTRACTOR_STYLES[
            field_integer(
                seed,
                block=block,
                family_key=family.match_group,
                slot=family_slot,
                field="distractor_style",
                modulus=len(_DISTRACTOR_STYLES),
                counter=index,
            )
        ]
        for index in range(22)
    )
    proof_prefix = "causal_target" if family.polarity == POSITIVE else "direct_gold"
    proof_roles = tuple(f"{proof_prefix}_{index}" for index in range(gold_count))
    draw_payload = {
        "gold_count": gold_count,
        "role_order": role_order,
        "distractor_styles": distractor_styles,
        "surface_variant": family.surface_variant,
    }
    return LatentWorld(
        schema=f"{VERSION}_latent_AST",
        block=block,
        block_ordinal=block_ordinal,
        family_slot=family_slot,
        world_serial=world_serial,
        family=family,
        gold_count=gold_count,
        symbols=_symbol_table(
            seed=seed,
            block=block,
            family=family,
            family_slot=family_slot,
            world_serial=world_serial,
        ),
        role_order=role_order,
        distractor_styles=distractor_styles,
        proof_roles=proof_roles,
        pair_key=f"{block}:{family.match_group}:{family_slot}",
        structural_draw_sha256=_semantic_hash(draw_payload),
    )


def _list_markers(variant: int) -> tuple[str, str, str, str]:
    if variant == 1:
        return ("(a)", "(b)", "(c)", "(d)")
    if variant == 2:
        return ("1.", "2.", "3.", "4.")
    if variant == 3:
        return ("(i)", "(ii)", "(iii)", "(iv)")
    raise SyntheticCausalGrammarError("surface variant is not frozen")


def _xref_label(variant: int) -> tuple[str, str]:
    if variant == 1:
        return "section", "17"
    if variant == 2:
        return "clause", "b"
    if variant == 3:
        return "paragraph", "3.2"
    raise SyntheticCausalGrammarError("surface variant is not frozen")


def _node_texts(world: LatentWorld) -> dict[str, str]:
    family = world.family
    term = world.symbol("term")
    concept = world.symbol("concept")
    actor = world.symbol("actor")
    payload = world.symbol("payload")
    direct_cue = world.symbol("direct_cue")
    texts: dict[str, str] = {}

    for index in range(3):
        texts[f"permutation_decoy_{index}"] = (
            f"Neutral decoy {world.symbol(f'decoy_{index}')} stores {payload} token {index}."
        )
        texts[f"direct_gold_{index}"] = (
            f"Direct cue {direct_cue} record {index} stores {payload}."
        )
    for index, style in enumerate(world.distractor_styles):
        lexeme = public_lexeme(world.world_serial * 64 + 16 + index)
        texts[f"distractor_{index:02d}"] = (
            f"Neutral {style} {lexeme} records synthetic payload {index}."
        )

    if family.edge_family == MENTIONS_DEFINITION:
        if family.surface_variant == 1:
            texts["anchor"] = f'"{term}" means the {concept} role for {actor}.'
        elif family.surface_variant == 2:
            texts["anchor"] = f"{term} shall mean the {concept} role for {actor}."
        else:
            texts["anchor"] = f"“{term}” refers to the {concept} role for {actor}."
        for index in range(3):
            texts[f"causal_target_{index}"] = (
                f"The {term} record {world.symbol(f'target_{index}')} stores {payload}."
            )
    elif family.edge_family == EXCEPTION_SCOPE:
        texts["anchor"] = f"The {actor} shall preserve {concept} with {payload}."
        cue = {1: "except that", 2: "subject to", 3: "notwithstanding"}[
            family.surface_variant
        ]
        for index in range(3):
            texts[f"causal_target_{index}"] = (
                f"{cue} {world.symbol(f'target_{index}')} limits {payload} case {index}."
            )
    elif family.edge_family == LIST_SIBLING:
        markers = _list_markers(family.surface_variant)
        texts["anchor"] = f"{markers[0]} The {actor} list establishes {concept}."
        for index in range(3):
            texts[f"causal_target_{index}"] = (
                f"{markers[index + 1]} Entry {world.symbol(f'target_{index}')} stores {payload}."
            )
    elif family.edge_family == EXPLICIT_CROSS_REFERENCE:
        heading, label = _xref_label(family.surface_variant)
        texts["anchor"] = (
            f"The {actor} shall consult {heading} {label} for {concept}."
        )
        for index in range(3):
            texts[f"causal_target_{index}"] = (
                f"{heading} {label} {world.symbol(f'target_{index}')} stores {payload}."
            )
    else:
        raise SyntheticCausalGrammarError("edge family is not frozen")
    if set(texts) != set(_NODE_ROLES):
        raise SyntheticCausalGrammarError("compiler did not create all 32 node roles")
    return texts


def _question(world: LatentWorld) -> str:
    family = world.family
    if family.polarity == NEGATIVE:
        return f"Which records contain direct cue {world.symbol('direct_cue')}?"
    if family.edge_family == MENTIONS_DEFINITION:
        return f"Which records govern the {world.symbol('concept')} synthetic role?"
    if family.edge_family == EXCEPTION_SCOPE:
        return f"Which exceptions limit the {world.symbol('actor')} preservation rule?"
    if family.edge_family == LIST_SIBLING:
        return f"Which entries accompany the {world.symbol('actor')} list rule?"
    heading, label = _xref_label(family.surface_variant)
    return (
        f"Which {heading} {label} records are referenced for "
        f"{world.symbol('actor')}?"
    )


def _designated_role_edges(world: LatentWorld) -> tuple[tuple[str, str], ...]:
    if world.family.edge_family == LIST_SIBLING:
        return (
            ("anchor", "causal_target_0"),
            ("causal_target_0", "causal_target_1"),
            ("causal_target_1", "causal_target_2"),
        )
    return tuple(("anchor", f"causal_target_{index}") for index in range(3))


def compile_world(world: LatentWorld) -> CompiledItem:
    if not isinstance(world, LatentWorld) or world.schema != f"{VERSION}_latent_AST":
        raise SyntheticCausalGrammarError("compiler input is not a frozen latent AST")
    texts = _node_texts(world)
    nodes: list[SyntheticNode] = []
    context_parts: list[str] = []
    offset = 0
    role_to_index: dict[str, int] = {}
    for span_i, role in enumerate(world.role_order):
        text = texts[role]
        if not text or "\n" in text or "\r" in text:
            raise SyntheticCausalGrammarError("node text violates the synthetic envelope")
        if context_parts:
            context_parts.append("\n\n")
            offset += 2
        start = offset
        context_parts.append(text)
        offset += len(text)
        nodes.append(SyntheticNode(span_i, start, offset, text, role))
        role_to_index[role] = span_i
    context = "".join(context_parts)
    gold = tuple(sorted(role_to_index[role] for role in world.proof_roles))
    if not MIN_GOLD <= len(gold) <= MAX_GOLD:
        raise SyntheticCausalGrammarError("compiled gold cardinality drifted")
    designated: list[SyntheticEdge] = []
    for left_role, right_role in _designated_role_edges(world):
        left = role_to_index[left_role]
        right = role_to_index[right_role]
        designated.append(
            SyntheticEdge(world.family.edge_family, min(left, right), max(left, right))
        )
    endpoint_permutation = tuple(
        sorted(
            (
                role_to_index[f"causal_target_{index}"],
                role_to_index[f"permutation_decoy_{index}"],
            )
            for index in range(3)
        )
    )
    question = _question(world)
    label_free = {
        "block": world.block,
        "block_ordinal": world.block_ordinal,
        "family_id": world.family.family_id,
        "question": question,
        "nodes": [node.identity_text for node in nodes],
    }
    matching = {
        "block": world.block,
        "match_group": world.family.match_group,
        "family_slot": world.family_slot,
        "gold_count": world.gold_count,
        "role_order": world.role_order,
        "distractor_styles": world.distractor_styles,
        "surface_variant": world.family.surface_variant,
    }
    label_free_commitment = _semantic_hash(label_free)
    item_commitment = _semantic_hash(
        [DOMAIN, "item", label_free_commitment, _semantic_hash(list(gold))]
    )
    item = CompiledItem(
        schema=f"{VERSION}_compiled_item",
        block=world.block,
        block_ordinal=world.block_ordinal,
        family_slot=world.family_slot,
        family_id=world.family.family_id,
        family_role=world.family.family_role,
        template_split=world.family.template_split,
        polarity=world.family.polarity,
        negative_kind=world.family.negative_kind,
        edge_family=world.family.edge_family,
        pair_key=world.pair_key,
        item_commitment_sha256=item_commitment,
        label_free_commitment_sha256=label_free_commitment,
        matching_signature_sha256=_semantic_hash(matching),
        question=question,
        context=context,
        nodes=tuple(nodes),
        gold_node_indices=gold,
        designated_edges=tuple(sorted(set(designated))),
        endpoint_permutation=endpoint_permutation,
        structural_draw_sha256=world.structural_draw_sha256,
    )
    validate_compiled_item(item)
    return item


def validate_compiled_item(item: CompiledItem) -> None:
    if not isinstance(item, CompiledItem) or item.schema != f"{VERSION}_compiled_item":
        raise SyntheticCausalGrammarError("compiled item schema drifted")
    if len(item.nodes) != NODE_COUNT or tuple(node.span_i for node in item.nodes) != tuple(
        range(NODE_COUNT)
    ):
        raise SyntheticCausalGrammarError("compiled node identity drifted")
    for node in item.nodes:
        if not (0 <= node.start < node.end <= len(item.context)):
            raise SyntheticCausalGrammarError("compiled node offset is invalid")
        if item.context[node.start : node.end] != node.identity_text:
            raise SyntheticCausalGrammarError("compiled node offset does not preserve text")
    if not MIN_GOLD <= len(item.gold_node_indices) <= MAX_GOLD:
        raise SyntheticCausalGrammarError("compiled gold cardinality is invalid")
    if len(set(item.gold_node_indices)) != len(item.gold_node_indices) or any(
        type(index) is not int or not 0 <= index < NODE_COUNT
        for index in item.gold_node_indices
    ):
        raise SyntheticCausalGrammarError("compiled gold indices are invalid")
    if item.polarity == POSITIVE:
        if any(
            not item.nodes[index].latent_role.startswith("causal_target_")
            for index in item.gold_node_indices
        ):
            raise SyntheticCausalGrammarError("positive proof trace is not causal")
    elif item.polarity == NEGATIVE:
        if any(
            not item.nodes[index].latent_role.startswith("direct_gold_")
            for index in item.gold_node_indices
        ):
            raise SyntheticCausalGrammarError("negative proof trace is not a control")
    else:
        raise SyntheticCausalGrammarError("compiled polarity is invalid")


def generate_block(seed: bytes, block: str) -> tuple[CompiledItem, ...]:
    """Compile all 64 frozen slots directly, with no selection operation."""

    _validate_seed(seed)
    quota = family_quota(block)
    items: list[CompiledItem] = []
    block_ordinal = 0
    for family_id, count in quota:
        for family_slot in range(count):
            world = build_latent_world(
                seed=seed,
                block=block,
                block_ordinal=block_ordinal,
                family_id=family_id,
                family_slot=family_slot,
            )
            items.append(compile_world(world))
            block_ordinal += 1
    if len(items) != BLOCK_SIZE:
        raise SyntheticCausalGrammarError("direct block generation did not yield 64 items")
    if len({item.item_commitment_sha256 for item in items}) != BLOCK_SIZE:
        raise SyntheticCausalGrammarError("direct block item commitments overlap")
    return tuple(items)


def generate_all_blocks(seed: bytes) -> dict[str, tuple[CompiledItem, ...]]:
    """Compile the exact 4x64 design; this function never creates a seed."""

    _validate_seed(seed)
    blocks = {block: generate_block(seed, block) for block in BLOCK_ORDER}
    commitments = [
        item.item_commitment_sha256
        for block in BLOCK_ORDER
        for item in blocks[block]
    ]
    if len(commitments) != 4 * BLOCK_SIZE or len(set(commitments)) != 4 * BLOCK_SIZE:
        raise SyntheticCausalGrammarError("study item commitments are not disjoint")
    train_families = {
        item.family_id
        for block in ("A_form", "F_search", "A_hold")
        for item in blocks[block]
    }
    familyout_families = {item.family_id for item in blocks["M_search"]}
    if train_families & familyout_families:
        raise SyntheticCausalGrammarError("family-out templates overlap train templates")
    return blocks


def _canonical_edges(edges: Sequence[SyntheticEdge]) -> tuple[SyntheticEdge, ...]:
    if isinstance(edges, (str, bytes)):
        raise SyntheticCausalGrammarError("edges must be a sequence")
    normalized = tuple(edges)
    if any(not isinstance(edge, SyntheticEdge) for edge in normalized):
        raise SyntheticCausalGrammarError("edge sequence contains a foreign type")
    return tuple(sorted(set(normalized)))


def apply_graph_ablation(
    item: CompiledItem,
    edges: Sequence[SyntheticEdge],
    *,
    mode: str,
) -> tuple[SyntheticEdge, ...]:
    """Apply a graph-only intervention; query, nodes, labels, and ranks are absent."""

    validate_compiled_item(item)
    if mode not in ABLATION_MODES:
        raise SyntheticCausalGrammarError("graph ablation mode is not frozen")
    full = _canonical_edges(edges)
    designated = set(item.designated_edges)
    if mode == FULL_GRAPH:
        return full
    if mode == DROP_DESIGNATED:
        return tuple(edge for edge in full if edge not in designated)
    if mode == WRONG_TYPE:
        return _canonical_edges(
            tuple(
                SyntheticEdge(
                    _WRONG_EDGE[edge.edge_family],
                    edge.left_span_i,
                    edge.right_span_i,
                )
                if edge in designated
                else edge
                for edge in full
            )
        )
    permutation: dict[int, int] = {}
    for left, right in item.endpoint_permutation:
        permutation[left] = right
        permutation[right] = left
    permuted: list[SyntheticEdge] = []
    for edge in full:
        left = permutation.get(edge.left_span_i, edge.left_span_i)
        right = permutation.get(edge.right_span_i, edge.right_span_i)
        if left == right:
            raise SyntheticCausalGrammarError("endpoint permutation created a self edge")
        permuted.append(
            SyntheticEdge(edge.edge_family, min(left, right), max(left, right))
        )
    return _canonical_edges(permuted)


def evaluator_label_derangement(
    items: Sequence[CompiledItem], *, seed: bytes
) -> tuple[tuple[str, str], ...]:
    """Return the sole frozen within-stratum A-form label derangement plan."""

    _validate_seed(seed)
    normalized = tuple(items)
    if len(normalized) != BLOCK_SIZE or any(
        item.block != "A_form" for item in normalized
    ):
        raise SyntheticCausalGrammarError("derangement requires the full A_form block")
    groups: dict[tuple[str, int, int], list[CompiledItem]] = {}
    for item in normalized:
        key = (item.edge_family, len(item.nodes), len(item.gold_node_indices))
        groups.setdefault(key, []).append(item)
    pairs: list[tuple[str, str]] = []
    for key in sorted(groups):
        group = groups[key]
        if len(group) < 2:
            raise SyntheticCausalGrammarError("derangement stratum is a singleton")
        ordered = sorted(
            group,
            key=lambda item: (
                field_digest(
                    seed,
                    block="A_form",
                    family_key=item.edge_family,
                    slot=len(item.gold_node_indices),
                    field="evaluator_label_derangement_order",
                    counter=int(item.label_free_commitment_sha256[:8], 16),
                ),
                item.label_free_commitment_sha256,
            ),
        )
        shift = 1 + field_integer(
            seed,
            block="A_form",
            family_key=key[0],
            slot=key[2],
            field="evaluator_label_derangement_shift",
            modulus=len(ordered) - 1,
        )
        for index, destination in enumerate(ordered):
            source = ordered[(index + shift) % len(ordered)]
            if (
                source.label_free_commitment_sha256
                == destination.label_free_commitment_sha256
            ):
                raise SyntheticCausalGrammarError("derangement contains a fixed point")
            pairs.append(
                (
                    destination.label_free_commitment_sha256,
                    source.label_free_commitment_sha256,
                )
            )
    pairs.sort()
    if len(pairs) != BLOCK_SIZE:
        raise SyntheticCausalGrammarError("derangement does not cover A_form exactly")
    return tuple(pairs)


def block_commitment(items: Sequence[CompiledItem]) -> str:
    normalized = tuple(items)
    return _semantic_hash([item.item_commitment_sha256 for item in normalized])


__all__ = [
    "ABLATION_MODES",
    "BLOCK_ORDER",
    "BLOCK_SIZE",
    "CompiledItem",
    "DROP_DESIGNATED",
    "E00_CONTROL_EVALUATOR_ID",
    "EDGE_FAMILIES",
    "ENDPOINT_PERMUTED",
    "FAMILYOUT_SPLIT",
    "FAMILY_REGISTRY",
    "FULL_GRAPH",
    "FamilySpec",
    "LatentWorld",
    "NEGATIVE",
    "NODE_COUNT",
    "OFFICIAL_CONCURRENCY_CAP",
    "POSITIVE",
    "SyntheticCausalGrammarError",
    "SyntheticEdge",
    "SyntheticNode",
    "TRAIN_SPLIT",
    "VERSION",
    "WRONG_TYPE",
    "apply_graph_ablation",
    "block_commitment",
    "build_latent_world",
    "compile_world",
    "evaluator_label_derangement",
    "family_quota",
    "family_registry_rows",
    "field_digest",
    "field_integer",
    "generate_all_blocks",
    "generate_block",
    "public_lexeme",
    "quota_rows",
    "validate_compiled_item",
]

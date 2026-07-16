"""Frozen QASC counterfactual-chain recipe and evaluator core.

This module is deliberately model- and source-free.  It accepts a label-free
private retrieval view, produces auditable two-wave NLI pair plans for an
injected scorer, and keeps the label envelope out of every action-building
function.  Labels enter only through :func:`score_recipe_action` after an
action is terminal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Mapping, Sequence


VERSION = "qasc_counterfactual_chain_margin_v1"
DESIGN_COMMIT = "ac95a656"
DESIGN_SHA256 = "7c52b7e43d02ffa986683c49ca61863c3f36985b97a1a4677a40b6cddef8c150"
DESIGN_FILE_SHA256 = "fdd1bd1d088cee851a20015227d1f3dea1d086bcaf5c0f435f1bf52e943ab003"
VIEW_SCHEMA = "qasc_evaluator_direct_action_acquisition_v1_private_view"
LABEL_SCHEMA = "qasc_evaluator_direct_action_acquisition_v1_private_label_envelope"
DOMAIN_SEPARATOR = "qasc_evaluator_direct_action_coevolution_acquisition_v1"
DOCUMENT_COUNT = 32
CHOICE_COUNT = 8
TOP_K = 5
SEED_EXPANSION_COUNT = 4
FOLD_COUNT = 4
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_BLOCK_MEMBER = {
    "A_form": "TRAIN",
    "F_search": "TRAIN",
    "A_hold": "TRAIN",
    "M_search": "DEV",
}
_VIEW_KEYS = frozenset(
    {
        "schema",
        "block",
        "source_member",
        "formatted_question",
        "choices",
        "documents",
        "raw_ranking",
    }
)
_LABEL_KEYS = frozenset(
    {
        "schema",
        "block",
        "source_member",
        "identity_commitment_sha256",
        "view_sha256",
        "answerKey",
        "gold_document_ids",
        "fact1_document_id",
        "fact2_document_id",
    }
)
_STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does doing
    for from had has have having he her here hers herself him himself his how
    i if in into is it its itself may might more most must my myself no nor not
    of on once only or other our ours ourselves out over same she should so some
    such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves""".split()
)


class QASCCounterfactualError(RuntimeError):
    """Raised when a frozen recipe, view, label, or score contract is broken."""


@dataclass(frozen=True)
class Choice:
    label: str
    text: str


@dataclass(frozen=True)
class RetrievalDocument:
    doc_id: int
    text: str
    bm25_score_int: int


@dataclass(frozen=True)
class RetrievalView:
    schema: str
    block: str
    source_member: str
    formatted_question: str
    choices: tuple[Choice, ...]
    documents: tuple[RetrievalDocument, ...]
    raw_ranking: tuple[int, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "block": self.block,
            "source_member": self.source_member,
            "formatted_question": self.formatted_question,
            "choices": [
                {"label": choice.label, "text": choice.text}
                for choice in self.choices
            ],
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "text": document.text,
                    "bm25_score_int": document.bm25_score_int,
                }
                for document in self.documents
            ],
            "raw_ranking": list(self.raw_ranking),
        }

    @property
    def view_sha256(self) -> str:
        return stable_sha256(self.to_mapping())


@dataclass(frozen=True)
class LabelEnvelope:
    schema: str
    block: str
    source_member: str
    identity_commitment_sha256: str
    view_sha256: str
    answerKey: str
    gold_document_ids: tuple[int, int]
    fact1_document_id: int
    fact2_document_id: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "block": self.block,
            "source_member": self.source_member,
            "identity_commitment_sha256": self.identity_commitment_sha256,
            "view_sha256": self.view_sha256,
            "answerKey": self.answerKey,
            "gold_document_ids": list(self.gold_document_ids),
            "fact1_document_id": self.fact1_document_id,
            "fact2_document_id": self.fact2_document_id,
        }

    @property
    def envelope_sha256(self) -> str:
        return stable_sha256(self.to_mapping())


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    first_query: str
    bridge_budget: int
    second_query: str
    aggregation: str


@dataclass(frozen=True)
class NLIPair:
    premise: str
    hypothesis: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.premise, self.hypothesis)


@dataclass(frozen=True)
class FirstStageRequest:
    first_query: str
    choice_index: int
    doc_id: int
    pair_index: int


@dataclass(frozen=True)
class SecondStageRequest:
    first_query: str
    bridge_budget: int
    second_query: str
    choice_index: int
    seed_doc_id: int
    second_doc_id: int
    pair_index: int


@dataclass(frozen=True)
class FirstStagePlan:
    view_sha256: str
    recipe_ids: tuple[str, ...]
    pairs: tuple[NLIPair, ...]
    requests: tuple[FirstStageRequest, ...]
    conceptual_request_count: int

    @property
    def pairs_sha256(self) -> str:
        return stable_sha256(
            [{"premise": pair.premise, "hypothesis": pair.hypothesis} for pair in self.pairs]
        )


@dataclass(frozen=True)
class SecondStagePlan:
    view_sha256: str
    recipe_ids: tuple[str, ...]
    pairs: tuple[NLIPair, ...]
    requests: tuple[SecondStageRequest, ...]
    conceptual_request_count: int

    @property
    def pairs_sha256(self) -> str:
        return stable_sha256(
            [{"premise": pair.premise, "hypothesis": pair.hypothesis} for pair in self.pairs]
        )


@dataclass(frozen=True)
class ChoicePath:
    choice_label: str
    score: tuple[int, int, int, int]
    selected_pair: tuple[int, int]


@dataclass(frozen=True)
class RecipeAction:
    view_sha256: str
    recipe_id: str
    predicted_choice_label: str
    choice_paths: tuple[ChoicePath, ...]
    ordered_top5: tuple[int, int, int, int, int]
    action_sha256: str


@dataclass(frozen=True)
class ScoredRecipeItem:
    identity_commitment_sha256: str
    view_sha256: str
    recipe_id: str
    invalid: bool
    support_hits_at_5: int
    complete: bool
    U: int
    auc2: int
    top1: bool
    gold_pair: bool
    ordered_top5: tuple[int, ...]
    action_sha256: str


@dataclass(frozen=True)
class FormationSelection:
    incumbent_recipe_id: str
    challenger_recipe_id: str
    incumbent_key: tuple[object, ...]
    challenger_key: tuple[object, ...]
    same_behavior: bool


BatchScorer = Callable[[tuple[tuple[str, str], ...]], Sequence[int]]


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QASCCounterfactualError("value is not canonical JSON") from exc


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def unicode_tokens(value: str) -> tuple[str, ...]:
    """Frozen Unicode NFKC/casefold alphanumeric tokenizer."""

    if not isinstance(value, str):
        raise QASCCounterfactualError("tokenizer input must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(match.group(0) for match in _TOKEN.finditer(normalized))


def normalized_fact(value: str) -> str:
    if not isinstance(value, str):
        raise QASCCounterfactualError("fact must be text")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise QASCCounterfactualError(f"{field} must be lowercase sha256")
    return value


def _require_exact_keys(
    row: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    if not isinstance(row, Mapping) or set(row) != expected:
        raise QASCCounterfactualError(f"{field} keys do not match frozen schema")


def _require_block_member(block: object, source_member: object) -> tuple[str, str]:
    if not isinstance(block, str) or block not in _BLOCK_MEMBER:
        raise QASCCounterfactualError("block is invalid")
    if source_member != _BLOCK_MEMBER[block]:
        raise QASCCounterfactualError("source member does not match block")
    return block, str(source_member)


def load_retrieval_view(row: Mapping[str, object]) -> RetrievalView:
    _require_exact_keys(row, _VIEW_KEYS, "view")
    if row["schema"] != VIEW_SCHEMA:
        raise QASCCounterfactualError("view schema is invalid")
    block, source_member = _require_block_member(row["block"], row["source_member"])
    question = row["formatted_question"]
    if not isinstance(question, str) or not question.strip():
        raise QASCCounterfactualError("formatted question must be nonempty")

    choice_rows = row["choices"]
    if not isinstance(choice_rows, list) or len(choice_rows) != CHOICE_COUNT:
        raise QASCCounterfactualError("view must contain exactly eight choices")
    choices: list[Choice] = []
    for choice_row in choice_rows:
        if not isinstance(choice_row, Mapping) or set(choice_row) != {"label", "text"}:
            raise QASCCounterfactualError("choice keys are invalid")
        label, text = choice_row["label"], choice_row["text"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise QASCCounterfactualError("choice label and text must be nonempty")
        choices.append(Choice(label=label, text=text))
    if len({choice.label for choice in choices}) != CHOICE_COUNT:
        raise QASCCounterfactualError("choice labels must be unique")
    if len({choice.text for choice in choices}) != CHOICE_COUNT:
        raise QASCCounterfactualError("choice texts must be unique")

    document_rows = row["documents"]
    if not isinstance(document_rows, list) or len(document_rows) != DOCUMENT_COUNT:
        raise QASCCounterfactualError("view must contain exactly 32 documents")
    documents: list[RetrievalDocument] = []
    for expected_id, document_row in enumerate(document_rows):
        if not isinstance(document_row, Mapping) or set(document_row) != {
            "doc_id",
            "text",
            "bm25_score_int",
        }:
            raise QASCCounterfactualError("document keys are invalid")
        doc_id = document_row["doc_id"]
        text = document_row["text"]
        score = document_row["bm25_score_int"]
        if type(doc_id) is not int or doc_id != expected_id:
            raise QASCCounterfactualError("document IDs must be canonical contiguous order")
        if not isinstance(text, str) or not text.strip():
            raise QASCCounterfactualError("document text must be nonempty")
        if type(score) is not int:
            raise QASCCounterfactualError("BM25 score must be an integer")
        documents.append(RetrievalDocument(doc_id=doc_id, text=text, bm25_score_int=score))
    if len({normalized_fact(document.text) for document in documents}) != DOCUMENT_COUNT:
        raise QASCCounterfactualError("normalized document facts must be unique")

    raw_row = row["raw_ranking"]
    if (
        not isinstance(raw_row, list)
        or len(raw_row) != TOP_K
        or any(type(doc_id) is not int or not 0 <= doc_id < DOCUMENT_COUNT for doc_id in raw_row)
        or len(set(raw_row)) != TOP_K
    ):
        raise QASCCounterfactualError("RAW ranking must contain five unique document IDs")
    expected_raw = tuple(
        document.doc_id
        for document in sorted(documents, key=lambda document: (-document.bm25_score_int, document.doc_id))[:TOP_K]
    )
    if tuple(raw_row) != expected_raw:
        raise QASCCounterfactualError("RAW ranking does not match frozen BM25 order")
    return RetrievalView(
        schema=VIEW_SCHEMA,
        block=block,
        source_member=source_member,
        formatted_question=question,
        choices=tuple(choices),
        documents=tuple(documents),
        raw_ranking=tuple(raw_row),
    )


def load_label_envelope(row: Mapping[str, object]) -> LabelEnvelope:
    _require_exact_keys(row, _LABEL_KEYS, "label envelope")
    if row["schema"] != LABEL_SCHEMA:
        raise QASCCounterfactualError("label schema is invalid")
    block, source_member = _require_block_member(row["block"], row["source_member"])
    identity = _require_sha256(row["identity_commitment_sha256"], "identity commitment")
    view_sha256 = _require_sha256(row["view_sha256"], "view")
    answer_key = row["answerKey"]
    if not isinstance(answer_key, str) or not answer_key.strip():
        raise QASCCounterfactualError("answerKey must be nonempty")
    gold_row = row["gold_document_ids"]
    fact1, fact2 = row["fact1_document_id"], row["fact2_document_id"]
    if (
        not isinstance(gold_row, list)
        or len(gold_row) != 2
        or any(type(value) is not int or not 0 <= value < DOCUMENT_COUNT for value in gold_row)
        or len(set(gold_row)) != 2
        or type(fact1) is not int
        or type(fact2) is not int
        or list(gold_row) != sorted((fact1, fact2))
    ):
        raise QASCCounterfactualError("gold document IDs are invalid")
    return LabelEnvelope(
        schema=LABEL_SCHEMA,
        block=block,
        source_member=source_member,
        identity_commitment_sha256=identity,
        view_sha256=view_sha256,
        answerKey=answer_key,
        gold_document_ids=tuple(gold_row),
        fact1_document_id=fact1,
        fact2_document_id=fact2,
    )


def validate_view_label_binding(
    view: RetrievalView,
    label: LabelEnvelope,
) -> None:
    if (
        view.block != label.block
        or view.source_member != label.source_member
        or view.view_sha256 != label.view_sha256
        or label.answerKey not in {choice.label for choice in view.choices}
    ):
        raise QASCCounterfactualError("view and authorized label envelope do not bind")


def recipe_registry() -> tuple[RecipeSpec, ...]:
    recipes: list[RecipeSpec] = []
    for first_query in ("stem", "stem_choice"):
        for bridge_budget in (2, 4):
            for second_query in ("choice_bridge", "stem_choice_bridge"):
                for aggregation in ("bottleneck_rank", "sum_rank"):
                    recipes.append(
                        RecipeSpec(
                            recipe_id=(
                                f"q1_{first_query}__b{bridge_budget}__"
                                f"q2_{second_query}__agg_{aggregation}"
                            ),
                            first_query=first_query,
                            bridge_budget=bridge_budget,
                            second_query=second_query,
                            aggregation=aggregation,
                        )
                    )
    return tuple(recipes)


def _normalize_recipe_ids(recipe_ids: Sequence[str] | None) -> tuple[str, ...]:
    registry = recipe_registry()
    known = {recipe.recipe_id for recipe in registry}
    requested = known if recipe_ids is None else set(recipe_ids)
    if not requested or any(not isinstance(value, str) for value in requested):
        raise QASCCounterfactualError("recipe subset must be nonempty text IDs")
    if recipe_ids is not None and len(requested) != len(recipe_ids):
        raise QASCCounterfactualError("recipe subset contains duplicate IDs")
    if not requested <= known:
        raise QASCCounterfactualError("recipe subset contains an unknown ID")
    return tuple(recipe.recipe_id for recipe in registry if recipe.recipe_id in requested)


def _recipe_by_id() -> dict[str, RecipeSpec]:
    return {recipe.recipe_id: recipe for recipe in recipe_registry()}


def _first_hypothesis(view: RetrievalView, choice_index: int, mode: str) -> str:
    choice = view.choices[choice_index]
    if mode == "stem":
        return view.formatted_question
    if mode == "stem_choice":
        return f"{view.formatted_question} [CHOICE] {choice.text}"
    raise QASCCounterfactualError("first-query mode is invalid")


def _bridge_tokens(
    view: RetrievalView,
    seed_doc_id: int,
    budget: int,
) -> tuple[str, ...]:
    if budget not in {2, 4} or not 0 <= seed_doc_id < DOCUMENT_COUNT:
        raise QASCCounterfactualError("bridge request is invalid")
    document_token_sets = tuple(set(unicode_tokens(document.text)) for document in view.documents)
    seed_tokens = unicode_tokens(view.documents[seed_doc_id].text)
    first_positions: dict[str, int] = {}
    for position, token in enumerate(seed_tokens):
        if len(token) >= 3 and token not in _STOPWORDS and token not in first_positions:
            first_positions[token] = position
    ranked = sorted(
        first_positions,
        key=lambda token: (
            -(
                math.log(
                    33
                    / (
                        sum(token in document_tokens for document_tokens in document_token_sets)
                        + 1
                    )
                )
                + 1
            ),
            first_positions[token],
            token.encode("utf-8"),
        ),
    )
    return tuple(ranked[:budget])


def bridge_tokens(
    view: RetrievalView,
    seed_doc_id: int,
    budget: int,
) -> tuple[str, ...]:
    """Expose the frozen label-free bridge selector for audit and tests."""

    return _bridge_tokens(view, seed_doc_id, budget)


def _second_hypothesis(
    view: RetrievalView,
    choice_index: int,
    seed_doc_id: int,
    bridge_budget: int,
    mode: str,
) -> str:
    choice = view.choices[choice_index]
    bridge = " ".join(_bridge_tokens(view, seed_doc_id, bridge_budget))
    if mode == "choice_bridge":
        return f"{choice.text} [BRIDGE] {bridge}"
    if mode == "stem_choice_bridge":
        return (
            f"{view.formatted_question} [CHOICE] {choice.text} "
            f"[BRIDGE] {bridge}"
        )
    raise QASCCounterfactualError("second-query mode is invalid")


def _pair_index(
    pairs: list[NLIPair],
    indices: dict[NLIPair, int],
    pair: NLIPair,
) -> int:
    existing = indices.get(pair)
    if existing is not None:
        return existing
    index = len(pairs)
    pairs.append(pair)
    indices[pair] = index
    return index


def _validated_score_vector(
    plan_pairs: Sequence[NLIPair], scores: Sequence[int], field: str
) -> tuple[int, ...]:
    normalized = tuple(scores)
    if len(normalized) != len(plan_pairs) or any(type(value) is not int for value in normalized):
        raise QASCCounterfactualError(f"{field} must contain one integer per unique NLI pair")
    return normalized


def path_score_tuple(
    aggregation: str,
    *,
    first_rank: int,
    second_rank: int,
    first_margin: int,
    second_margin: int,
) -> tuple[int, int, int, int]:
    """Return the exact four-integer choice/pair comparison tuple."""

    if (
        aggregation not in {"sum_rank", "bottleneck_rank"}
        or type(first_rank) is not int
        or type(second_rank) is not int
        or first_rank <= 0
        or second_rank <= 0
        or type(first_margin) is not int
        or type(second_margin) is not int
    ):
        raise QASCCounterfactualError("path-score inputs are invalid")
    if aggregation == "sum_rank":
        return (
            -(first_rank + second_rank),
            -max(first_rank, second_rank),
            first_margin + second_margin,
            min(first_margin, second_margin),
        )
    return (
        -max(first_rank, second_rank),
        -(first_rank + second_rank),
        min(first_margin, second_margin),
        first_margin + second_margin,
    )


def build_first_stage_plan(
    view: RetrievalView,
    recipe_ids: Sequence[str] | None = None,
) -> FirstStagePlan:
    if not isinstance(view, RetrievalView):
        raise QASCCounterfactualError("first-stage view is invalid")
    normalized_ids = _normalize_recipe_ids(recipe_ids)
    recipes = _recipe_by_id()
    modes = tuple(
        mode
        for mode in ("stem", "stem_choice")
        if any(recipes[recipe_id].first_query == mode for recipe_id in normalized_ids)
    )
    pairs: list[NLIPair] = []
    indices: dict[NLIPair, int] = {}
    requests: list[FirstStageRequest] = []
    for mode in modes:
        for choice_index in range(CHOICE_COUNT):
            hypothesis = _first_hypothesis(view, choice_index, mode)
            for document in view.documents:
                pair = NLIPair(premise=document.text, hypothesis=hypothesis)
                requests.append(
                    FirstStageRequest(
                        first_query=mode,
                        choice_index=choice_index,
                        doc_id=document.doc_id,
                        pair_index=_pair_index(pairs, indices, pair),
                    )
                )
    return FirstStagePlan(
        view_sha256=view.view_sha256,
        recipe_ids=normalized_ids,
        pairs=tuple(pairs),
        requests=tuple(requests),
        conceptual_request_count=len(normalized_ids) * CHOICE_COUNT * DOCUMENT_COUNT,
    )


def _first_score_lookup(
    plan: FirstStagePlan,
    scores: Sequence[int],
) -> dict[tuple[str, int, int], int]:
    normalized = _validated_score_vector(plan.pairs, scores, "first-stage score vector")
    lookup: dict[tuple[str, int, int], int] = {}
    for request in plan.requests:
        key = (request.first_query, request.choice_index, request.doc_id)
        if key in lookup:
            raise QASCCounterfactualError("duplicate first-stage logical request")
        lookup[key] = normalized[request.pair_index]
    return lookup


def _top_seed_ids(
    lookup: Mapping[tuple[str, int, int], int],
    mode: str,
    choice_index: int,
) -> tuple[int, ...]:
    try:
        return tuple(
            sorted(
                range(DOCUMENT_COUNT),
                key=lambda doc_id: (-lookup[(mode, choice_index, doc_id)], doc_id),
            )[:SEED_EXPANSION_COUNT]
        )
    except KeyError as exc:
        raise QASCCounterfactualError("first-stage logical score is missing") from exc


def build_second_stage_plan(
    view: RetrievalView,
    recipe_ids: Sequence[str] | None,
    first_score_vector: Sequence[int],
    first_plan: FirstStagePlan | None = None,
) -> SecondStagePlan:
    normalized_ids = (
        first_plan.recipe_ids
        if recipe_ids is None and first_plan is not None
        else _normalize_recipe_ids(recipe_ids)
    )
    if first_plan is None:
        first_plan = build_first_stage_plan(view, normalized_ids)
    if first_plan.view_sha256 != view.view_sha256 or first_plan.recipe_ids != normalized_ids:
        raise QASCCounterfactualError("first-stage plan does not bind requested view and recipes")
    lookup = _first_score_lookup(first_plan, first_score_vector)
    recipes = _recipe_by_id()
    semantic_factors: list[tuple[str, int, str]] = []
    for recipe_id in normalized_ids:
        recipe = recipes[recipe_id]
        factor = (recipe.first_query, recipe.bridge_budget, recipe.second_query)
        if factor not in semantic_factors:
            semantic_factors.append(factor)

    pairs: list[NLIPair] = []
    indices: dict[NLIPair, int] = {}
    requests: list[SecondStageRequest] = []
    for first_query, bridge_budget, second_query in semantic_factors:
        for choice_index in range(CHOICE_COUNT):
            for seed_doc_id in _top_seed_ids(lookup, first_query, choice_index):
                hypothesis = _second_hypothesis(
                    view,
                    choice_index,
                    seed_doc_id,
                    bridge_budget,
                    second_query,
                )
                for document in view.documents:
                    if document.doc_id == seed_doc_id:
                        continue
                    pair = NLIPair(premise=document.text, hypothesis=hypothesis)
                    requests.append(
                        SecondStageRequest(
                            first_query=first_query,
                            bridge_budget=bridge_budget,
                            second_query=second_query,
                            choice_index=choice_index,
                            seed_doc_id=seed_doc_id,
                            second_doc_id=document.doc_id,
                            pair_index=_pair_index(pairs, indices, pair),
                        )
                    )
    return SecondStagePlan(
        view_sha256=view.view_sha256,
        recipe_ids=normalized_ids,
        pairs=tuple(pairs),
        requests=tuple(requests),
        conceptual_request_count=(
            len(normalized_ids)
            * CHOICE_COUNT
            * SEED_EXPANSION_COUNT
            * (DOCUMENT_COUNT - 1)
        ),
    )


def consume_stage_scores(
    view: RetrievalView,
    first_plan: FirstStagePlan,
    first_score_vector: Sequence[int],
    second_plan: SecondStagePlan,
    second_score_vector: Sequence[int],
    recipe_ids: Sequence[str] | None = None,
) -> tuple[RecipeAction, ...]:
    normalized_ids = first_plan.recipe_ids if recipe_ids is None else _normalize_recipe_ids(recipe_ids)
    if (
        first_plan.view_sha256 != view.view_sha256
        or second_plan.view_sha256 != view.view_sha256
        or not set(normalized_ids) <= set(first_plan.recipe_ids)
        or not set(normalized_ids) <= set(second_plan.recipe_ids)
    ):
        raise QASCCounterfactualError("NLI plans do not cover requested view and recipes")
    first_lookup = _first_score_lookup(first_plan, first_score_vector)
    second_scores = _validated_score_vector(
        second_plan.pairs, second_score_vector, "second-stage score vector"
    )
    second_lookup: dict[tuple[str, int, str, int, int, int], int] = {}
    for request in second_plan.requests:
        key = (
            request.first_query,
            request.bridge_budget,
            request.second_query,
            request.choice_index,
            request.seed_doc_id,
            request.second_doc_id,
        )
        if key in second_lookup:
            raise QASCCounterfactualError("duplicate second-stage logical request")
        second_lookup[key] = second_scores[request.pair_index]

    recipes = _recipe_by_id()
    actions: list[RecipeAction] = []
    for recipe_id in normalized_ids:
        recipe = recipes[recipe_id]
        choice_paths: list[ChoicePath] = []
        first_rankings: list[tuple[int, ...]] = []
        second_ranking_by_choice_seed: dict[tuple[int, int], tuple[int, ...]] = {}
        for choice_index, choice in enumerate(view.choices):
            first_ranking = tuple(
                sorted(
                    range(DOCUMENT_COUNT),
                    key=lambda doc_id: (
                        -first_lookup[(recipe.first_query, choice_index, doc_id)],
                        doc_id,
                    ),
                )
            )
            first_rankings.append(first_ranking)
            first_rank = {doc_id: rank for rank, doc_id in enumerate(first_ranking, 1)}
            best_score: tuple[int, int, int, int] | None = None
            best_pair: tuple[int, int] | None = None
            for seed_doc_id in first_ranking[:SEED_EXPANSION_COUNT]:
                second_ranking = tuple(
                    sorted(
                        (doc_id for doc_id in range(DOCUMENT_COUNT) if doc_id != seed_doc_id),
                        key=lambda doc_id: (
                            -second_lookup[
                                (
                                    recipe.first_query,
                                    recipe.bridge_budget,
                                    recipe.second_query,
                                    choice_index,
                                    seed_doc_id,
                                    doc_id,
                                )
                            ],
                            doc_id,
                        ),
                    )
                )
                second_ranking_by_choice_seed[(choice_index, seed_doc_id)] = second_ranking
                for second_rank, second_doc_id in enumerate(second_ranking, 1):
                    r1 = first_rank[seed_doc_id]
                    r2 = second_rank
                    m1 = first_lookup[(recipe.first_query, choice_index, seed_doc_id)]
                    m2 = second_lookup[
                        (
                            recipe.first_query,
                            recipe.bridge_budget,
                            recipe.second_query,
                            choice_index,
                            seed_doc_id,
                            second_doc_id,
                        )
                    ]
                    score = path_score_tuple(
                        recipe.aggregation,
                        first_rank=r1,
                        second_rank=r2,
                        first_margin=m1,
                        second_margin=m2,
                    )
                    pair = (seed_doc_id, second_doc_id)
                    if (
                        best_score is None
                        or score > best_score
                        or (score == best_score and pair < best_pair)
                    ):
                        best_score, best_pair = score, pair
            if best_score is None or best_pair is None:
                raise QASCCounterfactualError("recipe failed to produce a choice path")
            choice_paths.append(
                ChoicePath(
                    choice_label=choice.label,
                    score=best_score,
                    selected_pair=best_pair,
                )
            )

        winning_score = max(path.score for path in choice_paths)
        winning_index = next(
            index for index, path in enumerate(choice_paths) if path.score == winning_score
        )
        winning_path = choice_paths[winning_index]
        selected_first, selected_second = winning_path.selected_pair
        first_ranking = first_rankings[winning_index]
        second_ranking = second_ranking_by_choice_seed[(winning_index, selected_first)]
        first_rank = {doc_id: rank for rank, doc_id in enumerate(first_ranking, 1)}
        second_rank = {doc_id: rank for rank, doc_id in enumerate(second_ranking, 1)}
        remaining = sorted(
            (
                doc_id
                for doc_id in range(DOCUMENT_COUNT)
                if doc_id not in {selected_first, selected_second}
            ),
            key=lambda doc_id: (
                -exact_rrf_score(first_rank[doc_id], second_rank[doc_id]),
                doc_id,
            ),
        )
        ordered_top5 = (selected_first, selected_second, *remaining[:3])
        action_body = {
            "view_sha256": view.view_sha256,
            "recipe_id": recipe_id,
            "predicted_choice_label": view.choices[winning_index].label,
            "choice_paths": [
                {
                    "choice_label": path.choice_label,
                    "score": list(path.score),
                    "selected_pair": list(path.selected_pair),
                }
                for path in choice_paths
            ],
            "ordered_top5": list(ordered_top5),
        }
        actions.append(
            RecipeAction(
                view_sha256=view.view_sha256,
                recipe_id=recipe_id,
                predicted_choice_label=view.choices[winning_index].label,
                choice_paths=tuple(choice_paths),
                ordered_top5=ordered_top5,
                action_sha256=stable_sha256(action_body),
            )
        )
    return tuple(actions)


def execute_recipes(
    view: RetrievalView,
    scorer: BatchScorer,
    recipe_ids: Sequence[str] | None = None,
) -> tuple[RecipeAction, ...]:
    first_plan = build_first_stage_plan(view, recipe_ids)
    first_scores = tuple(scorer(tuple(pair.as_tuple() for pair in first_plan.pairs)))
    second_plan = build_second_stage_plan(
        view,
        first_plan.recipe_ids,
        first_scores,
        first_plan,
    )
    second_scores = tuple(scorer(tuple(pair.as_tuple() for pair in second_plan.pairs)))
    return consume_stage_scores(
        view,
        first_plan,
        first_scores,
        second_plan,
        second_scores,
        first_plan.recipe_ids,
    )


def score_recipe_action(
    view: RetrievalView,
    action: RecipeAction,
    label: LabelEnvelope,
) -> ScoredRecipeItem:
    validate_view_label_binding(view, label)
    if (
        action.view_sha256 != view.view_sha256
        or action.recipe_id not in _recipe_by_id()
        or len(action.choice_paths) != CHOICE_COUNT
        or tuple(path.choice_label for path in action.choice_paths)
        != tuple(choice.label for choice in view.choices)
        or len(action.ordered_top5) != TOP_K
        or len(set(action.ordered_top5)) != TOP_K
        or any(not 0 <= doc_id < DOCUMENT_COUNT for doc_id in action.ordered_top5)
    ):
        raise QASCCounterfactualError("recipe action does not bind the view")
    for path in action.choice_paths:
        if (
            len(path.score) != 4
            or any(type(value) is not int for value in path.score)
            or len(set(path.selected_pair)) != 2
            or any(not 0 <= doc_id < DOCUMENT_COUNT for doc_id in path.selected_pair)
        ):
            raise QASCCounterfactualError("choice path is invalid")
    winning_score = max(path.score for path in action.choice_paths)
    winning_index = next(
        index for index, path in enumerate(action.choice_paths) if path.score == winning_score
    )
    if action.predicted_choice_label != view.choices[winning_index].label:
        raise QASCCounterfactualError("predicted choice violates frozen tie order")
    action_body = {
        "view_sha256": action.view_sha256,
        "recipe_id": action.recipe_id,
        "predicted_choice_label": action.predicted_choice_label,
        "choice_paths": [
            {
                "choice_label": path.choice_label,
                "score": list(path.score),
                "selected_pair": list(path.selected_pair),
            }
            for path in action.choice_paths
        ],
        "ordered_top5": list(action.ordered_top5),
    }
    if stable_sha256(action_body) != action.action_sha256:
        raise QASCCounterfactualError("recipe action hash is invalid")

    correct_index = next(
        index for index, choice in enumerate(view.choices) if choice.label == label.answerKey
    )
    correct_path = action.choice_paths[correct_index]
    wrong_paths = tuple(
        path for index, path in enumerate(action.choice_paths) if index != correct_index
    )
    auc2 = 2 * sum(correct_path.score > path.score for path in wrong_paths) + sum(
        correct_path.score == path.score for path in wrong_paths
    )
    top1 = all(correct_path.score > path.score for path in wrong_paths)
    gold_pair = set(correct_path.selected_pair) == set(label.gold_document_ids)
    support_hits = len(set(action.ordered_top5) & set(label.gold_document_ids))
    complete = support_hits == 2
    utility = support_hits + int(complete)
    return ScoredRecipeItem(
        identity_commitment_sha256=label.identity_commitment_sha256,
        view_sha256=view.view_sha256,
        recipe_id=action.recipe_id,
        invalid=False,
        support_hits_at_5=support_hits,
        complete=complete,
        U=utility,
        auc2=auc2,
        top1=top1,
        gold_pair=gold_pair,
        ordered_top5=action.ordered_top5,
        action_sha256=action.action_sha256,
    )


def invalid_scored_item(
    *,
    identity_commitment_sha256: str,
    view_sha256: str,
    recipe_id: str,
) -> ScoredRecipeItem:
    """Create a fail-closed formation item after an already-recorded error."""

    _require_sha256(identity_commitment_sha256, "identity commitment")
    _require_sha256(view_sha256, "view")
    if recipe_id not in _recipe_by_id():
        raise QASCCounterfactualError("invalid item recipe is unknown")
    return ScoredRecipeItem(
        identity_commitment_sha256=identity_commitment_sha256,
        view_sha256=view_sha256,
        recipe_id=recipe_id,
        invalid=True,
        support_hits_at_5=0,
        complete=False,
        U=0,
        auc2=0,
        top1=False,
        gold_pair=False,
        ordered_top5=(),
        action_sha256=stable_sha256(
            {
                "invalid": True,
                "identity_commitment_sha256": identity_commitment_sha256,
                "view_sha256": view_sha256,
                "recipe_id": recipe_id,
            }
        ),
    )


def assign_hmac_folds(
    identity_commitments: Sequence[str],
    selection_secret: bytes,
    *,
    block: str,
    domain_separator: str = DOMAIN_SEPARATOR,
) -> dict[str, int]:
    identities = tuple(identity_commitments)
    if len(identities) != 64 or len(set(identities)) != 64:
        raise QASCCounterfactualError("formal fold assignment requires 64 unique items")
    for identity in identities:
        _require_sha256(identity, "identity commitment")
    if not isinstance(selection_secret, bytes) or len(selection_secret) != 32:
        raise QASCCounterfactualError("selection secret must be exactly 32 bytes")
    if block not in {"A_form", "F_search"}:
        raise QASCCounterfactualError("fold block must be A_form or F_search")
    if not isinstance(domain_separator, str) or not domain_separator:
        raise QASCCounterfactualError("fold domain separator is invalid")
    ranked = sorted(
        identities,
        key=lambda identity: (
            hmac.new(
                selection_secret,
                f"{domain_separator}\0fold\0{block}\0{identity}".encode("utf-8"),
                hashlib.sha256,
            ).digest(),
            identity,
        ),
    )
    return {identity: ordinal % FOLD_COUNT for ordinal, identity in enumerate(ranked)}


def _validate_scored_item(item: ScoredRecipeItem, recipe_id: str) -> None:
    _require_sha256(item.identity_commitment_sha256, "identity commitment")
    _require_sha256(item.view_sha256, "view")
    _require_sha256(item.action_sha256, "action")
    if item.recipe_id != recipe_id or type(item.invalid) is not bool:
        raise QASCCounterfactualError("scored item recipe or invalid flag is wrong")
    if item.invalid:
        if (
            item.support_hits_at_5 != 0
            or item.complete is not False
            or item.U != 0
            or item.auc2 != 0
            or item.top1 is not False
            or item.gold_pair is not False
            or item.ordered_top5
        ):
            raise QASCCounterfactualError("invalid scored item must have zero metrics")
        return
    if (
        type(item.support_hits_at_5) is not int
        or not 0 <= item.support_hits_at_5 <= 2
        or type(item.complete) is not bool
        or item.complete != (item.support_hits_at_5 == 2)
        or type(item.U) is not int
        or item.U != item.support_hits_at_5 + int(item.complete)
        or type(item.auc2) is not int
        or not 0 <= item.auc2 <= 14
        or type(item.top1) is not bool
        or item.top1 != (item.auc2 == 14)
        or type(item.gold_pair) is not bool
        or len(item.ordered_top5) != TOP_K
        or len(set(item.ordered_top5)) != TOP_K
        or any(not 0 <= doc_id < DOCUMENT_COUNT for doc_id in item.ordered_top5)
    ):
        raise QASCCounterfactualError("valid scored item metrics are inconsistent")


def select_formation_recipes(
    evidence_by_recipe: Mapping[str, Sequence[ScoredRecipeItem]],
    fold_by_identity: Mapping[str, int],
) -> FormationSelection:
    registry_ids = tuple(recipe.recipe_id for recipe in recipe_registry())
    if set(evidence_by_recipe) != set(registry_ids):
        raise QASCCounterfactualError("formation evidence must cover all 16 recipes")
    normalized: dict[str, dict[str, ScoredRecipeItem]] = {}
    identity_set: set[str] | None = None
    for recipe_id in registry_ids:
        items = tuple(evidence_by_recipe[recipe_id])
        if len(items) != 64:
            raise QASCCounterfactualError("each formation recipe requires 64 items")
        by_identity: dict[str, ScoredRecipeItem] = {}
        for item in items:
            _validate_scored_item(item, recipe_id)
            if item.identity_commitment_sha256 in by_identity:
                raise QASCCounterfactualError("duplicate formation identity")
            by_identity[item.identity_commitment_sha256] = item
        if identity_set is None:
            identity_set = set(by_identity)
        elif set(by_identity) != identity_set:
            raise QASCCounterfactualError("recipe formation item sets differ")
        normalized[recipe_id] = by_identity
    assert identity_set is not None
    if set(fold_by_identity) != identity_set or any(
        type(fold) is not int or not 0 <= fold < FOLD_COUNT
        for fold in fold_by_identity.values()
    ):
        raise QASCCounterfactualError("fold map does not match formation items")
    if [sum(fold == expected for fold in fold_by_identity.values()) for expected in range(4)] != [16] * 4:
        raise QASCCounterfactualError("formation folds must contain exactly 16 items each")

    incumbent_rows: list[tuple[tuple[object, ...], str]] = []
    challenger_rows: list[tuple[tuple[object, ...], str]] = []
    natural_incumbent: dict[str, tuple[object, ...]] = {}
    natural_challenger: dict[str, tuple[object, ...]] = {}
    for recipe_id in registry_ids:
        items = normalized[recipe_id]
        invalid = sum(item.invalid for item in items.values())
        fold_hits = [
            sum(
                item.support_hits_at_5
                for identity, item in items.items()
                if fold_by_identity[identity] == fold
            )
            for fold in range(FOLD_COUNT)
        ]
        fold_complete = [
            sum(
                item.complete
                for identity, item in items.items()
                if fold_by_identity[identity] == fold
            )
            for fold in range(FOLD_COUNT)
        ]
        fold_auc2 = [
            sum(item.auc2 for identity, item in items.items() if fold_by_identity[identity] == fold)
            for fold in range(FOLD_COUNT)
        ]
        fold_top1 = [
            sum(item.top1 for identity, item in items.items() if fold_by_identity[identity] == fold)
            for fold in range(FOLD_COUNT)
        ]
        fold_gold_pair = [
            sum(item.gold_pair for identity, item in items.items() if fold_by_identity[identity] == fold)
            for fold in range(FOLD_COUNT)
        ]
        fold_utility = [
            sum(item.U for identity, item in items.items() if fold_by_identity[identity] == fold)
            for fold in range(FOLD_COUNT)
        ]
        inc_natural = (
            invalid,
            min(fold_hits),
            sum(fold_hits),
            min(fold_complete),
            sum(fold_complete),
            recipe_id,
        )
        chal_natural = (
            invalid,
            min(fold_auc2),
            sum(fold_auc2),
            min(fold_top1),
            sum(fold_top1),
            min(fold_gold_pair),
            sum(fold_gold_pair),
            min(fold_utility),
            sum(fold_utility),
            recipe_id,
        )
        natural_incumbent[recipe_id] = inc_natural
        natural_challenger[recipe_id] = chal_natural
        incumbent_rows.append(
            ((invalid, -min(fold_hits), -sum(fold_hits), -min(fold_complete), -sum(fold_complete), recipe_id), recipe_id)
        )
        challenger_rows.append(
            (
                (
                    invalid,
                    -min(fold_auc2),
                    -sum(fold_auc2),
                    -min(fold_top1),
                    -sum(fold_top1),
                    -min(fold_gold_pair),
                    -sum(fold_gold_pair),
                    -min(fold_utility),
                    -sum(fold_utility),
                    recipe_id,
                ),
                recipe_id,
            )
        )
    incumbent_id = min(incumbent_rows)[1]
    challenger_id = min(challenger_rows)[1]
    same_behavior = all(
        normalized[incumbent_id][identity].ordered_top5
        == normalized[challenger_id][identity].ordered_top5
        for identity in identity_set
    )
    return FormationSelection(
        incumbent_recipe_id=incumbent_id,
        challenger_recipe_id=challenger_id,
        incumbent_key=natural_incumbent[incumbent_id],
        challenger_key=natural_challenger[challenger_id],
        same_behavior=same_behavior,
    )


def exact_rrf_score(first_rank: int, second_rank: int) -> Fraction:
    """Expose the frozen exact RRF primitive for independent verification."""

    return Fraction(1, 60 + first_rank) + Fraction(1, 60 + second_rank)

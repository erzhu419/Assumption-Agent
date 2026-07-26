"""Pure typed evidence-set action and evaluator core for AVeriTeC P1.

The core has no file, source-loader, network, model-loader, API, retry, or
benchmark-label entrypoint.  It consumes already-quantized local MiniLM
coordinates and constructs five-document evidence sets through a closed
relation-slot assignment grammar.  The evaluator challenger is fitted once
from sealed A_form action slates and exact offline utility.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
import re
import unicodedata
from typing import Mapping, Sequence

import numpy as np


STUDY_ID = "AVERITEC_P1_TYPED_QA_SET_EVALUATOR_V1"
VERSION = "averitec_p1_typed_core_v1"
SCALE = 1_000_000
TOP_K = 5
RIDGE_L2 = 1.0
PROMOTION_ALPHA = Fraction(1, 10)

DIRECT = "DIRECT"
CAUSE = "CAUSE"
EFFECT = "EFFECT"
QUOTE = "QUOTE"
SOURCE = "SOURCE"
NUMBER = "NUMBER"
COMPARE = "COMPARE"
CONTEXT = "CONTEXT"
QUERY_VARIANT_IDS = (
    DIRECT,
    CAUSE,
    EFFECT,
    QUOTE,
    SOURCE,
    NUMBER,
    COMPARE,
    CONTEXT,
)

R0_DIRECT_DENSE = "R0_DIRECT_DENSE"
R1_CAUSAL_CHAIN = "R1_CAUSAL_CHAIN"
R2_QUOTE_ATTRIBUTION = "R2_QUOTE_ATTRIBUTION"
R3_NUMERICAL_COMPARISON = "R3_NUMERICAL_COMPARISON"
R4_CAUSAL_QUOTE = "R4_CAUSAL_QUOTE"
R5_CAUSAL_NUMERICAL = "R5_CAUSAL_NUMERICAL"
R6_QUOTE_NUMERICAL = "R6_QUOTE_NUMERICAL"
R7_ALL_TYPED = "R7_ALL_TYPED"
RECIPE_IDS = (
    R0_DIRECT_DENSE,
    R1_CAUSAL_CHAIN,
    R2_QUOTE_ATTRIBUTION,
    R3_NUMERICAL_COMPARISON,
    R4_CAUSAL_QUOTE,
    R5_CAUSAL_NUMERICAL,
    R6_QUOTE_NUMERICAL,
    R7_ALL_TYPED,
)
RECIPE_SLOTS: dict[str, tuple[str, ...]] = {
    R0_DIRECT_DENSE: (DIRECT, DIRECT, DIRECT, DIRECT, DIRECT),
    R1_CAUSAL_CHAIN: (CAUSE, EFFECT, SOURCE, CONTEXT, DIRECT),
    R2_QUOTE_ATTRIBUTION: (QUOTE, SOURCE, CONTEXT, DIRECT, EFFECT),
    R3_NUMERICAL_COMPARISON: (NUMBER, COMPARE, CONTEXT, SOURCE, DIRECT),
    R4_CAUSAL_QUOTE: (CAUSE, EFFECT, QUOTE, SOURCE, DIRECT),
    R5_CAUSAL_NUMERICAL: (CAUSE, EFFECT, NUMBER, COMPARE, DIRECT),
    R6_QUOTE_NUMERICAL: (QUOTE, SOURCE, NUMBER, COMPARE, DIRECT),
    R7_ALL_TYPED: (CAUSE, EFFECT, QUOTE, NUMBER, COMPARE),
}

FEATURE_NAMES = (
    "mean_direct_selected",
    "minimum_direct_selected",
    "mean_slot_selected",
    "minimum_slot_selected",
    "mean_slot_minus_direct",
    "raw_overlap_fraction",
    "pairwise_lexical_diversity",
    "typed_slot_fraction",
)
MODEL_FEATURE_COUNT = 1 + len(RECIPE_IDS) + len(FEATURE_NAMES)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class AveritecP1TypedCoreError(ValueError):
    """The frozen typed action, evaluator, or exact metric drifted."""


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AveritecP1TypedCoreError(
            "typed-core value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise AveritecP1TypedCoreError("text input is not a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(value)))


def typed_query_variants(query: str) -> dict[str, str]:
    normalized = " ".join(query.split())
    if not normalized or "\x00" in normalized or len(normalized) > 4_000:
        raise AveritecP1TypedCoreError("query text is invalid")
    prefixes = {
        DIRECT: "",
        CAUSE: "What cause, reason, or prior event determines this claim? ",
        EFFECT: "What result, consequence, or outcome follows in this claim? ",
        QUOTE: "What exact statement or quotation is being verified? ",
        SOURCE: "Who made, reported, or witnessed the statement and in what context? ",
        NUMBER: "What quantities, dates, percentages, rates, or amounts establish this claim? ",
        COMPARE: "Which values or measurements must be compared to verify this claim? ",
        CONTEXT: "What entity, place, time, definition, or background fact resolves this claim? ",
    }
    return {
        variant: prefixes[variant] + normalized
        for variant in QUERY_VARIANT_IDS
    }


def _strict_coordinate(value: object, field: str) -> int:
    if type(value) is not int or not 0 <= value <= SCALE:
        raise AveritecP1TypedCoreError(
            f"{field} is not a million-scale unit coordinate"
        )
    return int(value)


def _mean_integer(values: Sequence[int]) -> int:
    if not values:
        raise AveritecP1TypedCoreError("cannot average an empty vector")
    return int(round(sum(values) / len(values)))


def _lexical_diversity(
    document_texts: Sequence[str], ordinals: Sequence[int]
) -> int:
    token_sets = [set(lexical_tokens(document_texts[index])) for index in ordinals]
    distances: list[Fraction] = []
    for left in range(len(token_sets)):
        for right in range(left + 1, len(token_sets)):
            union = token_sets[left] | token_sets[right]
            if not union:
                distances.append(Fraction(0, 1))
            else:
                distances.append(
                    Fraction(
                        len(union - (token_sets[left] & token_sets[right])),
                        len(union),
                    )
                )
    if not distances:
        return 0
    mean = sum(distances, Fraction(0, 1)) / len(distances)
    return int(round(float(mean) * SCALE))


@dataclass(frozen=True)
class RecipeAction:
    recipe_id: str
    top5_document_ordinals: tuple[int, ...]
    assigned_slots: tuple[str, ...]
    assigned_slot_scores: tuple[int, ...]
    selected_direct_scores: tuple[int, ...]
    raw_top5_document_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise AveritecP1TypedCoreError("recipe_id is not frozen")
        if (
            len(self.top5_document_ordinals) != TOP_K
            or len(set(self.top5_document_ordinals)) != TOP_K
            or any(type(value) is not int or value < 0 for value in self.top5_document_ordinals)
        ):
            raise AveritecP1TypedCoreError("typed top5 is malformed")
        if self.assigned_slots != RECIPE_SLOTS[self.recipe_id]:
            raise AveritecP1TypedCoreError("assigned slots drifted")
        if len(self.assigned_slot_scores) != TOP_K or len(
            self.selected_direct_scores
        ) != TOP_K:
            raise AveritecP1TypedCoreError("typed action coordinates drifted")
        for index, value in enumerate(self.assigned_slot_scores):
            _strict_coordinate(value, f"assigned_slot_scores[{index}]")
        for index, value in enumerate(self.selected_direct_scores):
            _strict_coordinate(value, f"selected_direct_scores[{index}]")
        if (
            len(self.raw_top5_document_ordinals) != TOP_K
            or len(set(self.raw_top5_document_ordinals)) != TOP_K
        ):
            raise AveritecP1TypedCoreError("raw top5 is malformed")


def _stable_top5(scores: Sequence[int]) -> tuple[int, ...]:
    if len(scores) < TOP_K:
        raise AveritecP1TypedCoreError("document corpus is smaller than top_k")
    checked = [
        _strict_coordinate(score, f"scores[{index}]")
        for index, score in enumerate(scores)
    ]
    return tuple(
        sorted(range(len(checked)), key=lambda index: (-checked[index], index))[
            :TOP_K
        ]
    )


def _typed_assignment(
    *,
    slots: Sequence[str],
    direct_scores: Sequence[int],
    variant_scores: Mapping[str, Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(slots) != TOP_K:
        raise AveritecP1TypedCoreError("recipe does not contain five slots")
    document_count = len(direct_scores)
    if document_count < TOP_K:
        raise AveritecP1TypedCoreError("typed corpus is too small")
    candidates: list[tuple[int, int, int, int, int]] = []
    for slot_index, slot in enumerate(slots):
        scores = variant_scores.get(slot)
        if scores is None or len(scores) != document_count:
            raise AveritecP1TypedCoreError("typed coordinate table drifted")
        for document_index, slot_score in enumerate(scores):
            checked_slot = _strict_coordinate(
                slot_score, f"{slot}[{document_index}]"
            )
            checked_direct = _strict_coordinate(
                direct_scores[document_index],
                f"{DIRECT}[{document_index}]",
            )
            combined = 3 * checked_slot + 2 * checked_direct
            candidates.append(
                (
                    -combined,
                    slot_index,
                    document_index,
                    checked_slot,
                    checked_direct,
                )
            )
    candidates.sort()
    assigned: dict[int, tuple[int, int, int]] = {}
    used_documents: set[int] = set()
    for _negative, slot_index, document_index, slot_score, direct_score in candidates:
        if slot_index in assigned or document_index in used_documents:
            continue
        assigned[slot_index] = (document_index, slot_score, direct_score)
        used_documents.add(document_index)
        if len(assigned) == TOP_K:
            break
    if len(assigned) != TOP_K:
        raise AveritecP1TypedCoreError("typed slot assignment is incomplete")
    ordered = [assigned[index] for index in range(TOP_K)]
    return (
        tuple(row[0] for row in ordered),
        tuple(row[1] for row in ordered),
        tuple(row[2] for row in ordered),
    )


def materialize_recipe_actions(
    *,
    document_texts: Sequence[str],
    variant_scores: Mapping[str, Sequence[int]],
) -> dict[str, RecipeAction]:
    if isinstance(document_texts, (str, bytes)) or len(document_texts) < TOP_K:
        raise AveritecP1TypedCoreError("document text corpus is malformed")
    document_count = len(document_texts)
    if set(variant_scores) != set(QUERY_VARIANT_IDS):
        raise AveritecP1TypedCoreError("query variant coordinate set drifted")
    for variant in QUERY_VARIANT_IDS:
        if len(variant_scores[variant]) != document_count:
            raise AveritecP1TypedCoreError("query variant row width drifted")
    direct_scores = tuple(
        _strict_coordinate(value, f"DIRECT[{index}]")
        for index, value in enumerate(variant_scores[DIRECT])
    )
    raw_top5 = _stable_top5(direct_scores)
    actions: dict[str, RecipeAction] = {}
    for recipe_id in RECIPE_IDS:
        slots = RECIPE_SLOTS[recipe_id]
        if recipe_id == R0_DIRECT_DENSE:
            ordinals = raw_top5
            slot_scores = tuple(direct_scores[index] for index in ordinals)
            selected_direct = slot_scores
        else:
            ordinals, slot_scores, selected_direct = _typed_assignment(
                slots=slots,
                direct_scores=direct_scores,
                variant_scores=variant_scores,
            )
        actions[recipe_id] = RecipeAction(
            recipe_id=recipe_id,
            top5_document_ordinals=ordinals,
            assigned_slots=slots,
            assigned_slot_scores=slot_scores,
            selected_direct_scores=selected_direct,
            raw_top5_document_ordinals=raw_top5,
        )
    return actions


def action_payload(action: RecipeAction) -> dict[str, object]:
    return {
        "assigned_slot_scores": list(action.assigned_slot_scores),
        "assigned_slots": list(action.assigned_slots),
        "raw_top5_document_ordinals": list(action.raw_top5_document_ordinals),
        "recipe_id": action.recipe_id,
        "selected_direct_scores": list(action.selected_direct_scores),
        "top5_document_ordinals": list(action.top5_document_ordinals),
    }


def action_from_payload(value: object) -> RecipeAction:
    if not isinstance(value, Mapping) or set(value) != {
        "assigned_slot_scores",
        "assigned_slots",
        "raw_top5_document_ordinals",
        "recipe_id",
        "selected_direct_scores",
        "top5_document_ordinals",
    }:
        raise AveritecP1TypedCoreError("action payload shape drifted")
    try:
        return RecipeAction(
            recipe_id=str(value["recipe_id"]),
            top5_document_ordinals=tuple(value["top5_document_ordinals"]),  # type: ignore[arg-type]
            assigned_slots=tuple(value["assigned_slots"]),  # type: ignore[arg-type]
            assigned_slot_scores=tuple(value["assigned_slot_scores"]),  # type: ignore[arg-type]
            selected_direct_scores=tuple(value["selected_direct_scores"]),  # type: ignore[arg-type]
            raw_top5_document_ordinals=tuple(
                value["raw_top5_document_ordinals"]  # type: ignore[arg-type]
            ),
        )
    except (KeyError, TypeError) as exc:
        raise AveritecP1TypedCoreError("action payload is invalid") from exc


@dataclass(frozen=True)
class ActionFeatures:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(FEATURE_NAMES) or any(
            type(value) is not int or not -SCALE <= value <= SCALE
            for value in self.values
        ):
            raise AveritecP1TypedCoreError("action features drifted")


def compute_action_features(
    *, action: RecipeAction, document_texts: Sequence[str]
) -> ActionFeatures:
    direct_mean = _mean_integer(action.selected_direct_scores)
    direct_min = min(action.selected_direct_scores)
    slot_mean = _mean_integer(action.assigned_slot_scores)
    slot_min = min(action.assigned_slot_scores)
    slot_minus_direct = max(-SCALE, min(SCALE, slot_mean - direct_mean))
    raw_overlap = (
        len(
            set(action.top5_document_ordinals)
            & set(action.raw_top5_document_ordinals)
        )
        * SCALE
        // TOP_K
    )
    diversity = _lexical_diversity(
        document_texts, action.top5_document_ordinals
    )
    typed_fraction = (
        sum(slot != DIRECT for slot in action.assigned_slots) * SCALE // TOP_K
    )
    return ActionFeatures(
        (
            direct_mean,
            direct_min,
            slot_mean,
            slot_min,
            slot_minus_direct,
            raw_overlap,
            diversity,
            typed_fraction,
        )
    )


def utility(
    *, top5_document_ordinals: Sequence[int], qrel_document_ordinals: Sequence[int]
) -> Fraction:
    if (
        len(top5_document_ordinals) != TOP_K
        or len(set(top5_document_ordinals)) != TOP_K
        or not qrel_document_ordinals
        or len(set(qrel_document_ordinals)) != len(qrel_document_ordinals)
    ):
        raise AveritecP1TypedCoreError("utility inputs are malformed")
    qrels = set(qrel_document_ordinals)
    hits = len(qrels & set(top5_document_ordinals))
    return Fraction(hits, len(qrels))


@dataclass(frozen=True)
class AFormAction:
    recipe_id: str
    features: ActionFeatures
    utility: Fraction


@dataclass(frozen=True)
class AFormSlate:
    actions: tuple[AFormAction, ...]

    def __post_init__(self) -> None:
        if tuple(action.recipe_id for action in self.actions) != RECIPE_IDS:
            raise AveritecP1TypedCoreError("A_form slate recipe order drifted")


@dataclass(frozen=True)
class E1Model:
    weights: tuple[float, ...]
    training_item_count: int
    training_row_count: int
    target: str = "recipe_utility_minus_R0_utility"

    def __post_init__(self) -> None:
        if (
            len(self.weights) != MODEL_FEATURE_COUNT
            or any(not math.isfinite(value) for value in self.weights)
            or self.training_item_count <= 0
            or self.training_row_count != self.training_item_count * len(RECIPE_IDS)
            or self.target != "recipe_utility_minus_R0_utility"
        ):
            raise AveritecP1TypedCoreError("E1 model drifted")


def _model_row(recipe_id: str, features: ActionFeatures) -> np.ndarray:
    if recipe_id not in RECIPE_IDS:
        raise AveritecP1TypedCoreError("model recipe is not frozen")
    row = np.zeros(MODEL_FEATURE_COUNT, dtype=np.float64)
    row[0] = 1.0
    row[1 + RECIPE_IDS.index(recipe_id)] = 1.0
    offset = 1 + len(RECIPE_IDS)
    row[offset:] = np.asarray(features.values, dtype=np.float64) / SCALE
    return row


def fit_e1(slates: Sequence[AFormSlate]) -> E1Model:
    if not slates:
        raise AveritecP1TypedCoreError("A_form slates are empty")
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for slate in slates:
        baseline = slate.actions[0].utility
        for action in slate.actions:
            rows.append(_model_row(action.recipe_id, action.features))
            targets.append(float(action.utility - baseline))
    matrix = np.stack(rows).astype(np.float64, copy=False)
    target = np.asarray(targets, dtype=np.float64)
    penalty = np.eye(MODEL_FEATURE_COUNT, dtype=np.float64) * RIDGE_L2
    penalty[0, 0] = 0.0
    gram = matrix.T @ matrix + penalty
    rhs = matrix.T @ target
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise AveritecP1TypedCoreError("E1 ridge solve failed") from exc
    rounded = tuple(float(f"{value:.15g}") for value in weights)
    return E1Model(
        weights=rounded,
        training_item_count=len(slates),
        training_row_count=len(rows),
    )


def model_payload(model: E1Model) -> dict[str, object]:
    body = {
        "feature_names": [
            "intercept",
            *[f"recipe_one_hot:{recipe_id}" for recipe_id in RECIPE_IDS],
            *FEATURE_NAMES,
        ],
        "ridge_l2": RIDGE_L2,
        "schema": f"{VERSION}_E1_model_v1",
        "study_id": STUDY_ID,
        "target": model.target,
        "training_item_count": model.training_item_count,
        "training_row_count": model.training_row_count,
        "weights": list(model.weights),
    }
    body["self_sha256"] = stable_hash(body)
    return body


def model_from_payload(value: object) -> E1Model:
    if not isinstance(value, Mapping):
        raise AveritecP1TypedCoreError("E1 model payload is not an object")
    body = dict(value)
    self_sha256 = body.pop("self_sha256", None)
    if self_sha256 != stable_hash(body):
        raise AveritecP1TypedCoreError("E1 model self hash drifted")
    expected_features = [
        "intercept",
        *[f"recipe_one_hot:{recipe_id}" for recipe_id in RECIPE_IDS],
        *FEATURE_NAMES,
    ]
    if (
        body.get("schema") != f"{VERSION}_E1_model_v1"
        or body.get("study_id") != STUDY_ID
        or body.get("ridge_l2") != RIDGE_L2
        or body.get("feature_names") != expected_features
        or body.get("target") != "recipe_utility_minus_R0_utility"
    ):
        raise AveritecP1TypedCoreError("E1 model contract drifted")
    weights = body.get("weights")
    if not isinstance(weights, list):
        raise AveritecP1TypedCoreError("E1 model weights drifted")
    return E1Model(
        weights=tuple(float(value) for value in weights),
        training_item_count=int(body["training_item_count"]),
        training_row_count=int(body["training_row_count"]),
    )


def select_e0(actions: Mapping[str, RecipeAction]) -> str:
    if set(actions) != set(RECIPE_IDS):
        raise AveritecP1TypedCoreError("E0 action slate drifted")
    return R0_DIRECT_DENSE


def select_e1(
    *,
    model: E1Model,
    actions: Mapping[str, RecipeAction],
    document_texts: Sequence[str],
) -> str:
    if set(actions) != set(RECIPE_IDS):
        raise AveritecP1TypedCoreError("E1 action slate drifted")
    weights = np.asarray(model.weights, dtype=np.float64)
    scored: list[tuple[float, str]] = []
    for recipe_id in RECIPE_IDS:
        features = compute_action_features(
            action=actions[recipe_id], document_texts=document_texts
        )
        score = float(_model_row(recipe_id, features) @ weights)
        if not math.isfinite(score):
            raise AveritecP1TypedCoreError("E1 prediction is not finite")
        scored.append((score, recipe_id))
    return min(scored, key=lambda row: (-row[0], row[1]))[1]


@dataclass(frozen=True)
class ExactComparison:
    net_utility: Fraction
    positive_count: int
    negative_count: int
    tie_count: int
    reference_tail: Fraction


def exact_sign_flip(deltas: Sequence[Fraction]) -> Fraction:
    """Return the exact one-sided random-sign reference tail.

    Formal A_hold and M_search contain 36 rows.  A Counter-based convolution
    can have 2**36 distinct rational states when qrel denominators differ.
    This frozen meet-in-the-middle implementation instead materializes at most
    two 2**18 integer-sum tables, after an exact common-denominator lift.
    """

    nonzero = [delta for delta in deltas if delta]
    if not nonzero:
        return Fraction(1, 1)
    if len(nonzero) > 36:
        raise AveritecP1TypedCoreError(
            "exact sign-flip contract exceeds the frozen 36-pair bound"
        )
    denominator = 1
    for delta in nonzero:
        denominator = math.lcm(denominator, delta.denominator)
    magnitudes = [
        abs(delta.numerator) * (denominator // delta.denominator)
        for delta in nonzero
    ]
    observed = sum(
        delta.numerator * (denominator // delta.denominator)
        for delta in nonzero
    )

    def signed_sums(values: Sequence[int]) -> list[int]:
        sums = [0]
        for magnitude in values:
            sums = [total + magnitude for total in sums] + [
                total - magnitude for total in sums
            ]
        return sums

    midpoint = len(magnitudes) // 2
    left = signed_sums(magnitudes[:midpoint])
    right = signed_sums(magnitudes[midpoint:])
    right.sort()
    favorable = sum(
        len(right) - bisect_left(right, observed - left_total)
        for left_total in left
    )
    return Fraction(favorable, 2 ** len(nonzero))


def compare(
    candidate: Sequence[Fraction], baseline: Sequence[Fraction]
) -> ExactComparison:
    if len(candidate) != len(baseline) or not candidate:
        raise AveritecP1TypedCoreError("paired comparison shape drifted")
    deltas = [left - right for left, right in zip(candidate, baseline)]
    return ExactComparison(
        net_utility=sum(deltas, Fraction(0, 1)),
        positive_count=sum(delta > 0 for delta in deltas),
        negative_count=sum(delta < 0 for delta in deltas),
        tie_count=sum(delta == 0 for delta in deltas),
        reference_tail=exact_sign_flip(deltas),
    )


def comparison_payload(value: ExactComparison) -> dict[str, object]:
    def fraction_payload(number: Fraction) -> dict[str, int]:
        return {
            "denominator": number.denominator,
            "numerator": number.numerator,
        }

    return {
        "negative_count": value.negative_count,
        "net_utility": fraction_payload(value.net_utility),
        "positive_count": value.positive_count,
        "reference_tail": fraction_payload(value.reference_tail),
        "tie_count": value.tie_count,
    }

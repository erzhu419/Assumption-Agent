"""Pure frozen action/evaluator core for the EntailmentBank G1/E1 study."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence
import unicodedata

import numpy as np


VERSION = "entailmentbank_proof_retrieval_core_v1"
INTEGER_SCALE = 1_000_000
TOP_K = 5
NODE_FEATURE_COUNT = 8
EVALUATOR_FEATURE_COUNT = 16
ALPHA_REGISTRY = (0, 250_000, 500_000, 1_000_000)
SEED_REGISTRY = (
    "G_RIDGE",
    "NLI_HYPOTHESIS",
    "MINILM_HYPOTHESIS",
    "BORDA_G_RIDGE_NLI_HYPOTHESIS_MINILM_HYPOTHESIS_TOKEN_F1_HYPOTHESIS",
)
FAMILY_ORDER = ("TWO_LEAF", "THREE_LEAF", "FOUR_FIVE_LEAF")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EntailmentBankCoreError(RuntimeError):
    """A frozen feature, model, recipe, action, or score contract drifted."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EntailmentBankCoreError("value is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EntailmentBankCoreError(f"{field} is not an integer")
    if not -(2**63) <= value < 2**63:
        raise EntailmentBankCoreError(f"{field} is outside int64")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise EntailmentBankCoreError(f"{field} is not exact nonempty text")
    return value


def _commitment(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EntailmentBankCoreError(f"{field} is invalid")
    return value


def tokens(value: str) -> frozenset[str]:
    """Return the exact frozen ASCII alphanumeric token set."""

    text = unicodedata.normalize("NFKC", _text(value, "token input")).casefold()
    return frozenset(_TOKEN_RE.findall(text))


def token_f1(left: str, right: str) -> int:
    """Return set-token F1 quantized at one million."""

    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0
    return int(
        round(
            2
            * len(left_tokens & right_tokens)
            * INTEGER_SCALE
            / (len(left_tokens) + len(right_tokens))
        )
    )


@dataclass(frozen=True)
class LabelFreeItem:
    item_commitment_sha256: str
    question: str
    answer: str
    hypothesis: str
    node_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        _commitment(self.item_commitment_sha256, "item commitment")
        _text(self.question, "question")
        _text(self.answer, "answer")
        _text(self.hypothesis, "hypothesis")
        if not isinstance(self.node_texts, tuple) or len(self.node_texts) != 25:
            raise EntailmentBankCoreError("item must contain exactly 25 nodes")
        for index, value in enumerate(self.node_texts):
            _text(value, f"node_texts[{index}]")

    @property
    def answer_query(self) -> str:
        return f"{self.question}\nAnswer: {self.answer}"


@dataclass(frozen=True)
class ItemLabel:
    item_commitment_sha256: str
    family: str
    gold_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        _commitment(self.item_commitment_sha256, "label commitment")
        if self.family not in FAMILY_ORDER:
            raise EntailmentBankCoreError("label family is invalid")
        if (
            not isinstance(self.gold_ordinals, tuple)
            or not 2 <= len(self.gold_ordinals) <= 5
            or tuple(sorted(set(self.gold_ordinals))) != self.gold_ordinals
            or any(type(value) is not int or not 0 <= value < 25 for value in self.gold_ordinals)
        ):
            raise EntailmentBankCoreError("gold ordinals are invalid")
        expected = (
            "TWO_LEAF"
            if len(self.gold_ordinals) == 2
            else "THREE_LEAF"
            if len(self.gold_ordinals) == 3
            else "FOUR_FIVE_LEAF"
        )
        if self.family != expected:
            raise EntailmentBankCoreError("gold family and cardinality disagree")


@dataclass(frozen=True)
class ItemTensor:
    item_commitment_sha256: str
    node_features: tuple[tuple[int, ...], ...]
    pair_token_f1: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _commitment(self.item_commitment_sha256, "tensor commitment")
        if not isinstance(self.node_features, tuple) or len(self.node_features) != 25:
            raise EntailmentBankCoreError("node feature row count is invalid")
        for row in self.node_features:
            if not isinstance(row, tuple) or len(row) != NODE_FEATURE_COUNT:
                raise EntailmentBankCoreError("node feature width is invalid")
            for value in row:
                _integer(value, "node feature")
        if not isinstance(self.pair_token_f1, tuple) or len(self.pair_token_f1) != 25:
            raise EntailmentBankCoreError("pair matrix row count is invalid")
        for left, row in enumerate(self.pair_token_f1):
            if not isinstance(row, tuple) or len(row) != 25:
                raise EntailmentBankCoreError("pair matrix width is invalid")
            for right, value in enumerate(row):
                if (
                    type(value) is not int
                    or not 0 <= value <= INTEGER_SCALE
                    or (left == right and value != INTEGER_SCALE)
                    or value != self.pair_token_f1[right][left]
                ):
                    raise EntailmentBankCoreError("pair matrix is invalid")


def build_pair_token_f1(node_texts: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    if isinstance(node_texts, (str, bytes)) or len(node_texts) != 25:
        raise EntailmentBankCoreError("node text registry is invalid")
    matrix = [[0] * 25 for _ in range(25)]
    for left in range(25):
        matrix[left][left] = INTEGER_SCALE
        for right in range(left + 1, 25):
            value = token_f1(node_texts[left], node_texts[right])
            matrix[left][right] = value
            matrix[right][left] = value
    return tuple(tuple(row) for row in matrix)


@dataclass(frozen=True)
class QuantizedRidgeModel:
    feature_count: int
    means: tuple[int, ...]
    scales: tuple[int, ...]
    intercept: int
    coefficients: tuple[int, ...]
    training_row_count: int
    target_kind: str
    model_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.feature_count) is not int
            or self.feature_count <= 0
            or len(self.means) != self.feature_count
            or len(self.scales) != self.feature_count
            or len(self.coefficients) != self.feature_count
            or any(type(value) is not int for value in (*self.means, *self.scales, *self.coefficients))
            or any(value <= 0 for value in self.scales)
            or type(self.intercept) is not int
            or type(self.training_row_count) is not int
            or self.training_row_count <= 0
            or self.target_kind not in {"gold_leaf_binary", "direct_item_utility"}
        ):
            raise EntailmentBankCoreError("ridge model fields are invalid")
        if self.model_sha256 != _sha256(self.payload(include_hash=False)):
            raise EntailmentBankCoreError("ridge model self hash drifted")

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": f"{VERSION}_quantized_ridge_model",
            "feature_count": self.feature_count,
            "means": list(self.means),
            "scales": list(self.scales),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "training_row_count": self.training_row_count,
            "target_kind": self.target_kind,
        }
        if include_hash:
            value["model_sha256"] = self.model_sha256
        return value

    def predict(self, features: Sequence[int]) -> int:
        if len(features) != self.feature_count:
            raise EntailmentBankCoreError("ridge prediction feature width drifted")
        score = self.intercept
        for value, mean, scale, coefficient in zip(
            features,
            self.means,
            self.scales,
            self.coefficients,
            strict=True,
        ):
            _integer(value, "ridge feature")
            score += int(round(coefficient * (value - mean) / scale))
        return _integer(score, "ridge prediction")

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "QuantizedRidgeModel":
        if value.get("schema") != f"{VERSION}_quantized_ridge_model":
            raise EntailmentBankCoreError("ridge model schema drifted")
        try:
            return cls(
                feature_count=_integer(value["feature_count"], "feature count"),
                means=tuple(value["means"]),
                scales=tuple(value["scales"]),
                intercept=_integer(value["intercept"], "intercept"),
                coefficients=tuple(value["coefficients"]),
                training_row_count=_integer(
                    value["training_row_count"], "training row count"
                ),
                target_kind=_text(value["target_kind"], "target kind"),
                model_sha256=_commitment(value["model_sha256"], "model hash"),
            )
        except (KeyError, TypeError, ValueError, EntailmentBankCoreError) as exc:
            raise EntailmentBankCoreError("ridge model payload is invalid") from exc


def fit_quantized_ridge(
    rows: Sequence[Sequence[int]],
    targets: Sequence[int],
    *,
    target_kind: str,
    sample_weights: Sequence[float] | None = None,
) -> QuantizedRidgeModel:
    if not rows or len(rows) != len(targets):
        raise EntailmentBankCoreError("ridge training rows and targets drifted")
    width = len(rows[0])
    if width <= 0 or any(len(row) != width for row in rows):
        raise EntailmentBankCoreError("ridge training feature width drifted")
    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if matrix.shape != (len(rows), width) or target.shape != (len(rows),):
        raise EntailmentBankCoreError("ridge training matrix shape drifted")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise EntailmentBankCoreError("ridge training values are nonfinite")
    means = tuple(int(round(value)) for value in matrix.mean(axis=0))
    centered = matrix - np.asarray(means, dtype=np.float64)
    scales = tuple(
        max(1, int(round(value)))
        for value in np.sqrt(np.mean(centered * centered, axis=0))
    )
    standardized = centered / np.asarray(scales, dtype=np.float64)
    design = np.column_stack((np.ones(len(rows), dtype=np.float64), standardized))
    if sample_weights is None:
        weights = np.ones(len(rows), dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (len(rows),) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise EntailmentBankCoreError("ridge sample weights are invalid")
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    penalty = np.eye(width + 1, dtype=np.float64)
    penalty[0, 0] = 0.0
    gram = weighted_design.T @ weighted_design + penalty
    solution = np.linalg.pinv(gram, rcond=1e-12) @ weighted_design.T @ weighted_target
    if not np.isfinite(solution).all():
        raise EntailmentBankCoreError("ridge solution is nonfinite")
    intercept = int(round(float(solution[0]) * INTEGER_SCALE))
    coefficients = tuple(
        int(round(float(value) * INTEGER_SCALE)) for value in solution[1:]
    )
    body = {
        "schema": f"{VERSION}_quantized_ridge_model",
        "feature_count": width,
        "means": list(means),
        "scales": list(scales),
        "intercept": intercept,
        "coefficients": list(coefficients),
        "training_row_count": len(rows),
        "target_kind": target_kind,
    }
    return QuantizedRidgeModel(
        width,
        means,
        scales,
        intercept,
        coefficients,
        len(rows),
        target_kind,
        _sha256(body),
    )


def fit_g_model(
    tensors: Sequence[ItemTensor], labels: Sequence[ItemLabel]
) -> QuantizedRidgeModel:
    label_by_commitment = {label.item_commitment_sha256: label for label in labels}
    if len(label_by_commitment) != len(labels) or len(tensors) != len(labels):
        raise EntailmentBankCoreError("G tensor/label registry drifted")
    rows: list[tuple[int, ...]] = []
    targets: list[int] = []
    for tensor in tensors:
        label = label_by_commitment.get(tensor.item_commitment_sha256)
        if label is None:
            raise EntailmentBankCoreError("G tensor has no matching label")
        gold = set(label.gold_ordinals)
        for ordinal, features in enumerate(tensor.node_features):
            rows.append(features)
            targets.append(int(ordinal in gold))
    positives = sum(targets)
    negatives = len(targets) - positives
    if positives <= 0 or negatives <= 0:
        raise EntailmentBankCoreError("G class registry is degenerate")
    positive_weight = negatives / positives
    weights = [positive_weight if target else 1.0 for target in targets]
    return fit_quantized_ridge(
        rows,
        targets,
        target_kind="gold_leaf_binary",
        sample_weights=weights,
    )


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    seed: str
    alpha: int


def recipe_registry() -> tuple[Recipe, ...]:
    recipes: list[Recipe] = []
    index = 0
    for seed in SEED_REGISTRY:
        for alpha in ALPHA_REGISTRY:
            recipes.append(Recipe(f"R{index:02d}_{seed}_A{alpha:07d}", seed, alpha))
            index += 1
    if len(recipes) != 16 or len({recipe.recipe_id for recipe in recipes}) != 16:
        raise EntailmentBankCoreError("recipe registry drifted")
    return tuple(recipes)


RECIPE_REGISTRY = recipe_registry()
RECIPE_BY_ID = {recipe.recipe_id: recipe for recipe in RECIPE_REGISTRY}


def _rank_scores(values: Sequence[int]) -> tuple[int, ...]:
    if len(values) != 25:
        raise EntailmentBankCoreError("rank score vector length drifted")
    order = sorted(range(25), key=lambda ordinal: (-values[ordinal], ordinal))
    scores = [0] * 25
    for rank, ordinal in enumerate(order):
        scores[ordinal] = int(round((24 - rank) * INTEGER_SCALE / 24))
    return tuple(scores)


def base_rank_scores(
    tensor: ItemTensor, g_model: QuantizedRidgeModel
) -> Mapping[str, tuple[int, ...]]:
    ridge_raw = tuple(g_model.predict(row) for row in tensor.node_features)
    nli_raw = tuple(row[0] for row in tensor.node_features)
    minilm_raw = tuple(row[2] for row in tensor.node_features)
    lexical_raw = tuple(row[4] for row in tensor.node_features)
    ridge = _rank_scores(ridge_raw)
    nli = _rank_scores(nli_raw)
    minilm = _rank_scores(minilm_raw)
    lexical = _rank_scores(lexical_raw)
    borda = tuple(
        ridge[index] + nli[index] + minilm[index] + lexical[index]
        for index in range(25)
    )
    return {
        "G_RIDGE": ridge,
        "NLI_HYPOTHESIS": nli,
        "MINILM_HYPOTHESIS": minilm,
        "TOKEN_F1_HYPOTHESIS": lexical,
        "BORDA_G_RIDGE_NLI_HYPOTHESIS_MINILM_HYPOTHESIS_TOKEN_F1_HYPOTHESIS": _rank_scores(
            borda
        ),
    }


@dataclass(frozen=True)
class Action:
    recipe_id: str
    item_commitment_sha256: str
    selected_ordinals: tuple[int, ...]
    action_sha256: str

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_BY_ID:
            raise EntailmentBankCoreError("action recipe is invalid")
        _commitment(self.item_commitment_sha256, "action item commitment")
        _commitment(self.action_sha256, "action hash")
        if (
            not isinstance(self.selected_ordinals, tuple)
            or len(self.selected_ordinals) != TOP_K
            or len(set(self.selected_ordinals)) != TOP_K
            or any(type(value) is not int or not 0 <= value < 25 for value in self.selected_ordinals)
        ):
            raise EntailmentBankCoreError("action ordinals are invalid")
        if self.action_sha256 != _sha256(self.payload(include_hash=False)):
            raise EntailmentBankCoreError("action self hash drifted")

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema": f"{VERSION}_action",
            "recipe_id": self.recipe_id,
            "item_commitment_sha256": self.item_commitment_sha256,
            "selected_ordinals": list(self.selected_ordinals),
        }
        if include_hash:
            value["action_sha256"] = self.action_sha256
        return value


def execute_recipe(
    tensor: ItemTensor,
    g_model: QuantizedRidgeModel,
    recipe_id: str,
) -> Action:
    recipe = RECIPE_BY_ID.get(recipe_id)
    if recipe is None:
        raise EntailmentBankCoreError("unknown recipe")
    scores = base_rank_scores(tensor, g_model)[recipe.seed]
    selected: list[int] = []
    remaining = set(range(25))
    while len(selected) < TOP_K:
        best: tuple[int, int] | None = None
        best_ordinal: int | None = None
        for ordinal in sorted(remaining):
            bridge = (
                max(tensor.pair_token_f1[ordinal][prior] for prior in selected)
                if selected
                else 0
            )
            value = scores[ordinal] + recipe.alpha * bridge // INTEGER_SCALE
            key = (value, -ordinal)
            if best is None or key > best:
                best = key
                best_ordinal = ordinal
        assert best_ordinal is not None
        selected.append(best_ordinal)
        remaining.remove(best_ordinal)
    body = {
        "schema": f"{VERSION}_action",
        "recipe_id": recipe_id,
        "item_commitment_sha256": tensor.item_commitment_sha256,
        "selected_ordinals": selected,
    }
    return Action(recipe_id, tensor.item_commitment_sha256, tuple(selected), _sha256(body))


def direct_utility(action_ordinals: Sequence[int], label: ItemLabel) -> int:
    if len(action_ordinals) != TOP_K or len(set(action_ordinals)) != TOP_K:
        raise EntailmentBankCoreError("utility action ordinals are invalid")
    selected = set(action_ordinals)
    hits = len(selected & set(label.gold_ordinals))
    return hits + int(hits == len(label.gold_ordinals))


def evaluator_features(
    item: LabelFreeItem,
    tensor: ItemTensor,
    g_model: QuantizedRidgeModel,
    action: Action,
) -> tuple[int, ...]:
    if (
        item.item_commitment_sha256 != tensor.item_commitment_sha256
        or action.item_commitment_sha256 != item.item_commitment_sha256
    ):
        raise EntailmentBankCoreError("evaluator item binding drifted")
    ranks = base_rank_scores(tensor, g_model)
    selected = action.selected_ordinals
    pair_values = [
        tensor.pair_token_f1[left][right]
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    ]
    union_text = " ".join(item.node_texts[ordinal] for ordinal in selected)
    recipe = RECIPE_BY_ID[action.recipe_id]
    seed_flags = tuple(
        INTEGER_SCALE if recipe.seed == seed else 0 for seed in SEED_REGISTRY
    )
    values = (
        sum(ranks["G_RIDGE"][ordinal] for ordinal in selected),
        min(ranks["G_RIDGE"][ordinal] for ordinal in selected),
        sum(ranks["NLI_HYPOTHESIS"][ordinal] for ordinal in selected),
        min(ranks["NLI_HYPOTHESIS"][ordinal] for ordinal in selected),
        sum(ranks["MINILM_HYPOTHESIS"][ordinal] for ordinal in selected),
        min(ranks["MINILM_HYPOTHESIS"][ordinal] for ordinal in selected),
        sum(ranks["TOKEN_F1_HYPOTHESIS"][ordinal] for ordinal in selected),
        min(ranks["TOKEN_F1_HYPOTHESIS"][ordinal] for ordinal in selected),
        int(round(sum(pair_values) / len(pair_values))),
        min(pair_values),
        token_f1(union_text, item.hypothesis),
        recipe.alpha,
        *seed_flags,
    )
    if len(values) != EVALUATOR_FEATURE_COUNT:
        raise EntailmentBankCoreError("evaluator feature width drifted")
    return tuple(values)


def e0_score(features: Sequence[int]) -> int:
    if len(features) != EVALUATOR_FEATURE_COUNT:
        raise EntailmentBankCoreError("E0 feature width drifted")
    return features[0] + features[1] + features[8]


def fit_e1_model(
    feature_rows: Sequence[Sequence[int]], targets: Sequence[int]
) -> QuantizedRidgeModel:
    return fit_quantized_ridge(
        feature_rows,
        targets,
        target_kind="direct_item_utility",
    )


def select_global_recipe(
    feature_by_item_recipe: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    evaluator: str,
    e1_model: QuantizedRidgeModel | None = None,
) -> tuple[str, Mapping[str, int]]:
    if evaluator not in {"E0", "E1"}:
        raise EntailmentBankCoreError("unknown evaluator")
    if evaluator == "E1" and e1_model is None:
        raise EntailmentBankCoreError("E1 model is absent")
    totals = {recipe.recipe_id: 0 for recipe in RECIPE_REGISTRY}
    if not feature_by_item_recipe:
        raise EntailmentBankCoreError("search feature registry is empty")
    for item_key in sorted(feature_by_item_recipe):
        registry = feature_by_item_recipe[item_key]
        if set(registry) != set(totals):
            raise EntailmentBankCoreError("search recipe registry drifted")
        for recipe_id in totals:
            features = registry[recipe_id]
            totals[recipe_id] += (
                e0_score(features)
                if evaluator == "E0"
                else e1_model.predict(features)  # type: ignore[union-attr]
            )
    selected = min(totals, key=lambda recipe_id: (-totals[recipe_id], recipe_id))
    return selected, dict(sorted(totals.items()))


def exact_one_sided_signflip(differences: Sequence[int]) -> dict[str, int]:
    values = tuple(_integer(value, "paired difference") for value in differences)
    nonzero = tuple(abs(value) for value in values if value != 0)
    observed = sum(values)
    if not nonzero:
        return {
            "observed_sum": observed,
            "nonzero_pair_count": 0,
            "tail_numerator": 1,
            "tail_denominator": 1,
        }
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in nonzero:
        updated: Counter[int] = Counter()
        for total, count in distribution.items():
            updated[total + magnitude] += count
            updated[total - magnitude] += count
        distribution = updated
    numerator = sum(count for total, count in distribution.items() if total >= observed)
    denominator = 2 ** len(nonzero)
    divisor = math.gcd(numerator, denominator)
    return {
        "observed_sum": observed,
        "nonzero_pair_count": len(nonzero),
        "tail_numerator": numerator // divisor,
        "tail_denominator": denominator // divisor,
    }


__all__ = [
    "ALPHA_REGISTRY",
    "Action",
    "EVALUATOR_FEATURE_COUNT",
    "EntailmentBankCoreError",
    "FAMILY_ORDER",
    "INTEGER_SCALE",
    "ItemLabel",
    "ItemTensor",
    "LabelFreeItem",
    "NODE_FEATURE_COUNT",
    "QuantizedRidgeModel",
    "RECIPE_BY_ID",
    "RECIPE_REGISTRY",
    "SEED_REGISTRY",
    "TOP_K",
    "base_rank_scores",
    "build_pair_token_f1",
    "direct_utility",
    "e0_score",
    "evaluator_features",
    "exact_one_sided_signflip",
    "execute_recipe",
    "fit_e1_model",
    "fit_g_model",
    "fit_quantized_ridge",
    "recipe_registry",
    "select_global_recipe",
    "token_f1",
    "tokens",
]

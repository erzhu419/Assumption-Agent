"""Frozen, source-independent core for the EBM-NLP P1 PICO study.

This module contains only deterministic transformations over caller-supplied
tokens, embeddings, probe labels, and gold token positions.  It performs no
filesystem, archive, network, subprocess, or environment access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import hmac
import json
import math
from typing import Any, Callable, Mapping, Sequence


STUDY_ID = "EBMNLP_P1_TYPED_PICO_SET_EVALUATOR_V1"
INTEGER_SCALE = 1_000_000
WINDOW_WIDTH = 48
WINDOW_STRIDE = 24
WINDOW_ID_DIGITS = 8
TOP_K = 5
HARMONIC_5 = sum((Fraction(1, rank) for rank in range(1, TOP_K + 1)), Fraction())

PARTICIPANT = "PARTICIPANT"
INTERVENTION = "INTERVENTION"
OUTCOME = "OUTCOME"
ROLE_ORDER = (PARTICIPANT, INTERVENTION, OUTCOME)
ROLE_QUERIES = {
    PARTICIPANT: (
        "Which text describes the participants or patient population in this "
        "clinical trial?"
    ),
    INTERVENTION: (
        "Which text describes the intervention or treatment in this clinical "
        "trial?"
    ),
    OUTCOME: (
        "Which text describes the outcomes or endpoints measured in this "
        "clinical trial?"
    ),
}

R0_TARGET_POSTERIOR = "R0_TARGET_POSTERIOR"
R1_ROLE_CONTRAST = "R1_ROLE_CONTRAST"
R2_CONTIGUOUS_MAP = "R2_CONTIGUOUS_MAP"
R3_DISTINCT_POSTERIOR_COVERAGE = "R3_DISTINCT_POSTERIOR_COVERAGE"
R4_SEMANTIC_DIVERSE_SET = "R4_SEMANTIC_DIVERSE_SET"
R5_OVERLAP_PRESERVING_JOINT = "R5_OVERLAP_PRESERVING_JOINT"
RECIPE_IDS = (
    R0_TARGET_POSTERIOR,
    R1_ROLE_CONTRAST,
    R2_CONTIGUOUS_MAP,
    R3_DISTINCT_POSTERIOR_COVERAGE,
    R4_SEMANTIC_DIVERSE_SET,
    R5_OVERLAP_PRESERVING_JOINT,
)

BASE_FEATURE_ORDER = (
    "mean_target_probe_probability",
    "minimum_target_probe_probability",
    "maximum_target_probe_probability",
    "mean_target_minus_max_other_role_probability",
    "selected_union_target_posterior_mass_fraction",
    "mean_role_query_MiniLM_cosine_unit_interval",
    "mean_pairwise_MiniLM_diversity",
    "selected_token_position_range_fraction",
    "selected_window_overlap_fraction",
    "mean_target_probe_binary_entropy",
)
FEATURE_ORDER = BASE_FEATURE_ORDER + tuple(
    f"recipe_ID_is_{recipe_id}" for recipe_id in RECIPE_IDS
)
E1_FEATURE_COUNT = len(FEATURE_ORDER)


class EbmNlpP1CoreError(ValueError):
    """The frozen local core contract was violated."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Fraction):
        return {
            "denominator": value.denominator,
            "numerator": value.numerator,
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EbmNlpP1CoreError("non-finite float is not canonical JSON")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise EbmNlpP1CoreError("canonical JSON object keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise EbmNlpP1CoreError(
        f"value of type {type(value).__name__} is not canonical JSON"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return sorted, compact, ASCII JSON terminated by exactly one newline."""

    try:
        return (
            json.dumps(
                _jsonable(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except EbmNlpP1CoreError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EbmNlpP1CoreError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EbmNlpP1CoreError(f"{field} must be an exact integer")
    return value


def _quantized_unit(value: object, *, field: str) -> int:
    integer = _exact_int(value, field=field)
    if not 0 <= integer <= INTEGER_SCALE:
        raise EbmNlpP1CoreError(
            f"{field} must be in [0, {INTEGER_SCALE}]"
        )
    return integer


def _round_fraction_half_even(value: Fraction) -> int:
    return int(round(value))


def quantize_half_even(value: object, *, unit_interval: bool = False) -> int:
    """Quantize one finite real to signed integer scale 1,000,000."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, Fraction)):
        raise EbmNlpP1CoreError("value must be a finite real")
    if isinstance(value, float) and not math.isfinite(value):
        raise EbmNlpP1CoreError("value must be a finite real")
    if isinstance(value, Decimal) and not value.is_finite():
        raise EbmNlpP1CoreError("value must be a finite real")
    decimal_value = (
        value
        if isinstance(value, Decimal)
        else Decimal(value.numerator) / Decimal(value.denominator)
        if isinstance(value, Fraction)
        else Decimal(str(value))
    )
    if unit_interval and not Decimal(0) <= decimal_value <= Decimal(1):
        raise EbmNlpP1CoreError("unit-interval value is outside [0, 1]")
    return int(
        (decimal_value * INTEGER_SCALE).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
    )


@dataclass(frozen=True)
class EvidenceWindow:
    ordinal: int
    start: int
    end: int
    window_id: str
    text: str

    def __post_init__(self) -> None:
        ordinal = _exact_int(self.ordinal, field="window ordinal")
        start = _exact_int(self.start, field="window start")
        end = _exact_int(self.end, field="window end")
        if ordinal < 0 or start < 0 or end <= start:
            raise EbmNlpP1CoreError("window bounds and ordinal must be valid")
        expected = f"W:{start:0{WINDOW_ID_DIGITS}d}:{end:0{WINDOW_ID_DIGITS}d}"
        if self.window_id != expected:
            raise EbmNlpP1CoreError("window identity is not canonical")
        if not isinstance(self.text, str) or not self.text:
            raise EbmNlpP1CoreError("window text must be nonempty")

    @property
    def token_count(self) -> int:
        return self.end - self.start


def build_evidence_windows(tokens: Sequence[str]) -> tuple[EvidenceWindow, ...]:
    """Build the frozen 48-token/24-stride registry with one exact tail."""

    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise EbmNlpP1CoreError("tokens must be a sequence of token strings")
    canonical_tokens = tuple(tokens)
    if not canonical_tokens:
        raise EbmNlpP1CoreError("token sequence must be nonempty")
    for token in canonical_tokens:
        if (
            not isinstance(token, str)
            or not token
            or any(character.isspace() for character in token)
            or "\x00" in token
        ):
            raise EbmNlpP1CoreError(
                "each token must be one nonempty whitespace-free string"
            )

    token_count = len(canonical_tokens)
    if token_count <= WINDOW_WIDTH:
        intervals = [(0, token_count)]
    else:
        intervals = [
            (start, start + WINDOW_WIDTH)
            for start in range(
                0, token_count - WINDOW_WIDTH + 1, WINDOW_STRIDE
            )
        ]
        if intervals[-1][1] != token_count:
            tail_start = max(0, token_count - WINDOW_WIDTH)
            tail = (tail_start, token_count)
            if tail != intervals[-1]:
                intervals.append(tail)

    return tuple(
        EvidenceWindow(
            ordinal=ordinal,
            start=start,
            end=end,
            window_id=(
                f"W:{start:0{WINDOW_ID_DIGITS}d}:"
                f"{end:0{WINDOW_ID_DIGITS}d}"
            ),
            text=" ".join(canonical_tokens[start:end]),
        )
        for ordinal, (start, end) in enumerate(intervals)
    )


def _validate_role(role: object) -> str:
    if role not in ROLE_ORDER:
        raise EbmNlpP1CoreError("role is outside the frozen role registry")
    return str(role)


class ConstantProbabilityProbe:
    """One-class totalizer exposing the shared positive-probability interface."""

    def __init__(self, probability: Fraction) -> None:
        if probability < 0 or probability > 1:
            raise EbmNlpP1CoreError("constant probability is outside [0, 1]")
        self.probability = probability

    def positive_probabilities(
        self, rows: Sequence[Sequence[object]]
    ) -> tuple[Fraction, ...]:
        return tuple(self.probability for _ in rows)


class SklearnProbabilityProbe:
    """Thin adapter around a fitted binary sklearn-compatible estimator."""

    def __init__(self, estimator: object, positive_column: int) -> None:
        self.estimator = estimator
        self.positive_column = positive_column

    def positive_probabilities(
        self, rows: Sequence[Sequence[object]]
    ) -> tuple[float, ...]:
        prediction = getattr(self.estimator, "predict_proba")(rows)
        probabilities: list[float] = []
        for row in prediction:
            try:
                value = float(row[self.positive_column])
            except (IndexError, TypeError, ValueError) as exc:
                raise EbmNlpP1CoreError(
                    "probe predict_proba result has invalid shape"
                ) from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise EbmNlpP1CoreError(
                    "probe returned a non-probability"
                )
            probabilities.append(value)
        if len(probabilities) != len(rows):
            raise EbmNlpP1CoreError("probe prediction row count drifted")
        return tuple(probabilities)


@dataclass(frozen=True)
class FrozenRoleProbes:
    """Exactly three independently fitted probes with one shared interface."""

    models: tuple[object, object, object]

    def __post_init__(self) -> None:
        if len(self.models) != len(ROLE_ORDER):
            raise EbmNlpP1CoreError("exactly three role probes are required")
        if any(
            not callable(getattr(model, "positive_probabilities", None))
            for model in self.models
        ):
            raise EbmNlpP1CoreError(
                "every role probe must expose positive_probabilities"
            )

    def score_quantized(
        self, embeddings: Sequence[Sequence[object]]
    ) -> dict[str, tuple[int, ...]]:
        rows = _validate_embedding_rows(embeddings)
        scored: dict[str, tuple[int, ...]] = {}
        for role, model in zip(ROLE_ORDER, self.models):
            values = model.positive_probabilities(rows)
            if len(values) != len(rows):
                raise EbmNlpP1CoreError("probe prediction row count drifted")
            scored[role] = tuple(
                quantize_half_even(value, unit_interval=True)
                for value in values
            )
        return scored


def _validate_embedding_rows(
    embeddings: Sequence[Sequence[object]],
) -> tuple[tuple[float, ...], ...]:
    if isinstance(embeddings, (str, bytes)) or not isinstance(
        embeddings, Sequence
    ):
        raise EbmNlpP1CoreError("embeddings must be a row sequence")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for raw_row in embeddings:
        if isinstance(raw_row, (str, bytes)) or not isinstance(
            raw_row, Sequence
        ):
            raise EbmNlpP1CoreError("embedding row must be a sequence")
        row: list[float] = []
        for raw_value in raw_row:
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, float)
            ):
                raise EbmNlpP1CoreError(
                    "embedding coordinates must be finite real numbers"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise EbmNlpP1CoreError(
                    "embedding coordinates must be finite real numbers"
                )
            row.append(value)
        if not row:
            raise EbmNlpP1CoreError("embedding rows must be nonempty")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise EbmNlpP1CoreError("embedding width drifted")
        rows.append(tuple(row))
    if not rows:
        raise EbmNlpP1CoreError("embedding population must be nonempty")
    return tuple(rows)


def fit_independent_role_probes(
    embeddings: Sequence[Sequence[object]],
    labels_by_role: Mapping[str, Sequence[object]],
    *,
    logistic_regression_cls: Callable[..., object] | None = None,
) -> FrozenRoleProbes:
    """Fit the exact three liblinear probes, totalizing one-class roles.

    ``logistic_regression_cls`` is injectable for source-free contract tests.
    When omitted, sklearn is imported lazily only when a two-class fit exists.
    """

    rows = _validate_embedding_rows(embeddings)
    if set(labels_by_role) != set(ROLE_ORDER):
        raise EbmNlpP1CoreError("probe label role registry drifted")
    normalized: dict[str, tuple[int, ...]] = {}
    for role in ROLE_ORDER:
        raw_labels = tuple(labels_by_role[role])
        if len(raw_labels) != len(rows):
            raise EbmNlpP1CoreError("probe label row count drifted")
        labels: list[int] = []
        for raw_label in raw_labels:
            label = _exact_int(raw_label, field=f"{role} probe label")
            if label not in (0, 1):
                raise EbmNlpP1CoreError("probe labels must be binary")
            labels.append(label)
        normalized[role] = tuple(labels)

    models: list[object] = []
    for role in ROLE_ORDER:
        labels = normalized[role]
        observed = set(labels)
        if len(observed) == 1:
            models.append(
                ConstantProbabilityProbe(
                    Fraction(sum(labels), len(labels))
                )
            )
            continue
        estimator_cls = logistic_regression_cls
        if estimator_cls is None:
            try:
                from sklearn.linear_model import LogisticRegression
            except ImportError as exc:  # pragma: no cover - environment specific
                raise EbmNlpP1CoreError(
                    "sklearn is required for a two-class probe fit"
                ) from exc
            estimator_cls = LogisticRegression
        estimator = estimator_cls(
            solver="liblinear",
            penalty="l2",
            C=1,
            class_weight="balanced",
            fit_intercept=True,
            max_iter=1000,
            tol=1e-6,
            random_state=0,
        )
        fitted = getattr(estimator, "fit")(rows, labels)
        if fitted is not None:
            estimator = fitted
        classes = tuple(int(value) for value in getattr(estimator, "classes_"))
        if classes != (0, 1):
            raise EbmNlpP1CoreError(
                "binary probe class registry is not exactly (0, 1)"
            )
        models.append(
            SklearnProbabilityProbe(estimator, classes.index(1))
        )
    return FrozenRoleProbes(tuple(models))  # type: ignore[arg-type]


@dataclass(frozen=True)
class RecipeAction:
    recipe_id: str
    registry_ordinal: int
    window_ordinals: tuple[int, ...]
    window_ids: tuple[str, ...]
    quantized_rule_scores: tuple[int, ...]
    behavior_sha256: str

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise EbmNlpP1CoreError("recipe ID is outside the frozen registry")
        if self.registry_ordinal != RECIPE_IDS.index(self.recipe_id):
            raise EbmNlpP1CoreError("recipe registry ordinal drifted")
        if (
            not self.window_ordinals
            or len(self.window_ordinals) > TOP_K
            or len(set(self.window_ordinals)) != len(self.window_ordinals)
            or len(self.window_ids) != len(self.window_ordinals)
            or len(self.quantized_rule_scores) != len(self.window_ordinals)
        ):
            raise EbmNlpP1CoreError(
                "recipe action is not one distinct totalized top-k ranking"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.quantized_rule_scores
        ):
            raise EbmNlpP1CoreError("recipe rule scores must be exact integers")
        if (
            not isinstance(self.behavior_sha256, str)
            or len(self.behavior_sha256) != 64
        ):
            raise EbmNlpP1CoreError("recipe behavior hash is invalid")


@dataclass(frozen=True)
class CandidateFeatures:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != E1_FEATURE_COUNT:
            raise EbmNlpP1CoreError("candidate feature width drifted")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.values):
            raise EbmNlpP1CoreError(
                "candidate features must be exact quantized integers"
            )
        one_hot = self.values[len(BASE_FEATURE_ORDER) :]
        if sum(one_hot) != INTEGER_SCALE or any(
            value not in (0, INTEGER_SCALE) for value in one_hot
        ):
            raise EbmNlpP1CoreError("recipe coordinates are not one-hot")

    def as_mapping(self) -> dict[str, int]:
        return dict(zip(FEATURE_ORDER, self.values))


@dataclass(frozen=True)
class RecipeSlate:
    actions: tuple[RecipeAction, ...]
    features: tuple[CandidateFeatures, ...]

    def __post_init__(self) -> None:
        if (
            len(self.actions) != len(RECIPE_IDS)
            or len(self.features) != len(RECIPE_IDS)
        ):
            raise EbmNlpP1CoreError("slate must contain all six recipes")
        if tuple(action.recipe_id for action in self.actions) != RECIPE_IDS:
            raise EbmNlpP1CoreError("slate recipe order drifted")


def _validated_recipe_inputs(
    windows: Sequence[EvidenceWindow],
    target_role: str,
    role_probabilities: Mapping[str, Sequence[object]],
    query_cosines: Sequence[object],
    embeddings: Sequence[Sequence[object]],
) -> tuple[
    tuple[EvidenceWindow, ...],
    str,
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]:
    canonical_windows = tuple(windows)
    if not canonical_windows:
        raise EbmNlpP1CoreError("window registry must be nonempty")
    for ordinal, window in enumerate(canonical_windows):
        if window.ordinal != ordinal:
            raise EbmNlpP1CoreError("window registry ordinal drifted")
    if tuple(
        sorted(canonical_windows, key=lambda row: (row.start, row.end))
    ) != canonical_windows:
        raise EbmNlpP1CoreError("window registry ordering drifted")
    role = _validate_role(target_role)
    if set(role_probabilities) != set(ROLE_ORDER):
        raise EbmNlpP1CoreError("role probability registry drifted")
    probabilities: dict[str, tuple[int, ...]] = {}
    for candidate_role in ROLE_ORDER:
        row = tuple(
            _quantized_unit(
                value, field=f"{candidate_role} probe probability"
            )
            for value in role_probabilities[candidate_role]
        )
        if len(row) != len(canonical_windows):
            raise EbmNlpP1CoreError("role probability row count drifted")
        probabilities[candidate_role] = row
    cosines = tuple(
        _quantized_unit(value, field="query cosine")
        for value in query_cosines
    )
    if len(cosines) != len(canonical_windows):
        raise EbmNlpP1CoreError("query cosine row count drifted")
    raw_embeddings = tuple(tuple(row) for row in embeddings)
    if len(raw_embeddings) != len(canonical_windows):
        raise EbmNlpP1CoreError("embedding row count drifted")
    width: int | None = None
    quantized_embeddings: list[tuple[int, ...]] = []
    for row in raw_embeddings:
        if not row:
            raise EbmNlpP1CoreError("embedding rows must be nonempty")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise EbmNlpP1CoreError("embedding width drifted")
        quantized_row = tuple(
            _exact_int(value, field="quantized embedding coordinate")
            for value in row
        )
        if not any(quantized_row):
            raise EbmNlpP1CoreError("embedding row must have nonzero norm")
        quantized_embeddings.append(quantized_row)
    return (
        canonical_windows,
        role,
        probabilities,
        cosines,
        tuple(quantized_embeddings),
    )


def _rank_indices(
    indices: Sequence[int],
    scores: Mapping[int, int],
    windows: Sequence[EvidenceWindow],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            indices,
            key=lambda ordinal: (
                -scores[ordinal],
                windows[ordinal].start,
                windows[ordinal].end,
                ordinal,
            ),
        )
    )


def _log_odds_quantized(probability: int) -> int:
    clipped = min(INTEGER_SCALE - 1, max(1, probability))
    with localcontext() as context:
        context.prec = 50
        odds = Decimal(clipped) / Decimal(INTEGER_SCALE - clipped)
        return int(
            (odds.ln() * INTEGER_SCALE).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        )


def _pairwise_diversity(
    left: Sequence[int], right: Sequence[int]
) -> int:
    dot = sum(a * b for a, b in zip(left, right))
    left_square = sum(value * value for value in left)
    right_square = sum(value * value for value in right)
    if left_square <= 0 or right_square <= 0:
        raise EbmNlpP1CoreError("embedding row must have nonzero norm")
    with localcontext() as context:
        context.prec = 50
        denominator = (
            Decimal(left_square) * Decimal(right_square)
        ).sqrt()
        cosine = Decimal(dot) / denominator
        cosine = min(Decimal(1), max(Decimal(-1), cosine))
        unit_distance = (Decimal(1) - cosine) / Decimal(2)
        return int(
            (unit_distance * INTEGER_SCALE).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        )


def _token_posterior_mass(
    windows: Sequence[EvidenceWindow], target: Sequence[int]
) -> tuple[int, ...]:
    token_count = max(window.end for window in windows)
    mass = [0] * token_count
    for window, probability in zip(windows, target):
        for token_index in range(window.start, window.end):
            mass[token_index] = max(mass[token_index], probability)
    return tuple(mass)


def _make_action(
    recipe_id: str,
    ranking: Sequence[int],
    scores: Sequence[int],
    windows: Sequence[EvidenceWindow],
) -> RecipeAction:
    ordinals = tuple(ranking)
    window_ids = tuple(windows[ordinal].window_id for ordinal in ordinals)
    rule_scores = tuple(scores)
    behavior = {
        "recipe_id": recipe_id,
        "window_ids": window_ids,
    }
    return RecipeAction(
        recipe_id=recipe_id,
        registry_ordinal=RECIPE_IDS.index(recipe_id),
        window_ordinals=ordinals,
        window_ids=window_ids,
        quantized_rule_scores=rule_scores,
        behavior_sha256=canonical_sha256(behavior),
    )


def materialize_recipe_actions(
    *,
    windows: Sequence[EvidenceWindow],
    target_role: str,
    role_probabilities: Mapping[str, Sequence[object]],
    query_cosines: Sequence[object],
    embeddings: Sequence[Sequence[object]],
    top_k: int = TOP_K,
) -> tuple[RecipeAction, ...]:
    """Materialize exactly the six closed-grammar, label-free rankings."""

    (
        canonical_windows,
        role,
        probabilities,
        _,
        quantized_embeddings,
    ) = _validated_recipe_inputs(
        windows,
        target_role,
        role_probabilities,
        query_cosines,
        embeddings,
    )
    requested_k = _exact_int(top_k, field="top_k")
    if requested_k != TOP_K:
        raise EbmNlpP1CoreError("formal top_k is frozen at five")
    k = min(TOP_K, len(canonical_windows))
    all_indices = tuple(range(len(canonical_windows)))
    target = probabilities[role]
    other_roles = tuple(candidate for candidate in ROLE_ORDER if candidate != role)
    maximum_other = tuple(
        max(probabilities[other_roles[0]][index], probabilities[other_roles[1]][index])
        for index in all_indices
    )

    r0_scores = {index: target[index] for index in all_indices}
    r0_full = _rank_indices(all_indices, r0_scores, canonical_windows)
    r0 = r0_full[:k]

    r1_scores = {
        index: target[index] - maximum_other[index]
        for index in all_indices
    }
    r1 = _rank_indices(all_indices, r1_scores, canonical_windows)[:k]

    log_odds = tuple(_log_odds_quantized(value) for value in target)
    paths: list[tuple[int, int, int, int]] = []
    for start_index in all_indices:
        for length in range(
            1, min(TOP_K, len(canonical_windows) - start_index) + 1
        ):
            end_index = start_index + length
            paths.append(
                (
                    sum(log_odds[start_index:end_index]),
                    canonical_windows[start_index].start,
                    canonical_windows[end_index - 1].end,
                    end_index,
                )
            )
    best_score, best_start, _, best_end = min(
        paths,
        key=lambda row: (-row[0], row[1], row[2]),
    )
    path_start_index = next(
        index
        for index, window in enumerate(canonical_windows)
        if window.start == best_start
    )
    path_indices = tuple(range(path_start_index, best_end))
    ranked_path = _rank_indices(path_indices, r0_scores, canonical_windows)
    r2 = list(ranked_path)
    r2.extend(index for index in r0_full if index not in r2)
    r2 = r2[:k]
    r2_scores = {
        index: target[index] if index in path_indices else target[index]
        for index in all_indices
    }
    # ``best_score`` is intentionally computed and fixed above; selection of
    # the path, not the member ranking, is the contiguous-MAP operation.
    _ = best_score

    token_mass = _token_posterior_mass(canonical_windows, target)
    uncovered = set(range(len(token_mass)))
    remaining = set(all_indices)
    r3: list[int] = []
    r3_selection_scores: list[int] = []
    while remaining and len(r3) < k:
        gains = {
            index: sum(
                token_mass[position]
                for position in range(
                    canonical_windows[index].start,
                    canonical_windows[index].end,
                )
                if position in uncovered
            )
            for index in remaining
        }
        chosen = _rank_indices(
            tuple(remaining), gains, canonical_windows
        )[0]
        r3.append(chosen)
        r3_selection_scores.append(gains[chosen])
        remaining.remove(chosen)
        uncovered.difference_update(
            range(
                canonical_windows[chosen].start,
                canonical_windows[chosen].end,
            )
        )
    for index in r0_full:
        if len(r3) >= k:
            break
        if index not in r3:
            r3.append(index)
            r3_selection_scores.append(target[index])

    remaining = set(all_indices)
    r4: list[int] = []
    r4_selection_scores: list[int] = []
    while remaining and len(r4) < k:
        diverse_scores: dict[int, int] = {}
        for index in remaining:
            if not r4:
                mean_distance = 0
            else:
                mean_distance = _round_fraction_half_even(
                    Fraction(
                        sum(
                            _pairwise_diversity(
                                quantized_embeddings[index],
                                quantized_embeddings[selected],
                            )
                            for selected in r4
                        ),
                        len(r4),
                    )
                )
            diverse_scores[index] = target[index] + mean_distance
        chosen = _rank_indices(
            tuple(remaining), diverse_scores, canonical_windows
        )[0]
        r4.append(chosen)
        r4_selection_scores.append(diverse_scores[chosen])
        remaining.remove(chosen)
    for index in r0_full:
        if len(r4) >= k:
            break
        if index not in r4:
            r4.append(index)
            r4_selection_scores.append(target[index])

    r5_scores = {
        index: target[index] + min(target[index], maximum_other[index])
        for index in all_indices
    }
    r5 = _rank_indices(all_indices, r5_scores, canonical_windows)[:k]

    return (
        _make_action(
            R0_TARGET_POSTERIOR,
            r0,
            tuple(r0_scores[index] for index in r0),
            canonical_windows,
        ),
        _make_action(
            R1_ROLE_CONTRAST,
            r1,
            tuple(r1_scores[index] for index in r1),
            canonical_windows,
        ),
        _make_action(
            R2_CONTIGUOUS_MAP,
            r2,
            tuple(r2_scores[index] for index in r2),
            canonical_windows,
        ),
        _make_action(
            R3_DISTINCT_POSTERIOR_COVERAGE,
            r3,
            r3_selection_scores,
            canonical_windows,
        ),
        _make_action(
            R4_SEMANTIC_DIVERSE_SET,
            r4,
            r4_selection_scores,
            canonical_windows,
        ),
        _make_action(
            R5_OVERLAP_PRESERVING_JOINT,
            r5,
            tuple(r5_scores[index] for index in r5),
            canonical_windows,
        ),
    )


def _binary_entropy_quantized(probability: int) -> int:
    if probability in (0, INTEGER_SCALE):
        return 0
    with localcontext() as context:
        context.prec = 50
        p = Decimal(probability) / Decimal(INTEGER_SCALE)
        entropy = -(p * p.ln() + (Decimal(1) - p) * (Decimal(1) - p).ln())
        normalized = entropy / Decimal(2).ln()
        return int(
            (normalized * INTEGER_SCALE).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        )


def compute_candidate_features(
    *,
    action: RecipeAction,
    windows: Sequence[EvidenceWindow],
    target_role: str,
    role_probabilities: Mapping[str, Sequence[object]],
    query_cosines: Sequence[object],
    embeddings: Sequence[Sequence[object]],
) -> CandidateFeatures:
    (
        canonical_windows,
        role,
        probabilities,
        cosines,
        quantized_embeddings,
    ) = _validated_recipe_inputs(
        windows,
        target_role,
        role_probabilities,
        query_cosines,
        embeddings,
    )
    selected = action.window_ordinals
    if any(index >= len(canonical_windows) for index in selected):
        raise EbmNlpP1CoreError("action references an unknown window")
    if action.window_ids != tuple(
        canonical_windows[index].window_id for index in selected
    ):
        raise EbmNlpP1CoreError("action window identity drifted")
    target = probabilities[role]
    other_roles = tuple(candidate for candidate in ROLE_ORDER if candidate != role)
    contrast = tuple(
        target[index]
        - max(
            probabilities[other_roles[0]][index],
            probabilities[other_roles[1]][index],
        )
        for index in selected
    )
    selected_target = tuple(target[index] for index in selected)
    mean_target = _round_fraction_half_even(
        Fraction(sum(selected_target), len(selected_target))
    )
    mean_contrast = _round_fraction_half_even(
        Fraction(sum(contrast), len(contrast))
    )

    token_mass = _token_posterior_mass(canonical_windows, target)
    selected_positions: set[int] = set()
    for index in selected:
        selected_positions.update(
            range(canonical_windows[index].start, canonical_windows[index].end)
        )
    total_mass = sum(token_mass)
    union_mass = (
        0
        if total_mass == 0
        else _round_fraction_half_even(
            Fraction(
                INTEGER_SCALE
                * sum(token_mass[position] for position in selected_positions),
                total_mass,
            )
        )
    )
    mean_query_cosine = _round_fraction_half_even(
        Fraction(sum(cosines[index] for index in selected), len(selected))
    )
    pairs = [
        _pairwise_diversity(
            quantized_embeddings[left], quantized_embeddings[right]
        )
        for offset, left in enumerate(selected)
        for right in selected[offset + 1 :]
    ]
    mean_diversity = (
        0
        if not pairs
        else _round_fraction_half_even(Fraction(sum(pairs), len(pairs)))
    )
    token_count = max(window.end for window in canonical_windows)
    position_range = _round_fraction_half_even(
        Fraction(
            INTEGER_SCALE
            * (
                max(canonical_windows[index].end for index in selected)
                - min(canonical_windows[index].start for index in selected)
            ),
            token_count,
        )
    )
    selected_occurrences = sum(
        canonical_windows[index].token_count for index in selected
    )
    overlap = _round_fraction_half_even(
        Fraction(
            INTEGER_SCALE * (selected_occurrences - len(selected_positions)),
            selected_occurrences,
        )
    )
    entropy = _round_fraction_half_even(
        Fraction(
            sum(_binary_entropy_quantized(value) for value in selected_target),
            len(selected_target),
        )
    )
    one_hot = tuple(
        INTEGER_SCALE if recipe_id == action.recipe_id else 0
        for recipe_id in RECIPE_IDS
    )
    return CandidateFeatures(
        (
            mean_target,
            min(selected_target),
            max(selected_target),
            mean_contrast,
            union_mass,
            mean_query_cosine,
            mean_diversity,
            position_range,
            overlap,
            entropy,
            *one_hot,
        )
    )


def build_recipe_slate(
    *,
    windows: Sequence[EvidenceWindow],
    target_role: str,
    role_probabilities: Mapping[str, Sequence[object]],
    query_cosines: Sequence[object],
    embeddings: Sequence[Sequence[object]],
) -> RecipeSlate:
    actions = materialize_recipe_actions(
        windows=windows,
        target_role=target_role,
        role_probabilities=role_probabilities,
        query_cosines=query_cosines,
        embeddings=embeddings,
    )
    features = tuple(
        compute_candidate_features(
            action=action,
            windows=windows,
            target_role=target_role,
            role_probabilities=role_probabilities,
            query_cosines=query_cosines,
            embeddings=embeddings,
        )
        for action in actions
    )
    return RecipeSlate(actions=actions, features=features)


def raw_probe_ranking(slate: RecipeSlate) -> RecipeAction:
    """Return RAW's shared-probe descending ranking (the frozen R0 action)."""

    return slate.actions[RECIPE_IDS.index(R0_TARGET_POSTERIOR)]


def e0_score(features: CandidateFeatures) -> Fraction:
    row = features.as_mapping()
    return (
        Fraction(5, 20) * row["mean_target_probe_probability"]
        + Fraction(4, 20) * row["minimum_target_probe_probability"]
        + Fraction(3, 20)
        * row["mean_target_minus_max_other_role_probability"]
        + Fraction(3, 20)
        * row["selected_union_target_posterior_mass_fraction"]
        + Fraction(2, 20)
        * row["mean_role_query_MiniLM_cosine_unit_interval"]
        + Fraction(2, 20) * row["mean_pairwise_MiniLM_diversity"]
        + Fraction(1, 20)
        * row["selected_token_position_range_fraction"]
        - Fraction(2, 20) * row["selected_window_overlap_fraction"]
    )


def select_e0(slate: RecipeSlate) -> RecipeAction:
    return min(
        zip(slate.actions, slate.features),
        key=lambda pair: (
            -e0_score(pair[1]),
            pair[0].registry_ordinal,
        ),
    )[0]


def e1_feature_tensors(
    slates: Sequence[RecipeSlate],
    utility_slates: Sequence[Sequence[object]],
    *,
    standardization_slates: Sequence[RecipeSlate] | None = None,
) -> tuple[
    tuple[tuple[tuple[float, ...], ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[bool, ...],
]:
    """Build the exact standardized full-slate tensors without importing torch."""

    canonical_slates = tuple(slates)
    canonical_utilities = tuple(tuple(row) for row in utility_slates)
    if not canonical_slates or len(canonical_slates) != len(canonical_utilities):
        raise EbmNlpP1CoreError("E1 training slate population is invalid")
    normalization_population = (
        canonical_slates
        if standardization_slates is None
        else tuple(standardization_slates)
    )
    if not normalization_population:
        raise EbmNlpP1CoreError(
            "E1 standardization slate population is empty"
        )
    raw_features = [
        tuple(float(value) for value in features.values)
        for slate in normalization_population
        for features in slate.features
    ]
    means = tuple(
        sum(row[column] for row in raw_features) / len(raw_features)
        for column in range(E1_FEATURE_COUNT)
    )
    variances = tuple(
        sum((row[column] - means[column]) ** 2 for row in raw_features)
        / len(raw_features)
        for column in range(E1_FEATURE_COUNT)
    )
    standard_deviations = tuple(math.sqrt(value) for value in variances)
    zero_variance = tuple(value == 0.0 for value in standard_deviations)
    standardized: list[tuple[tuple[float, ...], ...]] = []
    targets: list[tuple[float, ...]] = []
    for slate, utilities in zip(canonical_slates, canonical_utilities):
        if len(utilities) != len(RECIPE_IDS):
            raise EbmNlpP1CoreError("E1 target slate width drifted")
        target_row: list[float] = []
        for value in utilities:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, Fraction)
            ):
                raise EbmNlpP1CoreError("E1 targets must be finite reals")
            converted = float(value)
            if not math.isfinite(converted):
                raise EbmNlpP1CoreError("E1 targets must be finite reals")
            target_row.append(converted)
        targets.append(tuple(target_row))
        standardized.append(
            tuple(
                tuple(
                    0.0
                    if zero_variance[column]
                    else (
                        features.values[column] - means[column]
                    )
                    / standard_deviations[column]
                    for column in range(E1_FEATURE_COUNT)
                )
                for features in slate.features
            )
        )
    return (
        tuple(standardized),
        tuple(targets),
        means,
        standard_deviations,
        zero_variance,
    )


@dataclass(frozen=True)
class E1DeepSetsModel:
    """Portable float64 inference state for the frozen DeepSets evaluator."""

    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    zero_variance: tuple[bool, ...]
    phi1_weight: tuple[tuple[float, ...], ...]
    phi1_bias: tuple[float, ...]
    phi2_weight: tuple[tuple[float, ...], ...]
    phi2_bias: tuple[float, ...]
    rho1_weight: tuple[tuple[float, ...], ...]
    rho1_bias: tuple[float, ...]
    rho2_weight: tuple[float, ...]
    rho2_bias: float
    training_slate_count: int
    standardization_slate_count: int
    training_epoch_count: int = 400

    def __post_init__(self) -> None:
        if (
            len(self.means) != E1_FEATURE_COUNT
            or len(self.standard_deviations) != E1_FEATURE_COUNT
            or len(self.zero_variance) != E1_FEATURE_COUNT
            or len(self.phi1_weight) != 24
            or any(len(row) != E1_FEATURE_COUNT for row in self.phi1_weight)
            or len(self.phi1_bias) != 24
            or len(self.phi2_weight) != 12
            or any(len(row) != 24 for row in self.phi2_weight)
            or len(self.phi2_bias) != 12
            or len(self.rho1_weight) != 16
            or any(len(row) != 24 for row in self.rho1_weight)
            or len(self.rho1_bias) != 16
            or len(self.rho2_weight) != 16
            or self.training_slate_count <= 0
            or self.standardization_slate_count
            < self.training_slate_count
            or self.training_epoch_count != 400
        ):
            raise EbmNlpP1CoreError("E1 model architecture drifted")
        floats = (
            self.means
            + self.standard_deviations
            + tuple(value for row in self.phi1_weight for value in row)
            + self.phi1_bias
            + tuple(value for row in self.phi2_weight for value in row)
            + self.phi2_bias
            + tuple(value for row in self.rho1_weight for value in row)
            + self.rho1_bias
            + self.rho2_weight
            + (self.rho2_bias,)
        )
        if not all(math.isfinite(value) for value in floats):
            raise EbmNlpP1CoreError("E1 model contains non-finite state")

    @staticmethod
    def _layer(
        inputs: Sequence[float],
        weights: Sequence[Sequence[float]],
        biases: Sequence[float],
        *,
        relu: bool,
    ) -> tuple[float, ...]:
        outputs = tuple(
            sum(weight * value for weight, value in zip(row, inputs)) + bias
            for row, bias in zip(weights, biases)
        )
        if relu:
            return tuple(max(0.0, value) for value in outputs)
        return outputs

    def predict_slate(self, slate: RecipeSlate) -> tuple[float, ...]:
        standardized = tuple(
            tuple(
                0.0
                if self.zero_variance[column]
                else (
                    features.values[column] - self.means[column]
                )
                / self.standard_deviations[column]
                for column in range(E1_FEATURE_COUNT)
            )
            for features in slate.features
        )
        phi = tuple(
            self._layer(
                self._layer(
                    row,
                    self.phi1_weight,
                    self.phi1_bias,
                    relu=True,
                ),
                self.phi2_weight,
                self.phi2_bias,
                relu=False,
            )
            for row in standardized
        )
        pooled = tuple(
            sum(row[column] for row in phi) / len(phi)
            for column in range(12)
        )
        scores: list[float] = []
        for row in phi:
            hidden = self._layer(
                row + pooled,
                self.rho1_weight,
                self.rho1_bias,
                relu=True,
            )
            score = sum(
                weight * value
                for weight, value in zip(self.rho2_weight, hidden)
            ) + self.rho2_bias
            if not math.isfinite(score):
                raise EbmNlpP1CoreError(
                    "E1 produced a non-finite prediction"
                )
            scores.append(score)
        return tuple(scores)


def fit_e1_deepsets(
    slates: Sequence[RecipeSlate],
    utility_slates: Sequence[Sequence[object]],
    *,
    standardization_slates: Sequence[RecipeSlate] | None = None,
    torch_module: object | None = None,
) -> E1DeepSetsModel:
    """Fit the frozen CPU float64 DeepSets model for exactly 400 epochs."""

    features, targets, means, deviations, zero_variance = e1_feature_tensors(
        slates,
        utility_slates,
        standardization_slates=standardization_slates,
    )
    standardization_count = (
        len(tuple(slates))
        if standardization_slates is None
        else len(tuple(standardization_slates))
    )
    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except ImportError as exc:  # pragma: no cover - environment specific
            raise EbmNlpP1CoreError(
                "torch is required to fit the frozen E1 evaluator"
            ) from exc
        torch = imported_torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(314159)
    nn = torch.nn

    class _FrozenDeepSets(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.phi1 = nn.Linear(E1_FEATURE_COUNT, 24, dtype=torch.float64)
            self.phi2 = nn.Linear(24, 12, dtype=torch.float64)
            self.rho1 = nn.Linear(24, 16, dtype=torch.float64)
            self.rho2 = nn.Linear(16, 1, dtype=torch.float64)

        def forward(self, rows: object) -> object:
            phi = self.phi2(torch.relu(self.phi1(rows)))
            pooled = phi.mean(dim=1, keepdim=True).expand(-1, len(RECIPE_IDS), -1)
            hidden = torch.relu(self.rho1(torch.cat((phi, pooled), dim=2)))
            return self.rho2(hidden).squeeze(-1)

    model = _FrozenDeepSets()
    feature_tensor = torch.tensor(features, dtype=torch.float64, device="cpu")
    target_tensor = torch.tensor(targets, dtype=torch.float64, device="cpu")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(feature_tensor)
        loss = torch.mean((prediction - target_tensor) ** 2)
        loss.backward()
        optimizer.step()

    def matrix(parameter: object) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(value) for value in row)
            for row in parameter.detach().cpu().tolist()
        )

    def vector(parameter: object) -> tuple[float, ...]:
        return tuple(float(value) for value in parameter.detach().cpu().tolist())

    return E1DeepSetsModel(
        means=means,
        standard_deviations=deviations,
        zero_variance=zero_variance,
        phi1_weight=matrix(model.phi1.weight),
        phi1_bias=vector(model.phi1.bias),
        phi2_weight=matrix(model.phi2.weight),
        phi2_bias=vector(model.phi2.bias),
        rho1_weight=matrix(model.rho1.weight),
        rho1_bias=vector(model.rho1.bias),
        rho2_weight=vector(model.rho2.weight[0]),
        rho2_bias=float(model.rho2.bias[0].detach().cpu()),
        training_slate_count=len(features),
        standardization_slate_count=standardization_count,
    )


def select_e1(model: E1DeepSetsModel, slate: RecipeSlate) -> RecipeAction:
    predictions = model.predict_slate(slate)
    return min(
        zip(slate.actions, predictions),
        key=lambda pair: (-pair[1], pair[0].registry_ordinal),
    )[0]


@dataclass(frozen=True)
class TokenCoverageScore:
    defined: bool
    primary_utility: Fraction | None
    undiscounted_coverage_at_5: Fraction | None
    complete_at_5: int | None
    newly_covered_positive_counts: tuple[int, ...]
    positive_token_count: int


def score_ranked_token_coverage(
    *,
    windows: Sequence[EvidenceWindow],
    ranking: Sequence[int],
    positive_token_positions: Sequence[object],
) -> TokenCoverageScore:
    """Score exact rank-discounted incremental positive-token coverage."""

    canonical_windows = tuple(windows)
    if not canonical_windows:
        raise EbmNlpP1CoreError("window registry must be nonempty")
    ordinals = tuple(
        _exact_int(value, field="ranked window ordinal") for value in ranking
    )
    if (
        not ordinals
        or len(ordinals) > TOP_K
        or len(set(ordinals)) != len(ordinals)
        or any(value < 0 or value >= len(canonical_windows) for value in ordinals)
    ):
        raise EbmNlpP1CoreError("ranking must be one distinct top-five prefix")
    token_count = max(window.end for window in canonical_windows)
    positives: set[int] = set()
    for raw_position in positive_token_positions:
        position = _exact_int(raw_position, field="positive token position")
        if position < 0 or position >= token_count:
            raise EbmNlpP1CoreError("positive token position is outside document")
        positives.add(position)
    if not positives:
        return TokenCoverageScore(
            defined=False,
            primary_utility=None,
            undiscounted_coverage_at_5=None,
            complete_at_5=None,
            newly_covered_positive_counts=tuple(0 for _ in ordinals),
            positive_token_count=0,
        )
    remaining = set(positives)
    newly_covered_counts: list[int] = []
    discounted = Fraction()
    for rank, ordinal in enumerate(ordinals, start=1):
        window = canonical_windows[ordinal]
        newly = {
            position
            for position in remaining
            if window.start <= position < window.end
        }
        count = len(newly)
        newly_covered_counts.append(count)
        discounted += Fraction(count, rank)
        remaining.difference_update(newly)
    covered = len(positives) - len(remaining)
    return TokenCoverageScore(
        defined=True,
        primary_utility=(
            discounted / len(positives) / HARMONIC_5
        ),
        undiscounted_coverage_at_5=Fraction(covered, len(positives)),
        complete_at_5=int(not remaining),
        newly_covered_positive_counts=tuple(newly_covered_counts),
        positive_token_count=len(positives),
    )


def aggregate_abstract_role_utilities(
    role_utilities: Mapping[str, Fraction | None],
) -> Fraction | None:
    if set(role_utilities) != set(ROLE_ORDER):
        raise EbmNlpP1CoreError("abstract role utility registry drifted")
    defined: list[Fraction] = []
    for role in ROLE_ORDER:
        value = role_utilities[role]
        if value is None:
            continue
        if not isinstance(value, Fraction):
            raise EbmNlpP1CoreError("role utility must be exact Fraction or None")
        defined.append(value)
    if not defined:
        return None
    return sum(defined, Fraction()) / len(defined)


def family_aggregate(
    abstract_role_utilities: Sequence[
        Mapping[str, Fraction | None]
    ],
) -> dict[str, Fraction | None]:
    aggregates: dict[str, Fraction | None] = {}
    for role in ROLE_ORDER:
        values = [
            row[role]
            for row in abstract_role_utilities
            if row[role] is not None
        ]
        if any(not isinstance(value, Fraction) for value in values):
            raise EbmNlpP1CoreError("role utility must be exact Fraction or None")
        aggregates[role] = (
            None
            if not values
            else sum(values, Fraction()) / len(values)  # type: ignore[arg-type]
        )
    return aggregates


@dataclass(frozen=True)
class ExactSignTest:
    gains: int
    harms: int
    ties: int
    nonzero_count: int
    one_sided_p: Fraction

    @property
    def passes_point_one(self) -> bool:
        return self.gains > self.harms and self.one_sided_p <= Fraction(1, 10)


def exact_one_sided_sign_test(
    deltas: Sequence[Fraction | int],
) -> ExactSignTest:
    exact = tuple(
        value if isinstance(value, Fraction) else Fraction(value)
        for value in deltas
    )
    gains = sum(value > 0 for value in exact)
    harms = sum(value < 0 for value in exact)
    ties = len(exact) - gains - harms
    nonzero = gains + harms
    if nonzero == 0:
        probability = Fraction(1)
    else:
        probability = Fraction(
            sum(
                math.comb(nonzero, count)
                for count in range(gains, nonzero + 1)
            ),
            2**nonzero,
        )
    return ExactSignTest(
        gains=gains,
        harms=harms,
        ties=ties,
        nonzero_count=nonzero,
        one_sided_p=probability,
    )


@dataclass(frozen=True)
class PairedAbstractComparison:
    paired_deltas: tuple[Fraction, ...]
    mean_delta: Fraction
    zero_defined_abstract_count: int
    sign_test: ExactSignTest
    family_deltas: Mapping[str, Fraction | None]


def compare_abstract_arms(
    left: Sequence[Mapping[str, Fraction | None]],
    right: Sequence[Mapping[str, Fraction | None]],
) -> PairedAbstractComparison:
    if len(left) != len(right):
        raise EbmNlpP1CoreError("paired abstract population drifted")
    deltas: list[Fraction] = []
    zero_defined = 0
    family_values: dict[str, list[Fraction]] = {
        role: [] for role in ROLE_ORDER
    }
    for left_row, right_row in zip(left, right):
        if set(left_row) != set(ROLE_ORDER) or set(right_row) != set(ROLE_ORDER):
            raise EbmNlpP1CoreError("abstract role utility registry drifted")
        for role in ROLE_ORDER:
            left_value = left_row[role]
            right_value = right_row[role]
            if (left_value is None) != (right_value is None):
                raise EbmNlpP1CoreError(
                    "paired arm role-definedness drifted"
                )
            if left_value is not None:
                if not isinstance(left_value, Fraction) or not isinstance(
                    right_value, Fraction
                ):
                    raise EbmNlpP1CoreError(
                        "role utility must be exact Fraction or None"
                    )
                family_values[role].append(left_value - right_value)
        left_cluster = aggregate_abstract_role_utilities(left_row)
        right_cluster = aggregate_abstract_role_utilities(right_row)
        if (left_cluster is None) != (right_cluster is None):
            raise EbmNlpP1CoreError("paired abstract-definedness drifted")
        if left_cluster is None:
            zero_defined += 1
        else:
            assert right_cluster is not None
            deltas.append(left_cluster - right_cluster)
    mean_delta = (
        Fraction()
        if not deltas
        else sum(deltas, Fraction()) / len(deltas)
    )
    family_deltas = {
        role: (
            None
            if not values
            else sum(values, Fraction()) / len(values)
        )
        for role, values in family_values.items()
    }
    return PairedAbstractComparison(
        paired_deltas=tuple(deltas),
        mean_delta=mean_delta,
        zero_defined_abstract_count=zero_defined,
        sign_test=exact_one_sided_sign_test(deltas),
        family_deltas=family_deltas,
    )


def hmac_assignment_digest(
    secret: bytes,
    official_split: str,
    pmid: str,
    *,
    study_id: str = STUDY_ID,
) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise EbmNlpP1CoreError("assignment secret must be exactly 32 bytes")
    for value, field in (
        (study_id, "study_id"),
        (official_split, "official_split"),
        (pmid, "PMID"),
    ):
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
        ):
            raise EbmNlpP1CoreError(f"{field} is not a canonical HMAC field")
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise EbmNlpP1CoreError(
                f"{field} must be ASCII"
            ) from exc
    message = (
        study_id.encode("utf-8")
        + b"\x00"
        + official_split.encode("utf-8")
        + b"\x00"
        + pmid.encode("utf-8")
    )
    return hmac.new(secret, message, hashlib.sha256).digest()


def hmac_assignment_order(
    pmids: Sequence[str],
    secret: bytes,
    official_split: str,
    *,
    study_id: str = STUDY_ID,
) -> tuple[str, ...]:
    canonical_pmids = tuple(pmids)
    if len(set(canonical_pmids)) != len(canonical_pmids):
        raise EbmNlpP1CoreError("eligible PMID registry contains duplicates")
    keyed = tuple(
        (
            hmac_assignment_digest(
                secret,
                official_split,
                pmid,
                study_id=study_id,
            ),
            pmid.encode("ascii"),
            pmid,
        )
        for pmid in canonical_pmids
    )
    return tuple(row[2] for row in sorted(keyed))


@dataclass(frozen=True)
class HmacBlockAssignment:
    blocks: tuple[tuple[str, tuple[str, ...]], ...]
    unused: tuple[str, ...]

    def as_mapping(self) -> dict[str, tuple[str, ...]]:
        return dict(self.blocks)


def assign_hmac_blocks(
    pmids: Sequence[str],
    secret: bytes,
    official_split: str,
    block_counts: Sequence[tuple[str, int]],
    *,
    study_id: str = STUDY_ID,
) -> HmacBlockAssignment:
    ordered = hmac_assignment_order(
        pmids, secret, official_split, study_id=study_id
    )
    names: set[str] = set()
    normalized: list[tuple[str, int]] = []
    required = 0
    for name, raw_count in block_counts:
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or "\x00" in name
        ):
            raise EbmNlpP1CoreError("HMAC block name registry drifted")
        count = _exact_int(raw_count, field=f"{name} block count")
        if count < 0:
            raise EbmNlpP1CoreError("HMAC block count cannot be negative")
        names.add(name)
        normalized.append((name, count))
        required += count
    if len(ordered) < required:
        raise EbmNlpP1CoreError(
            "eligible PMID capacity is below frozen block counts"
        )
    blocks: list[tuple[str, tuple[str, ...]]] = []
    cursor = 0
    for name, count in normalized:
        blocks.append((name, ordered[cursor : cursor + count]))
        cursor += count
    return HmacBlockAssignment(tuple(blocks), ordered[cursor:])


__all__ = [
    "BASE_FEATURE_ORDER",
    "CandidateFeatures",
    "ConstantProbabilityProbe",
    "E1DeepSetsModel",
    "E1_FEATURE_COUNT",
    "EbmNlpP1CoreError",
    "EvidenceWindow",
    "ExactSignTest",
    "FEATURE_ORDER",
    "FrozenRoleProbes",
    "HARMONIC_5",
    "HmacBlockAssignment",
    "INTEGER_SCALE",
    "INTERVENTION",
    "OUTCOME",
    "PARTICIPANT",
    "PairedAbstractComparison",
    "R0_TARGET_POSTERIOR",
    "R1_ROLE_CONTRAST",
    "R2_CONTIGUOUS_MAP",
    "R3_DISTINCT_POSTERIOR_COVERAGE",
    "R4_SEMANTIC_DIVERSE_SET",
    "R5_OVERLAP_PRESERVING_JOINT",
    "RECIPE_IDS",
    "ROLE_ORDER",
    "ROLE_QUERIES",
    "RecipeAction",
    "RecipeSlate",
    "STUDY_ID",
    "TOP_K",
    "TokenCoverageScore",
    "WINDOW_STRIDE",
    "WINDOW_WIDTH",
    "aggregate_abstract_role_utilities",
    "assign_hmac_blocks",
    "build_evidence_windows",
    "build_recipe_slate",
    "canonical_json_bytes",
    "canonical_sha256",
    "compare_abstract_arms",
    "compute_candidate_features",
    "e0_score",
    "e1_feature_tensors",
    "exact_one_sided_sign_test",
    "family_aggregate",
    "fit_e1_deepsets",
    "fit_independent_role_probes",
    "hmac_assignment_digest",
    "hmac_assignment_order",
    "materialize_recipe_actions",
    "quantize_half_even",
    "raw_probe_ranking",
    "score_ranked_token_coverage",
    "select_e0",
    "select_e1",
]

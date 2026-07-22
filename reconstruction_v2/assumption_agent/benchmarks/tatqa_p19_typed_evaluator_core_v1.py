"""Pure typed-action and evaluator algebra for the frozen TAT-QA P19 study.

The formal controller owns source access, model execution, action scheduling,
and label boundaries.  This module deliberately owns none of those things.  It
accepts only a validated typed plan and already-quantized, item-local unit
features.  Gold units enter only through :func:`item_utility`, after actions are
otherwise complete; baseline rankings never enter the evaluator feature space.

The implementation follows ``tatqa_p19_typed_evaluator_study_design_v1``:

* P0 ranks every unit by plan-facet coverage, query-anchor connectivity, and
  full-question relevance, with a canonical unit-ID tie break;
* P1 retains P0's first three units and adds two units by query-anchored typed
  residual gain, with zero authority for query-independent components;
* E1 uses six fixed P1-minus-P0 features and a lambda-one, population-
  standardized paired-delta ridge with an unpenalized intercept; and
* utility and the one-sided magnitude-preserving sign-flip test are exact
  :class:`fractions.Fraction` calculations.

There is intentionally no filesystem, dataset, network, model, concurrency, or
online-evaluator API in this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "tatqa_p19_typed_evaluator_core_v1"
TOP_K = 5
RIDGE_LAMBDA = 1.0
RIDGE_SINGULAR_RCOND = 1.0e-12
PROMOTION_ALPHA = Fraction(1, 10)

P0_POLICY_ID = "P0_QUERY_ANCHORED_COVERAGE"
P1_POLICY_ID = "P1_TYPED_CROSS_MODAL_RESIDUAL"
POLICY_IDS = (P0_POLICY_ID, P1_POLICY_ID)

OPERATIONS = (
    "LOOKUP",
    "COMPARE",
    "DIFFERENCE",
    "RATIO",
    "SUM",
    "AVERAGE",
    "COUNT",
    "OTHER",
)

TYPED_EDGE_ORDER = (
    "question_facet_to_unit",
    "table_header_to_row",
    "same_row_metric_to_value",
    "paragraph_to_table_shared_entity_metric_or_time",
    "adjacent_table_row_with_shared_header",
)

FEATURE_ORDER = (
    "typed_facet_coverage_delta",
    "numeric_or_time_operand_coverage_delta",
    "cross_modal_query_anchored_path_delta",
    "dense_relevance_mass_delta",
    "selected_unit_redundancy_delta",
    "P1_outside_P0_unit_count",
)

# These names are documentary as well as defensive.  Feature mappings are
# required to contain FEATURE_ORDER exactly, so every extra key is rejected;
# the frozen forbidden names make the intended late-label boundary explicit.
FORBIDDEN_FEATURES = frozenset(
    {
        "answer",
        "answer_from",
        "family",
        "gold_unit",
        "gold_mapping",
        "HippoRAG_ranking",
        "RAW_ranking",
        "question_identity",
        "table_identity",
        "context_identity",
    }
)

_PLAN_FIELDS = frozenset(
    {
        "entity_facets",
        "metric_facets",
        "time_facets",
        "operation",
        "relation_query",
    }
)
_CANONICAL_UNIT_ID = re.compile(r"(?P<kind>T|P):(?P<ordinal>0|[1-9][0-9]*)\Z")
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


class TatqaP19TypedEvaluatorError(ValueError):
    """Fail-closed error for malformed label-free or exact-offline inputs."""


def _canonical_text(value: object) -> str:
    if not isinstance(value, str):
        raise TatqaP19TypedEvaluatorError("plan facets must be strings")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if not normalized:
        raise TatqaP19TypedEvaluatorError("plan facets must be nonempty")
    return normalized


def _strict_facet_tuple(
    value: object, *, field: str, minimum: int, maximum: int
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TatqaP19TypedEvaluatorError(f"{field} must be an array of strings")
    normalized = tuple(_canonical_text(row) for row in value)
    if not minimum <= len(normalized) <= maximum:
        raise TatqaP19TypedEvaluatorError(
            f"{field} must contain between {minimum} and {maximum} strings"
        )
    folded = tuple(row.casefold() for row in normalized)
    if len(set(folded)) != len(folded):
        raise TatqaP19TypedEvaluatorError(f"{field} contains duplicate facets")
    return normalized


@dataclass(frozen=True)
class TypedPlan:
    """The exact frozen structured-plan schema shared by P0 and P1."""

    entity_facets: tuple[str, ...]
    metric_facets: tuple[str, ...]
    time_facets: tuple[str, ...]
    operation: str
    relation_query: str

    def __post_init__(self) -> None:
        entities = _strict_facet_tuple(
            self.entity_facets, field="entity_facets", minimum=1, maximum=4
        )
        metrics = _strict_facet_tuple(
            self.metric_facets, field="metric_facets", minimum=1, maximum=3
        )
        times = _strict_facet_tuple(
            self.time_facets, field="time_facets", minimum=0, maximum=3
        )
        if not isinstance(self.operation, str) or self.operation not in OPERATIONS:
            raise TatqaP19TypedEvaluatorError("operation is outside the frozen registry")
        relation = _canonical_text(self.relation_query)
        object.__setattr__(self, "entity_facets", entities)
        object.__setattr__(self, "metric_facets", metrics)
        object.__setattr__(self, "time_facets", times)
        object.__setattr__(self, "relation_query", relation)

    @property
    def facet_width(self) -> int:
        """Width of the injected coverage vector, including relation_query."""

        return (
            len(self.entity_facets)
            + len(self.metric_facets)
            + len(self.time_facets)
            + 1
        )

    def payload(self) -> dict[str, object]:
        return {
            "entity_facets": list(self.entity_facets),
            "metric_facets": list(self.metric_facets),
            "time_facets": list(self.time_facets),
            "operation": self.operation,
            "relation_query": self.relation_query,
        }


def validate_typed_plan(value: TypedPlan | Mapping[str, object]) -> TypedPlan:
    """Validate the exact five-field plan schema without repairing it."""

    if isinstance(value, TypedPlan):
        # Reconstructing also protects callers from forged instances produced
        # by bypassing dataclass initialization.
        return TypedPlan(**value.__dict__)
    if not isinstance(value, Mapping):
        raise TatqaP19TypedEvaluatorError("typed plan must be a mapping or TypedPlan")
    supplied = set(value)
    if supplied != _PLAN_FIELDS:
        missing = sorted(_PLAN_FIELDS - supplied)
        extra = sorted(supplied - _PLAN_FIELDS)
        raise TatqaP19TypedEvaluatorError(
            f"typed plan schema drifted; missing={missing}, extra={extra}"
        )
    return TypedPlan(
        entity_facets=_strict_facet_tuple(
            value["entity_facets"],
            field="entity_facets",
            minimum=1,
            maximum=4,
        ),
        metric_facets=_strict_facet_tuple(
            value["metric_facets"],
            field="metric_facets",
            minimum=1,
            maximum=3,
        ),
        time_facets=_strict_facet_tuple(
            value["time_facets"],
            field="time_facets",
            minimum=0,
            maximum=3,
        ),
        operation=value["operation"],  # type: ignore[arg-type]
        relation_query=value["relation_query"],  # type: ignore[arg-type]
    )


def _totalized_strings(value: object, fallback: object, maximum: int) -> tuple[str, ...]:
    rows: list[str] = []
    seen: set[str] = set()
    for source in (value, fallback):
        if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
            continue
        for candidate in source:
            try:
                normalized = _canonical_text(candidate)
            except TatqaP19TypedEvaluatorError:
                continue
            folded = normalized.casefold()
            if folded in seen:
                continue
            rows.append(normalized)
            seen.add(folded)
            if len(rows) == maximum:
                return tuple(rows)
    return tuple(rows)


def totalize_typed_plan(
    candidate: object,
    *,
    fallback_relation_query: object = "unspecified relation",
    fallback_entity_facets: object = (),
    fallback_metric_facets: object = (),
    fallback_time_facets: object = (),
) -> TypedPlan:
    """Deterministically totalize arbitrary structured-model output.

    The caller may inject question/header/lead-derived fallback strings, but
    this function performs no extraction itself.  Invalid elements are dropped,
    case-insensitive duplicates retain their first canonical occurrence, lists
    are capped in input order, an invalid operation becomes ``OTHER``, and the
    canonical fallback relation supplies any required empty entity/metric list.
    Consequently this function returns a valid plan for every Python object.
    """

    raw = candidate if isinstance(candidate, Mapping) else {}
    try:
        fallback_relation = _canonical_text(fallback_relation_query)
    except TatqaP19TypedEvaluatorError:
        fallback_relation = "unspecified relation"
    try:
        relation = _canonical_text(raw.get("relation_query"))
    except TatqaP19TypedEvaluatorError:
        relation = fallback_relation

    entities = _totalized_strings(
        raw.get("entity_facets"), fallback_entity_facets, 4
    )
    metrics = _totalized_strings(raw.get("metric_facets"), fallback_metric_facets, 3)
    times = _totalized_strings(raw.get("time_facets"), fallback_time_facets, 3)
    if not entities:
        entities = (relation,)
    if not metrics:
        metrics = (relation,)

    operation_value = raw.get("operation")
    operation = (
        operation_value.strip().upper()
        if isinstance(operation_value, str)
        else "OTHER"
    )
    if operation not in OPERATIONS:
        operation = "OTHER"
    return TypedPlan(entities, metrics, times, operation, relation)


def _unit_identity(value: object) -> tuple[int, int, str]:
    if not isinstance(value, str):
        raise TatqaP19TypedEvaluatorError("canonical unit ID must be text")
    match = _CANONICAL_UNIT_ID.fullmatch(value)
    if match is None:
        raise TatqaP19TypedEvaluatorError(
            "canonical unit ID must be T:<zero-based-row> or P:<official-order>"
        )
    kind_order = 0 if match.group("kind") == "T" else 1
    ordinal = int(match.group("ordinal"))
    if kind_order == 1 and ordinal == 0:
        raise TatqaP19TypedEvaluatorError(
            "paragraph canonical unit ID must use a positive official order"
        )
    return kind_order, ordinal, value


def _integer_tuple(
    value: object,
    *,
    field: str,
    width: int | None = None,
    binary: bool = False,
    nonnegative: bool = True,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TatqaP19TypedEvaluatorError(f"{field} must be an integer vector")
    normalized = tuple(value)
    if width is not None and len(normalized) != width:
        raise TatqaP19TypedEvaluatorError(f"{field} width drifted")
    if not normalized:
        raise TatqaP19TypedEvaluatorError(f"{field} must be nonempty")
    if any(type(row) is not int for row in normalized):
        raise TatqaP19TypedEvaluatorError(f"{field} must contain exact integers")
    if binary and any(row not in (0, 1) for row in normalized):
        raise TatqaP19TypedEvaluatorError(f"{field} must contain only zero or one")
    if not binary and nonnegative and any(row < 0 for row in normalized):
        raise TatqaP19TypedEvaluatorError(f"{field} must be nonnegative")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class CanonicalUnit:
    """One content-free canonical unit with prequantized action features.

    ``facet_coverage`` is a binary vector aligned to ``TypedPlan`` facets in
    entity, metric, time, relation order.  ``typed_edge_features`` follows
    :data:`TYPED_EDGE_ORDER` and contains only already-qualified,
    query-anchored path mass.  No text, gold mapping, answer, family, or
    baseline ranking is represented.
    """

    unit_id: str
    facet_coverage: tuple[int, ...]
    typed_edge_features: tuple[int, ...]
    numeric_or_time_operand_coverage: int
    full_question_similarity: int

    def __post_init__(self) -> None:
        _unit_identity(self.unit_id)
        facets = _integer_tuple(
            self.facet_coverage, field="facet coverage", binary=True
        )
        edges = _integer_tuple(
            self.typed_edge_features,
            field="typed edge features",
            width=len(TYPED_EDGE_ORDER),
        )
        if (
            type(self.numeric_or_time_operand_coverage) is not int
            or self.numeric_or_time_operand_coverage < 0
        ):
            raise TatqaP19TypedEvaluatorError(
                "numeric/time operand coverage must be a nonnegative integer"
            )
        if type(self.full_question_similarity) is not int:
            raise TatqaP19TypedEvaluatorError(
                "full-question similarity must be a prequantized integer"
            )
        object.__setattr__(self, "facet_coverage", facets)
        object.__setattr__(self, "typed_edge_features", edges)

    def payload(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "facet_coverage": list(self.facet_coverage),
            "typed_edge_features": {
                name: self.typed_edge_features[index]
                for index, name in enumerate(TYPED_EDGE_ORDER)
            },
            "numeric_or_time_operand_coverage": self.numeric_or_time_operand_coverage,
            "full_question_similarity": self.full_question_similarity,
        }


def _validate_units(
    plan: TypedPlan | Mapping[str, object], units: Sequence[CanonicalUnit]
) -> tuple[TypedPlan, tuple[CanonicalUnit, ...]]:
    normalized_plan = validate_typed_plan(plan)
    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise TatqaP19TypedEvaluatorError("canonical units must be a sequence")
    rows = tuple(units)
    if len(rows) < TOP_K or any(not isinstance(row, CanonicalUnit) for row in rows):
        raise TatqaP19TypedEvaluatorError("at least five CanonicalUnit rows are required")
    identifiers = tuple(row.unit_id for row in rows)
    if len(set(identifiers)) != len(identifiers):
        raise TatqaP19TypedEvaluatorError("canonical unit IDs must be distinct")
    if any(len(row.facet_coverage) != normalized_plan.facet_width for row in rows):
        raise TatqaP19TypedEvaluatorError(
            "unit facet coverage width does not match the shared typed plan"
        )
    return normalized_plan, rows


def rank_p0_units(
    plan: TypedPlan | Mapping[str, object], units: Sequence[CanonicalUnit]
) -> tuple[str, ...]:
    """Return the deterministic full P0 ranking over every canonical unit."""

    _normalized_plan, rows = _validate_units(plan, units)
    ranked = sorted(
        rows,
        key=lambda row: (
            -sum(row.facet_coverage),
            -row.typed_edge_features[0],
            -row.full_question_similarity,
            _unit_identity(row.unit_id),
        ),
    )
    return tuple(row.unit_id for row in ranked)


def _validated_top5(value: Sequence[str], *, field: str = "ordered top5") -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TatqaP19TypedEvaluatorError(f"{field} must be a sequence")
    rows = tuple(value)
    if len(rows) != TOP_K or len(set(rows)) != TOP_K:
        raise TatqaP19TypedEvaluatorError(f"{field} must contain five distinct units")
    for row in rows:
        _unit_identity(row)
    return rows


def rank_p1_units(
    plan: TypedPlan | Mapping[str, object],
    units: Sequence[CanonicalUnit],
    p0_top5: Sequence[str],
) -> tuple[str, ...]:
    """Return P1's top five: P0 top three plus two typed residual units.

    A candidate receives residual authority only from a newly covered plan
    facet or an injected query-anchored typed edge.  Operand coverage and dense
    relevance refine an authorized residual but cannot authorize a
    query-independent candidate.  If no authorized residual remains, the full
    P0 order is the deterministic totalizer.
    """

    normalized_plan, rows = _validate_units(plan, units)
    baseline = _validated_top5(p0_top5, field="P0 top5")
    full_p0 = rank_p0_units(normalized_plan, rows)
    if baseline != full_p0[:TOP_K]:
        raise TatqaP19TypedEvaluatorError("P0 top5 is not the deterministic P0 action")
    by_id = {row.unit_id: row for row in rows}
    selected = list(baseline[:3])
    covered = [
        max(by_id[unit_id].facet_coverage[index] for unit_id in selected)
        for index in range(normalized_plan.facet_width)
    ]
    p0_position = {unit_id: index for index, unit_id in enumerate(full_p0)}

    while len(selected) < TOP_K:
        candidates = [row for row in rows if row.unit_id not in selected]

        def residual_key(row: CanonicalUnit) -> tuple[object, ...]:
            new_facets = sum(
                bit == 1 and covered[index] == 0
                for index, bit in enumerate(row.facet_coverage)
            )
            typed_path_gain = sum(row.typed_edge_features)
            authorized = new_facets > 0 or typed_path_gain > 0
            if not authorized:
                # Query-independent components have exactly zero authority.
                return (1, 0, 0, 0, 0, 0, p0_position[row.unit_id])
            total_gain = (
                new_facets
                + typed_path_gain
                + row.numeric_or_time_operand_coverage
            )
            return (
                0,
                -total_gain,
                -new_facets,
                -typed_path_gain,
                -row.numeric_or_time_operand_coverage,
                -row.full_question_similarity,
                _unit_identity(row.unit_id),
            )

        chosen = min(candidates, key=residual_key)
        selected.append(chosen.unit_id)
        covered = [
            max(covered[index], chosen.facet_coverage[index])
            for index in range(normalized_plan.facet_width)
        ]
    return tuple(selected)


def _normalize_redundancy(
    value: Mapping[tuple[str, str], int] | None,
    *,
    valid_ids: frozenset[str],
) -> dict[tuple[str, str], int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TatqaP19TypedEvaluatorError("redundancy features must be a mapping")
    normalized: dict[tuple[str, str], int] = {}
    for pair, weight in value.items():
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or pair[0] == pair[1]
            or pair[0] not in valid_ids
            or pair[1] not in valid_ids
            or type(weight) is not int
            or weight < 0
        ):
            raise TatqaP19TypedEvaluatorError(
                "redundancy entries must bind two distinct known units to a nonnegative integer"
            )
        ordered = tuple(sorted(pair, key=_unit_identity))
        if ordered in normalized:
            raise TatqaP19TypedEvaluatorError("redundancy pair is duplicated in reverse")
        normalized[ordered] = weight
    return normalized


def _selection_components(
    selected: Sequence[str],
    *,
    plan: TypedPlan,
    by_id: Mapping[str, CanonicalUnit],
    redundancy: Mapping[tuple[str, str], int],
) -> tuple[int, int, int, int, int]:
    top5 = _validated_top5(selected)
    coverage = sum(
        max(by_id[unit_id].facet_coverage[index] for unit_id in top5)
        for index in range(plan.facet_width)
    )
    operand = sum(by_id[unit_id].numeric_or_time_operand_coverage for unit_id in top5)
    cross_modal_index = TYPED_EDGE_ORDER.index(
        "paragraph_to_table_shared_entity_metric_or_time"
    )
    cross_modal = sum(
        by_id[unit_id].typed_edge_features[cross_modal_index] for unit_id in top5
    )
    dense_mass = sum(by_id[unit_id].full_question_similarity for unit_id in top5)
    redundancy_mass = 0
    for left_index, left in enumerate(top5):
        for right in top5[left_index + 1 :]:
            pair = tuple(sorted((left, right), key=_unit_identity))
            redundancy_mass += redundancy.get(pair, 0)
    return coverage, operand, cross_modal, dense_mass, redundancy_mass


def p1_minus_p0_features(
    plan: TypedPlan | Mapping[str, object],
    units: Sequence[CanonicalUnit],
    p0_top5: Sequence[str],
    p1_top5: Sequence[str],
    *,
    redundancy_features: Mapping[tuple[str, str], int] | None = None,
) -> tuple[int, ...]:
    """Build the exact six-coordinate, label-free P1-minus-P0 feature vector."""

    normalized_plan, rows = _validate_units(plan, units)
    p0 = _validated_top5(p0_top5, field="P0 top5")
    p1 = _validated_top5(p1_top5, field="P1 top5")
    if p1[:3] != p0[:3]:
        raise TatqaP19TypedEvaluatorError("P1 must retain the ordered P0 top three")
    by_id = {row.unit_id: row for row in rows}
    if not set(p0).issubset(by_id) or not set(p1).issubset(by_id):
        raise TatqaP19TypedEvaluatorError("action selected an unknown canonical unit")
    redundancy = _normalize_redundancy(
        redundancy_features, valid_ids=frozenset(by_id)
    )
    p0_components = _selection_components(
        p0, plan=normalized_plan, by_id=by_id, redundancy=redundancy
    )
    p1_components = _selection_components(
        p1, plan=normalized_plan, by_id=by_id, redundancy=redundancy
    )
    return tuple(
        p1_value - p0_value
        for p1_value, p0_value in zip(p1_components, p0_components)
    ) + (len(set(p1) - set(p0)),)


@dataclass(frozen=True)
class Action:
    """One frozen P0 or P1 action and its content-free evaluator features."""

    policy_id: str
    plan: TypedPlan
    selected_unit_ids: tuple[str, ...]
    feature_vector: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.policy_id not in POLICY_IDS:
            raise TatqaP19TypedEvaluatorError("action policy is outside the frozen registry")
        normalized_plan = validate_typed_plan(self.plan)
        selected = _validated_top5(self.selected_unit_ids)
        features = _integer_tuple(
            self.feature_vector,
            field="evaluator feature vector",
            width=len(FEATURE_ORDER),
            nonnegative=False,
        )
        if self.policy_id == P0_POLICY_ID and any(features):
            raise TatqaP19TypedEvaluatorError("P0 action feature vector must be all zero")
        if not 0 <= features[-1] <= 2:
            raise TatqaP19TypedEvaluatorError("outside-P0 count must be between zero and two")
        object.__setattr__(self, "plan", normalized_plan)
        object.__setattr__(self, "selected_unit_ids", selected)
        object.__setattr__(self, "feature_vector", features)

    def feature_mapping(self) -> dict[str, int]:
        return dict(zip(FEATURE_ORDER, self.feature_vector))

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_action_v1",
            "policy_id": self.policy_id,
            "shared_typed_plan": self.plan.payload(),
            "ordered_top5": list(self.selected_unit_ids),
            "fixed_feature_order": list(FEATURE_ORDER),
            "P1_minus_P0_features": self.feature_mapping(),
        }

    @property
    def action_sha256(self) -> str:
        return canonical_action_hash(self)

    @property
    def behavior_sha256(self) -> str:
        return canonical_behavior_hash(self)


def build_p0_action(
    plan: TypedPlan | Mapping[str, object], units: Sequence[CanonicalUnit]
) -> Action:
    normalized_plan, rows = _validate_units(plan, units)
    selected = rank_p0_units(normalized_plan, rows)[:TOP_K]
    return Action(
        P0_POLICY_ID,
        normalized_plan,
        selected,
        (0,) * len(FEATURE_ORDER),
    )


def build_p1_action(
    plan: TypedPlan | Mapping[str, object],
    units: Sequence[CanonicalUnit],
    p0_action: Action,
    *,
    redundancy_features: Mapping[tuple[str, str], int] | None = None,
) -> Action:
    normalized_plan, rows = _validate_units(plan, units)
    if (
        not isinstance(p0_action, Action)
        or p0_action.policy_id != P0_POLICY_ID
        or p0_action.plan != normalized_plan
    ):
        raise TatqaP19TypedEvaluatorError("P1 requires the shared deterministic P0 action")
    expected_p0 = build_p0_action(normalized_plan, rows)
    if p0_action != expected_p0:
        raise TatqaP19TypedEvaluatorError("supplied P0 action drifted")
    selected = rank_p1_units(
        normalized_plan, rows, p0_action.selected_unit_ids
    )
    features = p1_minus_p0_features(
        normalized_plan,
        rows,
        p0_action.selected_unit_ids,
        selected,
        redundancy_features=redundancy_features,
    )
    return Action(P1_POLICY_ID, normalized_plan, selected, features)


def build_action_pair(
    plan: TypedPlan | Mapping[str, object],
    units: Sequence[CanonicalUnit],
    *,
    redundancy_features: Mapping[tuple[str, str], int] | None = None,
) -> tuple[Action, Action]:
    """Build the shared-plan P0/P1 pair without labels or baselines."""

    p0 = build_p0_action(plan, units)
    p1 = build_p1_action(
        p0.plan, units, p0, redundancy_features=redundancy_features
    )
    return p0, p1


def feature_vector(value: Mapping[str, object] | Sequence[int]) -> tuple[int, ...]:
    """Normalize exactly the frozen feature schema; reject every extra key."""

    if isinstance(value, Mapping):
        supplied = set(value)
        if supplied != set(FEATURE_ORDER):
            forbidden = sorted(supplied.intersection(FORBIDDEN_FEATURES))
            missing = sorted(set(FEATURE_ORDER) - supplied)
            extra = sorted(supplied - set(FEATURE_ORDER))
            raise TatqaP19TypedEvaluatorError(
                "fixed evaluator feature schema drifted; "
                f"forbidden={forbidden}, missing={missing}, extra={extra}"
            )
        rows = tuple(value[name] for name in FEATURE_ORDER)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TatqaP19TypedEvaluatorError("evaluator features must be a mapping or sequence")
        rows = tuple(value)
    if len(rows) != len(FEATURE_ORDER) or any(type(row) is not int for row in rows):
        raise TatqaP19TypedEvaluatorError(
            "evaluator feature vector must contain six exact integers"
        )
    return rows  # type: ignore[return-value]


@dataclass(frozen=True)
class PairedDeltaRidgeModel:
    """Frozen lambda-one standardized ridge with an unpenalized intercept."""

    population_mean: tuple[float, ...]
    population_std: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    solver: str

    def __post_init__(self) -> None:
        width = len(FEATURE_ORDER)
        if (
            len(self.population_mean) != width
            or len(self.population_std) != width
            or len(self.coefficients) != width
            or any(value < 0 for value in self.population_std)
            or not all(
                math.isfinite(value)
                for value in (
                    *self.population_mean,
                    *self.population_std,
                    self.intercept,
                    *self.coefficients,
                )
            )
        ):
            raise TatqaP19TypedEvaluatorError("ridge model is malformed or nonfinite")
        if self.solver not in {
            "numpy_float64_solve_v1",
            "numpy_float64_pinv_rcond_1e-12_v1",
        }:
            raise TatqaP19TypedEvaluatorError("ridge solver identity drifted")

    def standardize(self, features: Mapping[str, object] | Sequence[int]) -> tuple[float, ...]:
        values = feature_vector(features)
        return tuple(
            0.0 if std == 0.0 else (value - mean) / std
            for value, mean, std in zip(
                values, self.population_mean, self.population_std
            )
        )

    def predict(self, features: Mapping[str, object] | Sequence[int]) -> float:
        standardized = self.standardize(features)
        prediction = self.intercept + math.fsum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )
        if not math.isfinite(prediction):
            raise TatqaP19TypedEvaluatorError("ridge prediction is nonfinite")
        return prediction

    def payload(self) -> dict[str, object]:
        return {
            "feature_order": list(FEATURE_ORDER),
            "scaler": "A_form_population_mean_and_population_standard_deviation_v1",
            "zero_variance_maps_to_zero": True,
            "ridge_lambda": 1,
            "intercept_penalized": False,
            "population_mean_float64_hex": [value.hex() for value in self.population_mean],
            "population_std_float64_hex": [value.hex() for value in self.population_std],
            "intercept_float64_hex": self.intercept.hex(),
            "coefficient_float64_hex": [value.hex() for value in self.coefficients],
            "solver": self.solver,
        }


def _exact_delta(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TatqaP19TypedEvaluatorError(
            "paired utility deltas must be exact integers or Fractions"
        )
    return Fraction(value)


def fit_paired_delta_ridge(
    feature_rows: Sequence[Mapping[str, object] | Sequence[int]],
    utility_deltas: Sequence[Fraction | int],
) -> PairedDeltaRidgeModel:
    """Fit the single A_form E1 model with the frozen numerical contract."""

    if (
        isinstance(feature_rows, (str, bytes))
        or isinstance(utility_deltas, (str, bytes))
        or not isinstance(feature_rows, Sequence)
        or not isinstance(utility_deltas, Sequence)
        or not feature_rows
        or len(feature_rows) != len(utility_deltas)
    ):
        raise TatqaP19TypedEvaluatorError(
            "paired ridge requires equally sized nonempty feature and utility rows"
        )
    normalized_features = tuple(feature_vector(row) for row in feature_rows)
    normalized_targets = tuple(_exact_delta(row) for row in utility_deltas)
    x = np.asarray(normalized_features, dtype=np.float64)
    y = np.asarray(
        [row.numerator / row.denominator for row in normalized_targets],
        dtype=np.float64,
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise TatqaP19TypedEvaluatorError("paired ridge matrix is nonfinite")

    means = np.mean(x, axis=0, dtype=np.float64)
    centered = x - means
    stds = np.sqrt(np.mean(centered * centered, axis=0, dtype=np.float64))
    standardized = np.divide(
        centered,
        stds,
        out=np.zeros_like(centered),
        where=stds != 0.0,
    )
    design = np.column_stack(
        (np.ones(len(normalized_features), dtype=np.float64), standardized)
    )
    gram = design.T @ design
    gram[1:, 1:] += np.eye(len(FEATURE_ORDER), dtype=np.float64) * RIDGE_LAMBDA
    rhs = design.T @ y
    solver = "numpy_float64_solve_v1"
    try:
        fitted = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        # Lambda one makes the declared finite system positive definite for
        # every nonempty input.  This fixed pseudoinverse is nevertheless the
        # sole, deterministic handling for an implementation-level singularity.
        fitted = np.linalg.pinv(gram, rcond=RIDGE_SINGULAR_RCOND) @ rhs
        solver = "numpy_float64_pinv_rcond_1e-12_v1"
    if not np.isfinite(fitted).all():
        raise TatqaP19TypedEvaluatorError("ridge solution is nonfinite")
    return PairedDeltaRidgeModel(
        population_mean=tuple(float(value) for value in means),
        population_std=tuple(float(value) for value in stds),
        intercept=float(fitted[0]),
        coefficients=tuple(float(value) for value in fitted[1:]),
        solver=solver,
    )


def _validate_action_pair(p0_action: Action, p1_action: Action) -> None:
    if (
        not isinstance(p0_action, Action)
        or not isinstance(p1_action, Action)
        or p0_action.policy_id != P0_POLICY_ID
        or p1_action.policy_id != P1_POLICY_ID
        or p0_action.plan != p1_action.plan
        or p1_action.selected_unit_ids[:3] != p0_action.selected_unit_ids[:3]
        or p1_action.feature_vector[-1]
        != len(set(p1_action.selected_unit_ids) - set(p0_action.selected_unit_ids))
    ):
        raise TatqaP19TypedEvaluatorError("E0/E1 action pair drifted")


def select_e0_action(p0_action: Action, p1_action: Action) -> Action:
    """E0 always selects P0; P1 is accepted only to verify pair identity."""

    _validate_action_pair(p0_action, p1_action)
    return p0_action


def select_e1_action(
    model: PairedDeltaRidgeModel,
    p0_action: Action,
    p1_action: Action,
) -> Action:
    """Select P1 iff predicted P1-minus-P0 utility is strictly positive."""

    if not isinstance(model, PairedDeltaRidgeModel):
        raise TatqaP19TypedEvaluatorError("E1 requires the frozen paired-delta model")
    _validate_action_pair(p0_action, p1_action)
    return p1_action if model.predict(p1_action.feature_vector) > 0.0 else p0_action


def select_evaluator_action(
    evaluator_id: str,
    *,
    p0_action: Action,
    p1_action: Action,
    model: PairedDeltaRidgeModel | None = None,
) -> Action:
    """Dispatch the exact E0/E1 item selector without threshold search."""

    if evaluator_id == "E0":
        return select_e0_action(p0_action, p1_action)
    if evaluator_id == "E1" and model is not None:
        return select_e1_action(model, p0_action, p1_action)
    raise TatqaP19TypedEvaluatorError("evaluator must be E0 or E1 with its frozen model")


def item_utility(top5: Sequence[str], canonical_gold_units: Sequence[str]) -> Fraction:
    """Return exact gold-unit recall@5 plus one exact complete-set bonus."""

    selected = _validated_top5(top5)
    if (
        isinstance(canonical_gold_units, (str, bytes))
        or not isinstance(canonical_gold_units, Sequence)
    ):
        raise TatqaP19TypedEvaluatorError("canonical gold units must be a sequence")
    gold = tuple(canonical_gold_units)
    if not 1 <= len(gold) <= TOP_K or len(set(gold)) != len(gold):
        raise TatqaP19TypedEvaluatorError(
            "canonical gold must contain one through five distinct units"
        )
    for unit_id in gold:
        _unit_identity(unit_id)
    hits = len(set(selected).intersection(gold))
    complete = hits == len(gold)
    return Fraction(hits, len(gold)) + int(complete)


@dataclass(frozen=True)
class ExactSignFlipResult:
    observed_net_u: Fraction
    nonzero_pair_count: int
    exact_p: Fraction
    promoted: bool

    def payload(self) -> dict[str, object]:
        return {
            "test": "one_sided_exact_magnitude_preserving_sign_flip_v1",
            "observed_net_U": {
                "numerator": self.observed_net_u.numerator,
                "denominator": self.observed_net_u.denominator,
            },
            "nonzero_pair_count": self.nonzero_pair_count,
            "p_value": {
                "numerator": self.exact_p.numerator,
                "denominator": self.exact_p.denominator,
            },
            "alpha": {
                "numerator": PROMOTION_ALPHA.numerator,
                "denominator": PROMOTION_ALPHA.denominator,
            },
            "positive_observed_net": self.observed_net_u > 0,
            "exact_p_at_or_below_alpha": self.exact_p <= PROMOTION_ALPHA,
            "promoted": self.promoted,
        }


def exact_magnitude_preserving_sign_flip(
    deltas: Sequence[Fraction | int],
) -> ExactSignFlipResult:
    """Compute the frozen exact one-sided paired test without Monte Carlo."""

    if isinstance(deltas, (str, bytes)) or not isinstance(deltas, Sequence) or not deltas:
        raise TatqaP19TypedEvaluatorError("paired utility delta vector is empty")
    normalized = tuple(_exact_delta(row) for row in deltas)
    common_denominator = 1
    for row in normalized:
        common_denominator = math.lcm(common_denominator, row.denominator)
    integer_deltas = tuple(
        row.numerator * (common_denominator // row.denominator) for row in normalized
    )
    observed = sum(integer_deltas)
    magnitudes = tuple(abs(row) for row in integer_deltas if row != 0)
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
    p_value = Fraction(
        sum(count for subtotal, count in distribution.items() if subtotal >= observed),
        1 << len(magnitudes),
    )
    net = sum(normalized, Fraction(0))
    return ExactSignFlipResult(
        observed_net_u=net,
        nonzero_pair_count=len(magnitudes),
        exact_p=p_value,
        promoted=net > 0 and p_value <= PROMOTION_ALPHA,
    )


def _canonical_hash(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TatqaP19TypedEvaluatorError("hash payload is not canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def canonical_action_hash(action: Action) -> str:
    if not isinstance(action, Action):
        raise TatqaP19TypedEvaluatorError("action hash requires an Action")
    return _canonical_hash(action.payload())


def canonical_behavior_hash(action: Action | Sequence[str]) -> str:
    """Hash only observable ordered-top5 behavior, independent of policy."""

    selected = (
        action.selected_unit_ids
        if isinstance(action, Action)
        else _validated_top5(action)
    )
    return _canonical_hash(
        {
            "schema": f"{VERSION}_ordered_top5_behavior_v1",
            "ordered_top5": list(selected),
        }
    )


__all__ = [
    "Action",
    "CanonicalUnit",
    "ExactSignFlipResult",
    "FEATURE_ORDER",
    "FORBIDDEN_FEATURES",
    "OPERATIONS",
    "P0_POLICY_ID",
    "P1_POLICY_ID",
    "PROMOTION_ALPHA",
    "PairedDeltaRidgeModel",
    "TYPED_EDGE_ORDER",
    "TOP_K",
    "TatqaP19TypedEvaluatorError",
    "TypedPlan",
    "build_action_pair",
    "build_p0_action",
    "build_p1_action",
    "canonical_action_hash",
    "canonical_behavior_hash",
    "exact_magnitude_preserving_sign_flip",
    "feature_vector",
    "fit_paired_delta_ridge",
    "item_utility",
    "p1_minus_p0_features",
    "rank_p0_units",
    "rank_p1_units",
    "select_e0_action",
    "select_e1_action",
    "select_evaluator_action",
    "totalize_typed_plan",
    "validate_typed_plan",
]

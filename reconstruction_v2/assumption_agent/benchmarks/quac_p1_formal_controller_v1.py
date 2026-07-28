"""Exact offline scientific controller for the frozen QuAC RJMC study.

This module contains the label-release-independent scientific core only.  It
cannot read QuAC, call a model, access a network, choose a cohort, or open the
late M_search view.  A production runtime supplies already sealed, label-free
graphs and baseline actions.  Labels are supplied in a separate call only
after the corresponding action archive is durable.

The controller fits RJMC exactly once on A_form, creates E0/E1/RAW/HippoRAG
Set5 actions without labels, scores integer two-role utility offline, and
computes an exact magnitude-preserving sign-flip distribution by dynamic
programming.  The sole M_search authorization is the preregistered A_hold
E1-minus-E0 promotion decision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Mapping, Sequence

from . import quac_rjmc_evaluator_v1 as evaluator


VERSION = "quac_p1_formal_controller_v1"
STUDY_ID = "QUAC_P1_RJMC_DIALOGUE_EVIDENCE_L5_V1"
EXECUTION_DESIGN_SELF_SHA256 = (
    "def417300b3c25f127517eef1cdd61760757762f08cc5a9b9877b261036dace2"
)
FAMILY_ORDER = ("FOLLOW", "MAYBE_FOLLOW", "DONT_FOLLOW")
MEASUREMENT_ARMS = ("E0", "E1", "RAW", "official_HippoRAG")
PROMOTION_ALPHA = Fraction(1, 10)
MAX_GRAPH_UNITS = 11
MAX_REPLACEMENT_CANDIDATES = 6
MAX_COMPLETE_STATES = 181
FORMAL_A_FORM_COUNT = 192
FORMAL_MEASUREMENT_COUNT = 96
FORMAL_FAMILY_COUNT = {
    "A_form": 64,
    "A_hold": 32,
    "M_search": 32,
}
_OPAQUE_ID = re.compile(r"[0-9a-f]{64}\Z")


class QuacP1FormalControllerError(RuntimeError):
    """A frozen scientific, action, label, or exact-statistic contract drifted."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1FormalControllerError(
            "controller value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _opaque_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID.fullmatch(value) is None:
        raise QuacP1FormalControllerError(
            f"{field} is not an opaque lowercase SHA-256 ID"
        )
    return value


def _set5(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QuacP1FormalControllerError(f"{field} is not a Set5 sequence")
    rows = tuple(_opaque_id(row, f"{field} unit") for row in value)
    if len(rows) != evaluator.TOP_K or len(set(rows)) != evaluator.TOP_K:
        raise QuacP1FormalControllerError(f"{field} is not five distinct units")
    return rows


def _qrel(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QuacP1FormalControllerError(f"{field} is not a qrel sequence")
    rows = tuple(sorted(_opaque_id(row, f"{field} unit") for row in value))
    if not rows or len(set(rows)) != len(rows):
        raise QuacP1FormalControllerError(f"{field} is empty or duplicated")
    return rows


def _block_corpus_registry(
    value: object,
    *,
    field: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QuacP1FormalControllerError(
            f"{field} is not a corpus unit-ID sequence"
        )
    rows = tuple(_opaque_id(row, f"{field} unit") for row in value)
    if (
        len(rows) < evaluator.TOP_K
        or rows != tuple(sorted(rows))
        or len(set(rows)) != len(rows)
    ):
        raise QuacP1FormalControllerError(
            f"{field} must be a canonical complete corpus registry"
        )
    return rows


def _require_subset_of_corpus(
    values: Sequence[str],
    corpus: frozenset[str],
    *,
    field: str,
) -> None:
    if not set(values).issubset(corpus):
        raise QuacP1FormalControllerError(
            f"{field} escaped the corresponding complete block corpus"
        )


@dataclass(frozen=True)
class LabelFreeGraphItem:
    """One label-free graph whose provenance is enforced by the action adapter."""

    item_id: str
    fold: int
    graph: evaluator.RelationalGraph
    raw_top5: tuple[str, ...]

    def __post_init__(self) -> None:
        _opaque_id(self.item_id, "item ID")
        if type(self.fold) is not int or not 0 <= self.fold < evaluator.COMPONENT_COUNT:
            raise QuacP1FormalControllerError("item fold is outside [0, 5)")
        if not isinstance(self.graph, evaluator.RelationalGraph):
            raise QuacP1FormalControllerError("item graph drifted")
        raw = self.graph.canonical_set(self.raw_top5)
        candidate_count = len(self.graph.units) - evaluator.TOP_K
        states = evaluator.enumerate_complete_states(
            self.graph, raw_top5=raw
        )
        if (
            len(self.graph.units) > MAX_GRAPH_UNITS
            or candidate_count > MAX_REPLACEMENT_CANDIDATES
            or len(states) > MAX_COMPLETE_STATES
            or len(states) != evaluator.complete_state_count(candidate_count)
        ):
            raise QuacP1FormalControllerError(
                "bounded complete action grammar drifted"
            )
        object.__setattr__(self, "raw_top5", raw)


@dataclass(frozen=True)
class LateLabelRow:
    """One late-opened two-role qrel row."""

    item_id: str
    family: str
    previous_qrel: tuple[str, ...]
    current_qrel: tuple[str, ...]

    def __post_init__(self) -> None:
        _opaque_id(self.item_id, "label item ID")
        if self.family not in FAMILY_ORDER:
            raise QuacP1FormalControllerError("label family drifted")
        object.__setattr__(
            self,
            "previous_qrel",
            _qrel(self.previous_qrel, "previous qrel"),
        )
        object.__setattr__(
            self,
            "current_qrel",
            _qrel(self.current_qrel, "current qrel"),
        )


@dataclass(frozen=True)
class ActionRow:
    """One sealed four-arm label-free action row."""

    item_id: str
    E0: tuple[str, ...]
    E1: tuple[str, ...]
    RAW: tuple[str, ...]
    official_HippoRAG: tuple[str, ...]

    def __post_init__(self) -> None:
        _opaque_id(self.item_id, "action item ID")
        for field in MEASUREMENT_ARMS:
            object.__setattr__(
                self,
                field,
                _set5(getattr(self, field), f"{field} action"),
            )

    def arm(self, name: str) -> tuple[str, ...]:
        if name not in MEASUREMENT_ARMS:
            raise QuacP1FormalControllerError("unknown action arm")
        return getattr(self, name)

    def payload(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            **{arm: list(self.arm(arm)) for arm in MEASUREMENT_ARMS},
        }


@dataclass(frozen=True)
class SealedStageActions:
    block: str
    corpus_unit_ids_sha256: str
    rows: tuple[ActionRow, ...]

    def __post_init__(self) -> None:
        if self.block not in {"A_hold", "M_search"}:
            raise QuacP1FormalControllerError("action block drifted")
        _opaque_id(
            self.corpus_unit_ids_sha256,
            "action corpus registry commitment",
        )
        rows = tuple(self.rows)
        if (
            len(rows) != FORMAL_MEASUREMENT_COUNT
            or len({row.item_id for row in rows}) != len(rows)
        ):
            raise QuacP1FormalControllerError(
                "action rows are not the exact formal registry"
            )
        if tuple(sorted(rows, key=lambda row: row.item_id)) != rows:
            raise QuacP1FormalControllerError("action rows are not canonical")
        object.__setattr__(self, "rows", rows)

    @property
    def action_sha256(self) -> str:
        return stable_hash(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_sealed_stage_actions_v1",
            "block": self.block,
            "arms": list(MEASUREMENT_ARMS),
            "corpus_unit_ids_sha256": self.corpus_unit_ids_sha256,
            "rows": [row.payload() for row in self.rows],
            "label_or_family_present": False,
        }


@dataclass(frozen=True)
class ExactSignFlipResult:
    observed_sum: int
    nonzero_count: int
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if (
            type(self.observed_sum) is not int
            or type(self.nonzero_count) is not int
            or self.nonzero_count < 0
            or type(self.numerator) is not int
            or type(self.denominator) is not int
            or self.denominator != 1 << self.nonzero_count
            or not 0 <= self.numerator <= self.denominator
        ):
            raise QuacP1FormalControllerError("exact sign-flip result drifted")

    @property
    def p(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def payload(self) -> dict[str, object]:
        return {
            "method": "one_sided_exact_magnitude_preserving_sign_flip_DP",
            "observed_sum": self.observed_sum,
            "nonzero_count": self.nonzero_count,
            "p_numerator": self.numerator,
            "p_denominator": self.denominator,
        }


def exact_magnitude_preserving_sign_flip(
    deltas: Sequence[int],
) -> ExactSignFlipResult:
    """Compute the exact upper-tail sign-flip test without enumeration."""

    if isinstance(deltas, (str, bytes)) or not isinstance(deltas, Sequence):
        raise QuacP1FormalControllerError("paired deltas are not a sequence")
    rows = tuple(deltas)
    if not rows or any(type(value) is not int or not -4 <= value <= 4 for value in rows):
        raise QuacP1FormalControllerError("paired deltas must be integer [-4, 4]")
    observed = sum(rows)
    magnitudes = tuple(abs(value) for value in rows if value != 0)
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        next_distribution: Counter[int] = Counter()
        for partial, count in distribution.items():
            next_distribution[partial + magnitude] += count
            next_distribution[partial - magnitude] += count
        distribution = next_distribution
    denominator = 1 << len(magnitudes)
    numerator = sum(
        count for signed_sum, count in distribution.items()
        if signed_sum >= observed
    )
    if sum(distribution.values()) != denominator:
        raise QuacP1FormalControllerError("exact sign-flip mass drifted")
    return ExactSignFlipResult(
        observed_sum=observed,
        nonzero_count=len(magnitudes),
        numerator=numerator,
        denominator=denominator,
    )


def two_role_utility(
    selected: Sequence[str],
    *,
    previous_qrel: Sequence[str],
    current_qrel: Sequence[str],
) -> int:
    selected_set = set(_set5(selected, "selected evidence"))
    previous = set(_qrel(previous_qrel, "previous qrel"))
    current = set(_qrel(current_qrel, "current qrel"))
    previous_hit = bool(selected_set.intersection(previous))
    current_hit = bool(selected_set.intersection(current))
    return int(previous_hit) + int(current_hit) + 2 * int(
        previous_hit and current_hit
    )


def fit_a_form_once(
    items: Sequence[LabelFreeGraphItem],
    labels: Sequence[LateLabelRow],
    *,
    block_corpus_unit_ids: Sequence[str],
) -> evaluator.JackknifeMinimaxComparator:
    """Open A_form labels after graph sealing and perform the one frozen fit."""

    item_rows = tuple(sorted(items, key=lambda row: row.item_id))
    label_tuple = tuple(labels)
    label_rows = {row.item_id: row for row in label_tuple}
    corpus_rows = _block_corpus_registry(
        block_corpus_unit_ids,
        field="A_form corpus",
    )
    corpus = frozenset(corpus_rows)
    if (
        len(item_rows) != FORMAL_A_FORM_COUNT
        or len({row.item_id for row in item_rows}) != FORMAL_A_FORM_COUNT
        or len(label_tuple) != FORMAL_A_FORM_COUNT
        or len(label_rows) != len(label_tuple)
        or set(label_rows) != {row.item_id for row in item_rows}
    ):
        raise QuacP1FormalControllerError(
            "A_form action and label registries do not match exactly"
        )
    family_counts = Counter(row.family for row in label_tuple)
    if any(
        family_counts[family] != FORMAL_FAMILY_COUNT["A_form"]
        for family in FAMILY_ORDER
    ):
        raise QuacP1FormalControllerError(
            "A_form family quota registry drifted"
        )
    fold_counts = Counter(row.fold for row in item_rows)
    if tuple(fold_counts[index] for index in range(5)) != (39, 39, 38, 38, 38):
        raise QuacP1FormalControllerError("A_form fold balance drifted")
    training: list[evaluator.ListwiseTrainingItem] = []
    for item in item_rows:
        label = label_rows[item.item_id]
        _require_subset_of_corpus(
            item.graph.unit_ids,
            corpus,
            field="A_form Agent graph",
        )
        _require_subset_of_corpus(
            (*label.previous_qrel, *label.current_qrel),
            corpus,
            field="A_form qrel",
        )
        states = evaluator.enumerate_complete_states(
            item.graph, raw_top5=item.raw_top5
        )
        utility = tuple(
            two_role_utility(
                state.unit_ids,
                previous_qrel=label.previous_qrel,
                current_qrel=label.current_qrel,
            )
            for state in states
        )
        training.append(
            evaluator.ListwiseTrainingItem(
                item_id=item.item_id,
                component=item.fold,
                graph=item.graph,
                raw_top5=item.raw_top5,
                utility=utility,
            )
        )
    return evaluator.fit_component_jackknife(
        tuple(training),
        config=evaluator.FitConfig(),
    )


def select_measurement_actions(
    *,
    block: str,
    items: Sequence[LabelFreeGraphItem],
    model: evaluator.JackknifeMinimaxComparator,
    hipporag_top5: Mapping[str, Sequence[str]],
    block_corpus_unit_ids: Sequence[str],
) -> SealedStageActions:
    """Create all four label-free actions before any measurement labels open."""

    if block not in {"A_hold", "M_search"}:
        raise QuacP1FormalControllerError("measurement block drifted")
    item_rows = tuple(sorted(items, key=lambda row: row.item_id))
    corpus_rows = _block_corpus_registry(
        block_corpus_unit_ids,
        field=f"{block} corpus",
    )
    corpus = frozenset(corpus_rows)
    if (
        len(item_rows) != FORMAL_MEASUREMENT_COUNT
        or len({row.item_id for row in item_rows})
        != FORMAL_MEASUREMENT_COUNT
        or set(hipporag_top5) != {row.item_id for row in item_rows}
        or not isinstance(model, evaluator.JackknifeMinimaxComparator)
    ):
        raise QuacP1FormalControllerError(
            "measurement item/model/HippoRAG registry drifted"
        )
    rows: list[ActionRow] = []
    for item in item_rows:
        _require_subset_of_corpus(
            item.graph.unit_ids,
            corpus,
            field=f"{block} Agent graph",
        )
        official = _set5(
            hipporag_top5[item.item_id],
            "official_HippoRAG action",
        )
        _require_subset_of_corpus(
            official,
            corpus,
            field=f"{block} official_HippoRAG action",
        )
        states = evaluator.enumerate_complete_states(
            item.graph, raw_top5=item.raw_top5
        )
        e0_index = evaluator.select_e0_proof_coverage(
            item.graph, raw_top5=item.raw_top5
        )
        e1_index, _scores = model.select(
            item.graph, raw_top5=item.raw_top5
        )
        rows.append(
            ActionRow(
                item_id=item.item_id,
                E0=states[e0_index].unit_ids,
                E1=states[e1_index].unit_ids,
                RAW=item.raw_top5,
                official_HippoRAG=official,
            )
        )
    return SealedStageActions(
        block=block,
        corpus_unit_ids_sha256=stable_hash(list(corpus_rows)),
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class PairedComparison:
    left_arm: str
    right_arm: str
    deltas: tuple[int, ...]
    family_nets: tuple[tuple[str, int], ...]
    exact: ExactSignFlipResult

    def __post_init__(self) -> None:
        if self.left_arm not in MEASUREMENT_ARMS or self.right_arm not in MEASUREMENT_ARMS:
            raise QuacP1FormalControllerError("paired arm drifted")
        if (
            not self.deltas
            or any(type(value) is not int or not -4 <= value <= 4 for value in self.deltas)
            or tuple(name for name, _value in self.family_nets) != FAMILY_ORDER
            or any(type(value) is not int for _name, value in self.family_nets)
            or self.exact != exact_magnitude_preserving_sign_flip(self.deltas)
        ):
            raise QuacP1FormalControllerError("paired comparison drifted")

    @property
    def net(self) -> int:
        return sum(self.deltas)

    @property
    def aggregate_positive_and_exact(self) -> bool:
        return self.net > 0 and self.exact.p <= PROMOTION_ALPHA

    @property
    def all_families_positive(self) -> bool:
        return all(value > 0 for _family, value in self.family_nets)

    def payload(self) -> dict[str, object]:
        return {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "net_utility": self.net,
            "gain_count": sum(value > 0 for value in self.deltas),
            "harm_count": sum(value < 0 for value in self.deltas),
            "tie_count": sum(value == 0 for value in self.deltas),
            "family_net_utility": dict(self.family_nets),
            "exact": self.exact.payload(),
        }


@dataclass(frozen=True)
class StageScore:
    block: str
    item_count: int
    corpus_unit_ids_sha256: str
    arm_totals: tuple[tuple[str, int], ...]
    family_arm_totals: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    comparisons: tuple[PairedComparison, ...]
    complete_counts: tuple[tuple[str, int], ...]
    private_item_score_sha256: str

    def __post_init__(self) -> None:
        if (
            self.block not in {"A_hold", "M_search"}
            or self.item_count != FORMAL_MEASUREMENT_COUNT
        ):
            raise QuacP1FormalControllerError("stage score block/count drifted")
        _opaque_id(
            self.corpus_unit_ids_sha256,
            "stage corpus registry commitment",
        )
        if tuple(name for name, _value in self.arm_totals) != MEASUREMENT_ARMS:
            raise QuacP1FormalControllerError("stage arm totals drifted")
        if tuple(name for name, _value in self.complete_counts) != MEASUREMENT_ARMS:
            raise QuacP1FormalControllerError("stage complete counts drifted")
        if tuple(name for name, _value in self.family_arm_totals) != FAMILY_ORDER:
            raise QuacP1FormalControllerError("stage family totals drifted")

    def comparison(self, right_arm: str) -> PairedComparison:
        for row in self.comparisons:
            if row.left_arm == "E1" and row.right_arm == right_arm:
                return row
        raise QuacP1FormalControllerError("requested comparison is absent")

    @property
    def promotion(self) -> bool:
        return self.comparison("E0").aggregate_positive_and_exact

    @property
    def reality_primary(self) -> bool:
        raw = self.comparison("RAW")
        hippo = self.comparison("official_HippoRAG")
        return bool(
            raw.aggregate_positive_and_exact
            and hippo.aggregate_positive_and_exact
            and raw.all_families_positive
            and hippo.all_families_positive
        )

    @property
    def l5(self) -> bool:
        return self.comparison("E0").aggregate_positive_and_exact

    def safe_payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_safe_stage_score_v1",
            "block": self.block,
            "item_count": self.item_count,
            "corpus_unit_ids_sha256": self.corpus_unit_ids_sha256,
            "arm_total_utility": dict(self.arm_totals),
            "family_arm_total_utility": {
                family: dict(rows) for family, rows in self.family_arm_totals
            },
            "complete_role_pair_count": dict(self.complete_counts),
            "comparisons": [row.payload() for row in self.comparisons],
            "promotion": self.promotion if self.block == "A_hold" else None,
            "reality_primary": (
                self.reality_primary if self.block == "A_hold" else None
            ),
            "L5": self.l5 if self.block == "M_search" else None,
        }


def score_sealed_stage(
    actions: SealedStageActions,
    labels: Sequence[LateLabelRow],
    *,
    block_corpus_unit_ids: Sequence[str],
) -> StageScore:
    """Open late labels and score a previously sealed complete action barrier."""

    action_rows = tuple(actions.rows)
    label_tuple = tuple(labels)
    label_rows = {row.item_id: row for row in label_tuple}
    corpus_rows = _block_corpus_registry(
        block_corpus_unit_ids,
        field=f"{actions.block} corpus",
    )
    corpus_commitment = stable_hash(list(corpus_rows))
    corpus = frozenset(corpus_rows)
    if (
        actions.corpus_unit_ids_sha256 != corpus_commitment
        or len(action_rows) != FORMAL_MEASUREMENT_COUNT
        or len(label_tuple) != FORMAL_MEASUREMENT_COUNT
        or len(label_rows) != len(label_tuple)
        or set(label_rows) != {row.item_id for row in action_rows}
    ):
        raise QuacP1FormalControllerError(
            "sealed actions and late labels do not match exactly"
        )
    family_counts = Counter(row.family for row in label_tuple)
    if any(
        family_counts[family] != FORMAL_FAMILY_COUNT[actions.block]
        for family in FAMILY_ORDER
    ):
        raise QuacP1FormalControllerError(
            "measurement family quota registry drifted"
        )
    utility_rows: list[dict[str, int]] = []
    family_rows: list[str] = []
    for action in action_rows:
        label = label_rows[action.item_id]
        for arm in MEASUREMENT_ARMS:
            _require_subset_of_corpus(
                action.arm(arm),
                corpus,
                field=f"{actions.block} {arm} action",
            )
        _require_subset_of_corpus(
            (*label.previous_qrel, *label.current_qrel),
            corpus,
            field=f"{actions.block} qrel",
        )
        family_rows.append(label.family)
        utility_rows.append(
            {
                arm: two_role_utility(
                    action.arm(arm),
                    previous_qrel=label.previous_qrel,
                    current_qrel=label.current_qrel,
                )
                for arm in MEASUREMENT_ARMS
            }
        )
    arm_totals = tuple(
        (arm, sum(row[arm] for row in utility_rows))
        for arm in MEASUREMENT_ARMS
    )
    family_arm_totals = tuple(
        (
            family,
            tuple(
                (
                    arm,
                    sum(
                        utility[arm]
                        for observed_family, utility in zip(
                            family_rows, utility_rows
                        )
                        if observed_family == family
                    ),
                )
                for arm in MEASUREMENT_ARMS
            ),
        )
        for family in FAMILY_ORDER
    )
    complete_counts = tuple(
        (arm, sum(row[arm] == 4 for row in utility_rows))
        for arm in MEASUREMENT_ARMS
    )
    comparisons: list[PairedComparison] = []
    for right_arm in ("E0", "RAW", "official_HippoRAG"):
        deltas = tuple(
            row["E1"] - row[right_arm] for row in utility_rows
        )
        family_nets = tuple(
            (
                family,
                sum(
                    delta
                    for observed_family, delta in zip(family_rows, deltas)
                    if observed_family == family
                ),
            )
            for family in FAMILY_ORDER
        )
        comparisons.append(
            PairedComparison(
                left_arm="E1",
                right_arm=right_arm,
                deltas=deltas,
                family_nets=family_nets,
                exact=exact_magnitude_preserving_sign_flip(deltas),
            )
        )
    private_score_rows = [
        {
            "item_id": action.item_id,
            "family": family,
            "utility": utility,
        }
        for action, family, utility in zip(
            action_rows, family_rows, utility_rows
        )
    ]
    return StageScore(
        block=actions.block,
        item_count=len(action_rows),
        corpus_unit_ids_sha256=corpus_commitment,
        arm_totals=arm_totals,
        family_arm_totals=family_arm_totals,
        comparisons=tuple(comparisons),
        complete_counts=complete_counts,
        private_item_score_sha256=stable_hash(private_score_rows),
    )


def safe_terminal(
    *,
    a_hold: StageScore,
    m_search: StageScore | None,
    model_parameter_sha256: str,
    action_commitments: Mapping[str, str],
    runtime_commitments: Mapping[str, str],
    M_materialization_count_before_promotion: int,
) -> dict[str, object]:
    """Build the aggregate-only valid terminal after lifecycle checks."""

    if (
        a_hold.block != "A_hold"
        or type(M_materialization_count_before_promotion) is not int
        or M_materialization_count_before_promotion != 0
    ):
        raise QuacP1FormalControllerError("valid terminal inputs drifted")
    model_parameter_sha256 = _opaque_id(
        model_parameter_sha256,
        "model parameter commitment",
    )
    promoted = a_hold.promotion
    if (m_search is None) == promoted:
        raise QuacP1FormalControllerError(
            "M_search presence does not equal promotion authorization"
        )
    if m_search is not None and m_search.block != "M_search":
        raise QuacP1FormalControllerError("M_search score block drifted")
    expected_action_keys = {
        "A_form_label_free_actions",
        "A_hold_four_arm_actions",
    }
    expected_runtime_keys = {
        "A_form_runtime",
        "A_hold_runtime",
    }
    if promoted:
        expected_action_keys.add("M_search_four_arm_actions")
        expected_runtime_keys.add("M_search_runtime")
    if (
        not isinstance(action_commitments, Mapping)
        or set(action_commitments) != expected_action_keys
    ):
        raise QuacP1FormalControllerError(
            "action commitment registry drifted"
        )
    if (
        not isinstance(runtime_commitments, Mapping)
        or set(runtime_commitments) != expected_runtime_keys
    ):
        raise QuacP1FormalControllerError(
            "runtime commitment registry drifted"
        )
    safe_action_commitments = {
        key: _opaque_id(
            action_commitments[key],
            f"{key} commitment",
        )
        for key in sorted(expected_action_keys)
    }
    safe_runtime_commitments = {
        key: _opaque_id(
            runtime_commitments[key],
            f"{key} commitment",
        )
        for key in sorted(expected_runtime_keys)
    }
    l5 = bool(m_search is not None and m_search.l5)
    primary = a_hold.reality_primary
    body = {
        "schema": f"{VERSION}_safe_terminal_v1",
        "study_id": STUDY_ID,
        "status": (
            "VALID_COMPLETE_PROMOTED_M_MEASURED"
            if promoted
            else "VALID_NONPROMOTION_M_UNOPENED"
        ),
        "execution_design_self_sha256": EXECUTION_DESIGN_SELF_SHA256,
        "A_hold_primary": primary,
        "A_hold_promotion": promoted,
        "M_search_opened": promoted,
        "M_search_L5": l5 if promoted else None,
        "total_goal_success": bool(primary and promoted and l5),
        "A_hold": a_hold.safe_payload(),
        "M_search": None if m_search is None else m_search.safe_payload(),
        "model_parameter_sha256": model_parameter_sha256,
        "action_commitments": safe_action_commitments,
        "runtime_commitments": safe_runtime_commitments,
        "M_materialization_count_before_promotion": 0,
        "online_or_API_evaluation_count": 0,
        "retry_replay_resample_repair_count": 0,
        "private_item_query_document_qrel_action_or_score_values_present": False,
    }
    return {**body, "terminal_self_sha256": stable_hash(body)}


__all__ = [
    "ActionRow",
    "EXECUTION_DESIGN_SELF_SHA256",
    "ExactSignFlipResult",
    "FAMILY_ORDER",
    "LabelFreeGraphItem",
    "MAX_COMPLETE_STATES",
    "MAX_GRAPH_UNITS",
    "MAX_REPLACEMENT_CANDIDATES",
    "MEASUREMENT_ARMS",
    "PROMOTION_ALPHA",
    "PairedComparison",
    "QuacP1FormalControllerError",
    "STUDY_ID",
    "SealedStageActions",
    "StageScore",
    "VERSION",
    "canonical_bytes",
    "exact_magnitude_preserving_sign_flip",
    "fit_a_form_once",
    "safe_terminal",
    "score_sealed_stage",
    "select_measurement_actions",
    "stable_hash",
    "two_role_utility",
]

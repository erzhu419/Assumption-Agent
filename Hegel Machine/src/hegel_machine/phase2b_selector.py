"""Interval-aware structural selector for the Phase-2B recognizer image.

Unlike the Phase-2A unique-projection API, this selector can return a
preregistered admissible scale set.  It consumes only adapter-produced,
completed candidate evaluations; it has no answer-key or fixture dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Final

from .hashing import stable_hash
from .schema import LawKind, require_tuple

if TYPE_CHECKING:
    from .phase2b_adapter import Phase2BAdapterRegistry
    from .phase2b_wire import PublicEvidenceBundle


class CandidateIntervalStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class TypedSelectionDisposition(str, Enum):
    UNIQUE_IDENTIFICATION = "unique_identification"
    ADMISSIBLE_SCALE_SET = "admissible_scale_set"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class ClosedInterval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        for value in (self.lower, self.upper):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError("interval endpoints must be finite numbers")
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    law_kind: LawKind
    role_binding: tuple[tuple[str, str], ...]
    scale_hypothesis_id: str
    residual: ClosedInterval | None
    tolerance: ClosedInterval | None
    completed: bool
    error_code: str | None = None
    footprint_id: str = ""

    def __post_init__(self) -> None:
        require_tuple(self.role_binding, "candidate role binding")
        if not self.candidate_id or not self.scale_hypothesis_id:
            raise ValueError("candidate identity and scale hypothesis are required")
        if self.role_binding != tuple(sorted(self.role_binding)):
            raise ValueError("candidate role binding must use canonical order")
        roles = [role for role, _ in self.role_binding]
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("candidate role binding is empty or repeats a role")
        if type(self.completed) is not bool:
            raise TypeError("candidate completed flag must be boolean")
        if self.completed and self.error_code is None:
            if self.residual is None or self.tolerance is None:
                raise ValueError("completed candidate needs residual and tolerance")
            if self.residual.lower < 0:
                raise ValueError("violation residual cannot be negative")
            if self.tolerance.lower <= 0:
                raise ValueError("tolerance interval must be strictly positive")
        elif any(value is not None for value in (self.residual, self.tolerance)):
            raise ValueError("incomplete/error candidate cannot carry score intervals")
        if self.error_code is not None and not self.error_code:
            raise ValueError("candidate error code cannot be empty")

    @property
    def normalized_interval(self) -> ClosedInterval | None:
        if not self.completed or self.error_code is not None:
            return None
        assert self.residual is not None
        assert self.tolerance is not None
        return ClosedInterval(
            self.residual.lower / self.tolerance.upper,
            self.residual.upper / self.tolerance.lower,
        )

    @property
    def status(self) -> CandidateIntervalStatus:
        interval = self.normalized_interval
        if interval is None:
            return CandidateIntervalStatus.ERROR
        if interval.upper <= 1.0:
            return CandidateIntervalStatus.PASS
        if interval.lower > 1.0:
            return CandidateIntervalStatus.FAIL
        return CandidateIntervalStatus.INCONCLUSIVE

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase2b_candidate_evaluation_")


@dataclass(frozen=True, slots=True)
class Phase2BSelectionPolicy:
    minimum_structural_margin_lcb: float = 1.0
    maximum_candidate_count: int = 50_000
    require_complete_family_coverage: bool = True
    require_binding_competitor: bool = True
    require_scale_competitor: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_structural_margin_lcb, bool)
            or not isinstance(self.minimum_structural_margin_lcb, (int, float))
            or not isfinite(self.minimum_structural_margin_lcb)
            or self.minimum_structural_margin_lcb < 0
        ):
            raise ValueError("selector margin must be finite and nonnegative")
        if (
            isinstance(self.maximum_candidate_count, bool)
            or not isinstance(self.maximum_candidate_count, int)
            or self.maximum_candidate_count <= 0
        ):
            raise ValueError("candidate budget must be a positive integer")
        for value in (
            self.require_complete_family_coverage,
            self.require_binding_competitor,
            self.require_scale_competitor,
        ):
            if type(value) is not bool:
                raise TypeError("selector policy flags must be booleans")

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2b_selector_policy_")


DEFAULT_PHASE2B_SELECTION_POLICY: Final = Phase2BSelectionPolicy()


@dataclass(frozen=True, slots=True)
class CandidateGridCell:
    candidate_id: str
    law_kind: LawKind
    role_binding: tuple[tuple[str, str], ...]
    scale_hypothesis_id: str
    footprint_id: str

    def __post_init__(self) -> None:
        require_tuple(self.role_binding, "candidate-grid role binding")
        if not all((self.candidate_id, self.scale_hypothesis_id, self.footprint_id)):
            raise ValueError("candidate-grid cell identity is incomplete")
        if self.role_binding != tuple(sorted(self.role_binding)):
            raise ValueError("candidate-grid role binding must use canonical order")


@dataclass(frozen=True, slots=True)
class CandidateGridCommitment:
    """Commit the selector to the complete adapter enumeration before scoring."""

    adapter_result_id: str
    bundle_content_id: str
    registry_id: str
    expected_cells: tuple[CandidateGridCell, ...]

    def __post_init__(self) -> None:
        require_tuple(self.expected_cells, "expected candidate-grid cells")
        if not all((self.adapter_result_id, self.bundle_content_id, self.registry_id)):
            raise ValueError("candidate-grid commitment identity is incomplete")
        if not self.expected_cells:
            raise ValueError("candidate-grid commitment cannot be empty")
        if any(not isinstance(item, CandidateGridCell) for item in self.expected_cells):
            raise TypeError("candidate-grid commitment contains a non-cell")
        if self.expected_cells != tuple(
            sorted(self.expected_cells, key=lambda item: item.candidate_id)
        ):
            raise ValueError("expected candidate-grid cells must use canonical order")
        if len(set(self.expected_candidate_ids)) != len(self.expected_cells):
            raise ValueError("candidate-grid commitment repeats a candidate")
        if {item.law_kind for item in self.expected_cells} != set(LawKind):
            raise ValueError("candidate-grid commitment must cover every law family")
        global_scales = {
            item.scale_hypothesis_id for item in self.expected_cells
        }
        groups: dict[
            tuple[LawKind, tuple[tuple[str, str], ...]], set[str]
        ] = {}
        for item in self.expected_cells:
            groups.setdefault((item.law_kind, item.role_binding), set()).add(
                item.scale_hypothesis_id
            )
        if any(scales != global_scales for scales in groups.values()):
            raise ValueError("candidate-grid groups do not share the complete scale set")

    @property
    def expected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.expected_cells)

    @property
    def commitment_id(self) -> str:
        return stable_hash(self, prefix="phase2b_candidate_grid_")


@dataclass(frozen=True, slots=True)
class TypedSelectorDecision:
    disposition: TypedSelectionDisposition
    reason: str
    policy_id: str
    evaluated_candidate_ids: tuple[str, ...]
    candidate_grid_commitment_id: str | None = None
    selected_law_kind: LawKind | None = None
    selected_role_binding: tuple[tuple[str, str], ...] = ()
    admissible_scale_hypothesis_ids: tuple[str, ...] = ()
    normalized_structural_margin_lcb: float | None = None

    def __post_init__(self) -> None:
        require_tuple(self.evaluated_candidate_ids, "evaluated candidate ids")
        require_tuple(self.selected_role_binding, "selected role binding")
        require_tuple(
            self.admissible_scale_hypothesis_ids,
            "admissible scale hypotheses",
        )
        if not self.reason or not self.policy_id:
            raise ValueError("selector decision needs a reason and policy")
        if self.evaluated_candidate_ids != tuple(
            sorted(self.evaluated_candidate_ids)
        ):
            raise ValueError("evaluated candidate ids must use canonical order")
        if self.disposition is TypedSelectionDisposition.ABSTAIN:
            if any(
                (
                    self.selected_law_kind is not None,
                    bool(self.selected_role_binding),
                    bool(self.admissible_scale_hypothesis_ids),
                    self.normalized_structural_margin_lcb is not None,
                )
            ):
                raise ValueError("abstention cannot carry a selected answer")
        else:
            if (
                self.candidate_grid_commitment_id is None
                or
                self.selected_law_kind is None
                or not self.selected_role_binding
                or not self.admissible_scale_hypothesis_ids
                or self.normalized_structural_margin_lcb is None
            ):
                raise ValueError("identified decision is incomplete")
            expected = (
                TypedSelectionDisposition.UNIQUE_IDENTIFICATION
                if len(self.admissible_scale_hypothesis_ids) == 1
                else TypedSelectionDisposition.ADMISSIBLE_SCALE_SET
            )
            if self.disposition is not expected:
                raise ValueError("selector disposition disagrees with scale-set size")

    @property
    def decision_id(self) -> str:
        return stable_hash(self, prefix="phase2b_selector_decision_")


def _abstain(
    reason: str,
    policy: Phase2BSelectionPolicy,
    candidate_ids: tuple[str, ...],
    grid_commitment: CandidateGridCommitment | None = None,
) -> TypedSelectorDecision:
    return TypedSelectorDecision(
        disposition=TypedSelectionDisposition.ABSTAIN,
        reason=reason,
        policy_id=policy.policy_id,
        evaluated_candidate_ids=candidate_ids,
        candidate_grid_commitment_id=(
            None if grid_commitment is None else grid_commitment.commitment_id
        ),
    )


def _select_against_grid_commitment(
    evaluations: tuple[CandidateEvaluation, ...],
    *,
    grid_commitment: CandidateGridCommitment | None = None,
    policy: Phase2BSelectionPolicy = DEFAULT_PHASE2B_SELECTION_POLICY,
) -> TypedSelectorDecision:
    """Private interval core over an adapter-issued grid commitment."""

    require_tuple(evaluations, "Phase-2B candidate evaluations")
    canonical = tuple(sorted(evaluations, key=lambda item: item.content_id))
    candidate_ids = tuple(sorted(item.candidate_id for item in canonical))
    if not canonical:
        return _abstain("empty_candidate_set", policy, candidate_ids, grid_commitment)
    if len(set(candidate_ids)) != len(candidate_ids):
        return _abstain(
            "duplicate_candidate_id", policy, candidate_ids, grid_commitment
        )
    if grid_commitment is None:
        return _abstain("missing_candidate_grid_commitment", policy, candidate_ids)
    if candidate_ids != grid_commitment.expected_candidate_ids:
        return _abstain(
            "incomplete_or_drifted_candidate_grid",
            policy,
            candidate_ids,
            grid_commitment,
        )
    expected_by_id = {
        item.candidate_id: item for item in grid_commitment.expected_cells
    }
    if any(
        (
            item.law_kind,
            item.role_binding,
            item.scale_hypothesis_id,
            item.footprint_id,
        )
        != (
            expected_by_id[item.candidate_id].law_kind,
            expected_by_id[item.candidate_id].role_binding,
            expected_by_id[item.candidate_id].scale_hypothesis_id,
            expected_by_id[item.candidate_id].footprint_id,
        )
        for item in canonical
    ):
        return _abstain(
            "candidate_metadata_drift",
            policy,
            candidate_ids,
            grid_commitment,
        )
    if len(canonical) > policy.maximum_candidate_count:
        return _abstain(
            "candidate_budget_exceeded", policy, candidate_ids, grid_commitment
        )
    if any(item.status is CandidateIntervalStatus.ERROR for item in canonical):
        return _abstain(
            "candidate_evaluation_error", policy, candidate_ids, grid_commitment
        )
    if policy.require_complete_family_coverage and {
        item.law_kind for item in canonical
    } != set(LawKind):
        return _abstain(
            "incomplete_family_coverage", policy, candidate_ids, grid_commitment
        )

    groups: dict[
        tuple[LawKind, tuple[tuple[str, str], ...]],
        list[CandidateEvaluation],
    ] = {}
    for item in canonical:
        groups.setdefault((item.law_kind, item.role_binding), []).append(item)

    passing_groups = tuple(
        key
        for key, items in groups.items()
        if any(item.status is CandidateIntervalStatus.PASS for item in items)
    )
    if not passing_groups:
        reason = (
            "nonidentifiable_interval_overlap"
            if any(
                item.status is CandidateIntervalStatus.INCONCLUSIVE
                for item in canonical
            )
            else "no_passing_structure"
        )
        return _abstain(reason, policy, candidate_ids, grid_commitment)
    if len(passing_groups) != 1:
        return _abstain(
            "multiple_passing_structures", policy, candidate_ids, grid_commitment
        )

    selected_key = passing_groups[0]
    selected_items = tuple(groups[selected_key])
    if any(
        item.status is CandidateIntervalStatus.INCONCLUSIVE for item in selected_items
    ):
        return _abstain(
            "selected_structure_has_inconclusive_scale",
            policy,
            candidate_ids,
            grid_commitment,
        )
    passing_scales = tuple(
        sorted(
            {
                item.scale_hypothesis_id
                for item in selected_items
                if item.status is CandidateIntervalStatus.PASS
            }
        )
    )
    if policy.require_scale_competitor and len(
        {item.scale_hypothesis_id for item in selected_items}
    ) < 2:
        return _abstain(
            "missing_scale_competitor", policy, candidate_ids, grid_commitment
        )
    if policy.require_binding_competitor and not any(
        item.law_kind is selected_key[0] and item.role_binding != selected_key[1]
        for item in canonical
    ):
        return _abstain(
            "missing_binding_competitor", policy, candidate_ids, grid_commitment
        )

    selected_upper = min(
        item.normalized_interval.upper
        for item in selected_items
        if item.status is CandidateIntervalStatus.PASS
        and item.normalized_interval is not None
    )
    structural_competitors = tuple(
        item for key, items in groups.items() if key != selected_key for item in items
    )
    if not structural_competitors:
        return _abstain(
            "missing_structural_competitor", policy, candidate_ids, grid_commitment
        )
    if any(
        item.status is CandidateIntervalStatus.INCONCLUSIVE
        for item in structural_competitors
    ):
        return _abstain(
            "inconclusive_structural_competitor",
            policy,
            candidate_ids,
            grid_commitment,
        )
    competitor_lower = min(
        item.normalized_interval.lower
        for item in structural_competitors
        if item.normalized_interval is not None
    )
    margin = competitor_lower - selected_upper
    if margin < policy.minimum_structural_margin_lcb:
        return _abstain(
            "insufficient_structural_margin_lcb",
            policy,
            candidate_ids,
            grid_commitment,
        )

    disposition = (
        TypedSelectionDisposition.UNIQUE_IDENTIFICATION
        if len(passing_scales) == 1
        else TypedSelectionDisposition.ADMISSIBLE_SCALE_SET
    )
    return TypedSelectorDecision(
        disposition=disposition,
        reason="unique_structure_with_preregistered_admissible_scales",
        policy_id=policy.policy_id,
        evaluated_candidate_ids=candidate_ids,
        candidate_grid_commitment_id=grid_commitment.commitment_id,
        selected_law_kind=selected_key[0],
        selected_role_binding=selected_key[1],
        admissible_scale_hypothesis_ids=passing_scales,
        normalized_structural_margin_lcb=margin,
    )


def select_typed_candidate_evaluations(
    evaluations: tuple[CandidateEvaluation, ...],
    *,
    evidence_bundle: "PublicEvidenceBundle",
    adapter_registry: "Phase2BAdapterRegistry",
    policy: Phase2BSelectionPolicy = DEFAULT_PHASE2B_SELECTION_POLICY,
) -> TypedSelectorDecision:
    """Re-enumerate the adapter grid, then select conservatively.

    The public API deliberately does not accept a caller-provided grid
    commitment.  Re-enumeration from the content-bound evidence bundle and
    frozen registry prevents a projection compiler from truncating the grid and
    signing its own self-consistent subset.
    """

    from .phase2b_adapter import AdapterDisposition, enumerate_candidate_hypotheses

    result = enumerate_candidate_hypotheses(evidence_bundle, adapter_registry)
    if result.disposition is AdapterDisposition.ABSTAIN:
        candidate_ids = tuple(sorted(item.candidate_id for item in evaluations))
        return _abstain(
            "adapter_" + result.reason,
            policy,
            candidate_ids,
        )
    return _select_against_grid_commitment(
        evaluations,
        grid_commitment=result.candidate_grid_commitment,
        policy=policy,
    )


__all__ = (
    "CandidateEvaluation",
    "CandidateGridCell",
    "CandidateGridCommitment",
    "CandidateIntervalStatus",
    "ClosedInterval",
    "DEFAULT_PHASE2B_SELECTION_POLICY",
    "Phase2BSelectionPolicy",
    "TypedSelectionDisposition",
    "TypedSelectorDecision",
    "select_typed_candidate_evaluations",
)

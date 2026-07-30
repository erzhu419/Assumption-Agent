"""Bounded robustification and idealization engines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence

from .schema import ReductionMap


class UncertaintyKind(str, Enum):
    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    ADVERSARIAL = "adversarial"
    MIXED = "mixed"


class RobustModelKind(str, Enum):
    POINT_MASS = "point_mass"
    BOUNDED_SET = "bounded_set"
    PARAMETRIC = "parametric_distribution"
    NONPARAMETRIC = "nonparametric_distribution"
    AMBIGUITY_SET = "ambiguity_set"
    DYNAMIC_ADVERSARY = "dynamic_adversary"


@dataclass(frozen=True, slots=True)
class RobustificationCandidate:
    candidate_id: str
    parent_version_id: str
    model_kind: RobustModelKind
    uncertainty_kind: UncertaintyKind
    scope: tuple[str, ...]
    stress_test_ids: tuple[str, ...]
    reduction_map: ReductionMap
    nominal_utility: float
    tail_utility: float
    safety_utility: float
    coverage: float
    complexity_cost: float

    def __post_init__(self) -> None:
        if not all(
            (
                self.candidate_id,
                self.parent_version_id,
                self.scope,
                self.stress_test_ids,
            )
        ):
            raise ValueError("robustification candidate is missing scope or identity")
        values = (
            self.nominal_utility,
            self.tail_utility,
            self.safety_utility,
            self.coverage,
            self.complexity_cost,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("robustification utilities must be finite")
        if not 0 <= self.coverage <= 1:
            raise ValueError("robustification coverage must be in [0, 1]")
        if self.complexity_cost < 0:
            raise ValueError("robustification complexity cost cannot be negative")

    @property
    def vector_utility(self) -> tuple[tuple[str, float], ...]:
        return (
            ("nominal", self.nominal_utility),
            ("tail", self.tail_utility),
            ("safety", self.safety_utility),
            ("coverage", self.coverage),
            ("complexity_cost", self.complexity_cost),
        )


@dataclass(frozen=True, slots=True)
class RobustificationDecision:
    accepted: bool
    candidate_id: str
    reason: str
    utility_vector: tuple[tuple[str, float], ...]


def evaluate_robustification(
    candidate: RobustificationCandidate,
    *,
    minimum_tail_gain: float,
    minimum_safety_gain: float,
    minimum_coverage: float,
    maximum_nominal_loss: float,
    parent_nominal_utility: float,
    parent_tail_utility: float,
    parent_safety_utility: float,
) -> RobustificationDecision:
    """Retain vector utilities; do not silently scalarize them."""

    policy_values = (
        minimum_tail_gain,
        minimum_safety_gain,
        minimum_coverage,
        maximum_nominal_loss,
        parent_nominal_utility,
        parent_tail_utility,
        parent_safety_utility,
    )
    if any(not isfinite(value) for value in policy_values):
        raise ValueError("robustification policy values must be finite")
    if (
        minimum_tail_gain < 0
        or minimum_safety_gain < 0
        or not 0 <= minimum_coverage <= 1
        or maximum_nominal_loss < 0
    ):
        raise ValueError("robustification thresholds are outside their valid range")
    checks = {
        "stress_tests": bool(candidate.stress_test_ids),
        "singleton_reduction": candidate.reduction_map.maximum_error >= 0,
        "tail_gain": candidate.tail_utility - parent_tail_utility
        >= minimum_tail_gain,
        "safety_gain": candidate.safety_utility - parent_safety_utility
        >= minimum_safety_gain,
        "coverage": candidate.coverage >= minimum_coverage,
        "nominal_loss": parent_nominal_utility - candidate.nominal_utility
        <= maximum_nominal_loss,
    }
    accepted = all(checks.values())
    failures = [name for name, passed in checks.items() if not passed]
    return RobustificationDecision(
        accepted=accepted,
        candidate_id=candidate.candidate_id,
        reason="accepted_vector_pareto_contract"
        if accepted
        else "failed:" + ",".join(failures),
        utility_vector=candidate.vector_utility,
    )


@dataclass(frozen=True, slots=True)
class IdealizationContract:
    contract_id: str
    parent_version_id: str
    candidate_id: str
    operation: str
    deleted_degrees_of_freedom: tuple[str, ...]
    preserved_observables: tuple[str, ...]
    scale_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    maximum_probe_discrepancy: float
    discrepancy_budget: float
    failure_boundary: tuple[str, ...]
    counterexample_ids: tuple[str, ...]
    compute_gain: float
    sample_gain: float
    complexity_gain: float
    full_model_recovery: str

    def __post_init__(self) -> None:
        if self.operation not in {"restriction", "quotient", "projection"}:
            raise ValueError("unknown idealization operation")
        if not self.deleted_degrees_of_freedom or not self.preserved_observables:
            raise ValueError("idealization must say what is deleted and preserved")
        if not self.failure_boundary or not self.counterexample_ids:
            raise ValueError("idealization needs boundary and counterexamples")
        values = (
            self.maximum_probe_discrepancy,
            self.discrepancy_budget,
            self.compute_gain,
            self.sample_gain,
            self.complexity_gain,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("idealization measurements must be finite")
        if self.maximum_probe_discrepancy < 0 or self.discrepancy_budget < 0:
            raise ValueError("idealization discrepancy values cannot be negative")


@dataclass(frozen=True, slots=True)
class IdealizationDecision:
    accepted: bool
    contract_id: str
    reason: str


def evaluate_idealization(
    contract: IdealizationContract,
    *,
    minimum_complexity_gain: float,
) -> IdealizationDecision:
    if not isfinite(minimum_complexity_gain) or minimum_complexity_gain < 0:
        raise ValueError("minimum complexity gain must be finite and nonnegative")
    checks = {
        "probe_preservation": contract.maximum_probe_discrepancy
        <= contract.discrepancy_budget,
        "complexity_gain": contract.complexity_gain >= minimum_complexity_gain,
        "resource_gain": max(contract.compute_gain, contract.sample_gain) > 0,
        "recoverable": bool(contract.full_model_recovery),
    }
    accepted = all(checks.values())
    failures = [name for name, passed in checks.items() if not passed]
    return IdealizationDecision(
        accepted,
        contract.contract_id,
        "accepted_probe_preserving_simplification"
        if accepted
        else "failed:" + ",".join(failures),
    )


def pareto_frontier(
    candidates: Sequence[RobustificationCandidate],
) -> tuple[RobustificationCandidate, ...]:
    """Return non-dominated candidates without inventing scalar weights."""

    frontier: list[RobustificationCandidate] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            no_worse = (
                other.nominal_utility >= candidate.nominal_utility
                and other.tail_utility >= candidate.tail_utility
                and other.safety_utility >= candidate.safety_utility
                and other.coverage >= candidate.coverage
                and other.complexity_cost <= candidate.complexity_cost
            )
            strictly_better = (
                other.nominal_utility > candidate.nominal_utility
                or other.tail_utility > candidate.tail_utility
                or other.safety_utility > candidate.safety_utility
                or other.coverage > candidate.coverage
                or other.complexity_cost < candidate.complexity_cost
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda item: item.candidate_id))

"""Probe-induced geometry and observational quotients."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping, Sequence

from .schema import ProbeSpec

Distribution = Mapping[str, float]
ProbeOutcomes = Mapping[str, Distribution]


def normalize_distribution(distribution: Distribution) -> dict[str, float]:
    if not distribution:
        raise ValueError("distribution cannot be empty")
    if any(value < 0 or not isfinite(value) for value in distribution.values()):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(sum(distribution.values()))
    if total <= 0:
        raise ValueError("distribution must carry positive mass")
    return {key: float(value) / total for key, value in distribution.items()}


def total_variation(left: Distribution, right: Distribution) -> float:
    left_norm = normalize_distribution(left)
    right_norm = normalize_distribution(right)
    support = set(left_norm) | set(right_norm)
    return 0.5 * sum(
        abs(left_norm.get(key, 0.0) - right_norm.get(key, 0.0))
        for key in support
    )


@dataclass(frozen=True, slots=True)
class TaskGeometry:
    task_id: str
    evaluator_epoch: str
    probe_weights: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.probe_weights, tuple):
            raise TypeError("probe weights must be an immutable tuple")
        if not self.probe_weights:
            raise ValueError("task geometry needs at least one probe")
        if any(weight <= 0 or not isfinite(weight) for _, weight in self.probe_weights):
            raise ValueError("probe weights must be positive")
        if len({probe_id for probe_id, _ in self.probe_weights}) != len(
            self.probe_weights
        ):
            raise ValueError("probe ids must be unique")

    def distance(self, left: ProbeOutcomes, right: ProbeOutcomes) -> float:
        weighted = 0.0
        weight_sum = 0.0
        for probe_id, weight in self.probe_weights:
            if probe_id not in left or probe_id not in right:
                raise ValueError(f"missing outcome for probe {probe_id}")
            discrepancy = total_variation(left[probe_id], right[probe_id])
            weighted += weight * discrepancy**2
            weight_sum += weight
        return sqrt(weighted / weight_sum)

    def components(
        self, left: ProbeOutcomes, right: ProbeOutcomes
    ) -> tuple[tuple[str, float], ...]:
        return tuple(
            (
                probe_id,
                total_variation(left[probe_id], right[probe_id]),
            )
            for probe_id, _ in self.probe_weights
        )


def observationally_equivalent(
    left: ProbeOutcomes,
    right: ProbeOutcomes,
    geometry: TaskGeometry,
    *,
    tolerance: float,
) -> bool:
    if tolerance < 0 or not isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    return geometry.distance(left, right) <= tolerance


def quotient_classes(
    hypotheses: Mapping[str, ProbeOutcomes],
    geometry: TaskGeometry,
    *,
    tolerance: float,
) -> tuple[tuple[str, ...], ...]:
    """Build deterministic complete-link tolerance classes.

    Every pair inside a returned class is within ``tolerance``. This avoids
    treating an A~B~C chain as identity when A and C are still distinguishable.
    These remain task-relative tolerance classes, not mathematical quotient
    equivalence classes unless the underlying relation is proven transitive.
    """

    if tolerance < 0 or not isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    groups: list[list[str]] = []
    for identifier in sorted(hypotheses):
        for group in groups:
            if all(
                observationally_equivalent(
                    hypotheses[identifier],
                    hypotheses[member],
                    geometry,
                    tolerance=tolerance,
                )
                for member in group
            ):
                group.append(identifier)
                break
        else:
            groups.append([identifier])
    return tuple(tuple(group) for group in groups)


def choose_discriminating_probe(
    probes: Sequence[ProbeSpec],
    predicted_outcomes: Mapping[str, Mapping[str, Distribution]],
    competing_hypothesis_ids: Sequence[str],
    *,
    geometry: TaskGeometry,
    data_cutoff: str,
) -> str:
    """Choose a frozen-task/epoch/cutoff maximin separating probe."""

    if len(competing_hypothesis_ids) < 2:
        raise ValueError("at least two competing hypotheses are required")
    best: tuple[float, str] | None = None
    for probe in probes:
        if probe.semantic_only:
            continue
        if (
            geometry.task_id not in probe.task_ids
            or probe.evaluator_epoch != geometry.evaluator_epoch
            or probe.data_cutoff != data_cutoff
            or probe.probe_id not in dict(geometry.probe_weights)
        ):
            continue
        pair_distances: list[float] = []
        for index, left_id in enumerate(competing_hypothesis_ids):
            for right_id in competing_hypothesis_ids[index + 1 :]:
                try:
                    left = predicted_outcomes[left_id][probe.probe_id]
                    right = predicted_outcomes[right_id][probe.probe_id]
                except KeyError as exc:
                    raise ValueError(f"missing predicted probe outcome: {exc}") from exc
                pair_distances.append(total_variation(left, right))
        score = min(pair_distances) / probe.cost
        candidate = (score, probe.probe_id)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("no non-semantic probe is available")
    return best[1]


@dataclass(frozen=True, slots=True)
class CandidateFit:
    binding_id: str
    scale_id: str
    role_binding_id: str
    structural_violation: float
    hard_negative_margin: float
    unseen_prediction_score: float
    scope_valid: bool
    complexity_cost: float

    def __post_init__(self) -> None:
        values = (
            self.structural_violation,
            self.hard_negative_margin,
            self.unseen_prediction_score,
            self.complexity_cost,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("candidate fit values must be finite")


@dataclass(frozen=True, slots=True)
class MembershipResult:
    accepted: bool
    score: float
    best_fit: CandidateFit | None
    reason: str


def hypothesis_membership(
    fits: Sequence[CandidateFit],
    *,
    maximum_violation: float,
    minimum_hard_negative_margin: float,
    minimum_unseen_prediction: float,
    complexity_weight: float,
) -> MembershipResult:
    """Compute task/scale/role-conditioned membership by an explicit infimum."""

    thresholds = (
        maximum_violation,
        minimum_hard_negative_margin,
        minimum_unseen_prediction,
        complexity_weight,
    )
    if any(not isfinite(value) for value in thresholds):
        raise ValueError("membership thresholds must be finite")
    if maximum_violation < 0 or complexity_weight < 0:
        raise ValueError("violation and complexity thresholds must be nonnegative")
    admissible = [
        fit
        for fit in fits
        if fit.scope_valid
        and fit.hard_negative_margin >= minimum_hard_negative_margin
        and fit.unseen_prediction_score >= minimum_unseen_prediction
    ]
    if not admissible:
        return MembershipResult(False, float("inf"), None, "no_admissible_binding")
    best = min(
        admissible,
        key=lambda fit: fit.structural_violation
        + complexity_weight * fit.complexity_cost,
    )
    score = best.structural_violation + complexity_weight * best.complexity_cost
    if best.structural_violation > maximum_violation:
        return MembershipResult(False, score, best, "violation_above_threshold")
    return MembershipResult(True, score, best, "accepted_by_structural_infimum")

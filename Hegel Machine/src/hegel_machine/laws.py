"""Executable residuals for the frozen Phase-2 law library.

Natural-language labels are deliberately absent from every verifier input.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Iterable, Mapping

from .schema import LawKind


class InsufficientObservables(ValueError):
    """Raised internally when a law cannot be evaluated without fabrication."""


@dataclass(frozen=True, slots=True)
class LawEvaluation:
    kind: LawKind
    residual: float | None
    tolerance: float
    passed: bool
    abstained: bool
    reason: str
    components: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple):
            raise TypeError("law residual components must be an immutable tuple")
        if self.tolerance < 0 or not isfinite(self.tolerance) or (
            self.residual is not None and not isfinite(self.residual)
        ):
            raise ValueError("law residual must be finite and tolerance nonnegative")
        if any(not isfinite(value) for _, value in self.components):
            raise ValueError("law residual components must be finite")
        if self.abstained and (self.residual is not None or self.passed):
            raise ValueError("an abstention cannot carry a residual or pass")
        if not self.abstained and self.residual is None:
            raise ValueError("a completed evaluation needs a residual")
        if (
            not self.abstained
            and self.passed is not (self.residual <= self.tolerance)
        ):
            raise ValueError("law pass flag disagrees with residual and tolerance")
        expected_reason = (
            None
            if self.abstained
            else "within_tolerance"
            if self.passed
            else "law_violation"
        )
        if expected_reason is not None and self.reason != expected_reason:
            raise ValueError("completed law evaluation has an inconsistent reason")


def _number(episode: Mapping[str, Any], key: str) -> float:
    if key not in episode:
        raise InsufficientObservables(f"missing observable: {key}")
    value = episode[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InsufficientObservables(f"observable {key} is not numeric")
    result = float(value)
    if not isfinite(result):
        raise InsufficientObservables(f"observable {key} is not finite")
    return result


def _numbers(episode: Mapping[str, Any], key: str) -> tuple[float, ...]:
    if key not in episode or isinstance(episode[key], (str, bytes)):
        raise InsufficientObservables(f"missing numeric sequence: {key}")
    try:
        result = tuple(float(item) for item in episode[key])
    except (TypeError, ValueError) as exc:
        raise InsufficientObservables(f"observable {key} is not numeric") from exc
    if any(not isfinite(item) for item in result):
        raise InsufficientObservables(f"observable {key} contains non-finite values")
    return result


def _flag(episode: Mapping[str, Any], key: str) -> bool:
    if key not in episode or not isinstance(episode[key], bool):
        raise InsufficientObservables(f"missing boolean observable: {key}")
    return episode[key]


def _normalizer(*values: float) -> float:
    return max(1.0, *(abs(value) for value in values))


def _complete(
    kind: LawKind,
    residual: float,
    tolerance: float,
    components: Mapping[str, float],
) -> LawEvaluation:
    return LawEvaluation(
        kind=kind,
        residual=residual,
        tolerance=tolerance,
        passed=residual <= tolerance,
        abstained=False,
        reason="within_tolerance" if residual <= tolerance else "law_violation",
        components=tuple(sorted(components.items())),
    )


def _abstain(kind: LawKind, tolerance: float, reason: str) -> LawEvaluation:
    return LawEvaluation(
        kind=kind,
        residual=None,
        tolerance=tolerance,
        passed=False,
        abstained=True,
        reason=reason,
        components=(),
    )


def evaluate_symmetry(
    episode: Mapping[str, Any], tolerance: float = 1e-9
) -> LawEvaluation:
    """Check an involution/equivariance pair in a declared common codomain."""

    kind = LawKind.SYMMETRY
    try:
        if not _flag(episode, "common_codomains"):
            return _abstain(kind, tolerance, "outputs lack a declared common codomain")
        forward = _numbers(episode, "forward")
        reverse = _numbers(episode, "transformed")
        if not forward or len(forward) != len(reverse):
            return _abstain(kind, tolerance, "paired outputs have incompatible shapes")
        component_residuals = [
            abs(left - right) / _normalizer(left, right)
            for left, right in zip(forward, reverse, strict=True)
        ]
        residual = max(component_residuals)
        return _complete(
            kind,
            residual,
            tolerance,
            {
                "maximum_pair_residual": residual,
                "mean_pair_residual": sum(component_residuals)
                / len(component_residuals),
            },
        )
    except InsufficientObservables as exc:
        return _abstain(kind, tolerance, str(exc))


def evaluate_monotonicity(
    episode: Mapping[str, Any], tolerance: float = 1e-9
) -> LawEvaluation:
    """Check an explicitly oriented order relation."""

    kind = LawKind.MONOTONICITY
    try:
        x_low = _number(episode, "x_low")
        x_high = _number(episode, "x_high")
        y_low = _number(episode, "y_low")
        y_high = _number(episode, "y_high")
        direction = _number(episode, "direction")
        if x_low >= x_high:
            return _abstain(kind, tolerance, "input order is not strict")
        if direction not in {-1.0, 1.0}:
            return _abstain(kind, tolerance, "direction must be +1 or -1")
        signed_change = direction * (y_high - y_low)
        residual = max(0.0, -signed_change) / _normalizer(y_low, y_high)
        return _complete(
            kind,
            residual,
            tolerance,
            {"signed_change": signed_change, "order_violation": residual},
        )
    except InsufficientObservables as exc:
        return _abstain(kind, tolerance, str(exc))


def evaluate_conservation(
    episode: Mapping[str, Any], tolerance: float = 1e-9
) -> LawEvaluation:
    """Check a balance law with explicit boundary flows and time window."""

    kind = LawKind.CONSERVATION
    try:
        if not _flag(episode, "boundary_observed"):
            return _abstain(kind, tolerance, "unobserved boundary flow")
        storage = _number(episode, "storage_delta")
        inflows = _numbers(episode, "inflows")
        outflows = _numbers(episode, "outflows")
        sources = _numbers(episode, "sources")
        sinks = _numbers(episode, "sinks")
        raw_balance = (
            storage
            + sum(outflows)
            - sum(inflows)
            - sum(sources)
            + sum(sinks)
        )
        scale = _normalizer(
            storage,
            sum(inflows),
            sum(outflows),
            sum(sources),
            sum(sinks),
        )
        residual = abs(raw_balance) / scale
        return _complete(
            kind,
            residual,
            tolerance,
            {"raw_balance": raw_balance, "normalization_scale": scale},
        )
    except InsufficientObservables as exc:
        return _abstain(kind, tolerance, str(exc))


def evaluate_complementarity(
    episode: Mapping[str, Any], tolerance: float = 1e-9
) -> LawEvaluation:
    """Check second-order interaction with an explicit expected sign.

    ``expected_interaction`` is +1 for complementarity, -1 for redundancy,
    and 0 for additivity. ``interaction_margin`` is preregistered.
    """

    kind = LawKind.COMPLEMENTARITY
    try:
        u0 = _number(episode, "u_empty")
        ua = _number(episode, "u_a")
        ub = _number(episode, "u_b")
        uab = _number(episode, "u_ab")
        expected = _number(episode, "expected_interaction")
        margin = _number(episode, "interaction_margin")
        if expected not in {-1.0, 0.0, 1.0} or margin < 0:
            return _abstain(kind, tolerance, "invalid interaction sign or margin")
        interaction = uab - ua - ub + u0
        scale = _normalizer(u0, ua, ub, uab)
        if expected == 0:
            residual = abs(interaction) / scale
        else:
            residual = max(0.0, margin - expected * interaction) / scale
        return _complete(
            kind,
            residual,
            tolerance,
            {"interaction": interaction, "normalization_scale": scale},
        )
    except InsufficientObservables as exc:
        return _abstain(kind, tolerance, str(exc))


def evaluate_negative_feedback(
    episode: Mapping[str, Any], tolerance: float = 1e-9
) -> LawEvaluation:
    """Check sign opposition and mitigation with declared temporal order."""

    kind = LawKind.NEGATIVE_FEEDBACK
    try:
        if not _flag(episode, "controlled_quantity_observed"):
            return _abstain(kind, tolerance, "controlled quantity is unobserved")
        if not _flag(episode, "same_controlled_quantity"):
            return _abstain(kind, tolerance, "before/after quantities are not comparable")
        if not _flag(episode, "disturbance_precedes_response"):
            return _abstain(kind, tolerance, "feedback temporal order is absent")
        if not _flag(episode, "local_stability_window_observed"):
            return _abstain(kind, tolerance, "local stability window is unobserved")
        disturbance = _number(episode, "disturbance_delta")
        response = _number(episode, "response_delta")
        net_before = _number(episode, "deviation_before_response")
        net_after = _number(episode, "deviation_after_response")
        response_margin = _number(episode, "response_margin")
        mitigation_margin = _number(episode, "mitigation_margin")
        if response_margin <= 0 or mitigation_margin <= 0:
            return _abstain(kind, tolerance, "strict margins must be positive")
        if disturbance == 0 or response == 0:
            return _complete(
                kind,
                1.0,
                tolerance,
                {"sign_violation": 1.0, "mitigation_violation": 1.0},
            )
        if not _flag(episode, "system_induced_response"):
            return _complete(
                kind,
                1.0,
                tolerance,
                {"sign_violation": 1.0, "mitigation_violation": 0.0},
            )
        opposition = -(disturbance * response)
        mitigation = abs(net_before) - abs(net_after)
        sign_violation = max(0.0, response_margin - opposition) / _normalizer(
            response_margin, opposition
        )
        mitigation_violation = max(
            0.0, mitigation_margin - mitigation
        ) / _normalizer(
            mitigation_margin, net_before, net_after
        )
        residual = max(sign_violation, mitigation_violation)
        return _complete(
            kind,
            residual,
            tolerance,
            {
                "sign_violation": sign_violation,
                "mitigation_violation": mitigation_violation,
            },
        )
    except InsufficientObservables as exc:
        return _abstain(kind, tolerance, str(exc))


def _distribution(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(item) for item in values)
    if not result or any(item < 0 or not isfinite(item) for item in result):
        raise InsufficientObservables("invalid probability vector")
    total = sum(result)
    if total <= 0:
        raise InsufficientObservables("zero-mass probability vector")
    return tuple(item / total for item in result)


def evaluate_locality(
    episode: Mapping[str, Any], tolerance: float = 1e-9
) -> LawEvaluation:
    """Check conditional invariance outside a declared Markov blanket."""

    kind = LawKind.LOCALITY
    try:
        if not _flag(episode, "blanket_observed"):
            return _abstain(kind, tolerance, "Markov blanket is not observed")
        if not _flag(episode, "same_blanket_state"):
            return _abstain(kind, tolerance, "contexts do not share blanket state")
        first = _distribution(_numbers(episode, "conditional_a"))
        second = _distribution(_numbers(episode, "conditional_b"))
        if len(first) != len(second):
            return _abstain(kind, tolerance, "conditional supports differ")
        total_variation = 0.5 * sum(
            abs(left - right) for left, right in zip(first, second, strict=True)
        )
        return _complete(
            kind,
            total_variation,
            tolerance,
            {"total_variation": total_variation},
        )
    except InsufficientObservables as exc:
        return _abstain(kind, tolerance, str(exc))


VERIFIERS: dict[LawKind, Callable[[Mapping[str, Any], float], LawEvaluation]] = {
    LawKind.SYMMETRY: evaluate_symmetry,
    LawKind.MONOTONICITY: evaluate_monotonicity,
    LawKind.CONSERVATION: evaluate_conservation,
    LawKind.COMPLEMENTARITY: evaluate_complementarity,
    LawKind.NEGATIVE_FEEDBACK: evaluate_negative_feedback,
    LawKind.LOCALITY: evaluate_locality,
}


def evaluate_law(
    kind: LawKind, episode: Mapping[str, Any], tolerance: float = 1e-9
) -> LawEvaluation:
    if tolerance < 0 or not isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    return VERIFIERS[kind](episode, tolerance)

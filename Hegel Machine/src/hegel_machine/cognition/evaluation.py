"""Fixed task losses and measured candidate comparisons."""

from __future__ import annotations

from math import isfinite
from statistics import mean
from typing import Sequence

from .models import Observation
from .language import MissingHistory


def predict(program, observation: Observation, fallback: float, *, context=None) -> float:
    # A trajectory starts before lagged measurements exist. Every such row is
    # still scored, using the same data-fitted initial prediction for all arms.
    try:
        value = float(program.predict(observation, context=context))
    except MissingHistory:
        return fallback
    if not isfinite(value):
        raise ValueError("candidate produced a non-finite prediction")
    return value


def point_loss(prediction: float, target: float, risk_quantile: float) -> float:
    residual = target - prediction
    if risk_quantile == 0.5:
        return residual * residual
    return max(risk_quantile * residual, (risk_quantile - 1.0) * residual)


def loss(program, observations: Sequence[Observation], fallback: float, q: float, *, contexts=None) -> float:
    if not observations:
        raise ValueError("cannot evaluate a candidate without observations")
    if contexts is None:
        return mean(point_loss(predict(program, row, fallback), row.target, q) for row in observations)
    return mean(point_loss(predict(program, row, fallback, context=context), row.target, q)
                for row, context in zip(observations, contexts))


def initial_prediction(observations: Sequence[Observation], q: float) -> float:
    values = sorted(row.target for row in observations)
    if not values:
        return 0.0
    if q == 0.5:
        return mean(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))]


def decision_metrics(program, observations, fallback, decision_model, costs):
    from .uncertainty import action_and_interval, decision_loss

    results = [(*action_and_interval(program, row, fallback, decision_model, costs), row.target)
               for row in observations]
    return {"mean_decision_loss": mean(decision_loss(action, target, costs) for action, low, high, target in results),
            "coverage": mean(low <= target <= high for action, low, high, target in results),
            "mean_interval_width": mean(high - low for action, low, high, target in results),
            "worst_case_loss": mean(max(decision_loss(action, low, costs), decision_loss(action, high, costs))
                                    for action, low, high, target in results),
            "model": decision_model.get("kind", "nominal")}

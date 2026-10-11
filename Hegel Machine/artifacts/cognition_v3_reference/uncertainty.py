"""Learn residual uncertainty and choose decisions under fixed task costs."""

from __future__ import annotations

from statistics import mean

from .evaluation import predict
from .models import Observation, SearchBudget


def _quantile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (index - low) * (values[high] - values[low])


def action_and_interval(program, row, fallback, model, costs):
    center = predict(program, row, fallback)
    if not model or model.get("kind") == "nominal":
        return center, center, center
    residuals = model["residual_interval"]
    if model.get("kind") == "conditional_interval":
        residuals = model["intervals"][int(row.features[model["feature"]] >= model["threshold"])]
    low, high = center + residuals[0], center + residuals[1]
    under, over = costs
    # Minimizing the worst endpoint loss balances underage and overage costs.
    action = (under * high + over * low) / (under + over)
    return action, low, high


def decision_loss(action, target, costs):
    under, over = costs
    return under * max(target - action, 0.0) + over * max(action - target, 0.0)


def fit_decision_model(program, training: list[Observation], validation: list[Observation], *,
                       fallback: float, costs: tuple[float, float], budget: SearchBudget) -> dict:
    if not training or not validation or not budget.spend(len(training)):
        return {"kind": "nominal", "status": "insufficient_budget"}
    residuals = [(row, row.target - predict(program, row, fallback)) for row in training]

    def interval(values):
        return [_quantile(values, 0.05), _quantile(values, 0.95)]

    values = [residual for _, residual in residuals]
    full_interval = interval(values)
    models = [{"kind": "nominal"},
              {"kind": "global_interval", "residual_interval": full_interval}]
    for name in training[0].features:
        threshold = _quantile([row.features[name] for row in training], 0.5)
        groups = [[value for row, value in residuals if (row.features[name] >= threshold) == side]
                  for side in (False, True)]
        if min(map(len, groups)) >= 3:
            models.append({"kind": "conditional_interval", "feature": name, "threshold": threshold,
                           "residual_interval": full_interval, "intervals": [interval(group) for group in groups]})
    measured = []
    for model in models:
        if not budget.spend(len(validation)):
            break
        value = mean(decision_loss(action_and_interval(program, row, fallback, model, costs)[0], row.target, costs)
                     for row in validation)
        measured.append((value, model))
    if not measured:
        return {"kind": "nominal", "status": "insufficient_budget"}
    score, chosen = min(measured, key=lambda item: item[0])
    return {**chosen, "fixed_costs": list(costs), "validation_decision_loss": score,
            "alternatives": [{"kind": model["kind"], "loss": value} for value, model in measured],
            "status": "measured_selection"}

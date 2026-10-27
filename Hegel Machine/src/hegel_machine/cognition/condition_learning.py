"""Learn observable application conditions and method selectors from paid rows."""

from __future__ import annotations

from .language import Program
from .models import Observation, SearchBudget
from .synthesis import _fit_linear


def prediction_matches(program: Program, row: Observation) -> bool:
    if len(row.history) < program.minimum_history_depth:
        return False
    prediction = program.predict(row)
    return abs(prediction - row.target) <= 1e-6 * max(1.0, abs(prediction), abs(row.target))


def _measured_coverage(rows):
    names = sorted(set.intersection(*(set(row.features) for row in rows)) - {"__target__"}) if rows else []
    return {"scope_purpose": "measured_coverage", "row_count": len(rows),
        "feature_ranges": {name: [min(row.features[name] for row in rows),
                                  max(row.features[name] for row in rows)] for name in names},
        "history_depth_range": [min(len(row.history) for row in rows),
                                max(len(row.history) for row in rows)] if rows else None,
        "independent_confirmation": False}


def _condition_coordinates(rows, max_history):
    names = sorted(set.intersection(*(set(row.features) for row in rows)) - {"__target__"})
    terms = [Program("feature", name=name) for name in names]
    depth = min(max((len(row.history) for row in rows), default=0), max_history)
    for lag in range(1, depth + 1):
        present = [row for row in rows if len(row.history) >= lag]
        historical = sorted(set.intersection(*(set(row.history[-lag]) for row in present)))
        guard = Program("available", lag=lag)
        if len(present) != len(rows):
            terms.append(guard)
        for name in historical:
            coordinate = Program("lag", name=name, lag=lag)
            terms.append(coordinate if len(present) == len(rows) else Program("if",
                children=(guard, coordinate, Program("const", value=0.))))
    return terms


def learn_condition(support: list[Observation], counterexamples: list[Observation],
                    budget: SearchBudget, *, scope_purpose="application", max_history=4) -> dict:
    """Find a simple separating half-space; interior conflicts remain unresolved.

    Labels concern the declared application or relative-utility question only.
    Measured coverage describes acquired inputs and never creates a predicate.
    Reset rows retain guarded historical coordinates; no future target enters
    the returned executable condition. Inseparability is an unresolved result
    within this finite grammar, not a proof that all conditions are impossible.
    """
    start = budget.spent
    if scope_purpose not in {"application", "relative_utility", "measured_coverage"}:
        raise ValueError("scope_purpose must name application, relative_utility or measured_coverage")
    result = {"condition": None, "support_rows": len(support),
              "counterexample_rows": len(counterexamples), "status": "not_separable",
              "scope_purpose": scope_purpose, "scope_status": "unknown"}
    rows = support + counterexamples
    result["measured_coverage"] = _measured_coverage(rows)
    if scope_purpose == "measured_coverage":
        return {**result, "status": "measured_only", "compute_cost": 0}
    if len(support) < 2 or not counterexamples:
        return {**result, "status": "insufficient_support", "compute_cost": 0}
    terms = _condition_coordinates(rows, max_history)
    result["condition_grammar"] = {"kind": "observable_affine_halfspace",
        "max_history": max_history, "guarded_history": True,
        "coordinate_count": len(terms)}
    if not terms:
        return {**result, "status": "no_common_observables", "compute_cost": 0}

    def separator(coordinate):
        if not budget.spend(len(rows)):
            return None
        positive = [coordinate.predict(row) for row in support]
        negative = [coordinate.predict(row) for row in counterexamples]
        if min(positive) > max(negative):
            return Program("ge", children=(coordinate, Program("const", value=(min(positive) + max(negative)) / 2)))
        if max(positive) < min(negative):
            return Program("lt", children=(coordinate, Program("const", value=(max(positive) + min(negative)) / 2)))
        return None

    for term in terms:
        condition = separator(term)
        if condition is not None:
            return {**result, "condition": condition.to_dict(), "status": "separated",
                    "scope_status": "supported_on_supplied_rows", "compute_cost": budget.spent - start}
    if len(terms) > 1 and budget.spend(len(rows)):
        validity = [Observation(row.features, float(index < len(support)), row.history)
                    for index, row in enumerate(rows)]
        direction = _fit_linear(terms, validity)
        condition = separator(direction)
        if condition is not None:
            return {**result, "condition": condition.to_dict(), "status": "separated",
                    "scope_status": "supported_on_supplied_rows", "compute_cost": budget.spent - start}
    return {**result, "compute_cost": budget.spent - start}


def learn_relative_utility_condition(rows, baseline_predictions, candidate_predictions,
                                     budget, *, objective="pointwise_improvement",
                                     max_history=4, world_mse_threshold=1e-10):
    """Return a development method selector, never a world-validity condition.

    Predictions must already be paid and frozen. Historical truth supplies loss
    labels; the exported selector only reads observables and available history.
    A mean-loss objective must be selected by the caller before looking at this
    result. Its all-row utility can be positive despite retained worse rows.
    """
    rows = list(rows)
    if not (len(rows) == len(baseline_predictions) == len(candidate_predictions)):
        raise ValueError("relative utility needs one frozen prediction per supplied row")
    if objective not in {"pointwise_improvement", "mean_loss"}:
        raise ValueError("relative utility objective must be declared before learning")
    start = budget.spent
    result = {"scope_purpose": "relative_utility", "objective": objective,
        "selector": None, "status": "insufficient_budget", "scope_status": "unknown",
        "application_scope_status": "not_assessed",
        "measured_coverage": _measured_coverage(rows),
        "world_mse_threshold": world_mse_threshold, "compute_cost": 0}
    if not rows:
        return {**result, "status": "insufficient_support"}
    if not budget.spend(len(rows)):
        return result
    baseline = [(prediction - row.target) ** 2
                for row, prediction in zip(rows, baseline_predictions)]
    candidate = [(prediction - row.target) ** 2
                 for row, prediction in zip(rows, candidate_predictions)]
    positive = [row for row, before, after in zip(rows, baseline, candidate) if after < before]
    negative = [row for row, before, after in zip(rows, baseline, candidate) if after >= before]
    result.update(baseline_mse=sum(baseline) / len(rows),
        candidate_mse=sum(candidate) / len(rows),
        strict_improvement_rows=len(positive), nonimprovement_rows=len(negative),
        worst_loss_increase=max(after - before for before, after in zip(baseline, candidate)))
    if objective == "mean_loss":
        improved = result["candidate_mse"] < result["baseline_mse"]
        result.update(status="mean_improvement" if improved else "no_mean_improvement",
            scope_status="supported_on_supplied_rows" if improved else "not_supported_on_supplied_rows",
            selector=Program("const", value=float(improved)).to_dict())
    else:
        learned = learn_condition(positive, negative, budget,
            scope_purpose="relative_utility", max_history=max_history)
        result.update({key: value for key, value in learned.items()
                       if key not in {"condition", "compute_cost", "measured_coverage"}})
        result["selector"] = learned["condition"]
    result["compute_cost"] = budget.spent - start
    return result

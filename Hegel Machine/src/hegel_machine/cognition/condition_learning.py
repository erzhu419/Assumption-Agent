"""Learn observable validity conditions from supported rows and counterexamples."""

from __future__ import annotations

from .language import Program
from .models import Observation, SearchBudget
from .synthesis import _fit_linear


def prediction_matches(program: Program, row: Observation) -> bool:
    if len(row.history) < program.minimum_history_depth:
        return False
    prediction = program.predict(row)
    return abs(prediction - row.target) <= 1e-6 * max(1.0, abs(prediction), abs(row.target))


def learn_condition(support: list[Observation], counterexamples: list[Observation],
                    budget: SearchBudget) -> dict:
    """Find a simple separating half-space; interior conflicts remain unresolved.

    Raw coordinates compete with a direction fitted from validity labels. We do
    not cut arbitrary holes around individual errors: if supported observations
    surround a contradictory observation, a half-space cannot excuse it.
    """
    start = budget.spent
    result = {"condition": None, "support_rows": len(support),
              "counterexample_rows": len(counterexamples), "status": "not_separable"}
    rows = support + counterexamples
    if len(support) < 2 or not counterexamples:
        return {**result, "status": "insufficient_support", "compute_cost": 0}
    names = sorted(set.intersection(*(set(row.features) for row in rows)))
    terms = [Program("feature", name=name) for name in names]
    depth = min((len(row.history) for row in rows), default=0)
    for lag in range(1, min(depth, 4) + 1):
        historical = sorted(set.intersection(*(set(row.history[-lag]) for row in rows)))
        terms.extend(Program("lag", name=name, lag=lag) for name in historical)
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
                    "compute_cost": budget.spent - start}
    if len(terms) > 1 and budget.spend(len(rows)):
        validity = [Observation(row.features, float(index < len(support)), row.history)
                    for index, row in enumerate(rows)]
        direction = _fit_linear(terms, validity)
        condition = separator(direction)
        if condition is not None:
            return {**result, "condition": condition.to_dict(), "status": "separated",
                    "compute_cost": budget.spent - start}
    return {**result, "compute_cost": budget.spent - start}

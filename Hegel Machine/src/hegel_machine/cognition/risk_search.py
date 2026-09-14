"""Budgeted quantile regression in the ordinary executable language."""

from __future__ import annotations

from math import ceil
from statistics import mean
from typing import Sequence

from .language import Program
from .models import Candidate, Observation, SearchBudget
from .synthesis import _solve


def empirical_quantile(values: Sequence[float], probability: float) -> float:
    """A minimizer of empirical pinball loss, including small tail samples."""
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, ceil(probability * len(ordered)) - 1))]


def weighted_quantile(values: Sequence[tuple[float, float]], probability: float) -> float:
    ordered = sorted(values)
    target = probability * sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target - 1e-12:
            return value
    return ordered[-1][0]


def trajectory_blocks(observations: Sequence[Observation]) -> list[list[Observation]]:
    """Keep acquired trajectories intact when separating fitting and scoring."""
    blocks: list[list[Observation]] = []
    previous = 0
    for row in observations:
        depth = len(row.history)
        if not blocks or depth == 0 or depth < previous:
            blocks.append([])
        blocks[-1].append(row)
        previous = depth
    return blocks


def _expression(terms, means, scales, coefficients):
    slopes = [coefficient / scale for coefficient, scale in zip(coefficients[1:], scales)]
    intercept = coefficients[0] - sum(slope * center for slope, center in zip(slopes, means))
    program = Program("const", value=intercept)
    for term, slope in zip(terms, slopes):
        if abs(slope) > 1e-9:
            program = Program("add", children=(program,
                              Program("mul", children=(Program("const", value=slope), term))))
    return program


def _coordinate_candidates(observations: Sequence[Observation], *, budget: SearchBudget,
                           q: float, pooled_location: bool = False) -> list[Candidate]:
    """Fit fixed-quantile actions, without treating a mean fit as a quantile.

    The coordinate vocabulary contains current features and their absolute
    values. Each model starts from a fitted affine trend plus a residual
    quantile, then directly minimizes pinball loss by weighted-quantile
    coordinate updates. Models share computed columns and take turns using the
    remaining budget. Every fit and coefficient update is charged by row.
    """
    if not observations:
        return []
    if not 0 < q < 1:
        raise ValueError("risk quantile must be between zero and one")
    count = len(observations)
    if not budget.spend(count):
        return []
    targets = [row.target for row in observations]
    candidates = [Candidate(f"risk_constant:{budget.spent}",
                  Program("const", value=empirical_quantile(targets, q)), "robustification",
                  "Fit the task's fixed empirical quantile")]
    current = [Program("feature", name=name)
               for name in sorted(set.intersection(*(set(row.features) for row in observations)))]
    shapes = [tuple(current), tuple([*current, *[Program("abs", children=(term,)) for term in current]])]
    if pooled_location:
        # A robust location plus pooled global tail is already fitted. Spend
        # the direct pinball allowance on jointly adjusting conditional tails.
        shapes = shapes[-1:]
    elif len(current) > 1:
        shapes.extend((term,) for term in current)
    shapes = list(dict.fromkeys(shape for shape in shapes if shape))
    columns = {}
    states = []
    for terms in shapes:
        missing = [term for term in terms if term not in columns]
        if budget.remaining < count * (len(missing) + 2):
            continue
        for term in missing:
            budget.spend(count)
            columns[term] = [term.predict(row) for row in observations]
        centers = [mean(columns[term]) for term in terms]
        scales = [max(abs(value - center) for value in columns[term]) or 1.0
                  for term, center in zip(terms, centers)]
        design = [[1.0] + [(columns[term][index] - center) / scale
                          for term, center, scale in zip(terms, centers, scales)]
                  for index in range(count)]
        width = len(terms) + 1
        budget.spend(count)
        gram = [[sum(row[left] * row[right] for row in design)
                 for right in range(width)] for left in range(width)]
        for index in range(1, width):
            gram[index][index] += 1e-8
        coefficients = _solve(gram, [sum(row[index] * target for row, target in zip(design, targets))
                                    for index in range(width)])
        budget.spend(count)
        residuals = [target - sum(coef * value for coef, value in zip(coefficients, row))
                     for row, target in zip(design, targets)]
        offset = empirical_quantile(residuals, q)
        coefficients[0] += offset
        residuals = [value - offset for value in residuals]
        states.append({"terms": terms, "centers": centers, "scales": scales,
                       "design": design, "coefficients": coefficients,
                       "residuals": residuals, "cursor": 1 if pooled_location else 0, "updates": 0,
                       "sweep_change": 0.0, "finished": False})
    # A coordinate has an exact one-dimensional pinball minimizer. Negative
    # design values reverse q; the weighted quantile includes this reversal.
    while states and budget.remaining >= count and any(not state["finished"] for state in states):
        for state in states:
            width = len(state["coefficients"])
            if state["finished"] or not budget.spend(count):
                continue
            index = state["cursor"]
            column = [row[index] for row in state["design"]]
            ratios = [(residual / value, abs(value))
                      for residual, value in zip(state["residuals"], column) if abs(value) > 1e-10]
            if ratios:
                mass = sum(abs(value) for value in column if abs(value) > 1e-10)
                probability = sum(abs(value) * (q if value > 0 else 1.0 - q)
                                  for value in column if abs(value) > 1e-10) / mass
                change = weighted_quantile(ratios, probability)
                state["coefficients"][index] += change
                state["residuals"] = [residual - change * value
                                      for residual, value in zip(state["residuals"], column)]
                state["sweep_change"] += abs(change)
            state["cursor"] = (index + 1) % width
            state["updates"] += 1
            if state["cursor"] == 0:
                state["finished"] = state["sweep_change"] < 1e-9 or state["updates"] >= 8 * width
                state["sweep_change"] = 0.0
    for index, state in enumerate(states):
        program = _expression(state["terms"], state["centers"], state["scales"], state["coefficients"])
        candidates.append(Candidate(f"risk_quantile:{budget.spent}:{index}", program, "robustification",
                          "Direct pinball fit of observed coordinates under the task's fixed quantile"))
    return candidates


def _pooled_candidates(observations, *, budget, q):
    """Share the tail across an estimated location and absolute-residual scale.

    Central observations estimate the trend and the scale. The tail quantile
    then uses all standardized residuals, rather than fitting each tail slope
    from the few extreme rows. A bounded Huber reweighting limits the influence
    of an isolated tail observation on the location fit. The direct pinball
    alternative remains available when this location/scale approximation fails.
    """
    n = len(observations)
    names = sorted(set.intersection(*(set(row.features) for row in observations)))
    if not names or n < len(names) + 3 or budget.remaining < 12 * n:
        return []
    terms = [Program("feature", name=name) for name in names]
    targets = [row.target for row in observations]
    # Column materialization is shared by all four location fits.
    budget.spend(n)
    columns = [[row.features[name] for row in observations] for name in names]
    centers = [mean(column) for column in columns]
    scales = [max(abs(value - center) for value in column) or 1.0
              for column, center in zip(columns, centers)]
    design = [[1.0] + [(column[index] - center) / scale
                       for column, center, scale in zip(columns, centers, scales)]
              for index in range(n)]

    def fit(design, targets, weights):
        budget.spend(n)
        width = len(design[0])
        matrix = [[sum(weight * row[a] * row[b] for row, weight in zip(design, weights))
                   for b in range(width)] for a in range(width)]
        for index in range(1, width):
            matrix[index][index] += 1e-8
        return _solve(matrix, [sum(weight * row[a] * target
                                  for row, weight, target in zip(design, weights, targets))
                               for a in range(width)])

    weights = [1.0] * n
    coefficients = fit(design, targets, weights)
    for iteration in range(4):
        budget.spend(n)
        residuals = [target - sum(coef * value for coef, value in zip(coefficients, row))
                     for target, row in zip(targets, design)]
        if iteration == 3:
            break
        cutoff = 1.5 * empirical_quantile([abs(value) for value in residuals], .5)
        if cutoff <= 1e-10:
            break
        weights = [min(1.0, cutoff / max(abs(value), 1e-10)) for value in residuals]
        coefficients = fit(design, targets, weights)
    location = _expression(terms, centers, scales, coefficients)
    # The homoscedastic alternative uses the same robust location estimate.
    budget.spend(n)
    offset = empirical_quantile(residuals, q)
    global_program = Program("add", children=(location, Program("const", value=offset)))
    results = [Candidate(f"risk_pooled_global:{budget.spent}", global_program, "robustification",
                         "Pool the observed tail around a robustly estimated location")]
    scale_terms = [Program("abs", children=(term,)) for term in terms]
    absolute_columns = [[abs(value) for value in column] for column in columns]
    scale_centers = [mean(column) for column in absolute_columns]
    scale_scales = [max(abs(value - center) for value in column) or 1.0
                    for column, center in zip(absolute_columns, scale_centers)]
    scale_design = [[1.0] + [(column[index] - center) / scale
                             for column, center, scale in zip(absolute_columns, scale_centers, scale_scales)]
                    for index in range(n)]
    spread_coefficients = fit(scale_design, [abs(value) for value in residuals], [1.0] * n)
    spread = _expression(scale_terms, scale_centers, scale_scales, spread_coefficients)
    # A positive pooled floor makes standardization executable where a linear
    # scale estimate crosses zero. It is learned from observed residual size.
    floor = max(1e-10, .2 * mean(abs(value) for value in residuals))
    spread = Program("max", children=(Program("const", value=floor), spread))
    budget.spend(n)
    standardized = [value / spread.predict(row) for row, value in zip(observations, residuals)]
    quantile = empirical_quantile(standardized, q)
    program = Program("add", children=(location, Program("mul", children=(Program("const", value=quantile), spread))))
    results.append(Candidate(f"risk_pooled_scale:{budget.spent}", program, "robustification",
                            "Fit residual scale from central and tail rows, then pool their standardized tail"))
    return results


def risk_candidates(observations: Sequence[Observation], *, budget: SearchBudget,
                    q: float, tail_fit: bool = True) -> list[Candidate]:
    """Compare pooled location/scale and direct quantile fits within one budget.

    ``tail_fit=False`` runs the 0.6 coordinate search for the measured ablation.
    The new search uses two pooled fits, a conditional direct fit and a
    constant. It reduces the number of competing shapes and reserves the
    remaining direct-fit budget for conditional tail coefficients.
    """
    if not tail_fit or not observations:
        return _coordinate_candidates(observations, budget=budget, q=q)
    if not 0 < q < 1:
        raise ValueError("risk quantile must be between zero and one")
    # Reserve the empirical tail action even when little fitting is possible.
    pooled_budget = SearchBudget(min(budget.remaining // 2, max(0, budget.remaining - len(observations))))
    pooled = _pooled_candidates(observations, budget=pooled_budget, q=q)
    budget.spend(pooled_budget.spent)
    direct = _coordinate_candidates(observations, budget=budget, q=q, pooled_location=bool(pooled))
    return [*direct, *pooled]

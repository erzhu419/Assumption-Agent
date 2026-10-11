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


def risk_candidates(observations: Sequence[Observation], *, budget: SearchBudget,
                    q: float) -> list[Candidate]:
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
    if len(current) > 1:
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
                       "residuals": residuals, "cursor": 0, "updates": 0,
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

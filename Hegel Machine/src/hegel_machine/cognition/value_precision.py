"""Spend bounded prediction work to resolve a disputed action-value ranking."""

from copy import deepcopy
from math import sqrt
from random import Random
from statistics import mean, variance

from .models import SearchBudget


def compare_action_values(actions, score, *, seed, max_evaluations):
    """Compare raw KG values on shared, independent computational innovations.

    ``score(action, innovations)`` returns one raw KG value per innovation.
    Each candidate/draw costs one evaluation; a full paired block is bought
    for every candidate together. Four independent antithetic pairs are the
    minimum for an empirical comparison, and eight pairs are the fixed cap.
    Paired standard errors are a planning heuristic, not coverage guarantees.
    Initial MC1 values remain a baseline and do not enter the new comparison.
    ``action['tie_break']`` is the caller's existing feasibility certificate,
    posterior objective, replicate count, and control coordinates, in that
    order; this module does not acquire another prediction to construct it.
    """
    actions = sorted(deepcopy(actions), key=lambda item: tuple(item["point"]))
    if len(actions) < 2:
        raise ValueError("value precision needs at least two competing actions")
    budget = SearchBudget(max_evaluations)
    rng = Random(seed)
    pair_means = [[] for _ in actions]
    minimum_pairs, maximum_pairs = 4, 8
    blocks = 0
    comparisons = []
    separated = False
    flat = False
    best = None

    while blocks < maximum_pairs:
        if not budget.spend(2 * len(actions)):
            break
        z = [rng.gauss(0., 1.) for _ in range(2)]
        innovations = [z, [-value for value in z]]
        for index, action in enumerate(actions):
            values = score(action, deepcopy(innovations))
            if len(values) != 2:
                raise ValueError("each paid innovation needs its own raw action value")
            pair_means[index].append(mean(values))
        blocks += 1
        if blocks < minimum_pairs:
            continue
        averages = [mean(values) for values in pair_means]
        best = max(range(len(actions)), key=lambda index: averages[index])
        comparisons = []
        for index, action in enumerate(actions):
            if index == best:
                continue
            differences = [a - b for a, b in zip(pair_means[best], pair_means[index])]
            difference = mean(differences)
            standard_error = sqrt(variance(differences) / blocks)
            comparisons.append({"other_point": deepcopy(action["point"]),
                "mean_difference": difference, "standard_error": standard_error,
                "separated": difference > 2 * standard_error + 1e-12})
        separated = all(item["separated"] for item in comparisons)
        if separated:
            break
        unresolved = [item for item in comparisons if not item["separated"]]
        flat = all(abs(item["mean_difference"]) <= 1e-12
                   and item["standard_error"] <= 1e-12 for item in unresolved)
        if flat:
            break

    if separated:
        eligible = [best]
        selected = best
        reason = "paired_difference"
    else:
        excluded = {tuple(item["other_point"]) for item in comparisons if item["separated"]}
        eligible = [index for index, action in enumerate(actions) if tuple(action["point"]) not in excluded]
        # An unresolved comparison does not establish a gain. Prefer the
        # caller's existing certificate and recommendation ordering.
        selected = min(eligible, key=lambda index: tuple(actions[index]["tie_break"]))
        reason = ("numerically_flat_paired_values" if flat else
                  "sample_cap" if blocks == maximum_pairs else "evaluation_budget")

    return {"selected_point": deepcopy(actions[selected]["point"]),
        "status": "empirically_separated" if separated else "unresolved",
        "stop_reason": reason, "evaluations": budget.spent, "paired_blocks": blocks,
        "draws_per_action": 2 * blocks, "maximum_pairs": maximum_pairs,
        "uncertainty_kind": "empirical_paired_standard_error",
        "comparison_best_point": deepcopy(actions[best]["point"]) if best is not None else None,
        "eligible_points": [deepcopy(actions[index]["point"]) for index in eligible],
        "comparisons": comparisons,
        "action_statistics": [{**deepcopy(action),
            "mean_raw": mean(values) if values else None, "pair_means": values}
            for action, values in zip(actions, pair_means)]}

"""Compare expected native lexicographic terminal gains on paired draws."""

from copy import deepcopy
from math import sqrt
from random import Random
from statistics import mean, variance

from .models import SearchBudget


def _comparison(first, second):
    for component in range(len(first[0])):
        differences = [a[component] - b[component] for a, b in zip(first, second)]
        # A lower-priority objective cannot resolve an uncertain difference in
        # certificate probability. Descend only when paired values agree.
        if all(abs(value) <= 1e-12 for value in differences):
            continue
        average = mean(differences)
        error = sqrt(variance(differences) / len(differences))
        return {"component": component, "mean_difference": average,
            "standard_error": error, "separated": average > 2 * error + 1e-12,
            "equal": False}
    return {"component": None, "mean_difference": 0., "standard_error": 0.,
            "separated": False, "equal": True}


def compare_terminal_values(actions, score, *, seed, max_evaluations):
    """Keep probability, constraint distance and goal in separate units.

    ``score`` returns one complete ``current - future`` terminal vector per
    innovation. Average the components before lexicographic comparison. The
    paired sampling rule and budget are the same as ordinary value precision.
    """
    actions = sorted(deepcopy(actions), key=lambda item: tuple(item["point"]))
    if len(actions) < 2:
        raise ValueError("terminal comparison needs at least two actions")
    budget, rng = SearchBudget(max_evaluations), Random(seed)
    pairs = [[] for _ in actions]
    blocks, comparisons, best = 0, [], None
    separated = flat = False
    while blocks < 8:
        if not budget.spend(2 * len(actions)):
            break
        z = [rng.gauss(0., 1.) for _ in range(2)]
        for index, action in enumerate(actions):
            values = score(action, [z, [-value for value in z]])
            if len(values) != 2 or any(len(value) != 3 for value in values):
                raise ValueError("each innovation needs a native three-component terminal gain")
            pairs[index].append([mean(value[c] for value in values) for c in range(3)])
        blocks += 1
        if blocks < 4:
            continue
        averages = [tuple(mean(value[c] for value in values) for c in range(3)) for values in pairs]
        best = max(range(len(actions)), key=lambda index: averages[index])
        comparisons = [{"other_point": deepcopy(action["point"]),
                        **_comparison(pairs[best], pairs[index])}
                       for index, action in enumerate(actions) if index != best]
        separated = all(item["separated"] for item in comparisons)
        flat = all(item["equal"] for item in comparisons if not item["separated"])
        if separated or flat:
            break
    if separated:
        selected, eligible, reason = best, [best], "paired_lexicographic_difference"
    else:
        excluded = {tuple(item["other_point"]) for item in comparisons if item["separated"]}
        eligible = [i for i, action in enumerate(actions) if tuple(action["point"]) not in excluded]
        selected = min(eligible, key=lambda i: tuple(actions[i]["tie_break"]))
        reason = ("numerically_flat_paired_values" if flat else
                  "sample_cap" if blocks == 8 else "evaluation_budget")
    return {"selected_point": deepcopy(actions[selected]["point"]),
        "status": "empirically_separated" if separated else "unresolved",
        "stop_reason": reason, "evaluations": budget.spent, "paired_blocks": blocks,
        "draws_per_action": 2 * blocks, "maximum_pairs": 8,
        "uncertainty_kind": "paired_standard_error_in_first_differing_component",
        "comparison_best_point": deepcopy(actions[best]["point"]) if best is not None else None,
        "eligible_points": [deepcopy(actions[i]["point"]) for i in eligible],
        "comparisons": comparisons,
        "action_statistics": [{**deepcopy(action), "pair_means": values,
            "mean_gain": [mean(row[c] for row in values) for c in range(3)] if values else None}
            for action, values in zip(actions, pairs)]}

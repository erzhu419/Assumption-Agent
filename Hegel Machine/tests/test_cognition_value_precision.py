"""Additional KG precision spends predictions, not real observations."""

from copy import deepcopy
from math import sqrt
import random
from statistics import variance

import pytest

from hegel_machine.cognition.value_precision import compare_action_values


def _action(point, *, initial=-.01, objective=1., repeats=1, feasible=True):
    return {"point": list(point), "initial_raw": initial,
        "feasible": feasible, "posterior_value": objective, "replicate_count": repeats,
        "tie_break": [not feasible, objective, repeats, *point]}


def test_shared_innovations_resolve_relative_value_without_independent_candidate_noise():
    actions = [_action((1, 0)), _action((0, 0))]
    calls = []

    def score(action, innovations):
        calls.append((deepcopy(action["point"]), deepcopy(innovations)))
        return [.03 * action["point"][0] + 1000 * z[0] for z in innovations]

    result = compare_action_values(actions, score, seed=1701, max_evaluations=64)
    assert result["selected_point"] == [1, 0]
    assert result["status"] == "empirically_separated"
    assert result["stop_reason"] == "paired_difference"
    assert result["paired_blocks"] == 4 and result["evaluations"] == 16
    assert len(calls) == 8
    for first, second in zip(calls[::2], calls[1::2]):
        assert first[1] == second[1]
        assert first[1][1] == [-value for value in first[1][0]]
    assert result["comparisons"][0]["mean_difference"] == pytest.approx(.03)


def test_all_negative_values_can_have_a_supported_ranking_without_claiming_positive_gain():
    actions = [_action((0, 0)), _action((1, 0))]

    def score(action, innovations):
        return [-.1 if action["point"][0] else -.3 for _ in innovations]

    result = compare_action_values(actions, score, seed=12, max_evaluations=64)
    assert result["status"] == "empirically_separated" and result["selected_point"] == [1, 0]
    assert all(item["mean_raw"] < 0 for item in result["action_statistics"])


def test_unresolved_comparison_uses_certificate_then_posterior_and_measured_count_not_input_order():
    actions = [_action((0, 0), objective=-10., feasible=False),
               _action((1, 0), objective=.1, repeats=5),
               _action((2, 0), objective=.1, repeats=1)]

    def score(action, innovations):
        return [z[0] ** 2 for z in innovations]

    result = compare_action_values(actions, score, seed=72, max_evaluations=512)
    reversed_result = compare_action_values(list(reversed(actions)), score, seed=72, max_evaluations=512)
    assert result == reversed_result
    assert result["selected_point"] == [2, 0]
    assert result["status"] == "unresolved"
    assert result["stop_reason"] == "numerically_flat_paired_values"
    assert result["paired_blocks"] == 4 and result["evaluations"] == 24
    assert len(result["eligible_points"]) == 3
    assert all(item["mean_difference"] == 0 for item in result["comparisons"])


def test_uncertain_tie_rule_does_not_restore_a_candidate_already_separated_as_worse():
    actions = [_action((0, 0), objective=.2), _action((1, 0), objective=.1),
               _action((2, 0), objective=-10.)]

    def score(action, innovations):
        return [z[0] ** 2 - (10 if action["point"][0] == 2 else 0) for z in innovations]

    result = compare_action_values(actions, score, seed=54, max_evaluations=512)
    assert result["status"] == "unresolved"
    assert result["eligible_points"] == [[0, 0], [1, 0]]
    assert result["selected_point"] == [1, 0]
    assert result["stop_reason"] == "numerically_flat_paired_values"
    assert result["paired_blocks"] == 4


@pytest.mark.parametrize("available,expected_draws", [(0, 0), (3, 0), (6, 4), (14, 12)])
def test_budget_buys_only_complete_common_pairs_for_every_candidate(available, expected_draws):
    actions = [_action((0, 0), objective=.2), _action((1, 0), objective=.1)]
    counts = {tuple(action["point"]): 0 for action in actions}

    def score(action, innovations):
        counts[tuple(action["point"])] += len(innovations)
        return [1. if action["point"][0] else 0. for _ in innovations]

    result = compare_action_values(actions, score, seed=5, max_evaluations=available)
    assert result["evaluations"] == sum(counts.values()) == expected_draws
    assert result["evaluations"] <= available
    assert len(set(counts.values())) == 1
    assert result["status"] == "unresolved" and result["stop_reason"] == "evaluation_budget"
    assert result["paired_blocks"] < 4


def test_standard_error_counts_independent_pairs_instead_of_correlated_antithetic_draws():
    actions = [_action((0, 0)), _action((1, 0))]

    def score(action, innovations):
        return [z[0] ** 2 + .2 * action["point"][0] * z[1] ** 2 for z in innovations]

    result = compare_action_values(actions, score, seed=83, max_evaluations=64)
    by_point = {tuple(item["point"]): item["pair_means"] for item in result["action_statistics"]}
    best = tuple(result["comparison_best_point"])
    comparison, = result["comparisons"]
    other = tuple(comparison["other_point"])
    differences = [a - b for a, b in zip(by_point[best], by_point[other])]
    expected = sqrt(variance(differences) / len(differences))
    assert comparison["standard_error"] == pytest.approx(expected)
    assert expected > 0


def test_original_conditionally_negative_mc1_is_reported_but_not_mixed_into_new_evidence():
    actions = [_action((0, 0), initial=-1e6), _action((1, 0), initial=-.000001)]

    def score(action, innovations):
        return [.2 if action["point"][0] == 0 else -.1 for _ in innovations]

    result = compare_action_values(actions, score, seed=1, max_evaluations=64)
    assert result["selected_point"] == [0, 0]
    first, second = result["action_statistics"]
    assert first["initial_raw"] == -1e6 and first["mean_raw"] == pytest.approx(.2)
    assert second["initial_raw"] == -.000001 and second["mean_raw"] == pytest.approx(-.1)


def test_precision_does_not_consume_global_rng_or_modify_paid_action_metadata():
    actions = [_action((0, 0)), _action((1, 0))]
    saved_actions = deepcopy(actions)
    saved_random_state = random.getstate()

    def score(action, innovations):
        return [z[0] for z in innovations]

    compare_action_values(actions, score, seed=399, max_evaluations=64)
    assert actions == saved_actions
    assert random.getstate() == saved_random_state

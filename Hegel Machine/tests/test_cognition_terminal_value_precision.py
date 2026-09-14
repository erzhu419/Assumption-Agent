"""Terminal evidence preserves native feasibility priorities before goal value."""

from collections import defaultdict

import pytest

from hegel_machine.cognition.terminal_value_precision import compare_terminal_values


def actions():
    return [{"point": [0], "tie_break": [1]}, {"point": [1], "tie_break": [0]}]


def test_certificate_recovery_outweighs_any_lower_priority_goal_gain():
    draws = defaultdict(list)

    def score(action, innovations):
        draws[action["point"][0]].append(innovations)
        gain = [1., .1, -1e6] if action["point"] == [0] else [0., 0., 1e6]
        return [gain[:] for _ in innovations]

    result = compare_terminal_values(actions(), score, seed=19, max_evaluations=32)
    assert result["selected_point"] == [0]
    assert result["status"] == "empirically_separated"
    assert result["comparisons"][0]["component"] == 0
    assert result["paired_blocks"] == 4 and result["evaluations"] == 16
    assert draws[0] == draws[1]
    # Inconclusive certificate evidence cannot be resolved by a large goal gain.
    count = defaultdict(int)

    def uncertain(action, innovations):
        point = action["point"][0]
        count[point] += 1
        gain = [(-1.) ** count[point], 0., 1e6] if point == 0 else [0., 0., 0.]
        return [gain[:] for _ in innovations]

    unresolved = compare_terminal_values(actions(), uncertain, seed=19, max_evaluations=32)
    assert unresolved["status"] == "unresolved"
    assert unresolved["selected_point"] == [1]
    assert unresolved["comparisons"][0]["component"] == 0


def test_all_certified_futures_compare_goal_values_in_the_original_units():
    def score(action, innovations):
        goal_gain = .03 if action["point"] == [0] else .02
        return [[0., 0., goal_gain + .01 * innovation[0]] for innovation in innovations]

    result = compare_terminal_values(actions(), score, seed=7, max_evaluations=32)
    assert result["selected_point"] == [0]
    assert result["comparisons"][0]["component"] == 2
    assert result["comparisons"][0]["mean_difference"] == pytest.approx(.01)
    assert result["action_statistics"][0]["mean_gain"] == pytest.approx([0., 0., .03])


def test_mixed_futures_average_components_before_lexicographic_selection():
    outcomes = {0: [[1., .1, -100.], [0., 0., -100.]],
                1: [[0., 0., 1.], [0., 0., 1.]]}

    def score(action, innovations):
        return outcomes[action["point"][0]]

    # Averaging first-nonzero scalar diagnostics would prefer action one.
    scalar = {key: sum(next(value for value in gain if value != 0.)
                       for gain in gains) / 2 for key, gains in outcomes.items()}
    assert scalar[0] < scalar[1]
    result = compare_terminal_values(actions(), score, seed=11, max_evaluations=32)
    assert result["selected_point"] == [0]
    assert result["action_statistics"][0]["mean_gain"] == pytest.approx([.5, .05, -100.])
    assert result["comparisons"][0]["component"] == 0

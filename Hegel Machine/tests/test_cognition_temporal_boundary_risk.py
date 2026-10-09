"""Whole-window forecasts match the response loss of real gate events."""

from itertools import product

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.observed_condition_experiments import _boundary_response_risk


@pytest.mark.parametrize("placement", ["inside_window", "outside_window"])
def test_window_boundary_risk_matches_all_actual_reading_combinations(placement):
    drive = Program("feature", name="drive")
    gate = Program("ge", children=(Program("feature", name="meter"),
                                  Program("const", value=0.)))
    conditioned = Program("if", children=(gate, drive, Program("const", value=0.)))
    program = (Program("window_sum", lag=3, children=(conditioned,))
               if placement == "inside_window" else Program("if", children=(gate,
                   Program("window_sum", lag=3, children=(drive,)), Program("const", value=0.))))
    # Signed history contributions expose incorrect squared-sum shortcuts;
    # a current-only gate has a different risk despite the same window width.
    drives, probability = (2., -1., 3.), .2
    expected = 0.
    for events in product((False, True), repeat=3):
        weight = 1.
        for event in events:
            weight *= probability if event else 1. - probability
        difference = (sum(value for value, event in zip(drives, events) if event)
                      if placement == "inside_window" else sum(drives) * events[-1])
        expected += weight * difference ** 2
    row = Observation({"drive": drives[-1], "meter": -1.}, 0.,
        tuple({"drive": value, "meter": -1.} for value in drives[:-1]))
    budget = SearchBudget(6)
    forecast = probability * _boundary_response_risk(program, row,
        {"condition_field": "meter", "threshold_interval": [-.1, .1]}, probability, budget)
    assert forecast == pytest.approx(expected)
    assert budget.spent == (6 if placement == "inside_window" else 2)

"""Condition research shares executed scalar work without sharing model truth."""

import pytest

from hegel_machine.cognition.language import MissingHistory, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.transfer_condition_research import TransferConditionResearch


def const(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def calibration():
    x = Program("feature", name="x")
    return binary("add", binary("mul", x, x),
                  binary("mul", const(.4), Program("lag", name="x", lag=1)))


def observation(x=1.):
    return Observation({"x": x, "y": -.5}, 0., ({"x": -.5, "y": 1.}, {"x": .75, "y": -1.}))


def test_recursive_conditions_equal_complete_programs_on_current_and_actual_history_rows():
    value = calibration()
    low, high = binary("ge", value, const(-.2)), binary("lt", value, const(1.5))
    negative = Program("not", children=(binary("ge", Program("feature", name="y"), const(0.)),))
    programs = [low, high, negative, binary("and", low, high), binary("or", high, negative)]
    rows = [observation(-1.5), observation(.3)]
    research = TransferConditionResearch(())
    research.sync_rows(rows)
    budget = SearchBudget(1000)
    for index, row in enumerate(rows):
        for program in programs:
            for history_index in (None, 1):
                actual = row if history_index is None else Observation(
                    row.history[history_index], 0., row.history[:history_index])
                assert research.condition(program, index, budget, history_index) == (
                    program.predict(actual), "observed")


def test_two_questions_share_actual_calibration_execution_and_unchanged_paid_rows(monkeypatch):
    scalar = calibration()
    questions = [binary("ge", scalar, const(-.2)), binary("lt", scalar, const(1.5))]
    executed, predict = [], Program.predict

    def recording(program, row, context=None):
        if program == scalar:
            executed.append(row)
        return predict(program, row, context)

    monkeypatch.setattr(Program, "predict", recording)
    row = observation()
    cold_cost = 0
    for question in questions:
        cold = TransferConditionResearch(())
        cold.sync_rows([row])
        budget = SearchBudget(1000)
        assert cold.condition(question, 0, budget)[1] == "observed"
        cold_cost += budget.spent
    assert len(executed) == 2
    executed.clear()
    shared = TransferConditionResearch(())
    shared.sync_rows([row])
    budget = SearchBudget(1000)
    for question in questions:
        assert shared.condition(question, 0, budget)[1] == "observed"
    assert len(executed) == 1 and budget.spent < cold_cost

    shared.sync_rows([row, observation(2.)])
    assert shared.condition(questions[1], 0, SearchBudget(0)) == (1., "observed")
    assert shared.condition(questions[1], 1, budget) == (0., "observed")
    assert len(executed) == 2
    shared.sync_rows([observation(2.)])
    assert shared.condition(questions[1], 0, budget) == (0., "observed")
    assert len(executed) == 3


def test_eager_boolean_conditions_keep_missing_history_and_nonfinite_results_unknown():
    missing = binary("ge", Program("lag", name="x", lag=1), const(0.))
    program = binary("and", const(0.), missing)
    row = Observation({"x": 1.}, 0.)
    with pytest.raises(MissingHistory):
        program.predict(row)
    research = TransferConditionResearch(())
    research.sync_rows([row])
    assert research.condition(program, 0, SearchBudget(100)) == (None, "missing_observation")
    overflow = binary("mul", const(1e308), const(1e308))
    assert research.condition(overflow, 0, SearchBudget(10)) == (None, "nonfinite")
    enriched = Observation(row.features, row.target, ({"x": 1.},))
    research.sync_rows([enriched])
    assert research.condition(program, 0, SearchBudget(100)) == (program.predict(enriched), "observed")


def test_incomplete_boolean_evaluation_can_resume_but_never_reports_a_budget_stop_as_observed():
    program = binary("lt", calibration(), const(1.5))
    row = observation()
    research = TransferConditionResearch(())
    research.sync_rows([row])
    insufficient = SearchBudget(1)
    assert research.condition(program, 0, insufficient) == (None, "compute_exhausted")
    assert insufficient.spent <= insufficient.limit
    result = research.condition(program, 0, SearchBudget(100))
    assert result == (program.predict(row), "observed")
    assert research.condition(program, 0, SearchBudget(0)) == result

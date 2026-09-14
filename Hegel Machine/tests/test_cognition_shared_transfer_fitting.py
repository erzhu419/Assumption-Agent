"""Share numeric fitting work without identifying distinct response laws."""

from dataclasses import replace

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Definition, MissingHistory, Program
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.shared_transfer_fitting import SharedTransferFitter
from hegel_machine.cognition.unified_research import _fit


def const(value):
    return Program("const", value=value)


def field(name="x"):
    return Program("feature", name=name)


def binary(op, left, right):
    return Program(op, children=(left, right))


def call(identifier, body):
    definition = Definition(identifier, ("value",), body, source_concept_id=f"source_{identifier}")
    return Program("call", name=identifier, definition=definition, children=(field(),))


def rows_for(program, values=(-1., -.5, 0., .5, 1., -.75, .25, .75)):
    rows, history = [], []
    for value in values:
        row = Observation({"x": value}, 0., tuple(history))
        target = program.predict(row)
        rows.append(replace(row, target=target))
        history.append({"x": value, "__target__": target})
    return rows


@pytest.mark.parametrize("kind", ["affine", "quadratic", "guarded_history", "event", "empty", "risk"])
def test_shared_fit_matches_existing_center_scale_ridge_and_executable_loss(kind):
    x = field()
    guarded = Program("if", children=(Program("available", lag=2),
                                      Program("lag", name="x", lag=2), const(0.)))
    event = Program("event_delta", children=(x,), lag=2)
    terms = {"affine": [x], "quadratic": [x, binary("mul", x, x)],
             "guarded_history": [x, guarded], "event": [x, event], "empty": [], "risk": [x]}[kind]
    truth = binary("add", binary("mul", const(2.3), x), const(.7))
    if kind in {"guarded_history", "event"}:
        truth = binary("add", truth, terms[-1])
    rows = rows_for(truth)
    q = .8 if kind == "risk" else .5
    ordinary_budget, shared_budget = SearchBudget(10000), SearchBudget(10000)
    expected = _fit(terms, rows, ordinary_budget, kind, family="T03", q=q)
    actual = SharedTransferFitter().fit(terms, rows, shared_budget, kind, family="T03", q=q)
    assert actual["candidate"].program == expected["candidate"].program
    assert actual["training_loss"] == expected["training_loss"]
    assert actual["family_id"] == expected["family_id"] == "T03"
    assert shared_budget.spent == ordinary_budget.spent


def test_sixteen_gain_versions_share_exact_normalized_solve_but_keep_their_own_calls():
    argument = Program("arg", name="value")
    predicate = binary("ge", argument, const(0.))
    calls = [call(f"gain_{index}", binary("mul", const(float(2 ** index)), predicate)) for index in range(16)]
    truth = binary("mul", const(3.), binary("ge", field(), const(0.)))
    rows = rows_for(truth)
    fitter, budget = SharedTransferFitter(), SearchBudget(10000)
    candidates = []
    for original in calls:
        terms = [original, binary("mul", original, original), field()]
        fitted = fitter.fit(terms, rows, budget, "transfer", view=((), "fit"))
        ordinary = _fit(terms, rows, SearchBudget(10000), "transfer")
        assert fitted["candidate"].program == ordinary["candidate"].program
        assert fitted["training_loss"] == ordinary["training_loss"]
        candidates.append(fitted["candidate"])
    assert len({item.candidate_id for item in candidates}) == 16
    assert [_sources(item.program) for item in candidates] == [[f"source_gain_{index}"] for index in range(16)]
    assert fitter.stats()["linear_solves"] == 1 and fitter.stats()["linear_solve_reuses"] == 15
    assert fitter.stats()["normalized_columns"] == 33
    assert fitter.stats()["normalized_column_reuses"] == 15
    assert budget.spent < 16 * len(rows) * 5
    # Each call body runs once per row, reused by its square and final readout.
    assert fitter.stats()["call_evaluations"] == len(rows) * len(calls)
    assert fitter.stats()["call_cache_hits"] > 0


def test_equal_pilot_columns_do_not_merge_laws_that_differ_on_new_observations():
    argument = Program("arg", name="value")
    original = call("linear", argument)
    alternative = call("outside_shift", Program("if", children=(binary("lt", argument, const(0.)),
        argument, binary("add", argument, const(1.)))))
    truth = binary("mul", const(2.), field())
    rows = rows_for(truth, (-1., -.75, -.5, -.25))
    fitter, budget = SharedTransferFitter(), SearchBudget(10000)
    first = fitter.fit([original], rows, budget, "transfer")
    second = fitter.fit([alternative], rows, budget, "transfer")
    assert first["candidate"].program != second["candidate"].program
    assert fitter.stats()["linear_solves"] == fitter.stats()["linear_solve_reuses"] == 1
    assert _sources(first["candidate"].program) == ["source_linear"]
    assert _sources(second["candidate"].program) == ["source_outside_shift"]
    new = rows_for(truth, (.25, .5, .75))
    accepted = fitter.loss(first["candidate"].program, new, budget, 0., .5, view="validation")
    refuted = fitter.loss(second["candidate"].program, new, budget, 0., .5, view="validation")
    assert accepted < 1e-12 and refuted > 3.
    assert first["candidate"].program.predict(new[0]) != second["candidate"].program.predict(new[0])


def test_appended_fit_evidence_reuses_prefix_columns_without_validation_eviction():
    original = call("squared", binary("mul", Program("arg", name="value"), Program("arg", name="value")))
    terms = [original, binary("mul", original, original)]
    rows = rows_for(binary("add", const(.4), binary("mul", const(1.7), original)))
    fitter, budget = SharedTransferFitter(), SearchBudget(10000)
    first = fitter.fit(terms, rows[:4], budget, "transfer", view="fit")
    first_stats = fitter.stats()
    fitter.loss(first["candidate"].program, rows[4:], budget, 0., .5, view="validation")
    validation_columns = fitter.stats()["column_evaluations"]
    appended = fitter.fit(terms, rows, budget, "transfer", view="fit")
    assert fitter.stats()["column_evaluations"] - validation_columns == len(terms) * 4
    assert fitter.stats()["call_evaluations"] - first_stats["call_evaluations"] == 8
    expected = _fit(terms, rows, SearchBudget(10000), "transfer")
    assert appended["candidate"].program == expected["candidate"].program
    assert appended["training_loss"] == expected["training_loss"]
    before = budget.spent
    repeated = fitter.fit(terms, rows, budget, "transfer", view="fit")
    assert budget.spent == before
    assert repeated["candidate"].candidate_id != appended["candidate"].candidate_id


def test_changed_target_field_and_history_reset_only_their_actual_view():
    fitter, budget = SharedTransferFitter(), SearchBudget(10000)
    first_rows = rows_for(binary("mul", const(2.), field()))
    first = fitter.fit([field()], first_rows, budget, "transfer", view="fit")
    changed_target = [replace(row, target=4. * row.features["x"]) for row in first_rows]
    changed = fitter.fit([field()], changed_target, budget, "transfer", view="fit")
    assert fitter.stats()["view_resets"] == 1
    assert changed["candidate"].program != first["candidate"].program
    assert changed["candidate"].program == _fit([field()], changed_target, SearchBudget(10000), "transfer")["candidate"].program
    altered_history = [replace(row, history=({"x": 9., "__target__": 7.},)) for row in changed_target]
    fitter.fit([field()], altered_history, budget, "transfer", view="fit")
    assert fitter.stats()["view_resets"] == 2
    altered_fields = [replace(row, features={"x": row.features["x"] + 1.}) for row in altered_history]
    fitter.fit([field()], altered_fields, budget, "transfer", view="fit")
    assert fitter.stats()["view_resets"] == 3


def test_history_fallback_risk_and_partial_budget_never_reuse_the_wrong_loss():
    fitter, budget = SharedTransferFitter(), SearchBudget(10000)
    lag = Program("lag", name="x", lag=2)
    rows = rows_for(field(), (.2, .4, .6, .8))
    for fallback, q in ((0., .5), (2., .5), (2., .8), (0., .8)):
        actual = fitter.loss(lag, rows, budget, fallback, q)
        assert actual == loss(lag, rows, fallback, q)
    with pytest.raises(MissingHistory):
        fitter.fit([lag], rows, budget, "unavailable")
    partial, tiny = SharedTransferFitter(), SearchBudget(2)
    assert partial.fit([field()], rows, tiny, "partial") is None
    assert tiny.spent == 2
    enough = SearchBudget(10000)
    completed = partial.fit([field()], rows, enough, "completed")
    assert completed["candidate"].program == _fit([field()], rows, SearchBudget(10000), "ordinary")["candidate"].program
    assert tiny.spent + enough.spent == len(rows) * 3

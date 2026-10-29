"""Local startup repair preserves learned dynamics under control-only replay."""

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition import state_search


def threshold(value):
    return Program('ge', children=(Program('feature', name='u'),
                                  Program('const', value=value)))


def fixture():
    mature = Program('if', children=(Program('ge', children=(
        Program('lag', name='__target__', lag=1), Program('const', value=.5))),
        threshold(-.7), threshold(.7)))
    body = Program('if', children=(Program('available', lag=1), mature, threshold(.2)))
    program = Program('state_replay', children=(body,))
    rows = []
    for start in (-.9, .3, .5, .8):
        history, active = [], False
        for index, value in enumerate((start, .2, .2, -.9, .9, .2)):
            active = value >= .65 if index == 0 else value >= (-.7 if active else .7)
            rows.append(Observation({'u': value}, float(active), tuple(history)))
            history.append({'u': value})
    return program, rows


def test_closed_startup_repair_preserves_guard_and_mature_rule_and_removes_propagation():
    old, rows = fixture()
    assert loss(old, rows, 0., .5) == 6 / len(rows)
    budget = SearchBudget(1000)
    candidates = state_search.startup_revision_candidates([old], rows, budget=budget)
    assert candidates
    revised = candidates[0].program
    assert revised.op == 'state_replay'
    assert revised.children[0].children[0] is old.children[0].children[0]
    assert revised.children[0].children[1] is old.children[0].children[1]
    assert revised.children[0].children[2] != old.children[0].children[2]
    assert loss(revised, rows, 0., .5) == 0.
    assert revised.predict(Observation({'u': .2}, 0., ({'u': .8},))) == 1.
    assert revised.predict(Observation({'u': .2}, 0., ({'u': -.9},))) == 0.
    assert revised.predict(Observation({'u': .3}, 0.)) == 0.
    # Caller-provided measured targets do not become the replay's state.
    contaminated = [Observation(row.features, row.target,
        tuple({**point, '__target__': 123.} for point in row.history)) for row in rows]
    assert loss(revised, contaminated, 0., .5) == 0.


def test_bare_and_replay_forms_are_distinct_but_duplicate_inputs_share_work():
    closed, rows = fixture()
    bare = closed.children[0]
    clean_budget, duplicate_budget = SearchBudget(1000), SearchBudget(1000)
    clean = state_search.startup_revision_candidates([bare, closed], rows,
        budget=clean_budget, max_candidates=2)
    repeated = state_search.startup_revision_candidates([bare, closed, bare, closed], rows,
        budget=duplicate_budget, max_candidates=2)
    assert [candidate.program for candidate in repeated] == [candidate.program for candidate in clean]
    assert {candidate.program.op for candidate in clean} == {'if', 'state_replay'}
    assert clean_budget.spent == duplicate_budget.spent
    assert clean[0].program.children[1] is bare.children[1]
    assert clean[1].program.children[0].children[1] is bare.children[1]


def test_bridge_scores_complete_closed_trajectories_and_reports_only_training_work():
    old, rows = fixture()
    budget = SearchBudget(1000)
    claims = state_search.closed_startup_claims([old], rows, budget=budget)
    assert claims and claims[0]['training_loss'] == 0.
    for claim in claims:
        program = claim['candidate'].program
        assert claim['training_loss'] == loss(program, rows, 0., .5)
        assert claim['research_program'] is None
        receipt = claim['state_research']
        assert receipt['kind'] == 'startup_revision'
        assert receipt['fit_rows'] == len(rows) and receipt['startup_fit_rows'] == 4
        assert receipt['mature_transition_preserved'] and receipt['guard_preserved']
        assert not receipt['independent_confirmation']
    assert budget.spent == claims[0]['state_research']['shared_search_compute'] + len(rows) * len(claims)


def test_closed_comparison_keeps_mature_errors_instead_of_reporting_startup_fit_as_total():
    old, rows = fixture()
    guard, _, startup = old.children[0].children
    wrong = Program('state_replay', children=(Program('if', children=(guard,
        Program('const', value=1.), startup)),))
    claim = state_search.closed_startup_claims([wrong], rows, budget=SearchBudget(1000))[0]
    revised = claim['candidate'].program
    assert all(revised.predict(row) == row.target for row in rows if not row.history)
    assert claim['training_loss'] > 0.
    assert claim['training_loss'] == loss(revised, rows, 0., .5)


def test_already_correct_startup_does_not_refit_or_emit_candidates():
    old, rows = fixture()
    corrected = state_search.startup_revision_candidates([old], rows,
        budget=SearchBudget(1000))[0].program
    budget = SearchBudget(1000)
    assert state_search.closed_startup_claims([corrected], rows, budget=budget) == []
    assert budget.spent == len(rows) + 4  # Grouping and existing startup predictions only.


def test_bridge_reserves_all_closed_comparisons_before_spending_search(monkeypatch):
    old, rows = fixture()
    required = len(rows) * 7  # Six output slots plus the initial grouping pass.
    monkeypatch.setattr(state_search, 'startup_revision_candidates', lambda *args, **kwargs:
        pytest.fail('unfunded candidate comparison must not begin fitting'))
    budget = SearchBudget(required - 1)
    assert state_search.closed_startup_claims([old], rows, budget=budget) == []
    assert budget.spent == 0 and budget.last_denied_request['required_compute'] == required


def test_interrupted_local_fit_preserves_its_actual_spending(monkeypatch):
    old, rows = fixture()
    def interrupted(programs, observations, *, budget, **kwargs):
        budget.spend(11)
        raise RuntimeError('interrupted startup research')
    monkeypatch.setattr(state_search, 'startup_revision_candidates', interrupted)
    budget = SearchBudget(1000)
    with pytest.raises(RuntimeError, match='interrupted startup research'):
        state_search.closed_startup_claims([old], rows, budget=budget)
    assert budget.spent == 11

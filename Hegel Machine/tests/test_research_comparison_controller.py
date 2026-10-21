"""A separate design pool changes an actual planned horizon, not adoption."""

from hegel_machine.cognition.controller import _research_comparison
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.selection import representative_probe, screen_candidates


def test_candidate_outside_task_shortlist_can_change_next_representative_plan():
    x = Program('feature', name='x')
    duplicate_behavior = Program('add', children=(x, Program('const', value=0.)))
    history = Program('if', children=(Program('available', lag=4),
                                    Program('lag', name='x', lag=4), x))
    pool = [Candidate('current', x, 'refit'), Candidate('expanded_current', duplicate_behavior, 'composition'),
            Candidate('historical_alternative', history, 'composition')]
    observed = [Observation({'x': float(i)}, float(i), tuple({'x': -10.} for _ in range(4)))
                for i in range(3)]
    assessed = []
    task = screen_candidates(pool, observed, [], fallback=0., q=.5, budget=SearchBudget(100),
        complexity_weight=.001, maximum=2, assessment_sink=assessed)
    task_ids = [row.candidate.candidate_id for row in task]
    assert task_ids == ['current', 'expanded_current']
    spec = TaskSpec('design_fixture', 'Public controls', ('x',), (), (),
                    feature_bounds={'x': (-1., 1.)}, max_probe_length=12)
    budget = SearchBudget(24)
    candidates, details = _research_comparison(assessed, [row.candidate for row in task], spec, observed,
        seed=11, turn=0, remaining=40, maximum=2, fallback=0., budget=budget)
    assert 'historical_alternative' in {candidate.candidate_id for candidate in candidates}
    assert details['task_candidate_ids'] == task_ids
    assert details['adoption_changed'] is False and budget.spent == 24
    assert [row.candidate.candidate_id for row in task] == task_ids
    old = representative_probe(spec, seed=11, index=0, remaining=40,
                               candidates=[row.candidate for row in task])
    new = representative_probe(spec, seed=11, index=0, remaining=40, candidates=candidates)
    assert old.cost == 8 and new.cost == 12


def test_public_forecast_budget_exhaustion_retains_unknown_without_free_predictions():
    x = Program('feature', name='x')
    observed = [Observation({'x': 1.}, 1.)]
    assessed = []
    task = screen_candidates([Candidate('x', x, 'refit'),
        Candidate('square', Program('mul', children=(x, x)), 'composition')], observed, [],
        fallback=0., q=.5, budget=SearchBudget(2), complexity_weight=.001,
        maximum=1, assessment_sink=assessed)
    spec = TaskSpec('design_fixture', 'Public controls', ('x',), (), (), feature_bounds={'x': (-1., 1.)})
    budget = SearchBudget(8)
    _, details = _research_comparison(assessed, [row.candidate for row in task], spec, observed,
        seed=11, turn=0, remaining=20, maximum=2, fallback=0., budget=budget)
    assert budget.spent == details['compute_cost'] == 8
    assert details['unknown_prediction_ids'] == ['square']


def test_finite_public_panel_uses_the_same_remaining_probe_as_actual_design():
    x = Program('feature', name='x')
    observed = [Observation({'x': 1.}, 1.)]
    assessed = []
    task = screen_candidates([Candidate('x', x, 'refit')], observed, [],
        fallback=0., q=.5, budget=SearchBudget(1), complexity_weight=.001,
        maximum=1, assessment_sink=assessed)
    used = Probe('already_used', ({'x': 1.},))
    remaining = Probe('unmeasured', ({'x': 2.},))
    spec = TaskSpec('finite_probe_fixture', 'Finite actions', ('x',), (), (used, remaining))
    _, details = _research_comparison(assessed, [row.candidate for row in task], spec, observed,
        seed=11, turn=0, remaining=20, maximum=2, fallback=0., budget=SearchBudget(1), used=[used])
    assert details['panel_probe']['probe_id'] == 'unmeasured'

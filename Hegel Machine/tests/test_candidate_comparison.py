"""The shared adoption calculation uses frozen candidates and actual rows."""

import pytest

from hegel_machine.cognition.candidate_comparison import compare_candidate_adoption
from hegel_machine.cognition.evaluation import predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.selection import rank_candidates


SPEC = TaskSpec('frozen_comparison', 'Predict the public response', ('x', 'z'), (), ())
CONFIG = ResearchConfig()


def compare(candidates, observations, fresh, *, fallback=0.):
    budget = SearchBudget(len(candidates) * (len(observations) + 2 * len(fresh)))
    assessed = rank_candidates(candidates, observations, list(fresh), fresh=fresh,
        fresh_weight=.5, fallback=fallback, q=.5, budget=budget,
        complexity_weight=CONFIG.complexity_weight)
    return compare_candidate_adoption(assessed, candidates[0], observations, [], fresh,
        fallback=fallback, spec=SPEC, config=CONFIG)


def test_fresh_startup_errors_adopt_an_executable_guard_and_change_the_next_prediction():
    lag = Program('lag', name='x', lag=1)
    guarded = Program('if', children=(Program('available', lag=1), lag,
                                    Program('feature', name='x')))
    candidates = [Candidate('original_parent', lag, 'reuse'),
                  Candidate('frozen_guard', guarded, 'composition')]
    observations = [Observation({'x': value}, value, ({'x': value},))
                    for value in (1., 2., 3.)]
    fresh = [Observation({'x': 2.}, 2.),
             Observation({'x': 7.}, .75, ({'x': .75},))]
    decision = compare(candidates, observations, fresh)
    assert decision.current.candidate == candidates[0]
    assert decision.current.fresh_loss == 2.
    assert decision.best.candidate == candidates[1]
    assert decision.best.fresh_loss == 0.
    assert decision.adopt and decision.support and decision.fresh_gain == 2.
    continuation = Observation({'x': 3.}, 999.)
    assert predict(decision.best.candidate.program, continuation, 0.) == 3.
    assert predict(decision.current.candidate.program, continuation, 0.) == 0.


def test_acquired_counterexample_can_be_repaired_while_the_fresh_predictions_tie():
    old, new = Program('feature', name='x'), Program('feature', name='z')
    candidates = [Candidate('original_parent', old, 'reuse'),
                  Candidate('frozen_coordinate', new, 'composition')]
    observations = [Observation({'x': 0., 'z': value}, value) for value in (-1., 0., 1.)]
    fresh = [Observation({'x': 0., 'z': 0.}, 0.)]
    decision = compare(candidates, observations, fresh)
    assert decision.best.candidate == candidates[1]
    assert decision.counterexample_gain == pytest.approx(2 / 3)
    assert decision.fresh_gain == 0.
    assert decision.adopt


def test_fitting_rows_alone_cannot_be_passed_as_an_acquired_fresh_comparison():
    candidate = Candidate('parent', Program('const', value=0.), 'reuse')
    observations = [Observation({'x': 1.}, 1.)]
    assessed = rank_candidates([candidate], observations, [], fallback=0., q=.5,
        budget=SearchBudget(1), complexity_weight=CONFIG.complexity_weight)
    with pytest.raises(ValueError, match='acquired fresh comparison'):
        compare_candidate_adoption(assessed, candidate, observations, [], [],
            fallback=0., spec=SPEC, config=CONFIG)

"""Cold response terms can identify previously unenumerated temporal support."""

from dataclasses import replace
from itertools import combinations_with_replacement
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.temporal_coordinates import propose_temporal_coordinates


SPEC = TaskSpec('observed', 'Predict observed response', ('drive', 'load'), (), ())


def polynomial_basis():
    atoms = [Program('feature', name=name) for name in SPEC.feature_names]
    result = []
    for degree in range(1, 5):
        for factors in combinations_with_replacement(atoms, degree):
            term = factors[0]
            for factor in factors[1:]:
                term = Program('mul', children=(term, factor))
            result.append(term)
    return result


def rows(width, seed, length=24):
    rng, history, result = Random(seed), [], []
    values = []
    for _ in range(length):
        features = {'drive': rng.uniform(-1.5, 1.5), 'load': rng.uniform(-1.5, 1.5)}
        values.append(.3 + (features['drive'] ** 2 - .7 * features['load']) ** 2)
        row = Observation(features, .2 + sum(values[-width:]), tuple(history))
        result.append(row)
        history.append(features)
    return result


@pytest.mark.parametrize('width', [5, 9, 11])
def test_joint_coefficients_and_startup_offset_are_learned_with_data_selected_width(width):
    work = SearchBudget(18000)
    training = rows(width, 13) + rows(width, 29)
    claims = propose_temporal_coordinates(SPEC, training, work, basis=polynomial_basis())
    assert claims
    best = min(claims, key=lambda claim: loss(claim['candidate'].program, rows(width, 37), 0., .5))
    assert loss(best['candidate'].program, rows(width, 37), 0., .5) < 1e-12
    step = best['research_program']['selection_steps'][0]
    assert step['width'] == width and step['history_bound'] == 23
    assert all(2 <= trial['width'] <= 23 for trial in step['width_trials'])
    assert best['candidate'].parents == ()
    assert best['candidate'].candidate_id.startswith('temporal_coordinate:')
    assert best['research_program']['status'] == 'requires_fresh_validation'
    assert 0 < work.spent <= work.limit


def test_unfunded_search_and_absent_history_do_not_create_claims():
    training = rows(9, 13) + rows(9, 29)
    work = SearchBudget(100)
    assert propose_temporal_coordinates(SPEC, training, work, basis=polynomial_basis()) == []
    assert work.spent <= work.limit
    work = SearchBudget(18000)
    assert propose_temporal_coordinates(SPEC, [replace(row, history=()) for row in training],
        work, basis=polynomial_basis()) == []
    assert work.spent == 0


def test_fresh_validation_can_choose_simpler_response_than_noisy_training_prefers():
    rng = Random(43)
    training = [replace(row, target=row.target + rng.gauss(0., .01))
        for row in rows(7, 13) + rows(7, 29)]
    budget = SearchBudget(18000)
    claims = propose_temporal_coordinates(SPEC, training, budget, basis=polynomial_basis())
    fresh = rows(7, 37)
    selected = min(claims, key=lambda claim: loss(claim['candidate'].program, fresh, 0., .5))
    training_best = min(claims, key=lambda claim: claim['training_loss'])
    assert loss(selected['candidate'].program, fresh, 0., .5) < 5e-5
    assert selected['training_loss'] > training_best['training_loss']
    assert selected['candidate'].program.size < training_best['candidate'].program.size
    assert budget.spent <= budget.limit


def test_paid_development_case_searches_beyond_the_best_coarse_basin():
    # Reproduce only the first three purchased trajectories of development
    # root02/window5. No terminal observations or terminal means enter the fit.
    apparatus, noise = Random(9914002), Random(10316023 + 49979687)
    a, b = apparatus.uniform(.65, 1.35), apparatus.uniform(.65, 1.35)
    scale, offset = apparatus.uniform(.7, 1.4), apparatus.uniform(-.2, .2)
    trajectories = []
    for serial in range(3):
        rng, history, responses, observations = Random(10316023 + 1009 * serial), [], [], []
        for _ in range(24):
            features = {name: rng.uniform(-1.5, 1.5) for name in SPEC.feature_names}
            inner = a * features['drive'] ** 2 + .35 * b * features['load'] ** 2
            responses.append(offset + scale * inner ** 2)
            target = sum(responses[-5:]) + noise.gauss(0., .01)
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, '__target__': target})
        trajectories.append(observations)
    budget = SearchBudget(18000)
    claims = propose_temporal_coordinates(SPEC, trajectories[0] + trajectories[1], budget,
        basis=polynomial_basis())
    step = claims[0]['research_program']['selection_steps'][0]
    trials = {trial['width']: trial['loss'] for trial in step['width_trials']}
    assert trials[23] < trials[4]  # This misleading coarse basin caused the failure.
    assert step['width'] == 5 and trials[5] < 1e-4
    assert min(loss(claim['candidate'].program, trajectories[2], 0., .5) for claim in claims) < .001
    assert len({claim['candidate'].program.size for claim in claims}) > 1
    assert budget.spent <= budget.limit

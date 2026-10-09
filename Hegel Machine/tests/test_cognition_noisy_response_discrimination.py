"""Frozen noisy-response rivals buy informative trajectories before adoption."""

from copy import deepcopy
from random import Random
from statistics import mean

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.mechanism_revision import propose_response_refits
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.noisy_response_experiments import (
    assess_noisy_response_probe, plan_noisy_response_probe,
)
from hegel_machine.cognition.noise_replication import compare_response_refits


def _entry(name, gain):
    basis = Program('window_sum', lag=3, children=(Program('feature', name='drive'),))
    program = Program('mul', children=(Program('const', value=gain), basis))
    return {'candidate': Candidate(name, program, 'acquired_response'), 'kind': 'response_refit'}


class ResponseLab:
    """Only public commands and independently noisy measured responses are exposed."""

    channels = ()

    def __init__(self, gain, *, seed=971003, noise=.015):
        self.spec = TaskSpec('measured_response', 'Predict the measured response', ('drive',), (), (),
            feature_bounds={'drive': (-2., 2.)}, max_probe_length=64)
        self._gain, self._noise, self._rng = gain, noise, Random(seed)
        self.acquisitions = []

    def random_probe(self, seed, *, length=24, name='random'):
        rng = Random(seed)
        return Probe(name, tuple({'drive': rng.uniform(-2., 2.)} for _ in range(length)))

    def acquire(self, probe):
        history, rows, inputs = [], [], []
        for point in probe.sequence:
            inputs.append(point['drive'])
            response = self._gain * sum(inputs[-3:]) + self._rng.gauss(0., self._noise)
            rows.append(Observation(dict(point), response, tuple(history)))
            history.append({**point, '__target__': response})
        self.acquisitions.append((probe, rows))
        return MeasurementBatch(rows, len(rows), 0, ())


def _forecast(program, probe):
    history, predictions = [], []
    for point in probe.sequence:
        predictions.append(program.predict(Observation(dict(point), 0., tuple(history))))
        history.append(dict(point))
    return predictions


def test_complete_trajectory_amplifies_frozen_response_disagreement_without_observing():
    lab, entries = ResponseLab(1.), [_entry('old_response', 1.), _entry('refitted_response', 1.035)]
    budget = SearchBudget(6000)
    planned = plan_noisy_response_probe(lab.spec, entries, budget, length=24, seed=973013,
        observation_remaining=72, reserve_rows=48)
    probe = planned['probe']
    assert probe is not None and probe.cost == 24 and not lab.acquisitions
    assert 0 < planned['compute_cost'] == budget.spent <= budget.limit
    assert all(set(point) == set(lab.spec.feature_names) and -2. <= point['drive'] <= 2.
               for point in probe.sequence)
    forecasts = [_forecast(item['candidate'].program, probe) for item in entries]
    disagreement = mean((left - right) ** 2 for left, right in zip(*forecasts))
    assert planned['predicted_disagreement'] == pytest.approx(disagreement)
    ordinary = lab.random_probe(973013, length=24)
    ordinary_forecasts = [_forecast(item['candidate'].program, ordinary) for item in entries]
    assert disagreement > 2 * mean((left - right) ** 2
                                   for left, right in zip(*ordinary_forecasts))


@pytest.fixture(scope='module')
def measured_rivals():
    parent = _entry('prior_response', 1.)
    fitting_lab = ResponseLab(1.035, seed=977001)
    fitted_rows = fitting_lab.acquire(fitting_lab.random_probe(977007, length=48, name='training')).observations
    fit_budget = SearchBudget(4000)
    proposed = propose_response_refits(fitting_lab.spec, fitted_rows, fit_budget,
                                       parent=parent, max_candidates=1)[0]
    frozen = deepcopy([parent, proposed])
    research = []
    for position, gain in enumerate((1., 1.035)):
        lab = ResponseLab(gain, seed=977011 + position * 1009)
        budget = SearchBudget(6000)
        planned = plan_noisy_response_probe(lab.spec, [parent, proposed], budget,
            serial=position, length=24, seed=977021 + position * 1009,
            observation_remaining=72, reserve_rows=48)
        batch = lab.acquire(planned['probe'])
        compared = assess_noisy_response_probe(parent, proposed, batch.observations, budget)
        assert not {id(row) for row in fitted_rows} & {id(row) for row in batch.observations}
        # A selection experiment is not the final adoption acquisition.
        confirmation = lab.acquire(lab.random_probe(977027 + position * 1009,
            length=48, name=f'confirm_{position}'))
        confirmed = assess_noisy_response_probe(parent, proposed, confirmation.observations, budget)
        research.append({'gain': gain, 'lab': lab, 'planned': planned,
                         'compared': compared, 'confirmed': confirmed,
                         'confirmation': confirmation, 'compute_cost': budget.spent})
    assert [parent, proposed] == frozen
    # Both real research episodes finish before noiseless terminal targets are acquired.
    for position, item in enumerate(research):
        terminal_lab = ResponseLab(item['gain'], noise=0.)
        terminal = terminal_lab.acquire(terminal_lab.random_probe(977031 + position * 1009,
            length=64, name='terminal')).observations
        item['parent_mean_loss'] = loss(parent['candidate'].program, terminal, 0., .5)
        item['candidate_mean_loss'] = loss(proposed['candidate'].program, terminal, 0., .5)
    return parent, proposed, fitted_rows, fit_budget.spent, research


@pytest.mark.parametrize('position', (0, 1))
def test_paid_discrimination_protects_a_better_old_response_and_confirms_real_change(
        measured_rivals, position, record_property):
    parent, proposed, fitted_rows, fitting_cost, cases = measured_rivals
    item = cases[position]
    compared, confirmed, lab = item['compared'], item['confirmed'], item['lab']
    assert compared['supported'] == confirmed['supported'] == bool(position)
    assert compared['refuted'] == confirmed['refuted'] == (not bool(position))
    if position:
        assert compared['lower_bound'] > 1e-7 and confirmed['lower_bound'] > 1e-7
        assert item['candidate_mean_loss'] < item['parent_mean_loss']
    else:
        assert compared['upper_bound'] < -1e-7 and confirmed['upper_bound'] < -1e-7
        assert item['parent_mean_loss'] < item['candidate_mean_loss']
    assert len(lab.acquisitions) == 2
    assert lab.acquisitions[0][0].sequence != lab.acquisitions[1][0].sequence
    assert sum(len(rows) for _, rows in lab.acquisitions) == 72
    assert all(row not in fitted_rows for _, rows in lab.acquisitions for row in rows)
    assert item['compute_cost'] == (item['planned']['compute_cost']
        + compared['compute_cost'] + confirmed['compute_cost']) > 0
    assert fitting_cost > 0
    assert 'confirmed' not in proposed and parent['candidate'].program != proposed['candidate'].program
    record_property('measured_response_comparison', {
        'selection': compared, 'confirmation': confirmed,
        'observation_cost': 72, 'compute_cost': item['compute_cost'],
        'parent_mean_loss': item['parent_mean_loss'], 'candidate_mean_loss': item['candidate_mean_loss']})


@pytest.mark.parametrize('entries, compute, remaining', [([], 6000, 72),
    ([_entry('only_response', 1.)], 6000, 72),
    ([_entry('old_response', 1.), _entry('same_response', 1.)], 6000, 72),
    ([_entry('old_response', 1.), _entry('refitted_response', 1.035)], 0, 72),
    ([_entry('old_response', 1.), _entry('refitted_response', 1.035)], 6000, 23)])
def test_unfunded_or_absent_response_competition_does_not_buy_an_experiment(entries, compute, remaining):
    lab, budget = ResponseLab(1.), SearchBudget(compute)
    planned = plan_noisy_response_probe(lab.spec, entries, budget, length=24,
        observation_remaining=remaining, reserve_rows=48)
    assert planned['probe'] is None and not lab.acquisitions
    assert planned['compute_cost'] == budget.spent == planned['evaluated_probes'] == 0
    assert 'gain' not in planned


def test_targeted_gain_cannot_replace_a_response_refuted_by_an_ordinary_paid_repeat_pair():
    class EndpointResponseLab(ResponseLab):
        def acquire(self, probe):
            history, rows, responses = [], [], []
            for point in probe.sequence:
                drive = point['drive']
                responses.append(drive * (1.08 if abs(drive) > 1.95 else 1.))
                target = sum(responses[-3:]) + self._rng.gauss(0., self._noise)
                rows.append(Observation(dict(point), target, tuple(history)))
                history.append({**point, '__target__': target})
            self.acquisitions.append((probe, rows))
            return MeasurementBatch(rows, len(rows), 0, ())

    parent, lab = _entry('acquired_response', 1.), EndpointResponseLab(1., seed=979001)
    training = lab.acquire(Probe('measured_extremes', tuple(
        {'drive': 2. if step // 12 % 2 else -2.} for step in range(48)))).observations
    fitting_budget = SearchBudget(4000)
    proposed = propose_response_refits(lab.spec, training, fitting_budget,
                                       parent=parent, max_candidates=1)[0]
    frozen, budget = deepcopy([parent, proposed]), SearchBudget(6000)
    plan = plan_noisy_response_probe(lab.spec, [parent, proposed], budget,
        length=24, seed=979007, observation_remaining=120, reserve_rows=96)
    targeted = lab.acquire(plan['probe']).observations
    comparison = assess_noisy_response_probe(parent, proposed, targeted, budget)
    assert comparison['supported'] and comparison['lower_bound'] > 1e-7
    ordinary_probe = lab.random_probe(979017, length=48, name='ordinary_confirmation')
    original = lab.acquire(ordinary_probe).observations
    repeated = lab.acquire(Probe('repeat_confirmation', ordinary_probe.sequence)).observations
    confirmed = compare_response_refits(parent['candidate'].program, [proposed], original, repeated, budget)
    assert confirmed['selected'] is None
    event = confirmed['comparisons'][0]
    assert event['gain'] + event['margin'] < -1e-7 and not event['supported']
    assert [parent, proposed] == frozen
    assert len(lab.acquisitions) == 4
    assert sum(len(rows) for _, rows in lab.acquisitions) == 48 + 24 + 2 * 48
    assert lab.acquisitions[-2][0].sequence == lab.acquisitions[-1][0].sequence
    assert lab.acquisitions[-2][0].sequence != plan['probe'].sequence
    assert all(row not in training for row in [*targeted, *original, *repeated])
    assert budget.spent == plan['compute_cost'] + comparison['compute_cost'] + confirmed['compute_cost']

"""A real scope counterexample remains binding until its own noisy repeat."""

from copy import deepcopy
from dataclasses import replace
from random import Random

import pytest

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
from test_cognition_completed_challenges import CONFIG, challenged_session


class NoisyContinuationLab(UnknownTaskLab):
    def acquire(self, probe, channels=()):
        batch = super().acquire(probe, channels)
        rng = Random(f'{self.seed}:{probe.probe_id}:response_noise')
        rows, history = [], []
        for row in batch.observations:
            target = row.target + rng.gauss(0., .002)
            rows.append(Observation(row.features, target, tuple(history)))
            history.append({**row.features, '__target__': target})
        return replace(batch, observations=rows)


def _resume(stream, config):
    state = stream['queued']
    previous = state['results'][0]['research']
    offer, = state['pending_revisits']
    lab = NoisyContinuationLab(8, seed=11811)
    report = UnifiedResearchAgent(config, library=KnowledgeLibrary.from_dict(state['knowledge']),
        measurement_budget=768).run(lab, previous=previous,
            continuation_candidate=deepcopy(offer['candidate']))
    for resource in ('observation_cost', 'measurement_cost'):
        assert report['visit_costs'][resource] == lab.costs[resource] - previous['costs'][resource]
        assert report['costs'][resource] == lab.costs[resource]
    assert report['costs']['compute_cost'] <= config.compute_budget
    assert report['costs']['observation_cost'] <= config.observation_budget
    return report, previous


@pytest.fixture(scope='module')
def noisy_continuation(challenged_session):
    report, previous = _resume(challenged_session, CONFIG)
    return report, previous


def test_noisy_scope_error_requires_repetition_of_its_own_complete_history(noisy_continuation):
    report, previous = noisy_continuation
    events = report['actions'][len(previous['actions']):]
    scope = next(action for action in events if action['action'] == 'compare_scope_noise_replication')
    assert scope['noise_supported'] and not scope['repeatable_error']
    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    original, repeated = batches[scope['source_probe_id']], batches[scope['probe_id']]
    assert original['probe']['sequence'] == repeated['probe']['sequence']
    assert original['channels'] == repeated['channels']
    assert len(original['observations']) == len(repeated['observations']) == scope['matched_rows']
    for first, second in zip(original['observations'], repeated['observations']):
        assert first['features'] == second['features']
        assert [{k:v for k,v in row.items() if k != '__target__'} for row in first['history']] == [
            {k:v for k,v in row.items() if k != '__target__'} for row in second['history']]
    supported = next(action for action in reversed(events)
                     if action['action'] == 'validate_continued_program')
    assert supported['accepted'] and report['status'] == 'completed'
    assert supported['probe_id'] not in {scope['source_probe_id'], scope['probe_id']}
    assert supported['noise_assessment']['replication']['noise_supported']
    assert report['program'] != previous['program']
    acquired = next(action for action in events if action['action'] == 'acquire'
                    and action['probe_id'] == scope['probe_id'])
    assert acquired['purpose'] == 'retest_scope_counterexample_noise'
    assert acquired['observation_cost'] == scope['matched_rows']
    assert scope['compute_cost'] == 4 * scope['matched_rows']


def test_unfunded_scope_repeat_cannot_borrow_a_different_trajectorys_noise_support(
        noisy_continuation, challenged_session):
    full, _ = noisy_continuation
    scope = next(action for action in full['actions']
                 if action['action'] == 'compare_scope_noise_replication')
    config = replace(CONFIG, observation_budget=full['costs']['observation_cost'] - 1)
    report, previous = _resume(challenged_session, config)
    assert report['status'] == 'unresolved' and not report['concepts_added']
    events = report['actions'][len(previous['actions']):]
    validation = next(action for action in reversed(events)
                      if action['action'] == 'validate_continued_program')
    assert validation['noise_assessment']['reason'] == 'scope_noise_experiment_unfunded'
    assert validation['noise_assessment']['replication']['noise_supported']
    assert not any(action['action'] == 'compare_scope_noise_replication' for action in events)
    assert not any(batch['probe']['probe_id'] == scope['probe_id'] for batch in report['batches'])

"""A transferred mechanism can disappear and return without destroying its source."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_nested_relation_revisions import composition
from test_cognition_program_action_followup import NestedResponseLab


CONFIG = ResearchConfig(observation_budget=240, compute_budget=240000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)


class ChangingMechanismLab(NestedResponseLab):
    def __init__(self, event_gain, episode):
        super().__init__(.6)
        self.event_gain = event_gain
        self.spec = replace(self.spec, task_id=episode)

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, flows, combined = [], [], [], []
        for point in probe.sequence:
            flow = ((point['f0'] - point['f1']) * (point['q1'] - point['q0'])
                    if abs(point['x0'] - point['x1']) < self.radius else 0.)
            event = flow - flows[-1] if flows else 0.
            flows.append(flow)
            combined.append(1.3 * (point['q0'] + point['q1']) + self.event_gain * event)
            target = sum(combined[-3:])
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, '__target__': target})
        self.acquisitions.append({'probe': probe, 'observations': rows})
        return MeasurementBatch(rows, len(rows), 0, ())


def restored(library):
    return KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))


def source_knowledge():
    laboratory = ChangingMechanismLab(-.7, 'source_episode')
    paid = laboratory.sample(31, 64).observations
    source_program = composition(.6).expanded()
    assert loss(source_program, paid, 0., .5) < 1e-24
    library = KnowledgeLibrary()
    concept_id, added = library.learn(Candidate('observed_composition', source_program,
        'observed_response'), laboratory.spec, paid)
    assert added
    return restored(library), concept_id


def run_episode(library, gain, episode):
    laboratory = ChangingMechanismLab(gain, episode)
    report = UnifiedResearchAgent(CONFIG, library=library).run(laboratory)
    purchased = sum(len(batch['observations']) for batch in laboratory.acquisitions)
    assert report['costs']['observation_cost'] == purchased <= CONFIG.observation_budget
    assert report['costs']['measurement_cost'] == 0
    assert report['costs']['compute_cost'] <= CONFIG.compute_budget
    # These evaluation responses are obtained only after research has ended.
    evaluation = [row for seed, length in ((991, 1), (992, 3), (993, 47))
                  for row in laboratory.sample(seed, length).observations]
    error = loss(Program.from_dict(report['program']), evaluation, report['fallback'], .5)
    return report, laboratory, error


@pytest.fixture(scope='module')
def mechanism_history():
    library, source_id = source_knowledge()
    before = deepcopy(library.to_dict())
    removed, removed_lab, removed_error = run_episode(library, 0., 'changed_episode')
    after_removal = deepcopy(library.to_dict())
    returned_library = restored(library)
    returned, returned_lab, returned_error = run_episode(returned_library, -.7, 'returned_episode')
    necessary_library = KnowledgeLibrary.from_dict(deepcopy(before))
    necessary, necessary_lab, necessary_error = run_episode(
        necessary_library, -.35, 'other_changed_episode')
    return locals()


def _confirmed_response(report, laboratory):
    actions = report['actions']
    confirmed = next(event for event in reversed(actions)
                     if event['action'] == 'validate_frozen_program' and event['accepted'])
    confirmation_index = actions.index(confirmed)
    acquired = next(event for event in reversed(actions[:confirmation_index])
                    if event['action'] == 'acquire' and event['purpose'] == 'confirm_frozen_program')
    purchase_index = actions.index(acquired)
    batch = next(batch for batch in laboratory.acquisitions
                 if batch['probe'].probe_id == acquired['probe_id'])
    earlier = {event['probe_id'] for event in actions[:purchase_index] if event['action'] == 'acquire'}
    assert earlier and acquired['probe_id'] not in earlier
    assert all(batch['probe'].sequence != previous['probe'].sequence
               for previous in laboratory.acquisitions if previous['probe'].probe_id in earlier)
    assert acquired['observation_cost'] == confirmed['observation_rows'] == len(batch['observations'])
    assert confirmed['program'] == report['program']
    assert not any(event['action'].startswith(('propose_', 'select_', 'execute_'))
                   for event in actions[purchase_index + 1:confirmation_index])
    actual = loss(Program.from_dict(report['program']), batch['observations'], report['fallback'], .5)
    assert confirmed['loss'] == pytest.approx(actual)
    assert actual < CONFIG.minimum_improvement
    return batch['observations']


def _preserved_source(history, saved):
    before = next(concept for concept in history['before']['concepts']
                  if concept['concept_id'] == history['source_id'])
    after = next(concept for concept in saved['concepts']
                 if concept['concept_id'] == history['source_id'])
    assert after['program'] == before['program']
    assert all(definition in saved['definitions'] for definition in history['before']['definitions'])
    return Program.from_dict(before['program'])


def test_disappearing_mechanism_is_deleted_after_real_distinction_and_new_confirmation(mechanism_history):
    history = mechanism_history
    report = history['removed']
    assert report['status'] == 'completed' and history['removed_error'] < CONFIG.minimum_improvement
    confirmation = _confirmed_response(report, history['removed_lab'])
    source = _preserved_source(history, history['after_removal'])
    assert loss(source, confirmation, 0., .5) > CONFIG.minimum_improvement
    assert report['concepts_added']
    assert all(concept_id in {item['concept_id'] for item in history['after_removal']['concepts']}
               for concept_id in report['concepts_added'])


def test_necessary_mechanisms_survive_a_coefficient_change(mechanism_history):
    history = mechanism_history
    report = history['necessary']
    assert report['status'] == 'completed' and history['necessary_error'] < CONFIG.minimum_improvement
    confirmation = _confirmed_response(report, history['necessary_lab'])
    after = history['necessary_library'].to_dict()
    source = _preserved_source(history, after)
    assert loss(source, confirmation, 0., .5) > CONFIG.minimum_improvement
    assert report['concepts_added']
    assert all(concept_id in {item['concept_id'] for item in after['concepts']}
               for concept_id in report['concepts_added'])


def test_json_handoff_keeps_the_old_mechanism_available_when_it_returns(mechanism_history):
    history = mechanism_history
    returned = history['returned']
    after = history['returned_library'].to_dict()
    _preserved_source(history, history['after_removal'])
    _preserved_source(history, after)
    assert returned['status'] == 'completed' and history['returned_error'] < CONFIG.minimum_improvement
    _confirmed_response(returned, history['returned_lab'])
    assert history['source_id'] in returned['used_sources']
    assert all(concept_id in {item['concept_id'] for item in after['concepts']}
               for concept_id in history['removed']['concepts_added'])

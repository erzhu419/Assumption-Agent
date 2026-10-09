"""Empty-start discovery and learned nonlinear parameter revision buy real evidence."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


CONFIG = ResearchConfig(observation_budget=240, compute_budget=240000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)


class SignedCompositionLab:
    channels = ()

    def __init__(self, episode, *, names=('drive', 'load'), gain=-.65, squared=True):
        self._gain, self._squared = gain, squared
        self._names = names
        self.spec = TaskSpec(episode, 'Predict the response to controllable inputs',
            names, (), (), feature_bounds={name: (-1.5, 1.5) for name in names}, max_probe_length=64)
        self.acquisitions = []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def random_probe(self, seed, *, length=24, name='sample'):
        rng = Random(seed)
        return Probe(name, tuple({key: rng.uniform(*bounds)
            for key, bounds in self.spec.feature_bounds.items()} for _ in range(length)))

    def acquire(self, probe, channels=()):
        assert not channels
        observations, history = [], []
        for point in probe.sequence:
            inner = .8 * point[self._names[0]] ** 2 + self._gain * point[self._names[1]]
            target = .2 + 1.1 * inner ** 2 if self._squared else inner
            observations.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, '__target__': target})
        self.acquisitions.append({'probe': probe, 'observations': observations})
        return MeasurementBatch(observations, len(observations), 0, ())


def restore(library):
    return KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))


def research(lab, library):
    report = UnifiedResearchAgent(CONFIG, library=library).run(lab)
    assert report['costs']['observation_cost'] == sum(
        len(batch['observations']) for batch in lab.acquisitions) <= CONFIG.observation_budget
    assert report['costs']['measurement_cost'] == 0
    assert report['costs']['compute_cost'] <= CONFIG.compute_budget
    return report


def confirmation(report):
    validated = next(action for action in report['actions']
        if action['action'] == 'validate_frozen_program' and action['accepted'])
    index = report['actions'].index(validated)
    acquired = next(action for action in reversed(report['actions'][:index])
        if action['action'] == 'acquire' and action['purpose'] == 'confirm_frozen_program')
    assert acquired['observation_cost'] == validated['observation_rows'] > 0
    return validated, acquired


def fresh_error(report, lab, seed):
    program = Program.from_dict(report['program'])
    rows = [row for offset, length in ((0, 1), (1, 3), (2, 64), (3, 64))
        for row in lab.acquire(lab.random_probe(seed + offset, length=length,
            name=f'held_out_{offset}')).observations]
    return loss(program, rows, report['fallback'], .5)


def test_empty_library_discovers_signed_mixed_degree_response_and_confirms_it():
    lab, library = SignedCompositionLab('discovery_episode'), KnowledgeLibrary()
    assert not library.concepts and not library.definitions and not library.meta_priors
    report = research(lab, library)
    assert report['status'] == 'completed'
    assert report['concepts_added'] and not report['used_sources']
    validated, _ = confirmation(report)
    assert validated['loss'] <= CONFIG.minimum_improvement
    assert fresh_error(report, lab, 42091) < CONFIG.minimum_improvement
    assert Program.from_dict(report['program']).required_features() == {'drive', 'load'}
    saved = restore(library)
    assert {item.concept_id for item in saved.concepts} >= set(report['concepts_added'])
    assert any(prior['source_concept_id'] in report['concepts_added'] for prior in saved.meta_priors)


@pytest.fixture(scope='module')
def changed_inner_response():
    library, reports, laboratories, snapshots = KnowledgeLibrary(), [], [], []
    # The source is acquired by the same agent: neither a true source program
    # nor its coefficients enter the library or the public task specification.
    for index, (squared, gain) in enumerate(((False, -.65), (True, -.65),
                                           (True, .35), (True, -.65))):
        lab = SignedCompositionLab(f'acquired_episode_{index}', names=('rate', 'stress'),
                                   gain=gain, squared=squared)
        reports.append(research(lab, library))
        laboratories.append(lab)
        snapshots.append(deepcopy(library.to_dict()))
        library = restore(library)
    # Evaluation starts after the complete learn/change/return sequence.
    errors = [fresh_error(report, lab, 53001 + 1009 * index)
        for index, (report, lab) in enumerate(zip(reports, laboratories))]
    return locals()


def test_changed_inner_coefficient_is_researched_and_independently_confirmed(changed_inner_response):
    history = changed_inner_response
    source, changed = history['reports'][1:3]
    assert all(report['status'] == 'completed' for report in history['reports'])
    assert all(error < CONFIG.minimum_improvement for error in history['errors'])
    validated, acquired = confirmation(changed)
    observed_change = history['laboratories'][2].acquisitions[0]['observations']
    assert loss(Program.from_dict(source['program']), observed_change, source['fallback'], .5) > CONFIG.minimum_improvement
    if changed['selected_kind'] == 'mechanism_parameter_refit':
        tested = next(action for action in changed['actions']
                      if action['action'] == 'test_mechanism_predictions')
        batch = next(item for item in history['laboratories'][2].acquisitions
                     if item['probe'].probe_id == tested['probe_id'])
        outcomes = {}
        for alternative, hypothesis, outcome in zip(tested['alternatives'],
                tested['plan']['hypotheses'], tested['evidence']['hypotheses']):
            actual = loss(Program.from_dict(hypothesis['program']), batch['observations'], 0., .5)
            assert outcome['measured_loss'] == pytest.approx(actual)
            outcomes.setdefault(alternative['kind'], []).append(outcome)
        assert outcomes['retained_mechanism'][0]['status'] == 'counterexample'
        assert any(item['status'] == 'supported_on_probe'
                   for item in outcomes['mechanism_parameter_refit'])
        assert tested['evidence']['compute_cost'] == len(batch['observations']) * len(tested['plan']['hypotheses'])
        assert changed['actions'].index(tested) < changed['actions'].index(acquired)
        assert acquired['probe_id'] != tested['probe_id']
        assert set(source['concepts_added']) & set(changed['method_source_ids'])
    else:
        assert changed['selected_kind'] == 'research_program'
        assert any(step['operation'] in {'compose_measured_coordinates', 'refit_research_program'}
                   for step in changed['research_program']['selection_steps'])
        assert changed['research_program']['status'] == 'supported_on_fresh_trajectory'
        assert any(action['action'] in {'propose_coordinate_reconstruction', 'replay_research_programs'}
                   for action in changed['actions'][:changed['actions'].index(acquired)])


def test_json_return_preserves_and_reuses_the_old_successful_nonlinear_response(changed_inner_response):
    history = changed_inner_response
    source, changed, returned = history['reports'][1:]
    assert history['errors'][3] < CONFIG.minimum_improvement
    for definition in history['snapshots'][1]['definitions']:
        assert definition in history['snapshots'][2]['definitions']
        assert definition in history['snapshots'][3]['definitions']
    original_sources = {identifier for report in history['reports'][:2]
                        for identifier in report['concepts_added']}
    # The original inner call plus the learned square is also an executable
    # restoration; selecting the outer call is not required for correct reuse.
    assert original_sources & set(returned['used_sources'])
    final_ids = {item['concept_id'] for item in history['snapshots'][3]['concepts']}
    assert set(changed['concepts_added']) <= final_ids

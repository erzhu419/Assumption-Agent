"""An empty library learns a nonlinear response and its time scale together."""

from dataclasses import replace
import gzip
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import propose_coordinate_programs, replay_research_program
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


CONFIG = ResearchConfig(observation_budget=240, compute_budget=240000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)


class UnspecifiedResponseLab:
    channels = ()

    def __init__(self, episode, width, *, names=('drive', 'load'), noise=0., gain=-.65):
        self.spec = TaskSpec(episode, 'Predict the measured response to controllable inputs',
            names, (), (), feature_bounds={name: (-1.5, 1.5) for name in names},
            max_probe_length=64)
        self._width, self._names, self._noise, self._gain = width, names, noise, gain
        self._rng, self.acquisitions = Random(740013), []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def random_probe(self, seed, *, length=24, name='sample'):
        rng = Random(seed)
        return Probe(name, tuple({key: rng.uniform(*bounds)
            for key, bounds in self.spec.feature_bounds.items()} for _ in range(length)))

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, responses = [], [], []
        for point in probe.sequence:
            inner = .8 * point[self._names[0]] ** 2 + self._gain * point[self._names[1]]
            responses.append(.2 + 1.1 * inner ** 2)
            target = sum(responses if self._width is None else responses[-self._width:])
            if self._noise:
                target += self._rng.gauss(0., self._noise)
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, '__target__': target})
        self.acquisitions.append({'probe': probe, 'observations': rows})
        return MeasurementBatch(rows, len(rows), 0, ())


def _fresh_error(report, lab, seed):
    rows = [row for repeat, length in enumerate((1, 3, 9, 64, 64))
        for row in lab.acquire(lab.random_probe(seed + 1009 * repeat,
            length=length, name=f'terminal_{repeat}')).observations]
    return loss(Program.from_dict(report['program']), rows, report['fallback'], .5)


def _paid_confirmation(report):
    events = report['actions']
    validated = next(event for event in reversed(events)
        if event['action'] == 'validate_frozen_program' and event['accepted'])
    acquisition = next(event for event in reversed(events[:events.index(validated)])
        if event['action'] == 'acquire' and event['purpose'] in
        {'confirm_frozen_program', 'confirm_short_frozen_program'})
    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    confirmation = batches[acquisition['probe_id']]
    earlier = [batches[event['probe_id']] for event in events[:events.index(acquisition)]
               if event['action'] == 'acquire']
    assert earlier
    assert all(confirmation['probe']['sequence'] != batch['probe']['sequence'] for batch in earlier)
    assert acquisition['observation_cost'] == validated['observation_rows'] > 0


@pytest.fixture(scope='module')
def cold_research(tmp_path_factory):
    cases = [(5, 0., ('drive', 'load')), (7, 0., ('rate', 'stress')),
             (2, 0., ('u', 'v')), (7, .01, ('rate', 'stress')),
             (None, 0., ('drive', 'load'))]
    learned = []
    for position, (width, noise, names) in enumerate(cases):
        lab = UnspecifiedResponseLab(f'measured_episode_{position}', width, noise=noise, names=names)
        library = KnowledgeLibrary()
        assert not library.concepts and not library.definitions and not library.meta_priors
        config = replace(CONFIG, seed=9931701 + 1009 * position)
        report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
        assert report['costs']['observation_cost'] == sum(
            len(batch['observations']) for batch in lab.acquisitions) <= config.observation_budget
        assert report['costs']['measurement_cost'] == 0
        assert report['costs']['compute_cost'] <= config.compute_budget
        assert report['costs']['queries'] == len(lab.acquisitions)
        learned.append({'report': report, 'knowledge': json.loads(json.dumps(library.to_dict())),
                        'width': width, 'noise': noise, 'names': names})

    # Terminal mean responses never enter any of the independent learners.
    for position, item in enumerate(learned):
        mean_lab = UnspecifiedResponseLab(f'evaluation_{position}', item['width'], names=item['names'])
        item['mean_error'] = _fresh_error(item['report'], mean_lab, 9942701 + position * 5003)
    saved = tmp_path_factory.mktemp('cold_temporal') / 'research.json.gz'
    saved.write_bytes(gzip.compress(json.dumps(learned).encode()))
    return learned, saved


@pytest.mark.parametrize('position', [0, 1, 2])
def test_empty_library_discovers_nonlinear_response_and_unannounced_window(cold_research, position, record_property):
    learned, saved = cold_research
    item = learned[position]
    report = item['report']
    record_property('research_reports', str(saved))
    record_property('mean_response_mse', item['mean_error'])
    assert report['status'] == 'completed'
    assert item['mean_error'] < CONFIG.minimum_improvement
    assert report['concepts_added'] and not report['used_sources']
    assert Program.from_dict(report['program']).required_features() == set(item['names'])
    _paid_confirmation(report)
    restored = KnowledgeLibrary.from_dict(item['knowledge'])
    assert set(report['concepts_added']) <= {concept.concept_id for concept in restored.concepts}


def test_noisy_empty_start_is_checked_on_paid_repeated_response(cold_research, record_property):
    learned, saved = cold_research
    item = learned[3]
    report = item['report']
    record_property('research_reports', str(saved))
    record_property('mean_response_mse', item['mean_error'])
    assert report['status'] == 'completed' and report['concepts_added']
    assert item['mean_error'] < 3e-4
    _paid_confirmation(report)
    assert any(event['action'] == 'compare_noise_replication' and event['noise_supported']
               for event in report['actions'])


def test_unbounded_accumulation_cannot_be_confirmed_as_a_short_window(cold_research):
    learned, _ = cold_research
    item = learned[4]
    # Accumulation is a competing physical mechanism, not a request to force
    # the newly added finite-window route to explain every temporal response.
    if item['report']['status'] == 'completed':
        assert item['mean_error'] < CONFIG.minimum_improvement
        _paid_confirmation(item['report'])
    else:
        assert item['report']['status'] == 'unresolved'
        assert not item['report']['concepts_added']


def test_discovered_temporal_procedure_refits_after_json_and_sensor_rename():
    lab = UnspecifiedResponseLab('measured_training', 5)
    training = [row for seed in (93001, 93002)
        for row in lab.acquire(lab.random_probe(seed, length=24)).observations]
    budget = SearchBudget(32000)
    claims = propose_coordinate_programs(lab.spec, training, budget)
    assert claims
    selected = min(claims, key=lambda item: item['training_loss'])
    assert selected['training_loss'] < CONFIG.minimum_improvement
    assert not selected['candidate'].parents
    assert selected['research_program']['status'] == 'requires_fresh_validation'
    assert 0 < budget.spent <= budget.limit
    plan = json.loads(json.dumps(selected['research_program']))
    changed = UnspecifiedResponseLab('measured_changed', 5, names=('u', 'v'), gain=-.95)
    evidence = [row for seed in (94001, 94002)
        for row in changed.acquire(changed.random_probe(seed, length=24)).observations]
    replay_budget = SearchBudget(4000)
    replayed = replay_research_program(plan, evidence, replay_budget,
                                       feature_mapping={'drive': 'u', 'load': 'v'})
    assert replayed is not None and replayed['research_program']['reused_procedure']
    terminal = changed.acquire(changed.random_probe(95001, length=64)).observations
    assert loss(replayed['candidate'].program, terminal, 0., .5) < CONFIG.minimum_improvement
    assert 0 < replay_budget.spent <= replay_budget.limit

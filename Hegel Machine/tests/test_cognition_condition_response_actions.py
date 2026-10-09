"""Physical counterexamples change condition experiments and response models."""

from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch, MeasurementChannel
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph, bind_answers
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class ConditionResponseLab:
    channels = ()

    def __init__(self, contribution='linear', *, logic='and', placement='both_outside',
                 width=3, names=('alpha', 'beta', 'gamma', 'delta'),
                 control_scale=1., response_scale=1., thresholds=(.173, -.317)):
        self.spec = TaskSpec('unlabelled_response_apparatus',
            'Apply controllable input sequences and predict the measured response.', names, (), (),
            feature_bounds={name: (-1.5 * control_scale, 1.5 * control_scale) for name in names},
            max_probe_length=64)
        self._roles = names[0], names[2], names[1], names[3]
        self._contribution, self._logic, self._placement, self._width = contribution, logic, placement, width
        self._control_scale, self._response_scale, self._thresholds = control_scale, response_scale, thresholds
        self.observation_cost, self.acquisitions = 0, []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, responses, gates = [], [], [], []
        for point in probe.sequence:
            x, z, first, second = (point[name] / self._control_scale for name in self._roles)
            contribution = (.4 * first - .3 * second if self._contribution == 'linear' else
                            .5 * first ** 2 - .4 * second ** 2 if self._contribution == 'quadratic' else 0.)
            responses.append((.13 + .8 * x + .6 * z ** 2 + contribution,
                              -.11 - .9 * x - .45 * z ** 2 + contribution))
            gates.append((first >= self._thresholds[0], second >= self._thresholds[1]))
            target = 0.
            for index in range(max(0, len(responses) - self._width), len(responses)):
                left = gates[index if self._placement in ('both_inside', 'first_inside') else -1][0]
                right = gates[index if self._placement in ('both_inside', 'second_inside') else -1][1]
                active = True if self._logic is None else (
                    left and right if self._logic == 'and' else
                    left or right if self._logic == 'or' else left != right)
                target += responses[index][0 if active else 1]
            target *= self._response_scale
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, '__target__': target})
        self.observation_cost += len(rows)
        self.acquisitions.append({'probe': probe, 'observations': rows})
        return MeasurementBatch(rows, len(rows), 0, ())

    def restore(self, acquisitions):
        self.observation_cost, self.acquisitions = 0, []
        for item in acquisitions:
            self.acquire(Probe(item['probe_id'], tuple(item['sequence'])))


class ObservedConditionLab:
    """The auxiliary reading is available for purchase, never a control setting."""
    channels = (MeasurementChannel('meter', 1, 'Read the auxiliary process instrument.'),)

    def __init__(self, seed=9327301):
        self.spec = TaskSpec('observed_condition_apparatus',
            'Apply controllable input sequences and predict the measured response.',
            ('alpha', 'beta', 'gamma'), (), (),
            feature_bounds={name: (-1.5, 1.5) for name in ('alpha', 'beta', 'gamma')}, max_probe_length=64)
        self._random, self._experiments = Random(seed), {}
        self.observation_cost, self.measurement_cost, self.acquisitions = 0, 0, []

    def spec_for(self, channels=()):
        return replace(self.spec, feature_names=(*self.spec.feature_names, *channels))

    def acquire(self, probe, channels=()):
        assert set(channels) <= {'meter'}
        assert all(set(point) == set(self.spec.feature_bounds) for point in probe.sequence)
        key = probe.probe_id, tuple(tuple(point[name] for name in self.spec.feature_names) for point in probe.sequence)
        if key not in self._experiments:
            raw, selected = [], []
            for point in probe.sequence:
                meter = .35 * point['gamma'] + self._random.uniform(-1., 1.)
                x, z = point['alpha'], point['beta']
                selected.append((.13 + .8 * x + .6 * z ** 2 if meter >= .173 else
                                 -.11 - .9 * x - .45 * z ** 2) + .25 * meter)
                raw.append(({**point, 'meter': meter}, sum(selected[-3:])))
            self._experiments[key] = raw
        rows, history = [], []
        for readings, target in self._experiments[key]:
            features = {name: readings[name] for name in self.spec_for(channels).feature_names}
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, '__target__': target})
        measurement_cost = len(rows) if 'meter' in channels else 0
        self.observation_cost += len(rows)
        self.measurement_cost += measurement_cost
        self.acquisitions.append({'probe': probe, 'channels': tuple(channels), 'observations': rows})
        return MeasurementBatch(rows, len(rows), measurement_cost, tuple(channels))


def random_probe(spec, seed, *, length=24):
    random = Random(seed)
    return Probe(f'paid_{seed}', tuple({name: random.uniform(*bounds)
        for name, bounds in spec.feature_bounds.items()} for _ in range(length)))


CONFIG = ResearchConfig(observation_budget=320, compute_budget=320000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)


def fresh_error(report, laboratory, channels=()):
    rows = [row for seed in (9727301, 9728310) for row in
        laboratory.acquire(random_probe(laboratory.spec, seed, length=64), channels).observations]
    return loss(Program.from_dict(report['program']), rows, report['fallback'], .5)


@pytest.mark.parametrize('contribution', ['linear', 'quadratic'])
def test_counterexample_changes_the_paid_method_before_joint_response_recovery(contribution, tmp_path):
    laboratory = ConditionResponseLab(contribution)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9327301),
        library=KnowledgeLibrary(), measurement_budget=0).run(laboratory)
    saved = tmp_path / f'{contribution}_condition_response.json'
    saved.write_text(json.dumps(report))
    assert report['status'] == 'completed', str(saved)
    assert report['costs']['observation_cost'] == laboratory.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(laboratory.acquisitions)
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget
    assert fresh_error(report, ConditionResponseLab(contribution)) < .05, str(saved)
    # Purchasing a different conditional experiment is the observable revision;
    # saving an unchanged old recipe would not satisfy this assertion.
    assert any(item['probe'].probe_id.startswith('composed_response:')
        for item in laboratory.acquisitions), str(saved)


@pytest.mark.parametrize('logic,contribution', [(None, 'quadratic'), ('and', None)])
def test_smooth_and_pure_gate_controls_keep_their_already_effective_methods(logic, contribution, tmp_path):
    laboratory = ConditionResponseLab(contribution, logic=logic)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9327301),
        library=KnowledgeLibrary(), measurement_budget=0).run(laboratory)
    saved = tmp_path / 'control_response.json'
    saved.write_text(json.dumps(report))
    assert report['status'] == 'completed', str(saved)
    assert report['costs']['observation_cost'] == laboratory.observation_cost <= CONFIG.observation_budget
    assert fresh_error(report, ConditionResponseLab(contribution, logic=logic)) < .05, str(saved)
    assert not any(item['probe'].probe_id.startswith('composed_response:')
        for item in laboratory.acquisitions), str(saved)


def test_json_restored_pure_gate_method_recovers_after_response_logic_and_units_change(tmp_path):
    library = KnowledgeLibrary()
    source = ConditionResponseLab(None)
    first = UnifiedResearchAgent(replace(CONFIG, seed=9327301),
        library=library, measurement_budget=0).run(source)
    assert first['status'] == 'completed' and first['research_program'] is not None
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    changed = ConditionResponseLab('quadratic', logic='or', placement='both_inside', width=5,
        names=('p', 'q', 'r', 's'), control_scale=2.3, response_scale=1.25, thresholds=(.413, -.271))
    report = UnifiedResearchAgent(replace(CONFIG, seed=9328310),
        library=restored, measurement_budget=0).run(changed)
    saved = tmp_path / 'restored_condition_response.json'
    saved.write_text(json.dumps(report))
    assert report['status'] == 'completed', str(saved)
    assert report['costs']['observation_cost'] == changed.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(changed.acquisitions)
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget
    independent = ConditionResponseLab('quadratic', logic='or', placement='both_inside', width=5,
        names=('p', 'q', 'r', 's'), control_scale=2.3, response_scale=1.25, thresholds=(.413, -.271))
    fresh = [row for seed in (9727301, 9728310) for row in
        independent.acquire(random_probe(independent.spec, seed, length=64)).observations]
    actual = Program.from_dict(report['program'])
    assert loss(actual, fresh, report['fallback'], .5) < .05, str(saved)
    assert actual.required_features() <= set(changed.spec.feature_names)
    assert report['research_program'] is not None, str(saved)
    graph = ResearchGraph.from_dict(json.loads(json.dumps(report['research_program']['graph'])))
    output = graph.nodes[report['research_program']['output_node']]
    compiled = bind_answers(output.expression, {key: graph.nodes[key].answer for key in output.depends_on})
    assert [compiled.predict(row) for row in fresh] == pytest.approx([actual.predict(row) for row in fresh]), str(saved)


def test_an_observed_gate_is_purchased_and_never_written_as_a_control(tmp_path):
    laboratory = ObservedConditionLab()
    report = UnifiedResearchAgent(replace(CONFIG, seed=9327301),
        library=KnowledgeLibrary(), measurement_budget=320).run(laboratory)
    saved = tmp_path / 'observed_condition.json'
    saved.write_text(json.dumps(report))
    assert report['selected_channels'] == ['meter'], str(saved)
    assert report['costs']['observation_cost'] == laboratory.observation_cost <= CONFIG.observation_budget
    assert report['costs']['measurement_cost'] == laboratory.measurement_cost > 0
    assert all(set(point) == set(laboratory.spec.feature_bounds)
        for item in laboratory.acquisitions for point in item['probe'].sequence)
    assert 'meter' not in laboratory.spec_for(('meter',)).feature_bounds
    assert fresh_error(report, ObservedConditionLab(seed=9427301), ('meter',)) < .05, str(saved)

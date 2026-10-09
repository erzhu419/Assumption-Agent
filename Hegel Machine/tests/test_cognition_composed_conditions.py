"""Actual crossed controls discover unknown boolean gates and time layers."""

from dataclasses import replace
from itertools import combinations_with_replacement
import json
from random import Random

import pytest

from hegel_machine.cognition.conditional_experiments import assess_conditional_experiment
from hegel_machine.cognition.composed_condition_experiments import (
    assess_composed_boundary_probe, assess_composed_cross_experiment,
    propose_composed_boundary_probe, propose_composed_condition_experiments,
    propose_composed_cross_experiment)
from hegel_machine.cognition.composed_conditions import propose_composed_conditions
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph, bind_answers
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class UnspecifiedComposedLab:
    channels = ()

    def __init__(self, logic, placement, width, *, gate_axes=(1, 3),
                 thresholds=(.173, -.317), names=('alpha', 'beta', 'gamma', 'delta'),
                 control_scale=1., response_scale=1., task_id='unknown_apparatus'):
        self.spec = TaskSpec(task_id,
            'Apply controllable input sequences and predict the measured response.', names, (), (),
            feature_bounds={name: (-1.5 * control_scale, 1.5 * control_scale) for name in names},
            max_probe_length=64)
        self._logic, self._placement, self._width = logic, placement, width
        nongates = [name for index, name in enumerate(names) if index not in gate_axes]
        self._roles = (*nongates, *(names[index] for index in gate_axes))
        self._thresholds = thresholds
        self._control_scale, self._response_scale = control_scale, response_scale
        self.observation_cost, self.acquisitions = 0, []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def random_probe(self, seed, *, length=24, grid=False):
        rng = Random(seed)
        def value(bounds):
            low, high = bounds
            return low + (high - low) * rng.choice((0., 1 / 6, 2 / 6, .5, 4 / 6, 5 / 6, 1.)) \
                if grid else rng.uniform(low, high)
        return Probe(f'purchased_{seed}', tuple({name: value(bounds)
            for name, bounds in self.spec.feature_bounds.items()} for _ in range(length)))

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, responses, gates = [], [], [], []
        for point in probe.sequence:
            x, z, first, second = (point[name] / self._control_scale for name in self._roles)
            responses.append((.13 + .8 * x + .6 * z ** 2, -.11 - .9 * x - .45 * z ** 2))
            gates.append((first >= self._thresholds[0], second >= self._thresholds[1]))
            target = 0.
            for index in range(max(0, len(responses) - self._width), len(responses)):
                left = gates[index if self._placement in ('both_inside', 'first_inside') else -1][0]
                right = gates[index if self._placement in ('both_inside', 'second_inside') else -1][1]
                active = True if self._logic is None else (
                    (left and right) if self._logic == 'and' else
                    (left or right) if self._logic == 'or' else left != right)
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


def paid(lab, seeds=(53101, 54110), *, length=24, grid=False):
    return [row for seed in seeds for row in
            lab.acquire(lab.random_probe(seed, length=length, grid=grid)).observations]


def basis(spec):
    atoms = [Program('feature', name=name) for name in spec.feature_names]
    return [*atoms, *(Program('mul', children=(left, right))
        for left, right in combinations_with_replacement(atoms, 2))]


def measure_axis_contexts(lab, pilot, budget):
    packs = propose_composed_condition_experiments(lab.spec, pilot, (), budget)
    rows, evidence = [], []
    for pack in packs:
        reference = lab.acquire(pack['reference']).observations
        rows.extend(reference)
        for plan in pack['interventions']:
            measured = lab.acquire(plan['probe']).observations
            prefix = plan['prefix_length']
            assert [item.features for item in measured[:prefix]] == [item.features for item in reference[:prefix]]
            assert [item.target for item in measured[:prefix]] == pytest.approx(
                [item.target for item in reference[:prefix]])
            rows.extend(measured)
            evidence.append(assess_conditional_experiment(lab.spec, plan, reference, measured, budget))
    return packs, rows, evidence


def measure_cross(lab, evidence, budget):
    pack = propose_composed_cross_experiment(lab.spec, evidence, budget)
    assert pack is not None
    probes = [pack['reference'], *(plan['probe'] for plan in pack['interventions'])]
    rows_by_probe = {probe.probe_id: lab.acquire(probe).observations for probe in probes}
    shared = rows_by_probe[pack['reference'].probe_id][0]
    for plan in pack['interventions']:
        rows = rows_by_probe[plan['probe'].probe_id]
        assert rows[0].features == shared.features and rows[0].target == pytest.approx(shared.target)
        assert all(all(row.features[name] == pack['background'][name]
            for name in lab.spec.feature_names if name not in pack['controls']) for row in rows)
    joint = assess_composed_cross_experiment(lab.spec, pack, rows_by_probe, budget)
    return pack, [row for rows in rows_by_probe.values() for row in rows], joint


def measure_boundaries(lab, joint, budget):
    rows = []
    for control in joint['controls']:
        for _ in range(12):
            context = next(item for item in joint['active_contexts'] if item['control'] == control)
            before = list(context['threshold_interval'])
            probe = propose_composed_boundary_probe(lab.spec, context, serial=len(lab.acquisitions))
            measured = lab.acquire(probe).observations
            assert len(measured) == probe.cost == 1 and not measured[0].history
            rows.extend(measured)
            joint = assess_composed_boundary_probe(joint, context, measured[0], budget)
            assert joint is not None and not joint.get('boundary_counterexamples')
            after = joint['threshold_intervals'][joint['controls'].index(control)]
            assert before[0] <= after[0] < after[1] <= before[1]
            assert after[1] - after[0] == pytest.approx((before[1] - before[0]) / 2.)
    return rows, joint


def test_full_context_packs_are_funded_and_do_not_assign_hidden_gate_roles():
    lab = UnspecifiedComposedLab('and', 'both_inside', 5)
    budget = SearchBudget(10000)
    assert propose_composed_condition_experiments(lab.spec, [], (), budget, observation_budget=119) == []
    assert propose_composed_condition_experiments(lab.spec, [], (), SearchBudget(1)) == []
    packs = propose_composed_condition_experiments(lab.spec, [], (), budget)
    assert len(packs) == 2 and lab.observation_cost == 0
    assert sum(pack['observation_cost'] for pack in packs) == 120
    for context, pack in enumerate(packs):
        assert {plan['control'] for plan in pack['interventions']} == set(lab.spec.feature_names)
        assert all(pack['background'][name] == bounds[context] for name, bounds in lab.spec.feature_bounds.items())
        probes = [pack['reference'], *(plan['probe'] for plan in pack['interventions'])]
        assert pack['observation_cost'] == sum(probe.cost for probe in probes)
        assert all(all(low <= point[name] <= high
            for name, (low, high) in lab.spec.feature_bounds.items())
            for probe in probes for point in probe.sequence)


def test_actual_continuous_gate_response_keeps_the_measured_interval_and_counterexample():
    class ContinuousGateResponseLab(UnspecifiedComposedLab):
        def acquire(self, probe, channels=()):
            batch = super().acquire(probe, channels)
            rows = [Observation(dict(row.features), row.target + .25 * row.features['beta'], row.history)
                    for row in batch.observations]
            self.acquisitions[-1]['observations'] = rows
            return MeasurementBatch(rows, len(rows), 0, ())

    lab = ContinuousGateResponseLab('and', 'both_outside', 3)
    background = {name: (low + high) * .5 for name, (low, high) in lab.spec.feature_bounds.items()}
    background['alpha'], background['gamma'], background['delta'] = -.39, -.21, 1.5
    control, interval = 'beta', list(lab.spec.feature_bounds['beta'])
    endpoints = [lab.acquire(Probe(f'actual_endpoint_{index}',
        ({**background, control: value},))).observations[0] for index, value in enumerate(interval)]
    context = {'control': control, 'background': background, 'threshold_interval': interval,
        'low_response': endpoints[0].target, 'high_response': endpoints[1].target}
    evidence = {'controls': ['beta', 'delta'], 'threshold_intervals': [interval, [-1.5, 1.5]],
        'active_contexts': [context]}
    probe = propose_composed_boundary_probe(lab.spec, context, serial=2)
    row = lab.acquire(probe).observations[0]
    budget = SearchBudget(20)
    updated = assess_composed_boundary_probe(evidence, context, row, budget)
    assert updated['threshold_intervals'] == evidence['threshold_intervals']
    assert updated['boundary_counterexamples'][0]['target'] == row.target
    assert updated['boundary_counterexamples'][0]['features'] == row.features
    assert lab.observation_cost == 3 and budget.spent == 4


@pytest.mark.parametrize('logic,placement,width,gate_axes', [
    ('and', 'both_outside', 3, (1, 3)), ('or', 'both_inside', 5, (0, 2)),
    ('xor', 'first_inside', 7, (1, 3)), ('xor', 'second_inside', 3, (0, 2))])
def test_paid_crossed_effects_discover_logic_and_time_layers(logic, placement, width, gate_axes):
    lab = UnspecifiedComposedLab(logic, placement, width, gate_axes=gate_axes)
    pilot = paid(lab)
    budget = SearchBudget(80000)
    packs, measured, evidence = measure_axis_contexts(lab, pilot, budget)
    discovered = {item['control'] for item in evidence if item.get('gain', 0.) > 1e-10}
    assert discovered == {lab.spec.feature_names[index] for index in gate_axes}
    cross, crossed, joint = measure_cross(lab, evidence, budget)
    assert joint['status'] == 'conditional' and logic in joint['logic_candidates']
    assert width in joint['width_candidates'] and placement in joint['placement_candidates']
    assert lab.observation_cost == len(pilot) + sum(pack['observation_cost'] for pack in packs) + cross['observation_cost'] == 216
    boundaries, joint = measure_boundaries(lab, joint, budget)
    assert lab.observation_cost == 240
    fit_budget = SearchBudget(200000)
    claims = propose_composed_conditions(lab.spec, [*pilot, *measured, *crossed, *boundaries], fit_budget,
        basis=basis(lab.spec), evidence=joint)
    assert claims
    best = min(claims, key=lambda claim: claim['training_loss'])
    independent = UnspecifiedComposedLab(logic, placement, width, gate_axes=gate_axes)
    fresh = paid(independent, (86101, 87110), length=64, grid=True)
    assert loss(best['candidate'].program, fresh, 0., .5) < 1e-10
    assert 0 < budget.spent <= budget.limit and 0 < fit_budget.spent <= fit_budget.limit


CONFIG = ResearchConfig(observation_budget=320, compute_budget=320000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)


def fresh_error(report, lab):
    rows = paid(lab, (97101, 98110), length=64)
    return loss(Program.from_dict(report['program']), rows, report['fallback'], .5)


@pytest.mark.parametrize('logic,placement,width', [('and', 'both_outside', 3),
    ('xor', 'first_inside', 7)])
def test_cold_agent_buys_crossed_conditions_and_predicts_independent_inputs(logic, placement, width, tmp_path):
    lab = UnspecifiedComposedLab(logic, placement, width)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9325101),
        library=KnowledgeLibrary(), measurement_budget=0).run(lab)
    saved = tmp_path / 'composed_research.json'
    saved.write_text(json.dumps(report))
    assert report['research_program'] is not None, str(saved)
    assert any(step['operation'] == 'discover_composed_conditions'
        for step in report['research_program']['selection_steps']), str(saved)
    assert report['costs']['observation_cost'] == lab.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(lab.acquisitions)
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget
    independent = UnspecifiedComposedLab(logic, placement, width)
    assert fresh_error(report, independent) < .05, str(saved)


def test_cold_plain_window_keeps_the_smooth_joint_response_in_competition(tmp_path):
    lab = UnspecifiedComposedLab(None, 'both_inside', 5, gate_axes=(0, 2))
    report = UnifiedResearchAgent(replace(CONFIG, seed=9325101),
        library=KnowledgeLibrary(), measurement_budget=0).run(lab)
    saved = tmp_path / 'plain_window_research.json'
    saved.write_text(json.dumps(report))
    assert report['status'] == 'completed', str(saved)
    assert report['costs']['observation_cost'] == lab.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(lab.acquisitions)
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget
    assert not any(step['operation'] == 'discover_composed_conditions'
        for step in (report.get('research_program') or {}).get('selection_steps', [])), str(saved)
    independent = UnspecifiedComposedLab(None, 'both_inside', 5, gate_axes=(0, 2))
    assert fresh_error(report, independent) < 1e-10, str(saved)


def test_restored_wrong_mechanism_library_can_recover_a_short_call_through_run(tmp_path):
    from hegel_machine.cognition.research_programs import _Fitter

    library = KnowledgeLibrary()
    learned = []
    for width in (3, 5):
        old = UnspecifiedComposedLab(None, 'both_inside', width, gate_axes=(0, 2),
            task_id=f'previous_apparatus_{width}')
        rows = paid(old)
        work = SearchBudget(24000)
        terms = [Program('window_sum', lag=width, children=(term,))
                 for term in [Program('const', value=1.), *basis(old.spec)]]
        fitted = _Fitter(rows, work, .5).fit(terms)
        assert fitted is not None and fitted['loss'] < 1e-10
        learned.append(fitted['program'])
        library.learn(Candidate(f'prior_window_{width}', fitted['program'], 'measured_joint_fit'),
                      old.spec, rows)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    independent = UnspecifiedComposedLab(None, 'both_inside', 5, gate_axes=(0, 2))
    fresh = paid(independent, (97101, 98110), length=64)
    assert loss(learned[0], fresh, 0., .5) > .05  # The restored W3 mechanism is wrong here.
    lab = UnspecifiedComposedLab(None, 'both_inside', 5, gate_axes=(0, 2))
    report = UnifiedResearchAgent(replace(CONFIG, seed=9325101),
        library=restored, measurement_budget=0).run(lab)
    saved = tmp_path / 'restored_short_call_research.json'
    saved.write_text(json.dumps(report))
    assert any(action['action'] == 'test_learned_acquisition_method'
        and action['validation_loss'] < CONFIG.minimum_improvement for action in report['actions']), str(saved)
    assert not any(action['action'] == 'compare_research_directions' for action in report['actions']), str(saved)
    assert report['status'] == 'completed', str(saved)
    assert report['costs']['observation_cost'] == lab.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(lab.acquisitions)
    assert loss(Program.from_dict(report['program']), fresh, report['fallback'], .5) < 1e-10, str(saved)


def test_json_library_changes_logic_layers_names_and_scales(tmp_path):
    library = KnowledgeLibrary()
    source = UnspecifiedComposedLab('and', 'both_outside', 3)
    first = UnifiedResearchAgent(replace(CONFIG, seed=9325101),
        library=library, measurement_budget=0).run(source)
    assert first['research_program'] is not None
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    changed = UnspecifiedComposedLab('xor', 'second_inside', 7,
        names=('p', 'q', 'r', 's'), control_scale=2.3, response_scale=1.25,
        thresholds=(.413, -.271), task_id='changed_apparatus')
    report = UnifiedResearchAgent(replace(CONFIG, seed=9326101),
        library=restored, measurement_budget=0).run(changed)
    saved = tmp_path / 'changed_composed_research.json'
    saved.write_text(json.dumps(report))
    assert report['research_program'] is not None, str(saved)
    assert report['costs']['observation_cost'] == changed.observation_cost <= CONFIG.observation_budget
    assert report['costs']['compute_cost'] <= CONFIG.compute_budget
    independent = UnspecifiedComposedLab('xor', 'second_inside', 7,
        names=('p', 'q', 'r', 's'), control_scale=2.3, response_scale=1.25,
        thresholds=(.413, -.271))
    assert fresh_error(report, independent) < .05, str(saved)


def test_json_restored_unfinished_composed_task_acquires_refits_and_replays_current_graph(tmp_path):
    library = KnowledgeLibrary()
    limited = replace(CONFIG, observation_budget=96, compute_budget=24000, seed=9325101)
    lab = UnspecifiedComposedLab('xor', 'first_inside', 7)
    first = UnifiedResearchAgent(limited, library=library, measurement_budget=0).run(lab)
    (tmp_path / 'unfinished_composed_research.json').write_text(json.dumps(first))
    assert first['status'] == 'unresolved'
    stored = json.loads(json.dumps(first))
    restored = UnspecifiedComposedLab('xor', 'first_inside', 7)
    restored.restore([batch['probe'] for batch in stored['batches']])
    assert restored.observation_cost == stored['costs']['observation_cost']
    old_queries = len(restored.acquisitions)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9325101),
        library=KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict()))),
        measurement_budget=0).run(restored, previous=stored)
    saved = tmp_path / 'continued_composed_research.json'
    saved.write_text(json.dumps(report))
    added = report['actions'][len(stored['actions']):]
    assert any(action['action'] == 'assess_composed_cross_intervention' for action in added), str(saved)
    assert any(batch['probe'].probe_id.startswith('composed_cross:')
        for batch in restored.acquisitions[old_queries:]), str(saved)
    assert report['status'] == 'completed', str(saved)
    assert report['research_program'] is not None, str(saved)
    assert any(step['operation'] == 'discover_composed_conditions'
        for step in report['research_program']['selection_steps']), str(saved)
    assert report['costs']['observation_cost'] == restored.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(restored.acquisitions)
    assert report['visit_costs']['queries'] == len(restored.acquisitions) - old_queries
    assert 0 < report['visit_costs']['observation_cost'] == restored.observation_cost - stored['costs']['observation_cost']
    assert stored['costs']['compute_cost'] < report['costs']['compute_cost'] <= CONFIG.compute_budget
    graph = ResearchGraph.from_dict(json.loads(json.dumps(report['research_program']['graph'])))
    output = graph.nodes[report['research_program']['output_node']]
    compiled = bind_answers(output.expression, {key: graph.nodes[key].answer for key in output.depends_on})
    actual = Program.from_dict(report['program'])
    independent = UnspecifiedComposedLab('xor', 'first_inside', 7)
    fresh = paid(independent, (99101, 100110), length=64)
    assert loss(actual, fresh, report['fallback'], .5) < .05, str(saved)
    assert [compiled.predict(row) for row in fresh] == pytest.approx([actual.predict(row) for row in fresh])
    assert [output.answer.predict(row) for row in fresh] == pytest.approx([actual.predict(row) for row in fresh])

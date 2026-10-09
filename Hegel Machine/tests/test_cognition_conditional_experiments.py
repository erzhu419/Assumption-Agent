"""Matched physical interventions discover a cold conditional temporal response."""

from dataclasses import replace
from itertools import combinations_with_replacement
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph, bind_answers
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class UnspecifiedBranchLab:
    channels = ()

    def __init__(self, mode, width, *, gate_axis=2, threshold=.173,
                 names=('alpha', 'beta', 'gamma'), control_scale=1., response_scale=1.,
                 task_id='unlabelled_apparatus'):
        self.spec = TaskSpec(task_id,
            'Apply controllable input sequences and predict the measured response.', names, (), (),
            feature_bounds={name: (-1.5 * control_scale, 1.5 * control_scale) for name in names},
            max_probe_length=64)
        self._mode, self._width, self._threshold = mode, width, threshold
        others = [name for index, name in enumerate(names) if index != gate_axis]
        self._roles = (*others, names[gate_axis])
        self._control_scale, self._response_scale = control_scale, response_scale
        self.observation_cost, self.acquisitions = 0, []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def random_probe(self, seed, *, length=24, name='purchased'):
        rng = Random(seed)
        return Probe(name, tuple({key: rng.uniform(*bounds)
            for key, bounds in self.spec.feature_bounds.items()} for _ in range(length)))

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, first, second, selected = [], [], [], [], []
        for point in probe.sequence:
            x, z, c = (point[name] / self._control_scale for name in self._roles)
            # Equal offsets make a zero-background condition sweep uninformative.
            first.append(.13 + .8 * x + .6 * z ** 2)
            second.append(.13 - .9 * x - .45 * z ** 2)
            active = first if c >= self._threshold else second
            selected.append(active[-1])
            target = sum((active if self._mode == 'outside_window' else selected)[-self._width:])
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


def paid(lab, seeds=(5101, 6101), *, length=24):
    batches = [lab.acquire(lab.random_probe(seed, length=length, name=f'paid_{seed}'))
               for seed in seeds]
    assert all(batch.observation_cost == length and batch.measurement_cost == 0 for batch in batches)
    return [row for batch in batches for row in batch.observations]


def measured_basis(spec):
    atoms = [Program('feature', name=name) for name in spec.feature_names]
    return [*atoms, *(Program('mul', children=(left, right))
                     for left, right in combinations_with_replacement(atoms, 2))]


def latest_condition(payload):
    return next(step for step in reversed(payload['selection_steps'])
                if step['operation'] == 'discover_conditional_coordinates')


def fresh_error(report, laboratory, seed=85101):
    rows = paid(laboratory, (seed, seed + 1009), length=64)
    return loss(Program.from_dict(report['program']), rows, report['fallback'], .5)


def boundary_payments(report, laboratory):
    acquired = {item['probe'].probe_id: item['observations'] for item in laboratory.acquisitions}
    refinements = [action for action in report['actions']
                   if action['action'] == 'refine_conditional_measured_boundary']
    for action in refinements:
        assert action['observation_cost'] == len(action['boundary_trials'])
        before, after = action['before_interval'], action['after_interval']
        assert before[0] <= after[0] <= after[1] <= before[1]
        for trial in action['boundary_trials']:
            rows = acquired[trial['probe_id']]
            assert len(rows) == 1
            assert rows[0].target == trial['target']
    return refinements


def plan_experiments(lab, pilot, budget, *, observation_budget=96):
    from hegel_machine.cognition.conditional_experiments import propose_conditional_experiments
    return propose_conditional_experiments(lab.spec, pilot, (), budget,
        serial=len(lab.acquisitions), observation_budget=observation_budget)


def measure_experiments(lab, packs, budget):
    from hegel_machine.cognition.conditional_experiments import assess_conditional_experiment
    acquired, assessments = [], []
    for pack in packs:
        reference = lab.acquire(pack['reference']).observations
        acquired.extend(reference)
        for intervention in pack['interventions']:
            measured = lab.acquire(intervention['probe']).observations
            prefix = intervention['prefix_length']
            assert [row.features for row in measured[:prefix]] == [row.features for row in reference[:prefix]]
            assert [row.target for row in measured[:prefix]] == pytest.approx(
                [row.target for row in reference[:prefix]])
            acquired.extend(measured)
            plan = {**intervention, 'background': pack['background'], 'reference': pack['reference']}
            assessments.append(assess_conditional_experiment(lab.spec, plan, reference, measured, budget))
    return acquired, assessments


def test_cold_experiment_packs_cover_public_controls_at_matched_nonzero_backgrounds():
    lab = UnspecifiedBranchLab('outside_window', 5, gate_axis=0)
    pilot = paid(lab, (5101,), length=24)
    old_cost = lab.observation_cost
    work = SearchBudget(6000)
    packs = plan_experiments(lab, pilot, work)
    assert len(packs) == 2
    assert lab.observation_cost == old_cost
    assert packs[0]['background'] != packs[1]['background']
    for pack in packs:
        assert set(pack['background']) == set(lab.spec.feature_names)
        assert any(abs(value) > .01 for value in pack['background'].values())
        assert {trial['control'] for trial in pack['interventions']} == set(lab.spec.feature_names)
        assert pack['observation_cost'] == sum(probe.cost for probe in
            [pack['reference'], *(trial['probe'] for trial in pack['interventions'])])
        for trial in pack['interventions']:
            assert trial['mode'] in {'step', 'pulse', 'sweep'}
            assert trial['prefix_length'] > 0
        for probe in [pack['reference'], *(trial['probe'] for trial in pack['interventions'])]:
            assert 0 < probe.cost <= lab.spec.max_probe_length
            assert all(set(point) == set(lab.spec.feature_names)
                and all(low <= point[name] <= high for name, (low, high) in lab.spec.feature_bounds.items())
                for point in probe.sequence)
    assert sum(pack['observation_cost'] for pack in packs) <= 96
    assert 0 < work.spent <= work.limit


def test_an_unfunded_pack_is_not_silently_split_into_unmatched_observations():
    lab = UnspecifiedBranchLab('inside_window', 7)
    work = SearchBudget(6000)
    assert plan_experiments(lab, [], work, observation_budget=47) == []
    assert lab.observation_cost == 0
    tiny = SearchBudget(1)
    assert plan_experiments(lab, [], tiny) == []
    assert tiny.spent <= tiny.limit


def public_setting_rows(lab, seed, *, length=64):
    rng = Random(seed)
    fractions = (-1., -2 / 3, -1 / 3, 0., 1 / 3, 2 / 3, 1.)
    probe = Probe(f'unseen_public_settings_{seed}', tuple({name:
        (low + high) / 2 + (high - low) / 2 * rng.choice(fractions)
        for name, (low, high) in lab.spec.feature_bounds.items()} for _ in range(length)))
    return lab.acquire(probe).observations


@pytest.mark.parametrize('mode,width,gate_axis', [
    ('outside_window', 3, 0), ('inside_window', 5, 2), ('outside_window', 7, 1)])
def test_matched_measured_effects_identify_hidden_gate_window_and_order(mode, width, gate_axis):
    from hegel_machine.cognition.conditional_coordinates import propose_experimental_conditional_coordinates
    lab = UnspecifiedBranchLab(mode, width, gate_axis=gate_axis)
    pilot = paid(lab, (5101,), length=24)
    plan_budget = SearchBudget(6000)
    packs = plan_experiments(lab, pilot, plan_budget)
    assessment_budget = SearchBudget(48000)
    measured, evidence = measure_experiments(lab, packs, assessment_budget)
    assert lab.observation_cost == len(pilot) + sum(pack['observation_cost'] for pack in packs)
    discovered = [entry for entry in evidence
        if entry['control'] == lab.spec.feature_names[gate_axis] and entry.get('gain', 0.) > 1e-10]
    assert discovered
    assert all(entry['status'] == 'smooth' and entry['gain'] == 0.
               for entry in evidence if entry['control'] != lab.spec.feature_names[gate_axis])
    assert any(width in entry.get('width_candidates', [entry['width']])
               and mode in entry.get('placement_candidates', [entry['placement']])
               for entry in discovered)
    assert any(entry['threshold_interval'][0] <= .173 <= entry['threshold_interval'][1]
               for entry in discovered)
    work = SearchBudget(120000)
    claims = propose_experimental_conditional_coordinates(lab.spec, [*pilot, *measured], work,
        basis=measured_basis(lab.spec), evidence=evidence)
    assert claims
    best = min(claims, key=lambda claim: claim['training_loss'])
    step = latest_condition(best['research_program'])
    assert step['width'] == width and step['placement'] == mode
    assert step['gate_coordinate'] == Program('feature', name=lab.spec.feature_names[gate_axis]).to_dict()
    # The same public setting grid is used for every input, independently of its
    # private physical role. These new trajectories assess response composition.
    independent = UnspecifiedBranchLab(mode, width, gate_axis=gate_axis)
    fresh = [*public_setting_rows(independent, 85101), *public_setting_rows(independent, 86110)]
    assert loss(best['candidate'].program, fresh, 0., .5) < 1e-10
    assert 0 < plan_budget.spent <= plan_budget.limit
    assert 0 < assessment_budget.spent <= assessment_budget.limit
    assert 0 < work.spent <= work.limit


CONFIG = ResearchConfig(observation_budget=240, compute_budget=240000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)


@pytest.mark.parametrize('mode,width,gate_axis', [('outside_window', 5, 0), ('inside_window', 7, 2)])
def test_empty_agent_uses_actual_matched_experiments_to_learn_complex_branches(mode, width, gate_axis, tmp_path):
    lab = UnspecifiedBranchLab(mode, width, gate_axis=gate_axis)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9315101),
        library=KnowledgeLibrary(), measurement_budget=0).run(lab)
    saved = tmp_path / 'research.json'
    saved.write_text(json.dumps(report))
    assert report['research_program'] is not None, str(saved)
    step = latest_condition(report['research_program'])
    assert step['experiment_evidence'], str(saved)
    assert step['width'] == width and step['placement'] == mode, str(saved)
    assert report['costs']['observation_cost'] == lab.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(lab.acquisitions)
    assert report['costs']['measurement_cost'] == 0
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget
    refinements = boundary_payments(report, lab)
    if mode == 'outside_window':
        assert refinements, str(saved)
    independent = UnspecifiedBranchLab(mode, width, gate_axis=gate_axis)
    assert fresh_error(report, independent) < .05, str(saved)


def test_json_transferred_agent_rewrites_experiments_after_controls_scale_and_mechanism_change(tmp_path):
    library = KnowledgeLibrary()
    source = UnspecifiedBranchLab('outside_window', 3, gate_axis=0)
    first = UnifiedResearchAgent(replace(CONFIG, seed=9315101),
        library=library, measurement_budget=0).run(source)
    assert first['research_program'] is not None
    stored = json.loads(json.dumps(library.to_dict()))
    restored = KnowledgeLibrary.from_dict(stored)
    changed = UnspecifiedBranchLab('inside_window', 7, gate_axis=0,
        names=('p', 'q', 'r'), control_scale=2.3, response_scale=1.25,
        threshold=.413, task_id='unlabelled_apparatus_episode_2')
    report = UnifiedResearchAgent(replace(CONFIG, seed=9316101),
        library=restored, measurement_budget=0).run(changed)
    saved = tmp_path / 'changed_research.json'
    saved.write_text(json.dumps(report))
    assert report['research_program'] is not None, str(saved)
    step = latest_condition(report['research_program'])
    assert step['width'] == 7 and step['placement'] == 'inside_window', str(saved)
    assert Program.from_dict(report['program']).required_features() <= set(('p', 'q', 'r'))
    assert report['costs']['observation_cost'] == changed.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(changed.acquisitions)
    assert report['costs']['measurement_cost'] == 0
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget
    assert boundary_payments(report, changed), str(saved)
    independent = UnspecifiedBranchLab('inside_window', 7, gate_axis=0,
        names=('p', 'q', 'r'), control_scale=2.3, response_scale=1.25,
        threshold=.413, task_id='independent_episode_2')
    assert fresh_error(report, independent, seed=95101) < .05, str(saved)


def test_json_restored_unfinished_task_buys_and_validates_new_conditional_experiments(tmp_path):
    limited = replace(CONFIG, observation_budget=96, compute_budget=24000, seed=9315101)
    library = KnowledgeLibrary()
    laboratory = UnspecifiedBranchLab('outside_window', 5, gate_axis=0)
    first = UnifiedResearchAgent(limited, library=library, measurement_budget=0).run(laboratory)
    (tmp_path / 'unfinished_research.json').write_text(json.dumps(first))
    assert first['status'] == 'unresolved'
    assert not any(action['action'] == 'research_conditional_matched_experiments'
                   for action in first['actions'])
    stored = json.loads(json.dumps(first))
    restored_lab = UnspecifiedBranchLab('outside_window', 5, gate_axis=0)
    restored_lab.restore([batch['probe'] for batch in stored['batches']])
    assert restored_lab.observation_cost == stored['costs']['observation_cost']
    prior_queries = len(restored_lab.acquisitions)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9315101),
        library=KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict()))),
        measurement_budget=0).run(restored_lab, previous=stored)
    saved = tmp_path / 'continued_research.json'
    saved.write_text(json.dumps(report))
    added_actions = report['actions'][len(stored['actions']):]
    assert any(action['action'] == 'research_conditional_matched_experiments'
               for action in added_actions), str(saved)
    assert report['status'] == 'completed', str(saved)
    assert report['research_program'] is not None, str(saved)
    step = latest_condition(report['research_program'])
    assert step['width'] == 5 and step['placement'] == 'outside_window', str(saved)
    assert boundary_payments(report, restored_lab), str(saved)
    assert 0 < report['visit_costs']['observation_cost'] == (
        restored_lab.observation_cost - stored['costs']['observation_cost'])
    assert report['visit_costs']['queries'] == len(restored_lab.acquisitions) - prior_queries
    assert report['costs']['observation_cost'] == restored_lab.observation_cost <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(restored_lab.acquisitions)
    assert stored['costs']['compute_cost'] < report['costs']['compute_cost'] <= CONFIG.compute_budget
    graph = ResearchGraph.from_dict(json.loads(json.dumps(report['research_program']['graph'])))
    output = graph.nodes[report['research_program']['output_node']]
    compiled = bind_answers(output.expression, {key: graph.nodes[key].answer for key in output.depends_on})
    fresh = paid(UnspecifiedBranchLab('outside_window', 5, gate_axis=0), (75101, 76110), length=64)
    actual = Program.from_dict(report['program'])
    assert loss(actual, fresh, report['fallback'], .5) < .05, str(saved)
    assert [compiled.predict(row) for row in fresh] == pytest.approx([actual.predict(row) for row in fresh])
    assert [output.answer.predict(row) for row in fresh] == pytest.approx([actual.predict(row) for row in fresh])


def test_extra_pilot_does_not_spend_the_funding_for_measured_boundary_and_confirmation(tmp_path):
    laboratory = UnspecifiedBranchLab('outside_window', 5, gate_axis=0)
    fitting = paid(laboratory, (5101, 6101), length=24)
    validation = paid(laboratory, (7101,), length=24)
    extra_pilot = paid(laboratory, (8101,), length=24)
    assert laboratory.observation_cost == 96
    training = [*fitting, *extra_pilot]
    baseline = Program('const', value=0.)
    work = SearchBudget(CONFIG.compute_budget)
    assert work.spend(len(training) + len(validation))
    initial = {'candidate': Candidate('uninformed_baseline', baseline, 'initial'),
        'kind': 'initial', 'family_id': 'T03',
        'training_loss': loss(baseline, training, 0., .5),
        'validation_loss': loss(baseline, validation, 0., .5)}
    actions = []

    def acquire(channels, purpose, *, probe):
        assert not channels
        assert laboratory.observation_cost + probe.cost <= 192
        batch = laboratory.acquire(probe)
        actions.append({'action': 'acquire', 'purpose': purpose, 'probe_id': probe.probe_id,
            'observation_cost': batch.observation_cost, 'measurement_cost': batch.measurement_cost})
        return batch.observations

    selected, _, _ = UnifiedResearchAgent(CONFIG, library=KnowledgeLibrary(),
        measurement_budget=0)._research_conditional_experiments(laboratory.spec, initial,
            training, validation, [], work, actions, acquire=acquire,
            observation_allowance=96, serial=len(laboratory.acquisitions))
    matched = [action for action in actions if action.get('purpose') in
               ('conditional_matched_reference', 'conditional_matched_intervention')]
    assert len(matched) == 4 and sum(action['observation_cost'] for action in matched) == 48
    refinements = boundary_payments({'actions': actions}, laboratory)
    assert len(refinements) == 1 and refinements[0]['observation_cost'] == 12
    assert laboratory.observation_cost == 156
    confirmation = paid(laboratory, (9101,), length=24)
    assert work.spend(len(confirmation))
    confirmation_loss = loss(selected['candidate'].program, confirmation, 0., .5)
    independent = UnspecifiedBranchLab('outside_window', 5, gate_axis=0)
    future = paid(independent, (85101, 86110), length=64)
    fresh_loss = loss(selected['candidate'].program, future, 0., .5)
    saved = tmp_path / 'extra_pilot_research.json'
    saved.write_text(json.dumps({'program': selected['candidate'].program.to_dict(),
        'research_program': selected.get('research_program'), 'actions': actions,
        'costs': {'observation_cost': laboratory.observation_cost, 'compute_cost': work.spent},
        'confirmation_loss': confirmation_loss, 'fresh_loss': fresh_loss}))
    assert confirmation_loss < CONFIG.minimum_improvement, str(saved)
    assert fresh_loss < .05, str(saved)
    assert laboratory.observation_cost == 180 <= CONFIG.observation_budget
    assert 0 < work.spent <= CONFIG.compute_budget

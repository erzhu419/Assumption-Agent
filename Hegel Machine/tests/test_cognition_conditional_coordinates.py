"""Paid controls identify conditional response, temporal order, and new boundaries."""

from dataclasses import asdict, replace
import json
from random import Random

import pytest

from hegel_machine.cognition.conditional_coordinates import (
    propose_conditional_coordinates, propose_experimental_conditional_coordinates,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.knowledge_evolution import bind_answers
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.mechanism_revision import propose_response_refits
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import replay_research_program
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class ConditionalLab:
    channels = ()

    def __init__(self, episode, mode, width, *, threshold=.15,
                 names=('drive', 'load', 'gate'), gains=((.8, -.4), (-.3, 1.1)),
                 discrete_gate=True, gate_gains=(0., 0.)):
        self.spec = TaskSpec(episode, 'Predict the measured response to controllable inputs',
            names, (), (), feature_bounds={name: (-1., 1.) for name in names},
            max_probe_length=64)
        self._mode, self._width, self._threshold = mode, width, threshold
        self._names, self._gains, self._discrete_gate = names, gains, discrete_gate
        self._gate_gains = gate_gains
        self.acquisitions = []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def random_probe(self, seed, *, length=32, name='purchased'):
        rng = Random(seed)
        points = []
        for _ in range(length):
            point = {key: rng.uniform(-1., 1.) for key in self.spec.feature_names}
            if self._discrete_gate:
                # Public control levels, independent of this apparatus's boundary.
                point[self._names[2]] = rng.choice((-.9, -.6, -.3, 0., .3, .6, .9))
            points.append(point)
        return Probe(name, tuple(points))

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, branches, selected_values = [], [], [[], []], []
        for point in probe.sequence:
            active = 0 if point[self._names[2]] >= self._threshold else 1
            for index, (left, right) in enumerate(self._gains):
                response = (left * point[self._names[0]] + right * point[self._names[1]] ** 2 + .2
                            if self._mode == 'plain_window' else
                            left * point[self._names[0]] + right * point[self._names[1]])
                branches[index].append(response + self._gate_gains[index] * point[self._names[2]])
            selected_values.append(branches[active][-1])
            if self._mode == 'gate_outer':
                target = sum(branches[active][-self._width:])
            elif self._mode == 'temporal_outer':
                target = sum(selected_values[-self._width:])
            elif self._mode == 'no_branch':
                target = branches[0][-1]
            elif self._mode == 'plain_window':
                target = sum(branches[0][-self._width:])
            else:
                raise ValueError(self._mode)
            rows.append(Observation(dict(point), .17 + target, tuple(history)))
            history.append({**point, '__target__': .17 + target})
        self.acquisitions.append({'probe': probe, 'observations': rows})
        return MeasurementBatch(rows, len(rows), 0, ())


def purchased(lab, seeds=(701, 1709), *, length=32):
    batches = [lab.acquire(lab.random_probe(seed, length=length, name=f'paid_{seed}'))
               for seed in seeds]
    assert sum(batch.observation_cost for batch in batches) == len(seeds) * length
    assert all(batch.measurement_cost == 0 for batch in batches)
    return [row for batch in batches for row in batch.observations]


def basis(spec):
    return [Program('feature', name=name) for name in spec.feature_names]


def conditional_step(claim):
    return next(step for step in claim['research_program']['selection_steps']
                if step['operation'] == 'discover_conditional_coordinates')


def contains(program, operation):
    return program.op == operation or any(contains(child, operation) for child in program.children)


@pytest.mark.parametrize('mode,width', [('gate_outer', 3), ('temporal_outer', 5)])
def test_paid_trajectories_discover_branch_temporal_order(mode, width):
    lab = ConditionalLab('unlabelled_apparatus', mode, width)
    training = purchased(lab)
    work = SearchBudget(120000)
    claims = propose_conditional_coordinates(lab.spec, training, work, basis=basis(lab.spec))
    assert claims
    best = min(claims, key=lambda claim: (claim['training_loss'], claim['candidate'].program.size))
    plan, program = best['research_program'], best['candidate'].program
    assert plan['status'] == 'requires_fresh_validation'
    assert not best['candidate'].parents
    assert contains(program, 'ge') and contains(program, 'window_sum')
    assert conditional_step(best)['width'] == width
    left, right = conditional_step(best)['threshold_interval']
    assert left < .15 < right
    assert conditional_step(best)['placement'] == (
        'outside_window' if mode == 'gate_outer' else 'inside_window')
    assert all(trial['compute_cost'] > 0 for trial in conditional_step(best)['parameter_trials'])
    assert sum(trial['compute_cost'] for trial in conditional_step(best)['parameter_trials']) <= work.spent
    assert 0 < work.spent <= work.limit
    assert plan['search_compute'] == work.spent
    evaluation = ConditionalLab('independent_response', mode, width)
    fresh = purchased(evaluation, (73117, 81301), length=64)
    assert loss(program, fresh, 0., .5) < 1e-10


@pytest.mark.parametrize('source_mode,changed_mode', [
    ('gate_outer', 'gate_outer'), ('temporal_outer', 'temporal_outer'),
    ('gate_outer', 'temporal_outer')])
def test_saved_procedure_researches_changed_gate_window_and_response_after_rename(source_mode, changed_mode):
    source = ConditionalLab('source_response', source_mode, 3)
    training = purchased(source)
    claims = propose_conditional_coordinates(source.spec, training, SearchBudget(120000),
                                             basis=basis(source.spec))
    first = min(claims, key=lambda claim: claim['training_loss'])
    saved = json.loads(json.dumps(first['research_program']))
    changed = ConditionalLab('changed_response', changed_mode, 5, threshold=-.45,
        names=('u', 'v', 'switch'), gains=((-1.2, .6), (.5, -.7)))
    evidence = purchased(changed, (1093, 2089))
    work = SearchBudget(120000)
    revised = replay_research_program(saved, evidence, work,
        feature_mapping={'drive': 'u', 'load': 'v', 'gate': 'switch'})
    assert revised is not None and revised['research_program']['reused_procedure']
    assert revised['research_program']['status'] == 'requires_fresh_validation'
    assert conditional_step(revised)['width'] == 5
    assert conditional_step(revised)['placement'] == (
        'outside_window' if changed_mode == 'gate_outer' else 'inside_window')
    assert conditional_step(revised)['threshold'] != conditional_step(first)['threshold']
    left, right = conditional_step(revised)['threshold_interval']
    assert left < -.45 < right
    assert revised['candidate'].program.required_features() == set(changed.spec.feature_names)
    fresh = purchased(ConditionalLab('independent_changed', changed_mode, 5, threshold=-.45,
        names=('u', 'v', 'switch'), gains=((-1.2, .6), (.5, -.7))),
        (73117, 81301), length=64)
    old = first['candidate'].program.remap_features({'drive': 'u', 'load': 'v', 'gate': 'switch'})
    assert loss(old, fresh, 0., .5) > .1
    assert loss(revised['candidate'].program, fresh, 0., .5) < 1e-10
    assert 0 < work.spent <= work.limit


def test_unfunded_conditional_search_cannot_report_an_unpaid_candidate():
    lab = ConditionalLab('unlabelled_response', 'temporal_outer', 5)
    training = purchased(lab)
    work = SearchBudget(1)
    assert propose_conditional_coordinates(lab.spec, training, work, basis=basis(lab.spec)) == []
    assert work.spent == 0
    limited = SearchBudget(1700)
    claims = propose_conditional_coordinates(lab.spec, training, limited, basis=basis(lab.spec))
    assert limited.spent <= limited.limit
    assert all(claim['research_program']['search_compute'] <= limited.spent for claim in claims)


def measured_condition_search(*, threshold=.37, discrete_gate=False, budget=None):
    lab = ConditionalLab('paid_branch_response', 'temporal_outer', 5,
                         threshold=threshold, discrete_gate=discrete_gate)
    rows = purchased(lab, (5,), length=32)
    evidence = [{'control': 'gate', 'gain': 1., 'width': 5,
                 'placement': 'inside_window', 'threshold_interval': [-1., 1.]}]
    work = budget if budget is not None else SearchBudget(100000)
    claims = propose_experimental_conditional_coordinates(
        lab.spec, rows, work, basis=basis(lab.spec), evidence=evidence)
    return lab, rows, work, min(claims, key=lambda claim: claim['training_loss'])


def test_experimental_threshold_search_escapes_paid_local_minimum():
    # These paid trajectories give a nonunimodal joint-fit loss. Local
    # refinement alone stops at MSE .0209988 despite an exact available fit.
    lab, rows, work, best = measured_condition_search()
    assert best['training_loss'] < 1e-12
    assert loss(best['candidate'].program, rows, 0., .5) < 1e-12
    step = conditional_step(best)
    assert step['threshold_interval'][0] < .37 < step['threshold_interval'][1]
    assert len(step['parameter_trials']) <= len({row.features['gate'] for row in rows}) - 1
    assert len(lab.acquisitions) == 1
    assert best['research_program']['search_compute'] == work.spent <= work.limit


def test_experimental_threshold_fallback_respects_remaining_budget():
    work = SearchBudget(3700, reserve=200)
    _, rows, _, best = measured_condition_search(budget=work)
    assert best['training_loss'] > 1e-3
    assert 0 < work.spent <= work.limit - work.reserve
    assert work.reserve == 200
    assert len(conditional_step(best)['parameter_trials']) < len(rows) - 1


def test_experimental_threshold_exact_fit_does_not_expand_remaining_brackets():
    _, _, work, best = measured_condition_search(threshold=0., discrete_gate=True)
    assert best['training_loss'] < 1e-12
    # The three initial brackets already explain the paid data; the other
    # brackets remain untested even with a large available search allowance.
    assert len(conditional_step(best)['parameter_trials']) == 3
    assert work.spent < work.limit // 10


def test_discovered_boundaries_propose_a_new_legal_paid_control_trajectory():
    lab = ConditionalLab('unlabelled_control_response', 'gate_outer', 3)
    claims = propose_conditional_coordinates(lab.spec, purchased(lab), SearchBudget(120000),
                                             basis=basis(lab.spec))
    work, actions = SearchBudget(12000), []
    agent = UnifiedResearchAgent(ResearchConfig(seed=9981901), library=KnowledgeLibrary(),
                                 measurement_budget=0)
    probe = agent._probe(lab.spec, 3, 12, claims, work, actions)
    proposed = next(event for event in actions if event['action'] == 'propose_conditional_experiments')
    assert proposed['alternatives'] > 0
    assert proposed['boundary_intervals']
    assert 0 < proposed['compute_cost'] <= 3000
    assert proposed['compute_cost'] < work.spent <= work.limit
    assert len(probe.sequence) == 12
    assert all(set(point) == set(lab.spec.feature_names)
        and all(low <= point[name] <= high for name, (low, high) in lab.spec.feature_bounds.items())
        for point in probe.sequence)
    batch = lab.acquire(probe)
    assert batch.observation_cost == 12 and batch.measurement_cost == 0
    assert len(lab.acquisitions) == 3


def test_paid_mechanism_deletion_saves_its_current_graph_and_can_research_another_task():
    source = ConditionalLab('original_apparatus', 'gate_outer', 3)
    claims = propose_conditional_coordinates(source.spec, purchased(source), SearchBudget(120000),
                                             basis=basis(source.spec))
    parent = min(claims, key=lambda claim: claim['training_loss'])
    changed = ConditionalLab('changed_apparatus', 'gate_outer', 3,
                             gains=((.8, 0.), (-.3, 1.1)))
    work = SearchBudget(8000)
    revisions = propose_response_refits(changed.spec, purchased(changed, (3191, 4297)),
                                       work, parent=parent, max_candidates=8)
    deleted = min((claim for claim in revisions if claim['kind'] == 'mechanism_deletion'),
                  key=lambda claim: claim['training_loss'])
    assert deleted['training_loss'] < 1e-10
    saved = json.loads(json.dumps(deleted['research_program']))
    assert saved['status'] == 'requires_fresh_validation'
    assert saved['selection_steps'][-1]['operation'] == 'revise_conditional_response'
    graph = ResearchGraph.from_dict(saved['graph'])
    output = graph.nodes[saved['output_node']]
    compiled = bind_answers(output.expression,
        {key: graph.nodes[key].answer for key in output.depends_on})
    fresh = purchased(ConditionalLab('independent_changed_response', 'gate_outer', 3,
        gains=((.8, 0.), (-.3, 1.1))), (73117,), length=64)
    assert loss(deleted['candidate'].program, fresh, 0., .5) < 1e-10
    assert [compiled.predict(row) for row in fresh] == pytest.approx(
        [deleted['candidate'].program.predict(row) for row in fresh])
    assert [output.answer.predict(row) for row in fresh] == pytest.approx(
        [compiled.predict(row) for row in fresh])
    another = ConditionalLab('another_apparatus', 'temporal_outer', 5, threshold=-.45,
        names=('u', 'v', 'switch'), gains=((-1.2, 0.), (.5, -.7)))
    replay_work = SearchBudget(120000)
    replayed = replay_research_program(saved, purchased(another, (1093, 2089)), replay_work,
        feature_mapping={'drive': 'u', 'load': 'v', 'gate': 'switch'})
    assert replayed is not None and replayed['research_program']['reused_procedure']
    future = purchased(ConditionalLab('independent_another_response', 'temporal_outer', 5,
        threshold=-.45, names=('u', 'v', 'switch'), gains=((-1.2, 0.), (.5, -.7))),
        (73117,), length=64)
    assert loss(replayed['candidate'].program, future, 0., .5) < 1e-10
    assert 0 < work.spent <= work.limit and 0 < replay_work.spent <= replay_work.limit


def test_adopted_successor_saves_current_graph_and_replays_its_actual_terms():
    source = ConditionalLab('earlier_apparatus', 'gate_outer', 1, threshold=-.45)
    initial = propose_conditional_coordinates(source.spec, purchased(source, (901, 1901), length=16),
        SearchBudget(64000), basis=basis(source.spec))
    old = min(initial, key=lambda claim: claim['training_loss'])
    successor_lab = ConditionalLab('successor_apparatus', 'gate_outer', 1,
        gains=((1.3, -.6), (-.4, .8)), gate_gains=(.35, -.25))
    learned_rows = purchased(successor_lab, (2903, 3907), length=16)
    successor_claims = propose_conditional_coordinates(successor_lab.spec, learned_rows,
        SearchBudget(64000), basis=basis(successor_lab.spec))
    successor = min(successor_claims, key=lambda claim: claim['training_loss'])
    library = KnowledgeLibrary()
    source_id, _ = library.learn(successor['candidate'], successor_lab.spec, learned_rows)
    lab = ConditionalLab('earlier_task_revisited', 'gate_outer', 1,
        gains=((1.3, -.6), (-.4, .8)), gate_gains=(.35, -.25))
    lab.spec = replace(lab.spec, max_probe_length=12)
    probe = lab.random_probe(4909, length=12, name='earlier_paid')
    batch = lab.acquire(probe)
    previous = {'task_id': lab.spec.task_id, 'status': 'unresolved',
        'program': old['candidate'].program.to_dict(), 'fallback': 0.,
        'selected_channels': [], 'selected_kind': 'research_program',
        'research_program': json.loads(json.dumps(old['research_program'])),
        'actions': [], 'batches': [{'probe': asdict(probe), 'channels': [],
            'observations': [asdict(row) for row in batch.observations], 'transitions': []}],
        'costs': {'observation_cost': 12, 'measurement_cost': 0, 'compute_cost': 0, 'queries': 1}}
    proposal = {'program': successor['candidate'].program.to_dict(), 'channels': [],
        'source_concept_ids': [source_id], 'kind': 'successor_knowledge', 'fallback': 0.}
    config = ResearchConfig(observation_budget=60, compute_budget=24000, max_rounds=1,
        max_candidates=4, minimum_improvement=1e-7, active_probes=False, seed=5903)
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab,
        previous=previous, continuation_candidate=proposal)
    assert report['status'] == 'completed'
    assert source_id in report['continuation']['source_concept_ids']
    actual = Program.from_dict(report['program'])
    assert actual != old['candidate'].program
    saved = json.loads(json.dumps(report['research_program']))
    graph = ResearchGraph.from_dict(saved['graph'])
    output = graph.nodes[saved['output_node']]
    compiled = bind_answers(output.expression,
        {key: graph.nodes[key].answer for key in output.depends_on})
    fresh = purchased(ConditionalLab('unseen_revisited_response', 'gate_outer', 1,
        gains=((1.3, -.6), (-.4, .8)), gate_gains=(.35, -.25)), (6907,), length=24)
    assert loss(actual, fresh, report['fallback'], .5) < 1e-10
    assert [compiled.predict(row) for row in fresh] == pytest.approx([actual.predict(row) for row in fresh])
    assert [output.answer.predict(row) for row in fresh] == pytest.approx([actual.predict(row) for row in fresh])
    prior = next(prior for prior in library.meta_priors if prior.get('research_program')
        and Program.from_dict(prior['research_program']['graph']['nodes']['answer']['answer']) == actual)
    stored = ResearchGraph.from_dict(prior['research_program']['graph'])
    assert [stored.nodes[prior['research_program']['output_node']].answer.predict(row)
            for row in fresh] == pytest.approx([actual.predict(row) for row in fresh])
    replay_rows = purchased(ConditionalLab('another_paid_response', 'gate_outer', 1,
        gains=((1.3, -.6), (-.4, .8)), gate_gains=(.35, -.25)), (7907,), length=24)
    replay_work = SearchBudget((len(saved['term_nodes']) + 2) * len(replay_rows))
    replayed = replay_research_program(saved, replay_rows, replay_work)
    assert replayed is not None
    future = purchased(ConditionalLab('unseen_replayed_response', 'gate_outer', 1,
        gains=((1.3, -.6), (-.4, .8)), gate_gains=(.35, -.25)), (8909,), length=24)
    assert loss(replayed['candidate'].program, future, 0., .5) < 1e-10
    assert 0 < replay_work.spent <= replay_work.limit
    assert report['costs']['observation_cost'] == sum(len(item['observations']) for item in lab.acquisitions)


CONFIG = ResearchConfig(observation_budget=240, compute_budget=240000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)


@pytest.mark.parametrize('mode', ['gate_outer', 'temporal_outer'])
def test_empty_agent_actually_researches_conditional_temporal_response(mode, tmp_path):
    lab = ConditionalLab('measured_response', mode, 3, discrete_gate=False)
    library = KnowledgeLibrary()
    report = UnifiedResearchAgent(replace(CONFIG, seed=9981701),
        library=library, measurement_budget=0).run(lab)
    saved = tmp_path / 'research.json'
    saved.write_text(json.dumps(report))
    assert report['costs']['observation_cost'] == sum(len(item['observations'])
        for item in lab.acquisitions) <= CONFIG.observation_budget
    assert report['costs']['queries'] == len(lab.acquisitions)
    assert report['costs']['measurement_cost'] == 0
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget
    assert any(event['action'] == 'propose_conditional_reconstruction' and event['alternatives']
               for event in report['actions'])
    assert contains(Program.from_dict(report['program']), 'ge'), str(saved)
    assert report['research_program'] is not None, str(saved)
    assert conditional_step(report)['placement'] == (
        'outside_window' if mode == 'gate_outer' else 'inside_window')
    # Finite random pilot controls locate an interval around the actual boundary;
    # a new continuous trajectory tests its achieved predictive accuracy.
    fresh = purchased(ConditionalLab('unseen_response', mode, 3, discrete_gate=False),
                      (73117, 81301), length=64)
    assert loss(Program.from_dict(report['program']), fresh, report['fallback'], .5) < .08


def test_conditional_search_keeps_a_better_simple_explanation_when_no_branch_exists():
    lab = ConditionalLab('measured_linear_response', 'no_branch', 1)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9981803),
        library=KnowledgeLibrary(), measurement_budget=0).run(lab)
    program = Program.from_dict(report['program'])
    assert report['status'] == 'completed'
    assert not contains(program, 'ge')
    fresh = purchased(ConditionalLab('unseen_linear_response', 'no_branch', 1),
                      (73117,), length=64)
    assert loss(program, fresh, report['fallback'], .5) < 1e-10
    assert report['costs']['observation_cost'] == sum(len(item['observations'])
        for item in lab.acquisitions) <= CONFIG.observation_budget
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget


@pytest.mark.parametrize('width', [3, 5])
def test_empty_agent_keeps_a_joint_nonlinear_window_without_inventing_a_condition(width, tmp_path):
    lab = ConditionalLab('measured_common_response', 'plain_window', width,
                         discrete_gate=False)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9981701),
        library=KnowledgeLibrary(), measurement_budget=0).run(lab)
    saved = tmp_path / 'research.json'
    saved.write_text(json.dumps(report))
    program = Program.from_dict(report['program'])
    assert report['status'] == 'completed', str(saved)
    assert contains(program, 'window_sum') and not contains(program, 'ge')
    fresh = purchased(ConditionalLab('unseen_common_response', 'plain_window', width,
        discrete_gate=False), (73117, 81301), length=64)
    assert loss(program, fresh, report['fallback'], .5) < 1e-10
    assert report['costs']['observation_cost'] == sum(len(item['observations'])
        for item in lab.acquisitions) <= CONFIG.observation_budget
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget

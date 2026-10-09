"""A paid target counterexample withdraws the bound law, preserving its source."""

from dataclasses import asdict, replace
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.noise_replication import assess_noise_replication
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_noise_research_reopening import laboratory


def target(name, feature='x'):
    return TaskSpec(name, 'Predict the measured response', (feature,), (), (),
                    feature_bounds={feature: (-1., 1.)})


def measured_rows(feature='x', *, gain=1., seed=17, noise=0.):
    rng, rows, history = Random(seed), [], []
    for value in (-1., -.5, 0., .5, 1.) * 4:
        point = {feature: value}
        response = gain * value ** 2 + rng.gauss(0., noise)
        rows.append(Observation(point, response, tuple(history)))
        history.append({**point, '__target__': response})
    return rows


def learned_source():
    library = KnowledgeLibrary()
    x = Program('feature', name='x')
    program = Program('mul', children=(x, x))
    source_id, created = library.learn(Candidate('measured_square', program, 'refit'),
                                      target('source'), measured_rows())
    assert created
    return library, source_id, program


def counterexample(program, first, repeated):
    budget = SearchBudget(1000)
    evidence = assess_noise_replication(program, first, repeated, budget)
    assert evidence['compute_cost'] == budget.spent > 0
    return {'status': 'counterexample', 'rows': len(first),
            'measured_loss': loss(program, first, 0., .5), 'noise_replication': evidence}


def restored(library):
    return KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))


def test_known_target_conflict_cannot_return_through_preserved_parameter_transfer():
    library, source_id, program = learned_source()
    spec, rows = target('changed'), measured_rows(gain=2.)
    event = library.record_counterexample(source_id, spec, loss(program, rows, 0., .5), 0., rows[0])
    assert event['scope_change'] == 'restricted_context' and not library.retrieve(spec)
    candidates = library.propose(spec, rows, SearchBudget(500))
    assert candidates and all(candidate.program.expanded().simplified() != program for candidate in candidates)
    assert min(loss(candidate.program, rows, 0., .5) for candidate in candidates) < 1e-7
    assert library.retrieve(target('source'))[0].program == program


def test_real_repeat_withdraws_only_the_consumed_bound_program_after_json():
    library, source_id, program = learned_source()
    x = Program('feature', name='x')
    unrelated, _ = library.learn(Candidate('other', Program('abs', children=(x,)), 'refit'), target('other'))
    spec = target('changed', 'u')
    call = next(atom for atom in library.callable_atoms(spec)
                if atom.definition.source_concept_id == source_id)
    first, repeated = measured_rows('u', gain=2., noise=.001), measured_rows('u', gain=2., noise=.001, seed=71)
    outcome = counterexample(call, first, repeated)
    assert outcome['noise_replication']['repeatable_error']
    before = library.to_dict()
    event = library.record_transfer_counterexample(spec, call, [source_id, unrelated],
                                                   evidence=outcome, probe_id='paid_repeat')
    assert event['source_ids'] == [source_id]
    assert not next(c for c in library.concepts if c.concept_id == unrelated).reuse_history
    library = restored(library)
    assert library.transfer_refuted(spec, call) and library.transfer_refuted(spec, call.expanded())
    assert not library.transfer_refuted(target('new_target', 'u'), call)
    assert not library.transfer_refuted(target('source'), program)
    refitted = Program('mul', children=(Program('const', value=2.), call))
    assert not library.transfer_refuted(spec, refitted)
    assert loss(refitted, measured_rows('u', gain=2.), 0., .5) < 1e-7
    assert call in library.callable_atoms(spec)  # Still usable inside a revised response.
    assert library.to_dict()['definitions'] == before['definitions']
    source = next(c for c in library.concepts if c.concept_id == source_id)
    assert source.status == 'available' and not source.excluded_tasks
    assert library.record_transfer_counterexample(spec, call, [source_id],
        evidence=outcome, probe_id='same_failure_again') is None


@pytest.mark.parametrize('status', ['unresolved', 'counterexample'])
def test_observed_noise_without_repeatable_error_cannot_withdraw_a_transfer(status):
    library, source_id, program = learned_source()
    first, repeated = measured_rows(noise=.01), measured_rows(noise=.01, seed=71)
    outcome = counterexample(program, first, repeated)
    assert outcome['noise_replication']['noise_supported']
    assert not outcome['noise_replication']['repeatable_error']
    outcome['status'] = status
    before = library.to_dict()
    assert library.record_transfer_counterexample(target('changed'), program, [source_id],
        evidence=outcome, probe_id='noise_repeat') is None
    assert library.to_dict() == before


def test_withdrawal_ablation_saves_actual_screening_without_removing_refit_knowledge(record_property):
    library, source_id, program = learned_source()
    spec, rows = target('changed'), measured_rows(gain=2.)
    outcome = counterexample(program, rows, measured_rows(gain=2., seed=71))
    event = library.record_transfer_counterexample(spec, program, [source_id],
        evidence={key: value for key, value in outcome.items() if key != 'noise_replication'}, probe_id='deterministic')
    assert event is not None
    ablated = restored(library)
    for concept in ablated.concepts:
        concept.reuse_history = [item for item in concept.reuse_history if item.get('kind') != 'transfer_counterexample']
    results = []
    for arm in (library, ablated):
        work = SearchBudget(500)
        candidates = arm.propose(spec, rows, work)
        results.append((candidates, work.spent))
    scoped, without_scope = results
    assert all(candidate.program.expanded().simplified() != program for candidate in scoped[0])
    assert any(candidate.program.expanded().simplified() == program for candidate in without_scope[0])
    assert scoped[1] < without_scope[1] <= 500
    # Identical terminal measurements follow both searches; no target is fed
    # to one arm alone and no source concept is removed by this intervention.
    terminal = [Observation({'x': value}, 2. * value ** 2) for value in (-.8, -.3, .2, .7)]
    for candidates, _ in results:
        assert min(loss(candidate.program, terminal, 0., .5) for candidate in candidates) < 1e-7
    record_property('scoped_compute', scoped[1])
    record_property('withdrawal_ablated_compute', without_scope[1])


def test_json_queued_offer_is_withdrawn_before_any_new_observation_purchase():
    task = dict(apparatus_seed=9914301, seed=10314304, response='square',
                inner_weight=.35, window_width=3, noise_sigma=.01)
    config = ResearchConfig(observation_budget=240, compute_budget=240000,
                            minimum_improvement=1e-7, seed=task['seed'])
    lab = laboratory(task)
    previous = UnifiedResearchAgent(replace(config, compute_budget=512),
        library=KnowledgeLibrary(), measurement_budget=0).run(lab)
    library, source_id, program = learned_source()
    original = previous['batches'][0]
    first = [Observation(item['features'], item['target'], tuple(item['history']))
             for item in original['observations']]
    probe = Probe('queued_offer_repeat', tuple(original['probe']['sequence']))
    batch = lab.acquire(probe)
    outcome = counterexample(program, first, batch.observations)
    assert outcome['noise_replication']['repeatable_error']
    previous['batches'].append({'probe': asdict(probe), 'channels': [],
        'observations': [asdict(row) for row in batch.observations], 'transitions': []})
    previous['actions'].append({'action': 'acquire', 'purpose': 'test_queued_transfer',
        'probe_id': probe.probe_id, 'channels': [], 'observation_cost': batch.observation_cost,
        'measurement_cost': batch.measurement_cost})
    previous['costs']['observation_cost'] += batch.observation_cost
    previous['costs']['queries'] += 1
    previous['costs']['compute_cost'] += outcome['noise_replication']['compute_cost']
    opened = []

    def factory(item):
        acquired = laboratory(item)
        opened.append(acquired)
        return acquired

    session = UnifiedResearchSession(config, tasks=[task], library=library,
                                     measurement_budget=0, lab_factory=factory)
    session.state['results'].append({'task_position': 0, 'available_concepts': [], 'research': previous})
    session.state['pending_revisits'].append({'task_position': 0, 'source_task_position': 0,
        'candidate': {'program': program.to_dict(), 'channels': [], 'source_concept_ids': [source_id]},
        'relative_gain': 1., 'minimum_resources': {'observation_cost': 1, 'measurement_cost': 0, 'compute_cost': 1}})
    session.library.record_transfer_counterexample(lab.spec, program, [source_id],
                                                   evidence=outcome, probe_id=probe.probe_id)
    state = json.loads(json.dumps(session.to_dict()))
    resumed = UnifiedResearchSession(state=state, lab_factory=factory)
    offer = resumed.state['pending_revisits'].pop()
    before = resumed._costs()
    assert not resumed._perform_revisit(offer)
    assert len(opened) == 1 and not opened[0].acquisitions
    assert resumed._costs() == before
    assert resumed.state['results'][0]['research'] == state['results'][0]['research']
    assert resumed.state['scheduler_events'][-1]['action'] == 'withdraw_refuted_revisit'

"""Observed residuals revise coefficients inside shared nonlinear consumers."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.mechanism_revision import propose_mechanism_revisions
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch


def composed(coefficient, consumer):
    x, z = Program('feature', name='x'), Program('feature', name='z')
    context = Program('add', children=(
        Program('mul', children=(Program('const', value=.8), Program('mul', children=(x, x)))),
        Program('mul', children=(Program('const', value=coefficient), z))))
    if consumer == 'square':
        return Program('mul', children=(context, context))
    if consumer == 'abs':
        return Program('abs', children=(context,))
    return Program('mul', children=(context, Program('add', children=(x, Program('const', value=2.)))))


def observations(program, seed, count):
    random = Random(seed)
    rows = [Observation({'x': random.uniform(-1.3, 1.3), 'z': random.uniform(-1.3, 1.3)}, 0.)
            for _ in range(count)]
    return [replace(row, target=.2 + 1.1 * program.predict(row)) for row in rows]


@pytest.mark.parametrize('consumer', ['square', 'abs', 'product'])
@pytest.mark.parametrize('coefficient', [.35, -.25, -1.15])
def test_inner_coefficients_follow_observations_across_signs_and_consumers(consumer, coefficient):
    source, target = composed(-.65, consumer), composed(coefficient, consumer)
    parent = {'candidate': Candidate('old_response', source, 'learned_response', parents=('source',))}
    original = deepcopy(parent['candidate'].program.to_dict())
    spec = TaskSpec('unlabeled_episode', 'predict measurements', ('x', 'z'), (), ())
    paid = observations(target, 812, 48)
    budget = SearchBudget(16000)
    entries = propose_mechanism_revisions(spec, paid, budget, parent=parent)
    revised = next(entry for entry in entries if entry['kind'] == 'mechanism_parameter_refit')
    fresh = observations(target, 993, 120)
    assert loss(source, fresh, 0., .5) > 1e-3
    assert loss(revised['candidate'].program, fresh, 0., .5) < 1e-7
    assert entries[0]['kind'] == 'mechanism_full_refit' and len(entries) <= 8
    assert revised['candidate'].parents == ('source',)
    assert revised['mechanism_revision']['status'] == 'proposed'
    assert 'validation_loss' not in revised
    assert revised['mechanism_revision']['parameter_search_compute'] > 0
    changes = revised['mechanism_revision']['parameter_revisions']
    assert len(changes) == 1 and changes[0]['coefficients_before'] == [.8, -.65]
    assert changes[0]['shared_occurrences'] == (2 if consumer == 'square' else 1)
    assert parent['candidate'].program.to_dict() == original
    assert 0 < budget.spent < budget.limit


def test_parameter_candidate_keeps_a_slot_when_only_two_candidates_are_affordable():
    source, target = composed(-.65, 'square'), composed(.41, 'square')
    parent = {'candidate': Candidate('old_response', source, 'learned_response')}
    spec = TaskSpec('changed_response', 'predict measurements', ('x', 'z'), (), ())
    rows = observations(target, 891, 48)
    entries = propose_mechanism_revisions(spec, rows, SearchBudget(16000),
                                          parent=parent, max_candidates=2)
    assert [entry['kind'] for entry in entries] == ['mechanism_full_refit', 'mechanism_parameter_refit']
    assert loss(entries[1]['candidate'].program, observations(target, 192, 80), 0., .5) < 1e-7


def test_small_budget_preserves_original_control_without_unpaid_parameter_search():
    source, target = composed(-.65, 'square'), composed(.41, 'square')
    parent = {'candidate': Candidate('old_response', source, 'learned_response')}
    spec = TaskSpec('changed_response', 'predict measurements', ('x', 'z'), (), ())
    budget = SearchBudget(400)
    entries = propose_mechanism_revisions(spec, observations(target, 891, 48), budget, parent=parent)
    assert entries and entries[0]['kind'] == 'mechanism_full_refit'
    assert not any(entry['kind'] == 'mechanism_parameter_refit' for entry in entries)
    assert 0 < budget.spent <= budget.limit


def test_inner_parameter_revision_preserves_historical_relation_evidence_and_replay():
    from test_cognition_nonlinear_mechanism_research import NonlinearMechanismLab, nonlinear_source

    library, source_id = nonlinear_source()
    original = deepcopy(library.to_dict())
    source_lab = NonlinearMechanismLab(-.7, 'old_response')
    call = library.callable_atoms(source_lab.spec, limit=1)[0]
    parent = {'candidate': Candidate('acquired_source', call, 'callable_transfer'),
              'kind': 'callable_transfer'}
    old_rows = source_lab.sample(31, 64).observations
    upper = max(abs(row.features['x0'] - row.features['x1']) for row in old_rows) + 1e-6
    prior = {'source_concept_id': source_id,
             'structure': {'fits': [{'radius': .6}], 'radius_interval': [0., upper]}}
    budget = SearchBudget(40000)
    inherited = RelationIntervalResearch(call, [prior], budget)
    conditioned = inherited.alternatives(old_rows, budget, experiments=[old_rows])
    assert conditioned and inherited.evidence_contexts
    prior['research_program'] = conditioned[0]['research_program']
    prior_before = deepcopy(prior)

    target_lab = NonlinearMechanismLab(.27, 'changed_internal_response')
    rows = target_lab.sample(71, 48).observations
    fitting = SearchBudget(16000)
    proposals = propose_mechanism_revisions(target_lab.spec, rows, fitting, parent=parent, priors=[prior])
    revised = next(entry for entry in proposals if entry['kind'] == 'mechanism_parameter_refit')
    assert loss(revised['candidate'].program, target_lab.sample(913, 61).observations, 0., .5) < 1e-7
    plan = json.loads(json.dumps(revised['research_program']))
    replay = RelationIntervalResearch(revised['candidate'].program, [prior],
                                       SearchBudget(4000), research_program=plan)
    assert len(replay.parameters) == 1 and replay.status == 'unresolved'
    assert replay.parameters[0]['radius_intervals'] == inherited.parameters[0]['radius_intervals']
    assert replay.evidence_contexts == json.loads(json.dumps(inherited.evidence_contexts))
    assert replay.alternatives(rows, SearchBudget(16000), experiments=[rows])
    assert source_id in revised['candidate'].parents
    assert revised['mechanism_revision']['original_program'] == call.to_dict()
    assert library.to_dict() == original and prior == prior_before
    assert 0 < fitting.spent < fitting.limit

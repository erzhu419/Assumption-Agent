"""Object-count proposals preserve operators and require genuine full-source coverage."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.object_scope_revision import (
    object_scope_revision_terms, propose_object_scope_revisions,
)
from hegel_machine.cognition.research_programs import _plan, replay_research_program
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def schema(count, prefix='old'):
    objects = tuple(f'{prefix}_{i}' for i in range(count))
    return ObjectSchema(objects,
        {item: {role: f'{role}:{item}' for role in ('q', 'u', 'x', 'f')} for item in objects},
        {item: (item,) for item in objects},
        {item: {role: f'control:{role}:{item}' for role in ('q', 'u', 'x')} for item in objects})


def feature(view, item, role):
    return Program('feature', name=view.fields[item][role])


def total(view, role):
    return Program('object_sum', children=tuple(feature(view, item, role) for item in view.objects))


def saved(terms, view):
    terms = tuple(terms)
    weights = [1. + i / 10 for i in range(len(terms))]
    plan = _plan({'fit_terms': terms, 'coefficients': [.1, *weights],
        'program': _sum_terms(terms, weights, .1), 'loss': 0.}, [], 20, 100)
    plan['source_fit_ref'] = {'file': 'paid-source.json'}
    return {'status': 'supported', 'source_concept_id': 'learned_source',
        'object_schema': view.to_dict(), 'research_program': plan}


def walk(node):
    return [node, *(part for child in node.children for part in walk(child))]


def test_full_coordinates_expand_with_temporal_nesting_and_fixed_radius(monkeypatch):
    source, target = schema(3), schema(4, 'new')
    neighbors = Program('neighbor_sum', name='right', children=(Program('const', value=.813),
        *(feature(source, item, role) for item in reversed(source.objects) for role in ('f', 'x', 'q'))))
    terms = [Program('window_sum', lag=3, children=(total(source, role),)) for role in ('q', 'u')]
    terms.append(Program('window_sum', lag=3, children=(Program('event_delta', lag=1,
        children=(neighbors,)),)))
    prior = saved(terms, source)
    unchanged = deepcopy(prior)
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('Construction must not predict'))
    proposal = object_scope_revision_terms(prior, target)
    assert prior == unchanged
    assert proposal['fit_terms'][:2] == tuple(Program('window_sum', lag=3,
        children=(total(target, role),)) for role in ('q', 'u'))
    nodes = walk(proposal['fit_terms'][2])
    neighborhood = next(item for item in nodes if item.op == 'neighbor_sum')
    assert len(neighborhood.children) == 13 and neighborhood.name == 'right'
    assert neighborhood.children[0] == Program('const', value=.813)
    assert [node.lag for node in nodes if node.op == 'window_sum'] == [3]
    assert [node.lag for node in nodes if node.op == 'event_delta'] == [1]
    assert set.union(*(term.required_features() for term in proposal['fit_terms'])) == set(target.feature_names)
    assert proposal['proposal_scope']['status'] == 'untested_object_scope_revision'


@pytest.mark.parametrize('kind', ['partial_sum', 'bare_individual', 'object_specific_weight', 'partial_neighbor'])
def test_local_or_asymmetric_fields_do_not_become_whole_object_laws(kind):
    source, target = schema(3), schema(4, 'new')
    terms = [total(source, 'q')]
    if kind == 'partial_sum':
        terms = [Program('object_sum', children=terms[0].children[:2])]
    elif kind == 'bare_individual':
        terms.append(feature(source, source.objects[0], 'u'))
    elif kind == 'object_specific_weight':
        terms = [Program('object_sum', children=tuple(Program('mul', children=(
            Program('const', value=i + 1.), feature(source, item, 'q')))
            for i, item in enumerate(source.objects)))]
    else:
        terms = [Program('neighbor_sum', children=(Program('const', value=.8),
            *(feature(source, item, role) for item in source.objects[:2] for role in ('f', 'x', 'q'))))]
    assert object_scope_revision_terms(saved(terms, source), target) is None


def test_uniform_compound_summands_bind_every_new_role():
    source, target = schema(3), schema(2, 'new')
    term = Program('object_sum', children=tuple(Program('mul', children=(
        feature(source, item, 'f'), feature(source, item, 'q'))) for item in reversed(source.objects)))
    proposal = object_scope_revision_terms(saved([term], source), target)
    assert proposal is not None
    assert proposal['fit_terms'][0] == Program('object_sum', children=tuple(Program('mul', children=(
        feature(target, item, 'f'), feature(target, item, 'q'))) for item in target.objects))


def test_partial_view_missing_role_and_unchanged_count_do_not_claim_expansion():
    source, target = schema(3), schema(4, 'new')
    prior = saved([total(source, 'q')], source)
    assert object_scope_revision_terms(prior, schema(3, 'new')) is None
    partial = deepcopy(source.to_dict())
    partial['controls']['unobserved'] = {'q': 'unseen'}
    assert object_scope_revision_terms({**prior, 'object_schema': partial}, target) is None
    incomplete = deepcopy(target.to_dict())
    del incomplete['fields'][target.objects[-1]]['q']
    assert object_scope_revision_terms(prior, incomplete) is None


def rows(view, seed, coefficients):
    rng, history, result = Random(seed), [], []
    for _ in range(20):
        fields = {name: rng.uniform(-2., 2.) for name in view.feature_names}
        window = [*history[-2:], fields]
        q = sum(point[view.fields[item]['q']] for point in window for item in view.objects)
        u = sum(point[view.fields[item]['u']] for point in window for item in view.objects)
        result.append(Observation(fields, coefficients[0] + coefficients[1] * q + coefficients[2] * u,
                                  tuple(history)))
        history.append(fields)
    return result


def test_fitting_and_existing_replay_keep_source_and_refit_new_coefficients():
    source, target = schema(3), schema(4, 'new')
    prior = saved([Program('window_sum', lag=3, children=(total(source, role),)) for role in ('q', 'u')], source)
    unchanged = deepcopy(prior)
    spec = TaskSpec('independent', 'Measured roles', target.feature_names, (), (),
        context={'object_schema': json.dumps(target.to_dict())})
    training = rows(target, 91, (.7, 1.8, -.4))
    budget = SearchBudget(10000)
    entries = propose_object_scope_revisions(spec, training, budget, [prior])
    assert len(entries) == 1 and entries[0]['training_loss'] < 1e-12 and budget.spent > 0
    entry = entries[0]
    assert entry['method_source_ids'] == ['learned_source']
    assert entry['candidate'].parents == ('learned_source',)
    assert entry['research_program']['fit_rows'] == len(training)
    assert entry['research_program']['status'] == 'requires_fresh_validation'
    assert entry['research_program']['source_ids'] == ['learned_source']
    assert prior == unchanged
    new_rows = rows(target, 143, (-.2, -.5, 2.3))
    replayed = replay_research_program(json.loads(json.dumps(entry['research_program'])), new_rows,
                                      SearchBudget(10000))
    assert replayed['training_loss'] < 1e-12
    assert replayed['candidate'].program != entry['candidate'].program


def test_unfunded_fit_returns_no_candidate_without_changing_source():
    source, target = schema(3), schema(4, 'new')
    prior = saved([total(source, 'q')], source)
    spec = TaskSpec('bounded', '', target.feature_names, (), (), context={'object_schema': target.to_dict()})
    budget = SearchBudget(1)
    assert propose_object_scope_revisions(spec, rows(target, 12, (1., 2., 3.)), budget, [prior]) == []
    assert budget.spent == 0 and budget.denied_requests > 0


@pytest.mark.parametrize('old_replay_succeeds', [False, True])
def test_ordinary_program_path_only_revises_object_scope_after_failed_replay(monkeypatch, old_replay_succeeds):
    source, target = schema(3), schema(4, 'new')
    prior = saved([Program('window_sum', lag=3, children=(total(source, role),)) for role in ('q', 'u')], source)
    spec = TaskSpec('ordinary', 'Current measured schema', target.feature_names, (), (),
        context={'object_schema': target.to_dict()})
    training, validation = rows(target, 177, (.7, 1.8, -.4)), rows(target, 188, (.7, 1.8, -.4))
    budget = SearchBudget(10000)
    if old_replay_succeeds:
        old = saved([Program('window_sum', lag=3, children=(total(target, role),)) for role in ('q', 'u')], target)
        replayed = replay_research_program(old['research_program'], training, budget)
        assert replayed['training_loss'] < 1e-12
        monkeypatch.setattr('hegel_machine.cognition.object_scope_revision.propose_object_scope_revisions',
            lambda *args, **kwargs: pytest.fail('Successful old replay must not trigger scope research'))
    else:
        mapping = {source.fields[old][role]: target.fields[new][role]
                   for old, new in zip(source.objects, target.objects) for role in ('q', 'u', 'x', 'f')}
        replayed = replay_research_program(prior['research_program'], training, budget,
                                           feature_mapping=mapping)
        assert replayed['training_loss'] > .01
    monkeypatch.setattr('hegel_machine.cognition.relation_revisions.propose_relation_revisions',
        lambda *args, **kwargs: pytest.fail('A successful current proposal needs no unrelated search'))
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    agent.library.meta_priors.append(prior)
    actions, before = [], budget.spent
    claims = agent._program_claims(spec, training, budget, [], actions,
                                   relation_kernels={}, replayed=[replayed])
    assert claims[0] is replayed
    if old_replay_succeeds:
        assert claims == [replayed] and budget.spent == before
        assert not any(item['action'] == 'revise_object_scope' for item in actions)
    else:
        assert len(claims) == 2 and budget.spent > before
        assert any(item['action'] == 'revise_object_scope' for item in actions)
        compared = agent._assess(claims, validation, budget, 0., .5)
        winner = agent._best(compared)
        assert winner['candidate'].candidate_id.startswith('object_scope_revision:')
        assert winner['validation_loss'] < 1e-12
        assert compared[0]['candidate'] == replayed['candidate']

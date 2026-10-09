"""Current evidence selects paid risk knowledge and can reopen a rejected source."""
from copy import deepcopy
from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.knowledge_adoption import screen_risk_knowledge
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _fit, _terms
from hegel_machine.cognition.research_graph import ResearchGraph


SPEC = TaskSpec('observed_response', 'Predict the fixed quantile', ('x', 'y'), (), (), risk_quantile=.9)


def call(source, field='x'):
    value = Program('arg', name='input')
    definition = Definition(source + ':definition', ('input',), Program('mul', children=(value, value)), source)
    return Program('call', name=definition.definition_id, definition=definition,
                   children=(Program('feature', name=field),))


def rows(seed, nonlinear=True):
    rng = Random(seed)
    result = []
    for _ in range(3):
        history = []
        for _ in range(24):
            features = {'x': rng.uniform(-2, 2), 'y': rng.uniform(-2, 2)}
            target = (features['x'] ** 2 if nonlinear else 2 * features['x']) + rng.uniform(-.01, .01)
            result.append(Observation(features, target, tuple(history)))
            history.append({**features, '__target__': target})
    return result


def screen(training, validation, calls, priors=(), limit=20000):
    budget = SearchBudget(limit)
    baseline = _fit(_terms(SPEC.feature_names), training, budget, 'current_affine', q=.9)
    result = screen_risk_knowledge(SPEC, training, validation, baseline, calls, budget,
                                  tolerance=1e-7, priors=priors)
    return result, budget


def test_source_gain_is_marginal_and_redundant_sources_do_not_each_get_credit():
    primary, duplicate, wrong = call('primary'), call('duplicate'), call('wrong', 'y')
    priors = [{'source_concept_id': name, 'status': 'supported'} for name in ('primary', 'duplicate', 'wrong')]
    (selected, claims, events), budget = screen(rows(11), rows(22), [primary, duplicate, wrong], priors)
    assert selected == [primary]
    assert events[0]['status'] == 'supported_for_research'
    assert events[1]['status'] == 'no_marginal_gain'
    assert events[1]['against_source_ids'] == ['primary']
    assert events[2]['status'] != 'supported_for_research'
    assert claims and all(not event['confirmed'] for event in events)
    assert all(event['fixed_quantile'] == .9 and event['validation_trajectories'] == 3 for event in events)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(KnowledgeLibrary(meta_priors=priors).to_dict())))
    assert restored.meta_priors == priors
    assert all(prior['adoption_evidence'] and prior['status'] == 'supported' for prior in priors)
    assert 0 < budget.spent <= budget.limit


def test_current_counterexample_rejects_replay_and_new_conditions_reopen_it():
    source = call('old_quadratic')
    priors = [{'source_concept_id': 'old_quadratic', 'status': 'supported'}]
    (selected, _, events), _ = screen(rows(31, False), rows(42, False), [source], priors)
    assert selected == []
    assert events[0]['status'] != 'supported_for_research'
    preserved = deepcopy(priors[0]['adoption_evidence'])
    (reopened, _, events), _ = screen(rows(53), rows(64), [source], priors)
    assert reopened == [source]
    assert priors[0]['adoption_evidence'][:len(preserved)] == preserved
    assert priors[0]['status'] == 'supported'


def test_replacing_an_incumbent_removes_its_source_from_later_comparisons():
    x, y = Program('arg', name='x'), Program('arg', name='y')
    body = Program('add', children=(Program('mul', children=(Program('const', value=10),
        Program('mul', children=(x, x)))), Program('mul', children=(y, y))))
    definition = Definition('joint:def', ('x', 'y'), body, 'joint')
    joint = Program('call', name=definition.definition_id, definition=definition,
                    children=(Program('feature', name='x'), Program('feature', name='y')))
    training, validation = ([replace(row, target=10 * row.features['x'] ** 2 + row.features['y'] ** 2)
                             for row in rows(seed)] for seed in (123, 456))
    (_, claims, events), _ = screen(training, validation, [call('first'), joint, call('next')])
    assert events[0]['status'] == events[1]['status'] == 'supported_for_research'
    assert events[1]['against_source_ids'] == ['first']
    assert events[2]['against_source_ids'] == ['joint']
    assert claims[1]['method_source_ids'] == ['joint']


def test_research_view_does_not_replay_an_unhelpful_procedure(monkeypatch):
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, compute_budget=50000))
    source = call('irrelevant', 'y')
    monkeypatch.setattr(agent, '_recall', lambda spec: [source])
    def unexpected(*args, **kwargs):
        raise AssertionError('an unsupported source reached expensive procedure replay')
    monkeypatch.setattr(agent, '_program_claims', unexpected)
    actions = []
    claims = agent._research_view(SPEC, rows(75, False), rows(86, False), SearchBudget(30000),
                                  actions, [], ResearchGraph())
    assert claims
    event = next(item for item in actions if item['action'] == 'screen_risk_knowledge')
    assert event['research_calls'] == 0 and event['compute_cost'] > 0
    assert not any(item.get('selected_action') == 'program_revision' for item in actions)


def test_overlapping_evidence_is_not_reported_as_an_independent_gain():
    training = rows(97)
    (selected, claims, events), _ = screen(training, training, [call('first'), call('second')])
    assert len(selected) == 1 and not claims
    assert events[0]['status'] == 'comparison_unavailable'
    assert 'gain' not in events[0]

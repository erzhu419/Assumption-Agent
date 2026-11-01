"""Local direction research refits the full saved basis, not one weighted sum."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_revisions import (
    propose_parent_relation_revisions, quote_parent_relation_revisions,
)
from hegel_machine.cognition.relation_uncertainty import _bind
from hegel_machine.cognition.research_programs import _entry
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import transfer_partition_view
from test_transfer_partitions import plan


def scene(direction='left', count=83, source_direction=None):
    payload = plan(dormant=True)
    old = payload['selection_steps'][0]
    relation = replace(Program.from_dict(old['parameter_terms'][1]), name=source_direction)
    old['parameters'][0]['relation'] = source_direction or 'symmetric'
    symbolic = [Program('feature', name=f'z{i}') for i in range(6)] + [
        relation, Program('window_sum', lag=2, children=(relation,))]
    terms = [_bind(term, {'r': Program('const', value=.4)}) for term in symbolic]
    coefficients = [3., 1., 2., 0., 0., 0., 0., .7, .2]
    program = _sum_terms(terms, coefficients[1:], coefficients[0])
    fit = {'program': program, 'fit_terms': terms, 'terms': terms,
           'coefficients': coefficients, 'loss': 1.}
    binding = {**old, 'program': program,
        'parameter_template': _sum_terms(symbolic, coefficients[1:], coefficients[0]).to_dict(),
        'parameter_terms': [term.to_dict() for term in symbolic],
        'evidence_contexts': [{'source_ref': 'paid-old-response'}]}
    parent = _entry(fit, [binding], 20, 100)

    def redirect(node):
        return replace(node, name=direction) if node.op == 'neighbor_sum' else replace(
            node, children=tuple(redirect(child) for child in node.children))

    target = _sum_terms([redirect(term) for term in terms],
        [2., -1., .5, 0., 0., 0., 3., -2.], .5)
    history, rows = [], []
    for i in range(count):
        fields = {f'z{j}': (((i + 1) * (j + 3)) % (11 + j)) / 7. for j in range(5)}
        fields.update(z5=0., f0=1., f1=0., x0=.1,
            x1=.1 + (-1 if i % 2 else 1) * (.2 if i % 5 else .6),
            q0=.13 * (i % 7), q1=1. + .17 * (i % 13))
        row = Observation(fields, 0., tuple(history[-2:]))
        rows.append(replace(row, target=target.predict(row)))
        history.append(fields)
    return parent, rows


@pytest.mark.parametrize('direction', ['left', 'right'])
def test_local_directions_jointly_refit_all_columns_and_preserve_bindings(direction, monkeypatch):
    import hegel_machine.cognition.research_programs as module
    parent, rows = scene(direction)
    original = deepcopy(parent)
    calls, native = [], module._Fitter.fit

    def record(self, terms):
        calls.append(list(terms))
        return native(self, terms)

    monkeypatch.setattr(module._Fitter, 'fit', record)
    quote = quote_parent_relation_revisions(parent, len(rows), comparison_rows=103)
    assert quote['fit_compute_upper'] == 1660 and quote['comparison_compute'] == 206
    budget = SearchBudget(quote['required_compute'])
    result = propose_parent_relation_revisions(SimpleNamespace(risk_quantile=.5), rows, budget, parent)
    report = result['research']
    assert report['execution_complete'] and report['finite_domain_exhausted']
    assert len(calls) == len(result['entries']) == 2
    assert all(len(terms) == 8 for terms in calls)
    assert {entry['parent_relation_revision']['direction'] for entry in result['entries']} == {'right', 'left'}
    assert parent == original and report['parent_refitted'] is False
    assert report['compute_cost'] == budget.spent <= quote['required_compute']
    chosen = next(entry for entry in result['entries'] if entry['parent_relation_revision']['direction'] == direction)
    assert chosen['training_loss'] < 1e-12
    assert min(entry['training_loss'] for entry in result['entries']) < max(
        entry['training_loss'] for entry in result['entries'])
    for entry in result['entries']:
        body = entry['research_program']
        fit = research_program_fit_contract(body)
        binding = body['selection_steps'][0]
        assert len(fit['fit_terms']) == len(binding['parameter_terms']) == 8
        assert fit['coefficients'][6] == 0.  # explicitly saved dormant z5
        assert fit['program'] == entry['candidate'].program
        assert transfer_partition_view(body)['status'] == 'ready'
        assert binding['parameters'][0]['symbol'] == 'r'
        assert binding['parameters'][0]['radius'] == .4
        assert binding['parameters'][0]['radius_intervals'] == []
        assert binding['parameters'][0]['prior_radius_intervals'] == [[.39, .41000000000000003]]
        assert binding['evidence_contexts'] == []
        assert 'parameter_terms' not in binding['prior_binding']


def test_budget_partial_preserves_fitted_candidate_and_unresolved_attempt():
    parent, rows = scene()
    quote = quote_parent_relation_revisions(parent, len(rows))
    budget = SearchBudget(quote['preparation_compute'] + quote['fit_compute_upper'] // 2)
    result = propose_parent_relation_revisions(SimpleNamespace(risk_quantile=.5), rows, budget, parent)
    assert len(result['entries']) == 1
    report = result['research']
    assert report['budget_incomplete'] and not report['execution_complete']
    assert report['attempts'][-1]['status'] == 'compute_unresolved'
    assert report['attempts'][0]['fit_complete']
    assert report['compute_cost'] == budget.spent <= budget.limit


def test_equal_radius_values_do_not_merge_saved_symbols():
    body = plan(other=True, dormant=True)
    fit = research_program_fit_contract(body)
    parent = _entry(fit, [{**body['selection_steps'][0], 'program': fit['program']}], 20, 100)
    from test_transfer_partitions import observations
    rows = observations(other=True)
    result = propose_parent_relation_revisions(SimpleNamespace(risk_quantile=.5), rows,
        SearchBudget(100000), parent, max_candidates=2)
    assert result['research']['execution_complete']
    assert not result['research']['finite_domain_exhausted']
    assert len(result['entries']) == 2
    for entry in result['entries']:
        binding = entry['research_program']['selection_steps'][0]
        assert binding['parameter_terms'][2] == body['selection_steps'][0]['parameter_terms'][2]
        assert binding['parameters'][1] == body['selection_steps'][0]['parameters'][1]


@pytest.mark.parametrize('numerical_error', [False, True])
def test_fit_failures_remain_unresolved_even_if_every_form_was_attempted(monkeypatch, numerical_error):
    import hegel_machine.cognition.research_programs as module
    parent, rows = scene()
    def failed(*args):
        if numerical_error:
            raise ValueError('candidate produced a non-finite prediction')
        return None

    monkeypatch.setattr(module._Fitter, 'fit', failed)
    result = propose_parent_relation_revisions(SimpleNamespace(risk_quantile=.5), rows,
        SearchBudget(100000), parent)
    assert not result['entries']
    assert result['research']['attempts_complete']
    assert not result['research']['execution_complete']
    assert result['research']['status'] == 'fit_unresolved'
    assert [item['status'] for item in result['research']['attempts']] == ['fit_unavailable'] * 2
    if numerical_error:
        assert all('non-finite' in item['failure'] for item in result['research']['attempts'])


def test_directional_parent_also_proposes_symmetric_competitor():
    parent, rows = scene(source_direction='left')
    result = propose_parent_relation_revisions(SimpleNamespace(risk_quantile=.5), rows,
        SearchBudget(100000), parent)
    assert {item['direction'] for item in result['research']['attempts']} == {'symmetric', 'right'}
    assert result['research']['execution_complete']


def test_no_binding_is_not_silently_replaced_by_one_output_column():
    parent, rows = scene()
    del parent['research_program']['selection_steps'][0]['parameter_terms']
    budget = SearchBudget(100000)
    result = propose_parent_relation_revisions(SimpleNamespace(risk_quantile=.5), rows, budget, parent)
    assert result['research']['status'] == 'binding_unavailable'
    assert not result['entries'] and budget.spent == 0

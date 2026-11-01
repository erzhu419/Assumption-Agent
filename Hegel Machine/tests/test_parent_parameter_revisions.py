"""A paid parent can reuse a Pi1 operation without restoring its old world graph."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.research_programs import _entry, _Fitter
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import (
    _complete_fit, prepare_parent_parameter_revision, propose_parent_parameter_revisions,
    quote_transfer_partition, replay_transfer_partition, transfer_partition_view,
)
from test_parent_relation_revisions import scene
from test_transfer_partitions import plan


def paid_scene(*, unexplained=False):
    parent, rows = scene(source_direction='right', direction='right')
    original = research_program_fit_contract(parent['research_program'])

    def radius(node):
        if node.op == 'neighbor_sum':
            return replace(node, children=(Program('const', value=.85), *node.children[1:]))
        return replace(node, children=tuple(radius(child) for child in node.children))

    target = _sum_terms([radius(term) for term in original['fit_terms']],
        [2., -1., .5, 0., 0., 0., 3., -2.], .5)
    rows = [replace(row, target=target.predict(row) + (i ** 2 * .01 if unexplained else 0.))
        for i, row in enumerate(rows)]
    budget = SearchBudget(10000)
    fit = _complete_fit(_Fitter(rows, budget, .5).fit(original['fit_terms']), original['fit_terms'])
    saved = deepcopy(parent['research_program']['selection_steps'][0])
    symbolic = [Program.from_dict(term) for term in saved['parameter_terms']]
    saved.update(program=fit['program'], loss=fit['loss'], evidence_contexts=[],
        parameter_template=_sum_terms(symbolic, fit['coefficients'][1:], fit['coefficients'][0]).to_dict())
    saved['parameters'][0].update(radius_intervals=[], prior_radius_intervals=[[.39, .41]])
    parent = _entry(fit, [saved], len(rows), budget.spent)
    paid = {'entry': parent, 'training': rows, 'risk_quantile': .5,
            'cost_ref': {'file': 'paid-current-fit.json'}}
    source = plan()
    source['transfer_partition'] = {'kind': 'Pi1', 'released_groups': ['r'],
        'max_attempts': 48, 'max_candidates': 4, 'source_revision': {'old_evidence': True}}
    return parent, rows, paid, source


def prepared(parent, source):
    result = prepare_parent_parameter_revision(parent, source,
        source_method_ref={'file': 'old-frozen-Pi1.json'}, parent_ref={'file': 'current-parent.json'})
    assert result['status'] == 'ready'
    return result['research_program']


def test_operator_rebinding_keeps_current_full_parent_and_immutable_source():
    parent, _, _, source = paid_scene()
    old_parent, old_source = deepcopy(parent), deepcopy(source)
    source['transfer_partition']['second_stage'] = {
        'operation': 'resolve_relation_intervals', 'compute_limit': 15000,
        'trigger': 'fitted_training_with_unexplained_return'}
    old_source = deepcopy(source)
    body = prepared(parent, source)
    assert research_program_fit_contract(body) == research_program_fit_contract(parent['research_program'])
    assert body['selection_steps'][0]['parameters'][0]['relation'] == 'right'
    assert source['selection_steps'][0]['parameters'][0]['relation'] == 'symmetric'
    assert body['transfer_partition']['max_attempts'] == 48
    assert body['transfer_partition']['second_stage'] == source['transfer_partition']['second_stage']
    assert 'source_revision' not in body['transfer_partition']
    assert body['parent_parameter_revision']['new_parent_binding']
    assert not body['parent_parameter_revision']['source_support_inherited']
    assert parent == old_parent and source == old_source


def test_paid_initializer_is_reused_but_remaining_parameter_search_really_runs(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    parent, rows, paid, source = paid_scene()
    body = prepared(parent, source)
    original_terms = research_program_fit_contract(body)['fit_terms']
    calls, native = [], module._Fitter.fit

    def record(self, terms):
        assert tuple(terms) != original_terms, 'already paid initializer was fitted again'
        calls.append(tuple(terms))
        return native(self, terms)

    monkeypatch.setattr(module._Fitter, 'fit', record)
    quote = quote_transfer_partition(body, rows)
    budget = SearchBudget(quote['required_compute'])
    result = propose_parent_parameter_revisions(SimpleNamespace(risk_quantile=.5), rows,
        budget, parent=parent, method=body, paid_initial=paid)
    entry, = result['entries']
    report = result['research']
    assert report['execution_complete'] and not report['budget_incomplete']
    assert report['new_fits'] == len(calls) > 0
    assert report['search']['attempts'] == len(calls) + 1 <= 48
    assert report['search']['reused_initial_fits'] == 1
    assert report['search']['paid_initial']['cost_ref'] == paid['cost_ref']
    assert entry['training_loss'] < 1e-12 < parent['training_loss']
    fit = research_program_fit_contract(entry['research_program'])
    assert len(fit['fit_terms']) == 8 and fit['coefficients'][6] == 0.
    assert transfer_partition_view(entry['research_program'])['status'] == 'ready'
    assert all(len(terms) == 8 for terms in calls)
    assert budget.spent == report['compute_cost'] <= quote['required_compute']


@pytest.mark.parametrize('change', ['rows', 'risk', 'columns', 'loss', 'cost_ref'])
def test_initializer_reuse_requires_same_paid_inputs(change, monkeypatch):
    import hegel_machine.cognition.research_programs as module
    parent, rows, paid, source = paid_scene()
    body = prepared(parent, source)
    record = deepcopy(paid)
    if change == 'rows':
        record['training'][0] = replace(record['training'][0], target=999.)
    elif change == 'risk':
        record['risk_quantile'] = .9
    elif change == 'columns':
        record['entry'] = _entry(research_program_fit_contract(plan()), [], len(rows), 100)
    elif change == 'loss':
        record['entry']['training_loss'] += .1
    else:
        record.pop('cost_ref')
    calls, native = [], module._Fitter.fit

    def observe(self, terms):
        calls.append(tuple(terms))
        return native(self, terms)

    monkeypatch.setattr(module._Fitter, 'fit', observe)
    result = replay_transfer_partition(body, rows, SearchBudget(100000), paid_initial=record)
    assert calls[0] == research_program_fit_contract(body)['fit_terms']
    assert result['transfer_partition_search']['reused_initial_fits'] == 0


def test_partial_budget_keeps_paid_parent_without_calling_it_completed():
    parent, rows, paid, source = paid_scene()
    body = prepared(parent, source)
    quote = quote_transfer_partition(body, rows)
    budget = SearchBudget(quote['initialization_compute'] + quote['geometry_compute'])
    result = propose_parent_parameter_revisions(SimpleNamespace(risk_quantile=.5), rows,
        budget, parent=parent, method=body, paid_initial=paid)
    assert result['entries'][0]['candidate'].program == parent['candidate'].program
    report = result['research']
    assert report['status'] == 'compute_unresolved' and not report['execution_complete']
    assert report['new_fits'] == 0 and report['search']['reused_initial_fits'] == 1


def test_unexplained_joint_fit_does_not_force_fixed_readout_followup(monkeypatch):
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    parent, rows, paid, source = paid_scene(unexplained=True)
    source['transfer_partition']['second_stage'] = {
        'operation': 'resolve_relation_intervals', 'compute_limit': 15000}
    body = prepared(parent, source)
    monkeypatch.setattr(RelationIntervalResearch, 'alternatives',
        lambda *args, **kwargs: pytest.fail('unsupported fixed-readout continuation'))
    quote = quote_transfer_partition(body, rows, feedback_rows=rows[-6:])
    result = propose_parent_parameter_revisions(SimpleNamespace(risk_quantile=.5), rows,
        SearchBudget(quote['required_compute']), parent=parent, method=body,
        feedback_rows=rows[-6:], paid_initial=paid)
    report = result['research']
    assert report['bounded_first_search_complete'] and not report['execution_complete']
    assert report['status'] == 'parameter_followup_not_applicable'
    assert report['followup']['status'] == 'first_stage_training_unexplained'
    assert report['followup']['compute_cost'] == 0


def test_multiple_groups_do_not_get_an_unjustified_developer_choice():
    body = plan(other=True)
    fit = research_program_fit_contract(body)
    parent = _entry(fit, [{**body['selection_steps'][0], 'program': fit['program']}], 20, 100)
    source = deepcopy(body)
    source['transfer_partition'] = {'kind': 'Pi1', 'released_groups': ['r'],
        'max_attempts': 48, 'max_candidates': 4}
    assert prepare_parent_parameter_revision(parent, source)['status'] == 'one_explicit_parent_group_required'

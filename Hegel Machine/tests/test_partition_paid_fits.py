"""Exact paid solves feed the real method selector without repeating fitting."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.relation_revisions import propose_parent_relation_revisions
from hegel_machine.cognition.research_programs import _entry, replay_research_program
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import propose_transfer_partitions
from test_parent_parameter_revisions import paid_scene, prepared
from test_parent_relation_revisions import scene


def paid_directions():
    parent, rows = scene(direction='left')
    entries = propose_parent_relation_revisions(SimpleNamespace(risk_quantile=.5), rows,
        SearchBudget(10000), parent)['entries']
    source_right = next(entry for entry in entries if
        entry['parent_relation_revision']['direction'] == 'right')
    method = propose_transfer_partitions(parent['research_program'],
        source_right['research_program'])['candidates'][0]['research_program']
    records = [{'entry': entry, 'training': rows, 'risk_quantile': .5,
                'cost_ref': {'file': 'paid-directions.json', 'json_path': [index]}}
               for index, entry in enumerate(entries)]
    return method, rows, records


def test_paid_direction_fits_are_reselected_by_training_and_keep_all_columns(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    method, rows, records = paid_directions()
    frozen = deepcopy(records)
    monkeypatch.setattr(module._Fitter, 'fit',
        lambda *args: pytest.fail('the exact two direction fits were already paid'))
    budget = SearchBudget(0)
    result = replay_research_program(method, rows, budget, paid_fits=list(reversed(records)))
    search = result['transfer_partition_search']
    assert search['completed'] and search['attempts'] == search['reused_fits'] == 2
    assert search['new_fits'] == budget.spent == 0
    assert result['training_loss'] == min(item['entry']['training_loss'] for item in records)
    fit = research_program_fit_contract(result['research_program'])
    assert len(fit['fit_terms']) == 8 and fit['coefficients'][6] == 0.
    parameter, = result['research_program']['selection_steps'][0]['parameters']
    assert parameter['relation'] == 'left' and parameter['radius_intervals'] == []
    assert {item['paid_fit']['cost_ref']['json_path'][0] for item in
        search['alternative_outcomes']} == {0, 1}
    assert records == frozen


@pytest.mark.parametrize('change', ['rows', 'history', 'risk', 'loss', 'columns', 'cost_ref'])
def test_incompatible_paid_alternative_falls_back_to_exactly_one_new_fit(change, monkeypatch):
    import hegel_machine.cognition.research_programs as module
    method, rows, records = paid_directions()
    records = deepcopy(records)
    record = records[0]
    record['training'] = list(record['training'])
    if change == 'rows':
        record['training'][0] = replace(record['training'][0], target=999.)
    elif change == 'history':
        record['training'][-1] = replace(record['training'][-1], history=())
    elif change == 'risk':
        record['risk_quantile'] = .9
    elif change == 'loss':
        record['entry']['training_loss'] += 1.
    elif change == 'columns':
        record['entry'] = deepcopy(records[1]['entry'])
    else:
        record.pop('cost_ref')
    calls, original = [], module._Fitter.fit
    def fit(self, terms):
        calls.append(tuple(terms))
        return original(self, terms)
    monkeypatch.setattr(module._Fitter, 'fit', fit)
    result = replay_research_program(method, rows, SearchBudget(10000), paid_fits=records)
    search = result['transfer_partition_search']
    assert search['completed'] and search['new_fits'] == search['reused_fits'] == 1
    assert len(calls) == 1 and len(calls[0]) == 8


def test_one_paid_direction_does_not_mark_unfunded_other_direction_complete():
    method, rows, records = paid_directions()
    result = replay_research_program(method, rows, SearchBudget(0), paid_fits=records[:1])
    search = result['transfer_partition_search']
    assert search['reused_fits'] == 1
    assert search['status'] == 'compute_unresolved' and not search['completed']
    assert search['remaining'] == 1


def test_paid_complete_duplicate_columns_keep_the_original_coefficient_binding(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    method, rows, records = paid_directions()
    method = deepcopy(method)
    record = deepcopy(records[0])
    fit = research_program_fit_contract(record['entry']['research_program'])
    terms = (*fit['fit_terms'], fit['fit_terms'][0])
    complete = {**fit, 'fit_terms': terms, 'terms': terms,
                'coefficients': [*fit['coefficients'], 0.]}
    record['entry'] = _entry(complete, [], len(rows), 100)
    alternative = deepcopy(method['transfer_partition']['alternatives'][0])
    alternative['fit_terms'] = [term.to_dict() for term in terms]
    alternative['parameter_terms'].append(deepcopy(alternative['parameter_terms'][0]))
    method['transfer_partition']['alternatives'] = [alternative]
    monkeypatch.setattr(module._Fitter, 'fit', lambda *args: pytest.fail('already fitted'))
    result = replay_research_program(method, rows, SearchBudget(0), paid_fits=[record])
    retained = research_program_fit_contract(result['research_program'])
    assert retained['fit_terms'] == terms
    assert retained['coefficients'] == complete['coefficients']
    assert retained['program'] == fit['program']


def test_public_replay_forwards_paid_initializer_without_skipping_remaining_search(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    parent, rows, paid, source = paid_scene()
    body = prepared(parent, source)
    terms = research_program_fit_contract(body)['fit_terms']
    original, calls = module._Fitter.fit, []
    def fit(self, proposed):
        assert tuple(proposed) != terms
        calls.append(tuple(proposed))
        return original(self, proposed)
    monkeypatch.setattr(module._Fitter, 'fit', fit)
    result = replay_research_program(body, rows, SearchBudget(100000), paid_initial=paid)
    search = result['transfer_partition_search']
    assert search['reused_initial_fits'] == 1
    assert search['attempts'] - search['reused_initial_fits'] == len(calls) > 0

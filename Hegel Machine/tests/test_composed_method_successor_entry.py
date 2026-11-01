"""Freeze the learned chain and its shorter comparator before any new data."""

from copy import deepcopy

import pytest

from scripts import validate_partition_successor as runner
from scripts.run_partial_role_parent import load, save
from test_complete_method_successor_entry import prepared
from test_parent_bound_staged_replay import fixture


def composition(prepared):
    source, output, request, supplied, *_ = prepared
    method, _, _ = fixture()
    baseline = method['staged_research']['baseline']
    save(source / 'methods.json', {'Pi3': baseline})
    practice = source / 'practice'
    method_ref = runner.ref(practice / 'frozen_method.json')
    save(practice / 'frozen_method.json', method)
    save(practice / 'registration.json', {'method_ref': method_ref,
        'proposal_ref': runner.ref(source / 'native_composition.json'),
        'sources': [{'source_task_ref': runner.ref(source / 'task.json')}]})
    save(practice / 'receipt.json', {'status': 'paid_stage_paths_completed',
        'sources': [{'execution_complete': True, 'full_comparison': True}]})
    save(source / 'allocation.json', {**load(source / 'allocation.json'), 'maximum_compute': 300000})
    save(request, {**supplied, 'composition_practice_ref': runner.ref(practice / 'receipt.json')})
    return method, practice


def test_freezes_actual_chain_and_quotes_each_possible_parent(prepared):
    source, output, request, *_ = prepared
    method, practice = composition(prepared)
    original = deepcopy(method)
    cfg = runner.register_complete_method_successor(request, output)
    assert load(output / 'frozen_methods.json') == {
        'P0': original['staged_research']['baseline'], 'Pi1': original}
    assert cfg['source_pool_size'] == 262
    assert cfg['scope'] == 'frozen_generated_composition_vs_its_first_stage'
    assert cfg['composition_practice_ref'] in cfg['source_cost_refs']
    assert cfg['composition_proposal_ref'] in cfg['source_cost_refs']
    assert cfg['practice_scope'] == 'paid_path_verification_not_independent_support'
    quote = cfg['method_quotes']['Pi1']
    alternatives = quote['stages']['continuation']['parent_alternatives']
    assert len(alternatives) == 2 and all(item['fit_columns'] == 8 for item in alternatives)
    assert all(item['geometry_compute'] > 0 for item in alternatives)
    assert cfg['total_new_compute_upper'] <= cfg['total_compute_limit']
    assert load(practice / 'frozen_method.json') == original
    assert not (output / 'observations').exists()


def test_does_not_require_discarding_later_paid_source_rows(prepared, monkeypatch):
    from scripts import review_paid_object_pool
    source, output, request, *_ = prepared
    composition(prepared)
    prior = review_paid_object_pool.parent_inputs(source)
    extended = (*prior[:2], [*prior[2], *prior[2][:60]],
        [*prior[3], {'rows': 48}, {'rows': 12}], *prior[4:])
    monkeypatch.setattr(review_paid_object_pool, 'parent_inputs', lambda _: extended)
    cfg = runner.register_complete_method_successor(request, output)
    assert cfg['training_rows'] == 96 and cfg['working_rows'] == 24
    assert cfg['source_pool_size'] == 262


@pytest.mark.parametrize('change,reason', [
    ('unfinished', 'Complete the generated composition'),
    ('different_baseline', 'own actual first-stage'),
    ('exposed_seed', 'already in the retained source cohort'),
])
def test_missing_path_or_wrong_comparator_does_not_register(prepared, change, reason):
    source, output, request, supplied, *_ = prepared
    method, practice = composition(prepared)
    if change == 'unfinished':
        save(practice / 'receipt.json', {'status': 'paid_stage_path_incomplete_not_negative'})
    elif change == 'different_baseline':
        method['staged_research']['baseline']['transfer_partition']['max_attempts'] = 1
        save(practice / 'frozen_method.json', method)
    else:
        save(practice / 'other_task.json', {'seed': supplied['seed']})
        cfg = load(practice / 'registration.json')
        cfg['sources'].append({'source_task_ref': runner.ref(practice / 'other_task.json')})
        save(practice / 'registration.json', cfg)
    with pytest.raises(ValueError, match=reason):
        runner.register_complete_method_successor(request, output)
    assert not output.exists()

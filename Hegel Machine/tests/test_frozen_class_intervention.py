"""Frozen class removal changes organization, preserving source facts and fees."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.theory_consolidation import _read_index, _read_library
from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study
from test_prepare_frozen_successor import frozen_spec, prohibit_science
from test_prepare_pr3_source_successor import save


def class_spec(tmp_path):
    spec, candidate = frozen_spec(tmp_path)
    old = preparation._value(spec['materials']['initial_library_ref'])['candidates']['old']
    alias = {**deepcopy(candidate), 'id': 'new_alias'}
    alias['definition']['definition_id'] = 'new_alias_definition'
    other = {**deepcopy(candidate), 'id': 'other'}
    other['definition']['body'] = Program('abs', children=(Program('arg', name='x'),)).to_dict()
    library = {'candidates': {'old': old, 'new': candidate, 'new_alias': alias, 'other': other},
        'executions': {
            'old_transfer': {'request_basis': {'candidate_id': 'old'}, 'result': {'status': 'UNKNOWN'}},
            'new_transfer': {'request_basis': {'candidate_id': 'new'}, 'result': {'status': 'UNKNOWN'}},
            'alias_transfer': {'request_basis': {'candidate_id': 'new_alias'}, 'result': {'status': 'UNKNOWN'}},
            'other_transfer': {'request_basis': {'candidate_id': 'other'}, 'result': {'status': 'UNKNOWN'}},
            'mixed_source_construction': {'request_basis': {'common_source_refs': []}, 'result': {
                'frozen_candidates': [{'candidate_id': key} for key in ['new', 'new_alias', 'other']]}}},
        'transfers': [{'theory_id': f'instance_{key}', 'binding': {'candidate_id': key}}
                      for key in ['new', 'new_alias', 'old', 'other']]}
    full = save(tmp_path / 'complete_library.json', library)
    spec['frozen_source']['library_ref'] = save(tmp_path / 'thin_library.json', {'source_ref': full})
    spec['frozen_source']['new_candidate_ids'] = ['new', 'new_alias', 'other']
    spec['frozen_source']['redundancy_classes_ref'] = save(tmp_path / 'saved_classes.json', [
        {'original_member_ids': ['new', 'new_alias'], 'additional_same_contract_member_ids': []},
        {'original_member_ids': ['other'], 'additional_same_contract_member_ids': []}])

    def indexed(identifier, origin, version=1):
        return {'id': identifier, 'program': Program('const', value=float(version)).to_dict(),
            'version': version, 'family_id': origin, 'consolidation_ref': {'candidate_id': origin}}

    mixed = indexed('mixed', 'new', 2)
    mixed['history'] = [indexed('mixed', 'old', 1)]
    index = {'independent': indexed('independent', 'old'), 'derived': indexed('derived', 'new'),
             'mixed': mixed, 'other': indexed('other', 'other')}
    original_index = save(tmp_path / 'complete_index.json', index)
    spec['frozen_source']['index_ref'] = save(tmp_path / 'thin_index.json', {
        'source_ref': original_index, 'entries': {}})
    return spec, library, index


def test_class_removes_actual_aliases_and_only_their_direct_organization_contributions(tmp_path, monkeypatch):
    spec, original, original_index = class_spec(tmp_path)
    prohibit_science(monkeypatch)
    monkeypatch.setattr(study, 'run_target_intervention', lambda *a, **k: pytest.fail('prepare ran removal'))
    saved = deepcopy((original, original_index))
    result = preparation.prepare_frozen_successor(spec, tmp_path / 'prepared')
    selected = result['new_class_intervention']
    manifest = preparation._value(selected['manifest_ref'])
    removed = _read_library(selected['frozen_remove_library_ref'], tmp_path)
    assert manifest['status'] == 'frozen_ready'
    assert manifest['removed_candidate_ids'] == ['new', 'new_alias']
    assert removed['candidates'] == {key: original['candidates'][key] for key in ['old', 'other']}
    assert set(removed['executions']) == {'old_transfer', 'other_transfer', 'mixed_source_construction'}
    assert removed['executions']['mixed_source_construction'] == original['executions']['mixed_source_construction']
    assert [row['binding']['candidate_id'] for row in removed['transfers']] == ['old', 'other']
    plan = preparation._value(selected['plan_ref'])
    index = _read_index(plan['target_intervention']['index_view_ref'], tmp_path)
    assert set(index) == {'independent', 'mixed', 'other'}
    assert index['mixed'] == original_index['mixed']['history'][0]
    assert index['other'] == original_index['other']
    assert manifest['restored_independent_index_ids'] == ['mixed']
    assert len(manifest['retained_historical_construction_refs']) == 2
    assert all(preparation._value(item['source_ref'])['candidate_id'] in {'new', 'new_alias'}
               for item in manifest['retained_historical_construction_refs'])
    assert manifest['concrete_source_programs_parameters_facts_removed'] is False
    assert manifest['old_language_and_independent_old_reconstruction_legal'] is True
    assert result['closed_source_accounting_ref'] == spec['frozen_source']['accounting_ref']
    assert (original, original_index) == saved


def test_thin_source_candidate_references_address_actual_saved_members(tmp_path):
    spec, original, _ = class_spec(tmp_path)
    result = preparation.prepare_frozen_successor(spec, tmp_path / 'prepared')
    for row in result['proposal_qualification']:
        assert preparation._value(row['candidate_ref']) == original['candidates'][row['id']]


def test_removed_organization_survives_checkpoint_save_and_cold_read(tmp_path):
    from hegel_machine.cognition.theory_consolidation import (
        _save, install_theory_index, install_theory_library, save_theory_index,
    )
    from test_theory_consolidation import agent_scene

    spec, _, _ = class_spec(tmp_path)
    result = preparation.prepare_frozen_successor(spec, tmp_path / 'prepared')
    selected = result['new_class_intervention']
    plan = preparation._value(selected['plan_ref'])
    agent, _ = agent_scene(tmp_path / 'local')
    agent.theory_consolidation = install_theory_library(agent, selected['frozen_remove_library_ref'])
    agent.theory_index = install_theory_index(agent, plan['target_intervention']['index_view_ref'])
    _save(agent)
    save_theory_index(agent)
    restored = _read_library({'file': str(agent.workspace / 'shared/theory_library.json')}, tmp_path)
    index = _read_index({'file': str(agent.workspace / 'shared/theory_index.json')}, tmp_path)
    assert set(restored['candidates']) == {'old', 'other'}
    assert [row['binding']['candidate_id'] for row in restored['transfers']] == ['old', 'other']
    assert 'derived' not in index and index['mixed']['family_id'] == 'old'


@pytest.mark.parametrize('workers', [None, 3])
def test_mainline_intervention_wrapper_uses_full_plan_and_filtered_C_index(tmp_path, monkeypatch, workers):
    spec, _, _ = class_spec(tmp_path)
    output = tmp_path / 'prepared'
    result = preparation.prepare_frozen_successor(spec, output)
    result['successor_execution_authorized'] = True  # This stub only.
    preparation._write(output / 'registration.json', result)
    calls = []

    def run(plan, destination, **kwargs):
        calls.append((plan, destination, kwargs))
        assert plan['arms'] == list(study.ORGANIZATION_ARMS)
        assert kwargs['keep_study'] == output / 'target'
        assert kwargs['workers'] == (8 if workers is None else workers)
        assert plan['source_cost_ref'] == spec['frozen_source']['accounting_ref']
        return {'status': 'stub_only'}

    monkeypatch.setattr(study, 'run_target_intervention', run)
    options = {} if workers is None else {'workers': workers}
    assert preparation.run_prepared_intervention(output, 'new_redundancy_class', **options)['status'] == 'stub_only'
    plan, destination, kwargs = calls[0]
    assert destination == output / 'intervention_runs/new_redundancy_class'
    seen = []
    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', lambda *a, **k: seen.append(k))
    knowledge = {'finite_native_author': {}, 'theory_index': {'file': 'original_index.json'}}
    kwargs['agent_factory'](None, tmp_path / 'case_000/pr3', arm='pr3', knowledge=deepcopy(knowledge))
    assert seen[0]['knowledge']['theory_index'] == plan['target_intervention']['index_view_ref']
    assert knowledge['theory_index'] == {'file': 'original_index.json'}


def test_identical_single_member_class_references_one_execution_and_one_cost(tmp_path, monkeypatch):
    spec, _ = frozen_spec(tmp_path)
    spec['frozen_source']['redundancy_classes_ref'] = save(tmp_path / 'classes.json', [
        {'original_member_ids': ['new'], 'additional_same_contract_member_ids': []}])
    output = tmp_path / 'prepared'
    result = preparation.prepare_frozen_successor(spec, output)
    assert result['new_class_intervention']['reuse_execution_of'] == 'individual_object'
    result['successor_execution_authorized'] = True  # This stub only.
    preparation._write(output / 'registration.json', result)
    called = []

    def run(plan, destination, **kwargs):
        called.append(destination)
        preparation._write(destination / 'summary.json', {'costs': {'compute_cost': 17}})
        return {'status': 'stub_only', 'new_compute_cost': 17}

    monkeypatch.setattr(study, 'run_target_intervention', run)
    preparation.run_prepared_intervention(output, 'individual_object')
    reused = preparation.run_prepared_intervention(output, 'new_redundancy_class')
    assert len(called) == 1 and reused['new_compute_cost'] == 0
    assert reused['summary_ref']['file'] == str(called[0] / 'summary.json')


def test_removal_wrapper_keeps_original_execution_authorization_boundary(tmp_path, monkeypatch):
    spec, _, _ = class_spec(tmp_path)
    output = tmp_path / 'prepared'
    preparation.prepare_frozen_successor(spec, output)
    monkeypatch.setattr(study, 'run_target_intervention', lambda *a, **k: pytest.fail('unapproved execution'))
    with pytest.raises(ValueError, match='not authorized'):
        preparation.run_prepared_intervention(output, 'individual_object')

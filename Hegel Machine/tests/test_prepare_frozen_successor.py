"""Frozen local proposals enter a prepared successor without acquiring labels."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.theory_consolidation import _read_library
from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study
from test_prepare_pr3_source_successor import save, source_spec


def frozen_spec(tmp_path):
    spec = source_spec(tmp_path)
    original = tmp_path / 'original_preparation'
    preparation.prepare(spec, original)
    plan = preparation._read(original / 'source_plan.json')
    sources = [q['theory_target']['context_ref']['source_id']
               for q in plan['prepared_public_questions'].values()]
    body = Program('arg', name='x')
    candidate = {'id': 'new', 'version': 1, 'kind': 'program_revision_relation',
        'purpose': 'structural_revision', 'before_pattern': body.to_dict(),
        'definition': Definition('fixture_after', ('x',), body).to_dict(),
        'holes': [{'name': 'x', 'value_type': 'scalar', 'output_unit': '1', 'time_contract': ['current']}],
        'fixed_role_signature': {'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}},
        'output_unit': '1', 'source_ids': sources[:2],
        'source_reconstruction': [{'source_id': s, 'exact': True} for s in sources[:2]],
        'active_library': False, 'transfer_status': 'not_established',
        'scientific_support_added': False, 'old_closure_reconstructible': True}
    old = {**deepcopy(candidate), 'id': 'old'}
    baseline_ref = save(tmp_path / 'old_library.json', {'candidates': {'old': old}})
    library_ref = save(tmp_path / 'frozen_library.json', {'candidates': {'old': old, 'new': candidate}})
    spec['materials']['initial_library_ref'] = baseline_ref
    spec['frozen_source'] = {
        'manifest_ref': save(tmp_path / 'freeze.json', {'status': 'frozen', 'new_qualified_scientific_objects': 0}),
        'source_plan_ref': {'file': str(original / 'source_plan.json'), 'json_path': []},
        'source_summary_ref': save(tmp_path / 'source_summary.json', {'failures': 9}),
        'accounting_ref': save(tmp_path / 'source_accounting.json', {'accounting_and_preservation_checks_pass': True}),
        'library_ref': library_ref, 'new_candidate_ids': ['new'],
        'source_study': str(tmp_path / 'source'),
        'common_base_archive_ref': {'file': spec['materials']['baseline_archive'], 'json_path': []},
        'reconstruction_diagnostic_ref': save(tmp_path / 'reconstruction.json', {'old_reconstructible': True})}
    spec['design'] = {'endpoints': {'prediction': 'terminal MSE', 'cost': 'source plus successor',
        'wrong_transfer': 'paired degradation', 'confirmation': 'original96', 'actions': 'actual research actions'}}
    return spec, candidate


def prohibit_science(monkeypatch):
    monkeypatch.setattr(study, 'run_study', lambda *a, **k: pytest.fail('preparation ran a study'))
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda *a, **k: pytest.fail('preparation constructed a lab'))
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('preparation predicted'))


def test_reconstructible_inactive_source_proposal_is_prepared_without_world_support(tmp_path, monkeypatch):
    spec, candidate = frozen_spec(tmp_path)
    prohibit_science(monkeypatch)
    result = preparation.prepare_frozen_successor(spec, tmp_path / 'successor')
    assert result['qualified_local_proposal_count'] == 1
    assert result['proposal_qualification'][0]['status'] == 'qualified_local_proposal'
    assert not result['reconstruction_blocks_successor']
    assert not result['successor_execution_authorized']
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0
    keep = _read_library(result['automatic_C_library_ref'], tmp_path)
    assert keep['candidates']['new'] == candidate
    assert not keep['candidates']['new']['active_library']
    assert keep['candidates']['new']['transfer_status'] == 'not_established'
    assert result['individual_object']['selected_candidate_id'] == 'new'
    assert result['individual_object']['status'] == 'prepared_not_executed'
    assert result['new_class_intervention']['status'] == 'UNKNOWN'
    target = preparation._read(tmp_path / 'successor/target_plan.json')
    assert 'scripts/prepare_pr3_source_successor.py' in target['runtime_snapshot']
    assert target['arms'] == list(study.ORGANIZATION_ARMS)
    assert not (tmp_path / 'successor/individual_remove_plan.json').exists()
    assert result['individual_object']['matched_cases_ref']['json_path'] == ['cases']
    removal = preparation._value(result['individual_object']['plan_ref'])
    assert removal['arms'] == target['arms']
    assert removal['cases'] == target['cases'] and removal['limits'] == target['limits']
    assert 'new' not in _read_library(result['individual_object']['frozen_remove_library_ref'], tmp_path)['candidates']
    assert target['limits']['confirmation_rows'] == 96
    assert target['limits']['confirmation_mse_threshold'] == 1e-10
    assert target['research_organization']['routing_allocation'] == {'compute': 150000}
    assert not (tmp_path / 'successor/source').exists()
    assert not (tmp_path / 'successor/target').exists()


def test_no_qualified_source_keeps_whole_strategy_with_identical_B_library(tmp_path, monkeypatch):
    spec, candidate = frozen_spec(tmp_path)
    candidate['source_ids'] = candidate['source_ids'][:1]
    save(Path(spec['frozen_source']['library_ref']['file']), {'candidates': {**preparation._read(spec['materials']['initial_library_ref']['file'])['candidates'], 'new': candidate}})
    prohibit_science(monkeypatch)
    result = preparation.prepare_frozen_successor(spec, tmp_path / 'successor')
    assert result['qualified_local_proposal_count'] == 0
    assert result['whole_strategy']['arms'] == list(study.ORGANIZATION_ARMS)
    assert result['whole_strategy']['source_failure_denominator_preserved']
    assert result['whole_strategy']['no_new_object_C_falls_back_to_B']
    assert preparation._read(result['automatic_C_library_ref']['file']) == {
        'source_ref': spec['materials']['initial_library_ref']}
    b_input = save(tmp_path / 'B_fixed_input.json', {'source_ref': spec['materials']['initial_library_ref']})
    assert _read_library(result['automatic_C_library_ref'], tmp_path) == _read_library(b_input, tmp_path)
    assert result['individual_object']['status'] == 'not_applicable_no_qualified_new_object'
    assert not (tmp_path / 'successor/individual_remove_plan.json').exists()
    assert result['closed_source_accounting_ref'] == spec['frozen_source']['accounting_ref']


@pytest.mark.parametrize('purpose', ['structural_revision', 'parameter_proposal', 'exact_edit_map'])
def test_proposal_use_does_not_require_exact_after_parameter_prediction(tmp_path, purpose):
    spec, candidate = frozen_spec(tmp_path)
    candidate['purpose'] = purpose
    candidate['parameter_relations'] = []
    save(Path(spec['frozen_source']['library_ref']['file']), {'candidates': {**preparation._read(spec['materials']['initial_library_ref']['file'])['candidates'], 'new': candidate}})
    result = preparation.prepare_frozen_successor(spec, tmp_path / 'successor')
    assert result['qualified_local_proposal_count'] == 1
    assert result['proposal_qualification'][0]['purpose'] == purpose
    assert 'ordinary' in result['target_use']['parameter_proposal']
    assert 'Sufficient-condition' in result['target_use']['exact_edit_map']


def test_missing_source_type_contract_remains_unknown_and_old_pool_is_available(tmp_path):
    spec, candidate = frozen_spec(tmp_path)
    candidate['holes'][0].pop('output_unit')
    save(Path(spec['frozen_source']['library_ref']['file']), {'candidates': {**preparation._read(spec['materials']['initial_library_ref']['file'])['candidates'], 'new': candidate}})
    result = preparation.prepare_frozen_successor(spec, tmp_path / 'successor')
    assert result['proposal_qualification'][0]['status'] == 'UNKNOWN'
    assert 'declared_roles_units_and_time_required' in result['proposal_qualification'][0]['reasons']
    assert list(_read_library(result['automatic_C_library_ref'], tmp_path)['candidates']) == ['old']


def test_successor_seeds_must_match_the_unconsumed_registration(tmp_path):
    spec, _ = frozen_spec(tmp_path)
    original = tmp_path / 'original_preparation/target_plan.json'
    seeds = [case['seed'] for case in preparation._read(original)['cases']]
    spec['design']['target_seed_readiness'] = {
        'original_unexecuted_target_plan_ref': {'file': str(original), 'json_path': []},
        'expected_target_seeds': seeds,
        'unconsumed_target_directories_to_check': [str(tmp_path / 'consumed_target')]}
    (tmp_path / 'consumed_target').mkdir()
    with pytest.raises(ValueError, match='target directory already exists'):
        preparation.prepare_frozen_successor(spec, tmp_path / 'successor')
    assert not (tmp_path / 'successor/registration.json').exists()


def test_successor_entry_requires_its_separate_execution_authorization(tmp_path, monkeypatch):
    from hegel_machine.cognition import anonymous_research_archive as archives

    spec, _ = frozen_spec(tmp_path)
    output = tmp_path / 'successor'
    preparation.prepare_frozen_successor(spec, output)
    monkeypatch.setattr(archives, 'prepare_frozen_source_archive',
        lambda *a, **k: pytest.fail('unapproved successor materialized source contexts'))
    monkeypatch.setattr(study, 'run_study', lambda *a, **k: pytest.fail('unapproved successor ran'))
    with pytest.raises(ValueError, match='successor execution is not authorized'):
        preparation.run_prepared_successor(output)


@pytest.mark.parametrize('workers', [None, 3])
def test_successor_entry_passes_all_source_contexts_to_the_original_formal_loop(tmp_path, monkeypatch, workers):
    from hegel_machine.cognition import anonymous_research_archive as archives

    spec, _ = frozen_spec(tmp_path)
    output = tmp_path / 'successor'
    registration = preparation.prepare_frozen_successor(spec, output)
    # Authorization is granted to this stub only, never the real D3 registration.
    registration['successor_execution_authorized'] = True
    preparation._write(output / 'registration.json', registration)
    all_predictors = ['old999.py', 'source_failure/native_v1.py', 'source_failure/native_v2.py']
    calls = []

    def original_exporter(source, base):
        calls.append((source, base))
        return {'files': {'source_catalog.json': '[]'},
            'research_roles': {'predictors': list(all_predictors)}}

    def original_loop(plan, destination, **kwargs):
        assert destination == output / 'target'
        assert plan['arms'] == list(study.ORGANIZATION_ARMS)
        assert 'provider_factory' not in kwargs
        assert kwargs['workers'] == (8 if workers is None else workers)
        knowledge = kwargs['knowledge']
        assert knowledge['research_roles']['predictors'] == all_predictors
        assert knowledge['finite_native_author']['predictor_paths'] == all_predictors
        assert knowledge['files'][spec['author']['path']] == Path(registration['author_ref']['file']).read_text()
        return {'status': 'stub_only_not_a_successor_execution'}

    monkeypatch.setattr(archives, 'prepare_frozen_source_archive', original_exporter)
    monkeypatch.setattr(study, 'run_study', original_loop)
    options = {} if workers is None else {'workers': workers}
    assert preparation.run_prepared_successor(output, **options)['status'] == 'stub_only_not_a_successor_execution'
    assert calls == [(spec['frozen_source']['source_study'], spec['frozen_source']['common_base_archive_ref']['file'])]


def test_prepared_budget_uses_production_full_pool_completion_contract(tmp_path):
    spec, _ = frozen_spec(tmp_path)
    spec['target']['limits']['compute_budget'] = 1500000
    budget = preparation._successor_completion_budget([f'model_{i}.py' for i in range(1935)], spec)
    assert budget['completion_reserve']['compute_cost'] == 1116384
    assert budget['required_per_case_compute'] == 1316384
    assert budget['registered_cap_sufficient']


def test_successor_entry_stops_before_science_if_actual_pool_is_underfunded(tmp_path, monkeypatch):
    from hegel_machine.cognition import anonymous_research_archive as archives

    spec, _ = frozen_spec(tmp_path)
    output = tmp_path / 'successor'
    registration = preparation.prepare_frozen_successor(spec, output)
    registration['successor_execution_authorized'] = True  # Stub scope only.
    preparation._write(output / 'registration.json', registration)
    monkeypatch.setattr(archives, 'prepare_frozen_source_archive', lambda *a: {
        'files': {}, 'research_roles': {'predictors': [f'model_{i}.py' for i in range(999)]}})
    monkeypatch.setattr(study, 'run_study', lambda *a, **k: pytest.fail('underfunded whole pool reached science'))
    with pytest.raises(ValueError, match='777248 compute per case'):
        preparation.run_prepared_successor(output)
    budget = preparation._read(output / 'launch_budget_preflight.json')
    assert budget['actual_common_model_count'] == 999
    assert budget['status'] == 'insufficient_no_scientific_execution'

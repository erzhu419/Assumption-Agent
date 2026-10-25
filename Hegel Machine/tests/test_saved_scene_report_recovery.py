"""Saved branch results remain readable when an ordinary hook has no selection."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import method_effect_query
from scripts import prepare_pr3_source_successor as preparation


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def training():
    before = {'observation_cost': 97, 'measurement_cost': 0, 'compute_cost': 100,
              'python_execution_seconds': 3., 'model_calls': 0, 'reported_tokens': 0}
    delta = {key: 0 for key in before}
    delta.update(compute_cost=7, python_execution_seconds=2.)
    return {'status': 'method_step_limit', 'events': [
        {'feedback': {'selection': {'selected_experiment': {'id': 'old'}}}},
        {'feedback': {'selection': None}},
        {'feedback': {'selection': {'selected_experiment': {'id': 'current', 'actions': []}},
                      'frozen_recall': {'work': [{'candidate_id': 'new', 'theory_ids': ['t1']}]}}}],
        'costs': {key: value + delta[key] for key, value in before.items()},
        'prior_research_cost': before, 'continuation_cost': delta,
        'saved_scene_cutoff': {'event': 0, 'rows': 97}, 'fresh_confirmation': None,
        'same_paid_source_continuation': {'status': 'method_step_limit',
            'native_attempts': 0, 'new_method_results': 8, 'boundary': None},
        'model': None, 'paid_observations': [], 'evaluated_model': None,
        'limits': {'confirmation_mse_threshold': 1e-10}}


def test_branch_summary_keeps_no_selection_and_following_actual_selection(tmp_path):
    data = training()
    original = deepcopy(data)
    result = preparation._saved_scene_arm_result('pr3', data, tmp_path,
        ['new'], {'compute_cost': 1}, ['candidate.py'])

    assert result['selected_experiments'] == [{'id': 'current', 'actions': []}]
    assert result['new_object_usage']['new'] == {'attempts': 1, 'generated_theory_ids': ['t1']}
    assert result['continuation_cost'] == data['continuation_cost']
    assert result['candidate_paths'] == ['candidate.py']
    assert 'result' not in result  # The full history is already persisted in training.json.
    assert data == original
    assert method_effect_query.event_experiment_selection({'feedback': {'selection': None}}) == {}


def test_recover_saved_reports_executes_and_writes_nothing(tmp_path, monkeypatch):
    root = tmp_path / 'original/run'
    data = training()
    manifest = tmp_path / 'scene_manifest.json'
    save(manifest, {'library_views': {'new_ids': ['new']}})
    save(root / 'protocol.json', {'source_manifest_ref': {'file': str(manifest), 'json_path': []},
        'original_sunk_costs_once': data['prior_research_cost'],
        'cases': [{'id': 'saved_delay_event19'}], 'limits': data['limits'],
        'evaluation': {'rows': 0}, 'prefix_rows': 97, 'author': {},
        'research_organization': {'phase': 'development'}, 'post97_prediction_contract': {'rows': 48}})
    save(root.parent / 'run_status.json', {'status': 'stopped_failure_preserved',
        'error': "AttributeError: 'NoneType' object has no attribute 'get'"})
    for arm in ('fixed_pr2', 'pr3'):
        save(root / arm / 'training.json', data)
        save(root / arm / 'research_state.json', {'questions': [], 'revisions': [],
            'theory_index': {}, 'relation_research': {}, 'exported_candidate_paths': ['candidate.py']})
        save(root / arm / 'scene_acceptance.json', {'new_costs': {'compute_cost': 1}})
    originals = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}

    def prohibited(*_args, **_kwargs):
        pytest.fail('report recovery must not execute research, prediction or writes')

    monkeypatch.setattr(preparation, '_source_research_steps', prohibited)
    monkeypatch.setattr(preparation, 'run_prepared_saved_scene', prohibited)
    monkeypatch.setattr(preparation, '_write', prohibited)
    monkeypatch.setattr(method_effect_query, 'evaluate_saved_scene_predictions', prohibited)
    recovered = preparation.recover_saved_scene_reports(root)

    summary, effects = recovered['summary'], recovered['effects']
    assert summary['additional_costs']['compute_cost'] == 14
    assert summary['shared_original_sunk_costs_once'] == data['prior_research_cost']
    assert summary['post97_prediction_diagnostic'] == 'not_executed_candidates_require_separate_freeze'
    assert summary['comparison']['post97_prediction_change'] is None
    assert 'readout_ref' not in summary['comparison']
    assert effects['status'] == 'recorded_comparison'
    assert effects['new_predictions'] == effects['new_method_executions'] == 0
    assert effects['study_refs'] == []  # No nonexistent completed summary is cited.
    assert recovered['recovery']['files_written'] == 0
    assert {path: path.read_bytes() for path in tmp_path.rglob('*.json')} == originals


@pytest.mark.parametrize('entry,registration_name', [
    ('run_prepared_source_comparisons', 'comparison_registration.json'),
    ('run_prepared_source_research', 'research_registration.json'),
])
def test_source_continuation_freezes_its_own_registered_context(tmp_path, monkeypatch, entry, registration_name):
    from hegel_machine.cognition import code_agent
    from scripts import run_anonymous_revision_inquiry as study

    save(tmp_path / registration_name, {'source_study': str(tmp_path / 'old'),
        'plan_ref': {'file': str(tmp_path / 'plan.json'), 'json_path': []},
        'initial_archive_ref': {'file': str(tmp_path / 'archive.json'), 'json_path': []}})
    save(tmp_path / 'plan.json', {})
    save(tmp_path / 'archive.json', {})
    monkeypatch.setattr(study, '_prepare_organization', lambda *_args: ({}, {'files': {}}))
    class ReachedSourceFreeze(Exception):
        pass
    def stop_at_registered_source(*_args):
        raise ReachedSourceFreeze
    monkeypatch.setattr(code_agent, 'clone_or_copy', stop_at_registered_source)
    monkeypatch.setattr(preparation, '_source_research_steps', lambda *_args, **_kwargs:
                        pytest.fail('Checking source storage setup must not run research.'))
    with pytest.raises(ReachedSourceFreeze):
        getattr(preparation, entry)(tmp_path)

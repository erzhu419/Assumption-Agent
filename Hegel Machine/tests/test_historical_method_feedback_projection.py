"""Historical method returns preserve their original public input and phase."""

import json

import pytest

from hegel_machine.cognition.research_episodes import project_method_use_feedback


def source_scene(tmp_path, *, before_loss=1., after_loss=.25, invalid=False):
    def save(name, value, *path):
        file = tmp_path / name
        file.write_text(json.dumps(value))
        return {'file': str(file), 'json_path': list(path)}

    scope = save('scope.json', {'public_spec': {'controls': ['u'],
        'feature_bounds': {'u': [-1., 1.]}, 'max_probe_length': 24},
        'rows': [{'features': {'u': .1}, 'target': .2, 'history': []},
                 {'features': {'u': .2}, 'target': .4, 'history': [{'u': .1}]}]})
    binding = {'candidate_id': 'source_method', 'candidate_version': 2,
        'mapping': {'role_0': 'u'}, 'fit_split': {'training_records': [0]},
        'paid_cutoff': {'records': 1, 'rows': 2}, 'library_path': 'library.json'}
    binding_ref = save('library.json', {'transfers': [{'binding': binding,
        'theory_ref': {'id': 'actual_export', 'version': 1}}]}, 'transfers', 0, 'binding')
    validity = save('validity.json', {'comparison_requested': 2, 'comparison_completed': 2,
        'comparison_evaluations': 2, 'valid_complete_mse_entries': 1 if invalid else 2,
        'invalid_entries': [{'returncode': 1}] if invalid else [], 'timeouts': 0})
    row = {'method_ref': {'id': 'source_method', 'version': 2},
        'public_before_call': {'public_spec_ref': {**scope, 'json_path': ['public_spec']},
            'paid_evidence_ref': {**scope, 'json_path': ['rows']},
            'question_id': 'public_response', 'question_meaning_version': 1,
            'parent_question_ref': {**scope, 'json_path': ['question']},
            'parent_exposure': 'unmodeled_not_exposed', 'minimum_improvement': 1e-10},
        'source_provenance': {'method_binding_ref': binding_ref},
        'actual_application': {'replay_search_completed': True, 'method_replay_compute': 7,
            'method_work_ref': save('work.json', {'status': 'completed'}),
            'comparison': {'execution_validity_ref': validity, 'requested': 2, 'completed': 2,
                'rows': 2, 'old_pool_best_mse': before_loss, 'candidate_mse': after_loss,
                'score_ref': save('score.json', {'model': 'method.py'}, 'result'),
                'comparison_ref': save('comparison.json', {'complete': True})}},
        'subsequent_outcome': {'run_status': 'python_execution_budget_exhausted',
            'submitted_model': None, 'scientific_censoring': 'later_budget_exhaustion',
            'summary_trial_ref': save('summary.json', {'costs': {'compute_cost': 123}})}}
    return save('source.json', row), row


@pytest.mark.parametrize('after_loss, expected_gain', [(.25, True), (2., False)])
def test_projects_completed_return_without_post_comparison_parent(tmp_path, after_loss, expected_gain):
    reference, source = source_scene(tmp_path, after_loss=after_loss)
    result = project_method_use_feedback(reference, tmp_path)
    assert result['status'] == 'completed'
    assert result['execution_complete'] is True
    comparison = result['working_comparison']
    assert (comparison['after_loss'] < comparison['before_loss']) is expected_gain
    assert comparison['data_role'] == 'exposed_development'
    assert result['profile']['representation']['kind'] == 'unmodeled'
    assert result['profile']['residual']['loss'] is None
    assert result['profile']['residual']['minimum_improvement'] == 1e-10
    assert result['evidence_basis']['paid_evidence_ref'] == source['public_before_call']['paid_evidence_ref']
    assert result['evidence_basis']['mapping'] == {'role_0': 'u'}
    assert result['method_ref'] == source['method_ref']
    assert result['parent_question_ref'] == {'question_id': 'public_response', 'meaning_version': 1}
    assert result['parent_question_source_ref'] == source['public_before_call']['parent_question_ref']
    assert result['generated_theories'] == [{'id': 'actual_export', 'version': 1}]
    assert result['actual_model_refs'][0]['json_path'] == ['model']
    assert result['fit_compute_cost'] == 7
    assert 'administrative_censoring' not in result
    assert result['subsequent_outcome']['scientific_censoring'] == 'later_budget_exhaustion'
    assert result['independent_confirmation'] is False


def test_invalid_strong_pool_member_keeps_comparison_incomplete(tmp_path):
    reference, _ = source_scene(tmp_path, invalid=True)
    result = project_method_use_feedback(reference, tmp_path)
    assert result['status'] == 'incomplete'
    assert result['working_comparison']['complete'] is False
    assert result['comparison_execution_validity_ref']['file'].endswith('validity.json')


def test_after_only_confirmation_does_not_become_paired_future_gain(tmp_path):
    reference, source = source_scene(tmp_path)
    source['subsequent_outcome']['fresh_confirmation'] = {'passed': True, 'rows': 96, 'mse': 1e-12}
    source['subsequent_outcome']['submitted_model'] = 'method.py'
    (tmp_path / 'source.json').write_text(json.dumps(source))
    result = project_method_use_feedback(reference, tmp_path)
    assert result['confirmation']['passed'] is True
    assert result['confirmation']['paired_method_gain'] is None
    assert result['confirmation']['scope'] == 'submitted_predictor_only_no_before_predictor_column'
    assert result['world_support_updated'] is False

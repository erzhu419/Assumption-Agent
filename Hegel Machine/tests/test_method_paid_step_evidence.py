"""Paid split diagnostics retain provenance without replay or library expansion."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.research_episodes import project_method_paid_step_evidence


def paid_scene(tmp_path, *, common=False, replaced=False):
    def save(path, value, *keys):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(value))
        return {'file': str(file), 'json_path': list(keys)}

    rows = [{'features': {'u': index}, 'target': float(index), 'history': []}
            for index in range(6)]
    records = [{'row_indices': indices, 'rows': [rows[index] for index in indices]}
               for indices in ([0, 1], [2, 3, 4], [5])]
    save('laboratory_observations.json', records)
    scope = save('_paid_python_scores/_evaluated_model/scope_0000.json', {'rows': rows})
    score = save('_paid_python_scores/_evaluated_model/assessment_000000.json', {
        'scope_ref': scope, 'model': 'closed.py', 'execution_costs': {'compute_cost': 6},
        'closed_prediction_state': {'row_count': 6,
            'predictions': [0., 2., 4., 6., 8., 10.], 'representations': list(range(6))},
        'result': {'representation_diagnostic': {'status': 'no_exact_conflicts'},
            'paid_residuals': {'examples': [{'row_index': 5, 'squared_error': 25.}]}}})
    source_refs = [{'file': str(tmp_path / f'unread_source_{index}.json'),
        'json_path': ['research_program']} for index in range(2)]
    binding = {'candidate_id': 'actual_method', 'candidate_version': 2,
        'library_path': str(tmp_path / 'library.json'), 'mapping': {'role_0': 'u'},
        'source_refs': source_refs,
        'research_program_contract': {'operation': 'discover_conditional_coordinates'},
        'fit_split': {'training_records': [0, 2], 'validation_records': [1],
            'train_rows': 3, 'holdout_rows': 3}, 'paid_cutoff': {'records': 3, 'rows': 6}}
    work = [{'seed_index': index, 'source_ref': reference, 'compute_cost': 5,
        'search_completed': True, 'status': 'fitted_candidate'}
        for index, reference in enumerate(source_refs)]
    saved = deepcopy(binding)
    if replaced:
        saved['paid_cutoff'] = {'records': 4, 'rows': 8}
    key = 'research_program_transfer:frozen_recall:public@1:actual_method'
    theory_ref = {'id': 'paid_method_seed', 'version': 1}
    library = {'candidates': {'actual_method': {'source_outcomes': [
        {'source_ref': reference, 'input_mapping': {'role_0': 'u'}} for reference in source_refs]}},
        'transfers': [{'binding': binding, 'theory_ref': theory_ref}], 'executions': {key: {
            'basis': {'binding': saved}, 'result': {'seed_replay_work': work}}}}
    library_ref = save('library.json', library)
    save('shared/theory_index.json', {'entries': {theory_ref['id']: {
        'version': 1, 'consolidation_ref': {**binding, 'source_ref': source_refs[0], 'seed_index': 0},
        'program': {'op': 'mul', 'value': None, 'name': None, 'lag': 0, 'children': [
            {'op': 'const', 'value': 2., 'name': None, 'lag': 0, 'children': []},
            {'op': 'feature', 'value': None, 'name': 'u', 'lag': 0, 'children': []}]}}}})
    paid_ref = {**scope, 'json_path': ['rows']}
    method = {'id': 'actual_method', 'version': 2}
    if common:
        source = {'method_ref': method, 'invoked_method_ref': method,
            'invocation_binding': binding, 'evidence_basis': {'paid_evidence_ref': paid_ref},
            'fit_compute_cost': 10, 'execution_complete': True, 'seed_replay_work': work,
            'generated_theories': [theory_ref],
            'working_comparison': {'after_score_ref': score}}
    else:
        source = {'method_ref': method, 'source_provenance': {'method_binding_ref': {
            **library_ref, 'json_path': ['transfers', 0, 'binding']}},
            'public_before_call': {'paid_evidence_ref': paid_ref,
                'question_id': 'public', 'question_meaning_version': 1},
            'actual_application': {'comparison': {'score_ref': {**score, 'json_path': ['result']}},
                'method_replay_compute': 10, 'replay_search_completed': True}}
    return save('use.json', source), source, work


@pytest.mark.parametrize('common', [False, True])
def test_original_paid_split_and_seeds_are_read_without_loading_source_programs(tmp_path, common, monkeypatch):
    reference, source, work = paid_scene(tmp_path, common=common)
    original_read = Path.read_text

    def bounded_read(path, *args, **kwargs):
        assert path.name not in {'training.json', 'research_state.json'}
        assert not path.name.startswith('unread_source_')
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', bounded_read)
    result = project_method_paid_step_evidence(reference, tmp_path)
    assert result['status'] == 'completed' and result['fit_compute_cost'] == 10
    assert result['seed_replay_work'] == work
    assert result['method_ref'] == source['method_ref']
    assert result['split_rows']['training']['row_indices'] == [0, 1, 5]
    assert result['split_rows']['training']['loss'] == pytest.approx(26 / 3)
    assert result['split_rows']['holdout']['row_indices'] == [2, 3, 4]
    assert result['split_rows']['holdout']['loss'] == pytest.approx(29 / 3)
    assert result['source_method_payload_refs'][1]['input_mapping'] == {'role_0': 'u'}
    assert result['representation_diagnostic_ref']['json_path'] == ['result', 'representation_diagnostic']
    assert result['localization_scope'] == 'joint_replay_only_no_individual_step_ablation'
    assert result['new_fits'] == result['new_predictions'] == result['new_paid_observations'] == 0
    assert not result['world_support_updated'] and not result['independent_confirmation']


def test_replaced_execution_does_not_attribute_later_seed_bill_to_original_use(tmp_path):
    reference, _, _ = paid_scene(tmp_path, replaced=True)
    result = project_method_paid_step_evidence(reference, tmp_path)
    assert result['status'] == 'completed' and result['fit_compute_cost'] == 10
    assert result['seed_replay_work'] is None
    assert result['seed_work_status'] == 'UNKNOWN_execution_key_now_addresses_different_paid_fit'
    assert result['retained_execution_ref']['json_path'][0] == 'executions'


def test_changed_saved_paid_rows_do_not_acquire_old_prediction_diagnostics(tmp_path):
    reference, source, _ = paid_scene(tmp_path, common=True)
    paid = tmp_path / 'changed_paid.json'
    rows = json.loads(Path(source['evidence_basis']['paid_evidence_ref']['file']).read_text())['rows']
    rows[0]['target'] = 100.
    paid.write_text(json.dumps(rows))
    source['evidence_basis']['paid_evidence_ref'] = {'file': str(paid), 'json_path': []}
    Path(reference['file']).write_text(json.dumps(source))
    result = project_method_paid_step_evidence(reference, tmp_path)
    assert result['status'] == 'UNKNOWN' and result['split_rows'] == {}
    assert result['reason'] == 'saved_predictions_do_not_match_original_paid_view'


def test_thin_frozen_library_resolves_one_method_without_expanding_old_executions(tmp_path):
    reference, source, _ = paid_scene(tmp_path, common=True)
    library = tmp_path / 'library.json'
    declaration = json.loads(library.read_text())
    (tmp_path / 'frozen.json').write_text(json.dumps({'candidates': declaration['candidates'],
        'executions': {'unread': {'binding_ref': {'file': 'missing_old_pool.json', 'json_path': []}}}}))
    library.write_text(json.dumps({'source_ref': {'file': 'frozen.json', 'json_path': []}}))
    result = project_method_paid_step_evidence(reference, tmp_path)
    assert result['status'] == 'completed'
    assert result['source_method_payload_refs'][0]['source_outcome_ref']['file'].endswith('frozen.json')


def test_actual_common_retained_feedback_supplies_paid_step_diagnostics(tmp_path, monkeypatch):
    from test_common_method_feedback import scene, use_and_compare

    agent, lab, method, question = scene(tmp_path, useful=True)
    use_and_compare(agent, lab, question)
    reference, = agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs']
    paid = deepcopy(agent.costs)
    assert not (agent.workspace / 'shared/theory_index.json').exists()
    original_read = Path.read_text

    def bounded_read(path, *args, **kwargs):
        assert path.name not in {'research_state.json', 'training.json'}
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', bounded_read)
    result = project_method_paid_step_evidence(reference, agent.workspace)
    assert result['status'] == 'completed'
    assert result['split_rows']['training']['rows'] == result['split_rows']['holdout']['rows'] == 8
    assert result['split_rows']['training']['loss'] < 1e-16
    assert result['split_rows']['holdout']['loss'] < 1e-16
    assert len(result['source_method_payload_refs']) == len(result['seed_replay_work']) == 2
    assert len(result['paid_seed_baselines']) == 2
    for baseline in result['paid_seed_baselines']:
        theory = agent.theory_index[baseline['theory_ref']['id']]
        assert baseline['theory_ref']['version'] == theory['version']
        assert baseline['program'] == theory['program']
        assert baseline['program_ref']['file'] == str(agent.workspace / 'research_state.json')
        assert baseline['program_ref']['json_path'] == ['theory_index', theory['id'], 'program']
        assert baseline['fit_compute_cost'] is not None
    assert agent.costs == paid


def test_deduplicated_source_retains_actual_reused_seed_instead_of_source_position(tmp_path):
    reference, source, _ = paid_scene(tmp_path, common=True)
    source['seed_replay_work'][1] = {'source_ref': source['invocation_binding']['source_refs'][1],
        'status': 'same_replay_inputs', 'reused_seed_index': 0, 'compute_cost': 0}
    source['fit_compute_cost'] = 5
    Path(reference['file']).write_text(json.dumps(source))
    result = project_method_paid_step_evidence(reference, tmp_path)
    second = result['source_method_payload_refs'][1]
    assert second['source_index'] == 1 and second['seed_index'] == 0
    assert result['seed_replay_work'] == source['seed_replay_work']
    assert result['fit_compute_cost'] == 5


def test_exact_paid_programs_and_original_fees_are_reused_across_seed_dedup(tmp_path):
    reference, _, work = paid_scene(tmp_path, common=True)
    result = project_method_paid_step_evidence(reference, tmp_path)
    left, right = result['paid_seed_baselines']
    assert left['program'] == right['program']
    assert left['program_ref'] == right['program_ref']
    assert [item['fit_compute_cost'] for item in (left, right)] == [row['compute_cost'] for row in work]
    assert left['exact_program_deduplication'] is False and right['exact_program_deduplication'] is True
    assert left['training_loss'] == result['split_rows']['training']['loss']
    assert right['validation_loss'] == result['split_rows']['holdout']['loss']
    assert left['research_program_scope'] == 'original_source_plan_not_target_execution_graph'


def test_paid_program_uses_exact_old_version_in_thin_index_history(tmp_path):
    reference, _, _ = paid_scene(tmp_path, common=True)
    path = tmp_path / 'shared/theory_index.json'
    indexed = json.loads(path.read_text())
    original = indexed['entries']['paid_method_seed']
    newer = deepcopy(original)
    newer.update(version=2, program={'op': 'const', 'value': 100., 'name': None, 'lag': 0, 'children': []})
    newer['consolidation_ref']['paid_cutoff'] = {'records': 4, 'rows': 8}
    newer['history'] = [original]
    indexed['entries']['paid_method_seed'] = newer
    path.write_text(json.dumps(indexed))
    result = project_method_paid_step_evidence(reference, tmp_path)
    first = result['paid_seed_baselines'][0]
    assert first['program'] == original['program']
    assert first['program_ref']['json_path'] == ['entries', 'paid_method_seed', 'history', 0, 'program']
    assert first['theory_ref']['version'] == 1


def test_unknown_original_seed_cost_does_not_prevent_saved_program_recovery(tmp_path):
    reference, _, _ = paid_scene(tmp_path, replaced=True)
    result = project_method_paid_step_evidence(reference, tmp_path)
    first, = result['paid_seed_baselines']
    assert first['program_ref']['json_path'] == ['entries', 'paid_method_seed', 'program']
    assert first['fit_compute_cost'] is None
    assert result['fit_compute_cost'] == 10

"""Saved research differences are not IDs, new executions, or missing zeros."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_effect_query import (
    SUCCESSOR_EFFECT_KIND, evaluate_method_effect, registered_successor_effect,
    evaluate_saved_scene_predictions,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, *, right_program=None, right_predictions=None, right_context=None):
    plan = {'cases': [{'id': 'case_0', 'seed': 7, 'confirmation_seed': 9, 'evaluation_seed': 11}],
        'limits': {'compute_budget': 1000}, 'evaluation': {'rows': 512, 'mse_threshold': 1e-10},
        'prefix_rows': 72, 'author': {'kind': 'finite_native'}}
    save(tmp_path / 'protocol.json', plan)
    trials = []
    for i, arm in enumerate(('fixed_pr2', 'pr3')):
        workspace = tmp_path / 'case_0' / arm
        program = Program('feature', name='u').to_dict()
        if i and right_program:
            program = right_program
        model = f'candidate_{i}.py'
        context = right_context if i and right_context else {'source_id': 'same_target'}
        q = {'id': 'response', 'meaning_version': 1, 'context_ref': context,
            'theory_target': {'role_signature': {'u': {'role': 'control', 'unit': '1', 'time': 'current'}},
                             'output_unit': '1'},
            'query_semantics': {'query_expression': {'op': 'Readout', 'binding': {'features': {'u': 1.}}}},
            'native_condition_research': [{'candidates': [{'program': program, 'model': model}]}],
            'frozen_recall_execution': {'basis': {'candidates': [{'id': 'used'}, {'id': 'unused'}]},
                'result': {'work': [{'candidate_id': 'used', 'theory_ids': ['t']}]}},
            'python_forecast_execution': {'source_ref': {'file': 'forecast.json', 'json_path': []}}}
        state = {'questions': [q], 'theory_index': {}, 'revisions': [],
            'evaluated_model': {'model': model, 'source_model': model}}
        costs = dict(observation_cost=72, measurement_cost=0, compute_cost=100 + i,
                     python_execution_seconds=1. + i, model_calls=0, reported_tokens=0)
        training = {'model': model, 'workspace': str(workspace), 'events': [], 'paid_observations': [],
            'limits': {'confirmation_mse_threshold': 1e-10}, 'fresh_confirmation': {'passed': True},
            'costs': costs}
        save(workspace / 'training.json', training)
        save(workspace / 'research_state.json', state)
        save(workspace / 'forecast.json', {'forecasts': [{'model': 'same_saved_model.py', 'returncode': 0,
            'prediction_inputs': [{'features': {'u': 1.}, 'history': []}],
            'predictions': right_predictions if i and right_predictions else [1.]}]})
        trials.append({'case': 'case_0', 'arm': arm, 'status': 'completed', 'costs': costs,
            'fresh_confirmation_passed': True, 'terminal': {'returncode': 0, 'mse': 0., 'passed': True}})
    save(tmp_path / 'summary.json', {'status': 'completed', 'trials': trials})
    return tmp_path


def prohibit_execution(monkeypatch):
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('readout predicted'))


def paid_prediction_fixture(tmp_path):
    """Both trained branches have the same97 rows; later48 are source-only."""
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program

    records = []
    for index, count in enumerate((16, 16, 16, 16, 16, 17, 12, 12, 12, 12)):
        records.append({'action': {'kind': 'trajectory'}, 'rows': [
            {'features': {'u': float(index + row)}, 'history': [],
             'target': float(index + row)} for row in range(count)]})
    source = tmp_path / 'original_training.json'
    save(source, {'paid_observations': records})
    study = tmp_path / 'study'
    save(study / 'protocol.json', {'limits': {'compute_budget': 1000,
        'execution_seconds': 30., 'confirmation_mse_threshold': 1e-10}})
    trials = []
    for arm in ('fixed_pr2', 'pr3'):
        workspace = study / arm
        save(workspace / 'training.json', {'paid_observations': records[:6],
            'evaluated_model': {'model': '_evaluated_model/parent.py'},
            'costs': {'compute_cost': 100, 'python_execution_seconds': 1.,
                      'model_calls': 0, 'observation_cost': 145}})
        model = workspace / '_evaluated_model/parent.py'
        model.parent.mkdir()
        model.write_text(export_temporal_program(Program('feature', name='u')))
        trials.append({'arm': arm, 'result_ref': {'file': str(workspace / 'training.json'),
                                                'json_path': []}})
    save(study / 'summary.json', {'status': 'completed', 'trials': trials})
    return study, {'file': str(source), 'json_path': []}


def test_paid_later_prediction_requires_both_finished_training_jobs(tmp_path, monkeypatch):
    study, source = paid_prediction_fixture(tmp_path)
    document = json.loads((study / 'summary.json').read_text())
    document['status'] = 'running'
    save(study / 'summary.json', document)
    Path(source['file']).unlink()  # A pending run must not even open future labels.
    result = evaluate_saved_scene_predictions(study, source)
    assert result['reason'] == 'both_training_jobs_must_finish_before_freeze'
    assert not (study / 'post97_frozen_models').exists()


def test_paid_later_predictions_freeze_both_before_labels_and_charge_original_budget(
        tmp_path, monkeypatch):
    study, source = paid_prediction_fixture(tmp_path)
    original_reader = Path.read_text

    def tracked_reader(path, *args, **kwargs):
        if path == Path(source['file']):
            frozen = json.loads(original_reader(study / 'post97_parent_freeze.json'))
            assert frozen['status'] == 'both_parents_frozen_before_later_labels'
            assert len(frozen['arms']) == 2
            for entry in frozen['arms']:
                assert Path(entry['frozen_model']['workspace'], 'parent.py').is_file()
        return original_reader(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', tracked_reader)
    original = (study / 'fixed_pr2/_evaluated_model/parent.py').read_text()
    result = evaluate_saved_scene_predictions(study, source)
    assert result['status'] == 'recorded_comparison'
    assert result['comparison']['changed'] is False
    assert result['comparison']['right_minus_left_mse'] == 0.
    for entry in result['arms']:
        assert entry['evaluation']['mse'] == 0.
        assert len(entry['evaluation']['predictions']) == 48
        assert entry['charged_costs']['compute_cost'] == 48
        assert entry['charged_costs']['python_execution_seconds'] > 0
        assert entry['costs_after']['compute_cost'] == 148
        assert entry['costs_after']['model_calls'] == 0
        assert entry['costs_after']['observation_cost'] == 145
    assert result['new_fits'] == result['new_observations'] == result['new_model_calls'] == 0
    assert (study / 'fixed_pr2/_evaluated_model/parent.py').read_text() == original
    # A readout of the saved vectors must never execute the predictor again.
    import hegel_machine.cognition.code_agent as execution
    monkeypatch.setattr(execution, 'evaluate_python_model',
                        lambda *a, **kw: pytest.fail('saved prediction was repeated'))
    assert evaluate_saved_scene_predictions(study, source) == result


def test_three_registered_views_freeze_all_before_labels_and_reuse_predictions(tmp_path, monkeypatch):
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program

    study, source = paid_prediction_fixture(tmp_path)
    arms = ('fixed_pr2', 'pr3', 'remove_class')
    summary = json.loads((study / 'summary.json').read_text())
    training = study / 'remove_class/training.json'
    save(training, json.loads((study / 'pr3/training.json').read_text()))
    model = training.parent / '_evaluated_model/parent.py'
    model.parent.mkdir()
    model.write_text(export_temporal_program(Program('const', value=0.)))
    summary['trials'].append({'arm': 'remove_class',
        'result_ref': {'file': str(training), 'json_path': []}})
    save(study / 'summary.json', summary)
    original_reader = Path.read_text

    def tracked_reader(path, *args, **kwargs):
        if path == Path(source['file']):
            frozen = json.loads(original_reader(study / 'post97_parent_freeze.json'))
            assert frozen['status'] == 'all_parents_frozen_before_later_labels'
            assert [entry['arm'] for entry in frozen['arms']] == list(arms)
            assert all(Path(entry['frozen_model']['workspace'], 'parent.py').is_file()
                       for entry in frozen['arms'])
        return original_reader(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', tracked_reader)
    result = evaluate_saved_scene_predictions(study, source, arms=arms)
    assert result['status'] == 'recorded_comparison'
    assert [item['changed'] for item in result['comparisons']] == [False, True]
    assert [item['right_arm'] for item in result['comparisons']] == ['pr3', 'remove_class']
    assert all(entry['charged_costs']['compute_cost'] == 48 for entry in result['arms'])
    import hegel_machine.cognition.code_agent as execution
    monkeypatch.setattr(execution, 'evaluate_python_model',
                        lambda *a, **kw: pytest.fail('three-view receipt was predicted again'))
    assert evaluate_saved_scene_predictions(study, source, arms=arms) == result
    with pytest.raises(ValueError, match='different registered branches'):
        evaluate_saved_scene_predictions(study, source)


@pytest.mark.parametrize('failure', ['compute', 'python', 'fit'])
def test_unfunded_or_fit_initializing_paid_parent_is_unmeasured(tmp_path, monkeypatch, failure):
    study, source = paid_prediction_fixture(tmp_path)
    for arm in ('fixed_pr2', 'pr3'):
        path = study / arm / 'training.json'
        data = json.loads(path.read_text())
        if failure == 'compute':
            data['costs']['compute_cost'] = 953
        elif failure == 'python':
            data['costs']['python_execution_seconds'] = 30.
        else:
            (study / arm / '_evaluated_model/parent.py').write_text(
                'def fit(rows):\n raise RuntimeError("no new fit")\n'
                'def predict(row):\n return row["features"]["u"]\n')
        save(path, data)
    import hegel_machine.cognition.code_agent as execution
    monkeypatch.setattr(execution, 'evaluate_python_model',
                        lambda *a, **kw: pytest.fail('unfunded or fit model executed'))
    result = evaluate_saved_scene_predictions(study, source)
    assert result['status'] == 'UNKNOWN'
    assert result['comparison']['changed'] is None
    for arm in result['arms']:
        assert arm['status'] == 'UNKNOWN'
        assert arm['costs_before'] == arm['costs_after']
        assert arm['charged_costs'] == {'compute_cost': 0, 'python_execution_seconds': 0.}


def test_paid_later_prediction_detects_a_changed_frozen_parent(tmp_path):
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program

    study, source = paid_prediction_fixture(tmp_path)
    (study / 'pr3/_evaluated_model/parent.py').write_text(
        export_temporal_program(Program('const', value=0.)))
    result = evaluate_saved_scene_predictions(study, source)
    assert result['status'] == 'recorded_comparison'
    assert result['comparison']['changed'] is True
    assert result['comparison']['right_minus_left_mse'] > 0
    assert result['comparison']['left_within_threshold'] is True
    assert result['comparison']['right_within_threshold'] is False


def test_failed_paid_prediction_retains_attempted_cost_and_unknown_comparison(tmp_path, monkeypatch):
    study, source = paid_prediction_fixture(tmp_path)
    import hegel_machine.cognition.code_agent as execution

    def fail(*args, **kwargs):
        raise ValueError('recorded prediction failure')

    monkeypatch.setattr(execution, 'evaluate_python_model', fail)
    result = evaluate_saved_scene_predictions(study, source)
    assert result['status'] == 'UNKNOWN'
    assert result['comparison']['changed'] is None
    for arm in result['arms']:
        assert arm['reason'] == 'ValueError: recorded prediction failure'
        assert arm['charged_costs']['compute_cost'] == 48
        assert arm['charged_costs']['python_execution_seconds'] > 0.
        assert arm['status'] == 'UNKNOWN'


@pytest.mark.parametrize('saved_context', [False, True])
def test_real_parent_revision_refs_resolve_to_the_saved_event_document(
        tmp_path, monkeypatch, saved_context):
    root = fixture(tmp_path)
    prohibit_execution(monkeypatch)
    for index, arm in enumerate(('fixed_pr2', 'pr3')):
        workspace = root / 'case_0' / arm
        state_path = workspace / 'research_state.json'
        state = json.loads(state_path.read_text())
        state['questions'][0]['native_condition_research'][0]['candidates'].append({
            'model': 'before.py', 'program': Program('const', value=0.).to_dict()})
        save(state_path, state)
        path = workspace / 'training.json'
        training = json.loads(path.read_text())
        decision = {'status': 'candidate_retained', 'same_paid_rows': True,
            'incumbent': {'source_model': 'before.py', 'evaluated_rows': 1, 'mse': 1.},
            'candidate': {'model': f'candidate_{index}.py', 'evaluated_rows': 1}}
        training['paid_observations'] = [{'action': {'kind': 'trajectory'},
            'rows': [{'features': {'u': 1.}, 'history': [], 'y': 1.}]}]
        training['events'] = [{'action': 'run', 'feedback': {
            'operator': 'research_choice', 'research_question_id': 'response',
            'question_meaning_version': 1, 'paid_rows': 1,
            'evaluations': [
                {'path': 'before.py', 'returncode': 0, 'loss_kind': 'mse', 'mse': 1.,
                 'paid_residuals': {'evaluated_rows': 1}},
                {'path': f'candidate_{index}.py', 'returncode': 0, 'loss_kind': 'mse', 'mse': 0.,
                 'paid_residuals': {'evaluated_rows': 1}, 'retention_decision': decision}]}}]
        save(path, training)
        if saved_context:
            save(workspace / 'research_context.json', {'events': training['events']})
    result = registered_successor_effect(root)['cases'][0]['mechanism']
    for changes in result['actual_before_after_revision_refs']:
        change, = changes
        reference = change['action_ref']
        expected = 'research_context.json' if saved_context else 'training.json'
        assert Path(reference['file']).name == expected
        value = json.loads(Path(reference['file']).read_text())
        for key in reference['json_path']:
            value = value[key]
        assert value['status'] == 'candidate_retained'
        criterion = change['source_comparison']['criterion_ref']
        value = json.loads(Path(criterion['file']).read_text())
        for key in criterion['json_path']:
            value = value[key]
        assert value == 1e-10


def test_ids_are_not_program_or_prediction_changes_and_unused_objects_are_not_refuted(tmp_path, monkeypatch):
    root = fixture(tmp_path)
    prohibit_execution(monkeypatch)
    result = registered_successor_effect(root)
    row, = result['cases']
    m = row['mechanism']
    assert result['status'] == 'recorded_comparison'
    assert m['parent_model_ids_changed'] and m['parent_program_changed'] is False
    assert not m['candidate_programs_changed']
    assert m['stored_forecast_comparison']['changed'] is False
    assert m['objects'][1]['used']['usage_status'] == 'attempted'
    assert m['objects'][1]['unused']['usage_status'] == 'not_visited_under_recorded_policy'
    assert m['objects'][1]['unused']['utility_after_adoption'] == 'not_tested'
    assert row['right_minus_left']['target_costs']['compute_cost'] == 1
    assert not result['common_scheduler_repair_is_PR3_effect']


def test_final_training_parent_takes_precedence_over_the_last_hook_checkpoint(tmp_path, monkeypatch):
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program

    root = fixture(tmp_path)
    prohibit_execution(monkeypatch)
    literal = export_temporal_program(Program('feature', name='u'))
    for i, arm in enumerate(('fixed_pr2', 'pr3')):
        workspace = root / 'case_0' / arm
        model = f'_evaluated_model/final_{i}.py'
        (workspace / '_evaluated_model').mkdir()
        (workspace / model).write_text(literal)
        path = workspace / 'training.json'
        training = json.loads(path.read_text())
        training['evaluated_model'] = {'model': model, 'source_model': f'export_{i}.py'}
        save(path, training)
    mechanism = registered_successor_effect(root)['cases'][0]['mechanism']
    assert mechanism['parent_model_ids_changed'] is True
    assert mechanism['selected_parent_executable_comparison']['changed'] is False
    assert all('final_' in ref['file'] for ref in
               mechanism['selected_parent_executable_comparison']['model_refs'])
    for ref in mechanism['parent_refs']:
        assert Path(ref['file']).name == 'training.json'
        assert 'final_' in json.loads(Path(ref['file']).read_text())['evaluated_model']['model']


def test_program_changes_do_not_imply_changed_stored_predictions_or_independent_gain(tmp_path, monkeypatch):
    root = fixture(tmp_path, right_program=Program('abs', children=(Program('feature', name='u'),)).to_dict())
    prohibit_execution(monkeypatch)
    result = registered_successor_effect(root)
    row, = result['cases']
    assert row['mechanism']['candidate_programs_changed']
    assert row['mechanism']['parent_program_changed']
    assert row['mechanism']['stored_forecast_comparison']['changed'] is False
    assert row['right_minus_left']['terminal_mse'] == 0.
    assert row['object_causal_effect'] == 'requires_paired_removal'


def test_forecast_difference_requires_same_public_query_context_and_inputs(tmp_path):
    root = fixture(tmp_path, right_predictions=[2.])
    changed = registered_successor_effect(root)['cases'][0]['mechanism']['stored_forecast_comparison']
    assert changed['changed'] is True
    assert changed['difference_witness']['right_prediction'] == 2.
    state_path = root / 'case_0/pr3/research_state.json'
    state = json.loads(state_path.read_text())
    state['questions'][0]['context_ref'] = {'source_id': 'other_target'}
    save(state_path, state)
    unknown = registered_successor_effect(root)['cases'][0]['mechanism']['stored_forecast_comparison']
    assert unknown['changed'] is None and unknown['difference_witness'] is None


@pytest.mark.parametrize('different', [False, True])
def test_selected_parents_have_their_own_version_bound_columns_despite_different_model_names(
        tmp_path, different, monkeypatch):
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program

    right = Program('abs', children=(Program('feature', name='u'),)) if different else Program('feature', name='u')
    root = fixture(tmp_path, right_program=right.to_dict())
    for i, arm in enumerate(('fixed_pr2', 'pr3')):
        workspace = root / 'case_0' / arm
        model = f'candidate_{i}.py'
        program = right if i else Program('feature', name='u')
        snapshot = workspace / 'frozen_context'
        snapshot.mkdir()
        (snapshot / 'model.py').write_text(export_temporal_program(program))
        state_path = workspace / 'research_state.json'
        state = json.loads(state_path.read_text())
        state['questions'][0]['query_semantics']['query_expression']['binding']['features']['u'] = -1.
        save(state_path, state)
        save(workspace / 'forecast.json', {'forecasts': [{'model': model, 'returncode': 0,
            'source_context_ref': {'workspace': str(snapshot), 'model': 'model.py'},
            'prediction_inputs': [{'features': {'u': -1.}, 'history': []}],
            'predictions': [1. if different and i else -1.]}]})
    prohibit_execution(monkeypatch)
    result = registered_successor_effect(root)['cases'][0]['mechanism']
    assert result['parent_model_ids_changed']
    assert result['parent_program_changed'] is different
    assert result['stored_forecast_comparison']['changed'] is None  # No same-name column.
    selected = result['selected_parent_forecast_comparison']
    assert selected['status'] == 'recorded' and selected['changed'] is different
    assert selected['left_model'] == 'candidate_0.py' and selected['right_model'] == 'candidate_1.py'
    assert all(ref['json_path'] == ['forecasts', 0] for ref in selected['column_refs'])
    # A saved older parent version cannot stand in for the current one.
    (root / 'case_0/pr3/frozen_context/model.py').write_text(export_temporal_program(Program('const', value=9.)))
    stale = registered_successor_effect(root)['cases'][0]['mechanism']['selected_parent_forecast_comparison']
    assert stale['status'] == 'UNKNOWN' and stale['changed'] is None


def test_failed_unsubmitted_and_missing_results_keep_null_metrics_and_actual_costs(tmp_path):
    root = fixture(tmp_path)
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['trials'][1].update(status='budget_exhausted', terminal={
        'status': 'no_submitted_model', 'mse': None, 'passed': False})
    save(summary_path, summary)
    training_path = root / 'case_0/pr3/training.json'
    training = json.loads(training_path.read_text())
    training.update(model=None, fresh_confirmation=None)
    save(training_path, training)
    row = registered_successor_effect(root)['cases'][0]
    assert row['right']['terminal_mse'] is None and row['right_minus_left']['terminal_mse'] is None
    assert row['right_failure_with_left_success'] is True
    assert row['right']['confirmation_passed'] is None
    assert row['right']['external_terminal_cost']['rows'] == 0
    assert row['right']['costs']['compute_cost'] == 101
    summary['trials'].pop()
    save(summary_path, summary)
    row = registered_successor_effect(root)['cases'][0]
    assert row['right']['status'] == 'missing_registered_result'
    assert row['right']['terminal_mse'] is None
    assert row['right_minus_left']['target_costs']['compute_cost'] is None


def test_no_submission_is_failure_but_missing_terminal_after_submission_is_unknown(tmp_path):
    root = fixture(tmp_path)
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['trials'][1]['terminal'] = {'status': 'no_submitted_model', 'mse': None}
    save(summary_path, summary)
    result = registered_successor_effect(root)['cases'][0]
    assert result['right_failure_with_left_success'] is True
    summary['trials'][1]['terminal'] = {}
    save(summary_path, summary)
    missing = registered_successor_effect(root)['cases'][0]
    assert missing['right']['model_submitted'] is True
    assert missing['right_failure_with_left_success'] is None


def test_pending_queries_compare_paid_actions_in_feedback_not_unexpanded_requests(tmp_path):
    root = fixture(tmp_path)
    for i, arm in enumerate(('fixed_pr2', 'pr3')):
        path = root / 'case_0' / arm / 'training.json'
        training = json.loads(path.read_text())
        record = {'action': {'kind': 'trajectory', 'sequence': [{'u': 1. if i else -1.}]} , 'rows': []}
        training['paid_observations'] = [record]
        training['events'] = [{'action': 'query', 'arguments': {'pending_question': 'task'},
            'feedback': {'observations': [record]} if not i else {'record_indices': [0]}}]
        save(path, training)
    mechanism = registered_successor_effect(root)['cases'][0]['mechanism']
    assert mechanism['requested_query_arguments_changed'] is False
    assert mechanism['executed_research_actions_changed'] is True
    assert mechanism['executed_action_refs'][0][0]['json_path'][-1] == 'action'
    assert mechanism['executed_action_refs'][1][0]['json_path'] == ['paid_observations', 0, 'action']


def test_partial_query_failure_reports_only_successful_paid_prefix_and_unknown_is_not_equal(tmp_path):
    root = fixture(tmp_path)
    paid = {'kind': 'trajectory', 'sequence': [{'u': 1.}]}
    for i, arm in enumerate(('fixed_pr2', 'pr3')):
        path = root / 'case_0' / arm / 'training.json'
        training = json.loads(path.read_text())
        training['events'] = [{'action': 'query', 'arguments': {'actions': [paid, {'kind': 'bad'}] if i else [paid]},
            'feedback': {'observations': [{'action': paid, 'rows': []}], **({'error': 'invalid later action'} if i else {})}}]
        save(path, training)
    mechanism = registered_successor_effect(root)['cases'][0]['mechanism']
    assert mechanism['requested_query_arguments_changed'] is True
    assert mechanism['executed_research_actions_changed'] is False
    assert len(mechanism['executed_action_refs'][1]) == 1
    assert mechanism['partial_or_failed_query_refs'][1]
    path = root / 'case_0/pr3/training.json'
    training = json.loads(path.read_text())
    training['events'][0]['feedback'] = {}
    save(path, training)
    assert registered_successor_effect(root)['cases'][0]['mechanism']['executed_research_actions_changed'] is None


def test_project_C_and_unknown_ledger_accounts_are_not_merged_or_zero_filled(tmp_path):
    root = fixture(tmp_path)
    ledger = {'formal_compute_ledger': {'global_after_interval_excluding_legacy_unknown': [1000, 1100],
        'C_after_interval': [100, 200]}, 'separate_accounts': {
            'old_interrupted': {'actual': None, 'status': 'UNKNOWN', 'upper': 25}}}
    path = root / 'old_ledger.json'
    save(path, ledger)
    result = registered_successor_effect(root, source_cost_ref={'file': str(path), 'json_path': []})
    costs = result['source_costs']
    assert costs['project_development_compute_interval'] == [1000, 1100]
    assert costs['C_attributed_development_compute_interval'] == [100, 200]
    assert costs['C_source_learning_compute_interval'] is None
    assert costs['unresolved_accounts'] == ledger['separate_accounts']
    assert costs['B_source_compute_interval'] is None and costs['source_net_C_minus_B'] is None
    assert not result['source_learning_amortization']['C_development_cost_divided_by_target_count']


@pytest.mark.parametrize('shared', [True, False])
def test_actual_C_removal_shares_source_cost_difference_but_keeps_unknown_absolute_bill(tmp_path, shared):
    keep, removed = fixture(tmp_path / 'keep'), fixture(tmp_path / 'removed')
    source_ref = {'file': str(tmp_path / 'source_cost.json'), 'json_path': []}
    save(tmp_path / 'source_cost.json', {'formal_compute_ledger': {
        'global_after_interval_excluding_legacy_unknown': [1000, 1100], 'C_after_interval': [100, 200]}})
    for root in (keep, removed):
        plan = json.loads((root / 'protocol.json').read_text())
        plan['source_cost_ref'] = source_ref
        save(root / 'protocol.json', plan)
    manifest_ref = {'file': str(tmp_path / 'removal_manifest.json'), 'json_path': []}
    save(tmp_path / 'removal_manifest.json', {'status': 'frozen_ready', 'kind': 'individual_object',
        'source_cost_ref': source_ref if shared else {'file': 'other_source_cost.json', 'json_path': []}})
    summary = json.loads((removed / 'summary.json').read_text())
    summary['trials'] = [row for row in summary['trials'] if row['arm'] == 'pr3']
    summary['trials'][0]['costs']['compute_cost'] = 131
    summary['target_intervention'] = {'kind': 'individual_object', 'manifest_ref': manifest_ref,
        'keep_study': str(keep), 'executed_arms': ['pr3']}
    save(removed / 'summary.json', summary)
    result = registered_successor_effect(keep, removed, left_arm='pr3', right_arm='pr3',
        source_cost_ref=source_ref, intervention_ref=manifest_ref)
    assert result['source_costs']['shared_source_cancels_in_removal_difference'] is shared
    assert result['cases'][0]['right_minus_left']['target_costs']['compute_cost'] == 30
    assert result['source_costs']['C_source_learning_compute_interval'] is None
    assert result['source_costs']['C_attributed_development_compute_interval'] == [100, 200]
    assert result['source_costs']['source_net_C_minus_B'] is None
    assert result['full_source_plus_target_net_cost'].startswith('shared_source_cancels') is shared
    # A normal B/C contrast never cancels independently incurred learning.
    ordinary = registered_successor_effect(keep, source_cost_ref=source_ref)
    assert not ordinary['source_costs']['shared_source_cancels_in_removal_difference']
    assert ordinary['full_source_plus_target_net_cost'] == 'UNKNOWN_missing_comparable_B_source_account'


@pytest.mark.parametrize('removed_arm,kind', [
    ('individual_remove', 'individual_object'), ('class_remove', 'new_redundancy_class')])
@pytest.mark.parametrize('shared', [True, False])
def test_same_scene_registered_removals_cancel_only_their_common_source(
        tmp_path, monkeypatch, removed_arm, kind, shared):
    root = fixture(tmp_path)
    prohibit_execution(monkeypatch)
    source_ref = {'file': str(tmp_path / 'source_cost.json'), 'json_path': []}
    save(tmp_path / 'source_cost.json', {'formal_compute_ledger': {
        'global_after_interval_excluding_legacy_unknown': [1000, 1100], 'C_after_interval': [100, 200]}})
    manifest_refs = {}
    for label, removal_kind in [('individual_remove', 'individual_object'),
                                ('class_remove', 'new_redundancy_class')]:
        path = tmp_path / (label + '.json')
        manifest_refs[label] = {'file': str(path), 'json_path': []}
        save(path, {'status': 'frozen_ready', 'kind': removal_kind,
            'source_cost_ref': source_ref if shared else {'file': 'different_source.json'}})
    plan = json.loads((root / 'protocol.json').read_text())
    plan.update(scope='saved_scene_development_object_interventions', source_cost_ref=source_ref,
        registered_arms=['keep', 'individual_remove', 'class_remove'],
        branch_views={'keep': {'organization_arm': 'pr3'}, **{
            label: {'organization_arm': 'pr3', 'manifest_ref': ref}
            for label, ref in manifest_refs.items()}})
    save(root / 'protocol.json', plan)
    summary = json.loads((root / 'summary.json').read_text())
    keep, removed = summary['trials']
    keep['result_ref'] = {'file': str(root / 'case_0/fixed_pr2/training.json')}
    removed['result_ref'] = {'file': str(root / 'case_0/pr3/training.json')}
    keep['arm'] = 'keep'
    summary['trials'] = [keep, *[{**deepcopy(removed), 'arm': label}
        for label in ('individual_remove', 'class_remove')]]
    save(root / 'summary.json', summary)
    result = registered_successor_effect(root, left_arm='keep', right_arm=removed_arm,
        source_cost_ref=source_ref, intervention_ref=manifest_refs[removed_arm])
    assert result['status'] == 'recorded_comparison'
    assert result['source_costs']['shared_source_cancels_in_removal_difference'] is shared
    assert result['cases'][0]['right_minus_left']['target_costs']['compute_cost'] == 1
    assert result['source_costs']['C_source_learning_compute_interval'] is None
    assert result['source_costs']['source_net_C_minus_B'] is None
    assert result['source_costs']['C_attributed_development_compute_interval'] == [100, 200]
    assert result['full_source_plus_target_net_cost'] == (
        'shared_source_cancels_use_recorded_target_and_external_terminal_differences' if shared
        else 'UNKNOWN_shared_source_removal_not_established')


@pytest.mark.parametrize('different', [False, True])
def test_submitted_parent_terminal_vectors_keep_version_scope_and_external_costs(tmp_path, different, monkeypatch):
    root = fixture(tmp_path)
    plan_path = root / 'protocol.json'
    plan = json.loads(plan_path.read_text())
    plan['evaluation']['rows'] = 2
    save(plan_path, plan)
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text())
    for i, row in enumerate(summary['trials']):
        workspace = root / 'case_0' / row['arm']
        payload = {'model': f'candidate_{i}.py', 'training_ref': {
            'file': str(workspace / 'training.json'), 'json_path': []},
            'evaluation_contract': {'case': 'case_0', 'kind': None, 'evaluation_seed': 11,
                'rows': 2, 'trajectory_length': None},
            'predictions': [0., 2. if i and different else 1.]}
        path = workspace / 'terminal_predictions.json'
        save(path, payload)
        row['terminal'].update(prediction_ref={'file': str(path), 'json_path': []},
                              evaluation_rows=2, python_execution_seconds=.25 + i)
    save(summary_path, summary)
    prohibit_execution(monkeypatch)
    row = registered_successor_effect(root)['cases'][0]
    result = row['selected_parent_terminal_predictions']
    assert result['status'] == 'recorded' and result['changed'] is different
    assert result['rows'] == 2
    assert row['right']['external_terminal_cost'] == {
        'rows': 2, 'python_execution_seconds': 1.25, 'scope': 'outside_agent_training_costs'}
    assert row['right']['costs']['python_execution_seconds'] == 2.
    assert row['right_minus_left']['external_terminal_costs']['python_execution_seconds'] == 1.
    path = root / 'case_0/pr3/terminal_predictions.json'
    payload = json.loads(path.read_text())
    payload['model'] = 'different_parent_version.py'
    save(path, payload)
    unknown = registered_successor_effect(root)['cases'][0]['selected_parent_terminal_predictions']
    assert unknown['status'] == 'UNKNOWN' and unknown['changed'] is None


def test_saved_pair_must_match_registered_cases_budget_and_finish_before_readout(tmp_path):
    first = fixture(tmp_path / 'first')
    second = fixture(tmp_path / 'second')
    plan = json.loads((second / 'protocol.json').read_text())
    plan['limits']['compute_budget'] += 1
    save(second / 'protocol.json', plan)
    assert registered_successor_effect(first, second)['reason'] == 'registered_cases_budgets_or_evaluation_differ'
    summary = json.loads((first / 'summary.json').read_text())
    summary['status'] = 'researching'
    save(first / 'summary.json', summary)
    assert registered_successor_effect(first)['reason'] == 'all_training_must_finish_before_effect_readout'


def test_historical_source_cases_are_not_counted_as_successor_amortization_targets(tmp_path):
    root = fixture(tmp_path)
    plan_path = root / 'protocol.json'
    plan = json.loads(plan_path.read_text())
    plan['research_organization'] = {'phase': 'source'}
    save(plan_path, plan)
    result = registered_successor_effect(root)
    assert result['registered_case_count'] == 1 and result['registered_phase'] == 'source'
    assert result['source_learning_amortization']['actual_target_count'] == 0
    plan['research_organization']['phase'] = 'target'
    save(plan_path, plan)
    assert registered_successor_effect(root)['source_learning_amortization']['actual_target_count'] == 1


def test_existing_MethodEffect_dispatch_uses_the_readonly_successor_reader(tmp_path, monkeypatch):
    root = fixture(tmp_path)
    prohibit_execution(monkeypatch)
    result = evaluate_method_effect({'effect_kind': SUCCESSOR_EFFECT_KIND,
        'left_study': str(root)}, tmp_path)
    assert result['status'] == 'answered' and result['new_method_executions'] == 0
    assert result['value']['paired_terminal_case_count'] == 1


def test_Common_reads_and_withdraws_the_recorded_effect_without_execution(tmp_path, monkeypatch):
    from test_question_continuation import prepared

    root = fixture(tmp_path / 'saved_study')
    agent, lab = prepared(tmp_path / 'common')
    agent.research_mission = {'purpose': 'method_learning', 'mission_id': 'registered_successor_effect'}
    prohibit_execution(monkeypatch)
    before = deepcopy(agent.costs)
    question = {'id': 'successor_effect', 'parent_id': 'task', 'depends_on': [],
        'question': 'What changed in these completed registered successor runs?',
        'kind': 'method_effect', 'answer_type': 'relation', 'use_sites': [],
        'answer_resolution_ref': {'kind': 'registered_case_outcomes_and_cost'},
        'query_semantics': {'mode': 'model_query', 'query_expression': {
            'op': 'MethodEffect', 'effect_kind': SUCCESSOR_EFFECT_KIND, 'left_study': str(root)}}}
    result = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert result['status'] == 'answer_ready_no_parent_use'
    assert result['answer']['value']['paired_terminal_case_count'] == 1
    assert agent.costs == before and not agent.confirmations
    (root / 'summary.json').unlink()
    missing = agent._execute('run', {'operator': 'research_step',
        'research_question': {'id': question['id']}}, lab)
    assert missing['status'] == 'scoped_unknown'
    assert not next(q for q in agent.questions if q['id'] == question['id']).get('semantic_answer')
    assert agent.costs == before


@pytest.mark.parametrize('declared', [
    {'action': 'evaluate', 'arguments': {'model': 'candidate.py'}},
    {'action': 'run', 'arguments': {'operator': 'research_choice', 'models': ['candidate.py']}},
    {'action': 'run', 'arguments': {'operator': 'refit', 'model': 'candidate.py'}},
    {'action': 'run', 'arguments': {'operator': 'native_condition_research', 'max_compute': 50000}},
])
def test_query_program_pure_research_feedback_is_known_non_acquisition(tmp_path, declared):
    root = fixture(tmp_path)
    for arm in ('fixed_pr2', 'pr3'):
        path = root / 'case_0' / arm / 'training.json'
        training = json.loads(path.read_text())
        training['events'] = [{'action': 'query', 'arguments': {'path': 'shared/finite_native_author.py'},
            'feedback': {'program_action': declared, 'status': 'needs_matched_evidence'}}]
        save(path, training)
    mechanism = registered_successor_effect(root)['cases'][0]['mechanism']
    assert mechanism['executed_research_actions_changed'] is False
    assert mechanism['unknown_query_refs'] == [[], []]
    assert all(len(refs) == 1 for refs in mechanism['non_acquisition_query_refs'])
    path = root / 'case_0/pr3/training.json'
    training = json.loads(path.read_text())
    training['events'][0]['feedback'] = {'program_action': {'action': 'run',
        'arguments': {'operator': 'unidentified_research'}}}
    save(path, training)
    unknown = registered_successor_effect(root)['cases'][0]['mechanism']
    assert unknown['executed_research_actions_changed'] is None
    assert len(unknown['unknown_query_refs'][1]) == 1


@pytest.mark.parametrize('changed_source', ['different_literal', 'opaque', 'bound', 'unparseable'])
def test_inherited_parent_closed_export_comparison_does_not_invent_typed_Program(
        tmp_path, monkeypatch, changed_source):
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program

    root = fixture(tmp_path)
    prohibit_execution(monkeypatch)
    literal = export_temporal_program(Program('feature', name='u'))
    for arm in ('fixed_pr2', 'pr3'):
        workspace = root / 'case_0' / arm
        state_path = workspace / 'research_state.json'
        state = json.loads(state_path.read_text())
        state['questions'][0]['native_condition_research'] = []
        state['evaluated_model'] = {'model': '_evaluated_model/retained.py',
            'source_model': 'source_research/old_parent.py'}
        save(state_path, state)
        context = workspace / '_evaluated_model'
        context.mkdir()
        (context / 'retained.py').write_text(literal)
        # The retained snapshot is authoritative even after its source is edited.
        original = workspace / 'source_research'
        original.mkdir()
        (original / 'old_parent.py').write_text('def predict(row): return 99.\n')
    mechanism = registered_successor_effect(root)['cases'][0]['mechanism']
    assert mechanism['parent_program_changed'] is None
    executable = mechanism['selected_parent_executable_comparison']
    assert executable['status'] == 'recorded' and executable['changed'] is False
    assert all(ref['file'].endswith('_evaluated_model/retained.py') for ref in executable['model_refs'])
    context = root / 'case_0/pr3/_evaluated_model'
    if changed_source == 'different_literal':
        (context / 'retained.py').write_text(export_temporal_program(Program('const', value=3.)))
    elif changed_source == 'opaque':
        (context / 'retained.py').write_text('def predict(row): return row["features"]["u"]\n')
    elif changed_source == 'bound':
        save(context / 'model_bindings.json', {'retained.py': {'component': 'other.py'}})
    else:
        from hegel_machine.cognition.temporal_operator_agent import _PROGRAM_RUNNER

        (context / 'retained.py').write_text(
            '# Frozen paid-data predictor; controls and recorded history only.\n_PROGRAM = '
            'not a literal\n' + _PROGRAM_RUNNER)
    changed_result = registered_successor_effect(root)['cases'][0]['mechanism']
    assert changed_result['parent_program_changed'] is None
    executable = changed_result['selected_parent_executable_comparison']
    assert executable['status'] == ('recorded' if changed_source == 'different_literal' else 'UNKNOWN')
    assert executable['changed'] is (True if changed_source == 'different_literal' else None)

"""Finished saved scenes complete predictions once without restarting research."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import method_effect_query
from scripts import prepare_pr3_source_successor as preparation
from test_saved_scene_report_recovery import save, training


def finished_scene(tmp_path):
    root = tmp_path / 'original/run'
    data = training()
    manifest = tmp_path / 'scene_manifest.json'
    save(manifest, {'library_views': {'new_ids': ['new']},
        'source_training_ref': {'file': str(tmp_path / 'old.json'), 'json_path': []},
        'post97_prediction_contract': {'rows': 48}})
    save(root / 'protocol.json', {'source_manifest_ref': {'file': str(manifest), 'json_path': []},
        'original_sunk_costs_once': data['prior_research_cost'],
        'cases': [{'id': 'saved_delay_event19'}], 'limits': data['limits'],
        'evaluation': {'rows': 0}, 'prefix_rows': 97, 'author': {},
        'research_organization': {'phase': 'development'}})
    save(root.parent / 'run_status.json', {'status': 'stopped_failure_preserved', 'error': 'original'})
    for arm in ('fixed_pr2', 'pr3'):
        save(root / arm / 'training.json', data)
        save(root / arm / 'research_state.json', {'questions': [], 'revisions': [],
            'theory_index': {}, 'relation_research': {}, 'exported_candidate_paths': ['candidate.py']})
        save(root / arm / 'scene_acceptance.json', {'new_costs': {'compute_cost': 1}})
    return root, data


def test_complete_uses_retained_costs_once_and_preserves_failed_run(tmp_path, monkeypatch):
    root, data = finished_scene(tmp_path)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*.json')}
    executions = []

    def predictions(output, source_ref):
        path = output / 'post97_predictions.json'
        if path.exists():
            return json.loads(path.read_text())
        executions.append((output, source_ref))
        arms = []
        for arm in ('fixed_pr2', 'pr3'):
            after = deepcopy(data['costs'])
            after['compute_cost'] += 48
            after['python_execution_seconds'] += .25
            arms.append({'arm': arm, 'status': 'recorded', 'costs_before': data['costs'],
                'costs_after': after, 'evaluation': {'mse': .2}})
        receipt = {'status': 'recorded_comparison', 'arms': arms, 'rows': 48,
                   'comparison': {'changed': False, 'right_minus_left_mse': 0.}}
        save(path, receipt)
        return receipt

    def prohibited(*_args, **_kwargs):
        pytest.fail('report completion restarted research')

    monkeypatch.setattr(method_effect_query, 'evaluate_saved_scene_predictions', predictions)
    monkeypatch.setattr(preparation, '_source_research_steps', prohibited)
    monkeypatch.setattr(preparation, 'run_prepared_saved_scene', prohibited)
    first = preparation.complete_saved_scene_reports(root)
    second = preparation.complete_saved_scene_reports(root)
    assert first == second
    assert len(executions) == 1
    assert second['additional_costs']['compute_cost'] == 2 * (7 + 48)
    assert second['additional_costs']['python_execution_seconds'] == 2 * (2. + .25)
    assert json.loads((root / 'completion_receipt.json').read_text())['retained_prediction_receipt_reused']
    pair, = json.loads((root / 'research_effects.json').read_text())['cases']
    assert pair['left']['costs']['compute_cost'] == data['costs']['compute_cost'] + 48
    assert pair['right']['training_only_costs'] == data['costs']
    assert pair['post97_development_predictions']['rows'] == 48
    assert pair['post97_development_predictions']['left_mse'] == .2
    assert pair['post97_development_predictions']['changed'] is False
    assert {path: path.read_bytes() for path in protected} == protected


def test_completion_failure_has_separate_receipt_and_keeps_original(tmp_path, monkeypatch):
    root, _ = finished_scene(tmp_path)
    old_status = (root.parent / 'run_status.json').read_bytes()

    def fail(*_args, **_kwargs):
        raise ValueError('unavailable retained parent')

    monkeypatch.setattr(method_effect_query, 'evaluate_saved_scene_predictions', fail)
    with pytest.raises(ValueError, match='unavailable retained parent'):
        preparation.complete_saved_scene_reports(root)
    assert (root.parent / 'run_status.json').read_bytes() == old_status
    assert json.loads((root / 'completion_receipt.json').read_text())['status'] == 'failed_preserved'


@pytest.mark.parametrize('arms', [
    ('fixed_pr2', 'pr3'), ('keep', 'individual_remove', 'class_remove')])
def test_small_effect_settlement_updates_each_pair_once_without_reopening_training(
        tmp_path, monkeypatch, arms):
    root = tmp_path / 'run'
    before = training()['prior_research_cost']
    trials, prediction_arms, comparisons = [], [], []
    for index, arm in enumerate(arms):
        costs = {**before, 'compute_cost': before['compute_cost'] + 7 * index,
                 'python_execution_seconds': before['python_execution_seconds'] + index}
        after = {**costs, 'compute_cost': costs['compute_cost'] + 48,
                 'python_execution_seconds': costs['python_execution_seconds'] + .25 * (index + 1)}
        trials.append({'case': 'saved_delay_event19', 'arm': arm,
            'training_only_costs': costs, 'costs': after})
        prediction_arms.append({'arm': arm, 'status': 'recorded', 'costs_before': costs,
            'costs_after': after, 'evaluation': {'mse': .2 + .1 * index}})
        if index:
            comparisons.append({'left_arm': arms[0], 'right_arm': arm,
                'changed': index == 2, 'right_minus_left_mse': .1 * index})

    def effect_for(right_index):
        return {'left_arm': arms[0], 'right_arm': arms[right_index],
            'paired_terminal_case_count': 0, 'source_costs': {'absolute_bill': None},
            'cases': [{'case': 'saved_delay_event19', 'left': {
                'costs': trials[0]['training_only_costs'], 'terminal_mse': None},
                'right': {'costs': trials[right_index]['training_only_costs'], 'terminal_mse': None},
                'right_minus_left': {'target_costs': {key: 0 for key in before},
                    'terminal_mse': None, 'external_terminal_costs': {'rows': None}},
                'selected_parent_terminal_predictions': {'status': 'UNKNOWN', 'changed': None},
                'observed_prediction_degradation': None,
                'mechanism': {'parent_program_changed': True, 'selected_experiment_actions_changed': False}}]}

    effects = (effect_for(1) if len(arms) == 2 else {
        arm: effect_for(index) for index, arm in enumerate(arms) if index})
    predictions = {'status': 'recorded_comparison', 'rows': 48, 'arms': prediction_arms,
                   'comparison': comparisons[0]}
    if len(arms) == 3:
        predictions['comparisons'] = comparisons
    save(root / 'summary.json', {'trials': trials})
    save(root / 'post97_predictions.json', predictions)
    save(root / 'research_effects.json', effects)
    protected = {name: (root / name).read_bytes() for name in
                 ('summary.json', 'post97_predictions.json')}

    def prohibited(*_args, **_kwargs):
        pytest.fail('small receipt settlement executed research or reopened its large documents')

    read = preparation._read

    def small_read(path):
        assert path.parent == root
        assert path.name in {'summary.json', 'post97_predictions.json', 'research_effects.json'}
        return read(path)

    monkeypatch.setattr(preparation, '_read', small_read)
    monkeypatch.setattr(preparation, '_source_research_steps', prohibited)
    monkeypatch.setattr(method_effect_query, 'registered_successor_effect', prohibited)
    monkeypatch.setattr(method_effect_query, 'evaluate_saved_scene_predictions', prohibited)
    first = preparation.settle_saved_scene_effects(root)
    first_bytes = (root / 'research_effects.json').read_bytes()
    assert preparation.settle_saved_scene_effects(root) == first
    assert (root / 'research_effects.json').read_bytes() == first_bytes
    assert {name: (root / name).read_bytes() for name in protected} == protected
    actual_effects = [first] if len(arms) == 2 else first.values()
    originals = [effects] if len(arms) == 2 else effects.values()
    for index, (actual, original) in enumerate(zip(actual_effects, originals), 1):
        assert actual['paired_terminal_case_count'] == 0
        assert actual['source_costs'] == original['source_costs']
        pair, = actual['cases']
        old_pair, = original['cases']
        assert pair['mechanism'] == old_pair['mechanism']
        assert pair['selected_parent_terminal_predictions'] == old_pair['selected_parent_terminal_predictions']
        assert pair['right_minus_left']['terminal_mse'] is None
        assert pair['right_minus_left']['external_terminal_costs'] == {'rows': None}
        assert pair['observed_prediction_degradation'] is None
        for side, trial_index in [('left', 0), ('right', index)]:
            assert pair[side]['costs'] == trials[trial_index]['costs']
            assert pair[side]['training_only_costs'] == trials[trial_index]['training_only_costs']
            assert pair[side]['terminal_mse'] is None
        assert pair['right_minus_left']['target_costs'] == {
            key: trials[index]['costs'][key] - trials[0]['costs'][key] for key in before}
        development = pair['post97_development_predictions']
        assert development['status'] == 'recorded' and development['rows'] == 48
        assert development['left_mse'] == .2
        assert development['right_mse'] == pytest.approx(.2 + .1 * index)
        assert development['changed'] is (index == 2)
        assert development['right_minus_left_mse'] == pytest.approx(.1 * index)
        assert development['independent_confirmation'] is False

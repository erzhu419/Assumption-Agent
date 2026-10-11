"""Close point questions on actual final tuning feedback, including episode end."""

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.world_method_transfer import install_method_bundle
from scripts.run_response_matched_adoption import prepare_bundle
from test_world_reference_hypothesis import PATH, prepared, square, submit
from test_world_response_controller import ImmediateCompletionLab, _submit_point_prefix


def _immediate(tmp_path):
    agent, _ = prepared(tmp_path)
    lab = ImmediateCompletionLab()
    agent.lab = lab
    return agent, lab


def _point(agent, number):
    return next(q for q in agent.questions if q['id'] == f'reference_test_{number}_density_square')


def test_last_tuning_action_closes_ordinary_point_without_empty_intermediate_answers(tmp_path, monkeypatch):
    agent, lab = _immediate(tmp_path)
    execute = agent._execute
    evaluations = []

    def recording_execute(action, arguments, laboratory):
        result = execute(action, arguments, laboratory)
        qid = arguments.get('question_id', '')
        if action == 'evaluate' and qid.startswith('reference_test_'):
            evaluations.append(deepcopy(next(q for q in agent.questions if q['id'] == qid)))
        return result

    monkeypatch.setattr(agent, '_execute', recording_execute)
    result = submit(agent, lab, square('density', 'density_square'))

    assert result['status'] == 'completed' and lab.success
    assert lab.frequencies[5] == 350
    assert lab.actions[-1] == {'chosen_dialog_option_int': 8}  # Final -10 Hz, no Exit.
    assert [q['id'] for q in evaluations] == [f'reference_test_{n}_density_square' for n in (4, 5)]
    assert all(q['status'] == 'supported' and q['evidence'] for q in evaluations)
    last = _point(agent, 5)
    assert last['answer']['actual_frequency'] == 350 and last['answer']['activated']
    assert last['evidence'] == [agent.observations[-1]['index']]
    assert agent.costs['model_calls'] == 0


def test_activation_before_requested_point_does_not_confirm_that_point(tmp_path):
    agent, lab = _immediate(tmp_path)
    lab.required[5] = 360

    result = submit(agent, lab, square('density', 'density_square'))

    assert result['status'] == 'completed' and lab.success
    assert lab.frequencies[5] == 360
    point = _point(agent, 5)
    assert point['status'] == 'open' and point['evidence'] == []
    state = json.loads((agent.workspace / recipe.STATE).read_text())
    assert state['test']['prediction'] == pytest.approx(350)
    final_proposal = agent.method_results[-1]['proposal']
    assert 'after_observation' not in final_proposal
    assert agent.costs['model_calls'] == 0


def test_final_tuning_to_inactive_requested_point_records_refutation(tmp_path):
    agent, lab = _immediate(tmp_path)
    lab.required[5] = 351

    result = _submit_point_prefix(agent, lab, square('density', 'density_square'))

    assert result['status'] != 'completed' and not lab.success
    point = _point(agent, 5)
    assert point['status'] == 'refuted'
    assert point['answer']['prediction'] == pytest.approx(350)
    assert point['answer']['actual_frequency'] == 350 and not point['answer']['activated']
    assert point['evidence']
    record = agent.observations[point['evidence'][-1]]
    assert record['action'] == {'chosen_dialog_option_int': 8}
    assert agent.costs['model_calls'] == 0


def _view(frequency, steps):
    return {'dialog_box': {
        'is_in_dialog': True,
        'dialogIn': f'Reactor #5. The current resonance frequence is: {frequency} Hertz.',
        'dialogOptions': {'1': 'Exit', **{str(index):
            f'{"Increase" if step > 0 else "Decrease"} frequency by {abs(step)} Hz'
            for index, step in enumerate(steps, 2)}}}}


ANALYSIS = [{'action': 'evaluate', 'arguments': {'path': PATH, 'question_id': 'point'}}]


def test_half_hz_overshoot_keeps_reverse_adjustment_before_analysis():
    first = recipe._tune(_view(0, [10, 1, -1]), 9.495, after_observation=ANALYSIS)
    assert first == {'actions': [{'chosen_dialog_option_int': 2}]}
    final = recipe._tune(_view(10, [10, 1, -1]), 9.495, after_observation=ANALYSIS)
    assert final == {'actions': [{'chosen_dialog_option_int': 4}], 'after_observation': ANALYSIS}
    assert recipe._tune(_view(9, [10, 1, -1]), 9.495, after_observation=ANALYSIS) is None


@pytest.mark.parametrize('prediction, final_analysis', [(11.4, True), (11.6, False)])
def test_final_analysis_uses_public_step_sizes_and_reachable_tolerance(prediction, final_analysis):
    first = recipe._tune(_view(0, [8, 3, -3]), prediction, after_observation=ANALYSIS)
    assert first == {'actions': [{'chosen_dialog_option_int': 2}]}
    final = recipe._tune(_view(8, [8, 3, -3]), prediction, after_observation=ANALYSIS)
    assert final['actions'] == [{'chosen_dialog_option_int': 3}]
    assert ('after_observation' in final) is final_analysis
    assert recipe._tune(_view(8, [8, 3, -3]), prediction) == {'actions': [{'chosen_dialog_option_int': 3}]}


def test_archived_bundle_installs_finalization_without_replacing_scientific_declarations(tmp_path):
    archive = Path(__file__).resolve().parents[1] / (
        'artifacts/response_exhausted_stage_20261011/after/retained/sources/'
        'cubic_original_1__gain_lower/numeric_bundle.json')
    bundle = json.loads(archive.read_text())
    original = deepcopy(bundle)
    refreshed = prepare_bundle(bundle)
    assert bundle == original
    assert prepare_bundle(refreshed) == refreshed
    assert {k: v for k, v in refreshed.items() if k != 'files'} == {
        k: v for k, v in original.items() if k != 'files'}
    assert {k: v for k, v in refreshed['files'].items() if k != PATH} == {
        k: v for k, v in original['files'].items() if k != PATH}
    trees = [ast.parse(item['files'][PATH]) for item in (original, refreshed)]
    definitions = [{node.name: ast.dump(node) for node in tree.body
                    if isinstance(node, ast.FunctionDef)} for tree in trees]
    assert {name for name in definitions[0] if definitions[0][name] != definitions[1][name]} == {
        '_tune', 'select_actions', 'answer'}
    assert [ast.dump(node) for node in trees[0].body if not isinstance(node, ast.FunctionDef)] == [
        ast.dump(node) for node in trees[1].body if not isinstance(node, ast.FunctionDef)]

    agent, lab = prepared(tmp_path)
    installed = install_method_bundle(agent, refreshed, lab)
    assert installed['methods'] == [method['id'] for method in refreshed['methods']]
    assert agent.observations == [] and lab.actions == []
    namespace = {}
    exec(compile((agent.workspace / PATH).read_text(), PATH, 'exec'), namespace)
    assert namespace['_tune'](_view(8, [8, 3, -3]), 11.4, after_observation=ANALYSIS) == {
        'actions': [{'chosen_dialog_option_int': 3}], 'after_observation': ANALYSIS}

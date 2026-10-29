"""Restoring a complete scene must retain its already paid scoring scopes."""

from copy import deepcopy
import json

from scripts import prepare_pr3_source_successor as preparation
from test_python_paid_score_reuse import scene, choose, no_execution, MODELS


INDEX = '_paid_python_scores/_evaluated_model/assessments.json'


def test_exact_145_row_scores_survive_restore_and_skip_the_existing_evaluator(tmp_path, monkeypatch):
    source, lab = scene(tmp_path / 'source')
    source.rows = source.rows * 72 + source.rows[:1]
    source._write_evidence()
    paid = choose(source, lab)
    assert paid['new_scored_count'] == 2 and len(source.rows) == 145
    original_index = (source.workspace / INDEX).read_bytes()
    restored, restored_lab = scene(tmp_path / 'restored')
    restored.rows = deepcopy(source.rows)
    restored._write_evidence()
    # A genuine 97-row prefix still cannot receive later measurements.
    prefix = preparation._retain_scene_score_index(source.workspace, restored.workspace, paid_rows=97)
    assert prefix['retained_assessments'] == 0
    receipt = preparation._retain_scene_score_index(source.workspace, restored.workspace,
        paid_rows=len(restored.rows))
    assert receipt['available_paid_rows'] == 145 and receipt['retained_assessments'] == 2
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    before = deepcopy(restored.costs)
    reused = choose(restored, restored_lab)
    assert reused['models_requested'] == reused['models_completed'] == len(MODELS)
    assert reused['new_scored_count'] == 0 and reused['reused_score_count'] == 2
    assert [item['mse'] for item in reused['evaluations']] == [item['mse'] for item in paid['evaluations']]
    assert reused['evaluated_model']['source_model'] == paid['evaluated_model']['source_model']
    assert reused.get('next_action') == paid.get('next_action')
    assert restored.costs['compute_cost'] == before['compute_cost']
    assert (source.workspace / INDEX).read_bytes() == original_index


def test_restored_scope_keeps_all_paid_rows_but_not_a_future_scope(tmp_path):
    source = tmp_path / 'source'
    (source / INDEX).parent.mkdir(parents=True)
    scopes, assessments = [], []
    for rows in (72, 97, 145, 169):
        scope = {'file': str(source / f'scope_{rows}.json'), 'json_path': []}
        assessment = {'file': str(source / f'assessment_{rows}.json'), 'json_path': []}
        preparation._write(scope['file'], {'rows': [None] * rows})
        preparation._write(assessment['file'], {'scope_ref': scope,
            'result': {'mse': 1., 'costs': {'compute_cost': rows}}})
        scopes.append(scope)
        assessments.append(assessment)
    preparation._write(source / INDEX, {'scopes': scopes, 'assessments': assessments, 'contexts': []})
    receipt = preparation._retain_scene_score_index(source, tmp_path / 'restored', paid_rows=145)
    retained = json.loads((tmp_path / 'restored' / INDEX).read_text())
    assert retained['scopes'] == scopes[:3] and retained['assessments'] == assessments[:3]
    assert receipt['retained_scopes'] == receipt['retained_assessments'] == 3

"""An already purchased suffix must not rerun its immutable predictor."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.registered_source_scene import admit_exposed_representation_evidence
from test_exposed_representation_material import paid_scene


def material_with_paid_predictions(tmp_path):
    current, material = paid_scene(tmp_path)
    agent = current['agent']
    material['parent_assessment_ref'] = deepcopy(agent.operator_results[-1]['evaluations'][0][
        'paid_score_reuse']['execution_basis_ref'])
    path = tmp_path / 'frozen_evaluation.json'
    evaluated = json.loads(path.read_text())
    evaluated['evaluation'].update(predictions=[0.] * 4, risk_quantile=.5, loss_kind='mse',
        model_dependencies=[], research_bindings=[], research_prediction_count=0)
    evaluated['evaluation']['representation_diagnostic']['representation_contract'] = 'actual_downstream_inputs'
    path.write_text(json.dumps(evaluated))
    return current, material


def test_union_of_paid_predictions_enters_existing_score_cache_with_no_model_call(tmp_path, monkeypatch):
    current, material = material_with_paid_predictions(tmp_path)
    agent = current['agent']
    before = deepcopy(agent.costs)
    receipt = admit_exposed_representation_evidence(current, material)
    reused = receipt['paid_readout_reuse']
    assert reused['reused_readout_rows'] == 4 and reused['model_executions'] == 0
    assert agent.costs == before

    def no_python(*args, **kwargs):
        pytest.fail('the already paid frozen predictor was executed again')

    monkeypatch.setattr(agent, '_python', no_python)
    owner = agent.knowledge['finite_native_author']['question']['id']
    result = agent._execute('run', {'operator': 'research_choice', 'models': ['old.py'],
        'research_question': {'id': owner}}, current['lab'])
    assert result['models_completed'] == 1 and result['reused_score_count'] == 1
    assert result['new_scored_count'] == 0
    assert {key: value for key, value in agent.costs.items() if key != 'python_execution_seconds'} == {
        key: value for key, value in before.items() if key != 'python_execution_seconds'}
    assert result['evaluations'][0]['paid_score_reuse']['additional_scoring_python_seconds'] == 0
    saved = json.loads(open(reused['assessment_ref']['file']).read())
    assert saved['result']['mse'] == sum(row.target ** 2 for row in agent.rows) / len(agent.rows)
    assert saved['result']['evaluation_rows'] == len(agent.rows)
    assert saved['result']['representation_diagnostic']['rows'] == 4
    assert 'closed_prediction_state' not in saved
    assert saved['result']['paid_prediction_parts'][-1]['file'] == material['evaluation_ref']['file']


def test_readout_reuse_does_not_accept_a_changed_prefix_model(tmp_path):
    current, material = material_with_paid_predictions(tmp_path)
    prefix = json.loads(open(material['parent_assessment_ref']['file']).read())
    context = prefix['execution_context']
    from pathlib import Path
    # This is an isolated fixture context, never a historical model directory.
    changed = tmp_path / 'wrong_context'
    changed.mkdir()
    (changed / 'wrong.py').write_text('_PROGRAM = ("const", 100, None, 0, ())\n')
    prefix['execution_context'] = {'workspace': str(changed), 'model': 'wrong.py'}
    bad = tmp_path / 'different_assessment.json'
    bad.write_text(json.dumps(prefix))
    material['parent_assessment_ref'] = {'file': str(bad), 'json_path': []}
    with pytest.raises(ValueError, match='same successfully executed frozen model'):
        admit_exposed_representation_evidence(current, material)

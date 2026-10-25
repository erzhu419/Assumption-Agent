"""Saved fitting provenance supports shared external model execution roots."""

from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition import online_inquiry_llm
from hegel_machine.cognition.models import Observation


@pytest.mark.parametrize('external', [False, True])
@pytest.mark.parametrize('training_name', ['model_observations.json', 'observations.json'])
def test_saved_training_scope_keeps_local_relative_and_external_absolute_paths(
        tmp_path, monkeypatch, external, training_name):
    workspace = tmp_path / 'current'
    workspace.mkdir()
    context = (tmp_path / 'shared' if external else workspace) / '_evaluated_model'
    context.mkdir(parents=True)
    model = context / 'model.py'
    model.write_text('raise AssertionError("scope metadata must not run the model")\n')
    rows = [Observation({'u': float(i)}, float(i), ()) for i in (1, 2, 3)]
    training = context / training_name
    training.write_text(json.dumps([asdict(row) for row in rows[:2]]))
    monkeypatch.setattr(online_inquiry_llm, 'evaluate_python_model',
                        lambda *a, **kw: pytest.fail('metadata must not fit or score'))
    scope = online_inquiry_llm._paid_prediction_scope(
        workspace, str(model) if external else '_evaluated_model/model.py', rows, [1., 2., 4.])
    expected = str(training if external else training.relative_to(workspace))
    assert scope['fitting_view'] == {'scope': 'current_paid_prefix',
        'file': expected, 'rows': 2, 'response_row_range': [0, 1]}
    assert scope['evaluation_scope'] == 'paid_prediction'
    assert scope['later_readings'] == {'rows': 1, 'response_row_range': [2, 2],
                                     'mse': 1., 'max_abs_error': 1.}

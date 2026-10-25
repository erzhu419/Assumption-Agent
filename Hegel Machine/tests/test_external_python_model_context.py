"""External shared drafts retain their own execution root and score identity."""

import json
from pathlib import Path

from hegel_machine.cognition.code_agent import evaluate_python_model, python_model_context
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import (
    _matching_literal_score, _reused_python_score, export_temporal_program,
)


def test_external_shared_literal_executes_97_rows_and_reuses_named_score(tmp_path):
    workspace, shared = tmp_path / 'current', tmp_path / 'common_models'
    workspace.mkdir()
    shared.mkdir()
    model, renamed = shared / 'program_000.py', shared / 'program_001.py'
    source = export_temporal_program(Program('feature', name='u'))
    model.write_text(source)
    renamed.write_text(source)
    rows = [Observation({'u': float(i)}, float(i), ()) for i in range(97)]
    assert python_model_context(workspace, str(model)) == (shared, model.name)
    scored = evaluate_python_model(workspace, str(model), rows)
    assert scored['returncode'] == 0, scored['stderr']
    assert scored['mse'] == 0. and len(scored['predictions']) == 97
    basis = {'rows': 'the same 97 retained rows'}
    reference = {'file': 'saved_assessment.json', 'json_path': []}
    assessment = {'execution_context': {'workspace': str(shared), 'model': model.name},
        'scope_ref': {'file': 'same_scope.json'}, 'model': str(model), 'result': scored,
        'execution_costs': {'compute_cost': 97, 'python_execution_seconds': .1}}
    comparison = {'basis': basis, 'scopes': {'same_scope.json': basis},
                  'assessments': {str(model): [(reference, assessment)]}}
    found = _matching_literal_score(workspace, str(renamed), comparison)
    assert found == (reference, assessment)
    reused = _reused_python_score(str(renamed), *found)
    assert reused['predictions'] == scored['predictions']
    assert reused['paid_score_reuse']['additional_scoring_compute'] == 0
    assert reused['paid_score_reuse']['match_basis'] == 'identical_closed_literal_execution'


def test_local_absolute_draft_keeps_workspace_root_fitting_view(tmp_path):
    workspace = tmp_path / 'current'
    model = workspace / 'drafts' / 'model.py'
    model.parent.mkdir(parents=True)
    model.write_text('_value = None\ndef fit(rows):\n global _value\n'
                     ' _value = rows[0]["target"]\ndef predict(row):\n return _value\n')
    training = [{'features': {'u': 0.}, 'target': 7., 'history': []}]
    (workspace / 'model_observations.json').write_text(json.dumps(training))
    (model.parent / 'model_observations.json').write_text(json.dumps(
        [{'features': {'u': 0.}, 'target': 99., 'history': []}]))
    assert python_model_context(workspace, str(model)) == (workspace, 'drafts/model.py')
    result = evaluate_python_model(workspace, str(model), [Observation({'u': 0.}, 7., ())])
    assert result['returncode'] == 0, result['stderr']
    assert result['predictions'] == [7.] and result['mse'] == 0.


def test_absolute_evaluated_snapshot_still_uses_its_last_snapshot_root(tmp_path):
    workspace = tmp_path / 'current'
    model = tmp_path / 'source' / '_evaluated_model' / 'nested' / '_evaluated_model' / 'model.py'
    assert python_model_context(workspace, str(model)) == (model.parent, 'model.py')

"""Closed paid Programs expose the same inputs their predictor actually uses."""
from copy import deepcopy
import json
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.code_agent import (
    _closed_temporal_export, closed_temporal_prediction_contract, evaluate_python_model,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_operator_agent import (
    _PROGRAM_REPRESENTATION_RUNNER, closed_temporal_representation_diagnostic,
    export_temporal_program,
)
from test_literal_program_score_reuse import MODEL, scene
from test_python_paid_score_reuse import choose, no_execution


def native(program):
    namespace = {}
    exec(export_temporal_program(program), namespace)
    original = namespace['predict']
    exec(_PROGRAM_REPRESENTATION_RUNNER, namespace)
    return namespace, original


@pytest.mark.parametrize('kind', ['constant', 'lag', 'accumulate', 'nested_window', 'event', 'unused_temporal'])
def test_decomposition_preserves_prediction_and_temporal_scope(kind):
    u = Program('feature', name='u')
    one = Program('const', value=1.)
    if kind == 'constant':
        program = one
    elif kind == 'lag':
        program = Program('if', children=(Program('available', lag=2),
            Program('lag', name='u', lag=2), one))
    elif kind == 'accumulate':
        program = Program('accumulate', value=.63, children=(u, one))
    elif kind == 'nested_window':
        inner = Program('window_sum', lag=3, children=(u,))
        program = Program('window_sum', lag=4, children=(inner,))
    elif kind == 'event':
        program = Program('event_delta', lag=2, children=(
            Program('accumulate', value=.63, children=(u, one)),))
    else:
        unused = Program('accumulate', value=2.,
            children=(u, Program('const', value=1e308)))
        program = Program('if', children=(Program('const', value=0.), unused, one))
    model, original = native(program)
    rng, rows = Random(17), []
    for length in (0, 1, 2, 7, 48):
        rows.append({'features': {'u': rng.random()},
                     'history': [{'u': rng.random()} for _ in range(length)]})
    for row in rows + list(reversed(rows)):
        represented = json.loads(json.dumps(model['represent'](row)))
        assert model['predict_features'](represented) == original(row)
        # The downstream function receives no row or history to read again.
        changed = deepcopy(row)
        changed['features']['u'] = -999.
        changed['history'] = []
        assert model['predict_features'](represented) == original(row)
    if kind == 'constant':
        assert model['represent'](rows[0]) == []
    if kind == 'unused_temporal':
        assert model['represent'](rows[-1]) == [[0, 0.]]
    if kind == 'accumulate':
        assert len(model['_representation_terms']) == 1
        assert model['represent'](rows[0]) != model['represent'](rows[-1])


def test_short_window_does_not_expose_older_controls_or_whole_history():
    program = Program('window_sum', lag=2, children=(Program('feature', name='u'),))
    model, _ = native(program)
    a = {'features': {'u': .4}, 'history': [{'u': -3.}, {'u': .2}]}
    b = {'features': {'u': .4}, 'history': [{'u': 9.}, {'u': 8.}, {'u': .2}]}
    assert model['represent'](a) == model['represent'](b)
    assert len(model['represent'](a)) == 1


@pytest.mark.parametrize('arm,filename', [
    ('remove', 'remove_candidate_0016_v0001.py'),
    ('keep', 'keep_candidate_0133_v0001.py'),
])
def test_saved_models_reproduce_predictions_and_actual_view_floor(tmp_path, arm, filename):
    evidence = Path(__file__).resolve().parents[1] / 'docs/evidence/frozen_parent_terminal_259_455_20261025'
    source = (evidence / 'models' / filename).read_bytes()
    (tmp_path / 'model.py').write_bytes(source)
    rows = json.loads((evidence / 'terminal_rows.json').read_text())
    saved = json.loads((evidence / f'predictions_{arm}.json').read_text())['predictions']
    result = evaluate_python_model(tmp_path, 'model.py', rows)
    assert result['returncode'] == 0, result['stderr']
    assert result['predictions'] == saved
    diagnostic = result['representation_diagnostic']
    assert diagnostic['representation_contract'] == 'actual_downstream_inputs'
    assert diagnostic['conflicting_groups'] == 6
    assert diagnostic['empirical_mse_floor'] == .0068359375
    review = closed_temporal_representation_diagnostic(tmp_path, 'model.py', rows)
    assert review['diagnostic'] == diagnostic
    assert review['coordinate_evaluations'] == 2048
    assert review['prediction_evaluations'] == review['fitting_calls'] == 0
    assert _closed_temporal_export(tmp_path, 'model.py')
    assert (tmp_path / 'model.py').read_bytes() == source


@pytest.mark.parametrize('funded', [True, False])
def test_legacy_score_is_reused_review_is_charged_once_and_never_predicts(tmp_path, monkeypatch, funded):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    index_path = agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json'
    index = json.loads(index_path.read_text())
    scope_path = Path(index['scopes'][0]['file'])
    scope = json.loads(scope_path.read_text())
    scope['evaluation_contract'] = closed_temporal_prediction_contract(scope['evaluation_contract'])
    scope_path.write_text(json.dumps(scope))
    assessment_path = Path(index['assessments'][0]['file'])
    assessment = json.loads(assessment_path.read_text())
    assessment['result']['representation_diagnostic'] = {'status': 'not_exposed'}
    assessment_path.write_text(json.dumps(assessment))
    before = agent.costs['compute_cost']
    if not funded:
        object.__setattr__(agent.config, 'compute_budget', before)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    reviewed = choose(agent, lab, [MODEL])
    assert reviewed['new_scored_count'] == 0 and reviewed['reused_score_count'] == 1
    review = reviewed['evaluations'][0]['representation_review']
    if funded:
        assert review['status'] == 'computed_from_retained_controls'
        assert review['costs']['compute_cost'] == len(agent.rows)
        assert review['costs']['python_execution_seconds'] > 0
        assert review['prediction_evaluations'] == review['fitting_calls'] == 0
        assert agent.costs['compute_cost'] == before + len(agent.rows)
        repeated = choose(agent, lab, [MODEL])
        assert repeated['evaluations'][0]['representation_review']['status'] == 'reused'
        assert agent.costs['compute_cost'] == before + len(agent.rows)
    else:
        assert review['status'] == 'unresolved_budget'
        assert agent.costs['compute_cost'] == before
    assert calls == [MODEL]
    assert json.loads(assessment_path.read_text()) == assessment


def test_other_executor_changes_do_not_get_closed_contract_equivalence():
    from hegel_machine.cognition.temporal_operator_agent import _same_closed_prediction_scope
    base = {'rows': [], 'evaluation_contract': {'runner_source': 'same runtime',
        'evaluator_source': 'same evaluator', 'python_version': 'original'}}
    changed = deepcopy(base)
    changed['evaluation_contract']['evaluator_source'] += '\nnew prediction rule'
    assert not _same_closed_prediction_scope(base, changed)

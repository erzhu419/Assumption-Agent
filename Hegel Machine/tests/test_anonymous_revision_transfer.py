"""Source failures and unknown usage must survive genuine offline transfer."""

from dataclasses import asdict
import json
from random import Random

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from scripts.evaluate_anonymous_revision_transfer import run_transfer


MODEL = '''import json
import numpy as np

def fit(rows):
    global coefficients
    design = np.asarray([[1., row['features']['u']] for row in rows])
    coefficients = np.linalg.lstsq(design, [row['target'] for row in rows], rcond=None)[0]
    with open('actual_fit_rows.json', 'w') as stream:
        json.dump(rows, stream)

def predict(row):
    return float(coefficients @ [1., row['features']['u']])
'''


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_new_target_refit_preserves_sources_and_unknown_failed_usage(tmp_path):
    source_dir, output = tmp_path / 'source', tmp_path / 'transfer'
    cases = [dict(id='case_000', kind='memoryless', seed=113),
             dict(id='case_001', kind='delay', seed=149)]
    write_json(source_dir / 'protocol.json', {
        'cases': cases, 'arms': ['hegel'], 'limits': {'execution_seconds': 30.0}})

    workspace = source_dir / 'case_000' / 'hegel'
    retained = workspace / '_evaluated_model'
    lab = AnonymousHistoryLab('memoryless', seed=113)
    rows = []
    for action in lab.confirmation_actions(72, Random(193)):
        rows.extend(lab.observation_rows(lab.execute(action)))
    source_rows = json.loads(json.dumps([asdict(row) for row in rows]))
    write_json(retained / 'model_observations.json', source_rows)
    write_json(retained / 'observations.json', source_rows)
    (retained / 'model.py').write_text(MODEL)
    # The outer workspace has a different fitting view; the submitted snapshot
    # must resolve to its own original view, never these later source rows.
    write_json(workspace / 'model_observations.json', [{'unrelated': True}])
    write_json(workspace / 'training.json', {
        'workspace': str(workspace), 'status': 'completed',
        'model': '_evaluated_model/model.py', 'fresh_confirmation': {'passed': True},
        'costs': {'model_calls': 2, 'reported_tokens': 100, 'observation_cost': 72}})
    failed = source_dir / 'case_001' / 'hegel'
    write_json(failed / 'training.json', {
        'workspace': str(failed), 'status': 'provider_failed', 'model': None,
        'fresh_confirmation': None,
        'costs': {'model_calls': 3, 'reported_tokens': None, 'observation_cost': 72}})
    original = {str(path.relative_to(source_dir)): path.read_bytes()
                for path in source_dir.rglob('*') if path.is_file()}

    summary = run_transfer(source_dir, output)

    assert json.loads((output / 'summary.json').read_text()) == summary
    assert summary['status'] == 'completed'
    assert summary['registered_sources'] == 2
    assert summary['model_calls'] == 0
    assert summary['source_learning_costs_once'] == {
        'model_calls': 5, 'reported_tokens': None, 'observation_cost': 144}
    assert all(group['registered_sources'] == 2 and group['unavailable'] == 1
               and group['evaluated'] == 1 for group in summary['groups'])
    assert all(group['costs']['model_calls'] == 0 for group in summary['groups'])

    outcomes = json.loads((output / 'outcomes.json').read_text())
    assert outcomes[1]['source_learning_costs']['reported_tokens'] is None
    assert all(result['reason'] == 'no_submitted_source_model'
               for result in outcomes[1]['conditions'])
    frozen, refitted = outcomes[0]['conditions']
    assert frozen['source_fitting_rows'] == refitted['source_fitting_rows'] == 72
    assert frozen['status'] == refitted['status'] == 'evaluated'
    assert not frozen['passed'] and refitted['passed']
    assert refitted['mse'] < 1e-20 < frozen['mse']
    assert frozen['source_code_unchanged'] and refitted['source_code_unchanged']
    assert frozen['costs']['observation_cost'] == 0
    assert refitted['costs']['observation_cost'] == 72
    assert frozen['costs']['terminal_observations'] == refitted['costs']['terminal_observations'] == 512

    for result in (frozen, refitted):
        copied = output / 'case_000' / 'hegel' / result['condition']
        assert (copied / 'model.py').read_text() == MODEL
        fit_rows = json.loads((copied / 'actual_fit_rows.json').read_text())
        assert len(fit_rows) == 72
        assert (fit_rows == source_rows) == (result['condition'] == 'source_context_frozen')
        assert json.loads((copied / 'model_observations.json').read_text()) == fit_rows
    assert original == {str(path.relative_to(source_dir)): path.read_bytes()
                        for path in source_dir.rglob('*') if path.is_file()}

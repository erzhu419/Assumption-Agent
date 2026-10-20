"""Authored fit programs enter the existing paid-data candidate loop."""

from copy import deepcopy
import json

import pytest

from test_common_archive_online_research import NEW_MODEL, prepared
from test_inquiry_revision import PublicLab, action


FITTING_MODEL = """import json
from pathlib import Path
def fit(rows):
    global slope
    assert json.load(open('observations.json')) == rows
    assert json.load(open('model_observations.json')) == rows
    assert [r for batch in json.load(open('laboratory_observations.json'))
            for r in batch['rows']] == rows
    Path('fit_rows.json').write_text(json.dumps(rows))
    slope = sum(r['features']['u'] * r['target'] for r in rows) / sum(
        r['features']['u'] ** 2 for r in rows)
def predict(row): return slope * row['features']['u']
"""


def paid_scene(path, arm):
    agent, lab = prepared(path, arm), PublicLab()
    agent._execute('query', {'actions': [action(-2., -1.), action(0., 1.),
                                       action(2., 3.)]}, lab)
    return agent, lab


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_inline_refit_validates_exports_and_compares_the_authored_program(tmp_path, arm):
    agent, lab = paid_scene(tmp_path, arm)
    arguments = {'operator': 'refit', 'model': 'shared/hypothesis.py',
                 'code': FITTING_MODEL, 'path': 'shared/refitted',
                 'validation_records': [2]}
    original, before = deepcopy(arguments), deepcopy(agent.costs)

    result = agent._execute('run', arguments, lab)

    assert arguments == original
    assert result['written'] == result['source_model'] == 'shared/hypothesis.py'
    assert (tmp_path / result['source_model']).read_text() == FITTING_MODEL
    assert result['status'] == 'refit_supported_on_validation'
    assert result['validation']['mse'] < 1e-20
    assert result['split']['training_records'] == [0, 1]
    assert result['split']['validation_records'] == [2]
    assert result['split']['train_rows'] == 4 and result['split']['holdout_rows'] == 2
    assert agent.costs['compute_cost'] - before['compute_cost'] == 6
    assert agent.costs['observation_cost'] == before['observation_cost'] == lab.paid == 6
    assert result['path'] in agent.candidate_paths
    context = tmp_path / 'shared/refitted/_evaluated_model'
    training = json.loads((context / 'fit_rows.json').read_text())
    expected = [row for record in agent.observations[:2] for row in record['rows']]
    assert training == json.loads(json.dumps(expected))
    assert len(json.loads((context / 'model_observations.json').read_text())) == 6
    assert not (tmp_path / 'fit_rows.json').exists()
    current = agent.research_candidates['shared/hypothesis.py']['current_evaluation']
    assert current['evaluation_scope'] == 'paid_trajectory_holdout'
    assert current['mse'] == result['validation']['mse']
    assert agent.model is None and not agent.confirmations

    comparison = agent._execute('run', {'operator': 'compare_experiments',
        'models': [result['path'], 'new.py'],
        'experiments': [{'id': 'authored_probe', 'actions': [action(4., 5.)]}]}, lab)['comparison']
    assert comparison['status'] == 'compared'
    assert comparison['eligible_models'] == [result['path'], 'new.py']
    assert comparison['diagnosis'] == 'observational_ambiguity'
    assert comparison['forecasts'][0]['predictions'] == pytest.approx([4., 5.])
    assert agent.costs['observation_cost'] == lab.paid == 6
    assert agent.model is None and not agent.confirmations


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_inline_refit_failure_retains_the_supported_old_candidate(tmp_path, arm):
    agent, lab = paid_scene(tmp_path, arm)
    incumbent = agent._execute('evaluate', {'model': 'new.py'}, lab)['evaluated_model']['model']
    source = "def fit(rows): raise ValueError('authored hypothesis failed')\n" + NEW_MODEL
    before = deepcopy(agent.costs)

    result = agent._execute('run', {'operator': 'refit', 'model': 'shared/revision.py',
        'code': source, 'path': 'shared/failed_fit', 'validation_records': [2]}, lab)

    assert result['status'] == 'refit_execution_failed' and result['path'] is None
    assert 'authored hypothesis failed' in result['validation']['stderr']
    assert (tmp_path / 'shared/revision.py').read_text() == source
    assert agent.research_candidates['shared/revision.py']['status'] == 'execution_failed'
    assert agent._evaluated_summary()['model'] == incumbent
    assert agent._score(incumbent)['mse'] == 0
    assert agent.costs['observation_cost'] == before['observation_cost'] == lab.paid == 6
    assert agent.model is None and not agent.confirmations

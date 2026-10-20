"""Published source predictors keep their executable context across export."""

import json

import pytest

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry import InquiryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


@pytest.mark.parametrize('advertisement', ['research_roles', 'source_catalog',
                                         'nested_source_catalog', 'failed_source_catalog'])
def test_terminal_archive_retains_advertised_source_predictor_context(tmp_path, advertisement):
    directory = 'archive/source_001/_evaluated_model'
    prior = directory + '/model.py'
    knowledge = {'files': {
        prior: '''import json
from pathlib import Path
gain = json.loads(Path('settings.json').read_text())['gain']
def predict(row):
    return gain * row['features']['u']
''',
        directory + '/settings.json': '{"gain": 7}',
        directory + '/model_observations.json': '[]',
    }}
    field = 'research_roles' if advertisement == 'research_roles' else 'source_catalog'
    if advertisement == 'research_roles':
        knowledge[field] = {'predictors': [prior], 'predictor': prior}
    else:
        knowledge[field] = [{'id': 'source_001', 'directory': directory,
                                    'model': prior, 'status': 'completed'}]
    if advertisement == 'failed_source_catalog':
        knowledge[field][0].update(model=None, status='confirmation_budget_exhausted')
        knowledge['files'][directory + '/analysis.py'] = 'hypothesis = "unconfirmed source"\n'
        knowledge['files'][directory + '/draft_notes.md'] = 'Confirmation budget exhausted.\n'
    evaluated_prior, target = prior, 14.
    if advertisement == 'nested_source_catalog':
        nested = 'archive/source_004/_evaluated_model/_evaluated_model/model.py'
        evaluated_prior = directory + '/' + nested
        nested_directory = evaluated_prior.rsplit('/', 1)[0]
        knowledge['files'][directory + '/source_catalog.json'] = json.dumps([
            {'id': 'source_004', 'directory': nested.rsplit('/_evaluated_model/model.py', 1)[0],
             'model': nested, 'status': 'completed'}])
        knowledge['files'][evaluated_prior] = knowledge['files'][prior]
        knowledge['files'][nested_directory + '/settings.json'] = '{"gain": 11}'
        knowledge['files'][nested_directory + '/model_observations.json'] = '[]'
        target = 22.
    agent = CommonArchiveInquiryAgent(None, tmp_path / 'source', arm='code_react',
        knowledge=knowledge, source_task=False,
        config=InquiryLLMConfig(observation_budget=10, confirmation_rows=3,
                               compute_budget=100, max_model_calls=0, token_budget=0))
    lab = InquiryLab(seed=8, mechanism='linear', driver=0, orientation=1)
    agent._execute('write', {'path': 'current.py',
        'code': "def predict(row):\n return float(next(iter(row['features'].values())))\n"}, lab)
    finished = agent._execute('finish', {'model': 'current.py'}, lab)
    assert finished['fresh_confirmation']['passed']
    exported = agent.result('completed')['outgoing_knowledge']
    assert exported[field] == knowledge[field]
    assert prior in exported['files'], 'The archive advertises a predictor whose source was discarded'
    if advertisement == 'failed_source_catalog':
        for name in ('analysis.py', 'draft_notes.md'):
            path = directory + '/' + name
            assert exported['files'][path] == knowledge['files'][path]

    assert evaluated_prior in exported['files']

    restored = tmp_path / 'restored'
    for name, content in exported['files'].items():
        path = restored / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    scored = evaluate_python_model(restored, evaluated_prior, [{'features': {'u': 2.}, 'target': target}])
    assert scored['returncode'] == 0, scored
    assert scored['mse'] == 0.

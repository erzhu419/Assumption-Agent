"""Purchased-channel forecasts reuse the ordinary prospective comparison path."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_observable_research import MeasuredLab, PARENT as COMPOSED_MODEL
from test_common_archive_online_research import CHILD, PARENT, READOUT
from test_inquiry_revision import action, write


def setup(tmp_path, arm, *, measured=True):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm=arm, source_task=False,
        config=InquiryLLMConfig(observation_budget=30, confirmation_rows=2,
                               execution_seconds=30., compute_budget=2000))
    lab = MeasuredLab()
    agent._execute('query', {'actions': [action(0., channels=['meter'] if measured else [])],
                             'question': PARENT}, lab)
    agent._execute('evaluate', {'model': 'response.py',
        'code': "def predict(row): return 7*row['features']['u'] + 3.\n"}, lab)
    for model, slope in [('a.py', 2), ('b.py', 3)]:
        write(agent, model, f"""offset = 0.
def fit(rows):
    global offset
    assert len(rows) == 1 and rows[0]['target'] == 1., 'incorrect fitting evidence'
    offset = rows[0]['target']
def predict(row): return {slope}*row['features']['u'] + offset
""", lab)
    write(agent, 'readout.py', READOUT, lab)
    return agent, lab


def compare(agent, lab, *, models=None, channels=None, target='meter'):
    return agent._execute('run', {'operator': 'compare_experiments', 'target': target,
        'models': models or ['a.py', 'b.py'], 'experiments': [
            {'id': 'agreement', 'actions': [action(0., channels=['meter'])]},
            {'id': 'contrast', 'actions': [action(.5, channels=['meter'] if channels is None else channels)]},
        ]}, lab)['comparison']


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_meter_candidates_select_then_score_meter_and_return_authored_answer(tmp_path, arm):
    agent, lab = setup(tmp_path, arm)
    incumbent = deepcopy(agent.evaluated_model)
    response_evidence = (tmp_path / 'model_observations.json').read_text()
    result = compare(agent, lab)
    assert result['target'] == 'meter'
    assert result['eligible_models'] == ['a.py', 'b.py']
    assert [item['mse'] for item in result['prior_fit']] == [0., 0.]
    assert result['selected_experiment']['id'] == 'contrast'
    assert [item['predictions'] for item in result['forecasts']] == [[1., 2.], [1., 2.5]]
    assert result['query_arguments']['revision']['target'] == 'meter'
    assert agent._request(lab)['online_research']['latest_comparison']['target'] == 'meter'
    assert lab.paid == 1 and agent.evaluated_model == incumbent
    assert (tmp_path / 'model_observations.json').read_text() == response_evidence
    for candidate in result['candidate_contexts']:
        rows = json.loads((tmp_path / candidate['model']).with_name('model_observations.json').read_text())
        assert [row['target'] for row in rows] == [1.]

    arguments = deepcopy(result['query_arguments'])
    arguments['revision']['readout'] = 'readout.py'
    arguments['question'] = CHILD
    query = agent._execute('query', arguments, lab)
    forecasts = query['revision']['forecasts']
    assert [item['predictions'] for item in forecasts] == [[2.], [2.5]]
    assert [item['prospective_mse'] for item in forecasts] == [0., .25]
    assert [item['scored_rows'] for item in forecasts] == [1, 1]
    assert query['revision']['proposal']['original_models'] == {
        forecasts[0]['model']: 'a.py', forecasts[1]['model']: 'b.py'}
    assert query['revision']['authored_answer']['answer']['surviving'] == [forecasts[0]['model']]
    assert agent.evaluated_model == incumbent and lab.paid == 2
    assert agent.observable_models['meter']['training_observation_indices'] == [0]
    assert agent.observable_models['meter']['support'] == 'supported_on_later_readings'
    returned = next(question for question in agent.questions if question['id'] == 'parent')['child_answers'][-1]
    assert returned['answer']['surviving'] == [forecasts[0]['model']]


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_direct_meter_revision_uses_target_fitting_context(tmp_path, arm):
    agent, lab = setup(tmp_path, arm)
    incumbent = deepcopy(agent.evaluated_model)
    result = agent._execute('query', {'actions': [action(.5, channels=['meter'])],
        'revision': {'id': 'meter-check', 'affected_object': 'instrument', 'target': 'meter',
                     'models': ['a.py', 'b.py']}}, lab)['revision']
    assert [item['predictions'] for item in result['forecasts']] == [[2.], [2.5]]
    assert [item['prospective_mse'] for item in result['forecasts']] == [0., .25]
    assert agent.evaluated_model == incumbent


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_direct_revision_survivor_is_callable_from_parent_with_its_meter_fit(tmp_path, arm):
    agent, lab = setup(tmp_path, arm)
    result = agent._execute('query', {'actions': [action(.5, channels=['meter'])],
        'question': CHILD,
        'revision': {'id': 'meter-readout', 'affected_object': 'instrument', 'target': 'meter',
                     'models': ['a.py', 'b.py'], 'readout': 'readout.py'}}, lab)['revision']
    survivor = result['authored_answer']['answer']['surviving'][0]
    composed = agent._execute('evaluate', {'model': 'composed.py', 'code': COMPOSED_MODEL,
        'bindings': {'latent': survivor}, 'research_question': {'id': 'parent'}}, lab)
    assert composed['returncode'] == 0, composed.get('stderr')
    assert composed['mse'] == 0.
    assert composed['model_dependencies'][0]['calls'] == 2
    assert result['proposal']['original_models'][survivor] == 'a.py'
    assert survivor in result['proposal']['models']
    assert [row['target'] for row in json.loads(
        (tmp_path / survivor).with_name('model_observations.json').read_text())] == [1.]
    # Continue ordinary work after the answer; the executable return must stay
    # independent of both the edited draft and the expanded response evidence.
    write(agent, 'a.py', "def predict(row): return -999.\n", lab)
    done = agent._execute('finish', {'model': 'composed.py'}, lab)
    assert done['fresh_confirmation']['passed']
    assert done['fresh_confirmation']['model_dependencies'][0]['calls'] == 2
    assert lab.paid == 4


def test_retained_meter_candidate_keeps_original_fit_when_paid_evidence_grows(tmp_path):
    agent, lab = setup(tmp_path, 'hegel')
    agent._execute('evaluate', {'target': 'meter', 'model': 'a.py'}, lab)
    retained = agent.observable_models['meter']['model']
    agent._execute('query', {'actions': [action(0., channels=['meter'])]}, lab)
    write(agent, 'b.py', """offset = 0.
def fit(rows):
    global offset
    assert len(rows) == 2 and all(row['target'] == 1. for row in rows)
    offset = rows[0]['target']
def predict(row): return 3*row['features']['u'] + offset
""", lab)
    result = compare(agent, lab, models=[retained, 'b.py'])
    assert result['eligible_models'] == [retained, 'b.py']
    for item, count in zip(result['candidate_contexts'], [1, 2]):
        assert len(json.loads((tmp_path / item['model']).with_name('model_observations.json').read_text())) == count
    agent._execute('query', {'actions': [action(.2, channels=['meter'])]}, lab)
    query = agent._execute('query', result['query_arguments'], lab)['revision']
    assert [item['predictions'] for item in query['forecasts']] == [[2.], [2.5]]
    assert [item['prospective_mse'] for item in query['forecasts']] == [0., .25]


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_response_rows_cannot_establish_meter_support(tmp_path, arm):
    agent, lab = setup(tmp_path, arm, measured=False)
    result = compare(agent, lab)
    assert result['diagnosis'] == 'no_paid_support'
    assert result['eligible_models'] == [] and result['selected_experiment'] is None
    assert agent.observable_models == {} and lab.paid == 1


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_meter_forecast_requires_meter_purchase_before_scoring(tmp_path, arm):
    agent, lab = setup(tmp_path, arm)
    result = compare(agent, lab, channels=[])
    assert result['status'] == 'comparison_failed'
    assert result['selected_experiment'] is None and lab.paid == 1
    query = agent._execute('query', {'actions': [action(.5, channels=[])],
        'revision': {'id': 'missing-channel', 'affected_object': 'instrument',
                     'target': 'meter', 'models': ['a.py', 'b.py']}}, lab)
    assert query['revision']['status'] == 'preparation_failed'
    assert 'channels' in query['error'] and lab.paid == 1


def test_meter_models_do_not_gain_support_when_target_defaults_to_response(tmp_path):
    agent, lab = setup(tmp_path, 'hegel')
    result = compare(agent, lab, target='response')
    assert result['eligible_models'] == [] and result['selected_experiment'] is None

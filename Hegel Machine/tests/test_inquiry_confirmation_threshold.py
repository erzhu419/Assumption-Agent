"""The declared task criterion governs paid screening and fresh confirmation."""

import json

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.online_inquiry import InquiryLab
from hegel_machine.cognition.online_inquiry_llm import (
    CONFIRMATION_MSE_THRESHOLD, InquiryLLMAgent, InquiryLLMConfig,
)


def test_confirmation_criterion_is_instance_local_and_visible_to_both_arms(tmp_path):
    agents = []
    for arm, threshold in [('hegel', 1e-6), ('code_react', None)]:
        settings = {} if threshold is None else {'confirmation_mse_threshold': threshold}
        agent = InquiryLLMAgent(None, tmp_path / arm, arm=arm, source_task=False,
            config=InquiryLLMConfig(observation_budget=5, confirmation_rows=3, **settings))
        lab = InquiryLab(seed=8, mechanism='linear', driver=0, orientation=1)
        names = lab.public_spec()['controls']
        agent._query({'actions': [{'kind': 'response', 'point': dict(zip(names, point))}
            for point in ((1., 0.), (0., 1.))]}, lab)
        agent._execute('write', {'path': 'model.py', 'code': '''def fit(rows):
    global name, coefficient
    name = next(iter(rows[0]['features']))
    coefficient = rows[0]['target'] / rows[0]['features'][name]
def predict(row): return coefficient * row['features'][name] + 0.0001
'''}, lab)
        if arm == 'hegel':
            agent._execute('write', {'path': 'shared/method.py',
                'code': 'def select_actions(spec, rows, memory): return []\n'}, lab)
            agent._execute('write', {'path': 'shared/knowledge.json', 'code': json.dumps({
                'hypotheses': [{'id': 'h', 'path': 'model.py'}],
                'research_programs': [{'id': 'm', 'path': 'shared/method.py'}],
                'support_questions': []})}, lab)
        agents.append((agent, lab))

    # Both instances already exist; executing the relaxed arm cannot change
    # the default arm's declaration or acceptance criterion.
    for agent, lab in agents:
        request = agent._request(lab)
        threshold = agent.config.confirmation_mse_threshold
        assert request['limits']['confirmation_mse_threshold'] == threshold
        assert f'mean squared error (MSE) is at most {threshold}.' in request['task_interface']
        state = json.loads((agent.workspace / 'research_state.json').read_text())
        assert state['limits']['confirmation_mse_threshold'] == threshold
        fresh = agent._execute('finish', {'model': 'model.py'}, lab)['fresh_confirmation']
        assert fresh['mse'] == pytest.approx(1e-8)
        assert fresh['mse_threshold'] == threshold
        assert fresh['passed'] is (agent.arm == 'hegel')
        assert agent.costs['observation_cost'] == 5
    assert InquiryLLMConfig().confirmation_mse_threshold == CONFIRMATION_MSE_THRESHOLD == 1e-10


@pytest.mark.parametrize('threshold,accepted', [(1e-10, False), (1e-6, True)])
def test_paid_precondition_uses_the_same_criterion_as_fresh_confirmation(tmp_path, threshold, accepted):
    lab = AnonymousHistoryLab(kind='memoryless', seed=113)
    agent = InquiryLLMAgent(None, tmp_path, arm='code_react', config=InquiryLLMConfig(
        observation_budget=5, confirmation_rows=3, confirmation_mse_threshold=threshold))
    agent._query({'actions': [{'kind': 'trajectory', 'channels': [],
        'sequence': [{'u': -1.}, {'u': 1.}]}]}, lab)
    agent._execute('write', {'path': 'model.py', 'code': '''def fit(rows):
    global slope, intercept
    first, second = rows[:2]
    slope = (second['target'] - first['target']) / (second['features']['u'] - first['features']['u'])
    intercept = first['target'] - slope * first['features']['u']
def predict(row): return slope * row['features']['u'] + intercept + 0.0001
'''}, lab)
    result = agent._execute('finish', {'model': 'model.py'}, lab)
    assert result['mse'] == pytest.approx(1e-8)
    if accepted:
        fresh = result['fresh_confirmation']
        assert fresh['passed'] and fresh['mse_threshold'] == threshold
        assert fresh['mse'] == pytest.approx(1e-8)
        assert agent.model == 'model.py'
        assert agent.costs['observation_cost'] == lab.costs['observation_cost'] == 5
    else:
        assert 'paid evidence contradicts' in result['error']
        assert 'fresh_confirmation' not in result
        assert agent.model is None
        assert agent.costs['observation_cost'] == lab.costs['observation_cost'] == 2

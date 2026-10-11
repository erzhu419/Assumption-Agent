"""Authored prospective experiments use the common paid inquiry path."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, InquiryLLMConfig
from hegel_machine.cognition.temporal_paired_method import ProcedureConsumerAgent


class PublicLab:
    def __init__(self):
        self.paid = 0
        self.actions = []

    def public_spec(self):
        return {'controls': ['u'], 'action_kinds': ['trajectory']}

    def action_cost(self, action):
        return len(action['sequence'])

    def prediction_rows(self, action):
        return [Observation(deepcopy(point), 0., tuple(deepcopy(action['sequence'][:i])))
                for i, point in enumerate(action['sequence'])]

    def execute(self, action):
        if action.get('fail'):
            raise ValueError('apparatus unavailable')
        rows = self.prediction_rows(action)
        rows = [Observation(row.features, row.features['u'], row.history) for row in rows]
        first = self.paid
        self.paid += len(rows)
        self.actions.append(deepcopy(action))
        return {'action': deepcopy(action), 'rows': [asdict(row) for row in rows],
                'row_indices': list(range(first, self.paid)), 'observation_index': first,
                'observation_cost': len(rows), 'measurement_cost': 0}

    def observation_rows(self, record):
        return [Observation(row['features'], row['target'], tuple(row['history'])) for row in record['rows']]


def action(*values, **extra):
    return {'kind': 'trajectory', 'sequence': [{'u': value} for value in values], **extra}


def make_agent(tmp_path, arm='code_react', *, cls=InquiryLLMAgent, compute=1000):
    return cls(None, tmp_path, arm=arm, source_task=False,
        config=InquiryLLMConfig(observation_budget=100, confirmation_rows=1,
                               compute_budget=compute, execution_seconds=30.))


def write(agent, path, source, lab):
    agent._execute('write', {'path': path, 'code': source}, lab)


def revision(models, **extra):
    return {'id': 'r1', 'affected_object': 'representation', 'models': models,
            'parent_and_scope': {'parent': 'old', 'scope': 'public controls'}, **extra}


@pytest.mark.parametrize('arm', ['code_react', 'textual_memory', 'hegel'])
def test_all_arms_freeze_before_purchase_and_preserve_better_old_model(tmp_path, arm):
    agent, lab = make_agent(tmp_path, arm), PublicLab()
    write(agent, 'old.py', "def predict(row): return row['features']['u']\n", lab)
    write(agent, 'new.py', "def predict(row): return 0.\n", lab)
    agent._execute('query', {'actions': [action(2)]}, lab)
    agent._execute('evaluate', {'model': 'old.py'}, lab)
    result = agent._execute('query', {'actions': [action(4, 5), action(7)],
                                     'revision': revision(['old.py', 'new.py'])}, lab)
    record = result['revision']
    assert record['evidence_cutoff'] == {'rows': 1, 'records': 1}
    assert record['forecasts'][0]['predictions'] == [4, 5, 7]
    assert record['forecasts'][0]['prospective_mse'] == 0
    assert record['forecasts'][1]['prospective_mse'] == pytest.approx(30.)
    assert record['forecasts'][0]['prediction_inputs'][2]['history'] == ()
    assert all('target' not in row for row in record['forecasts'][0]['prediction_inputs'])
    assert agent.evaluated_model['source_model'] == 'old.py'
    assert record['evaluated_model']['source_model'] == 'old.py'
    assert agent.costs['compute_cost'] == 9  # prior evaluate + 6 predictions + 2 prior comparisons
    assert agent.costs['model_calls'] == 0
    write(agent, 'old.py', 'def predict(row): return -1.\n', lab)
    assert record['forecasts'][0]['source']['files']['old.py'] != (tmp_path / 'old.py').read_text()
    assert len(agent.result('researching')['revision_proposals']) == 1


def test_optional_fit_uses_only_the_preexperiment_paid_evidence(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    source = """value = 0.
def fit(rows):
    global value
    value = sum(r['target'] for r in rows) / len(rows)
def predict(row): return value
"""
    write(agent, 'mean.py', source, lab)
    agent._execute('query', {'actions': [action(1)]}, lab)
    result = agent._execute('query', {'actions': [action(9)], 'revision': revision(['mean.py'])}, lab)
    forecast = result['revision']['forecasts'][0]
    assert forecast['predictions'] == [1.]
    assert forecast['prospective_mse'] == 64.
    # An ordinary evaluate after purchase may refit; it must not change the earlier forecast.
    result_after = agent._execute('evaluate', {'model': 'mean.py'}, lab)
    assert result_after['mse'] == 16.
    assert agent.revisions[0]['forecasts'][0]['predictions'] == [1.]


def test_new_candidate_enters_current_paid_comparison(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'old.py', 'def predict(row): return 0.\n', lab)
    write(agent, 'new.py', "def predict(row): return row['features']['u']\n", lab)
    agent._execute('query', {'actions': [action(2)]}, lab)
    agent._execute('evaluate', {'model': 'old.py'}, lab)
    result = agent._execute('query', {'actions': [action(3)], 'revision': revision(['old.py', 'new.py'])}, lab)
    assert result['revision']['evaluated_model']['source_model'] == 'new.py'
    assert agent.evaluated_model['current_mse'] == 0


def test_retained_path_is_scored_before_new_candidate_replaces_its_context(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'old.py', 'def predict(row): return 0.\n', lab)
    agent._execute('query', {'actions': [action(2)]}, lab)
    agent._execute('evaluate', {'model': 'old.py'}, lab)
    retained = agent._evaluated_summary()['model']
    write(agent, 'old.py', 'def predict(row): return -99.\n', lab)
    write(agent, 'new.py', "def predict(row): return row['features']['u']\n", lab)
    result = agent._execute('query', {'actions': [action(3)],
        'revision': revision(['new.py', retained])}, lab)['revision']
    prior = next(item for item in result['prior_fit'] if item['model'] == retained)
    assert prior['mse'] == 4.
    assert result['forecasts'][1]['predictions'] == [0.]
    assert result['evaluated_model']['source_model'] == 'new.py'


def test_program_revision_returns_negative_child_answer_and_changes_parent_action(tmp_path):
    agent, lab = make_agent(tmp_path, 'hegel', cls=ProcedureConsumerAgent), PublicLab()
    write(agent, 'model.py', 'def predict(row): return 0.\n', lab)
    write(agent, 'readout.py', """def readout(result,spec,memory):
    return {'status':'refuted','answer':{'next_control':7},
            'evidence':result['observations'][0]['row_indices']}
""", lab)
    write(agent, 'plan.py', """import json
def select_actions(spec,observations,memory):
    state=json.load(open('research_state.json'))
    if state['revisions']:
        answer=state['questions'][0]['child_answers'][-1]
        assert answer['status']=='refuted'
        return {'actions':[{'kind':'trajectory','sequence':[{'u':answer['answer']['next_control']}]}]}
    return {'actions':[{'kind':'trajectory','sequence':[{'u':3}]}],
            'question':{'id':'child','question':'Does this prediction hold?', 'parent_id':'parent','depends_on':[]},
            'revision':{'id':'r1','affected_object':'representation','models':['model.py'],'readout':'readout.py'}}
""", lab)
    agent.questions.append({'id': 'parent', 'question': 'Predict response', 'depends_on': [], 'status': 'open'})
    first = agent._execute('query', {'path': 'plan.py'}, lab)
    assert first['question']['status'] == 'refuted'
    assert first['revision']['analysis_is_agent_authored']
    second = agent._execute('query', {'path': 'plan.py'}, lab)
    assert second['observations'][0]['action'] == action(7)
    assert agent.questions[0]['child_answers'][0]['question_id'] == 'child'
    assert lab.paid == 2


def test_failed_predictor_and_readout_remain_visible_without_accepting_them(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'bad.py', "def predict(row): raise ValueError('bad prediction')\n", lab)
    write(agent, 'bad_readout.py', "def readout(result,spec,memory): raise ValueError('bad analysis')\n", lab)
    result = agent._execute('query', {'actions': [action(4)],
        'revision': revision(['bad.py'], readout='bad_readout.py')}, lab)
    record = result['revision']
    assert record['forecasts'][0]['returncode'] != 0
    assert 'prospective_mse' not in record['forecasts'][0]
    assert 'bad analysis' in record['readout_error']
    assert record['status'] == 'observed'
    assert agent.evaluated_model is None
    assert lab.paid == 1


def test_partial_acquisition_keeps_failed_revision_without_complete_score(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'model.py', "def predict(row): return row['features']['u']\n", lab)
    result = agent._execute('query', {'actions': [action(4), action(5, fail=True)],
                                     'revision': revision(['model.py'])}, lab)
    assert result['error'] == 'apparatus unavailable'
    record = result['revision']
    assert record['status'] == 'acquisition_failed'
    assert record['record_indices'] == [0]
    assert record['forecasts'][0]['predictions'] == [4, 5]
    assert record['forecasts'][0]['scored_rows'] == 1
    assert record['forecasts'][0]['prospective_mse'] is None
    assert record['forecasts'][0]['complete_experiment'] is False
    assert agent.revisions[0] == record


def test_forecast_budget_failure_is_saved_without_buying_actions(tmp_path):
    agent, lab = make_agent(tmp_path, compute=1), PublicLab()
    write(agent, 'model.py', 'def predict(row): return 0.\n', lab)
    result = agent._execute('query', {'actions': [action(1, 2)], 'revision': revision(['model.py'])}, lab)
    assert 'budget exhausted' in result['error']
    assert result['revision']['status'] == 'preparation_failed'
    assert lab.paid == 0
    assert json.loads((tmp_path / 'research_state.json').read_text())['revisions'][0]['status'] == 'preparation_failed'


def test_readout_cannot_return_unpurchased_evidence_as_child_answer(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    write(agent, 'model.py', 'def predict(row): return 0.\n', lab)
    write(agent, 'readout.py', "def readout(result,spec,memory): return {'status':'supported','answer':'yes','evidence':[9]}\n", lab)
    result = agent._execute('query', {'actions': [action(4)], 'question': {
        'id': 'child', 'question': 'Check response', 'parent_id': None, 'depends_on': []},
        'revision': revision(['model.py'], readout='readout.py')}, lab)
    assert 'actual paid response row indices' in result['revision']['readout_error']
    assert result['question']['status'] == 'observed'
    assert 'answer' not in result['question']

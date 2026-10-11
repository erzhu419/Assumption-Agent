"""Scripted executable dependency checks; these are not Luna discovery results."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_online_research import ConfirmationLab
from test_inquiry_revision import action


class MeasuredLab(ConfirmationLab):
    target_rows = TemporalInquiryLab.target_rows
    offset = 1.

    def public_spec(self):
        return {**super().public_spec(), 'channels': [{'name': 'meter'}]}

    def execute(self, request):
        record = super().execute(request)
        record['measurements'] = []
        for index, row in zip(record['row_indices'], record['rows']):
            u = row['features']['u']
            meter = 2. * u + self.offset
            row['target'] = 3. * meter + u
            if request.get('channels') == ['meter']:
                record['measurements'].append({'observation_index': index, 'values': {'meter': meter}})
        record['measurement_cost'] = 3 * len(record['measurements'])
        return record


CHILD = """offset = 0.
def fit(rows):
    global offset
    offset = sum(r['target'] - 2*r['features']['u'] for r in rows)/len(rows)
def predict(row): return 2*row['features']['u'] + offset
"""
PARENT = """def bind_models(models):
    global auxiliary
    auxiliary = models['latent']
def predict(row): return 3*auxiliary(row) + row['features']['u']
"""


def setup(tmp_path, arm):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm=arm, source_task=False,
        config=InquiryLLMConfig(observation_budget=30, confirmation_rows=3,
                               execution_seconds=30., compute_budget=1000))
    lab = MeasuredLab()
    agent._execute('query', {'actions': [action(.1, .5, channels=['meter'])]}, lab)
    agent._execute('evaluate', {'target': 'meter', 'model': 'child.py', 'code': CHILD}, lab)
    return agent, lab


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_submodel_enters_parent_predictions_and_fresh_confirmation(tmp_path, arm):
    agent, lab = setup(tmp_path, arm)
    child = agent.observable_models['meter']
    assert child['mse'] == pytest.approx(0., abs=1e-28)
    assert child['support'] == 'paid_fit'
    assert agent.evaluated_model is None  # A meter model cannot replace the response.
    assert len(json.loads((tmp_path / 'model_observations.json').read_text())) == 2
    # Changing the draft cannot change a frozen executable child result.
    agent._execute('write', {'path': 'child.py', 'code': 'def predict(row): return -999.\n'}, lab)
    composed = agent._execute('evaluate', {'model': 'parent.py', 'code': PARENT,
        'bindings': {'latent': child['model']}}, lab)
    assert composed['mse'] < 1e-25
    assert composed['model_dependencies'][0]['calls'] == 2
    selected = composed['evaluated_model']['model']
    done = agent._execute('finish', {'model': selected}, lab)
    assert done['fresh_confirmation']['passed']
    assert done['fresh_confirmation']['model_dependencies'][0]['calls'] == 3
    assert lab.paid == 5
    assert agent.model == selected
    request = agent._request(lab)
    assert 'NONEMPTY' not in request['task_interface']
    if arm == 'hegel':
        child_question = next(q for q in agent.questions if q.get('observable_target') == 'meter')
        assert child_question['origin'] == 'framework_organization_of_author_selected_target'
        assert request['research_dependency_focus']['question']['id'] == 'task'
    else:
        assert not agent.questions
        assert 'research_dependency_focus' not in request


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_only_new_measurements_reassess_and_refute_frozen_submodel(tmp_path, arm):
    agent, lab = setup(tmp_path, arm)
    saved = deepcopy(agent.observable_models['meter'])
    compute = agent.costs['compute_cost']
    agent._execute('query', {'actions': [action(.3)]}, lab)
    assert agent.costs['compute_cost'] == compute
    assert agent.observable_models['meter'] == saved
    result = agent._execute('query', {'actions': [action(.8, channels=['meter'])]}, lab)
    entry = agent.observable_models['meter']
    assert entry['support'] == 'supported_on_later_readings'
    assert entry['training_observation_indices'] == [0, 1]
    assert entry['evaluated_observation_indices'] == [0, 1, 3]
    assert agent.costs['compute_cost'] == compute + 3
    assert result['observable_reassessments'][0]['evaluation']['mse'] < 1e-25
    lab.offset = 4.
    result = agent._execute('query', {'actions': [action(.8, channels=['meter'])]}, lab)
    assert agent.observable_models['meter']['support'] == 'refuted_on_paid_rows'
    assert agent.observable_models['meter']['model'] == saved['model']
    assert result['observable_reassessments'][0]['evaluation']['mse'] > 1.
    if arm == 'hegel':
        focus = agent._request(lab)['research_dependency_focus']['question']
        assert focus['observable_target'] == 'meter'
        assert focus['status'] == 'refuted_on_paid_rows'


def test_worse_auxiliary_draft_preserves_local_success_and_parent_incumbent(tmp_path):
    agent, lab = setup(tmp_path, 'hegel')
    child = deepcopy(agent.observable_models['meter'])
    parent = agent._execute('evaluate', {'model': 'parent.py', 'code': PARENT,
        'bindings': {'latent': child['model']}}, lab)['evaluated_model']
    failed = agent._execute('evaluate', {'target': 'meter', 'model': 'bad.py',
        'code': 'def predict(row): return 0.\n'}, lab)
    assert failed['mse'] > 0
    assert agent.observable_models['meter'] == child
    assert agent._evaluated_summary() == parent


def test_author_question_text_survives_executable_parent_return(tmp_path):
    agent, lab = setup(tmp_path, 'hegel')
    agent._execute('evaluate', {'target': 'meter', 'model': agent.observable_models['meter']['model'],
        'research_question': {'id': 'authored', 'question': 'Does this chosen history explain the auxiliary reading?',
                              'parent_id': 'task', 'depends_on': []}}, lab)
    question = next(q for q in agent.questions if q['id'] == 'authored')
    assert question['assignment']['question'] == 'Does this chosen history explain the auxiliary reading?'
    assert 'origin' not in question
    returned = next(q for q in agent.questions if q['id'] == 'task')['child_answers'][-1]
    assert returned['question_id'] == 'authored'
    assert returned['model'] == agent.observable_models['meter']['model']


def test_reselecting_old_snapshot_keeps_that_snapshots_training_indices(tmp_path):
    agent, lab = setup(tmp_path, 'code_react')
    original = deepcopy(agent.observable_models['meter'])
    lab.offset = 4.
    agent._execute('query', {'actions': [action(.8, channels=['meter'])]}, lab)
    agent._execute('evaluate', {'target': 'meter', 'model': 'child.py'}, lab)
    replacement = deepcopy(agent.observable_models['meter'])
    assert replacement['version'] == 1
    assert replacement['training_observation_indices'] == [0, 1, 2]
    lab.offset = 1.
    agent._execute('query', {'actions': [action(*([.2] * 15), channels=['meter'])]}, lab)
    agent._execute('evaluate', {'target': 'meter', 'model': original['model']}, lab)
    selected = agent.observable_models['meter']
    assert selected['version'] == 2
    assert selected['training_observation_indices'] == [0, 1]
    assert selected['evaluated_observation_indices'] == list(range(18))
    assert selected['support'] == 'refuted_on_paid_rows'


def test_resume_keeps_observable_support_and_executable_context(tmp_path):
    from test_common_archive_resume import agent_at, execute, SEED

    old, laboratory = agent_at(tmp_path / 'source'), TemporalInquiryLab(task_index=1)
    execute(old, laboratory, 'query', {'actions': [action(.1, .5, channels=['meter'])]})
    execute(old, laboratory, 'evaluate', {'target': 'meter', 'model': 'child.py', 'code': CHILD})
    prior = old.result('model_call_budget_exhausted')
    resumed, target = agent_at(tmp_path / 'resumed'), TemporalInquiryLab(task_index=1)
    resumed.restore_research(prior, old.workspace, target, confirmation_seed=SEED)
    assert resumed.observable_models == old.observable_models
    assert resumed.questions == old.questions
    assert resumed.costs == old.costs
    entry = resumed.observable_models['meter']
    score = resumed._execute('evaluate', {'target': 'meter', 'model': entry['model']}, target)
    assert score['returncode'] == 0
    assert score['mse'] == pytest.approx(entry['mse'])

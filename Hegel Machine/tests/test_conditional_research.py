"""Measured-coordinate components through the ordinary online execution path.

These supplied toy relations check integration, not autonomous discovery.
"""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_observable_research import MeasuredLab, CHILD
from test_common_archive_resume import agent_at, execute, SEED
from test_inquiry_revision import action


VIEW = """import json
def bind(spec, observations, memory):
    field = json.load(open('field.json'))
    rows = []
    for record in observations:
        measured = {item['observation_index']: item['values']['meter']
                    for item in record['measurements']}
        for index, row in zip(record['row_indices'], record['rows']):
            if index in measured:
                rows.append({'features': {'u': row['features']['u'], field: measured[index]},
                             'target': row['target'], 'evidence': [index]})
    return {'rows': rows}
"""

RELATION = """offset = 0.
def fit(rows):
    global offset
    offset = sum(row['target']-3*row['features']['observed']-row['features']['u']
                 for row in rows)/len(rows)
def predict(row):
    return 3*row['features']['observed']+row['features']['u']+offset
"""

PARENT = """def bind_models(models):
    global meter, relation
    meter, relation = models['meter'], models['response_relation']
def predict(row):
    return relation({'features': {'u': row['features']['u'], 'observed': meter(row)},
                     'history': []})
"""


def setup(tmp_path, arm='hegel'):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm=arm, source_task=False,
        config=InquiryLLMConfig(observation_budget=30, confirmation_rows=3,
                               execution_seconds=30., compute_budget=1000))
    lab = MeasuredLab()
    for name, source in [('field.json', '"observed"'), ('view.py', VIEW)]:
        agent._execute('write', {'path': name, 'code': source}, lab)
    agent._execute('query', {'actions': [action(.1, .5, channels=['meter'])]}, lab)
    result = agent._execute('evaluate', {'component': 'response_relation',
        'view': 'view.py', 'model': 'relation.py', 'code': RELATION}, lab)
    assert result['mse'] < 1e-25, result
    return agent, lab


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_measured_component_is_retained_and_actually_called_by_control_only_parent(tmp_path, arm):
    agent, lab = setup(tmp_path, arm)
    entry = agent.conditional_models['response_relation']
    rows = json.loads((tmp_path / entry['model']).parent.joinpath('model_observations.json').read_text())
    assert [row['features']['observed'] for row in rows] == [1.2, 2.]
    assert entry['support'] == 'paid_fit' and agent.evaluated_model is None
    # Root fitting inputs remain control-only response rows.
    assert all(set(row['features']) == {'u'} for row in json.loads((tmp_path / 'observations.json').read_text()))
    agent._execute('evaluate', {'target': 'meter', 'model': 'meter.py', 'code': CHILD}, lab)
    agent._execute('write', {'path': 'relation.py', 'code': 'def predict(row): return -99.\n'}, lab)
    result = agent._execute('evaluate', {'model': 'parent.py', 'code': PARENT,
        'bindings': {'meter': agent.observable_models['meter']['model'],
                     'response_relation': entry['model']}}, lab)
    assert result['mse'] < 1e-25
    assert {row['slot']: row['calls'] for row in result['model_dependencies']} == {
        'meter': 2, 'response_relation': 2}
    confirmed = agent._execute('finish', {'model': result['evaluated_model']['model']}, lab)
    assert confirmed['fresh_confirmation']['passed']
    assert {row['slot']: row['calls'] for row in confirmed['fresh_confirmation']['model_dependencies']} == {
        'meter': 3, 'response_relation': 3}
    request = agent._request(lab)
    state = json.loads((tmp_path / 'research_state.json').read_text())
    assert request['conditional_models'] == state['conditional_models'] == agent.conditional_models
    assert agent.result('agent_finished')['conditional_models'] == agent.conditional_models
    if arm == 'hegel':
        child = next(q for q in agent.questions if q.get('conditional_component') == 'response_relation')
        assert child['model_answer']['model'] == entry['model']
        assert request['research_dependency_focus']['available_components'] == agent.conditional_models
    else:
        assert agent.questions == []


def test_frozen_binding_and_helper_context_survive_draft_edits_and_preserve_old_predictor(tmp_path):
    agent, lab = setup(tmp_path)
    original = deepcopy(agent.conditional_models['response_relation'])
    agent._execute('write', {'path': 'field.json', 'code': '"wrong_coordinate"'}, lab)
    agent._execute('write', {'path': 'view.py', 'code': VIEW.replace("row['target']", "999.")}, lab)
    with pytest.raises(ValueError, match='distinct component ID'):
        agent._execute('evaluate', {'component': 'response_relation', 'view': 'view.py',
                                  'model': 'relation.py'}, lab)
    # An unmeasured response cannot change this relation's measurement evidence.
    compute = agent.costs['compute_cost']
    agent._execute('query', {'actions': [action(.3)]}, lab)
    assert agent.costs['compute_cost'] == compute
    agent._execute('query', {'actions': [action(.8, channels=['meter'])]}, lab)
    current = agent.conditional_models['response_relation']
    assert current['support'] == 'supported_on_later_readings'
    assert current['training_observation_indices'] == [0, 1]
    assert current['evaluated_observation_indices'] == [0, 1, 3]
    assert current['model'] == original['model']
    assert agent.costs['compute_cost'] == compute + 3
    result = agent._execute('evaluate', {'component': 'response_relation', 'model': 'bad.py',
                                      'code': 'def predict(row): return 0.\n'}, lab)
    assert result['mse'] > 1.
    assert agent.conditional_models['response_relation']['model'] == original['model']
    # Fitting a replacement still uses the frozen measured coordinates.
    result = agent._execute('evaluate', {'component': 'response_relation',
                                      'model': 'replacement.py', 'code': RELATION}, lab)
    assert result['mse'] < 1e-25


def test_same_target_components_coexist_and_later_counterexample_reopens_child(tmp_path):
    agent, lab = setup(tmp_path)
    first = deepcopy(agent.conditional_models['response_relation'])
    agent._execute('evaluate', {'component': 'second_response_relation', 'view': 'view.py',
                              'model': 'second.py', 'code': RELATION}, lab)
    second = agent.conditional_models['second_response_relation']
    assert first['model'] != second['model']
    assert len(agent.conditional_models) == 2 and agent.observable_models == {}
    original_execute = lab.execute
    def changed_response(request):
        result = original_execute(request)
        for row in result['rows']:
            row['target'] += 5.
        return result
    lab.execute = changed_response
    agent._execute('query', {'actions': [action(.7, channels=['meter'])]}, lab)
    assert all(row['support'] == 'refuted_on_paid_rows' for row in agent.conditional_models.values())
    assert agent.conditional_models['response_relation']['model'] == first['model']
    question = next(q for q in agent.questions if q['id'] == agent._active_question_id)
    assert question['conditional_component'] == 'second_response_relation'
    assert question['status'] == 'refuted_on_paid_rows'


def test_new_candidate_and_frozen_incumbent_are_compared_on_same_later_rows(tmp_path):
    agent, lab = setup(tmp_path)
    first = deepcopy(agent.conditional_models['response_relation'])
    original_execute = lab.execute
    def shifted_response(request):
        result = original_execute(request)
        for row in result['rows']:
            row['target'] += 3.
        return result
    lab.execute = shifted_response
    agent._execute('query', {'actions': [action(.7, channels=['meter'])]}, lab)
    assert agent.conditional_models['response_relation']['mse'] == pytest.approx(3.)
    result = agent._execute('evaluate', {'component': 'response_relation',
                                      'model': 'relation.py'}, lab)
    selected = agent.conditional_models['response_relation']
    assert result['mse'] == pytest.approx(2.)
    assert selected['version'] == 1 and selected['model'] != first['model']
    assert selected['training_observation_indices'] == [0, 1, 2]
    assert selected['view'] == first['view']
    assert selected['support'] == 'refuted_on_paid_rows'


def test_reused_retained_model_freezes_new_views_own_helper_context(tmp_path):
    agent, lab = setup(tmp_path)
    previous = agent.conditional_models['response_relation']
    agent._execute('write', {'path': 'new_view.py', 'code': VIEW}, lab)
    result = agent._execute('evaluate', {'component': 'new_relation', 'view': 'new_view.py',
                                        'model': previous['model']}, lab)
    assert result['mse'] < 1e-25
    new = agent.conditional_models['new_relation']
    assert (tmp_path / new['view']).read_text() == VIEW
    agent._execute('write', {'path': 'field.json', 'code': '"wrong_coordinate"'}, lab)
    agent._execute('write', {'path': 'new_view.py', 'code': 'def bind(*args): return {"rows": []}'}, lab)
    agent._execute('query', {'actions': [action(.7, channels=['meter'])]}, lab)
    assert agent.conditional_models['new_relation']['mse'] < 1e-25
    assert agent.conditional_models['new_relation']['model'] == new['model']


def test_restore_keeps_conditional_view_model_and_paid_account(tmp_path):
    source, lab = agent_at(tmp_path / 'source'), TemporalInquiryLab(task_index=1)
    for name, code in [('field.json', '"observed"'), ('view.py', VIEW)]:
        execute(source, lab, 'write', {'path': name, 'code': code})
    execute(source, lab, 'query', {'actions': [action(.1, .5, channels=['meter'])]})
    execute(source, lab, 'evaluate', {'component': 'relation', 'view': 'view.py',
                                    'model': 'relation.py', 'code': RELATION})
    previous = source.result('model_call_budget_exhausted')
    resumed, new_lab = agent_at(tmp_path / 'continued'), TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, source.workspace, new_lab, confirmation_seed=SEED)
    assert resumed.conditional_models == source.conditional_models
    assert resumed.costs == source.costs
    assert new_lab.costs == lab.costs
    entry = resumed.conditional_models['relation']
    result = resumed._execute('evaluate', {'component': 'relation', 'model': entry['model']}, new_lab)
    assert result['mse'] == pytest.approx(entry['mse'])
    assert (resumed.workspace / entry['view']).read_text() == VIEW

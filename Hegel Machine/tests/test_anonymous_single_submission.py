"""A first-submission study cannot learn from its final confirmation labels."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.anonymous_method_learning import (
    AnonymousPreparedMethodAgent, SingleSubmissionMethodAgent,
)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_common_archive_online_research import ConfirmationLab, NEW_MODEL, OLD_MODEL


def prepared(tmp_path, source, *, cls=SingleSubmissionMethodAgent, observations=300):
    manifest = {'hypotheses': [{'id': 'old', 'path': 'old.py'}],
        'research_programs': [{'id': 'method', 'path': METHOD_PATH}],
        'support_questions': []}
    knowledge = {'files': {'old.py': OLD_MODEL, 'good.py': NEW_MODEL,
        'shared/custom.json': json.dumps(manifest)}, 'manifest': manifest}
    agent = cls(None, tmp_path, arm='hegel', source_task=False, knowledge=knowledge,
        config=InquiryLLMConfig(observation_budget=observations, confirmation_rows=96,
            compute_budget=2000, execution_seconds=30., max_model_calls=0, token_budget=0))
    agent.bind_method(source)
    return agent, ConfirmationLab()


def declaration(model, explicit):
    if explicit:
        return {'action': 'finish', 'arguments': {'model': model,
            'manifest': 'shared/custom.json', 'lesson': 'Selected from paid evidence.'}}
    return {'model': model, 'lesson': 'Selected from paid evidence.'}


@pytest.mark.parametrize('body,status,count', [
    ("return {'stop': 'Computed, but no candidate was submitted.'}", 'method_stopped', 1),
    ("raise ValueError('fixture calculation failed')", 'method_no_progress', 2),
])
def test_unsubmitted_practice_saves_its_final_state_without_new_evidence(
        tmp_path, body, status, count):
    source = f'''def select_actions(spec, records, memory):
    total = sum(range(1000))
    print('completed_work=' + str(total))
    {body}
'''
    agent, lab = prepared(tmp_path, source)
    agent._prepare_program_state()
    assert json.loads((tmp_path / 'research_state.json').read_text())['method_results'] == []

    result = agent.execute_method(lab, max_steps=3)
    state = json.loads((tmp_path / 'research_state.json').read_text())

    assert result['status'] == status
    assert len(state['method_results']) == count
    assert state['method_results'] == result['method_results']
    assert state['costs'] == result['costs']
    assert state['costs']['python_execution_seconds'] > 0
    assert state['method_results'][-1]['feedback']['program_execution']['returncode'] == (
        0 if status == 'method_stopped' else 1)
    assert state['costs']['observation_cost'] == lab.paid == 0
    assert result['fresh_confirmation'] is None and agent.confirmations == []
    assert result['model'] is None


def repair_program(explicit):
    return f'''import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    if not steps:
        return {{'actions':[{{'kind':'trajectory','sequence':[{{'u':0.}}]}}]}}
    if len(steps) == 1:
        return {declaration('old.py', explicit)!r}
    if len(steps) == 2:
        assert steps[-1]['feedback']['fresh_confirmation']['passed'] is False
        return {{'action':'write','arguments':{{'path':'repaired.py','code':{NEW_MODEL!r}}}}}
    return {declaration('repaired.py', explicit)!r}
'''


@pytest.mark.parametrize('explicit', [False, True])
def test_failed_first_submission_never_reenters_method_but_normal_method_can_repair(tmp_path, explicit):
    source = repair_program(explicit)
    agent, lab = prepared(tmp_path / 'first_submission', source)
    checkpoints = []
    result = agent.execute_method(lab, max_steps=4,
        on_checkpoint=lambda data: checkpoints.append(deepcopy(data)))

    assert result['status'] == 'confirmation_failed'
    assert result['model'] is None
    assert result['fresh_confirmation']['passed'] is False
    assert result['fresh_confirmation']['rows'] == 96
    assert len(result['method_results']) == len(result['events']) == 2
    assert len(agent.confirmations) == 1
    assert result['costs']['observation_cost'] == lab.paid == 97
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 0
    assert not (agent.workspace / 'repaired.py').exists()
    submitted = result['method_results'][-1]['feedback']['submitted_candidate']
    assert submitted['source_model'] == 'old.py' and submitted['paid_rows'] == 1
    assert (agent.workspace / submitted['model']).read_text() == OLD_MODEL
    assert checkpoints[-2]['fresh_confirmation'] is None
    assert checkpoints[-2]['method_results'][-1]['feedback']['status'] == 'confirmation_not_authorized'
    assert checkpoints[-1]['status'] == 'confirmation_failed'
    assert result['events'][-1]['feedback']['fresh_confirmation']['rows'] == 96

    ordinary, ordinary_lab = prepared(tmp_path / 'ordinary', source, cls=AnonymousPreparedMethodAgent)
    repaired = ordinary.execute_method(ordinary_lab, max_steps=4)
    assert repaired['status'] == 'completed' and repaired['model'] == 'repaired.py'
    assert len(ordinary.confirmations) == 2 and len(repaired['method_results']) == 4
    assert repaired['costs']['observation_cost'] == 193


@pytest.mark.parametrize('explicit', [False, True])
def test_paid_experiment_then_selected_inherited_predictor_receives_real_confirmation(tmp_path, explicit):
    source = f'''import json
def select_actions(spec, records, memory):
    if not records:
        return {{'actions':[{{'kind':'trajectory','sequence':[{{'u':.75}}]}}]}}
    assert records[0]['rows'][0]['target'] == .75
    return {declaration('good.py', explicit)!r}
'''
    agent, lab = prepared(tmp_path, source)
    result = agent.execute_method(lab, max_steps=2)
    assert result['status'] == 'completed' and result['model'] == 'good.py'
    assert agent.lesson == 'Selected from paid evidence.'
    assert 'Selected from paid evidence.' in result['outgoing_knowledge']['lessons']
    assert result['fresh_confirmation']['passed'] and result['fresh_confirmation']['rows'] == 96
    assert result['costs']['observation_cost'] == lab.paid == 97
    assert result['costs']['compute_cost'] == 97
    assert len(result['method_results']) == 2
    feedback = result['method_results'][-1]['feedback']
    assert feedback['submitted_candidate']['source_model'] == 'good.py'
    if explicit:
        assert feedback['program_action'] == declaration('good.py', True)
        assert agent.manifest['research_programs'] == [{'id': 'method', 'path': METHOD_PATH}]
    else:
        assert feedback['submitted_model_inherited'] and feedback['submitted_model_unchanged']
        assert agent.manifest['hypotheses'] == [{'id': 'declared_predictor', 'path': 'good.py'}]
    state = json.loads((tmp_path / 'research_state.json').read_text())
    assert state['method_results'][-1]['feedback']['fresh_confirmation']['passed']


@pytest.mark.parametrize('source,observations,steps,status', [
    ("def select_actions(spec,records,memory): return {'stop':'insufficient evidence'}\n",
     300, 2, 'method_stopped'),
    ("def select_actions(spec,records,memory): return {'actions':[{'kind':'trajectory','sequence':[{'u':0.}]}]}\n",
     300, 1, 'method_step_limit'),
    ("def select_actions(spec,records,memory): return {'model':'good.py'}\n",
     95, 2, 'confirmation_budget_exhausted'),
    ("def select_actions(spec,records,memory): return {'model':'missing.py'}\n",
     300, 2, 'method_error'),
])
def test_no_submission_and_failed_execution_keep_distinct_outcomes(tmp_path, source, observations, steps, status):
    agent, lab = prepared(tmp_path, source, observations=observations)
    result = agent.execute_method(lab, max_steps=steps)
    assert result['status'] == status
    assert result['model'] is None and result['fresh_confirmation'] is None
    assert agent.confirmations == []
    assert result['costs']['observation_cost'] <= 1


def test_child_procedure_returns_a_proposal_without_receiving_confirmation_labels(tmp_path):
    source = f'''import json
def select_actions(spec, records, memory):
    steps = [step for step in json.load(open('research_state.json'))['method_results']
             if step['path'] == {METHOD_PATH!r}]
    if not steps:
        return {{'action':'run','arguments':{{'operator':'procedure','path':'child.py',
                                           'max_steps':1,'confirm':True}}}}
    json.dump({{'rows':sum(len(record['rows']) for record in records),
               'child':steps[-1]['feedback']}}, open('parent_view.json','w'))
    return {{'model':'good.py'}}
'''
    child = "def select_actions(spec,records,memory): return {'model':'old.py'}\n"
    for label, cls, expected_rows in [('single', SingleSubmissionMethodAgent, 96),
                                     ('ordinary', AnonymousPreparedMethodAgent, 192)]:
        agent, lab = prepared(tmp_path / label, source, cls=cls)
        agent._save('child.py', child)
        result = agent.execute_method(lab, max_steps=2)
        assert result['status'] == 'completed' and result['model'] == 'good.py'
        assert result['costs']['observation_cost'] == lab.paid == expected_rows
        view = json.loads((agent.workspace / 'parent_view.json').read_text())
        assert view['rows'] == expected_rows - 96
        if label == 'single':
            assert len(agent.confirmations) == 1
            assert view['child']['procedure_status'] == 'confirmation_not_authorized'
            assert view['child']['fresh_confirmation'] is None
            assert view['child']['steps'][-1]['feedback']['next_action'] == {
                'action': 'finish', 'arguments': {'model': 'old.py'}}
            own = [step for step in result['method_results'] if step['path'] == METHOD_PATH]
            assert own[0]['proposal']['action'] == 'run'
            assert own[-1]['feedback']['executed_path'] == METHOD_PATH
        else:
            assert len(agent.confirmations) == 2
            assert view['child']['fresh_confirmation']['passed'] is False


def test_direct_child_query_returns_to_parent_before_paid_experiment_and_final_submission(tmp_path):
    source = f'''import json
def select_actions(spec, records, memory):
    steps = [step for step in json.load(open('research_state.json'))['method_results']
             if step['path'] == {METHOD_PATH!r}]
    if not steps:
        return {{'action':'query','arguments':{{'path':'child.py'}}}}
    if len(steps) == 1:
        assert not records
        assert steps[-1]['feedback']['procedure_status'] == 'confirmation_not_authorized'
        assert steps[-1]['feedback']['next_action'] == {{'action':'finish','arguments':{{'model':'old.py'}}}}
        return {{'actions':[{{'kind':'trajectory','sequence':[{{'u':.75}}]}}]}}
    assert records[0]['rows'][0]['target'] == .75
    return {{'model':'good.py'}}
'''
    agent, lab = prepared(tmp_path, source)
    agent._save('child.py', "def select_actions(spec,records,memory): return {'model':'old.py'}\n")
    result = agent.execute_method(lab, max_steps=3)
    assert result['status'] == 'completed' and result['model'] == 'good.py'
    assert len(agent.confirmations) == 1 and result['fresh_confirmation']['rows'] == 96
    assert result['costs']['observation_cost'] == lab.paid == 97
    own = [step for step in result['method_results'] if step['path'] == METHOD_PATH]
    assert len(own) == 3
    assert own[0]['proposal'] == {'action': 'query', 'arguments': {'path': 'child.py'}}
    assert own[0]['feedback']['executed_path'] == METHOD_PATH
    assert own[-1]['feedback']['executed_path'] == METHOD_PATH
    assert result['events'][0]['feedback']['program_action'] == own[0]['proposal']

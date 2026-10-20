"""Existing procedure declarations must reach the ordinary inquiry handlers."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


class PublicLab:
    def __init__(self):
        self.rows = 0

    def public_spec(self):
        return {'controls': ['u']}

    def action_cost(self, action):
        return len(action['sequence'])

    def prediction_rows(self, action):
        return [Observation(point, 0., tuple(action['sequence'][:index]))
                for index, point in enumerate(action['sequence'])]

    def observation_rows(self, record):
        return [Observation(row['features'], row['target'], tuple(row['history'])) for row in record['rows']]

    def execute(self, action):
        start = self.rows
        rows = [Observation(row.features, row.features['u'], row.history)
                for row in self.prediction_rows(action)]
        self.rows += len(rows)
        return {'action': deepcopy(action), 'rows': [asdict(row) for row in rows],
                'row_indices': list(range(start, self.rows)), 'observation_index': start,
                'observation_cost': len(rows), 'measurement_cost': 0}


def make_agent(tmp_path):
    return CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(max_model_calls=0, observation_budget=20,
                               compute_budget=1000, execution_seconds=30.))


def install(agent, path, source):
    target = agent.workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)


def test_procedure_registers_existing_object_then_id_and_executes_query_readout(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    declaration = {'id': 'prospective', 'question': 'Does the saved prediction hold?'}
    install(agent, 'readout.py', """def readout(result,spec,memory):
    return {'status':'refuted','answer':{'saved_error':result['forecasts'][0]['prospective_mse']},
            'evidence':result['observations'][0]['row_indices']}
""")
    install(agent, 'procedure.py', """import json
def select_actions(spec,observations,memory):
    state=json.load(open('research_state.json'))
    if not state['method_results']:
        return {'action':'write','arguments':{'path':'model.py','code':'def predict(row): return 0.\\n'},
                'research_question':{'id':'prospective','question':'Does the saved prediction hold?'}}
    return {'action':'query','arguments':{
        'actions':[{'kind':'trajectory','sequence':[{'u':3.}]}],
        'revision':{'id':'r1','affected_object':'prediction','models':['model.py']},
        'readout':'readout.py'},'research_question':'prospective'}
""")
    first = agent._execute('query', {'path': 'procedure.py'}, lab)
    assert first['research_question_id'] == declaration['id'] and lab.rows == 0
    second = agent._execute('query', {'path': 'procedure.py'}, lab)
    record = second['revision']
    assert record['question_id'] == declaration['id']
    assert record['proposal']['readout'] == 'readout.py'
    assert record['analysis_is_agent_authored'] and record['returned_answer']['status'] == 'refuted'
    assert record['returned_answer']['answer']['saved_error'] == 9.
    assert record['evidence_status'] == 'not_checked'
    assert next(q for q in agent.questions if q['id'] == declaration['id'])['evidence'] == [0]
    assert agent.method_results[0]['proposal']['research_question'] == declaration
    assert agent.method_results[1]['proposal']['research_question'] == 'prospective'
    assert 'readout' not in agent.method_results[1]['proposal']['arguments']['revision']
    assert lab.rows == 1 and agent.costs['model_calls'] == 0


def test_explicit_argument_question_keeps_priority_over_procedure_metadata(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    proposed = {'action': 'write', 'arguments': {'path': 'note.txt', 'code': 'note',
        'research_question': {'id': 'explicit', 'question': 'The explicit question'}},
        'research_question': {'id': 'outer', 'question': 'The outer question'}}
    install(agent, 'procedure.py', f'def select_actions(spec,observations,memory): return {proposed!r}\n')
    result = agent._execute('query', {'path': 'procedure.py'}, lab)
    assert result['research_question_id'] == 'explicit'
    assert 'outer' not in {q['id'] for q in agent.questions}
    assert agent.method_results[0]['proposal'] == proposed
    assert lab.rows == 0


def test_procedure_id_cannot_invent_an_undeclared_question(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    install(agent, 'procedure.py', """def select_actions(spec,observations,memory):
    return {'action':'query','arguments':{'actions':[{'kind':'trajectory','sequence':[{'u':1.}]}]},
            'research_question':'not_declared'}
""")
    result = agent._execute('query', {'path': 'procedure.py'}, lab)
    assert 'a new research question needs its question text' in result['error']
    assert 'not_declared' not in {q['id'] for q in agent.questions} and lab.rows == 0


def test_revision_readout_keeps_priority_over_query_readout(tmp_path):
    agent, lab = make_agent(tmp_path), PublicLab()
    install(agent, 'model.py', 'def predict(row): return 0.\n')
    for name in ('revision', 'query'):
        install(agent, f'{name}.py', "def readout(result,spec,memory): return "
            + repr({'status': 'refuted', 'answer': name, 'evidence': [0]}) + '\n')
    arguments = {'actions': [{'kind': 'trajectory', 'sequence': [{'u': 2.}]}],
        'revision': {'id': 'r1', 'affected_object': 'prediction', 'models': ['model.py'],
                     'readout': 'revision.py'}, 'readout': 'query.py'}
    before = deepcopy(arguments)
    result = agent._execute('query', arguments, lab)
    assert result['revision']['returned_answer']['answer'] == 'revision'
    assert result['revision']['proposal']['readout'] == 'revision.py'
    assert arguments == before


@pytest.mark.parametrize('failure', ['observation_budget', 'missing_readout'])
def test_raw_procedure_failures_retain_question_without_observations_or_support(tmp_path, failure):
    from dataclasses import replace

    agent, lab = make_agent(tmp_path), PublicLab()
    declaration = {'id': 'raw', 'question': 'Does the declared forecast hold?'}
    proposal = {'question': declaration,
        'actions': [{'kind': 'trajectory', 'sequence': [{'u': 1.}]}],
        'revision': {'id': 'raw_revision', 'affected_object': 'prediction',
                     'models': ['model.py'], 'readout': 'missing.py'},
        'after_observation': [{'action': 'run', 'arguments': {'operator': 'analyze_revision'}}]}
    install(agent, 'procedure.py', f'def select_actions(spec,observations,memory): return {proposal!r}\n')
    if failure == 'observation_budget':
        agent.config = replace(agent.config, observation_budget=0)
        # Reusing this original assignment must not create a second question.
        agent._open_question(deepcopy(declaration))
    result = agent._execute('query', {'path': 'procedure.py'}, lab)
    question, = [q for q in agent.questions if q['id'] == 'raw']
    assert result['question']['id'] == 'raw'
    assert question['status'] == 'open' and question['evidence'] == []
    assert question.get('acquired_evidence') == [] and 'executable_answer' not in question
    assert question['after_observation'] == proposal['after_observation']
    assert question['continuation_origin'] == 'procedure.py'
    assert lab.rows == agent.costs['observation_cost'] == 0 and agent.observations == []
    if failure == 'observation_budget':
        assert result['error'] == 'observation budget exhausted' and agent.revisions == []
    else:
        assert 'missing.py' in result['error']
        assert result['revision']['status'] == 'preparation_failed'
        assert result['revision']['question_id'] == 'raw'
        assert result['revision']['question_meaning_version'] == question['meaning_version']
        assert 'returned_answer' not in result['revision']


def test_original_window_readout_runs_without_a_model_or_acquisition(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'scripts/replay_window_authored_readout.py'
    spec = importlib.util.spec_from_file_location('window_authored_readout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not (module.SOURCE / 'research_state.json').exists():
        pytest.skip('The original small retained state is unavailable')
    result = module.replay_authored_readout(output=tmp_path / 'receipt.json')
    assert result['readout']['status'] == 'refuted'
    assert result['readout']['evidence_status'] == 'not_checked'
    assert result['readout']['supported_models'] == []
    assert len(result['readout']['refuted_models']) == 2
    assert result['readout']['evidence_rows'] == {'first': 72, 'last': 119, 'count': 48}
    assert result['current_parent']['fresh_confirmation']['passed']
    assert result['current_parent']['current_failure_established'] is False
    assert result['costs']['new_model_executions'] == result['costs']['new_acquisitions'] == 0
    assert result['question_boundary']['generated_question_count_after_readout'] == 0

"""Compiled research keeps online comparison and executable parent binding tools."""

import json

from hegel_machine.cognition.anonymous_method_learning import AnonymousPreparedMethodAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_observable_research import MeasuredLab, PARENT
from test_common_archive_online_research import READOUT
from test_inquiry_revision import action


METHOD = '''import json
def select_actions(spec, records, memory):
    state = json.load(open('research_state.json'))
    step = len(state['method_results'])
    if step == 0:
        return {'action':'run','arguments':{
            'operator':'compare_experiments','target':'meter','models':['a.py','b.py'],
            'experiments':[{'id':'contrast','actions':[{
                'kind':'trajectory','sequence':[{'u':.5}], 'channels':['meter']}]}]}}
    if step == 1:
        arguments = state['experiment_comparisons'][-1]['query_arguments']
        arguments['revision']['readout'] = 'readout.py'
        arguments['question'] = {'id':'meter-test','question':'Which predictor survives?',
                                 'parent_id':'task','depends_on':[]}
        return {'action':'query','arguments':arguments}
    if step == 2:
        survivor = state['revisions'][-1]['authored_answer']['answer']['surviving'][0]
        return {'action':'evaluate','arguments':{
            'model':'parent.py','bindings':{'latent':survivor}}}
    return {'model':state['evaluated_model']['model']}
'''


def test_frozen_method_can_compare_auxiliary_experiments_and_use_returned_predictor(tmp_path):
    agent = AnonymousPreparedMethodAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=20, confirmation_rows=3,
                               execution_seconds=30., compute_budget=1000))
    lab = MeasuredLab()
    agent._execute('query', {'actions':[action(0., channels=['meter'])]}, lab)
    for name, slope in [('a.py', 2), ('b.py', 3)]:
        agent._execute('write', {'path':name, 'code':f'''offset = 0.
def fit(rows):
    global offset
    assert len(rows) == 1 and rows[0]['target'] == 1.
    offset = rows[0]['target']
def predict(row): return {slope} * row['features']['u'] + offset
'''}, lab)
    agent._execute('write', {'path':'parent.py','code':PARENT}, lab)
    agent._execute('write', {'path':'readout.py','code':READOUT}, lab)
    agent.bind_method(METHOD)
    result = agent.execute_method(lab, max_steps=4)
    assert result['status'] == 'completed'
    assert result['method_source'] == METHOD and result['costs']['model_calls'] == 0
    assert result['fresh_confirmation']['passed']
    assert result['fresh_confirmation']['model_dependencies'][0]['calls'] == 3
    assert result['experiment_comparisons'][0]['target'] == 'meter'
    forecasts = result['revision_proposals'][0]['forecasts']
    assert [entry['prospective_mse'] for entry in forecasts] == [0., .25]
    parent = next(q for q in result['online_questions'] if q['id'] == 'task')
    assert any(answer.get('question_id') == 'meter-test' for answer in parent['child_answers'])
    assert result['costs']['observation_cost'] == 5
    assert len(json.loads((tmp_path/'model_observations.json').read_text())) == 2

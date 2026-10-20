"""Saved executable checks remain discoverable after later notes replace an answer."""

import json

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.executable_evidence import _local_relations
from test_executable_evidence_binding import setup


def readout(agent, lab, *, count=0, revision_index=None, broken=False):
    answer = {'status': 'supported', 'answer': 'Current local interpretation',
              'evidence': [0, 1]}
    if count:
        answer['checks'] = [
            {'id': f'contrast_{index}',
             'terms': [{'row': 1, 'coefficient': 1}, {'row': 0, 'coefficient': -1}],
             'relation': 'eq', 'expected': 2. if index % 2 else 0.,
             'atol': 1e-9, 'scope': {'controls': [-1., 1.]}}
            for index in range(count)]
    code = ('def readout(result,spec,memory): raise ValueError("analysis unfinished")\n'
            if broken else f'def readout(result,spec,memory): return {answer!r}\n')
    agent._execute('write', {'path': 'readout.py', 'code': code}, lab)
    arguments = ({'operator': 'analyze_evidence', 'research_question': {'id': 'task'}}
                 if revision_index is None else
                 {'operator': 'analyze_revision', 'revision_index': revision_index})
    return agent._execute('run', {**arguments, 'path': 'readout.py'}, lab)['revision']


def test_old_checks_remain_indexed_after_same_question_answer_and_recent_revisions_change(tmp_path):
    agent, lab = setup(tmp_path)
    first = readout(agent, lab, count=16)
    second = readout(agent, lab, count=48)
    for _ in range(5):
        readout(agent, lab)

    notebook = agent._evidence_notebook()
    assert next(q for q in agent.questions if q['id'] == 'task')['answer'] == 'Current local interpretation'
    assert f"revision:{first['index']}" not in notebook['facts']
    assert f"revision:{second['index']}" not in notebook['facts']
    entries = notebook['checked_relations']
    assert [entry['revision_index'] for entry in entries] == [first['index'], second['index']]
    assert [entry['check_count'] for entry in entries] == [16, 48]
    for entry in entries:
        selected = _local_relations(agent, entry['selector'])
        assert entry['version'] == selected['version'] == 1
        assert entry['question_id'] == 'task'
        assert entry['satisfied_count'] == entry['violated_count'] == entry['check_count'] // 2
        assert entry['source'] == {'file': 'research_state.json',
                                   'json_path': ['revisions', entry['revision_index']]}
    compact = prepare_code_context(tmp_path, agent._request(lab), schema={}, instructions='')
    assert compact['evidence_notebook']['checked_relations'] == entries
    index_text = json.dumps(compact['evidence_notebook']['checked_relations'])
    assert len(index_text) < 800
    assert 'witnesses' not in index_text and 'history' not in index_text


def test_failed_latest_readout_does_not_advertise_stale_checks(tmp_path):
    agent, lab = setup(tmp_path)
    record = readout(agent, lab, count=2)
    assert agent._evidence_notebook()['checked_relations'][0]['check_count'] == 2
    readout(agent, lab, revision_index=record['index'], broken=True)
    assert record['analysis_attempts'][0]['check_results']
    assert agent._evidence_notebook()['checked_relations'] == []
    readout(agent, lab, revision_index=record['index'], count=4)
    current = agent._evidence_notebook()['checked_relations']
    assert len(current) == 1 and current[0]['version'] == 3 and current[0]['check_count'] == 4

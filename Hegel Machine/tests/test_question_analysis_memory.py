"""Question memory retains executed work without inventing scientific answers."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_common_archive_online_research import prepared
from test_common_archive_resume import execute, SEED
from test_executed_evidence_notebook import AccountedPublicLab


def question(identifier, parent=None):
    return {'id': identifier, 'question': f'Investigate {identifier}',
            'parent_id': parent, 'depends_on': []}


def research_scene(path, arm):
    agent, lab = prepared(path, arm), AccountedPublicLab()
    declarations = [question('task'), question('sibling', 'task'),
                    question('parent', 'task'), question('child', 'parent'),
                    question('other', 'task'), question('other', 'task')]
    for index, declaration in enumerate(declarations):
        execute(agent, lab, 'run', {'code': f"print('computed {index}')",
            'research_question': declaration, 'notes': 'Current interpretation without citations.'})
    # Age every computation beyond the provider's latest eight events. The
    # next author works on the child, even though its notebook omits old facts.
    for index in range(9):
        execute(agent, lab, 'write', {'path': 'scratch_note.txt', 'code': str(index),
            'research_question': question('child', 'parent'),
            'notes': 'Current interpretation without citations.'})
    return agent, lab


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_active_child_and_ancestors_keep_actual_work_after_notebook_omits_it(tmp_path, arm):
    agent, lab = research_scene(tmp_path, arm)
    before = deepcopy((agent.costs, agent.questions, agent.events))
    request = agent._request(lab)
    online = request['online_research']
    questions = {q['id']: q for q in online['questions']}

    for identifier, index in [('task', 0), ('parent', 2), ('child', 3)]:
        entry = questions[identifier]['executed_analyses'][0]
        fact = request['evidence_notebook']['facts'][entry['ref']]
        assert entry['ref'] == f'analysis:{index}'
        assert entry['event_index'] == index
        assert entry['executed_path'] == f'scratch_{index:03d}.py'
        assert entry['returncode'] == 0
        assert entry['source'] == fact['source'] == {
            'file': 'research_context.json', 'json_path': ['events', index, 'feedback']}
        assert fact['research_question_id'] == identifier
        assert fact['stdout'] == f'computed {index}\n'
        assert questions[identifier]['status'] == 'open'
        assert 'answer' not in questions[identifier]
        assert questions[identifier]['evidence'] == []
    assert questions['sibling']['executed_analyses'][0]['ref'] == 'analysis:1'
    assert 'analysis:1' not in request['evidence_notebook']['facts']
    assert [entry['ref'] for entry in questions['other']['executed_analyses']] == [
        'analysis:4', 'analysis:5']
    assert online['parent_returns'] == []
    assert agent.costs == before[0] and agent.questions == before[1] and agent.events == before[2]
    assert lab.paid == 0

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    assert context['event_history']['preview_start'] > 3
    assert context['evidence_notebook'] == request['evidence_notebook']
    compact_questions = {q['id']: q for q in context['online_research']['questions']}
    assert compact_questions['task']['executed_analyses'] == questions['task']['executed_analyses']
    assert context['research_notes'] == 'Current interpretation without citations.'


def test_silent_runs_errors_and_large_outputs_keep_execution_semantics(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), AccountedPublicLab()
    execute(agent, lab, 'run', {'code': 'pass', 'research_question': question('task')})
    execute(agent, lab, 'run', {'code': "raise ValueError('unfinished calculation')",
                              'research_question': question('task')})
    execute(agent, lab, 'run', {'code': "print('large output ' * 1000)",
                              'research_question': question('task')})
    request = agent._request(lab)
    records = request['online_research']['questions'][0]['executed_analyses']
    facts = request['evidence_notebook']['facts']

    assert [entry['returncode'] for entry in records] == [0, 1, 0]
    assert facts['analysis:0']['stdout'] == ''
    assert 'unfinished calculation' in facts['analysis:1']['stderr']
    for index in range(3):
        fact = facts[f'analysis:{index}']
        feedback = agent.events[index]['feedback']
        for stream in ('stdout', 'stderr'):
            output = (tmp_path / fact[f'{stream}_file']).read_text(encoding='utf-8')
            assert fact[f'{stream}_characters'] == len(output)
            assert fact[f'{stream}_truncated'] is (output != feedback[stream])
            if isinstance(fact[stream], str):
                assert fact[stream] == output
    large_output = facts['analysis:2']['stdout']
    assert large_output == {
        'full_content_file': facts['analysis:2']['stdout_file'],
        'characters': len('large output ' * 1000 + '\n')}
    assert (tmp_path / large_output['full_content_file']).read_text(encoding='utf-8') == (
        'large output ' * 1000 + '\n')
    assert facts['analysis:2']['stdout_truncated'] is True
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    assert context['evidence_notebook']['facts']['analysis:2']['stdout'] == large_output
    for record in records:
        assert 'supported' not in record and 'answer' not in record
    assert request['online_research']['questions'][0]['status'] == 'open'


def test_restore_preserves_question_history_and_fact_selection_without_repurchase(tmp_path):
    agent, lab = research_scene(tmp_path / 'old', 'hegel')
    request = agent._request(lab)
    (agent.workspace / 'research_context.json').write_text(json.dumps(request))
    previous = agent.result('model_call_budget_exhausted')
    restored, fresh = prepared(tmp_path / 'restored', 'hegel'), AccountedPublicLab()

    result = restored.restore_research(previous, agent.workspace, fresh, confirmation_seed=SEED)
    new_request = restored._request(fresh)

    assert result['new_observations'] == 0 and fresh.actions == []
    assert new_request['online_research']['questions'] == request['online_research']['questions']
    assert new_request['evidence_notebook'] == request['evidence_notebook']
    assert new_request['online_research']['active_question'] == 'child'
    assert restored.costs == agent.costs

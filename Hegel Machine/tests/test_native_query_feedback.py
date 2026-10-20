"""Query feedback references cumulative research without losing executable evidence."""

from copy import deepcopy
import json

from hegel_machine.cognition.anonymous_method_learning import AnonymousPreparedMethodAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig

from test_native_condition_inquiry import (
    COMPOSITION, PaidConditionalLab, buy_fixture, execute, full_round, prepared, research,
)


def test_query_keeps_current_continuation_and_restorable_native_rounds(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path), PaidConditionalLab()
    buy_fixture(agent, lab)
    first = research(agent, lab)['native_condition_research']
    original = json.loads(json.dumps(full_round(agent, first)))
    boundary = next(item for item in first['proposed_experiments']
                    if item.get('purpose') == 'conditional_boundary')['query_action']
    feedback = execute(agent, lab, boundary['action'], boundary['arguments'])
    question = next(q for q in agent.questions if q['id'] == COMPOSITION['id'])
    reference = feedback['question']['native_condition_research']
    assert reference['round_count'] == 2
    assert reference['full_content_file'] == 'research_state.json'
    state = json.loads((tmp_path / reference['full_content_file']).read_text())
    resolved = state
    for field in reference['json_path']:
        resolved = resolved[field]
    assert resolved == json.loads(json.dumps(question['native_condition_research']))
    assert resolved[0] == original
    assert resolved[1]['exported_models']
    assert feedback['after_observation_results']
    assert feedback['question']['after_observation_results'] == feedback['after_observation_results']
    assert {k: v for k, v in feedback['question'].items() if k != 'native_condition_research'} == {
        k: v for k, v in question.items() if k != 'native_condition_research'}
    # The event carries a reference, while the checkpoint still owns all rounds.
    checkpoint = agent.result('researching')
    saved_question = next(q for q in checkpoint['online_questions'] if q['id'] == COMPOSITION['id'])
    assert json.loads(json.dumps(saved_question['native_condition_research'])) == resolved
    assert checkpoint['events'][-1]['feedback']['question']['native_condition_research'] == reference
    (tmp_path / 'research_context.json').write_text(json.dumps(agent._request(lab)))
    restored, fresh = prepared(tmp_path / 'restored'), PaidConditionalLab()
    restored.restore_research(checkpoint, tmp_path, fresh, confirmation_seed=1709)
    restored_question = next(q for q in restored.questions if q['id'] == COMPOSITION['id'])
    assert json.loads(json.dumps(restored_question['native_condition_research'])) == resolved
    assert all((restored.workspace / item['model']).is_file()
               for round_result in resolved for item in round_result['exported_models'])
    assert fresh.paid == lab.paid and fresh.actions == []
    # The runtime reads the complete canonical recipe after restoration; the
    # smaller returned recipe is only a view, not the update algorithm's input.
    from hegel_machine.cognition import structural_inquiry

    def repeat_search(*args, **kwargs):
        raise AssertionError('Restored boundary research must reuse its complete saved recipe')

    monkeypatch.setattr(structural_inquiry, 'research_native_conditions', repeat_search)
    selected = restored.operator_results[-1]['native_condition_research'][
        'research_decision']['next_experiment']
    continued = execute(restored, fresh, selected['action'], selected['arguments'])
    assert 'after_observation_error' not in continued
    latest = continued['after_observation_results'][0]['native_condition_research']
    assert latest['boundary_update']['status'] == 'boundary_refined'
    assert latest['remaining_boundary']['threshold_interval'] == [0., .125]
    assert fresh.paid == lab.paid + 1 and len(fresh.actions) == 1


def test_practice_outgoing_history_references_only_current_native_banks(tmp_path):
    prior = {'id': 'previous_task', 'status': 'supported',
             'native_condition_research': [{'paid_result': 'original archived trace'}]}
    knowledge = {'files': {}, 'question_traces': [deepcopy(prior)]}
    config = InquiryLLMConfig(observation_budget=300, confirmation_rows=16,
        compute_budget=2_000_000, execution_seconds=60., max_model_calls=0)
    agent = AnonymousPreparedMethodAgent(None, tmp_path, arm='hegel', source_task=False,
        confirmation_seed=1709, knowledge=knowledge, config=config)
    lab = PaidConditionalLab()
    buy_fixture(agent, lab)
    agent.bind_method('''import json
def select_actions(spec, observations, memory):
    state = json.load(open('research_state.json'))
    rounds = [item for item in state['operator_results']
              if item['operator'] == 'native_condition_research']
    if not rounds:
        return {'action':'run', 'arguments':{
            'operator':'native_condition_research', 'max_compute':500000,
            'question':{'id':'compositions','question':'Test measured compositions',
                        'parent_id':'task','depends_on':[]},
            'matched_pairs':[{'reference_record_index':0,'record_index':1}]}}
    if len(rounds) == 1:
        return rounds[-1]['native_condition_research']['research_decision']['next_experiment']
    return {'stop':'Retain the paid boundary research for further study'}
''')
    result = agent.execute_method(lab, max_steps=3)
    assert result['status'] == 'method_stopped'
    current = next(q for q in agent.questions if 'native_condition_research' in q)
    current.update(question='Continue from the measured boundary with the current interpretation',
                   status='inconclusive', answer='Latest interpretation after the saved bank')
    result = agent.result(result['status'])
    outgoing = result['outgoing_knowledge']
    assert result['knowledge_protocol'] == 'method_practice_workspace'
    assert 'files' not in outgoing and outgoing['workspace'] == str(tmp_path)
    assert outgoing['question_traces'][0] == prior
    state = json.loads((tmp_path / 'research_state.json').read_text())
    assert state['questions'][agent.questions.index(current)]['question'] != current['question']
    for index, question in enumerate(result['online_questions']):
        trace = outgoing['question_traces'][1 + index]
        assert {k: v for k, v in trace.items() if k != 'native_condition_research'} == {
            k: v for k, v in question.items() if k != 'native_condition_research'}
        if 'native_condition_research' not in question:
            continue
        reference = trace['native_condition_research']
        assert reference['full_content_file'] == 'research_state.json'
        assert reference['round_count'] == 2
        assert question['native_condition_research'] == reference
        resolved = state
        for part in reference['json_path']:
            resolved = resolved[part]
        assert resolved == json.loads(json.dumps(agent.questions[index]['native_condition_research']))
        assert any(item.get('research_program', {}).get('graph')
                   for round_result in resolved for item in round_result['candidates'])
    (tmp_path / 'research_context.json').write_text(json.dumps(agent._request(lab)))
    restored = AnonymousPreparedMethodAgent(None, tmp_path / 'restored', arm='hegel',
        source_task=False, confirmation_seed=1709, knowledge=knowledge, config=config)
    fresh = PaidConditionalLab()
    restored.restore_research(result, tmp_path, fresh, confirmation_seed=1709)
    assert json.loads(json.dumps(restored.questions)) == json.loads(json.dumps(agent.questions))
    assert fresh.paid == lab.paid and fresh.actions == []

    # A bank that has changed since the saved state must retain its unique data.
    current['native_condition_research'][0]['retained_note'] = 'New information after state was saved'
    latest = agent.result('researching')
    index = agent.questions.index(current)
    expected = deepcopy(current['native_condition_research'])
    assert latest['online_questions'][index]['native_condition_research'] == expected
    assert latest['outgoing_knowledge']['question_traces'][1 + index]['native_condition_research'] == expected
    assert state['questions'][index]['native_condition_research'] != json.loads(json.dumps(expected))

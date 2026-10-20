"""Current executed evidence stays distinct from repeated authored notebooks."""

from copy import deepcopy
import json

from hegel_machine.cognition.code_provider_context import prepare_code_context


def resolve(stored, reference):
    assert reference['full_content_file'] == 'research_context.json'
    for key in reference['json_path']:
        stored = stored[key]
    return stored


def test_current_notebook_occurs_once_beside_actual_scope_and_history_is_retrievable(tmp_path):
    wrong = 'The old discriminator tested rows 2792–2815.'
    scope = {'revision_index': 0, 'record_indices': [2],
             'response_row_ranges': [[72, 95]], 'forecast_evidence_cutoff': {'rows': 72}}
    request = {'files': {'research_notes.md': wrong},
        'session_knowledge': {'files': {'research_notes.md': wrong}},
        'events': [{'step': index, 'action': 'run',
                    'arguments': {'notes': wrong, 'operator': 'analyze_revision'},
                    'research_notes': wrong,
                    'feedback': {'analysis_scope': scope, 'returncode': 0}}
                   for index in range(12)],
        'research_notes': wrong,
        'online_research': {'questions': [], 'recent_revisions': [scope]},
        'evidence_notebook': {'revision:0': scope}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    stored = json.loads((tmp_path / 'research_context.json').read_text())
    assert request == original == stored
    assert list(context)[:2] == ['evidence_notebook', 'research_notes']
    assert json.dumps(context, ensure_ascii=False).count(wrong) == 1
    assert context['evidence_notebook']['revision:0'] == scope
    assert resolve(stored, context['files']['research_notes.md']) == wrong
    assert context['event_history']['preview_start'] == 4
    for index, event in enumerate(context['events'], start=4):
        assert event['arguments']['notes']['json_path'] == ['events', index, 'arguments', 'notes']
        assert resolve(stored, event['arguments']['notes']) == wrong
        assert resolve(stored, event['research_notes']) == wrong
        assert event['feedback']['analysis_scope'] == scope


def test_question_compaction_retains_live_dependencies_and_returned_children(tmp_path):
    questions = [{'id': f'q{index}', 'status': 'open', 'question': f'Prior inquiry {index}',
                  'parent_id': None, 'depends_on': [], 'acquired_evidence': list(range(50))}
                 for index in range(14)]
    questions[2].update(parent_id='q0', depends_on=['q1'])
    questions[5].update(parent_id='q4')
    action = {'kind': 'trajectory', 'sequence': [{'x': 0.0}, {'x': 1.0}], 'channels': []}
    request = {'online_research': {'active_question': 'q2', 'questions': questions,
        'pending_analyses': [{'question_id': 'q6'}],
        'parent_returns': [{'parent_id': 'q4', 'answers': [{'question_id': 'q5', 'status': 'refuted'}]}],
        'recent_paid_records': [{'record_index': 20, 'response_row_range': [72, 73], 'action': action}]}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    online = context['online_research']
    stored = json.loads((tmp_path / 'research_context.json').read_text())
    assert request == original == stored
    for index in (0, 1, 2, 4, 5, 6, 10, 11, 12, 13):
        assert online['questions'][index] == questions[index]
    for index in (3, 7, 8, 9):
        assert 'acquired_evidence' not in online['questions'][index]
        assert resolve(stored, online['questions'][index]) == questions[index]
    compact = online['recent_paid_records'][0]['action']
    assert 'sequence' not in compact
    assert compact['sequence_rows'] == 2 and compact['controls'] == ['x']
    assert resolve(stored, compact) == action


def test_context_budget_references_bulk_history_instead_of_canonical_evidence(tmp_path):
    facts = {f'revision:{index}': {'record_indices': [index],
             'response_row_ranges': [[index * 24, index * 24 + 23]],
             'forecast_results': [{'model': f'model_{index}.py', 'prospective_mse': 1.2}]}
             for index in range(90)}
    request = {'evidence_notebook': facts, 'research_notes': 'Current interpretation.',
               'online_research': {'questions': []},
               'old_research': [{'analysis': 'retained old output ' * 60000}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    assert len(json.dumps(facts)) > 12000
    assert len(json.dumps(context)) <= 1_000_000
    assert context['evidence_notebook'] == facts
    assert context['research_notes'] == request['research_notes']
    assert list(context)[:2] == ['evidence_notebook', 'research_notes']
    stored = json.loads((tmp_path / 'research_context.json').read_text())
    assert stored == original == request
    assert resolve(stored, context['old_research']) == original['old_research']


def test_numeric_bank_and_old_analysis_do_not_displace_current_checked_parent_return(tmp_path):
    program = {'op': 'add', 'children': [{'op': 'const', 'value': index / 100.}
                                        for index in range(6000)]}
    bank = {'kind': 'executed_structural_research', 'question_id': 'structure',
        'status': 'refuted', 'bound_row_count': 6000, 'evidence': list(range(6000)),
        'candidates': [{'program': deepcopy(program), 'status': 'refuted', 'fit_mse': .2,
            'fit_evidence': list(range(6000)), 'paid_fit_residuals': {
                'evaluated_rows': 6000, 'examples': [{'target': 2., 'prediction': 1.,
                                                     'squared_error': 1., 'evidence': [5999]}]}}
            for _ in range(3)]}
    previous = {'question_id': 'structure', 'status': 'refuted', 'answer': deepcopy(bank)}
    latest = {'question_id': 'new_child', 'status': 'refuted', 'evidence': [6000],
        'answer': {'claim': 'The measured contrast rejects the proposed equality.',
                   'local_checks': [{'status': 'violated', 'observed': 1., 'expected': 0.}]}}
    earlier_checks = [{'id': f'contrast_{index}', 'status': 'satisfied', 'value': 0.,
        'expected': 0., 'scope': {'claim': 'The compared rows agree.'},
        'witnesses': [{'row': index, 'target': 0., 'features': {'x': 1.},
                       'history': [{'x': float(offset)} for offset in range(100)]}]}
        for index in range(30)]
    earlier_returns = [{'question_id': 'new_child', 'revision_index': index,
                        'status': 'supported', 'evidence_status': 'supported',
                        'answer': f'Local equality checked in revision {index}.',
                        'check_results': deepcopy(earlier_checks)} for index in range(9, 14)]
    parent = {'id': 'task', 'status': 'reopened',
              'child_answers': [previous, *earlier_returns, latest],
              'reviewed_child_answers': [deepcopy(previous)], 'depends_on': [],
              'executed_analyses': [{'ref': f'analysis:{index}', 'event_index': index}
                                    for index in range(200)]}
    facts = {'relation:structure': bank,
        **{f'analysis:{index}': {'kind': 'executed_python_analysis', 'event_index': index,
            'returncode': 0, 'research_question_id': 'task',
            'stdout': f'Actual old calculation {index}. ' * 60, 'stderr': '',
            'source': {'file': 'research_context.json', 'json_path': ['events', index, 'feedback']}}
           for index in range(200)}}
    facts['analysis:189'].update(research_question_id='new_child',
                                stdout='Actual newest child contrast rejects equality.')
    events = [{'action': 'run', 'arguments': {'path': f'analysis_{index}.py'},
               'feedback': {'returncode': 0, 'stdout': f'Actual result {index}.'}}
              for index in range(200)]
    events[-1]['feedback'].update(question=deepcopy(parent), checked_answer=deepcopy(latest))
    request = {'laboratory_observations': [{'rows': [{'features': {'x': 0.}, 'target': 0.,
                                                    'history': []}], 'row_indices': [0]}],
        'online_research': {'active_question': 'task', 'questions': [parent,
            {'id': 'new_child', 'parent_id': 'task', 'depends_on': [],
             'status': 'refuted', 'answer': deepcopy(latest['answer'])}],
            'pending_analyses': [], 'parent_returns': [{'parent_id': 'task', 'answers': [latest]}]},
        'parent_research_focus': {'question_id': 'task', 'parent_id': 'task',
            'status': 'running', 'phase': 'prediction_method_revision',
            'assignment': {'question': 'Deliver a revised executable predictor.'},
            'returned_research': {'question_id': 'new_child', 'status': 'inquiry_finished'},
            'executions': [{'step': index, 'status': 'completed'} for index in range(30)]},
        'llm_inquiry': {'question_id': 'new_child', 'parent_question_id': 'task',
                        'remaining_model_calls': 4},
        'events': events, 'evidence_notebook': {'facts': facts, 'current_paid_rows': 6001},
        'research_notes': 'Retain analysis:3 and the new child result.'}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    assert len(json.dumps(context)) < 1_000_000
    assert request == original
    stored = json.loads((tmp_path / 'research_context.json').read_text())
    assert stored == original
    assert context['parent_research_focus']['phase'] == 'prediction_method_revision'
    assert context['parent_research_focus']['returned_research'] == {
        'question_id': 'new_child', 'status': 'inquiry_finished'}
    assert context['llm_inquiry'] == request['llm_inquiry']
    online = context['online_research']
    assert online['active_question'] == 'task'
    assert online['parent_returns'][0]['answers'][-1] == latest
    assert online['questions'][0]['child_answers'][-1] == latest
    assert resolve(stored, online['questions'][0]['child_answers'][0]) == previous
    for shown_return, earlier in zip(online['questions'][0]['child_answers'][1:-1], earlier_returns):
        assert shown_return['revision_index'] == earlier['revision_index']
        assert shown_return['evidence_status'] == 'supported'
        assert shown_return['answer'] == earlier['answer']
        assert shown_return['check_results']['status_counts'] == {'satisfied': 30}
        witness = shown_return['check_results']['preview'][0]['witnesses'][0]
        assert witness['target'] == 0. and witness['history_length'] == 100
        assert resolve(stored, witness['history']) == earlier_checks[0]['witnesses'][0]['history']
        assert resolve(stored, shown_return) == earlier
    assert resolve(stored, online['questions'][0]['reviewed_child_answers']) == [previous]
    assert online['questions'][0]['executed_analyses']['count'] == 200
    assert context['events'][-1]['feedback']['checked_answer'] == latest
    assert context['events'][-1]['feedback']['stdout'] == 'Actual result 199.'
    shown = context['evidence_notebook']['facts']
    candidate = shown['relation:structure']['candidates'][0]
    assert candidate['status'] == 'refuted' and candidate['fit_mse'] == .2
    assert candidate['paid_fit_residuals'] == bank['candidates'][0]['paid_fit_residuals']
    assert resolve(stored, candidate['program']) == program
    assert resolve(stored, candidate['fit_evidence']) == list(range(6000))
    assert shown['analysis:3']['stdout'] == facts['analysis:3']['stdout']
    assert shown['analysis:189']['stdout'] == facts['analysis:189']['stdout']
    assert resolve(stored, shown['analysis:5']['stdout']) == facts['analysis:5']['stdout']
    assert 'preview' not in shown['analysis:5']['stdout']
    assert shown['analysis:5']['returncode'] == 0

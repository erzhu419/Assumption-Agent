"""Executed toy reviews check parent handoff, not autonomous scientific ability."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.common_archive_inquiry import (
    CommonArchiveInquiryAgent, FlatScientificRevisionAgent)
from test_common_archive_llm_inquiry import ContextSavingProvider, RestorableConfirmationLab
from test_common_archive_online_research import ScriptedProvider
from test_inquiry_revision import action
from test_scientific_revision_inquiry import PARENT, prepared, prefix


IDENTITY = "def predict(row): return row['features']['u']\n"
VIEW = """def bind(spec, observations, memory):
    return {'rows': [dict(row, evidence=[index]) for record in observations
        for index, row in zip(record['row_indices'], record['rows'])]}
"""
COMPOSED = """def bind_models(models):
    global local
    local = models['local']
def predict(row): return local(row)
"""


def review_state(result):
    question = next(q for q in result['online_questions']
                    if q.get('origin') == 'fresh_confirmation_failure')
    return question['parent_research']


def is_running(request):
    return request.get('parent_research_focus', {}).get('status') == 'running'


@pytest.mark.parametrize('narrowed', [False, True])
def test_checked_refutation_only_closes_its_unchanged_assignment(tmp_path, narrowed):
    question = {**PARENT, **({'question': 'Are these paid responses all zero?'}
                            if narrowed else {})}
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Parent prediction remains unresolved.'}),
        ('run', {'operator': 'analyze_evidence', 'path': 'shared/zero_readout.py',
                 'record_indices': [0, 1], 'research_question': question}),
        ('run', {'code': "print('Interpret the measured refutation')"}),
    ])
    agent = prepared(tmp_path, provider, calls=len(provider.steps))
    if not narrowed:
        provider.steps[2][1]['research_question'] = deepcopy(
            next(q for q in agent.questions if q['id'] == 'task')['assignment'])
    agent._save('shared/zero_readout.py', """def readout(result, spec, memory):
    rows = [i for record in result['observations'] for i in record['row_indices']]
    return {'status': 'refuted', 'answer': 'The tested responses are not all zero.',
            'evidence': rows,
            'checks': [{'id': str(i), 'terms': [{'row': i, 'coefficient': 1.}],
                        'relation': 'eq', 'expected': 0., 'atol': 1e-10}
                       for i in rows]}
""")
    lab = RestorableConfirmationLab()
    prefix(agent, lab)

    result = agent.run(lab)

    state = review_state(result)
    checked = state['executions'][-1]['checked_answer']
    assert checked['evidence_status'] == 'refuted' and checked['check_results']
    assert is_running(provider.requests[3]) is narrowed
    assert state['status'] == ('running' if narrowed else 'refuted')
    assert result['model'] is None and len(agent.confirmations) == 1
    assert result['costs']['observation_cost'] == lab.paid == 3


def test_narrow_refutation_does_not_override_explicit_parent_inconclusive(tmp_path):
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Parent prediction remains unresolved.'}),
        ('run', {'operator': 'analyze_evidence', 'path': 'shared/zero_readout.py',
                 'record_indices': [0, 1], 'research_question': {
                     **PARENT, 'question': 'Are these paid responses all zero?'}}),
        ('finish', {'inconclusive': True, 'lesson': 'No revised predictor is ready.'}),
        ('run', {'code': "print('Keep the explicit unresolved return')"}),
    ])
    agent = prepared(tmp_path, provider, calls=len(provider.steps))
    agent._save('shared/zero_readout.py', """def readout(result, spec, memory):
    row = result['observations'][1]['row_indices'][0]
    return {'status': 'refuted', 'answer': 'The tested response is not zero.',
            'evidence': [row], 'checks': [{'id': 'zero_response',
                'terms': [{'row': row, 'coefficient': 1.}], 'relation': 'eq',
                'expected': 0., 'atol': 1e-10}]}
""")
    lab = RestorableConfirmationLab()
    prefix(agent, lab)

    result = agent.run(lab)

    assert is_running(provider.requests[3])
    assert not is_running(provider.requests[4])
    state = review_state(result)
    assert state['status'] == 'inconclusive'
    assert state['completed_at_model_call'] == 4
    assert state['executions'][-1]['checked_answer']['evidence_status'] == 'refuted'
    assert result['model'] is None and len(agent.confirmations) == 1
    assert result['costs']['observation_cost'] == lab.paid == 3


@pytest.mark.parametrize('agent_type', [CommonArchiveInquiryAgent, FlatScientificRevisionAgent])
def test_parent_focus_requires_executable_synthesis_after_return(tmp_path, agent_type):
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('run', {'code': "print('A local contrast still needs a response model')"}),
        ('finish', {'lesson': 'Return local analysis for parent synthesis.',
                    'artifacts': ['parent.py']}),
        ('query', {'actions': [action(.5)]}),
        ('run', {'code': "print('Additional ordinary analysis')"}),
        ('write', {'path': 'handoff.txt', 'code': 'parent.py'}),
        ('evaluate', {'model': 'parent.py'}),
        ('evaluate', {'component': 'local_response', 'view': 'view.py',
                      'model': 'local.py', 'code': IDENTITY}),
        ('evaluate', {'model': 'composed.py', 'code': COMPOSED,
                      'bindings': {'local': '_observable_models/_conditional/'
                          'local_response/0/_evaluated_model/local.py'}}),
        ('run', {'code': "print('Parent synthesis evaluated; confirmation remains separate')"}),
    ])
    agent = prepared(tmp_path, provider, agent_type=agent_type, calls=len(provider.steps))
    agent._save('parent.py', (agent.workspace / 'parent.py').read_text()
        + "\ndef represent(row): return row['features']['u']\n"
          "def predict_features(value): return predict({'features': {'u': value}})\n")
    agent._save('view.py', VIEW)
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    # Extra purchases, prose, a path, repeating the unchanged parent, and a
    # component-only fit cannot stand in for an actual parent composition.
    assert all(is_running(request) for request in provider.requests[3:9])
    assert is_running(provider.requests[9])
    state = review_state(result)
    assert state['status'] == 'running'
    assert state['phase'] == 'awaiting_independent_confirmation'
    composed = next(event['feedback'] for event in result['events']
                    if event.get('arguments', {}).get('model') == 'composed.py')
    assert composed['mse'] == pytest.approx(0.)
    assert composed['model_dependencies'][0]['calls'] == 4
    assert agent.research_candidates['composed.py']['components']['local']['id'] == 'local_response'
    assert lab.paid == result['costs']['observation_cost'] == 4
    assert len(agent.confirmations) == 1 and result['fresh_confirmation']['passed'] is False
    assert result['model'] is None


def test_running_parent_focus_resumes_without_reopening_review_or_rebuying_rows(tmp_path):
    provider = ContextSavingProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'The local review has not produced a parent predictor.'}),
        ('run', {'code': "print('Pending parent synthesis')"}),
    ])
    old = prepared(tmp_path / 'old', provider, calls=len(provider.steps))
    provider.workspace = old.workspace
    lab = RestorableConfirmationLab()
    prefix(old, lab)
    previous = old.run(lab)
    assert review_state(previous)['status'] == 'running'
    paid = deepcopy(previous['paid_observations'])

    later = ScriptedProvider([
        ('write', {'path': 'parent.py', 'code': IDENTITY}),
        ('evaluate', {'model': 'parent.py'}),
        ('run', {'code': "print('Changed parent was evaluated on inherited counterexamples')"}),
    ])
    restored = prepared(tmp_path / 'resumed', later, calls=6)
    fresh = RestorableConfirmationLab()
    restored.restore_research(previous, old.workspace, fresh, confirmation_seed=1709)
    result = restored.run(fresh)

    assert is_running(later.requests[0]) and is_running(later.requests[1])
    assert is_running(later.requests[2])
    assert review_state(result)['status'] == 'running'
    assert review_state(result)['phase'] == 'awaiting_independent_confirmation'
    assert result['paid_observations'] == paid and fresh.actions == []
    assert result['costs']['observation_cost'] == fresh.paid == 3
    assert result['costs']['model_calls'] == 6
    assert len([q for q in result['online_questions']
                if q.get('origin') == 'fresh_confirmation_failure']) == 1
    assert result['model'] is None and result['fresh_confirmation']['passed'] is False


@pytest.mark.parametrize('agent_type', [CommonArchiveInquiryAgent, FlatScientificRevisionAgent])
def test_local_finish_model_is_evaluated_and_execution_error_stays_local(tmp_path, agent_type):
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'model': 'returned.py', 'code': 'def predict(:\n pass\n'}),
        ('finish', {'model': 'returned.py', 'code': IDENTITY,
                    'lesson': 'Executable local repair; independent confirmation is still due.'}),
        ('evaluate', {'model': 'returned.py'}),
        ('run', {'code': "print('Parent receives a measured local candidate')"}),
    ])
    agent = prepared(tmp_path, provider, agent_type=agent_type, calls=len(provider.steps))
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    # Failed execution cannot be presented as an evaluated local return.
    child = provider.requests[2]
    assert child.get('scientific_revision_assignment')
    if agent_type is CommonArchiveInquiryAgent:
        assert child.get('llm_inquiry')
    else:
        assert child['scientific_revision_review']['organization'] == 'flat_review'
    parent = provider.requests[3]
    assert not parent.get('llm_inquiry')
    assert is_running(parent)
    assert is_running(provider.requests[4])
    assert review_state(result)['status'] == 'running'
    assert review_state(result)['phase'] == 'awaiting_independent_confirmation'
    assert result['evaluated_model']['mse'] == pytest.approx(0.)
    assert len(agent.confirmations) == 1 and result['fresh_confirmation']['passed'] is False
    assert lab.paid == result['costs']['observation_cost'] == 3
    assert result['model'] is None


def test_explicit_inconclusive_returns_control_without_submission_or_false_success(tmp_path):
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Local study still unresolved.'}),
        ('finish', {'inconclusive': True, 'lesson': 'No supported parent explanation yet.'}),
        ('run', {'code': "print('Ordinary research can continue after the explicit decision')"}),
    ])
    agent = prepared(tmp_path, provider, calls=len(provider.steps))
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    assert is_running(provider.requests[2])
    assert not is_running(provider.requests[3])
    assert review_state(result)['status'] == 'inconclusive'
    assert len(agent.confirmations) == 1 and result['fresh_confirmation']['passed'] is False
    assert result['model'] is None and result['status'] == 'model_call_budget_exhausted'
    assert lab.paid == result['costs']['observation_cost'] == 3


def test_first_paid_parent_run_recovers_focus_question_and_single_accounting(tmp_path, monkeypatch):
    from hegel_machine.cognition.interrupted_online_inquiry import recover_interrupted_online_run

    code = "print('Combine the returned local evidence into a parent explanation')"

    class PaidReplyProvider(ContextSavingProvider):
        def next_action(self, request):
            decision = super().next_action(request)
            if decision['action'] == 'run':
                saved = json.loads((self.workspace / 'active.json').read_text())
                assert saved['costs']['model_calls'] == 2
                assert review_state(saved)['status'] == 'running'
                assert not saved['inquiry_stack']
                (self.workspace / 'code_action_003.json').write_text(json.dumps({
                    'action': 'run', 'arguments_json': '{}', 'source_code': code,
                    'research_notes': 'Parent synthesis calculation.',
                    'research_question': None}))
                (self.workspace / 'code_call_003.jsonl').write_text(json.dumps({
                    'type': 'turn.completed', 'usage': {'input_tokens': 1,
                        'output_tokens': 0, 'cached_input_tokens': 0}}) + '\n')
            return decision

    provider = PaidReplyProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Return unresolved local evidence to the parent.'}),
        ('run', {'code': code}),
    ])
    old = prepared(tmp_path / 'old', provider, calls=5)
    provider.workspace = old.workspace
    lab = RestorableConfirmationLab()
    prefix(old, lab)
    checkpoints = []

    def save(result):
        checkpoints.append(deepcopy(result))
        (old.workspace / 'active.json').write_text(json.dumps(result))

    original_python = old._python

    def interrupted(command, *args, **kwargs):
        if len(command) > 1 and Path(command[1]).name.startswith('scratch_'):
            raise RuntimeError('Worker stopped during paid parent synthesis')
        return original_python(command, *args, **kwargs)

    monkeypatch.setattr(old, '_python', interrupted)
    with pytest.raises(RuntimeError, match='paid parent synthesis'):
        old.run(lab, on_checkpoint=save)
    previous = recover_interrupted_online_run(old.workspace, model='scripted',
        interruption={'executed_path': 'scratch_003.py', 'interrupted_python_seconds': 0.1},
        result_path=old.workspace / 'interrupted_training.json')
    assert review_state(previous)['status'] == 'running'
    assert previous['events'][-1]['feedback']['research_question_id'] == 'task'
    assert previous['costs']['model_calls'] == previous['costs']['reported_tokens'] == 3
    assert previous['costs']['python_execution_seconds'] == pytest.approx(
        checkpoints[-1]['costs']['python_execution_seconds'] + 0.1)

    continuation = ScriptedProvider([
        ('write', {'path': 'parent.py', 'code': IDENTITY}),
        ('evaluate', {'model': 'parent.py'}),
    ])
    resumed = prepared(tmp_path / 'resumed', continuation, calls=5)
    fresh = RestorableConfirmationLab()
    resumed.restore_research(previous, old.workspace, fresh, confirmation_seed=1709)
    result = resumed.run(fresh)
    assert is_running(continuation.requests[0])
    parent = next(q for q in continuation.requests[0]['online_research']['questions']
                  if q['id'] == 'task')
    assert any(row['event_index'] == 3 for row in parent['executed_analyses'])
    assert review_state(result)['status'] == 'running'
    assert result['paid_observations'] == previous['paid_observations']
    assert fresh.actions == [] and fresh.paid == result['costs']['observation_cost'] == 3
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 5
    assert sum(event['feedback'].get('status') == 'runtime_interrupted'
               for event in result['events']) == 1
    assert result['model'] is None and result['fresh_confirmation']['passed'] is False


@pytest.mark.parametrize('exposed', [False, True])
def test_changed_high_residual_parent_stays_under_research(tmp_path, exposed):
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'The child has returned evidence, not a solution.'}),
        ('evaluate', {'model': 'revision.py', 'code': (
            "def represent(row): return row['features']['u']\n"
            "def predict_features(value): return .5 * value\n" if exposed else
            "def predict(row): return .5 * row['features']['u']\n")}),
    ])
    agent = prepared(tmp_path, provider, calls=3)
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    state = review_state(result)
    assert state['status'] == 'running'
    assert state['phase'] == ('candidate_revision' if exposed else 'representation_unassessed')
    assert state['executions'][-1]['source_changed'] is True
    assert state['executions'][-1]['mse'] > agent.config.confirmation_mse_threshold
    assert result['model'] is None and len(agent.confirmations) == 1


def test_low_paid_error_only_awaits_then_fresh_confirmation_closes_parent(tmp_path):
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Return the failed predictor for actual parent revision.'}),
        ('evaluate', {'model': 'revision.py', 'code': IDENTITY}),
        ('finish', {'model': 'revision.py'}),
    ])
    agent = prepared(tmp_path, provider, calls=4)
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    assert provider.requests[3]['parent_research_focus']['phase'] == 'awaiting_independent_confirmation'
    assert is_running(provider.requests[3])
    assert review_state(result)['status'] == 'independent_confirmation_passed'
    assert result['fresh_confirmation']['passed'] is True
    assert len(agent.confirmations) == 2 and lab.paid == 5


@pytest.mark.parametrize('outcome', ['supported', 'inadequate', 'execution_failed'])
def test_parent_refit_tracks_selected_source_and_separate_holdout_result(tmp_path, outcome):
    fit = ("raise ValueError('authored fitting failure')" if outcome == 'execution_failed'
           else "assert len(targets) == 1, 'validation labels entered fitting'")
    factor = 1. if outcome == 'supported' else .5
    source = ("def represent(row): return row['features']['u']\n"
              f"def fit_features(features, targets): {fit}\n"
              f"def predict_features(value): return {factor} * value\n")
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Return the failed predictor for parent research.'}),
        ('evaluate', {'model': 'parent.py'}),
        ('write', {'path': 'revision.py', 'code': source}),
        ('run', {'operator': 'refit', 'model': 'revision.py', 'path': 'shared/heldout',
                 'validation_records': [1]}),
        ('run', {'code': "print('Interpret the separate trajectory validation')"}),
        ('finish', {'inconclusive': True, 'lesson': 'Leave independent prediction unresolved.'}),
        ('run', {'operator': 'refit', 'model': 'revision.py', 'path': 'shared/after_return',
                 'validation_records': [1]}),
        ('run', {'code': "print('The explicitly returned parent stays unresolved')"}),
    ])
    agent = prepared(tmp_path, provider, calls=len(provider.steps))
    agent._save('parent.py', (agent.workspace / 'parent.py').read_text()
        + "\ndef represent(row): return row['features']['u']\n"
          "def predict_features(value): return predict({'features': {'u': value}})\n")
    lab = RestorableConfirmationLab()
    prefix(agent, lab)

    result = agent.run(lab)

    request = provider.requests[5]
    focus = request['parent_research_focus']
    assert is_running(request) and focus['evaluated_model'] == 'revision.py'
    review = focus['method_review_result']
    refit = agent.operator_results[-2]
    assert review['source_model'] == 'revision.py'
    assert review['tested_model'] == refit['tested_model'] != 'revision.py'
    assert review['train_rows'] == 1 and review['holdout_rows'] == 2
    assert review['validation_records'] == [1] and review['paid_rows_at_refit'] == 3
    assert review['status'] == refit['status'] == 'refit_' + (
        'supported_on_validation' if outcome == 'supported' else
        'inadequate_on_validation' if outcome == 'inadequate' else 'execution_failed')
    assert review['validation_mse'] == refit['validation'].get('mse')
    assert focus['executions'][-1]['operator'] == 'refit'
    assert focus['executions'][-1]['evaluation_scope'] == 'paid_trajectory_holdout'
    assert focus['phase'] == ('awaiting_independent_confirmation' if outcome == 'supported'
                               else 'candidate_revision')
    evaluation = agent.research_candidates['revision.py']['current_evaluation']
    assert evaluation['evaluation_scope'] == 'paid_trajectory_holdout'
    if outcome != 'execution_failed':
        assert evaluation['mse'] == review['validation_mse']
        assert refit['validation']['training_representation_diagnostic']['rows'] == 1
        assert refit['validation']['representation_diagnostic']['rows'] == 2
    assert refit['final_fitting_rows'] == 3
    assert not is_running(provider.requests[7]) and not is_running(provider.requests[8])
    state = review_state(result)
    assert state['status'] == 'inconclusive' and state['completed_at_model_call'] == 7
    assert len(state['executions']) == 2  # Old evaluation and the active refit only.
    assert result['model'] is None and len(agent.confirmations) == 1
    assert lab.paid == 3 and result['costs']['observation_cost'] == 3


def test_saved_parent_evaluated_reopens_with_stale_evaluation_and_keeps_executions(tmp_path):
    provider = ContextSavingProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Return for parent revision.'}),
        ('evaluate', {'model': 'revision.py', 'code': IDENTITY}),
        ('query', {'actions': [action(.5)]}),
    ])
    old = prepared(tmp_path / 'old', provider, calls=4)
    provider.workspace = old.workspace
    lab = RestorableConfirmationLab()
    prefix(old, lab)
    previous = old.run(lab)
    # Reproduce the persisted delivery state written by the previous runtime.
    state = review_state(previous)
    state['status'] = 'parent_evaluated'
    state['completed_at_model_call'] = 3
    state.pop('phase')
    executions = deepcopy(state['executions'])
    paid = deepcopy(previous['paid_observations'])

    later = ScriptedProvider([('run', {'code': "print('Continue the existing unpaid obligation')"})])
    restored = prepared(tmp_path / 'restored', later, calls=5)
    fresh = RestorableConfirmationLab()
    restored.restore_research(previous, old.workspace, fresh, confirmation_seed=1709)
    result = restored.run(fresh)

    focus = later.requests[0]['parent_research_focus']
    assert focus['status'] == 'running' and focus['phase'] == 'representation_unassessed'
    assert focus['evaluation_current'] is False
    assert focus['candidate_paid_rows'] == 3 and focus['current_paid_rows'] == 4
    assert review_state(result)['executions'] == executions
    assert result['paid_observations'] == paid and fresh.actions == []
    question = next(q for q in result['online_questions'] if q.get('parent_research'))
    assert question['parent_research_history'][-1]['status'] == 'parent_evaluated'


def test_computed_representation_conflict_remains_active_until_actual_repair(tmp_path):
    provider = ScriptedProvider([
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'The unexplained rows require parent revision.'}),
        ('evaluate', {'model': 'alias.py', 'code':
            'def represent(row): return 0.\ndef predict_features(value): return 0.\n'}),
        ('run', {'code': "print('Inspect the actual collision witnesses')"}),
        ('evaluate', {'model': 'alias.py', 'code':
            "def represent(row): return row['features']['u']\n"
            'def predict_features(value): return value\n'}),
        ('finish', {'model': 'alias.py'}),
    ])
    agent = prepared(tmp_path, provider, calls=6)
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    for request in provider.requests[3:5]:
        assert request['parent_research_focus']['phase'] == 'representation_conflict'
        assert request['research_dependency_focus']['question']['id'] == 'representation:alias.py'
    inspected = next(event for event in result['events']
                     if 'Inspect the actual collision' in event.get('arguments', {}).get('code', ''))
    assert inspected['feedback']['research_question_id'] == 'representation:alias.py'
    assert provider.requests[5]['parent_research_focus']['phase'] == 'awaiting_independent_confirmation'
    question = next(q for q in result['online_questions'] if q['id'] == 'representation:alias.py')
    assert question['status'] == 'resolved_on_paid_rows'
    assert review_state(result)['status'] == 'independent_confirmation_passed'

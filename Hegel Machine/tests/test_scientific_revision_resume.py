"""Resume the selected paid source without replaying a different arm or its costs."""

from copy import deepcopy
from dataclasses import asdict
import json
import shutil

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts import run_scientific_revision_comparison as runner
from test_common_archive_resume import checkpoint, SEED
from test_common_archive_online_research import ScriptedProvider
from test_common_archive_resume import execute


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_resume_selected_paid_arm_keeps_registered_case_and_costs(tmp_path, monkeypatch):
    old, lab, previous = checkpoint(tmp_path)
    source = tmp_path / 'completed_pair'
    case = {'id': 'case', 'kind': 'fixture', 'confirmation_seed': SEED}
    workspace = source / 'online' / case['id'] / 'recursive'
    # The run fixture uses the real old workspace; protocol assertions below
    # first establish the correct on-disk selection for this cohort layout.
    write(source / 'protocol.json', {'online': {'model': 'gpt-6-luna',
        'cases': [case], 'arms': ['recursive', 'flat'], 'runtime_snapshot': [],
        'limits': asdict(InquiryLLMConfig()), 'provider': {}},
        'initial_inquiry': {'question': {'id': 'already_delegated'}, 'max_model_calls': 8},
        'next_stage': 'No automatic method compilation or target rerun.'})
    write(source / 'summary.json', {'status': 'completed', 'trials': [
        {'case': 'case', 'arm': 'recursive', 'model': None,
         'fresh_confirmation_passed': False, 'costs': previous['costs']},
        {'case': 'case', 'arm': 'flat', 'model': None,
         'fresh_confirmation_passed': False, 'costs': {'model_calls': 999}}]})
    write(source / 'online' / 'initial_archive.json', old.knowledge)
    plan = runner.protocol(source, resume_arm='recursive')
    assert 'initial_inquiry' not in plan
    assert plan['resume_result'] == str(workspace / 'training.json')
    assert plan['online']['cases'] == [case] and plan['online']['arms'] == ['recursive']
    assert plan['online']['limits']['max_model_calls'] == previous['costs']['model_calls'] + 48
    assert plan['online']['limits']['token_budget'] == previous['costs']['reported_tokens'] + 12_000_000
    assert plan['online']['limits']['observation_budget'] == previous['costs']['observation_cost'] + 4096
    recovered = workspace / 'interrupted_training.json'
    recovered_plan = runner.protocol(source, resume_arm='recursive', resume_result=recovered)
    assert recovered_plan['resume_result'] == str(recovered)
    assert recovered_plan['resume_workspace'] == plan['resume_workspace']
    assert recovered_plan['online']['limits'] == plan['online']['limits']
    latest = deepcopy(previous)
    latest['costs']['model_calls'] += 28
    latest['costs']['reported_tokens'] += 500
    latest_plan = runner.protocol(source, resume_arm='recursive', resume_result=recovered,
                                  previous=latest)
    assert latest_plan['online']['limits']['max_model_calls'] == latest['costs']['model_calls'] + 48
    assert latest_plan['online']['limits']['token_budget'] == latest['costs']['reported_tokens'] + 12_000_000

    write(old.workspace / 'training.json', previous)
    plan.update(resume_workspace=str(old.workspace), resume_result=str(old.workspace / 'training.json'))
    plan['online']['runtime_snapshot'] = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Restoring this paid source must not call the model or buy observations')

    def inspect(part, output, *, knowledge, prepare_prefix, agent_factory, **kwargs):
        agent = agent_factory(None, output / 'recursive', arm='recursive', knowledge=knowledge,
                              source_task=False, config=InquiryLLMConfig(**part['limits']))
        current = TemporalInquiryLab(task_index=1)
        monkeypatch.setattr(current, 'execute', forbidden)
        extra = prepare_prefix(agent, current, case)
        assert isinstance(agent, CommonArchiveInquiryAgent) and agent.arm == 'hegel'
        assert agent.costs == previous['costs'] and current.costs == lab.costs
        assert agent.rows == old.rows and agent.events == json.loads(json.dumps(previous['events']))
        assert agent.confirmations == json.loads(json.dumps(old.confirmations))
        assert extra['prior_research_cost'] == previous['costs']
        assert agent.knowledge == old.knowledge
        return {'status': 'restored_fixture', 'trials': []}

    monkeypatch.setattr(runner, 'run_study', inspect)
    assert runner.run_comparison(plan, tmp_path / 'resumed', provider_factory=forbidden)['status'] == 'restored_fixture'


@pytest.mark.parametrize('status', ['open', 'supported', 'refuted'])
def test_unsummarized_active_restores_actual_account_and_same_question(tmp_path, monkeypatch, status):
    old, lab, _ = checkpoint(tmp_path)
    old.provider = ScriptedProvider([])
    old._start_inquiry({'question': {'id': 'new_focus', 'parent_id': 'task'},
                        'max_model_calls': 2}, parent_event_index=len(old.events) - 1)
    execute(old, lab, 'write', {'path': 'local_draft.py', 'code': '# Authored local work\n',
                              'notes': 'Paid local findings retained for the next invocation.'})
    old._execute('finish', {'lesson': 'The current local explanation remains unresolved.'}, lab)
    old._settle_inquiries()
    assert not old._inquiry_stack
    # The paid author can revise hypothesis metadata; explicit resumption still
    # returns to the original caller recorded by the question's assignment.
    question = next(q for q in old.questions if q['id'] == 'new_focus')
    question.update(parent_id=None, status=status)
    old.costs['model_calls'] = 649
    old.costs['reported_tokens'] = 139_960_786
    old._prepare_program_state()
    write(old.workspace / 'research_context.json', old._request(lab))
    previous = old.result('token_budget_exhausted')
    source = tmp_path / 'unsummarized'
    case = {'id': 'case', 'kind': 'fixture', 'confirmation_seed': SEED}
    workspace = source / 'online' / 'case' / 'recursive'
    workspace.parent.mkdir(parents=True)
    shutil.copytree(old.workspace, workspace)
    write(source / 'protocol.json', {'online': {'model': 'gpt-6-luna',
        'cases': [case], 'arms': ['recursive'], 'runtime_snapshot': [],
        'limits': asdict(InquiryLLMConfig(max_model_calls=621)), 'provider': {}},
        'next_stage': 'No automatic method compilation or target rerun.'})
    write(source / 'online' / 'initial_archive.json', old.knowledge)
    active = workspace / 'active.json'
    write(active, previous)
    plan = runner.protocol(source, resume_arm='recursive', resume_result=active,
                           previous=previous, resume_question='new_focus')
    assert plan['resume_workspace'] == str(workspace)
    assert plan['resume_result'] == str(active)
    assert plan['online']['cases'] == [case] and plan['online']['arms'] == ['recursive']
    for limit, cost in runner.LIMIT_COSTS:
        assert plan['online']['limits'][limit] == previous['costs'][cost] + plan['online_allowance'][limit]
    assert plan['online']['limits']['max_model_calls'] == 697
    assert plan['online']['limits']['token_budget'] == 151_960_786
    plan['online']['runtime_snapshot'] = []
    original_question = next(q for q in previous['online_questions'] if q['id'] == 'new_focus')
    session = previous['llm_inquiries']['new_focus']
    real_read = runner.read_json

    def read_without_reloading_active(path):
        assert path != str(active), 'Reuse the already loaded paid checkpoint'
        return real_read(path)

    def forbidden(*args, **kwargs):
        raise AssertionError('Restoration must not call the model or buy observations')

    def inspect(part, output, *, knowledge, prepare_prefix, agent_factory, **kwargs):
        provider = ScriptedProvider([])
        agent = agent_factory(provider, output / 'recursive', arm='recursive',
            knowledge=knowledge, source_task=False, config=InquiryLLMConfig(**part['limits']))
        current = TemporalInquiryLab(task_index=1)
        monkeypatch.setattr(current, 'execute', forbidden)
        extra = prepare_prefix(agent, current, case)
        assert agent.costs == previous['costs'] and agent.rows == old.rows
        assert agent.events == previous['events'] and current.costs == lab.costs
        assert provider.calls == 649 and provider.requests == []
        assert extra['prior_research_cost'] == previous['costs']
        frame = agent._inquiry_stack[-1]
        assert frame['question_id'] == 'new_focus' and frame['call_limit'] == 661
        for key in ('question', 'depends_on'):
            assert frame['assignment'][key] == original_question[key]
        assert frame['assignment']['parent_id'] == original_question['assignment']['parent_id'] == 'task'
        assert frame['parent_question'] == 'task'
        resumed = agent.llm_inquiries['new_focus']
        assert resumed['notes'] == session['notes']
        assert resumed['event_indices'] == session['event_indices']
        assert resumed['invocations'] == session['invocations'] + 1
        assert (agent.workspace / 'local_draft.py').read_text() == '# Authored local work\n'
        restoration = real_read(agent.workspace / 'restoration.json')
        assert restoration['host_scheduled_resume'] == {
            'question_id': 'new_focus', 'parent_id': 'task',
            'max_model_calls': 12, 'status': 'inquiry_started'}
        assert restoration['new_observations'] == 0 and restoration['restored_model_calls'] == 649
        agent._before_model_turn(current)
        assert agent._inquiry_stack[-1]['question_id'] == 'new_focus'
        request = agent._request(current)
        assert request['llm_inquiry']['question_id'] == 'new_focus'
        assert request['llm_inquiry']['question']['status'] == (
            'open' if status == 'open' else 'reopened')
        assert agent.costs == previous['costs'] and provider.requests == []
        return {'status': 'restored_fixture', 'trials': []}

    monkeypatch.setattr(runner, 'read_json', read_without_reloading_active)
    monkeypatch.setattr(runner, 'run_study', inspect)
    result = runner.run_comparison(plan, tmp_path / 'continued', provider_factory=forbidden,
                                   previous=previous)
    assert result['status'] == 'restored_fixture'
    assert not (source / 'summary.json').exists()


def test_cli_parses_explicit_checkpoint_once_and_reuses_loaded_result(tmp_path, monkeypatch):
    active = tmp_path / 'active.json'
    write(active, {'model': None, 'costs': {'model_calls': 649}})
    parsed = []
    captured = []
    real_read = runner.read_json

    def read(path):
        result = real_read(path)
        parsed.append((path, result))
        return result

    def plan(source, **kwargs):
        assert kwargs['previous'] is parsed[0][1]
        assert kwargs['resume_question'] == 'existing_local_question'
        captured.append(kwargs['previous'])
        return {'online': {'model': 'gpt-6-luna', 'provider': {
            'reasoning_effort': 'high', 'timeout_seconds': 1800.}}}

    def run(plan, output, *, previous, **kwargs):
        assert previous is captured[0]
        return {'status': 'restored_fixture', 'trials': []}

    monkeypatch.setattr(runner, 'read_json', read)
    monkeypatch.setattr(runner, 'protocol', plan)
    monkeypatch.setattr(runner, 'run_comparison', run)
    monkeypatch.setattr(runner, 'verify_codex_subscription', lambda: 'ChatGPT_subscription')
    monkeypatch.setattr(runner.sys, 'argv', ['run_scientific_revision_comparison.py',
        '--source-study', str(tmp_path), '--output', str(tmp_path / 'continued'),
        '--resume-arm', 'recursive', '--resume-result', str(active),
        '--resume-question', 'existing_local_question'])
    runner.main()
    assert len(parsed) == 1 and parsed[0][0] == active

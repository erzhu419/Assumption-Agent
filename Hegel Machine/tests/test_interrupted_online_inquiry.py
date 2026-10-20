"""Interrupted executions retain paid decisions without rerunning science."""

from copy import deepcopy
import json
import os

import pytest

from hegel_machine.cognition.interrupted_online_inquiry import (
    recover_completed_online_query, recover_interrupted_online_run,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def stopped_run(tmp_path):
    source = tmp_path / 'online/case/hegel'
    costs = {'model_calls': 7, 'reported_tokens': 1000, 'python_execution_seconds': 12.5,
             'observation_cost': 24, 'measurement_cost': 0, 'compute_cost': 36}
    previous = {'model': None, 'status': 'researching', 'workspace': str(source),
        'events': [{'step': -1, 'action': 'query'}, {'step': 6, 'action': 'run'}],
        'costs': costs, 'online_questions': [{'id': 'task', 'question': 'Explain the data',
                                            'status': 'open'}],
        'paid_observations': [{'row_indices': [0, 1]}],
        'fresh_confirmation': {'passed': False, 'mse': 2.5},
        'outgoing_knowledge': {'files': {'research_notes.md': 'old interpretation',
                                       'model.py': 'old model'}, 'manifest': {}}}
    write_json(source / 'active.json', previous)
    os.utime(source / 'active.json', (100, 100))
    question = {'id': 'child', 'question': 'Does this representation fit?',
                'parent_id': 'task', 'depends_on': []}
    payload = {'action': 'run', 'arguments_json': '{}', 'research_question': question,
        'source_code': "open('must_not_execute.txt','w').write('executed')",
        'research_notes': 'authored interpretation citing analysis:1'}
    write_json(source / 'code_action_008.json', payload)
    (source / 'code_call_008.jsonl').write_text(json.dumps({'type': 'turn.completed',
        'usage': {'input_tokens': 100, 'output_tokens': 25, 'cached_input_tokens': 80}}) + '\n')
    (source / 'research_notes.md').write_text(payload['research_notes'])
    (source / 'scratch_002.py').write_text(payload['source_code'])
    (source / 'model.py').write_text('new partial model')
    (source / 'dependency.py').write_text('written_before_interrupt = True\n')
    partial = tmp_path / 'interrupted_stdout.txt'
    partial.write_text('first candidate error 1.5\n')
    stop = {'reason': 'repair known runtime defects', 'status': 'stopped_for_repair',
        'executed_path': 'scratch_002.py', 'interrupted_python_seconds': 7.25,
        'partial_outputs': {'stdout': {'path': str(partial), 'bytes': partial.stat().st_size}}}
    return source, previous, payload, stop


def recover(source, stop):
    return recover_interrupted_online_run(source, model='gpt-6-luna', interruption=stop,
                                          result_path=source / 'interrupted_training.json')


def test_paid_call_elapsed_notes_and_files_survive_without_replaying(stopped_run):
    source, previous, payload, stop = stopped_run
    result = recover(source, stop)

    assert result['costs'] == {**previous['costs'], 'model_calls': 8,
        'reported_tokens': 1125, 'python_execution_seconds': 19.75}
    assert result['paid_observations'] == previous['paid_observations']
    assert result['fresh_confirmation'] == previous['fresh_confirmation']
    assert result['model'] is None and result['status'] == 'runtime_interrupted'
    assert result['events'][:-1] == previous['events']
    event = result['events'][-1]
    assert event['step'] == 7 and event['usage']['total_tokens'] == 125
    assert event['usage']['cached_input_tokens'] == 80
    assert event['arguments']['code'] == payload['source_code']
    assert event['arguments']['notes'] == payload['research_notes']
    assert event['feedback']['status'] == 'runtime_interrupted'
    assert event['feedback']['executed_path'] == 'scratch_002.py'
    assert event['feedback']['stdout'] == 'first candidate error 1.5\n'
    assert 'mse' not in event['feedback'] and 'passed' not in event['feedback']
    files = result['outgoing_knowledge']['files']
    assert files['research_notes.md'] == payload['research_notes']
    assert files['model.py'] == 'new partial model'
    assert files['dependency.py'] == 'written_before_interrupt = True\n'
    assert not any(name.startswith(('code_action_', 'code_call_')) for name in files)
    assert 'active.json' not in files and 'interrupted_training.json' not in files
    assert not (source / 'must_not_execute.txt').exists()
    assert json.loads((source / 'active.json').read_text()) == previous
    assert json.loads((source / 'code_action_008.json').read_text()) == payload
    assert json.loads((source / 'interrupted_training.json').read_text()) == result


def test_pending_question_is_registered_without_becoming_supported(stopped_run):
    source, _, payload, stop = stopped_run
    result = recover(source, stop)
    question = result['online_questions'][-1]
    assert question['assignment'] == payload['research_question']
    assert question['status'] == 'open'
    assert question['evidence'] == question['acquired_evidence'] == []
    assert question['opened_before_observation'] == 24
    assert result['events'][-1]['feedback']['research_question_id'] == 'child'


def test_recovery_does_not_inline_paid_history_or_candidate_snapshot_trees(stopped_run):
    source, _, _, stop = stopped_run
    write_json(source / 'observations.json', [{'target': 1.0}])
    write_json(source / 'research_trace.json', {'events': ['large history']})
    write_json(source / '_evaluated_model/snapshot/observations.json', [{'target': 1.0}])
    (source / '_evaluated_model/snapshot/nested.py').write_text('old copied program')
    shared = source / 'shared'
    shared.mkdir()
    (shared / 'new_model.py').write_text('new learned dependency')
    write_json(shared / 'coefficients.json', [1.0, 2.0])

    files = recover(source, stop)['outgoing_knowledge']['files']

    assert 'observations.json' not in files and 'research_trace.json' not in files
    assert not any(name.startswith('_evaluated_model/') for name in files)
    assert files['shared/new_model.py'] == 'new learned dependency'
    assert json.loads(files['shared/coefficients.json']) == [1.0, 2.0]
    assert (source / '_evaluated_model/snapshot/observations.json').exists()


def test_existing_question_and_repeated_export_preserve_account_once(stopped_run):
    source, previous, payload, stop = stopped_run
    previous['online_questions'].append({**payload['research_question'], 'status': 'reopened',
        'child_answers': [{'question_id': 'deeper', 'status': 'refuted'}]})
    write_json(source / 'active.json', previous)
    result = recover(source, stop)
    assert len(result['online_questions']) == 2
    assert result['online_questions'][-1]['child_answers'] == [{'question_id': 'deeper', 'status': 'refuted'}]
    assert result['online_questions'][-1]['status'] == 'reopened'
    assert recover(source, stop) == result


def test_explicit_path_uses_pending_call_not_scratch_number(stopped_run):
    source, _, payload, stop = stopped_run
    payload['arguments_json'] = json.dumps({'path': 'analysis.py'})
    payload['source_code'] = None
    write_json(source / 'code_action_008.json', payload)
    (source / 'analysis.py').write_text('print(1)\n')
    result = recover(source, {**stop, 'executed_path': 'analysis.py'})
    assert result['events'][-1]['feedback']['executed_path'] == 'analysis.py'
    assert 'code' not in result['events'][-1]['arguments']
    assert result['runtime_interruption']['saved_reply'] == 'code_action_008.json'


def test_response_evaluation_retains_paid_reply_and_scoring_charge_once(stopped_run):
    source, previous, payload, stop = stopped_run
    previous['paid_observations'].append({'row_indices': [2, 3, 4]})
    failed = {'passed': False, 'mse': 2.5, 'observation_indices': [2, 3, 4]}
    previous['events'][-1]['feedback'] = {'fresh_confirmation': failed}
    previous['evaluated_model'] = {'model': '_evaluated_model/old.py', 'mse': 0.0,
                                   'evaluated_rows': 2}
    previous['research_candidates'] = {'shared/revised.py': {'status': 'needs_refit'}}
    write_json(source / 'active.json', previous)
    original_checkpoint = (source / 'active.json').read_bytes()
    payload.update(action='evaluate', arguments_json='{"model":"shared/revised.py"}',
                   source_code=None)
    write_json(source / 'code_action_008.json', payload)
    shared = source / 'shared'
    shared.mkdir()
    revised = shared / 'revised.py'
    revised.write_text("from pathlib import Path\nPath('must_not_execute.txt').touch()\n")
    snapshot = source / '_evaluated_model/old.py'
    snapshot.parent.mkdir()
    snapshot.write_text('old_supported_predictor = True\n')
    interruption = {**stop, 'executed_path': 'shared/revised.py'}

    result = recover(source, interruption)

    assert result['costs'] == {**previous['costs'], 'model_calls': 8,
        'reported_tokens': 1125, 'python_execution_seconds': 19.75, 'compute_cost': 41}
    event = result['events'][-1]
    assert event['action'] == 'evaluate'
    assert event['arguments']['model'] == 'shared/revised.py'
    assert 'code' not in event['arguments']
    assert event['arguments']['notes'] == payload['research_notes']
    assert event['usage']['cached_input_tokens'] == 80
    assert event['feedback']['executed_path'] == 'shared/revised.py'
    assert event['feedback']['research_question_id'] == 'child'
    assert event['feedback']['scoring_rows'] == event['feedback']['compute_cost'] == 5
    assert not {'mse', 'passed', 'returncode', 'evaluation_scope'} & event['feedback'].keys()
    assert result['events'][:-1] == previous['events']
    assert result['paid_observations'] == previous['paid_observations']
    assert result['fresh_confirmation'] == previous['fresh_confirmation']
    assert result['evaluated_model'] == previous['evaluated_model']
    assert result['research_candidates'] == previous['research_candidates']
    assert result['outgoing_knowledge']['files']['shared/revised.py'] == revised.read_text()
    assert snapshot.read_text() == 'old_supported_predictor = True\n'
    assert not (source / 'must_not_execute.txt').exists()
    assert (source / 'active.json').read_bytes() == original_checkpoint
    assert json.loads((source / 'code_action_008.json').read_text()) == payload
    assert recover(source, interruption) == result


@pytest.mark.parametrize('action,arguments', [
    ('finish', {'model': 'model.py'}),
    ('evaluate', {'model': 'model.py', 'component': 'local_relation'}),
    ('evaluate', {'model': 'model.py', 'target': 'instrument'}),
    ('evaluate', {'operator': 'refit', 'model': 'model.py'}),
])
def test_only_response_evaluation_adds_a_recoverable_boundary(stopped_run, action, arguments):
    source, previous, payload, stop = stopped_run
    payload.update(action=action, arguments_json=json.dumps(arguments), source_code=None)
    write_json(source / 'code_action_008.json', payload)
    with pytest.raises(ValueError, match='ordinary Python run or response evaluation'):
        recover(source, {**stop, 'executed_path': 'model.py'})
    assert not (source / 'interrupted_training.json').exists()
    assert json.loads((source / 'active.json').read_text()) == previous


def test_response_evaluation_requires_the_pending_model_path(stopped_run):
    source, _, payload, stop = stopped_run
    payload.update(action='evaluate', arguments_json='{"model":"shared/revised.py"}',
                   source_code=None)
    write_json(source / 'code_action_008.json', payload)
    with pytest.raises(ValueError, match='does not match'):
        recover(source, {**stop, 'executed_path': 'another.py'})
    assert not (source / 'interrupted_training.json').exists()


@pytest.mark.parametrize('change,match', [
    ('operator', 'ordinary Python run'), ('wrong_worker', 'does not match'),
    ('submitted', 'submitted model'), ('missing_usage', 'completed usage record'),
])
def test_wrong_recovery_boundary_does_not_publish_result(stopped_run, change, match):
    source, previous, payload, stop = stopped_run
    if change == 'operator':
        payload['arguments_json'] = '{"operator":"procedure"}'
        write_json(source / 'code_action_008.json', payload)
    elif change == 'wrong_worker':
        stop['executed_path'] = 'another.py'
    elif change == 'submitted':
        previous['model'] = 'model.py'
        write_json(source / 'active.json', previous)
    else:
        (source / 'code_call_008.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match=match):
        recover(source, stop)
    assert not (source / 'interrupted_training.json').exists()


def test_cannot_overwrite_original_active_checkpoint(stopped_run):
    source, previous, _, stop = stopped_run
    with pytest.raises(ValueError, match='original active'):
        recover_interrupted_online_run(source, model='gpt-6-luna', interruption=stop,
                                        result_path=source / 'active.json')
    assert json.loads((source / 'active.json').read_text()) == previous


@pytest.fixture
def completed_query(tmp_path):
    from test_common_archive_resume import agent_at, execute, SEED
    from test_temporal_prospective_inputs import trajectory
    from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab

    agent, lab = agent_at(tmp_path / 'source'), TemporalInquiryLab(task_index=1)
    execute(agent, lab, 'query', {'actions': [trajectory(0.)]})
    previous = agent.result('researching')
    write_json(agent.workspace / 'active.json', previous)
    previous = json.loads((agent.workspace / 'active.json').read_text())
    declaration = {'id': 'paid_tail', 'question': 'Inspect the newly purchased trajectory.',
                   'parent_id': 'task', 'depends_on': []}
    arguments = {'actions': [trajectory(.2, .3), trajectory(-.2, -.3)],
                 'question': declaration}
    call = previous['costs']['model_calls'] + 1
    payload = {'action': 'query', 'arguments_json': json.dumps(arguments),
               'source_code': None, 'research_notes': 'My pending comparison.'}
    usage = {'input_tokens': 100, 'output_tokens': 25, 'cached_input_tokens': 80,
             'total_tokens': 125}

    class Provider:
        def next_action(self, request):
            write_json(agent.workspace / 'research_context.json', request)
            write_json(agent.workspace / f'code_action_{call:03d}.json', payload)
            (agent.workspace / f'code_call_{call:03d}.jsonl').write_text(json.dumps(
                {'type': 'turn.completed', 'usage': usage}) + '\n')
            return {'action': 'query', 'arguments': {**deepcopy(arguments),
                    'notes': payload['research_notes']}, 'usage': usage}

    agent.provider = Provider()

    def failed_checkpoint(result):
        assert result['events'][-1]['action'] == 'query'
        raise OSError('checkpoint writing failed after the paid query')

    with pytest.raises(OSError, match='checkpoint writing failed'):
        agent.run(lab, on_checkpoint=failed_checkpoint)
    assert len(agent.observations) == len(previous['paid_observations']) + 2
    assert json.loads((agent.workspace / 'active.json').read_text()) == previous
    return agent, lab, previous, payload, SEED


def test_completed_query_recovers_actual_paid_tail_and_readout_without_replay(completed_query, monkeypatch, tmp_path):
    from test_common_archive_resume import agent_at
    from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab

    old, lab, previous, payload, seed = completed_query
    original = (old.workspace / 'active.json').read_bytes()
    account = dict(lab.costs)

    def no_purchase(*args, **kwargs):
        raise AssertionError('recovery and restore must not replay the paid query')

    monkeypatch.setattr(TemporalInquiryLab, 'execute', no_purchase)
    path = old.workspace / 'interrupted_training_0002.json'
    recovered = recover_completed_online_query(old.workspace, model='gpt-6-luna',
        result_path=path, previous=deepcopy(previous))
    assert recovered['costs'] == old.costs
    assert recovered['paid_observations'] == json.loads(json.dumps(old.observations))
    event = recovered['events'][-1]
    assert event['step'] == previous['costs']['model_calls']
    assert event['feedback']['status'] == 'query_completed'
    assert event['feedback']['record_indices'] == [1, 2]
    assert event['usage']['total_tokens'] == 125 and event['usage']['cached_input_tokens'] == 80
    assert recovered['fresh_confirmation'] == previous['fresh_confirmation']
    question = next(q for q in recovered['online_questions'] if q['id'] == 'paid_tail')
    assert question['assignment'] == json.loads(payload['arguments_json'])['question']
    assert question['status'] == 'observed'
    assert question['evidence'] == question['acquired_evidence'] == [1, 2, 3, 4]
    assert question['evidence_readout_proposal']['record_indices'] == [1, 2]
    assert recovered['outgoing_knowledge']['files']['research_notes.md'] == payload['research_notes']
    assert json.loads(path.read_text()) == recovered
    assert recover_completed_online_query(old.workspace, model='gpt-6-luna', result_path=path) == recovered
    assert recover_completed_online_query(old.workspace, model='gpt-6-luna',
        result_path=path, previous=recovered) == recovered
    resumed, target = agent_at(tmp_path / 'continued'), TemporalInquiryLab(task_index=1)
    restored = resumed.restore_research(recovered, old.workspace, target, confirmation_seed=seed)
    assert restored['new_observations'] == 0 and resumed.costs == old.costs
    request = resumed._request(target)
    assert request['evidence_readout_proposal']['record_indices'] == [1, 2]
    assert request['evidence_readout_proposal']['response_row_count'] == 4
    assert request['laboratory_observations'] == json.loads(json.dumps(old.observations))
    assert resumed._active_question_id == 'paid_tail'
    assert (old.workspace / 'active.json').read_bytes() == original
    assert lab.costs == account


@pytest.mark.parametrize('change', ['after_observation', 'action_mismatch', 'prefix_mismatch', 'incomplete_record'])
def test_completed_query_rejects_unfinished_or_mismatched_acquisitions(completed_query, change):
    old, _, previous, payload, _ = completed_query
    call = previous['costs']['model_calls'] + 1
    if change == 'after_observation':
        arguments = json.loads(payload['arguments_json'])
        arguments['after_observation'] = [{'operator': 'refit', 'model': 'model.py'}]
        payload['arguments_json'] = json.dumps(arguments)
        write_json(old.workspace / f'code_action_{call:03d}.json', payload)
    else:
        records = json.loads((old.workspace / 'laboratory_observations.json').read_text())
        if change == 'action_mismatch':
            records[-1]['action']['sequence'][0]['u'] = .9
        elif change == 'prefix_mismatch':
            records[0]['rows'][0]['target'] += 1.
        else:
            records[-1]['rows'].pop()
        write_json(old.workspace / 'laboratory_observations.json', records)
    destination = old.workspace / 'interrupted_training_0002.json'
    with pytest.raises(ValueError):
        recover_completed_online_query(old.workspace, model='gpt-6-luna',
                                       result_path=destination, previous=deepcopy(previous))
    assert not destination.exists()
    assert json.loads((old.workspace / 'active.json').read_text()) == previous

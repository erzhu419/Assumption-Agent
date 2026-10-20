"""Recover a paid run, evaluation or completed query without replaying its work.

This is the online counterpart of interrupted_method_author: the operator stops
the original worker, records its elapsed time, and retains the original workspace.
The normal restore_research path subsequently copies those files and paid data.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .code_agent import _model_context_ignored
from .evidence_readout_proposal import prepare_evidence_readout
from .proposers import _events


def recover_completed_online_query(previous_workspace, *, model, result_path, previous=None):
    """Recover a completed explicit trajectory query after checkpoint writing failed.

    The already-loaded checkpoint may be supplied and is updated in place. No
    action, author code or laboratory is executed; the original files stay put.
    """
    source, destination = Path(previous_workspace).resolve(), Path(result_path).resolve()
    checkpoint = source / 'active.json'
    if destination == checkpoint:
        raise ValueError('retain the original active checkpoint')
    retained = previous if previous is not None else json.loads(checkpoint.read_text())
    if retained.get('model') is not None:
        raise ValueError('recover unfinished research, not a submitted model')
    records = json.loads((source / 'laboratory_observations.json').read_text())
    recovery = retained.get('completed_query_recovery')
    if recovery is not None:
        if (recovery['source_workspace'] != str(source)
                or recovery['model_call'] != retained['costs']['model_calls']
                or records != retained['paid_observations']):
            raise ValueError('the completed query recovery does not match the paid boundary')
        return retained
    call = retained['costs']['model_calls'] + 1
    response_name, trace_name = f'code_action_{call:03d}.json', f'code_call_{call:03d}.jsonl'
    payload = json.loads((source / response_name).read_text())
    arguments = json.loads(payload['arguments_json'], strict=False)
    actions = arguments.get('actions')
    if (payload['action'] != 'query' or not isinstance(actions, list) or not actions
            or payload.get('source_code') is not None
            or any(key in arguments for key in ('path', 'revision', 'after_observation'))
            or retained.get('observable_models') or retained.get('conditional_models')):
        raise ValueError('recovery requires a completed ordinary explicit-actions query')
    prefix = retained['paid_observations']
    acquired = records[len(prefix):]
    if (records[:len(prefix)] != prefix or len(acquired) != len(actions)
            or [record.get('action') for record in acquired] != actions):
        raise ValueError('persisted records do not match the paid prefix and complete query actions')
    first = retained['costs']['observation_cost']
    for action, record in zip(actions, acquired):
        count = len(action.get('sequence', []))
        if (action.get('kind') != 'trajectory' or not count
                or record.get('observation_index') != first
                or record.get('row_indices') != list(range(first, first + count))
                or len(record.get('rows', [])) != count
                or record.get('observation_cost') != count
                or not isinstance(record.get('measurement_cost'), int)):
            raise ValueError('persisted query records are not complete trajectory acquisitions')
        first += count
    events, _ = _events((source / trace_name).read_text())
    completion = next((event for event in reversed(events)
                       if event.get('type') == 'turn.completed'), None)
    if completion is None:
        raise ValueError('the saved paid call has no completed usage record')
    usage = {'backend': 'codex_cli', 'calls': 1, 'model': model,
             **{key: completion['usage'].get(key)
                for key in ('input_tokens', 'output_tokens', 'cached_input_tokens')}}
    if not all(isinstance(usage[key], int) for key in ('input_tokens', 'output_tokens')):
        raise ValueError('the saved paid call has no reported input/output token usage')
    usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
    declaration = payload.get('research_question') or arguments.get('research_question') or arguments.get('question')
    if not isinstance(declaration, dict) or not declaration.get('id'):
        raise ValueError('recover the explicit query with its authored question')
    questions = retained['online_questions']
    question = next((q for q in questions if q['id'] == declaration['id']), None)
    if question is None:
        if not declaration.get('question'):
            raise ValueError('a new research question needs its question text')
        question = {'parent_id': None, 'depends_on': [], **deepcopy(declaration),
            'opened_before_observation': retained['costs']['observation_cost'],
            'evidence': [], 'status': 'open', 'experiment_path': None, 'acquired_evidence': []}
        question['assignment'] = {key: deepcopy(question[key])
                                 for key in ('id', 'question', 'parent_id', 'depends_on')}
        questions.append(question)
    else:
        question.update(deepcopy(declaration))
    rows = [index for record in acquired for index in record['row_indices']]
    question['reviewed_child_answers'] = deepcopy(question.get('child_answers', []))
    question.setdefault('evidence', []).extend(rows)
    question['acquired_evidence'] = sorted(set(question.get('acquired_evidence', [])) | set(rows))
    question['status'] = 'observed'
    indices = list(range(len(prefix), len(records)))
    retained['paid_observations'] = records
    prepare_evidence_readout(SimpleNamespace(questions=questions,
        revisions=retained.get('revision_proposals', []), observations=records,
        _active_question_id=question['id']), question)
    if payload.get('research_question') is not None:
        arguments['research_question'] = payload['research_question']
    if 'research_notes' in payload:
        arguments['notes'] = payload['research_notes']
    cost = sum(record['observation_cost'] for record in acquired)
    feedback = {'status': 'query_completed', 'observations': acquired,
        'observation_cost': cost, 'executed_path': None, 'inherited_program': False,
        'inherited_source_unchanged': False,
        'question': deepcopy(question), 'research_question_id': question['id'],
        'record_indices': indices, 'after_observation': [], 'continuation_origin': None,
        'knowledge_sources': list(arguments.get('knowledge_sources', []))}
    retained['events'].append({'step': call - 1, 'action': 'query', 'arguments': arguments,
                               'feedback': feedback, 'usage': usage})
    retained['costs']['model_calls'] = call
    retained['costs']['reported_tokens'] += usage['total_tokens']
    retained['costs']['observation_cost'] += cost
    retained['costs']['measurement_cost'] += sum(record['measurement_cost'] for record in acquired)
    recovery = {'reason': 'The explicit query completed; checkpoint writing failed afterward.',
        'source_workspace': str(source), 'model_call': call,
        'saved_reply': response_name, 'saved_usage': trace_name, 'record_indices': indices,
        'scope': 'The persisted paid observations and reply are retained without replaying the query.'}
    retained.update(status='checkpoint_interrupted', workspace=str(source), completed_query_recovery=recovery)
    files = retained['outgoing_knowledge']['files']
    for name in list(files):
        if _model_context_ignored(source / name, source):
            files.pop(name)
    if 'research_notes' in payload:
        files['research_notes.md'] = payload['research_notes']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as output:
        json.dump(retained, output, ensure_ascii=False)
    return retained


def recover_interrupted_online_run(previous_workspace, *, model, interruption, result_path):
    """Export the stopped execution boundary; make no model or laboratory call."""
    source, destination = Path(previous_workspace).resolve(), Path(result_path).resolve()
    checkpoint = source / 'active.json'
    if destination == checkpoint:
        raise ValueError('retain the original active checkpoint')
    # This checkpoint contains a large executable archive. Mutate the loaded
    # result once instead of copying the archive and parsing it a second time.
    retained = json.loads(checkpoint.read_text())
    if retained.get('model') is not None:
        raise ValueError('recover unfinished research, not a submitted model')
    call = retained['costs']['model_calls'] + 1
    response_name, trace_name = f'code_action_{call:03d}.json', f'code_call_{call:03d}.jsonl'
    payload = json.loads((source / response_name).read_text())
    arguments = json.loads(payload['arguments_json'], strict=False)
    action = payload['action']
    ordinary_run = action == 'run' and arguments.get('operator') is None
    response_evaluation = (action == 'evaluate' and arguments.get('operator') is None
        and 'component' not in arguments and arguments.get('target', 'response') == 'response')
    if not ordinary_run and not response_evaluation:
        raise ValueError('recovery requires an interrupted ordinary Python run or response evaluation')
    if payload.get('source_code') is not None:
        arguments['code'] = payload['source_code']
    if payload.get('research_question') is not None:
        arguments['research_question'] = payload['research_question']
    if 'research_notes' in payload:
        arguments['notes'] = payload['research_notes']
    executed_path = (arguments.get('model', 'model.py') if response_evaluation
                     else arguments.get('path', f"scratch_{len(retained['events']):03d}.py"))
    if interruption['executed_path'] != executed_path:
        raise ValueError('stopped worker does not match the saved paid action')
    elapsed = interruption['interrupted_python_seconds']

    # Use the same completed CLI event and token accounting as the historical
    # method-author recovery. Cached tokens are included in input_tokens.
    events, _ = _events((source / trace_name).read_text())
    completion = next((event for event in reversed(events)
                       if event.get('type') == 'turn.completed'), None)
    if completion is None:
        raise ValueError('the saved paid call has no completed usage record')
    usage = {'backend': 'codex_cli', 'calls': 1, 'model': model,
             **{key: completion['usage'].get(key)
                for key in ('input_tokens', 'output_tokens', 'cached_input_tokens')}}
    if not all(isinstance(usage[key], int) for key in ('input_tokens', 'output_tokens')):
        raise ValueError('the saved paid call has no reported input/output token usage')
    usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']

    recovery = {'reason': ('The operator stopped the response evaluation before its return.'
                           if response_evaluation else
                           'The operator stopped the ordinary Python run before its return.'),
        'source_workspace': str(source), 'saved_reply': response_name, 'saved_usage': trace_name,
        'operator_stop': deepcopy(interruption),
        'scope': 'The paid reply and files are retained. The unfinished computation is not '
                 'a successful or refuted scientific result; neither the action nor data was replayed.'}
    feedback = {'status': 'runtime_interrupted', 'executed_path': executed_path,
                'execution_seconds': elapsed, 'runtime_interruption': recovery}
    if response_evaluation:
        # _score charges all current response rows before starting its worker.
        # The pre-action checkpoint predates that charge; retain it without
        # evaluating the model or turning interrupted work into a loss value.
        scoring_rows = sum(len(record.get('row_indices', []))
                           for record in retained['paid_observations'])
        retained['costs']['compute_cost'] += scoring_rows
        feedback.update(scoring_rows=scoring_rows, compute_cost=scoring_rows)
    for stream in ('stdout', 'stderr'):
        output = interruption.get('partial_outputs', {}).get(stream)
        text = Path(output['path']).read_text() if output else ''
        feedback.update({stream: text[-12000:], f'{stream}_characters': len(text),
                         f'{stream}_truncated': len(text) > 12000})
        if output:
            feedback[f'{stream}_file'] = output['path']

    declaration = arguments.get('research_question') or arguments.get('question')
    if declaration:
        questions = retained['online_questions']
        question = next((q for q in questions if q['id'] == declaration['id']), None)
        if question is None:
            question = {'parent_id': None, 'depends_on': [], **deepcopy(declaration),
                'opened_before_observation': retained['costs']['observation_cost'],
                'evidence': [], 'status': 'open', 'experiment_path': None,
                'acquired_evidence': []}
            question['assignment'] = {key: deepcopy(question[key])
                                      for key in ('id', 'question', 'parent_id', 'depends_on')}
            questions.append(question)
        else:
            question.update(deepcopy(declaration))
        feedback['research_question_id'] = question['id']
    elif retained.get('inquiry_stack'):
        feedback['research_question_id'] = retained['inquiry_stack'][-1]['question_id']
    else:
        review = next((q for q in reversed(retained.get('online_questions', []))
                       if q.get('flat_review', {}).get('status') == 'running'), None)
        if review is not None:
            feedback['research_question_id'] = review['id']
        else:
            parent = next((q['parent_research'] for q in reversed(retained.get('online_questions', []))
                           if q.get('parent_research', {}).get('status') == 'running'), None)
            if parent is not None:
                feedback['research_question_id'] = parent['parent_id']

    retained['events'].append({'step': call - 1, 'action': action, 'arguments': arguments,
                               'feedback': feedback, 'usage': usage})
    retained['costs']['model_calls'] += 1
    retained['costs']['reported_tokens'] += usage['total_tokens']
    retained['costs']['python_execution_seconds'] += elapsed
    retained.update(status='runtime_interrupted', workspace=str(source),
                    runtime_interruption=recovery)

    # Ordinary Python may have written dependencies before being stopped. The
    # workspace is retained in full; refresh its changed textual knowledge so
    # the restored author also sees those writes and its already-paid notebook.
    files = retained['outgoing_knowledge']['files']
    checkpoint_mtime = checkpoint.stat().st_mtime
    names = set(files)
    names.update(path.name for path in source.glob('*.py'))
    names.add('research_notes.md')
    # Only the authored shared directory needs discovery. Candidate snapshots
    # and paid histories are retained by COW restoration, never traversed here.
    for directory, subdirs, filenames in os.walk(source / 'shared'):
        root = Path(directory)
        subdirs[:] = [name for name in subdirs
                      if not _model_context_ignored(root / name, source)]
        names.update(str((root / name).relative_to(source)) for name in filenames)
    paid_files = {'observations.json', 'model_observations.json', 'laboratory_observations.json'}
    for name in names:
        path = source / name
        if (not path.is_file() or path.suffix not in {'.py', '.json', '.md', '.txt'}
                or _model_context_ignored(path, source) or path == destination
                or (path.parent == source and path.name in paid_files)):
            continue
        name = str(path.relative_to(source))
        if name not in files or path.stat().st_mtime > checkpoint_mtime:
            files[name] = path.read_text(encoding='utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as output:
        json.dump(retained, output, ensure_ascii=False)
    return retained

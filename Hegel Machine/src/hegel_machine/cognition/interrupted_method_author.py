"""Retain a paid method edit or practice request interrupted before its return.

The caller stops the old process and copies its workspace before recovery. This
exports the saved evidence; it executes neither the reply nor the apparatus.
"""

from copy import deepcopy
import json
from pathlib import Path

from .proposers import _events
from .checkpoint_storage import write_text_incrementally
from .temporal_method_learning import METHOD_PATH


def _edit_practice_case(previous, source, cases):
    # Same choice as AnonymousResearchMethodLearner after a changed source.
    latest = {trial['case']: trial for trial in previous['practice_trials']
              if trial['method_source'] == source and trial['case'] in cases}
    confirmed = {case for case, trial in latest.items()
                 if trial['status'] == 'completed'
                 and (trial.get('fresh_confirmation') or {}).get('passed') is True}
    missing = set(cases) - confirmed
    first_missing = next(case for case in cases if case in missing)
    return next((trial['case'] for trial in reversed(previous['practice_trials'])
                 if trial['case'] in missing and (trial['status'] != 'completed'
                     or (trial.get('fresh_confirmation') or {}).get('passed') is not True)),
                first_missing)


def recover_interrupted_method_author(previous_workspace, workspace):
    """Write an honest interruption result into an already copied workspace.

    Supported boundary: a saved paid write or explicit run/practice launched a trial,
    whose research state exists but whose result was never written. The author
    checkpoint precedes that action. Partial steps retain their recorded scope;
    a comparison saved ahead of its enclosing step does not complete that step.
    An operator stop record may return already observed non-progress and charge
    the interrupted worker's measured time without completing its pending step.
    """
    previous_workspace, workspace = Path(previous_workspace).resolve(), Path(workspace).resolve()
    if previous_workspace == workspace:
        raise ValueError('recover into the copied workspace, retaining the stopped original')
    previous = json.loads((previous_workspace / 'active.json').read_text())
    plan = json.loads((previous_workspace.parent / 'protocol.json').read_text())
    call = previous['costs']['model_calls'] + 1
    response_name, trace_name = f'code_action_{call:03d}.json', f'code_call_{call:03d}.jsonl'
    payload = json.loads((previous_workspace / response_name).read_text())
    arguments = json.loads(payload['arguments_json'], strict=False)
    action = payload['action']
    if action == 'write' and arguments.get('path') == METHOD_PATH:
        source = payload.get('source_code', arguments.get('code'))
        if not isinstance(source, str):
            raise ValueError('the saved method write has no source')
        arguments['code'] = source
        case = _edit_practice_case(previous, source, plan['practice_cases'])
    elif action == 'run' and arguments.get('operator') == 'practice':
        case = arguments.get('case')
        if case not in plan['practice_cases']:
            raise ValueError('the saved practice case is not in this author protocol')
        source = (previous_workspace / METHOD_PATH).read_text()
    else:
        raise ValueError('recovery requires the saved method write or explicit practice run')
    directory = f"trials/trial_{len(previous['practice_trials']):03d}"
    old_trial, trial = previous_workspace / directory, workspace / directory
    if (old_trial / 'result.json').exists():
        raise ValueError('practice already has a result; use its completed author checkpoint')
    for path in (previous_workspace / METHOD_PATH, old_trial / METHOD_PATH,
                 workspace / METHOD_PATH, trial / METHOD_PATH):
        if path.read_text() != source:
            raise ValueError('saved paid reply and copied method sources differ')
    events, _ = _events((previous_workspace / trace_name).read_text())
    completion = next(event for event in reversed(events) if event.get('type') == 'turn.completed')
    usage = {'backend': 'codex_cli', 'calls': 1, 'model': plan['model'],
             **{key: completion['usage'].get(key)
                for key in ('input_tokens', 'output_tokens', 'cached_input_tokens')}}
    if not all(isinstance(usage[key], int) for key in ('input_tokens', 'output_tokens')):
        raise ValueError('the saved completed call has no reported input/output token usage')
    usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
    state = json.loads((old_trial / 'research_state.json').read_text())
    observations = json.loads((old_trial / 'laboratory_observations.json').read_text())
    stop_path = previous_workspace / 'runtime_stop.json'
    stop = json.loads(stop_path.read_text()) if stop_path.exists() else None
    costs = deepcopy(state['costs'])
    if stop is not None:
        # The interrupted worker has not returned, so its elapsed execution is
        # absent from the last checkpoint. Charge the operator's measured wall
        # time once, separately from the already completed Python invocations.
        costs['python_execution_seconds'] += stop['interrupted_python_seconds']
    no_progress = stop.get('no_progress') if stop is not None else None
    practice_status = 'method_no_progress' if no_progress is not None else 'runtime_interrupted'
    interruption = {
        'reason': 'Runtime interrupted before the paid practice returned to its author.',
        'source_workspace': str(previous_workspace), 'saved_reply': response_name,
        'saved_usage': trace_name, 'saved_practice_state': f'{directory}/research_state.json',
        'scope': 'Recorded steps and observations are retained. The unfinished action is not '
                 'declared successful or scientifically failed; no reply or acquisition was replayed.',
    }
    if stop is not None:
        interruption['operator_stop'] = deepcopy(stop)
    result = {
        'agent_arm': 'hegel', 'status': practice_status, 'workspace': str(trial),
        'model': None, 'costs': deepcopy(costs), 'events': [],
        'method_results': state['method_results'], 'experiment_comparisons': state['experiment_comparisons'],
        'paid_observations': observations, 'fresh_confirmation': None,
        'evaluated_model': state['evaluated_model'], 'online_questions': state['questions'],
        'revision_proposals': state['revisions'], 'limits': state['limits'],
        'outgoing_knowledge': {'files': {METHOD_PATH: source}},
        'runtime_interruption': interruption,
    }
    if no_progress is not None:
        result['no_progress'] = deepcopy(no_progress)
    retained = deepcopy(previous)
    retained.update(status='runtime_interrupted', workspace=str(workspace), model=None,
                    method_source=None, method_path=None, runtime_interruption=interruption)
    retained['outgoing_knowledge']['files'][METHOD_PATH] = source
    for key in ('observation_cost', 'measurement_cost', 'compute_cost', 'python_execution_seconds'):
        retained['costs'][key] += costs[key]
    retained['costs']['model_calls'] += 1
    retained['costs']['reported_tokens'] += usage['total_tokens']
    practice = {'case': case, 'directory': directory, 'status': practice_status,
                'costs': deepcopy(costs), 'selected_prior_predictor': False,
                'fresh_confirmation': None}
    if no_progress is not None:
        practice['no_progress'] = deepcopy(no_progress)
    retained['practice_trials'].append({**practice, 'method_source': source})
    feedback = {**practice, 'model': None, 'steps': deepcopy(state['method_results']),
                'runtime_interruption': interruption}
    if action == 'write':
        feedback = {'written': METHOD_PATH, 'edit_practice': feedback}
    retained['events'].append({'step': call - 1, 'action': action, 'arguments': arguments,
                              'feedback': feedback, 'usage': usage})
    write_text_incrementally(trial / 'result.json', json.dumps(result, indent=2))
    destination = workspace / 'runtime_interrupted.json'
    write_text_incrementally(destination, json.dumps(retained, indent=2),
        base=previous_workspace / 'active.json' if not destination.exists() else None)
    return retained

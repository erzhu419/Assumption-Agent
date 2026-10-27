"""One registered, locally funded action using the current Common operators."""
import os
import sys
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from time import monotonic

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT / 'source_at_run'
sys.path[:0] = [str(REPOSITORY / 'src'), str(REPOSITORY), str(ROOT / 'prepared')]
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

from prepare_case_scene import prepare_scene
from paid_prediction_readout import read_paid_prediction_pair
from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_json_checkpoint, clone_or_copy
from hegel_machine.cognition.code_agent import python_model_context, _closed_temporal_export
from hegel_machine.cognition.experiment_runtime import (
    checkpoint_execution, python_execution_deadline, physical_worker_cpus, _initialize_worker,
)
from hegel_machine.cognition.finite_native_author import _comparison_models, paid_experiment_record_indices
from hegel_machine.cognition.question_continuation import pending_query_arguments
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from hegel_machine.cognition.registered_source_control import _comparison_external_python_timing


def save(path, value, **kwargs):
    write_json_checkpoint(path, value, ensure_ascii=False, indent=None, **kwargs)


def models_for(agent, question):
    rounds = question.get('native_condition_research', [])
    return _comparison_models(agent.knowledge['finite_native_author']['predictor_paths'],
        agent.candidate_paths, rounds)


def model_source(agent, model):
    directory, inner = python_model_context(agent.workspace, model)
    return (directory / inner).read_bytes() if _closed_temporal_export(directory, inner) else None


def paid_ref_for_source(agent, feedback, source):
    if source is None:
        return None
    for item in feedback.get('evaluations', []):
        reference = item.get('paid_score_reuse', {}).get('execution_basis_ref')
        if reference and item.get('returncode') == 0 and model_source(agent, item['path']) == source:
            return reference
    return None


def main():
    run = ROOT / 'run'
    if run.exists():
        raise ValueError('this registered unit was already started; preserve its outcome')
    run.mkdir()
    registration = read_json_checkpoint(ROOT / 'prepared/budget_registration.json')
    unit = registration['registered_unit']
    state = {'status': 'restoring', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'registration_ref': {'file': str(ROOT / 'prepared/budget_registration.json'), 'json_path': []},
        'actions': [], 'new_world_terminal_evaluations': 0, 'automatic_retry': False}
    save(run / 'status.json', state)
    restored_at = monotonic()
    scene = prepare_scene(run / 'case_002/pr3', registration,
        source=ROOT / 'prepared/case_002/pr3')
    state['restoration_engineering_wall_seconds'] = monotonic() - restored_at
    agent, lab = scene['agent'], scene['lab']
    before = deepcopy(agent.costs)
    initial_steps = agent._organization_state['finite_author']['completed_steps']
    question = next(q for q in agent.questions
        if q['id'] == agent.knowledge['finite_native_author']['question']['id'])
    agent._active_question_id = question['id']
    quote = scene['acceptance']['native_quote']['compute_continuation']
    if quote['status'] != 'available' or quote['research_unit']['minimum_compute'] > unit['compute_cap']:
        state.update(status='admission_blocked', actual_quote=quote)
        save(run / 'summary.json', state)
        return
    cpus = physical_worker_cpus()[:unit['max_physical_cores']]
    _initialize_worker(cpus)
    state.update(status='researching', before_parent=scene['before_parent'], before_costs=before,
        accepted_quote=quote, assigned_physical_cpus=cpus, original_author_cursor=initial_steps)
    save(run / 'status.json', state)
    source = (agent.workspace / scene['plan']['author']['path']).read_text()
    namespace = {'__name__': 'registered_complete_action',
        '__file__': str(agent.workspace / scene['plan']['author']['path'])}
    exec(compile(source, namespace['__file__'], 'exec'), namespace)
    before_source = model_source(agent, scene['before_parent']['model'])
    before_ref = after_ref = None
    meaningful = None
    native_started = native_committed = False
    final_checkpoint_saved = False
    original_native_rounds = len(question.get('native_condition_research', []))
    action_at = monotonic()
    legacy_ceiling = before['python_execution_seconds'] + unit['active_wall_limit_seconds']

    def remaining():
        return max(0., unit['active_wall_limit_seconds'] - (monotonic() - action_at))

    def clamp_time():
        agent.config = replace(agent.config, execution_seconds=min(legacy_ceiling,
            agent.costs['python_execution_seconds'] + remaining()))

    def checkpoint(status):
        with checkpoint_execution():
            data = agent.checkpoint_result('researching')
            data['registered_unit_status'] = status
            save(agent.workspace / 'active.json', data,
                 base=agent.workspace / 'training.json')

    try:
        with python_execution_deadline(unit['active_wall_limit_seconds']):
            for _ in range(16):
                clamp_time()
                agent._active_question_id = question['id']
                agent._prepare_program_state()
                selected_at, old_cwd = monotonic(), Path.cwd()
                try:
                    os.chdir(agent.workspace)
                    proposal = namespace['select_actions'](lab.public_spec(), agent.observations,
                        agent._program_memory(scene['plan']['author']['path'], {}))
                finally:
                    os.chdir(old_cwd)
                    select_seconds = monotonic() - selected_at
                    agent.costs['python_execution_seconds'] += select_seconds
                action, arguments = proposal.get('action'), proposal.get('arguments', {})
                if action is None or action == 'finish':
                    state.update(status='author_boundary', boundary=proposal)
                    break
                operator = arguments.get('operator')
                if action == 'query':
                    query = (pending_query_arguments(agent, arguments['pending_question'])
                        if 'pending_question' in arguments else arguments)
                    actions = query.get('actions', [])
                    cached = paid_experiment_record_indices(actions, agent.observations,
                        lab.public_spec(), question, agent.knowledge['finite_native_author'],
                        prediction_rows=lab.prediction_rows)
                    answered = bool(actions) and all(index is not None for index in cached)
                    rounds = question.get('native_condition_research', [])
                    experiment = next((item for item in (rounds[-1].get('proposed_experiments', [])
                            if rounds else [])
                        if item.get('query_action', {}).get('arguments', {}).get('actions') == actions
                        and item.get('predicted_separation', 0) > 0), None)
                    if not answered and experiment is None:
                        state.update(status='new_query_consequence_unresolved',
                            boundary={'proposal': proposal,
                                'reason': 'No same-scope paid answer or retained distinct predicted consequence.'})
                        break
                    state.setdefault('query_evidence', []).append({'paid_record_indices': cached,
                        'kind': 'existing_paid_answer' if answered else 'retained_prediction_separation',
                        'predicted_separation': experiment.get('predicted_separation') if experiment else None})
                native_started = native_started or operator == 'native_condition_research'
                prepared = {'path': scene['plan']['author']['path'], 'proposal': proposal,
                    'program_execution': {'path': scene['plan']['author']['path'], 'returncode': 0,
                        'timeout': False, 'python_execution_seconds': select_seconds}}
                count = len(agent.method_results)
                candidates_before = set(agent.candidate_paths)
                execute_procedure(agent, lab, prepared['path'], max_steps=1,
                    allow_confirmation=False, record_events=True, _prepared_program_result=prepared)
                if len(agent.method_results) == count:
                    state['status'] = 'selected_action_not_executed'
                    break
                agent._organization_state['finite_author']['completed_steps'] += 1
                feedback = agent.method_results[-1]['feedback']
                entry = {'action': action, 'operator': operator,
                    'method_result_ref': {'file': str(agent.workspace / 'training.json'),
                        'json_path': ['method_results', len(agent.method_results) - 1]},
                    'feedback_status': feedback.get('status'), 'error': feedback.get('error'),
                    'new_known_compute': agent.costs['compute_cost'] - before['compute_cost']}
                state['actions'].append(entry)
                if operator == 'research_choice':
                    before_ref = paid_ref_for_source(agent, feedback, before_source) or before_ref
                native = feedback.get('native_condition_research')
                if operator == 'native_condition_research' and native:
                    native_committed = True
                    entry['native_result'] = {k: native.get(k) for k in
                        ('status', 'round_index', 'searched_candidates', 'search_stages', 'compute_cost')}
                    meaningful = 'native_condition_research'
                elif action == 'query' and agent.costs['observation_cost'] > before['observation_cost']:
                    meaningful = 'new_distinguishing_query'
                elif operator in {'refit', 'response', 'research_update', 'research_route', 'theory_transfer'}:
                    delivered = set(agent.candidate_paths) - candidates_before
                    if delivered and not feedback.get('error') and feedback.get('status') not in {
                            'refit_unavailable', 'method_use_deferred', 'scoped_unknown',
                            'author_work_request', 'research_deferred', 'no_candidates'}:
                        meaningful = operator
                        entry['new_exported_candidates'] = sorted(delivered)
                checkpoint('registered_complete_action_progress')
                save(run / 'status.json', state)
                if feedback.get('error'):
                    state['status'] = 'action_failed'
                    break
                if meaningful:
                    clamp_time()
                    completion = {'action': 'run', 'arguments': {'operator': 'research_choice',
                        'research_question': {'id': question['id']},
                        'models': models_for(agent, question)}}
                    completed_count = len(agent.method_results)
                    execute_procedure(agent, lab, prepared['path'], max_steps=1,
                        allow_confirmation=False, record_events=True,
                        _prepared_program_result={'path': prepared['path'], 'proposal': completion,
                            'program_execution': {'path': prepared['path'], 'returncode': 0,
                                'timeout': False, 'python_execution_seconds': 0.,
                                'basis': 'registered_complete_unit_strong_pool_comparison'}})
                    if len(agent.method_results) == completed_count:
                        state['status'] = 'completion_comparison_not_executed'
                        break
                    agent._organization_state['finite_author']['completed_steps'] += 1
                    comparison = agent.method_results[-1]['feedback']
                    state['completion_comparison_ref'] = {
                        'file': str(agent.workspace / 'training.json'),
                        'json_path': ['method_results', len(agent.method_results) - 1, 'feedback']}
                    state['comparison'] = {k: comparison.get(k) for k in
                        ('status', 'models_requested', 'models_completed', 'truncated', 'error')}
                    state['comparison_external_timing'] = _comparison_external_python_timing(comparison)
                    before_ref = paid_ref_for_source(agent, comparison, before_source) or before_ref
                    after = agent._evaluated_summary()
                    after_ref = paid_ref_for_source(agent, comparison,
                        model_source(agent, after['model'])) if after else None
                    state['status'] = ('completed_action_and_strong_pool_comparison' if
                        not comparison.get('error') and not comparison.get('truncated')
                        and comparison.get('models_requested') == comparison.get('models_completed')
                        else 'comparison_incomplete')
                    if (native and native.get('status') == 'search_incomplete'
                            and state['status'] == 'completed_action_and_strong_pool_comparison'):
                        state['status'] = 'bounded_action_compared_search_incomplete'
                    break
            else:
                state['status'] = 'bounded_preparation_steps_exhausted'
            if before_ref and after_ref:
                state['frozen_paid_predictions'] = read_paid_prediction_pair(
                    before_ref, after_ref, agent.workspace, expected_rows=145)
            with checkpoint_execution():
                data = agent.checkpoint_result('researching')
                data['registered_unit_status'] = state['status']
                save(agent.workspace / 'training.json', data,
                     base=agent.workspace / 'training.json')
            final_checkpoint_saved = True
    except Exception as error:
        state.update(status='interrupted_preserve_paid_work',
            error={'type': type(error).__name__, 'message': str(error)})
    finally:
        committed_native_compute = sum(item.get('compute_cost', 0) for item in
            question.get('native_condition_research', [])[original_native_rounds:])
        state.update(scientific_active_wall_seconds=monotonic() - action_at,
            after_parent=agent._evaluated_summary(), costs=deepcopy(agent.costs),
            new_costs={k: agent.costs[k] - value for k, value in before.items()},
            meaningful_action=meaningful, final_author_cursor=agent._organization_state['finite_author']['completed_steps'],
            before_score_ref=before_ref, after_score_ref=after_ref,
            frozen_prediction_status='recorded' if 'frozen_paid_predictions' in state else 'UNKNOWN',
            uncommitted_native_compute_hold_upper=(max(0,
                unit['search_compute_limit'] - committed_native_compute) if
                native_started and not native_committed else 0),
            old_unknown_hold=registration['original_compute_account']['old_unknown_compute_hold'])
        preserve_at = monotonic()
        if not final_checkpoint_saved:
            with checkpoint_execution():
                data = agent.checkpoint_result('researching')
                data['registered_unit_status'] = state['status']
                save(agent.workspace / 'training.json', data,
                     base=agent.workspace / 'training.json')
        state['final_preservation_wall_seconds'] = monotonic() - preserve_at
        state['elapsed_through_preservation_seconds'] = monotonic() - action_at
        state['active_window_exceeded'] = state['elapsed_through_preservation_seconds'] > unit['active_wall_limit_seconds']
        state['preservation_after_cutoff_seconds'] = max(0.,
            state['elapsed_through_preservation_seconds'] - unit['active_wall_limit_seconds'])
        if state['active_window_exceeded']:
            state['outcome_before_window_accounting'] = state['status']
            state['status'] = 'window_exhausted_paid_work_preserved'
        save(run / 'summary.json', state)
        save(run / 'status.json', state)
        print(state['status'], state['new_costs'], flush=True)


if __name__ == '__main__':
    main()

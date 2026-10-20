#!/usr/bin/env python3
"""Continue a paid failed practice online, then compile and transfer its method.

This is a new development phase. Earlier failures and their costs remain intact;
the source study's publication rules and registered target episodes are retained.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.interrupted_online_inquiry import recover_interrupted_online_run
from scripts.run_anonymous_method_learning import laboratory
from scripts.run_anonymous_method_revision import run_revision
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_native_method_compilation import run_compilation
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_python_method_learning_study import LIMIT_COSTS, usage_summary


def read_json(path):
    return json.loads(Path(path).read_text())


def source_evaluation(case, evaluation):
    """Describe the trajectory cap actually used by conditional evaluation_rows."""
    result = deepcopy(evaluation)
    if case.get('kind') == 'conditional_window':
        result['trajectory_length'] = laboratory(case).public_spec()['max_probe_length']
    return result


def protocol(source_study, *, trial='trial_095', model='gpt-6-luna',
             timeout_seconds=1800., online_limits=None, author_limits=None):
    source = Path(source_study).resolve()
    old = read_json(source / 'protocol.json')
    author = read_json(source / 'author/training.json')
    native = old.get('study') == 'luna_native_research_method_compilation'
    if (not (old.get('conditional_design') or native) or author.get('method_source')):
        raise ValueError('continue an unpublished conditional-design study')
    index, practice = next((index, item) for index, item in enumerate(author['practice_trials'])
                           if Path(item['directory']).name == trial)
    workspace = source / 'author' / practice['directory']
    previous = read_json(workspace / 'result.json')
    if previous['status'] != 'confirmation_failed':
        raise ValueError('select the actual paid failed-confirmation practice')
    case = {'id': practice['case'], **deepcopy(old['practice_cases'][practice['case']])}
    if case.get('kind') != 'conditional_window':
        raise ValueError('select a conditional-window practice for this continuation')
    # The practice learner offsets its RNG by the draft's original trial index.
    case['confirmation_seed'] += index
    # Native repair practices intentionally omit the old target's terminal seed.
    # Register a separate development check before obtaining any new outcomes.
    case.setdefault('evaluation_seed', case['confirmation_seed'] + 60_000_043)
    default_online = (InquiryLLMConfig(
        max_model_calls=32, token_budget=12_000_000, observation_budget=4096,
        execution_seconds=7200., compute_budget=2_000_000,
        confirmation_rows=96, confirmation_mse_threshold=1e-6) if native else InquiryLLMConfig(
        max_model_calls=128, token_budget=20_000_000, observation_budget=120_000,
        execution_seconds=36_000., compute_budget=8_000_000,
        confirmation_rows=96, confirmation_mse_threshold=1e-6))
    allowance = asdict(online_limits or default_online)
    allowance['confirmation_rows'] = old['limits']['confirmation_rows']
    allowance['confirmation_mse_threshold'] = case.get('limits', {}).get(
        'confirmation_mse_threshold', old['limits'].get('confirmation_mse_threshold', 1e-10))
    limits = deepcopy(allowance)
    for limit, cost in LIMIT_COSTS:
        limits[limit] += previous['costs'][cost]
    limits['token_budget'] += previous['costs']['reported_tokens']
    provider = {**old['provider'], 'timeout_seconds': timeout_seconds}
    runtime = sorted(set(old.get('runtime_snapshot', [])) | {
        'scripts/run_conditional_online_method_repair.py',
        'scripts/run_anonymous_revision_inquiry.py',
        'scripts/run_native_method_compilation.py'})
    online = {
        'study': 'conditional_paid_practice_online_continuation',
        'model': model, 'provider': provider, 'cases': [case], 'arms': ['hegel'],
        'limits': limits, 'prefix_rows': old['prefix_rows'],
        'evaluation': source_evaluation(case, {**old['evaluation'],
                       'mse_threshold': allowance['confirmation_mse_threshold']}),
        'runtime_snapshot': runtime,
        'scope': 'Same paid development apparatus and failed hypothesis, continued with live Luna. '
            'New independent confirmation and source terminal evaluation precede method compilation. '
            'This source was selected after failure and is not held-out transfer evidence.'}
    historical = read_json(source / 'summary.json')['prior_author_phases']
    method = deepcopy(old)
    method.update(model=model, provider=provider, runtime_snapshot=runtime,
        continued_author=str(source / 'author/training.json'),
        author_limits=(asdict(author_limits) if author_limits is not None else
                       deepcopy(old['author_limits'] if native else
                                old.get('author_total_limits', old['author_limits']))),
        prior_author_phases=[*historical, {
            'result': str(source / 'author/training.json'), 'status': author['status'],
            'costs': author['costs'], 'usage': usage_summary(author)}],
        continuation='New development allowance after online source research. Restore every paid '
            'draft and failure. The earlier exhausted account stays exhausted; its costs are retained. '
            'Only independently confirmed online experience is added to the method author. '
            'Practice cases, publication rules, original K0 and target protocols are unchanged.')
    method['author_total_limits'] = deepcopy(method['author_limits'])
    method.pop('continuation_usage', None)
    return {'study': 'conditional_online_research_then_method_transfer',
        'source_study': str(source), 'practice_workspace': str(workspace),
        'practice_result': str(workspace / 'result.json'), 'practice_index': index,
        'restored_practice_trials': len(author['practice_trials']),
        'online_allowance': allowance, 'online': online, 'method': method,
        'stage_order': ['restore_paid_practice', 'online_research', 'independent_confirmation',
                        'source_terminal_evaluation', 'method_compilation',
                        'existing_publication_gate', 'frozen_targets'],
        'accounting': 'The restored practice is already in the failed author account. Only online '
            'continuation_cost is additional. All old author phases, including missing token usage, '
            'remain separate. New method authoring has an explicitly new allowance.',
        'target_episodes': len(old['transfer_cases']) if native else 56}


def rescue_protocol(source_study, *, model='gpt-6-luna', timeout_seconds=1800.,
                    online_limits=None):
    """Add bounded development to a finished online account, without rebuying data."""
    source = Path(source_study).resolve()
    old = read_json(source / 'protocol.json')
    summary = read_json(source / 'summary.json')
    if old['study'] != 'conditional_online_research_then_method_transfer':
        raise ValueError('select a completed conditional online-research pipeline')
    if summary['status'] not in {'online_research_incomplete', 'online_confirmation_failed'}:
        raise ValueError('rescue an unresolved source, not an already published method')
    training_path = Path(summary['online_training'])
    previous = read_json(training_path)
    if previous.get('model') is not None:
        raise ValueError('this rescue requires unfinished research, not a terminally evaluated submission')
    allowance = asdict(online_limits or InquiryLLMConfig(
        max_model_calls=32, token_budget=4_000_000, observation_budget=4096,
        execution_seconds=3600., compute_budget=1_000_000,
        confirmation_rows=96, confirmation_mse_threshold=1e-6))
    for name in ('confirmation_rows', 'confirmation_mse_threshold'):
        allowance[name] = old['online']['limits'][name]
    plan = deepcopy(old)
    plan.update(rescue_study=str(source), resume_result=str(training_path),
        resume_workspace=str(training_path.parent), online_allowance=allowance,
        development_origin='Rescue using all paid evidence from a finished online investigation; '
            'not a restart from the original 192-row prefix or a sample-efficiency comparison.')
    online = plan['online']
    online.update(study='conditional_all_paid_evidence_rescue', model=model,
        limits=deepcopy(allowance), scope=plan['development_origin'])
    online['evaluation'] = source_evaluation(online['cases'][0], online['evaluation'])
    for limit, cost in LIMIT_COSTS:
        online['limits'][limit] += previous['costs'][cost]
    online['limits']['token_budget'] += previous['costs']['reported_tokens']
    for stage in ('online', 'method'):
        plan[stage]['model'] = model
        plan[stage]['provider']['timeout_seconds'] = timeout_seconds
        plan[stage]['runtime_snapshot'] = sorted(set(plan[stage]['runtime_snapshot']) | {
            str(path.relative_to(ROOT)) for path in (ROOT / 'src/hegel_machine').rglob('*.py')})
    # Older source costs are already in the author account. Record each online
    # increment once, rather than adding its restored cumulative account again.
    prior_path = old.get('resume_result', old['practice_result'])
    prior_events = len(read_json(prior_path)['events'])
    plan['method']['prior_author_phases'].append({
        'phase': 'online_source_research', 'result': str(training_path),
        'status': previous['status'], 'costs': previous['continuation_cost'],
        'usage': usage_summary({'events': previous['events'][prior_events:]}),
        'cost_scope': 'Only this historical online increment; the restored account appears '
                      'in earlier phases. Failed and paid work remains part of method-learning costs.'})
    plan['accounting'] = ('Restore the full prior online account and confirmation stream without '
        'new observations. online_additional_costs reports only this rescue; online_cumulative_costs '
        'includes its restored source. Prior author and online increments are retained separately.')
    return plan


def interrupted_protocol(source_study, *, model='gpt-6-luna', timeout_seconds=1800.):
    """Resume a stopped paid run under its unchanged cumulative limits."""
    source = Path(source_study).resolve()
    old = read_json(source / 'protocol.json')
    if old['study'] != 'conditional_online_research_then_method_transfer':
        raise ValueError('select an interrupted conditional online-research pipeline')
    if (source / 'summary.json').exists():
        raise ValueError('this pipeline already finished; use its recorded result')
    interruption = read_json(source / 'operator_pause.json')
    if interruption.get('status') != 'stopped_for_repair':
        raise ValueError('stop the old execution before recovering its paid decision')
    case = old['online']['cases'][0]
    workspace = source / 'online' / case['id'] / 'hegel'
    training_path = workspace / 'interrupted_training.json'
    if training_path.exists():
        previous = read_json(training_path)
    else:
        previous = recover_interrupted_online_run(workspace, model=old['online']['model'],
            interruption=interruption, result_path=training_path)
    if previous.get('model') is not None or previous['status'] != 'runtime_interrupted':
        raise ValueError('continue an interrupted investigation without a submitted model')

    plan = deepcopy(old)
    plan.update(interrupted_study=str(source), resume_result=str(training_path),
        resume_workspace=str(workspace),
        development_origin='Resume the same paid investigation after a runtime repair. '
            'The interrupted decision and elapsed execution remain charged; '
            'no additional observations, model calls, tokens or execution allowance are granted.')
    online = plan['online']
    online.update(study='conditional_interrupted_online_continuation',
                  scope=plan['development_origin'])
    online['evaluation'] = source_evaluation(online['cases'][0], online['evaluation'])
    online['limits'].setdefault('run_timeout_seconds', InquiryLLMConfig().run_timeout_seconds)
    for stage in ('online', 'method'):
        plan[stage]['model'] = model
        plan[stage]['provider']['timeout_seconds'] = timeout_seconds
        plan[stage]['runtime_snapshot'] = sorted(set(plan[stage]['runtime_snapshot']) | {
            'scripts/run_conditional_online_method_repair.py',
            *(str(path.relative_to(ROOT)) for path in (ROOT / 'src/hegel_machine').rglob('*.py'))})

    prior_path = old.get('resume_result', old['practice_result'])
    prior = read_json(prior_path)
    increment = {key: value - prior['costs'].get(key, 0) if value is not None else None
                 for key, value in previous['costs'].items()}
    plan['method']['prior_author_phases'].append({
        'phase': 'online_source_research', 'result': str(training_path),
        'status': previous['status'], 'costs': increment,
        'usage': usage_summary({'events': previous['events'][len(prior['events']):]}),
        'cost_scope': 'Only the interrupted phase increment, including its pending paid decision. '
            'Earlier restored phases are already recorded; later continuation is charged separately.'})
    plan['accounting'] = ('Keep the old cumulative limits and paid account. '
        'online_additional_costs reports only work after this recovered checkpoint; '
        'the interrupted phase increment is retained once in prior_author_phases.')
    return plan


def run_pipeline(plan, output, *, provider_factory):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('use a fresh output; retain paid earlier phases')
    write_json(output / 'protocol.json', plan)
    for name in plan['online']['runtime_snapshot']:
        target = output / 'source_at_run' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    resume_result = plan.get('resume_result', plan['practice_result'])
    resume_workspace = plan.get('resume_workspace', plan['practice_workspace'])
    previous = read_json(resume_result)

    def restore(agent, lab, case):
        restore_paid = agent.restore_research if 'resume_result' in plan else agent.restore_practice_research
        restored = restore_paid(previous, resume_workspace, lab,
            confirmation_seed=case['confirmation_seed'])
        restored.update(resume_origin=resume_result, restored_events=len(previous['events']),
                        restored_model_calls=previous['costs']['model_calls'])
        write_json(output / 'restoration.json', restored)
        # Checkpoint before the first paid call, so a provider interruption does
        # not hide the restored data or require buying the original prefix again.
        write_json(agent.workspace / 'active.json', agent.result('researching'))
        return {'prior_research_cost': deepcopy(previous['costs']),
                'resume_origin': resume_result}

    write_json(output / 'progress.json', {'stage': 'online_research'})
    online = run_study(plan['online'], output / 'online', provider_factory=provider_factory,
        knowledge=read_json(Path(plan['source_study']) / 'initial_archive.json'),
        prepare_prefix=restore, laboratory_factory=laboratory)
    record = online['trials'][0]
    workspace = output / 'online' / record['case'] / 'hegel'
    training = read_json(workspace / 'training.json')
    ready = (training['status'] == 'completed' and bool(training.get('model'))
        and (training.get('fresh_confirmation') or {}).get('passed', False)
        and record['terminal']['passed'])
    result = {'study': plan['study'], 'status': 'online_research_incomplete',
        'online': record, 'online_training': str(workspace / 'training.json'),
        'prior_author_phases': plan['method']['prior_author_phases'],
        'online_additional_costs': training['continuation_cost'],
        'online_cumulative_costs': training['costs'],
        'online_prior_costs': previous['costs'],
        'online_prior_events': len(previous['events']),
        'online_added_events': len(training['events']) - len(previous['events']),
        'method_published': False, 'target_episodes_completed': 0,
        'target_episodes_pending': plan['target_episodes'],
        'original_source': plan['method']['source'],
        'original_result_scope': 'Original ten successful target results remain unchanged. '
            'Retention by a revised method requires its new frozen target results.'}
    if not ready:
        if training['status'] == 'completed':
            result['status'] = 'online_confirmation_failed'
        write_json(output / 'summary.json', result)
        write_json(output / 'progress.json', {'stage': result['status']})
        return result

    method_plan = deepcopy(plan['method'])
    method_plan['online_research_experience'] = str(workspace)
    native = method_plan.get('study') == 'luna_native_research_method_compilation'
    if native:
        # The standalone source-answer control must use the model from the
        # new confirmed workspace, not a path belonging to its earlier source.
        method_plan['previous_confirmed_source'] = {
            'workspace': plan['method']['online_research_experience'],
            'model': plan['method']['source_model'],
            'scope': 'Retained source provenance; its costs remain in the existing historical phases.'}
        method_plan.update(source_model=training['model'], online_study=str(output / 'online'),
                           source_research_costs=deepcopy(training['costs']))
    method_plan['prior_author_phases'].append({
        'phase': 'online_source_research', 'result': str(workspace / 'training.json'),
        'status': training['status'], 'costs': training['continuation_cost'],
        'usage': usage_summary({'events': training['events'][len(previous['events']):]}),
        'cost_scope': 'Additional online research only; restored practice costs already appear '
                      'in the earlier failed author phase.'})
    write_json(output / 'progress.json', {'stage': 'method_compilation',
        'online_training': str(workspace / 'training.json')})
    compile_method = run_compilation if native else run_revision
    method = compile_method(method_plan, output / 'method', provider_factory=provider_factory)
    completed = (len(method.get('retention', [])) + len(method.get('new_parameters', []))
                 if native else sum(len(rows) for rows in method['frozen'].values()))
    result.update(status='completed' if method['method_published'] else 'method_unpublished',
        method_published=method['method_published'],
        target_episodes_completed=completed,
        target_episodes_pending=(plan['target_episodes'] - completed if native else
                                 method['unavailable_target_episodes']),
        method_summary=str(output / 'method/summary.json'))
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', {'stage': result['status']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--source-study', type=Path)
    origin.add_argument('--rescue-study', type=Path,
                        help='Finished online pipeline: restore all paid evidence and decisions')
    origin.add_argument('--resume-interrupted-study', type=Path,
                        help='Stopped online pipeline: retain its paid pending decision and remaining budget')
    parser.add_argument('--trial', default='trial_095')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--timeout-seconds', type=float, default=1800.)
    args = parser.parse_args()
    if args.resume_interrupted_study:
        plan = interrupted_protocol(args.resume_interrupted_study, model=args.model,
                                    timeout_seconds=args.timeout_seconds)
    elif args.rescue_study:
        plan = rescue_protocol(args.rescue_study, model=args.model,
                               timeout_seconds=args.timeout_seconds)
    else:
        plan = protocol(args.source_study, trial=args.trial, model=args.model,
                        timeout_seconds=args.timeout_seconds)
    auth = verify_codex_subscription()
    for stage in ('online', 'method'):
        plan[stage]['provider']['auth_mode'] = auth

    def provider(workspace):
        return CodexCodeProvider(workspace, model=args.model,
                                 reasoning_effort=plan['online']['provider']['reasoning_effort'],
                                 timeout_seconds=args.timeout_seconds)

    result = run_pipeline(plan, args.output, provider_factory=provider)
    print(json.dumps({key: result[key] for key in (
        'status', 'method_published', 'target_episodes_completed', 'target_episodes_pending')}), flush=True)


if __name__ == '__main__':
    main()

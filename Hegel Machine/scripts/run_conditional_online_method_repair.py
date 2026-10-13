#!/usr/bin/env python3
"""Continue a paid failed practice online, then compile and transfer its method.

This is a new development phase. Earlier failures and their costs remain intact;
the existing publication rules and the 56 registered target episodes are retained.
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
from scripts.run_anonymous_method_learning import laboratory
from scripts.run_anonymous_method_revision import run_revision
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_python_method_learning_study import LIMIT_COSTS, usage_summary


def read_json(path):
    return json.loads(Path(path).read_text())


def protocol(source_study, *, trial='trial_095', model='gpt-6-luna',
             timeout_seconds=1800., online_limits=None, author_limits=None):
    source = Path(source_study).resolve()
    old = read_json(source / 'protocol.json')
    author = read_json(source / 'author/training.json')
    if not old.get('conditional_design') or author.get('method_source'):
        raise ValueError('continue an unpublished conditional-design study')
    index, practice = next((index, item) for index, item in enumerate(author['practice_trials'])
                           if Path(item['directory']).name == trial)
    workspace = source / 'author' / practice['directory']
    previous = read_json(workspace / 'result.json')
    if previous['status'] != 'confirmation_failed':
        raise ValueError('select the actual paid failed-confirmation practice')
    case = {'id': practice['case'], **deepcopy(old['practice_cases'][practice['case']])}
    # The practice learner offsets its RNG by the draft's original trial index.
    case['confirmation_seed'] += index
    allowance = asdict(online_limits or InquiryLLMConfig(
        max_model_calls=128, token_budget=20_000_000, observation_budget=120_000,
        execution_seconds=36_000., compute_budget=8_000_000,
        confirmation_rows=96, confirmation_mse_threshold=1e-6))
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
        'scripts/run_anonymous_revision_inquiry.py'})
    online = {
        'study': 'conditional_paid_practice_online_continuation',
        'model': model, 'provider': provider, 'cases': [case], 'arms': ['hegel'],
        'limits': limits, 'prefix_rows': old['prefix_rows'],
        'evaluation': {**old['evaluation'],
                       'mse_threshold': allowance['confirmation_mse_threshold']},
        'runtime_snapshot': runtime,
        'scope': 'Same paid development apparatus and failed hypothesis, continued with live Luna. '
            'New independent confirmation and source terminal evaluation precede method compilation. '
            'This source was selected after failure and is not held-out transfer evidence.'}
    historical = read_json(source / 'summary.json')['prior_author_phases']
    method = deepcopy(old)
    method.update(model=model, provider=provider, runtime_snapshot=runtime,
        continued_author=str(source / 'author/training.json'),
        author_limits=(asdict(author_limits) if author_limits is not None else
                       deepcopy(old.get('author_total_limits', old['author_limits']))),
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
        'target_episodes': 56}


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
    previous = read_json(plan['practice_result'])

    def restore(agent, lab, case):
        restored = agent.restore_practice_research(previous, plan['practice_workspace'], lab,
            confirmation_seed=case['confirmation_seed'])
        write_json(output / 'restoration.json', restored)
        # Checkpoint before the first paid call, so a provider interruption does
        # not hide the restored data or require buying the original prefix again.
        write_json(agent.workspace / 'active.json', agent.result('researching'))
        return {'prior_research_cost': deepcopy(previous['costs']),
                'resume_origin': plan['practice_result']}

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
    method_plan['prior_author_phases'].append({
        'phase': 'online_source_research', 'result': str(workspace / 'training.json'),
        'status': training['status'], 'costs': training['continuation_cost'],
        'usage': usage_summary({'events': training['events'][len(previous['events']):]}),
        'cost_scope': 'Additional online research only; restored practice costs already appear '
                      'in the earlier failed author phase.'})
    write_json(output / 'progress.json', {'stage': 'method_compilation',
        'online_training': str(workspace / 'training.json')})
    method = run_revision(method_plan, output / 'method', provider_factory=provider_factory)
    completed = sum(len(rows) for rows in method['frozen'].values())
    result.update(status='completed' if method['method_published'] else 'method_unpublished',
        method_published=method['method_published'],
        target_episodes_completed=completed,
        target_episodes_pending=method['unavailable_target_episodes'],
        method_summary=str(output / 'method/summary.json'))
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', {'stage': result['status']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-study', type=Path, required=True)
    parser.add_argument('--trial', default='trial_095')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--timeout-seconds', type=float, default=1800.)
    args = parser.parse_args()
    plan = protocol(args.source_study, trial=args.trial, model=args.model,
                    timeout_seconds=args.timeout_seconds)
    auth = verify_codex_subscription()
    for stage in ('online', 'method'):
        plan[stage]['provider']['auth_mode'] = auth

    def provider(workspace):
        return CodexCodeProvider(workspace, model=args.model, reasoning_effort='medium',
                                 timeout_seconds=args.timeout_seconds)

    result = run_pipeline(plan, args.output, provider_factory=provider)
    print(json.dumps({key: result[key] for key in (
        'status', 'method_published', 'target_episodes_completed', 'target_episodes_pending')}), flush=True)


if __name__ == '__main__':
    main()

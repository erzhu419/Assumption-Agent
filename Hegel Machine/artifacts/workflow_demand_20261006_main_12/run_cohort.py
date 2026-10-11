#!/usr/bin/env python3
"""Run the registered demand study against its frozen source snapshot."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'frozen_source' / 'scripts'))
from run_workflow_demand_study import ARMS, COST_KEYS, run_study


def eligible(action):
    return [item for item in action.get('alternatives', [])
        if item['selection_value']['basis'] != 'method_not_applicable']


def summarize(cases, failures, protocol):
    prefix = {'tasks': sum(len(phase['tasks']) for case in cases for phase in case['prefix'].values())}
    for key in ('solved', 'fresh_confirmed', 'confirmed_and_solved'):
        prefix[key] = sum(phase[key] for case in cases for phase in case['prefix'].values())
    prefix['costs'] = {key: sum(case['common_prefix_costs'][key] for case in cases) for key in COST_KEYS}
    prefix['evaluation_costs'] = {key: sum(phase['evaluation_costs'][key]
        for case in cases for phase in case['prefix'].values()) for key in COST_KEYS}
    totals = {}
    for arm in ARMS:
        rows = [case['results'][arm] for case in cases]
        counts = Counter()
        for row in rows:
            counts.update(row['action_counts'])
        totals[arm] = {'suffix_tasks': sum(len(row['tasks']) for row in rows),
            **{key: sum(row[key] for row in rows) for key in
                ('solved', 'fresh_confirmed', 'confirmed_and_solved', 'revisit_count', 'nonzero_value_replays')},
            'costs': {key: sum(row['costs'][key] for row in rows) for key in COST_KEYS},
            'task_sequence_costs': {key: sum(row['task_sequence_costs'][key] for row in rows) for key in COST_KEYS},
            'evaluation_costs': {key: sum(row['full_evaluation_costs'][key] for row in rows) for key in COST_KEYS},
            'training_cpu_seconds': sum(row['training_cpu_seconds'] for row in rows),
            'evaluation_cpu_seconds': sum(row['evaluation_cpu_seconds'] for row in rows),
            'mean_suffix_mse': mean(row['mean_mse'] for row in rows) if rows else None,
            'nonzero_value_cases': sum(row['nonzero_value_replays'] > 0 for row in rows),
            'competing_eligible_replay_cases': sum(any(len(eligible(action)) >= 2
                for action in row['replays']) for row in rows),
            'producer_choices': dict(Counter(method for case in cases
                for method in case['producer_workflow_choices'][arm])),
            'action_counts': dict(counts),
            'unconfirmed_or_failed_tasks': [{'seed': case['protocol']['seed'],
                'task_position': task['task_position'], 'status': task['status'], 'mse': task['mse'],
                'confirmation_loss': task['confirmation_loss'], 'costs': task['costs'],
                'action_tail': task['actions'][-5:]}
                for case in cases for task in case['results'][arm]['tasks'] if not task['confirmed_and_solved']]}
        totals[arm]['full_confirmed_and_solved'] = totals[arm]['confirmed_and_solved'] + prefix['confirmed_and_solved']
        totals[arm]['full_tasks'] = totals[arm]['suffix_tasks'] + prefix['tasks']
    metrics = (*COST_KEYS, 'confirmed_and_solved', 'mean_mse', 'training_cpu_seconds')
    differences = {key: [(case['paired_retained_minus_removed']['costs'][key] if key in COST_KEYS else
        case['results'][ARMS[0]]['training_cpu_seconds'] - case['results'][ARMS[1]]['training_cpu_seconds']
        if key == 'training_cpu_seconds' else case['paired_retained_minus_removed'][key]) for case in cases]
        for key in metrics}
    draws = {key: [] for key in metrics}
    random = Random(protocol['bootstrap_seed'])
    if cases:
        for _ in range(protocol['bootstrap_samples']):
            sample = [random.randrange(len(cases)) for _ in cases]
            for key in metrics:
                draws[key].append(mean(differences[key][index] for index in sample))
    intervals = {}
    for key, sampled in draws.items():
        sampled.sort()
        intervals[key] = {'mean_difference': mean(differences[key]) if cases else None,
            'interval_95': [sampled[int(.025 * (len(sampled)-1))], sampled[int(.975 * (len(sampled)-1))]]
                if sampled else None}
    inactive = [{'seed': case['protocol']['seed'], 'workflow_count': case['learned_workflow_count'],
        'confirmed_returns': case['confirmed_positive_returns'], 'partial_status': case['partial_demand']['status'],
        'condition_matches': case['demand_matches_confirmed_return'],
        'choices': case['producer_workflow_choices']}
        for case in cases if not case['producer_choice_changed']]
    return {'protocol': protocol, 'completed_cases': len(cases), 'execution_failures': failures,
        'prefix': prefix, 'totals': totals, 'paired_retained_minus_removed': intervals,
        'choice_changed_cases': sum(case['producer_choice_changed'] for case in cases),
        'confirmed_source_return_cases': sum(case['confirmed_positive_returns'] > 0 for case in cases),
        'matched_demand_cases': sum(case['demand_matches_confirmed_return'] for case in cases),
        'multiple_learned_workflow_cases': sum(case['learned_workflow_count'] >= 2 for case in cases),
        'unresolved_incoming_cases': sum(case['partial_demand']['status'] == 'unresolved' for case in cases),
        'inactive_cases': inactive,
        'physical_research_costs': {key: sum(case['common_prefix_costs'][key]
            + case['partial_demand']['costs'][key]
            + sum(case['results'][arm]['continuation_costs'][key] for arm in ARMS)
            for case in cases) for key in COST_KEYS},
        'physical_terminal_costs': {key: prefix['evaluation_costs'][key] + sum(
            case['results'][arm]['evaluation_costs'][key] for case in cases for arm in ARMS)
            for key in COST_KEYS}, 'external_model_calls': 0}


def main():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    output = HERE / 'results'
    output.mkdir(exist_ok=False)
    completed, failures = [], []
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(run_study, plan): plan for plan in protocol['cases']}
        for future in as_completed(pending):
            plan = pending[future]
            try:
                result = future.result()
                completed.append(result)
                receipt = {'seed': plan['seed'], 'status': 'completed',
                    'choice_changed': result['producer_choice_changed'],
                    'joint': {arm: result['results'][arm]['confirmed_and_solved'] for arm in ARMS},
                    'paired_costs': result['paired_retained_minus_removed']['costs']}
            except Exception:
                result = {'seed': plan['seed'], 'protocol': plan, 'status': 'execution_failed',
                    'traceback': traceback.format_exc()}
                failures.append(result)
                receipt = {'seed': plan['seed'], 'status': 'execution_failed'}
            (output / f"case_{plan['seed']}.json").write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(receipt), flush=True)
    completed.sort(key=lambda case: case['protocol']['seed'])
    summary = summarize(completed, failures, protocol)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'completed_cases': len(completed), 'execution_failures': len(failures),
        'choice_changed_cases': summary['choice_changed_cases'], 'summary': str(output / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()

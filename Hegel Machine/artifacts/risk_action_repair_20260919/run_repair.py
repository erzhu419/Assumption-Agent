#!/usr/bin/env python3
"""Paired development rerun of two previously exposed risk streams."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_risk_action_study as risk

risk.ARMS = risk.common.ARMS = ('learned',)


def summarize(protocol):
    original = json.loads((ROOT / protocol['original_results']).read_text())
    seeds = protocol['cohorts']['development']
    results = {'scope': protocol['known_exposure'], 'arms': {}}
    tasks, cpu = [], 0.
    for seed in seeds:
        path = risk.common.run_path(OUTPUT, 'development', 'learned', seed)
        state = json.loads(path.read_text())
        scores = {row['task_position']: row for row in state['evaluation']}
        cpu += sum(row['seconds'] for row in state['learning_process_cpu'])
        for row in state['results']:
            report = row['research']
            tasks.append({'replicate_seed': seed, 'position': row['task_position'],
                'stage': risk.STAGES[row['task_position']], 'status': report['status'],
                'kind': report['selected_kind'], 'channels': report['selected_channels'],
                'costs': report['costs'] | {'queries': row['actual_costs']['queries']},
                'evaluation': scores[row['task_position']],
                'learning_wall_seconds': row['learning_wall_seconds'],
                'risk_events': [event for event in report['actions'] if any(
                    token in event['action'] for token in ('risk', 'tail', 'noise'))]})
    def group(rows):
        return {'tasks': rows, 'metrics': {key: mean(row['evaluation'][key] for row in rows)
                for key in risk.METRICS},
            'learning_status': dict(Counter(row['status'] for row in rows)),
            'learning_costs': {key: sum(row['costs'][key] for row in rows) for key in risk.common.COSTS},
            'terminal_costs': {key: sum(row['evaluation']['costs'][key] for row in rows)
                               for key in risk.common.COSTS},
            'learning_wall_seconds': sum(row['learning_wall_seconds'] for row in rows),
            'by_seed': {str(seed): mean(row['evaluation']['pinball_loss'] for row in rows
                         if row['replicate_seed'] == seed) for seed in seeds}}
    results['arms']['repaired_v0961'] = group(tasks) | {'learning_process_cpu_seconds': cpu}
    for arm, data in original['arms'].items():
        results['arms'][arm] = group([row for row in data['tasks'] if row['replicate_seed'] in seeds])
    repaired = results['arms']['repaired_v0961']
    results['paired_new_minus_original_pinball'] = {
        arm: {str(seed): repaired['by_seed'][str(seed)] - data['by_seed'][str(seed)] for seed in seeds}
        for arm, data in results['arms'].items() if arm != 'repaired_v0961'}
    risk.common.save(OUTPUT / 'development/summary.json', results)
    digest = {'arms': {arm: {key: value for key, value in data.items() if key != 'tasks'}
               for arm, data in results['arms'].items()},
               'paired_new_minus_original_pinball': results['paired_new_minus_original_pinball']}
    risk.common.save(OUTPUT / 'development/final_digest.json', digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol):
    execution = {'status': 'running', 'completed': []}
    def run(mode, seed):
        log = OUTPUT / 'development/logs' / f'{mode}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--seed', str(seed)], stdout=stream, stderr=subprocess.STDOUT)
        return {'mode': mode, 'seed': seed, 'exit_code': code}
    for mode in ('learn', 'evaluate'):
        risk.common.save(OUTPUT / 'development/execution.json', execution | {'phase': mode})
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda seed: run(mode, seed), protocol['cohorts']['development']))
        execution['completed'].extend(outcomes)
        if any(row['exit_code'] for row in outcomes):
            risk.common.save(OUTPUT / 'development/execution.json', execution | {'status': 'failed'})
            raise SystemExit(1)
        print(json.dumps(outcomes), flush=True)
    summarize(protocol)
    risk.common.save(OUTPUT / 'development/execution.json', execution | {'status': 'completed', 'exit_code': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'learn', 'evaluate', 'queue', 'summarize'))
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    if args.mode == 'freeze':
        risk.common.freeze(OUTPUT)
    elif args.mode == 'queue':
        queue(protocol)
    elif args.mode == 'summarize':
        summarize(protocol)
    else:
        getattr(risk, args.mode)(protocol, OUTPUT, 'development', 'learned', args.seed)


if __name__ == '__main__':
    main()

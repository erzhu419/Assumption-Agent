#!/usr/bin/env python3
"""Known-sequence development follow-up for revising a residual question's parent."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import run_callable_residual_policy_study as study

ROOT = study.ROOT
PREVIOUS = study.OUTPUT
OUTPUT = ROOT / 'artifacts/residual_parent_revision_20260919'
ARMS = ('learned', 'policy_disabled')


def prepare(output):
    if (output / 'protocol.json').exists():
        return
    original = json.loads((PREVIOUS / 'protocol.json').read_text())
    protocol = {key: original[key] for key in ('task_order', 'profile', 'research_config',
        'measurement_budget', 'shared_laboratory_source', 'terminal', 'cost_accounting', 'information')}
    protocol.update(experiment='residual_parent_revision_known_sequences',
        fixed_at=datetime.now(timezone.utc).isoformat(), trajectory_seeds=[94701901, 94701907],
        observation_budgets={'94701901': 192, '94701907': 160},
        arms={arm: {**original['arms'][arm], 'runtime': str((output / 'runtime/src').relative_to(ROOT))}
              for arm in ARMS},
        scope='Development follow-up on two already-observed trajectory sequences, not additional independent confirmation. '
            'Full empty-library six-task streams, matched policy-disabled control, and JSON persistence. '
            'Frozen0.88 and0.89 results for these exact sequences are referenced unchanged.',
        reference_directory=str(PREVIOUS.relative_to(ROOT)),
        reference_arms=['baseline_v088', 'learned', 'policy_disabled'],
        maximum_concurrent_research_processes=1, formal_evidence=False, main_tasks=24,
        maximum_main_learning_costs={'observation_cost': 4224, 'measurement_cost': 18432,
                                      'compute_cost': 2880000})
    study.common.save(output / 'protocol.json', protocol)


def ensure_learning(protocol, output):
    for arm in ARMS:
        for seed in protocol['trajectory_seeds']:
            state = json.loads(study.common.run_path(output, arm, seed).read_text())
            if len(state['results']) != len(protocol['task_order']):
                raise RuntimeError('complete all development learning before terminal scoring')


def evaluate(protocol, output, arm, seed):
    ensure_learning(protocol, output)
    module, _ = study.load(protocol, arm)
    path = study.common.run_path(output, arm, seed)
    session = module.UnifiedResearchSession(state=json.loads(path.read_text()))
    study.common.save(path, session.evaluate(**{key: protocol['terminal'][key]
        for key in ('length', 'trajectories', 'seed_offset')}))


def summarize(protocol, output):
    result = {'scope': protocol['scope'], 'current': {}, 'references': {}}
    for group, directory, arms in [('current', output, ARMS),
            ('references', ROOT / protocol['reference_directory'], protocol['reference_arms'])]:
        for arm in arms:
            rows = []
            for seed in protocol['trajectory_seeds']:
                state = json.loads(study.common.run_path(directory, arm, seed).read_text())
                evaluated = {row['task_position']: row for row in state['evaluation']}
                rows.extend(study.compact_record(record, evaluated.get(record['task_position']), seed,
                    record['task_position'], protocol['observation_budgets'][str(seed)]) for record in state['results'])
            result[group][arm] = study.aggregate(rows)
    study.common.save(output / 'summary.json', result)
    digest = {group: {arm: {key: value for key, value in data.items() if key != 'tasks'}
        for arm, data in result[group].items()} for group in ('current', 'references')}
    study.common.save(output / 'final_digest.json', digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, output):
    jobs = [('learn', arm, seed) for i, seed in enumerate(protocol['trajectory_seeds'])
            for arm in (ARMS if i % 2 == 0 else tuple(reversed(ARMS)))]
    jobs.extend(('evaluate', arm, seed) for seed in protocol['trajectory_seeds'] for arm in ARMS)
    execution = {'status': 'running', 'completed': []}
    for mode, arm, seed in jobs:
        log = output / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        study.common.save(output / 'execution.json', {**execution, 'active': [mode, arm, seed]})
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(output), '--arm', arm, '--seed', str(seed)], stdout=stream, stderr=subprocess.STDOUT)
        if code:
            study.common.save(output / 'execution.json', {**execution, 'status': 'failed',
                'failed': [mode, arm, seed], 'exit_code': code})
            raise SystemExit(code)
        execution['completed'].append([mode, arm, seed])
        print(json.dumps({'finished': [mode, arm, seed]}), flush=True)
    summarize(protocol, output)
    study.common.save(output / 'execution.json', {**execution, 'status': 'completed', 'exit_code': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'queue', 'learn', 'evaluate', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        study.common.freeze(args.output)
    elif args.mode == 'queue':
        queue(protocol, args.output)
    elif args.mode == 'learn':
        study.learning(protocol, args.output, args.arm, args.seed)
    elif args.mode == 'evaluate':
        evaluate(protocol, args.output, args.arm, args.seed)
    else:
        summarize(protocol, args.output)


if __name__ == '__main__':
    main()

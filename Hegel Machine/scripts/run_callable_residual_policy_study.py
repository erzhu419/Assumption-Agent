#!/usr/bin/env python3
"""Full scalar task streams for callable and dependent-residual action selection."""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_unified_action_policy_study as common
import run_action_confirmation_study as recovery

ROOT = common.ROOT
OUTPUT = ROOT / 'artifacts/callable_residual_actions_20260919'
PREVIOUS = ROOT / 'artifacts/action_confirmation_returns_20260919'
ARMS = ('baseline_v088', 'learned', 'policy_disabled')
COSTS = (*common.COSTS, 'queries')


def prepare(output):
    if (output / 'protocol.json').exists():
        return
    previous = json.loads((PREVIOUS / 'protocol.json').read_text())
    seeds = list(range(94701901, 94701913))
    protocol = {'experiment': 'callable_and_dependent_residual_research_actions',
        'fixed_at': datetime.now(timezone.utc).isoformat(), 'trajectory_seeds': seeds,
        'observation_budgets': {str(seed): 192 if i < 6 else 160 for i, seed in enumerate(seeds)},
        'task_order': [2, 3, 4, 7, 5, 2], 'profile': 'development',
        'research_config': {'compute_budget': 120000, 'minimum_improvement': 1e-7,
            'max_rounds': 4, 'max_candidates': 16, 'active_probes': True},
        'measurement_budget': 768,
        'arms': {'baseline_v088': {'runtime': str((PREVIOUS / 'runtime/src').relative_to(ROOT))},
            'learned': {'runtime': str((output / 'runtime/src').relative_to(ROOT))},
            'policy_disabled': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                                'research_config_overrides': {'learn_research_policy': False}}},
        'shared_laboratory_source': str((PREVIOUS / 'runtime/src/hegel_machine/cognition').relative_to(ROOT)),
        'terminal': previous['terminal'], 'formal_evidence': False,
        'scope': 'Twelve new trajectory sequences over existing fixed scalar mechanisms; six per observation budget. '
            'Each arm learns from an empty library and restores complete JSON after every task. '
            'These are not twelve independent mechanism apparatus. Short reuse bypasses the new view policy and is reported separately.',
        'cost_accounting': 'All six tasks, including source discovery, included once per arm. Terminal acquisition and scoring separate.',
        'information': 'Public controls, instrument prices and paid observations only; neutral task label. No supplied source or action scores.',
        'maximum_concurrent_research_processes': 1,
        'maximum_main_learning_costs': {'observation_cost': 38016, 'measurement_cost': 165888,
                                        'compute_cost': 25920000},
        'main_tasks': 216, 'new_recovery_tasks': 6,
        'recovery': {**previous['recovery'], 'source_seeds': [31], 'learner_seeds': {'31': 7},
            'shared_lab': str((output / 'shared_nested_lab.py').relative_to(ROOT)),
            'baseline_reference': str((PREVIOUS / 'recovery/learned/31.json').relative_to(ROOT)),
            'scope': 'Known source31 development recovery only, with supplied paid family. Reuses frozen0.88 source and results. '
                     'Fresh0.89 learned and disabled sequences; excluded from new scalar trajectory totals.'}}
    common.save(output / 'protocol.json', protocol)
    shutil.copyfile(PREVIOUS / 'shared_nested_lab.py', output / 'shared_nested_lab.py')
    path = output / 'recovery_sources/31.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(PREVIOUS / 'recovery_sources/31.json')


def load(protocol, arm):
    source = ROOT / protocol['arms'][arm]['runtime']
    sys.path.insert(0, str(source))
    session = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(session.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('each runtime must run in its own process')
    name = 'hegel_machine.cognition.unknown_task_lab'
    spec = importlib.util.spec_from_file_location(name, ROOT / protocol['shared_laboratory_source'] / 'unknown_task_lab.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return session, module.UnknownTaskLab


def learning(protocol, output, arm, seed):
    module, laboratory_type = load(protocol, arm)
    path = common.run_path(output, arm, seed)
    laboratories = []

    class PublicLaboratory:
        def __init__(self, item):
            self.lab = laboratory_type(**item)
            self.channels = self.lab.channels
            self.spec = replace(self.lab.spec, task_id='unlabeled_experiment')
            laboratories.append(self.lab)
        def spec_for(self, channels=()):
            return replace(self.lab.spec_for(channels), task_id='unlabeled_experiment')
        def acquire(self, probe, channels=()):
            return self.lab.acquire(probe, channels)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    if path.exists():
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=PublicLaboratory)
    else:
        config = module.ResearchConfig(**protocol['research_config'],
            **protocol['arms'][arm].get('research_config_overrides', {}),
            observation_budget=protocol['observation_budgets'][str(seed)], seed=seed)
        tasks = [{'task_index': task, 'seed': seed + 10007 * index, 'profile': protocol['profile']}
                 for index, task in enumerate(protocol['task_order'])]
        session = module.UnifiedResearchSession(config, tasks=tasks,
            measurement_budget=protocol['measurement_budget'], lab_factory=PublicLaboratory)
    while len(session.state['results']) < len(session.state['tasks']):
        incoming = json.loads(json.dumps(session.library.action_experience))
        started = time.perf_counter()
        state = session.run(max_tasks=1)
        record = state['results'][-1]
        record.update(learning_wall_seconds=time.perf_counter() - started,
            action_experience_before=incoming, action_experience_after=state['knowledge'].get('action_experience', {}),
            actual_costs=laboratories[-1].costs, acquisitions=laboratories[-1].acquisitions)
        for key in ('observation_cost', 'measurement_cost'):
            if record['actual_costs'][key] != record['research']['costs'][key]:
                raise RuntimeError('charged research differs from actual acquisition')
        common.save(path, state)
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=PublicLaboratory)
        report = record['research']
        print(json.dumps({'arm': arm, 'seed': seed, 'position': record['task_position'],
            'status': report['status'], 'costs': report['costs'], 'selected_kind': report['selected_kind']}), flush=True)


def ensure_all_learning(protocol, output):
    for arm in ARMS:
        for seed in protocol['trajectory_seeds']:
            state = json.loads(common.run_path(output, arm, seed).read_text())
            if len(state['results']) != len(protocol['task_order']):
                raise RuntimeError('all main learning must precede terminal scoring')
    for arm in ('learned', 'policy_disabled'):
        state = json.loads(recovery.recovery_path(output, arm, 31).read_text())
        if len(state['results']) != 3:
            raise RuntimeError('all recovery learning must precede terminal scoring')


def evaluate(protocol, output, arm, seed):
    ensure_all_learning(protocol, output)
    module, _ = load(protocol, arm)
    path = common.run_path(output, arm, seed)
    session = module.UnifiedResearchSession(state=json.loads(path.read_text()))
    common.save(path, session.evaluate(**{key: protocol['terminal'][key]
        for key in ('length', 'trajectories', 'seed_offset')}))


def compact_record(record, terminal, seed, position, budget):
    report = record['research']
    events = report['actions']
    return {'seed': seed, 'position': position, 'observation_budget': budget,
        'status': report['status'], 'selected_kind': report['selected_kind'],
        'costs': {**report['costs'], 'queries': record['actual_costs']['queries']},
        'learning_wall_seconds': record['learning_wall_seconds'],
        'terminal': terminal, 'solved': terminal is not None and terminal.get('mse', terminal.get('loss', float('inf'))) <= 1e-6,
        'policy_events': [{k: v for k, v in event.items() if k not in ('program', 'baseline_program')}
            for event in events if event['action'] == 'choose_research_action'
                or event['action'].startswith('research_action_')],
        'short_trial': any(event['action'] == 'test_learned_acquisition_method' for event in events),
        'short_bypass': report['selected_kind'] in ('short_callable_transfer',
            'retained_callable_transfer', 'retained_calibrated_call'),
        'action_counts': dict(Counter(event['action'] for event in events)),
        'incoming_experience': record['action_experience_before'],
        'outgoing_experience': record['action_experience_after']}


def aggregate(rows):
    sequences = sorted({row['seed'] for row in rows})
    return {'tasks': rows, 'learning_status': dict(Counter(row['status'] for row in rows)),
        'solved': sum(row['solved'] for row in rows), 'evaluated_tasks': sum(row['terminal'] is not None for row in rows),
        'complete_sequences': sum(all(row['solved'] for row in rows if row['seed'] == seed) for seed in sequences),
        'learning_costs': {key: sum(row['costs'][key] for row in rows) for key in COSTS},
        'terminal_costs': {key: sum(row['terminal'].get(key, 0) for row in rows if row['terminal']) for key in common.COSTS},
        'short_bypass_tasks': sum(row['short_bypass'] for row in rows),
        'learning_wall_seconds': sum(row['learning_wall_seconds'] for row in rows)}


def summarize(protocol, output):
    result = {'scope': protocol['scope'], 'main': {}, 'budget_layers': {}, 'recovery': {}}
    for arm in ARMS:
        rows = []
        for seed in protocol['trajectory_seeds']:
            state = json.loads(common.run_path(output, arm, seed).read_text())
            evaluated = {row['task_position']: row for row in state['evaluation']}
            rows.extend(compact_record(record, evaluated.get(record['task_position']), seed,
                record['task_position'], protocol['observation_budgets'][str(seed)]) for record in state['results'])
        result['main'][arm] = aggregate(rows)
        result['budget_layers'][arm] = {str(budget): aggregate([row for row in rows if row['observation_budget'] == budget])
                                       for budget in (192, 160)}
        path = ROOT / protocol['recovery']['baseline_reference'] if arm == 'baseline_v088' else recovery.recovery_path(output, arm, 31)
        state = json.loads(path.read_text())
        evaluated = {row['position']: row for row in state['evaluation']}
        rows = [compact_record(record, evaluated.get(position), 31, position,
            protocol['recovery']['stages'][position]['observation_budget']) for position, record in enumerate(state['results'])]
        result['recovery'][arm] = aggregate(rows)
        for key in COSTS:
            result['recovery'][arm]['learning_costs'][key] += state['source_costs'].get(key, 0)
        result['recovery'][arm]['source_costs'] = state['source_costs']
        result['recovery'][arm]['scope'] = protocol['recovery']['scope']
    common.save(output / 'summary.json', result)
    digest = {group: {arm: {k: v for k, v in data.items() if k != 'tasks'} for arm, data in result[group].items()}
              for group in ('main', 'recovery')}
    digest['budget_layers'] = {arm: {budget: {k: v for k, v in data.items() if k != 'tasks'}
        for budget, data in layers.items()} for arm, layers in result['budget_layers'].items()}
    common.save(output / 'final_digest.json', digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, output, phase):
    jobs = []
    for index, seed in enumerate(protocol['trajectory_seeds']):
        arms = ('baseline_v088',) if phase == 'baseline' else (('learned', 'policy_disabled') if index % 2 == 0 else ('policy_disabled', 'learned'))
        jobs.extend(('learn', arm, seed) for arm in arms)
    if phase == 'remaining':
        jobs.extend(('learn-recovery', arm, 31) for arm in ('learned', 'policy_disabled'))
        jobs.extend(('evaluate', arm, seed) for seed in protocol['trajectory_seeds'] for arm in ARMS)
        jobs.extend(('evaluate-recovery', arm, 31) for arm in ('learned', 'policy_disabled'))
    execution = {'phase': phase, 'status': 'running', 'completed': []}
    path = output / f'execution_{phase}.json'
    for mode, arm, seed in jobs:
        log = output / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        common.save(path, {**execution, 'active': [mode, arm, seed]})
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(output), '--arm', arm, '--seed', str(seed)], stdout=stream, stderr=subprocess.STDOUT)
        if code:
            common.save(path, {**execution, 'status': 'failed', 'failed': [mode, arm, seed], 'exit_code': code})
            raise SystemExit(code)
        execution['completed'].append([mode, arm, seed])
        print(json.dumps({'finished': [mode, arm, seed]}), flush=True)
    if phase == 'remaining':
        summarize(protocol, output)
    common.save(path, {**execution, 'status': 'completed', 'exit_code': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'learn', 'evaluate', 'learn-recovery',
        'evaluate-recovery', 'summarize', 'queue'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--phase', choices=('baseline', 'remaining'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        common.freeze(args.output)
    elif args.mode == 'queue':
        queue(protocol, args.output, args.phase)
    elif args.mode == 'summarize':
        summarize(protocol, args.output)
    elif args.mode == 'learn':
        learning(protocol, args.output, args.arm, args.seed)
    elif args.mode == 'evaluate':
        evaluate(protocol, args.output, args.arm, args.seed)
    elif args.mode == 'learn-recovery':
        recovery.learn_recovery(protocol, args.output, args.arm, args.seed)
    elif args.mode == 'evaluate-recovery':
        ensure_all_learning(protocol, args.output)
        recovery.evaluate_recovery(protocol, args.output, args.arm, args.seed)


if __name__ == '__main__':
    main()

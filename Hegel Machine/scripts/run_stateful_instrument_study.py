#!/usr/bin/env python3
"""Whole research sessions for learned recurrent instrument mappings."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/stateful_instrument_study_20260919'
ARMS = ('baseline_v0891', 'learned')
COSTS = ('observation_cost', 'measurement_cost', 'compute_cost', 'queries')


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def prepare(output):
    if (output / 'protocol.json').exists():
        return
    baseline = ROOT / 'artifacts/residual_parent_revision_20260919/runtime/src'
    if not (baseline / 'hegel_machine/cognition/unified_session.py').exists():
        raise FileNotFoundError(baseline)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'src/hegel_machine/cognition/stateful_instrument_lab.py',
                    output / 'shared_stateful_instrument_lab.py')
    save(output / 'protocol.json', {
        'experiment': 'recurrent_instrument_learning_and_joint_revision',
        'fixed_at': datetime.now(timezone.utc).isoformat(),
        'cohorts': {'development': [94901900], 'main': [94901901, 94901902, 94901903, 94901904]},
        'instrument_states': {'94901900': 2, '94901901': 1, '94901902': 1,
                              '94901903': 2, '94901904': 2},
        'development_runtime': 'src',
        'arms': {'baseline_v0891': {'runtime': str(baseline.relative_to(ROOT))},
                 'learned': {'runtime': str((output / 'runtime/src').relative_to(ROOT))}},
        'shared_laboratory': str((output / 'shared_stateful_instrument_lab.py').relative_to(ROOT)),
        'stages': ['source', 'instrument_only_change', 'instrument_and_response_change', 'return_source'],
        'research_config': {'observation_budget': 384, 'compute_budget': 360000,
            'minimum_improvement': 1e-7, 'max_rounds': 5, 'max_history': 4, 'active_probes': True},
        'measurement_budget': 1536, 'maximum_concurrent_research_processes': 1,
        'terminal': {'length': 64, 'trajectories': 4, 'seed_offset': 1000003,
            'impulse_length': 96, 'impulse_amplitudes': [-.85, .85], 'mse_threshold': 1e-6},
        'scope': 'Recurrent instrument hypotheses from paid readings. All arms start empty; no supplied source '
            'program, private parameter, required measurement, diagnostic choice or action score. '
            'Finite-history and direct-control response explanations compete normally. '
            'Development and main cohorts are evaluated separately, after all declared learning in each cohort.',
        'persistence': 'Restore complete UnifiedResearchSession JSON between stages.',
        'cost_accounting': 'Source learning and all unsuccessful research included; terminal resources separate.',
        'capability_endpoints': ['random_and_long_tail_response_prediction', 'learned_recurrent_instrument_fields',
            'paid_instrument_counterexamples', 'joint_revision_and_cross_task_call', 'reset_and_json_continuation'],
        'formal_evidence': False})


def freeze(output):
    destination = output / 'runtime/src/hegel_machine'
    if destination.exists():
        raise ValueError('runtime already frozen; preserve it and use a new output for a revised release')
    shutil.copytree(ROOT / 'src/hegel_machine', destination,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def load(protocol, arm, cohort):
    source = ROOT / (protocol['development_runtime'] if cohort == 'development' and arm == 'learned'
                     else protocol['arms'][arm]['runtime'])
    sys.path.insert(0, str(source))
    session = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(session.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('execute each runtime in a separate process')
    name = 'hegel_machine.cognition.stateful_instrument_lab'
    spec = importlib.util.spec_from_file_location(name, ROOT / protocol['shared_laboratory'])
    laboratory = importlib.util.module_from_spec(spec)
    sys.modules[name] = laboratory
    spec.loader.exec_module(laboratory)
    return session, laboratory.StatefulInstrumentLab


def run_path(output, cohort, arm, seed):
    return output / cohort / 'runs' / arm / str(seed) / 'checkpoint.json'


def learn(protocol, output, cohort, arm, seed):
    module, lab_type = load(protocol, arm, cohort)
    path, laboratories = run_path(output, cohort, arm, seed), []
    def factory(item):
        lab = lab_type(**item)
        laboratories.append(lab)
        return lab
    if path.exists():
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=factory)
    else:
        config = module.ResearchConfig(**protocol['research_config'], seed=seed)
        tasks = [{'task_index': i, 'seed': seed + 10007 * i, 'apparatus_seed': seed,
                  'states': protocol['instrument_states'][str(seed)]} for i in range(4)]
        session = module.UnifiedResearchSession(config, tasks=tasks,
            measurement_budget=protocol['measurement_budget'], lab_factory=factory)
    while len(session.state['results']) < 4:
        started = time.perf_counter()
        state = session.run(max_tasks=1)
        record = state['results'][-1]
        record.update(learning_wall_seconds=time.perf_counter() - started,
                      actual_costs=laboratories[-1].costs, acquisitions=laboratories[-1].acquisitions)
        for key in ('observation_cost', 'measurement_cost'):
            if record['actual_costs'][key] != record['research']['costs'][key]:
                raise RuntimeError('reported costs differ from actual instrument purchases')
        save(path, state)
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=factory)
        report = record['research']
        print(json.dumps({'arm': arm, 'seed': seed, 'position': record['task_position'],
            'status': report['status'], 'selected_kind': report['selected_kind'],
            'selected_channels': report['selected_channels'], 'costs': report['costs']}), flush=True)


def ensure_learning(protocol, output, cohort):
    for arm in ARMS:
        for seed in protocol['cohorts'][cohort]:
            state = json.loads(run_path(output, cohort, arm, seed).read_text())
            if len(state['results']) != 4:
                raise RuntimeError('finish all cohort learning before terminal scoring')


def evaluate(protocol, output, cohort, arm, seed):
    ensure_learning(protocol, output, cohort)
    module, lab_type = load(protocol, arm, cohort)
    from hegel_machine.cognition.models import Probe
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.evaluation import loss
    path = run_path(output, cohort, arm, seed)
    session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=lambda item: lab_type(**item))
    state = session.evaluate(**{key: protocol['terminal'][key] for key in ('length', 'trajectories', 'seed_offset')})
    tails = state.setdefault('long_tail_evaluation', [])
    for position in range(len(tails), 4):
        report = state['results'][position]['research']
        lab = lab_type(**state['tasks'][position])
        rows = []
        for index, amplitude in enumerate(protocol['terminal']['impulse_amplitudes']):
            controls = ({'u': amplitude}, *({'u': 0.} for _ in range(protocol['terminal']['impulse_length'] - 1)))
            rows.extend(lab.acquire(Probe(f'long_tail_{index}', controls), report['selected_channels']).observations)
        measured = loss(Program.from_dict(report['program']), rows, report['fallback'], .5)
        tails.append({'task_position': position, 'mse': measured,
                      'costs': {**lab.costs, 'compute_cost': len(rows)}})
    save(path, state)


def summarize(protocol, output, cohort):
    from collections import defaultdict
    result = {'scope': protocol['scope'], 'cohort': cohort, 'arms': {}}
    for arm in ARMS:
        rows, costs, terminal_costs, models = [], Counter(), Counter(), []
        for seed in protocol['cohorts'][cohort]:
            state = json.loads(run_path(output, cohort, arm, seed).read_text())
            measured = {row['task_position']: row for row in state['evaluation']}
            tails = {row['task_position']: row for row in state['long_tail_evaluation']}
            models.extend({'apparatus_seed': seed, **model} for model in state['knowledge'].get('instrument_models', []))
            for record in state['results']:
                p, report = record['task_position'], record['research']
                events = report['actions']
                row = {'apparatus_seed': seed, 'position': p, 'stage': protocol['stages'][p],
                    'instrument_states': protocol['instrument_states'][str(seed)],
                    'status': report['status'], 'kind': report['selected_kind'], 'channels': report['selected_channels'],
                    'costs': {**report['costs'], 'queries': record['actual_costs']['queries']},
                    'learning_wall_seconds': record['learning_wall_seconds'],
                    'random_mse': measured[p]['mse'], 'long_tail_mse': tails[p]['mse'],
                    'solved': max(measured[p]['mse'], tails[p]['mse']) <= protocol['terminal']['mse_threshold'],
                    'actions': dict(Counter(event['action'] for event in events)),
                    'instrument_events': [event for event in events if 'instrument' in event['action']],
                    'used_sources': report['used_sources']}
                rows.append(row)
                costs.update(row['costs'])
                terminal_costs.update({key: measured[p][key] for key in COSTS if key in measured[p]})
                terminal_costs['queries'] += protocol['terminal']['trajectories']
                terminal_costs.update(tails[p]['costs'])
        sequences = defaultdict(list)
        for row in rows:
            sequences[row['apparatus_seed']].append(row)
        result['arms'][arm] = {'tasks': rows, 'instrument_models': models,
            'learning_status': dict(Counter(row['status'] for row in rows)),
            'solved': sum(row['solved'] for row in rows), 'tasks_count': len(rows),
            'instrument_free_solved_tasks': sum(row['solved'] and not row['channels'] for row in rows),
            'complete_sequences': sum(all(row['solved'] for row in sequence) for sequence in sequences.values()),
            'learning_costs': dict(costs), 'terminal_costs': dict(terminal_costs),
            'learning_wall_seconds': sum(row['learning_wall_seconds'] for row in rows)}
    save(output / cohort / 'summary.json', result)
    digest = {arm: {key: value for key, value in data.items() if key not in ('tasks', 'instrument_models')}
              for arm, data in result['arms'].items()}
    save(output / cohort / 'final_digest.json', digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, output, cohort):
    jobs = [('learn', arm, seed) for i, seed in enumerate(protocol['cohorts'][cohort])
            for arm in (tuple(reversed(ARMS)) if cohort == 'development' or i % 2 else ARMS)]
    jobs.extend(('evaluate', arm, seed) for seed in protocol['cohorts'][cohort] for arm in ARMS)
    execution = {'status': 'running', 'completed': []}
    for mode, arm, seed in jobs:
        log = output / cohort / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        save(output / cohort / 'execution.json', {**execution, 'active': [mode, arm, seed]})
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(output), '--cohort', cohort, '--arm', arm, '--seed', str(seed)],
                stdout=stream, stderr=subprocess.STDOUT)
        if code:
            save(output / cohort / 'execution.json', {**execution, 'status': 'failed', 'exit_code': code})
            raise SystemExit(code)
        execution['completed'].append([mode, arm, seed])
        print(json.dumps({'finished': [mode, arm, seed]}), flush=True)
    summarize(protocol, output, cohort)
    save(output / cohort / 'execution.json', {**execution, 'status': 'completed', 'exit_code': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'learn', 'evaluate', 'queue', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cohort', choices=('development', 'main'), default='development')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        freeze(args.output)
    elif args.mode == 'queue':
        queue(protocol, args.output, args.cohort)
    elif args.mode == 'summarize':
        summarize(protocol, args.output, args.cohort)
    else:
        (learn if args.mode == 'learn' else evaluate)(protocol, args.output, args.cohort, args.arm, args.seed)


if __name__ == '__main__':
    main()

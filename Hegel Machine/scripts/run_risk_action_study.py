#!/usr/bin/env python3
"""Compare risk research actions on a fixed noisy task stream, without a solved threshold."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import threading
import time

import run_observation_action_study as common

ROOT = common.ROOT
OUTPUT = ROOT / 'artifacts/risk_action_study_20260919'
ARMS = ('baseline_v095', 'learned', 'policy_disabled')
STAGES = ('control_risk_source', 'new_independent_signal', 'changed_response', 'return_control_risk')
METRICS = ('pinball_loss', 'action_loss', 'oracle_pinball_loss', 'excess_pinball_loss',
           'mean_underage', 'mean_overage', 'empirical_coverage', 'oracle_empirical_coverage',
           'oracle_prediction_mse')
common.ARMS = ARMS


def prepare(output):
    if (output / 'protocol.json').exists():
        return
    baseline = ROOT / 'artifacts/object_action_study_20260919/runtime/src'
    if not (baseline / 'hegel_machine/cognition/unified_session.py').exists():
        raise FileNotFoundError(baseline)
    seeds = [95401900, *range(95401901, 95401909)]
    common.save(output / 'protocol.json', {
        'experiment': 'common_risk_research_action_selection',
        'fixed_at': datetime.now(timezone.utc).isoformat(),
        'cohorts': {'development': seeds[:1], 'main': seeds[1:]},
        'stages': list(STAGES),
        'tasks_by_seed': {str(seed): [{'environment': 'scalar', 'task_index': task,
            'profile': profile, 'seed': seed + 10007 * position}
            for position, (task, profile) in enumerate(((12, 'development'), (13, 'development'),
                                                       (13, 'confirmation'), (12, 'development')))]
            for seed in seeds},
        'development_runtime': 'src',
        'arms': {'baseline_v095': {'runtime': str(baseline.relative_to(ROOT))},
                 'learned': {'runtime': str((output / 'runtime/src').relative_to(ROOT))},
                 'policy_disabled': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                                     'research_config_overrides': {'learn_research_policy': False}}},
        'shared_laboratory_source': str((baseline / 'hegel_machine/cognition/unknown_task_lab.py').relative_to(ROOT)),
        'research_config': {'observation_budget': 384, 'compute_budget': 240000,
            'minimum_improvement': 1e-7, 'max_rounds': 4, 'active_probes': False},
        'measurement_budget': 1536, 'maximum_concurrent_research_processes': 4,
        'terminal': {'length': 96, 'trajectories': 16, 'seed_offset': 1000003,
                     'risk_quantile': .9, 'underage_cost': 9., 'overage_cost': 1.},
        'scope': 'Eight new stochastic trajectory repeats of the existing two noisy mechanisms and '
            'two parameter profiles, not eight independently parameterized physical apparatus. Each '
            'four-task session starts with empty knowledge and restores full JSON between stages. '
            'No source model, diagnostic choice or action score is supplied. Development is separate.',
        'risk_contract': 'Primary endpoint is pinball loss at the fixed public quantile .9. Deploy the '
            'returned Program prediction directly as the action. Fixed (9,1) underage/overage action '
            'loss is exactly ten times pinball and is not an independent performance endpoint. '
            'Coverage and mean underage/overage remain descriptive. There is no risk solved threshold; '
            'completed/unresolved are learning statuses, and all returned models remain in evaluation.',
        'oracle_contract': 'Only the terminal scorer uses the existing evaluation_oracle after every '
            'learning task in the cohort has finished. Its full-state conditional quantile includes '
            'unpurchased instrument state; it is a reference, not an attainable target for every view. '
            'Report realized excess loss without clipping. Oracle values never enter learning or memory.',
        'evaluation_contract': 'All arms see the same independent terminal controls and disturbances '
            'per task. Save per-trajectory metrics, not observation dumps. Score every row including '
            'startup. Compare matched complete streams and stages; do not tune from terminal results.',
        'cost_accounting': 'All source learning and unsuccessful research included; terminal costs separate.',
        'formal_evidence': False})


def load(protocol, arm, cohort):
    source = ROOT / (protocol['development_runtime'] if cohort == 'development' and arm != 'baseline_v095'
                     else protocol['arms'][arm]['runtime'])
    sys.path.insert(0, str(source))
    canonical = 'hegel_machine.cognition.unknown_task_lab'
    spec = importlib.util.spec_from_file_location(canonical, ROOT / protocol['shared_laboratory_source'])
    laboratory = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = laboratory
    spec.loader.exec_module(laboratory)
    session = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(session.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('execute each runtime in a separate process')
    def factory(item):
        arguments = dict(item)
        arguments.pop('environment')
        return laboratory.UnknownTaskLab(**arguments)
    return session, factory


common.load = load
common.task_sequence = lambda protocol, seed: protocol['tasks_by_seed'][str(seed)]


def learn(protocol, output, cohort, arm, seed):
    path = common.run_path(output, cohort, arm, seed)
    previous = len(json.loads(path.read_text())['results']) if path.exists() else 0
    started = time.process_time()
    common.learn(protocol, output, cohort, arm, seed, stop_after=len(STAGES))
    elapsed = time.process_time() - started
    state = json.loads(path.read_text())
    if len(state['results']) > previous:
        state.setdefault('learning_process_cpu', []).append({
            'task_positions': list(range(previous, len(state['results']))), 'seconds': elapsed})
        common.save(path, state)


def evaluate(protocol, output, cohort, arm, seed):
    common.ensure_learning(protocol, output, cohort)
    _, factory = load(protocol, arm, cohort)
    from hegel_machine.cognition.evaluation import point_loss, predict
    from hegel_machine.cognition.language import Program
    path = common.run_path(output, cohort, arm, seed)
    state = json.loads(path.read_text())
    settings = protocol['terminal']
    for position in range(len(state['evaluation']), len(STAGES)):
        started = time.perf_counter()
        task, report = state['tasks'][position], state['results'][position]['research']
        with factory(task) as laboratory:
            reference = laboratory.evaluation_oracle(report['selected_channels'],
                seed=task['seed'] + settings['seed_offset'], length=settings['length'],
                trajectories=settings['trajectories'])
        q = reference['risk_quantile']
        if q != settings['risk_quantile']:
            raise ValueError('the fixed risk objective differs from the laboratory interface')
        program = Program.from_dict(report['program'])
        rows, optimal = reference['observations'], reference['conditional_quantiles']
        predictions = [predict(program, row, report['fallback']) for row in rows]
        trajectories = []
        for index in range(settings['trajectories']):
            start, end = index * settings['length'], (index + 1) * settings['length']
            targets = [row.target for row in rows[start:end]]
            values, truth = predictions[start:end], optimal[start:end]
            under = mean(max(y - p, 0.) for y, p in zip(targets, values))
            over = mean(max(p - y, 0.) for y, p in zip(targets, values))
            observed = mean(point_loss(p, y, q) for p, y in zip(values, targets))
            oracle = mean(point_loss(p, y, q) for p, y in zip(truth, targets))
            trajectories.append({'trajectory': index, 'rows': len(targets),
                'pinball_loss': observed, 'oracle_pinball_loss': oracle,
                'excess_pinball_loss': observed - oracle,
                'action_loss': settings['underage_cost'] * under + settings['overage_cost'] * over,
                'mean_underage': under, 'mean_overage': over,
                'empirical_coverage': mean(y <= p for y, p in zip(targets, values)),
                'oracle_empirical_coverage': mean(y <= p for y, p in zip(targets, truth)),
                'oracle_prediction_mse': mean((p - t) ** 2 for p, t in zip(values, truth))})
        state['evaluation'].append({'task_position': position, 'loss_kind': 'pinball', 'risk_quantile': q,
            **{key: mean(row[key] for row in trajectories) for key in METRICS},
            'trajectories': trajectories, 'costs': reference['costs'] | {'compute_cost': len(rows)},
            'evaluation_wall_seconds': time.perf_counter() - started})
        common.save(path, state)


def summarize(protocol, output, cohort):
    result = {'scope': protocol['scope'], 'cohort': cohort, 'arms': {}}
    def metrics(rows):
        return {key: mean(row['evaluation'][key] for row in rows) for key in METRICS}
    for arm in ARMS:
        tasks, learning, terminal, cpu = [], Counter(), Counter(), []
        for seed in protocol['cohorts'][cohort]:
            state = json.loads(common.run_path(output, cohort, arm, seed).read_text())
            cpu_records = state.get('learning_process_cpu', [])
            if {p for row in cpu_records for p in row['task_positions']} == set(range(len(STAGES))):
                cpu.append(sum(row['seconds'] for row in cpu_records))
            scores = {row['task_position']: row for row in state['evaluation']}
            for record in state['results']:
                p, report = record['task_position'], record['research']
                events = report['actions']
                task = {'replicate_seed': seed, 'position': p, 'stage': STAGES[p],
                    'status': report['status'], 'kind': report['selected_kind'],
                    'channels': report['selected_channels'],
                    'costs': report['costs'] | {'queries': record['actual_costs']['queries']},
                    'learning_wall_seconds': record['learning_wall_seconds'],
                    'evaluation': scores[p], 'noise_assessment': report.get('noise_assessment'),
                    'actions': dict(Counter(e['action'] for e in events)),
                    'risk_events': [{k: v for k, v in e.items() if k not in ('program', 'baseline_program')}
                        for e in events if any(word in e['action'] for word in ('risk', 'tail', 'noise'))
                        or e['action'] == 'test_learned_acquisition_method'],
                    'action_experience_before': record['action_experience_before'],
                    'action_experience_after': record['action_experience_after'],
                    'acquisitions': record['acquisitions'], 'used_sources': report['used_sources']}
                tasks.append(task)
                learning.update(task['costs'])
                terminal.update(scores[p]['costs'])
        result['arms'][arm] = {'tasks': tasks, 'tasks_count': len(tasks),
            'learning_status': dict(Counter(row['status'] for row in tasks)),
            'evaluated_sequences': len(protocol['cohorts'][cohort]),
            'metrics': metrics(tasks),
            'by_stage': {stage: metrics([row for row in tasks if row['stage'] == stage]) for stage in STAGES},
            'by_seed': {str(seed): metrics([row for row in tasks if row['replicate_seed'] == seed])
                        for seed in protocol['cohorts'][cohort]},
            'learning_costs': dict(learning), 'terminal_costs': dict(terminal),
            'learning_process_cpu_seconds': sum(cpu) if len(cpu) == len(protocol['cohorts'][cohort]) else None,
            'learning_wall_seconds': sum(row['learning_wall_seconds'] for row in tasks)}
        events = [event for task in tasks for event in task['risk_events']]
        choices = [event for event in events if event['action'] == 'choose_risk_research']
        feedback = [event for event in events if event['action'] == 'risk_research_feedback']
        unscored = [event for event in events if event['action'] == 'risk_research_unscored']
        result['arms'][arm]['risk_research'] = {
            'choices': dict(Counter(event.get('selected_action') or 'none' for event in choices)),
            'choices_with_measured_history': sum(event.get('trials', 0) > 0 for event in choices),
            'paired_feedback_status': dict(Counter(event['status'] for event in feedback
                if event.get('before_loss') is not None and event.get('after_loss') is not None)),
            'unscored_status': dict(Counter(event['status'] for event in unscored)),
            'unscored_costs': {key: sum(event.get(key, 0) for event in unscored)
                               for key in common.COSTS if key != 'queries'}}
    result['paired_pinball_changes'] = {}
    learned = result['arms']['learned']
    for control in ('baseline_v095', 'policy_disabled'):
        deltas = {str(seed): learned['by_seed'][str(seed)]['pinball_loss']
                  - result['arms'][control]['by_seed'][str(seed)]['pinball_loss']
                  for seed in protocol['cohorts'][cohort]}
        result['paired_pinball_changes'][control] = {'new_minus_control_by_seed': deltas,
            'mean_change': mean(deltas.values()), 'lower': sum(value < 0 for value in deltas.values()),
            'equal': sum(value == 0 for value in deltas.values()), 'higher': sum(value > 0 for value in deltas.values())}
    common.save(output / cohort / 'summary.json', result)
    digest = {'arms': {arm: {key: value for key, value in data.items() if key != 'tasks'}
                      for arm, data in result['arms'].items()},
              'paired_pinball_changes': result['paired_pinball_changes']}
    common.save(output / cohort / 'final_digest.json', digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, output, cohort):
    learning_jobs = [('learn', arm, seed) for i, seed in enumerate(protocol['cohorts'][cohort])
            for arm in (('learned', 'policy_disabled', 'baseline_v095') if cohort == 'development'
                        else ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)])]
    evaluation_jobs = [('evaluate', arm, seed) for seed in protocol['cohorts'][cohort] for arm in ARMS]
    execution = {'status': 'running', 'completed': [], 'active': [], 'failed': [],
                 'maximum_concurrent_processes': protocol['maximum_concurrent_research_processes']}
    lock = threading.Lock()
    def run(job):
        mode, arm, seed = job
        log = output / cohort / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            execution['active'].append(list(job))
            common.save(output / cohort / 'execution.json', execution)
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(output), '--cohort', cohort, '--arm', arm, '--seed', str(seed)],
                stdout=stream, stderr=subprocess.STDOUT)
        with lock:
            execution['active'].remove(list(job))
            execution['failed' if code else 'completed'].append(
                {'job': list(job), 'exit_code': code} if code else list(job))
            common.save(output / cohort / 'execution.json', execution)
        print(json.dumps({'finished': list(job), 'exit_code': code}), flush=True)
        return code
    for phase, jobs in (('learning', learning_jobs), ('terminal', evaluation_jobs)):
        execution['phase'] = phase
        with ThreadPoolExecutor(max_workers=protocol['maximum_concurrent_research_processes']) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            for future in as_completed(futures):
                if not future.cancelled() and future.result():
                    for pending in futures:
                        pending.cancel()
        if execution['failed']:
            common.save(output / cohort / 'execution.json', {**execution, 'status': 'failed', 'exit_code': 1})
            raise SystemExit(1)
    summarize(protocol, output, cohort)
    common.save(output / cohort / 'execution.json', {**execution, 'status': 'completed', 'exit_code': 0})


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
        common.freeze(args.output)
    elif args.mode == 'queue':
        queue(protocol, args.output, args.cohort)
    elif args.mode == 'summarize':
        summarize(protocol, args.output, args.cohort)
    elif args.mode == 'learn':
        learn(protocol, args.output, args.cohort, args.arm, args.seed)
    else:
        evaluate(protocol, args.output, args.cohort, args.arm, args.seed)


if __name__ == '__main__':
    main()

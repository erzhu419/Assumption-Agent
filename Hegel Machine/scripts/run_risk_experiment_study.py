#!/usr/bin/env python3
"""Fixed-risk paired streams: frozen 0.95, frozen 0.96.1, and new 0.97."""
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

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/risk_experiment_study_20260920'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_risk_action_study as risk

ARMS = ('baseline_v095', 'baseline_v0961', 'learned')
CONTROLS = ARMS[:2]
risk.ARMS = risk.common.ARMS = ARMS
original_run_path = risk.common.run_path


def prepare():
    path = OUTPUT / 'protocol.json'
    if path.exists():
        return
    previous = json.loads((ROOT / 'artifacts/risk_action_study_20260919/protocol.json').read_text())
    development, main = [95401901, 95401903], list(range(95502001, 95502009))
    protocol = {**previous,
        'experiment': 'risk_counterexplanation_measurement_and_continuation',
        'fixed_at': datetime.now(timezone.utc).isoformat(),
        'cohorts': {'development': development, 'main': main},
        'tasks_by_seed': {str(seed): [{'environment': 'scalar', 'task_index': task,
            'profile': profile, 'seed': seed + 10007 * position}
            for position, (task, profile) in enumerate(((12, 'development'), (13, 'development'),
                (13, 'confirmation'), (12, 'development')))] for seed in [*development, *main]},
        'development_runtime': 'src',
        'arms': {
            'baseline_v095': {'runtime': 'artifacts/object_action_study_20260919/runtime/src',
                             'version': '0.95.0'},
            'baseline_v0961': {'runtime': 'artifacts/risk_action_repair_20260919/runtime/src',
                              'version': '0.96.1'},
            'learned': {'runtime': str((OUTPUT / 'runtime/src').relative_to(ROOT)),
                        'version': '0.97.0'}},
        'development_references': {
            'baseline_v095': 'artifacts/risk_action_study_20260919/main/runs/baseline_v095/{seed}/checkpoint.json',
            'baseline_v0961': 'artifacts/risk_action_repair_20260919/development/runs/learned/{seed}/checkpoint.json'},
        'maximum_concurrent_research_processes': 4,
        'scope': 'Eight prospectively fixed new stochastic trajectory streams of the same two noisy '
            'mechanisms and parameter profiles, not eight independent physical apparatus. Each '
            'four-task stream starts with empty knowledge and restores the full JSON between tasks. '
            'Every arm is rerun on every main seed. Two known, previously exposed development roots '
            'are reported separately, using the completed frozen baseline records as read-only references.',
        'capability_under_test': 'A positive competing residual explanation remains available when '
            'finite evidence cannot distinguish it from the incumbent. The new agent can pay for '
            'another independent extended full-coordinate trajectory using the observed gain and '
            'variance, compare frozen risk predictions, then separately confirm a resulting revision. '
            'No private mechanism labels or oracle values enter proposal, selection, stopping, or memory. '
            'The common active_probes setting remains false; risk-specific measurement is itself a '
            'paid research action, not a different baseline observation policy.',
        'known_exposure': 'Development roots 95401901 and 95401903, their terminal losses and the '
            '0.96.1 repair outcomes were already inspected. They cannot establish prospective performance. '
            'Main roots 95502001 through 95502008 were fixed before any execution of this protocol.',
        'execution_contract': 'Known development uses released current 0.97 source; freeze the final '
            'source after development and required checks, before main learning. Run at most four independent stream processes concurrently. Finish all declared cohort learning in every arm '
            'before terminal evaluation. Preserve every returned model, including unresolved tasks. '
            'The main cohort is run once after the known-development check; no tuning from its terminal results.',
        'comparisons': 'Paired new-minus-0.96.1 and new-minus-0.95 pinball loss per complete stream '
            'and stage. Include source learning, failed research, all three paid resources, query count, '
            'process CPU, and wall time. Action loss remains ten times pinball, not an independent endpoint.',
        'formal_evidence': False}
    risk.common.save(path, protocol)


def run_path(output, cohort, arm, seed):
    if cohort == 'development' and arm in CONTROLS:
        protocol = json.loads((OUTPUT / 'protocol.json').read_text())
        return ROOT / protocol['development_references'][arm].format(seed=seed)
    return original_run_path(output, cohort, arm, seed)


def load(protocol, arm, cohort):
    source = ROOT / (protocol['development_runtime'] if cohort == 'development' and arm == 'learned'
                     else protocol['arms'][arm]['runtime'])
    sys.path.insert(0, str(source))
    canonical = 'hegel_machine.cognition.unknown_task_lab'
    spec = importlib.util.spec_from_file_location(canonical, ROOT / protocol['shared_laboratory_source'])
    laboratory = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = laboratory
    spec.loader.exec_module(laboratory)
    module = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('each frozen runtime must execute in its own process')
    def factory(item):
        arguments = dict(item)
        arguments.pop('environment')
        return laboratory.UnknownTaskLab(**arguments)
    return module, factory


risk.load = risk.common.load = load
risk.common.run_path = run_path


def summarize(protocol, cohort):
    seeds, result = protocol['cohorts'][cohort], {'cohort': cohort, 'scope': protocol['scope'], 'arms': {}}
    for arm in ARMS:
        tasks, learning, terminal, cpu = [], Counter(), Counter(), []
        for seed in seeds:
            state = json.loads(run_path(OUTPUT, cohort, arm, seed).read_text())
            evaluations = {row['task_position']: row for row in state['evaluation']}
            timing = state.get('learning_process_cpu', [])
            if {p for row in timing for p in row['task_positions']} == set(range(len(risk.STAGES))):
                cpu.append(sum(row['seconds'] for row in timing))
            for record in state['results']:
                position, report = record['task_position'], record['research']
                events = report['actions']
                experiments = [event for event in events if event['action'] == 'acquire'
                    and event.get('purpose') == 'distinguish_risk_explanations']
                task = {'replicate_seed': seed, 'position': position, 'stage': risk.STAGES[position],
                    'status': report['status'], 'kind': report['selected_kind'],
                    'channels': report['selected_channels'], 'evaluation': evaluations[position],
                    'costs': report['costs'] | {'queries': record['actual_costs']['queries']},
                    'learning_wall_seconds': record['learning_wall_seconds'],
                    'actions': dict(Counter(event['action'] for event in events)),
                    'risk_events': [{key: value for key, value in event.items() if key not in
                        ('program', 'baseline_program', 'previous_program', 'correction', 'selected_program')}
                        for event in events if any(token in event['action'] for token in ('risk', 'tail', 'noise'))],
                    'action_experience_before': record['action_experience_before'],
                    'action_experience_after': record['action_experience_after'],
                    'acquisitions': record['acquisitions'], 'used_sources': report['used_sources'],
                    'noise_assessment': report.get('noise_assessment'),
                    'risk_experiments': {'paid_experiments': len(experiments),
                        'observation_rows': sum(event['observation_cost'] for event in experiments),
                        'measurement_cost': sum(event['measurement_cost'] for event in experiments),
                        'finish_status': dict(Counter(event['status'] for event in events
                            if event['action'] == 'finish_risk_experiment'))}}
                tasks.append(task)
                learning.update(task['costs'])
                terminal.update(evaluations[position]['costs'])
        def metrics(rows):
            return {key: mean(row['evaluation'][key] for row in rows) for key in risk.METRICS}
        result['arms'][arm] = {'tasks': tasks, 'tasks_count': len(tasks),
            'learning_status': dict(Counter(row['status'] for row in tasks)),
            'evaluated_sequences': len(seeds), 'metrics': metrics(tasks),
            'by_stage': {stage: metrics([row for row in tasks if row['stage'] == stage]) for stage in risk.STAGES},
            'by_seed': {str(seed): metrics([row for row in tasks if row['replicate_seed'] == seed]) for seed in seeds},
            'learning_costs': dict(learning), 'terminal_costs': dict(terminal),
            'costs_by_stage': {stage: {
                'learning': {key: sum(row['costs'][key] for row in tasks if row['stage'] == stage)
                             for key in risk.common.COSTS},
                'terminal': {key: sum(row['evaluation']['costs'][key] for row in tasks if row['stage'] == stage)
                             for key in risk.common.COSTS}} for stage in risk.STAGES},
            'risk_experiments': {
                **{key: sum(row['risk_experiments'][key] for row in tasks)
                   for key in ('paid_experiments', 'observation_rows', 'measurement_cost')},
                'finish_status': dict(sum((Counter(row['risk_experiments']['finish_status'])
                                          for row in tasks), Counter()))},
            'learning_process_cpu_seconds': sum(cpu) if len(cpu) == len(seeds) else None,
            'learning_wall_seconds': sum(row['learning_wall_seconds'] for row in tasks),
            'events': dict(sum((Counter(row['actions']) for row in tasks), Counter()))}
    result['paired_pinball_changes'] = {}
    learned = result['arms']['learned']
    for arm in CONTROLS:
        control = result['arms'][arm]
        deltas = {str(seed): learned['by_seed'][str(seed)]['pinball_loss']
            - control['by_seed'][str(seed)]['pinball_loss'] for seed in seeds}
        stage_deltas = {stage: {str(seed): next(row['evaluation']['pinball_loss']
            for row in learned['tasks'] if row['replicate_seed'] == seed and row['stage'] == stage)
            - next(row['evaluation']['pinball_loss'] for row in control['tasks']
                if row['replicate_seed'] == seed and row['stage'] == stage) for seed in seeds}
            for stage in risk.STAGES}
        result['paired_pinball_changes'][arm] = {'new_minus_control_by_seed': deltas,
            'by_stage_and_seed': stage_deltas, 'mean_change': mean(deltas.values()),
            'lower': sum(value < 0 for value in deltas.values()),
            'equal': sum(value == 0 for value in deltas.values()),
            'higher': sum(value > 0 for value in deltas.values())}
    risk.common.save(OUTPUT / cohort / 'summary.json', result)
    digest = {**result, 'arms': {arm: {key: value for key, value in data.items() if key != 'tasks'}
                               for arm, data in result['arms'].items()}}
    risk.common.save(OUTPUT / cohort / 'final_digest.json', digest)
    interpretation = {'cohort': cohort, 'known_development': cohort == 'development',
        'arms': {arm: {'pinball_loss': data['metrics']['pinball_loss'],
            'learning_costs': data['learning_costs'],
            'terminal_measurement_cost': data['terminal_costs']['measurement_cost'],
            'learning_process_cpu_seconds': data['learning_process_cpu_seconds'],
            'learning_status': data['learning_status'], 'risk_experiments': data['risk_experiments'],
            'unresolved_by_stage': dict(Counter(row['stage'] for row in data['tasks']
                                               if row['status'] == 'unresolved'))}
                 for arm, data in result['arms'].items()},
        'paired': {arm: {'mean_change_percent': 100 * (
                learned['metrics']['pinball_loss'] / result['arms'][arm]['metrics']['pinball_loss'] - 1),
            **{key: value for key, value in result['paired_pinball_changes'][arm].items()
               if key != 'by_stage_and_seed'},
            'by_stage_mean_change': {stage: learned['by_stage'][stage]['pinball_loss']
                - result['arms'][arm]['by_stage'][stage]['pinball_loss'] for stage in risk.STAGES}}
                   for arm in CONTROLS}}
    risk.common.save(OUTPUT / cohort / 'interpretation_digest.json', interpretation)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, cohort):
    jobs = [(arm, seed) for i, seed in enumerate(protocol['cohorts'][cohort])
            for arm in (('learned',) if cohort == 'development'
                        else ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)])]
    execution = {'status': 'running', 'active': [], 'completed': [], 'failed': [],
                 'maximum_concurrent_processes': protocol['maximum_concurrent_research_processes']}
    lock = threading.Lock()
    def run(mode, arm, seed):
        job = [mode, arm, seed]
        log = OUTPUT / cohort / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            execution['active'].append(job)
            risk.common.save(OUTPUT / cohort / 'execution.json', execution)
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(OUTPUT), '--cohort', cohort, '--arm', arm, '--seed', str(seed)], stdout=stream, stderr=subprocess.STDOUT)
        with lock:
            execution['active'].remove(job)
            execution['failed' if code else 'completed'].append({'job': job, 'exit_code': code})
            risk.common.save(OUTPUT / cohort / 'execution.json', execution)
        print(json.dumps({'finished': job, 'exit_code': code}), flush=True)
        return code
    for phase in ('learn', 'evaluate'):
        execution['phase'] = phase
        with ThreadPoolExecutor(max_workers=protocol['maximum_concurrent_research_processes']) as pool:
            futures = [pool.submit(run, phase, arm, seed) for arm, seed in jobs]
            for future in as_completed(futures):
                if not future.cancelled() and future.result():
                    for pending in futures:
                        pending.cancel()
        if execution['failed']:
            risk.common.save(OUTPUT / cohort / 'execution.json', {**execution, 'status': 'failed', 'exit_code': 1})
            raise SystemExit(1)
    summarize(protocol, cohort)
    risk.common.save(OUTPUT / cohort / 'execution.json', {**execution, 'status': 'completed', 'exit_code': 0})


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'learn', 'evaluate', 'queue', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cohort', choices=('development', 'main'), default='development')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    if args.mode == 'prepare':
        prepare()
        return
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    if args.mode == 'freeze':
        risk.common.freeze(OUTPUT)
    elif args.mode == 'queue':
        queue(protocol, args.cohort)
    elif args.mode == 'summarize':
        summarize(protocol, args.cohort)
    else:
        if args.cohort == 'development' and args.arm in CONTROLS:
            raise ValueError('known development baseline checkpoints are read-only references')
        getattr(risk, args.mode)(protocol, OUTPUT, args.cohort, args.arm, args.seed)


if __name__ == '__main__':
    main()

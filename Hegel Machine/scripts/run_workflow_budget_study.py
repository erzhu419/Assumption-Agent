#!/usr/bin/env python3
"""Compare frozen and repaired research-package allocation from one paid state."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import importlib
import json
from pathlib import Path
from random import Random
from statistics import mean
import subprocess
import sys
from time import process_time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/workflow_demand_20261006_main_12/frozen_source'
COST_KEYS = ('observation_cost', 'measurement_cost', 'compute_cost')
ARMS = ('baseline', 'repair')


def protocol(seed=65007001):
    original = json.loads((BASELINE.parent / 'protocol.json').read_text())['cases'][0]
    plan = deepcopy(original)
    previous = plan['seed']
    for group in ('return_tasks', 'competition_tasks', 'tasks'):
        for task in plan[group]:
            task['seed'] += seed - previous
    plan.update(experiment='workflow_research_package_budget_repair', seed=seed,
        partial_probe_seed=seed, arms=list(ARMS), arm_order='baseline_then_repair',
        intervention='Both arms retain all measured workflow returns and start from the exact same '
            'old-implementation source-return, competing-method and paid incoming-consumer checkpoint. '
            'Only the implementation performing subsequent research differs.',
        information_contract='Generate the common paid prefix with the frozen implementation. '
            'Give both continuations identical reports, predictive knowledge, learned workflows '
            'and measured returns; neither arm deletes feedback or receives supplied models.',
        accounting='Initial task caps and two-task pooled ceilings are unchanged. Include common '
            'prefix and incoming-consumer work once in each full bill; charge all screening, failed '
            'research and confirmation. Score both arms only after both finish learning. Source '
            'prefix is learned and scored once. Report incoming demand and producer separately.',
        scope='A paired comparison of the research-package repair on a previously failing task '
            'family. Baseline runs first; CPU is descriptive. Retain failures and non-activation.')
    plan['research_config']['seed'] = seed
    plan.pop('arm_order_seed', None)
    return plan


def cohort_protocol():
    return {'experiment': 'workflow_budget_repair_12_new_seeds', 'bootstrap_samples': 10000,
        'bootstrap_seed': 65007019, 'cases': [protocol(65107001 + 100003*i) for i in range(12)],
        'case_policy': 'All registered seeds remain, including unresolved tasks and execution failures.'}


def _write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')


def _runtime_module(runtime):
    # The worker is a fresh interpreter, so baseline and repair cannot share imports.
    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime / 'src'))
    sys.path.insert(0, str(runtime / 'scripts'))
    return importlib.import_module('run_workflow_demand_study')


def worker(phase, runtime, input_path, output_path):
    study = _runtime_module(runtime)
    value = json.loads(Path(input_path).read_text())
    if phase == 'prepare':
        plan = value
        config = study.ResearchConfig(**plan['research_config'])
        start = process_time()
        source = study.UnifiedResearchSession(config, tasks=plan['return_tasks'],
            measurement_budget=plan['measurement_budget'], library=study.KnowledgeLibrary())
        initial = study._partial_visit(source, plan, study.laboratory_for)
        source.run(max_tasks=1)
        offers = deepcopy(source.to_dict()['pending_revisits'])
        source.run()
        competing = study.UnifiedResearchSession(config, tasks=plan['competition_tasks'],
            measurement_budget=plan['measurement_budget'],
            library=study.KnowledgeLibrary.from_dict(source.library.to_dict()))
        competing.run()
        prefix_cpu = process_time() - start
        start = process_time()
        demand = study.UnifiedResearchSession(config, tasks=plan['tasks'],
            measurement_budget=plan['measurement_budget'],
            library=study.KnowledgeLibrary.from_dict(competing.library.to_dict()))
        partial = study._partial_visit(demand, plan, study.laboratory_for, incoming=True)
        result = {'plan': plan, 'prefix_states': {'source_return': source.to_dict(),
            'competing_method': competing.to_dict()}, 'checkpoint': demand.to_dict(),
            'source_initial_status': initial['status'], 'source_queued_offers': offers,
            'partial_demand': partial, 'demands': demand._workflow_demands(),
            'prefix_training_cpu_seconds': prefix_cpu,
            'demand_training_cpu_seconds': process_time() - start}
    elif phase == 'train':
        from hegel_machine.cognition.workflow_execution import WorkflowEpisode
        original_replay, fits = WorkflowEpisode.replay, []

        def observe_replay(episode, **arguments):
            native_fit, remaining = arguments['fit'], arguments['remaining']

            def observe_fit(channels, training, validation):
                before = deepcopy(remaining())
                claims = native_fit(channels, training, validation)
                after = deepcopy(remaining())
                fits.append({'task_id': episode.spec.task_id, 'attempt': episode.attempt,
                    'channels': list(channels), 'before': before, 'after': after,
                    'compute_cost': before['compute_remaining'] - after['compute_remaining']})
                return claims

            arguments['fit'] = observe_fit
            return original_replay(episode, **arguments)

        WorkflowEpisode.replay = observe_replay
        session = study.UnifiedResearchSession(state=value['checkpoint'])
        initial = session.to_dict()
        start = process_time()
        session.run()
        result = {'state': session.to_dict(), 'training_cpu_seconds': process_time() - start,
            'initial_costs': initial['costs'],
            'initial_concept_ids': [item['concept_id'] for item in initial['knowledge']['concepts']],
            'initial_workflows': initial['knowledge']['research_workflows'],
            'initial_tasks': initial['tasks'], 'initial_config': initial['config'], 'workflow_fits': fits}
    else:
        common, trained = value['common'], value['trained']
        plan = common['plan']
        start = process_time()
        result = {'suffix': study._summary(study.UnifiedResearchSession(state=trained['state']),
            plan, plan['tasks']), 'suffix_evaluation_cpu_seconds': process_time() - start}
        if value['include_prefix']:
            start = process_time()
            result['prefix'] = {name: study._summary(study.UnifiedResearchSession(state=state),
                plan, plan['return_tasks'] if name == 'source_return' else plan['competition_tasks'])
                for name, state in common['prefix_states'].items()}
            result['prefix_evaluation_cpu_seconds'] = process_time() - start
    result['runtime_root'] = str(Path(runtime).resolve())
    result['runtime_module'] = str(Path(study.__file__).resolve())
    _write(output_path, result)


def _call(phase, runtime, input_path, output_path):
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', phase,
            '--runtime', str(runtime), '--input', str(input_path), '--output', str(output_path)],
            check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr or error.stdout) from error


def prepare_case(plan, output, *, baseline_root=BASELINE):
    """Buy the shared prefix and train only baseline; do not score either arm."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'protocol.json', plan)
    _call('prepare', baseline_root, output / 'protocol.json', output / 'common.json')
    _call('train', baseline_root, output / 'common.json', output / 'baseline_training.json')
    return output


def complete_case(output, *, baseline_root=BASELINE, repair_root=ROOT):
    """Resume a prepared case with repair, then score the two finished arms."""
    output = Path(output)
    _call('train', repair_root, output / 'common.json', output / 'repair_training.json')
    common = json.loads((output / 'common.json').read_text())
    trained = {arm: json.loads((output / f'{arm}_training.json').read_text()) for arm in ARMS}
    scored = {}
    for arm, runtime in (('baseline', baseline_root), ('repair', repair_root)):
        _write(output / f'{arm}_score_input.json', {'common': common, 'trained': trained[arm],
            'include_prefix': arm == 'baseline'})
        _call('score', runtime, output / f'{arm}_score_input.json', output / f'{arm}_scored.json')
        scored[arm] = json.loads((output / f'{arm}_scored.json').read_text())
        (output / f'{arm}_score_input.json').unlink()
    prefix = scored['baseline']['prefix']
    prefix_costs = {key: sum(phase['costs'][key] for phase in prefix.values()) for key in COST_KEYS}
    prefix_evaluation = {key: sum(phase['evaluation_costs'][key] for phase in prefix.values()) for key in COST_KEYS}
    results = {}
    for arm in ARMS:
        result = scored[arm]['suffix']
        for task, role in zip(result['tasks'], ('incoming_demand', 'producer')):
            task['role'] = role
        result.update(task_sequence_costs=deepcopy(result['costs']),
            task_sequence_resource_limits=deepcopy(result['resource_limits']),
            costs={key: result['costs'][key] + prefix_costs[key] for key in COST_KEYS},
            resource_limits={key: result['resource_limits'][key]
                + sum(phase['resource_limits'][key] for phase in prefix.values()) for key in COST_KEYS},
            continuation_costs={key: result['costs'][key] - common['partial_demand']['costs'][key] for key in COST_KEYS},
            full_evaluation_costs={key: result['evaluation_costs'][key] + prefix_evaluation[key] for key in COST_KEYS},
            training_cpu_seconds=trained[arm]['training_cpu_seconds'] + common['prefix_training_cpu_seconds']
                + common['demand_training_cpu_seconds'],
            continuation_cpu_seconds=trained[arm]['training_cpu_seconds'],
            evaluation_cpu_seconds=scored[arm]['suffix_evaluation_cpu_seconds'],
            initial_workflows=trained[arm]['initial_workflows'], initial_costs=trained[arm]['initial_costs'],
            initial_config=trained[arm]['initial_config'], runtime_root=trained[arm]['runtime_root'])
        actions = result['tasks'][1]['actions']
        fits = [fit for fit in trained[arm].get('workflow_fits', [])
                if fit['attempt'] == common['plan']['tasks'][1]['seed']]
        remeasure = next((action for action in actions
            if action['action'] == 'choose_observation_research'
            and action.get('selected_action') == 'remeasure_inputs'), None)
        joint = [action for action in actions if action['action'] == 'choose_condition_response_research']
        result['producer_diagnostics'] = {'workflow_fits': fits,
            'first_workflow_fit_compute': fits[0]['compute_cost'] if fits else None,
            'first_remeasurement': {key: deepcopy(remeasure.get(key)) for key in
                ('observation_remaining', 'measurement_remaining', 'compute_remaining',
                 'quoted_observation_cost', 'quoted_compute_cost', 'eligible_actions')} if remeasure else None,
            'joint_choices': [{key: deepcopy(action.get(key)) for key in
                ('selected_action', 'eligible_actions', 'observation_remaining', 'measurement_remaining',
                 'compute_remaining', 'quoted_observation_cost', 'quoted_compute_cost')} for action in joint],
            'joint_eligible': any('joint_observed_condition_response' in action.get('eligible_actions', [])
                                  for action in joint),
            'state_identification_choices': [{key: deepcopy(action.get(key)) for key in
                ('context', 'compute_remaining', 'quoted_compute_cost', 'observation_remaining')}
                for action in actions if action.get('selected_action') == 'state_identification'],
            'confirmed': result['tasks'][1]['fresh_confirmation_passed'],
            'confirmed_and_solved': result['tasks'][1]['confirmed_and_solved']}
        results[arm] = result
    baseline, repair = (results[arm] for arm in ARMS)
    record = {'protocol': common['plan'], 'case_path': str((output / 'case.json').resolve()),
        'prefix': prefix, 'common_prefix_costs': prefix_costs,
        'common_prefix_evaluation_costs': prefix_evaluation,
        'source_initial_status': common['source_initial_status'], 'source_queued_offers': common['source_queued_offers'],
        'partial_demand': {key: common['partial_demand'].get(key) for key in ('status', 'costs', 'actions')},
        'demands': common['demands'], 'results': results, 'paired_repair_minus_baseline': {
            'costs': {key: repair['costs'][key] - baseline['costs'][key] for key in COST_KEYS},
            'confirmed_and_solved': repair['confirmed_and_solved'] - baseline['confirmed_and_solved'],
            'mean_mse': repair['mean_mse'] - baseline['mean_mse']}, 'external_model_calls': 0}
    _write(output / 'case.json', record)
    return record


def run_study(plan, output, *, baseline_root=BASELINE, repair_root=ROOT):
    prepare_case(plan, output, baseline_root=baseline_root)
    return complete_case(output, baseline_root=baseline_root, repair_root=repair_root)


def summarize(cases, failures, plan):
    totals = {}
    for arm in ARMS:
        rows = [case['results'][arm] for case in cases]
        stages = {}
        for position, role in enumerate(('incoming_demand', 'producer')):
            tasks = [row['tasks'][position] for row in rows]
            stages[role] = {'tasks': len(tasks), 'confirmed_and_solved': sum(t['confirmed_and_solved'] for t in tasks),
                'mean_mse': mean(t['mse'] for t in tasks) if tasks else None,
                'costs': {key: sum(t['costs'][key] for t in tasks) for key in COST_KEYS}}
        actions = Counter(action['action'] for row in rows for task in row['tasks'] for action in task['actions'])
        totals[arm] = {'stages': stages, 'confirmed_and_solved': sum(row['confirmed_and_solved'] for row in rows),
            'costs': {key: sum(row['costs'][key] for row in rows) for key in COST_KEYS},
            'evaluation_costs': {key: sum(row['full_evaluation_costs'][key] for row in rows) for key in COST_KEYS},
            'training_cpu_seconds': sum(row['training_cpu_seconds'] for row in rows),
            'revisit_count': sum(row['revisit_count'] for row in rows), 'action_counts': dict(actions),
            'producer_joint_eligible_cases': sum(row['producer_diagnostics']['joint_eligible'] for row in rows),
            'producer_state_identification_cases': sum(bool(row['producer_diagnostics']['state_identification_choices'])
                for row in rows),
            'producer_diagnostics': [{'seed': case['protocol']['seed'], 'case_path': case['case_path'],
                **{key: case['results'][arm]['producer_diagnostics'][key] for key in
                    ('first_workflow_fit_compute', 'first_remeasurement', 'joint_eligible',
                     'state_identification_choices', 'confirmed')}}
                for case in cases],
            'unconfirmed_tasks': [{'seed': case['protocol']['seed'], 'role': task['role'], 'mse': task['mse'],
                'status': task['status'], 'confirmation_loss': task['confirmation_loss']}
                for case in cases for task in case['results'][arm]['tasks'] if not task['confirmed_and_solved']]}
    random, intervals = Random(plan['bootstrap_seed']), {}
    for metric in (*COST_KEYS, 'confirmed_and_solved', 'mean_mse'):
        differences = [(case['paired_repair_minus_baseline']['costs'][metric] if metric in COST_KEYS
            else case['paired_repair_minus_baseline'][metric]) for case in cases]
        draws = sorted(mean(random.choice(differences) for _ in differences)
            for _ in range(plan['bootstrap_samples'])) if cases else []
        intervals[metric] = {'mean_difference': mean(differences) if cases else None,
            'interval_95': [draws[int(.025*(len(draws)-1))], draws[int(.975*(len(draws)-1))]] if draws else None}
    return {'protocol': plan, 'completed_cases': len(cases), 'execution_failures': failures, 'totals': totals,
        'prefix_tasks': sum(len(phase['tasks']) for case in cases for phase in case['prefix'].values()),
        'prefix_confirmed_and_solved': sum(phase['confirmed_and_solved'] for case in cases for phase in case['prefix'].values()),
        'paired_repair_minus_baseline': intervals, 'external_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=65007001)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, default=BASELINE)
    parser.add_argument('--repair-root', type=Path, default=ROOT)
    parser.add_argument('--phase', choices=('prepare', 'complete', 'case', 'cohort'), default='case')
    parser.add_argument('--worker', choices=('prepare', 'train', 'score'))
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.runtime, args.input, args.output)
        return
    if args.phase == 'prepare':
        prepare_case(protocol(args.seed), args.output, baseline_root=args.baseline_root)
        print(json.dumps({'status': 'baseline_trained_no_evaluation', 'output': str(args.output)}), flush=True)
    elif args.phase == 'complete':
        result = complete_case(args.output, baseline_root=args.baseline_root, repair_root=args.repair_root)
        print(json.dumps(result['paired_repair_minus_baseline']), flush=True)
    elif args.phase == 'case':
        result = run_study(protocol(args.seed), args.output, baseline_root=args.baseline_root, repair_root=args.repair_root)
        print(json.dumps(result['paired_repair_minus_baseline']), flush=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        registration = args.output / 'protocol.json'
        plan = json.loads(registration.read_text()) if registration.exists() else cohort_protocol()
        if not registration.exists():
            _write(registration, plan)
        if (args.output / 'summary.json').exists():
            raise FileExistsError('preserve the completed cohort and use a new output directory')
        cases, failures = [], []
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(run_study, case, args.output / f"case_{case['seed']}",
                baseline_root=args.baseline_root, repair_root=args.repair_root): case for case in plan['cases']}
            for job in as_completed(jobs):
                case = jobs[job]
                try:
                    result = job.result()
                    cases.append(result)
                    print(json.dumps({'seed': case['seed'], 'status': 'completed',
                        'joint': {arm: result['results'][arm]['confirmed_and_solved'] for arm in ARMS}}), flush=True)
                except Exception:
                    error = {'seed': case['seed'], 'traceback': traceback.format_exc()}
                    failures.append(error)
                    _write(args.output / f"failure_{case['seed']}.json", error)
                    print(json.dumps({'seed': case['seed'], 'status': 'execution_failed'}), flush=True)
        cases.sort(key=lambda case: case['protocol']['seed'])
        _write(args.output / 'summary.json', summarize(cases, failures, plan))
        print(json.dumps({'completed': len(cases), 'failed': len(failures)}), flush=True)


if __name__ == '__main__':
    main()

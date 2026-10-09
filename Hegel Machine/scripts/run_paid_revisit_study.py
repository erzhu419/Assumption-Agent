#!/usr/bin/env python3
"""Compare native paid-demand revisits after identical completed production."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
from random import Random
from statistics import mean
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNTIME = ROOT / 'artifacts/workflow_demand_20261006_main_12/frozen_source'
BASELINE = ROOT / 'artifacts/workflow_budget_20261006_main_12/frozen_source'
COST_KEYS = ('observation_cost', 'measurement_cost', 'compute_cost')
ARMS = ('baseline', 'repair')


def protocol(seed=67007001):
    original = json.loads((BASELINE.parent / 'protocol.json').read_text())['cases'][0]
    plan, previous = deepcopy(original), original['seed']
    for group in ('return_tasks', 'competition_tasks', 'tasks'):
        for task in plan[group]:
            task['seed'] += seed - previous
    plan.update(experiment='native_paid_demand_revisit', seed=seed, partial_probe_seed=seed,
        intervention='Generate the source prefix with the original demand-study implementation, '
            'then finish production with the frozen budget repair. Clone that completed-production '
            'checkpoint. Both implementations retain every source and return and invoke only native '
            'session.run(); no candidate or source is selected by this runner.',
        information_contract='Both arms inherit identical paid evidence, response programs, workflows, '
            'source returns, finished producer and remaining resource pool. Reconsideration and '
            'confirmation are native researcher decisions. Gains cannot be attributed to the newly '
            'produced source merely because it is present: older sources are also available.',
        accounting='Each counterfactual full bill includes shared source-prefix, incoming partial visit '
            'and completed production once. Add all native revisit screening, acquisition, search and '
            'confirmation. The original observation, measurement and compute ceilings are unchanged. '
            'Both continuations finish learning before terminal evaluation; score the source prefix once.',
        scope='A new-seed comparison on the same task family with an already purchased incoming '
            'consumer. Record actual remaining resources and inactive or unsuccessful revisits. '
            'Baseline runs first; CPU is descriptive.')
    plan['research_config']['seed'] = seed
    return plan


def cohort_protocol():
    return {'experiment': 'paid_revisit_12_new_seeds', 'bootstrap_samples': 10000,
        'bootstrap_seed': 67007019, 'cases': [protocol(67107001 + 100003*i) for i in range(12)],
        'case_policy': 'Keep all registered cases, including inactive demand, failed confirmation and exceptions.'}


def _write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')


def _call(phase, runtime, input_path, output_path, *, baseline_root=BASELINE):
    try:
        subprocess.run([sys.executable, str(Path(baseline_root) / 'scripts/run_workflow_budget_study.py'), '--worker', phase,
            '--runtime', str(runtime), '--input', str(input_path), '--output', str(output_path)],
            check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr or error.stdout) from error


def _called_sources(program):
    sources, definitions = set(), program.get('definitions', {})

    def visit(node):
        if node.get('op') == 'call':
            definition = definitions[node['name']]
            if definition.get('source_concept_id'):
                sources.add(definition['source_concept_id'])
            visit(definition['body'])
        for child in node.get('children', []):
            visit(child)

    visit(program)
    return sorted(sources)


def prepare_case(plan, output, *, source_runtime=SOURCE_RUNTIME, baseline_root=BASELINE):
    """Buy shared work, complete production, and run only the old scheduler."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'protocol.json', plan)
    _call('prepare', source_runtime, output / 'protocol.json', output / 'common.json', baseline_root=baseline_root)
    _call('train', baseline_root, output / 'common.json', output / 'producer_training.json', baseline_root=baseline_root)
    producer = json.loads((output / 'producer_training.json').read_text())
    _write(output / 'fork.json', {'checkpoint': producer['state']})
    _call('train', baseline_root, output / 'fork.json', output / 'baseline_training.json', baseline_root=baseline_root)
    return output


def complete_case(output, *, baseline_root=BASELINE, repair_root=ROOT):
    output = Path(output)
    _call('train', repair_root, output / 'fork.json', output / 'repair_training.json', baseline_root=baseline_root)
    common = json.loads((output / 'common.json').read_text())
    producer = json.loads((output / 'producer_training.json').read_text())
    checkpoint, plan = producer['state'], common['plan']
    trained = {arm: json.loads((output / f'{arm}_training.json').read_text()) for arm in ARMS}
    scored = {}
    for arm, runtime in (('baseline', baseline_root), ('repair', repair_root)):
        _write(output / f'{arm}_score_input.json', {'common': common, 'trained': trained[arm],
            'include_prefix': arm == 'baseline'})
        _call('score', runtime, output / f'{arm}_score_input.json', output / f'{arm}_scored.json', baseline_root=baseline_root)
        scored[arm] = json.loads((output / f'{arm}_scored.json').read_text())
        (output / f'{arm}_score_input.json').unlink()
    prefix = scored['baseline']['prefix']
    prefix_costs = {key: sum(phase['costs'][key] for phase in prefix.values()) for key in COST_KEYS}
    prefix_evaluation = {key: sum(phase['evaluation_costs'][key] for phase in prefix.values()) for key in COST_KEYS}
    results = {}
    for arm in ARMS:
        result = scored[arm]['suffix']
        incremental_actions = []
        for task, role, initial in zip(result['tasks'], ('incoming_demand', 'producer'), checkpoint['results']):
            task['role'] = role
            task['called_source_ids'] = _called_sources(task['program'])
            task['post_fork_actions'] = deepcopy(task['actions'][len(initial['research']['actions']):])
            incremental_actions.extend(task['post_fork_actions'])
        visits = [{key: deepcopy(visit.get(key)) for key in
            ('task_position', 'source_task_position', 'candidate', 'visit_costs', 'resource_limits',
             'available_concepts', 'status', 'basis')} for visit in result['revisits']]
        new_visits = visits[len(checkpoint['revisits']):]
        result.update(task_sequence_costs=deepcopy(result['costs']),
            task_sequence_resource_limits=deepcopy(result['resource_limits']),
            costs={key: result['costs'][key] + prefix_costs[key] for key in COST_KEYS},
            resource_limits={key: result['resource_limits'][key]
                + sum(phase['resource_limits'][key] for phase in prefix.values()) for key in COST_KEYS},
            incremental_costs={key: result['costs'][key] - checkpoint['costs'][key] for key in COST_KEYS},
            incremental_screening_compute=result['screening_compute_cost'] - checkpoint['screening_compute_cost'],
            full_evaluation_costs={key: result['evaluation_costs'][key] + prefix_evaluation[key] for key in COST_KEYS},
            revisits=visits, new_revisits=new_visits,
            selected_offer_source_ids=sorted({source for visit in new_visits
                for source in visit['candidate'].get('source_concept_ids', [])}),
            paid_selection_receipts=[{key: deepcopy(visit['candidate'].get('paid_selection', {}).get(key))
                for key in ('evidence_rows', 'evidence_batches', 'channels', 'baseline_loss', 'candidate_loss')}
                for visit in new_visits],
            fresh_confirmation_receipts=[{key: deepcopy(action.get(key)) for key in
                ('observation_rows', 'accepted', 'frozen_paid_selection', 'loss', 'probe_id')}
                for action in incremental_actions if action['action'] == 'validate_continued_program'],
            new_scheduler_events=deepcopy(result['scheduler_events'][len(checkpoint['scheduler_events']):]),
            incremental_action_counts=dict(Counter(action['action'] for action in incremental_actions)),
            training_cpu_seconds=common['prefix_training_cpu_seconds'] + common['demand_training_cpu_seconds']
                + producer['training_cpu_seconds'] + trained[arm]['training_cpu_seconds'],
            revisit_cpu_seconds=trained[arm]['training_cpu_seconds'],
            evaluation_cpu_seconds=scored[arm]['suffix_evaluation_cpu_seconds'],
            initial_workflows=trained[arm]['initial_workflows'], initial_costs=trained[arm]['initial_costs'],
            initial_config=trained[arm]['initial_config'], initial_concept_ids=trained[arm]['initial_concept_ids'],
            runtime_root=trained[arm]['runtime_root'])
        results[arm] = result
    base, repair = (results[arm] for arm in ARMS)
    limits = base['task_sequence_resource_limits']
    record = {'protocol': plan, 'case_path': str((output / 'case.json').resolve()), 'prefix': prefix,
        'common_prefix_costs': prefix_costs, 'common_prefix_evaluation_costs': prefix_evaluation,
        'common_production_costs': {key: checkpoint['costs'][key] - common['partial_demand']['costs'][key]
                                    for key in COST_KEYS},
        'fork_costs': checkpoint['costs'],
        'fork_remaining': {key: limits[key] - checkpoint['costs'][key] for key in COST_KEYS},
        'fork_statuses': [row['research']['status'] for row in checkpoint['results']],
        'fork_concept_ids': [concept['concept_id'] for concept in checkpoint['knowledge']['concepts']],
        'producer_added_concepts': checkpoint['results'][1]['research'].get('concepts_added', []),
        'partial_demand': {key: common['partial_demand'].get(key) for key in ('status', 'costs')},
        'results': results, 'paired_repair_minus_baseline': {
            'costs': {key: repair['costs'][key] - base['costs'][key] for key in COST_KEYS},
            'confirmed_and_solved': repair['confirmed_and_solved'] - base['confirmed_and_solved'],
            'incoming_mse': repair['tasks'][0]['mse'] - base['tasks'][0]['mse'],
            'mean_mse': repair['mean_mse'] - base['mean_mse']}, 'external_model_calls': 0}
    _write(output / 'case.json', record)
    return record


def run_study(plan, output, *, source_runtime=SOURCE_RUNTIME, baseline_root=BASELINE, repair_root=ROOT):
    prepare_case(plan, output, source_runtime=source_runtime, baseline_root=baseline_root)
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
        totals[arm] = {'stages': stages,
            **{field: {key: sum(row[field][key] for row in rows) for key in COST_KEYS}
               for field in ('costs', 'incremental_costs', 'full_evaluation_costs')},
            'incremental_screening_compute': sum(row['incremental_screening_compute'] for row in rows),
            'native_revisit_cases': sum(bool(row['new_revisits']) for row in rows),
            'training_cpu_seconds': sum(row['training_cpu_seconds'] for row in rows),
            'incoming_called_sources': dict(Counter(source for row in rows for source in row['tasks'][0]['called_source_ids'])),
            'selected_offer_sources': dict(Counter(source for row in rows for source in row['selected_offer_source_ids'])),
            'fresh_confirmation_rows': [receipt['observation_rows'] for row in rows
                for receipt in row['fresh_confirmation_receipts']],
            'cases': [{'seed': case['protocol']['seed'], 'path': case['case_path'],
                'incoming_joint': case['results'][arm]['tasks'][0]['confirmed_and_solved'],
                'producer_joint': case['results'][arm]['tasks'][1]['confirmed_and_solved'],
                'revisits': len(case['results'][arm]['new_revisits']),
                'screening_compute': case['results'][arm]['incremental_screening_compute'],
                'selected_sources': case['results'][arm]['selected_offer_source_ids'],
                'paid_selection': case['results'][arm]['paid_selection_receipts'],
                'fresh_confirmation': case['results'][arm]['fresh_confirmation_receipts'],
                'incremental_costs': case['results'][arm]['incremental_costs']}
                for case in cases]}
    random, intervals = Random(plan['bootstrap_seed']), {}
    for metric in (*COST_KEYS, 'confirmed_and_solved', 'incoming_mse', 'mean_mse'):
        differences = [(case['paired_repair_minus_baseline']['costs'][metric] if metric in COST_KEYS
            else case['paired_repair_minus_baseline'][metric]) for case in cases]
        draws = sorted(mean(random.choice(differences) for _ in differences)
            for _ in range(plan['bootstrap_samples'])) if cases else []
        intervals[metric] = {'mean_difference': mean(differences) if cases else None,
            'interval_95': [draws[int(.025*(len(draws)-1))], draws[int(.975*(len(draws)-1))]] if draws else None}
    return {'protocol': plan, 'completed_cases': len(cases), 'execution_failures': failures,
        'prefix_confirmed_and_solved': sum(phase['confirmed_and_solved'] for case in cases for phase in case['prefix'].values()),
        'totals': totals, 'paired_repair_minus_baseline': intervals,
        'fork_observation_remaining': [case['fork_remaining']['observation_cost'] for case in cases],
        'external_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=67007001)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-runtime', type=Path, default=SOURCE_RUNTIME)
    parser.add_argument('--baseline-root', type=Path, default=BASELINE)
    parser.add_argument('--repair-root', type=Path, default=ROOT)
    parser.add_argument('--phase', choices=('prepare', 'complete', 'case', 'cohort'), default='case')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare_case(protocol(args.seed), args.output, source_runtime=args.source_runtime, baseline_root=args.baseline_root)
        print(json.dumps({'status': 'native_baseline_trained_no_evaluation', 'output': str(args.output)}), flush=True)
    elif args.phase == 'complete':
        result = complete_case(args.output, baseline_root=args.baseline_root, repair_root=args.repair_root)
        print(json.dumps(result['paired_repair_minus_baseline']), flush=True)
    elif args.phase == 'case':
        result = run_study(protocol(args.seed), args.output, source_runtime=args.source_runtime,
            baseline_root=args.baseline_root, repair_root=args.repair_root)
        print(json.dumps(result['paired_repair_minus_baseline']), flush=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        registration = args.output / 'protocol.json'
        plan = json.loads(registration.read_text()) if registration.exists() else cohort_protocol()
        if not registration.exists():
            _write(registration, plan)
        if (args.output / 'summary.json').exists():
            raise FileExistsError('preserve completed results and use a new output directory')
        cases, failures = [], []
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(run_study, case, args.output / f"case_{case['seed']}",
                source_runtime=args.source_runtime, baseline_root=args.baseline_root, repair_root=args.repair_root): case
                for case in plan['cases']}
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

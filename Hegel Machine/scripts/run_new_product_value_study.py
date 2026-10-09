#!/usr/bin/env python3
"""Post-hoc marginal value of products from already paid, completed production."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from random import Random
from statistics import mean
import subprocess
import sys
from time import process_time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/paid_revisit_20261006_main_12'
ARMS = ('retain_product', 'remove_product')
COST_KEYS = ('observation_cost', 'measurement_cost', 'compute_cost')


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort_protocol(source=SOURCE):
    source = Path(source).resolve()
    old = _read(source / 'protocol.json')
    return {'experiment': 'new_product_posthoc_attribution', 'source_cohort': str(source),
        'runtime': str(source / 'frozen_source'), 'arms': list(ARMS),
        'bootstrap_samples': 10000, 'bootstrap_seed': 67107119,
        'cases': [{'seed': case['seed'], 'source_case': str(source / f"case_{case['seed']}")}
                  for case in old['cases']],
        'intervention': 'After completed production, retain its newly learned concepts or apply '
            'KnowledgeLibrary.without to exactly the producer concepts_added. Preserve all old '
            'knowledge, workflows, paid data, producer reports, feedback and costs. Invoke native run().',
        'accounting': 'Each arm includes the same original source prefix, incoming partial research '
            'and completed production once; add actual screening and continuation costs. Both arms '
            'finish learning before terminal scoring; source prefix is scored once. Original caps remain.',
        'scope': 'Explicit post-hoc attribution on all 12 previously observed checkpoints, not an '
            'independent seed cohort or a comparison against skipping production. Keep zero/negative '
            'value and all failures. Retain runs first; CPU is descriptive.'}


def intervene(checkpoint, arm):
    """Remove callable products, never their already paid production history."""
    from hegel_machine.cognition.memory import KnowledgeLibrary
    state = deepcopy(checkpoint)
    products = list(state['results'][1]['research'].get('concepts_added', []))
    if arm == 'remove_product':
        state['knowledge'] = KnowledgeLibrary.from_dict(state['knowledge']).without(set(products)).to_dict()
    elif arm != 'retain_product':
        raise ValueError(arm)
    return state, products


def worker(arm, runtime, source_case, output):
    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime / 'src'))
    from hegel_machine.cognition.unified_session import UnifiedResearchSession
    original = _read(Path(source_case) / 'producer_training.json')['state']
    state, products = intervene(original, arm)
    session = UnifiedResearchSession(state=state)
    start = process_time()
    session.run()
    _write(output, {'state': session.to_dict(), 'training_cpu_seconds': process_time() - start,
        'initial_costs': state['costs'],
        'initial_concept_ids': [item['concept_id'] for item in state['knowledge']['concepts']],
        'product_ids': products,
        'changed_knowledge_fields': [key for key in original['knowledge']
            if original['knowledge'][key] != state['knowledge'].get(key)],
        'runtime_root': str(runtime)})


def _execute(command):
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr or error.stdout) from error


def run_study(plan, output, *, runtime):
    output, source = Path(output), Path(plan['source_case'])
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'protocol.json', plan)
    for arm in ARMS:
        _execute([sys.executable, str(Path(__file__).resolve()), '--worker', arm,
            '--runtime', str(runtime), '--source-case', str(source),
            '--output', str(output / f'{arm}_training.json')])
    # No evaluation occurs before both native continuations have completed.
    common, producer = _read(source / 'common.json'), _read(source / 'producer_training.json')
    checkpoint = producer['state']
    helper = _load(Path(runtime) / 'scripts/run_paid_revisit_study.py', 'frozen_paid_helper')
    scored, trained = {}, {}
    for arm in ARMS:
        trained[arm] = _read(output / f'{arm}_training.json')
        _write(output / 'score_input.json', {'common': common, 'trained': trained[arm],
            'include_prefix': arm == ARMS[0]})
        helper._call('score', runtime, output / 'score_input.json', output / f'{arm}_scored.json',
            baseline_root=runtime)
        scored[arm] = _read(output / f'{arm}_scored.json')
        (output / 'score_input.json').unlink()
    prefix = scored[ARMS[0]]['prefix']
    prefix_costs = {key: sum(item['costs'][key] for item in prefix.values()) for key in COST_KEYS}
    prefix_eval = {key: sum(item['evaluation_costs'][key] for item in prefix.values()) for key in COST_KEYS}
    results = {}
    for arm in ARMS:
        report = scored[arm]['suffix']
        tasks, actions = [], []
        for row, initial, role in zip(report['tasks'], checkpoint['results'], ('incoming_demand', 'producer')):
            post = row['actions'][len(initial['research']['actions']):]
            actions.extend(post)
            tasks.append({**{key: deepcopy(row[key]) for key in ('mse', 'costs', 'program',
                'status', 'confirmed_and_solved', 'fresh_confirmation_passed', 'confirmation_loss',
                'concepts_added', 'knowledge_return', 'evaluation_costs', 'selected_channels',
                'selected_kind', 'used_sources')}, 'role': role,
                'called_source_ids': helper._called_sources(row['program']), 'post_fork_actions': post})
        visits = [{key: deepcopy(item.get(key)) for key in ('task_position', 'source_task_position',
            'candidate', 'visit_costs', 'resource_limits', 'available_concepts', 'status', 'basis')}
            for item in report['revisits'][len(checkpoint['revisits']):]]
        results[arm] = {'tasks': tasks,
            **{key: report[key] for key in ('confirmed_and_solved', 'mean_mse', 'screening_compute_cost')},
            'task_sequence_costs': report['costs'], 'task_sequence_resource_limits': report['resource_limits'],
            'costs': {key: report['costs'][key] + prefix_costs[key] for key in COST_KEYS},
            'incremental_costs': {key: report['costs'][key] - checkpoint['costs'][key] for key in COST_KEYS},
            'evaluation_costs': report['evaluation_costs'],
            'full_evaluation_costs': {key: report['evaluation_costs'][key] + prefix_eval[key] for key in COST_KEYS},
            'incremental_screening_compute': report['screening_compute_cost'] - checkpoint['screening_compute_cost'],
            'new_revisits': visits,
            'selected_offer_source_ids': sorted({identifier for visit in visits
                for identifier in visit['candidate'].get('source_concept_ids', [])}),
            'paid_selection_receipts': [visit['candidate'].get('paid_selection') for visit in visits],
            'fresh_confirmation_receipts': [{key: action.get(key) for key in
                ('observation_rows', 'accepted', 'frozen_paid_selection', 'loss', 'probe_id')}
                for action in actions if action['action'] == 'validate_continued_program'],
            'new_scheduler_events': report['scheduler_events'][len(checkpoint['scheduler_events']):],
            'training_cpu_seconds': common['prefix_training_cpu_seconds'] + common['demand_training_cpu_seconds']
                + producer['training_cpu_seconds'] + trained[arm]['training_cpu_seconds'],
            'revisit_cpu_seconds': trained[arm]['training_cpu_seconds'],
            'evaluation_cpu_seconds': scored[arm]['suffix_evaluation_cpu_seconds'],
            'initial_concept_ids': trained[arm]['initial_concept_ids'],
            'changed_knowledge_fields': trained[arm]['changed_knowledge_fields']}
    retained, removed = (results[arm] for arm in ARMS)
    record = {'protocol': plan, 'case_path': str((output / 'case.json').resolve()),
        'runtime': str(Path(runtime).resolve()), 'producer_added_concepts': trained[ARMS[0]]['product_ids'],
        'prefix': {name: {key: value[key] for key in ('confirmed_and_solved', 'mean_mse',
            'costs', 'evaluation_costs')} for name, value in prefix.items()},
        'common_prefix_costs': prefix_costs, 'common_prefix_evaluation_costs': prefix_eval,
        'common_production_costs': {key: checkpoint['costs'][key] - common['partial_demand']['costs'][key]
                                  for key in COST_KEYS},
        'fork_costs': checkpoint['costs'],
        'fork_remaining': {key: retained['task_sequence_resource_limits'][key] - checkpoint['costs'][key]
                           for key in COST_KEYS},
        'programs_identical': [left['program'] == right['program']
            for left, right in zip(retained['tasks'], removed['tasks'])],
        'fresh_confirmation_identical': retained['fresh_confirmation_receipts'] == removed['fresh_confirmation_receipts'],
        'results': results, 'paired_retain_minus_remove': {
            'costs': {key: retained['costs'][key] - removed['costs'][key] for key in COST_KEYS},
            'confirmed_and_solved': retained['confirmed_and_solved'] - removed['confirmed_and_solved'],
            'incoming_mse': retained['tasks'][0]['mse'] - removed['tasks'][0]['mse'],
            'mean_mse': retained['mean_mse'] - removed['mean_mse']}, 'external_model_calls': 0}
    _write(output / 'case.json', record)
    # Original frozen paid evidence plus this runner reconstructs both continuations.
    # Preserve compact actual receipts; discard duplicate raw observation payloads.
    for arm in ARMS:
        (output / f'{arm}_training.json').unlink()
        (output / f'{arm}_scored.json').unlink()
    return record


def summarize(cases, failures, plan):
    totals = {}
    for arm in ARMS:
        rows = [case['results'][arm] for case in cases]
        totals[arm] = {
            'stages': {role: {'tasks': len(rows),
                'confirmed_and_solved': sum(row['tasks'][position]['confirmed_and_solved'] for row in rows),
                'mean_mse': mean(row['tasks'][position]['mse'] for row in rows) if rows else None,
                'costs': {key: sum(row['tasks'][position]['costs'][key] for row in rows) for key in COST_KEYS}}
                for position, role in enumerate(('incoming_demand', 'producer'))},
            **{field: {key: sum(row[field][key] for row in rows) for key in COST_KEYS}
               for field in ('costs', 'incremental_costs', 'full_evaluation_costs')},
            'incremental_screening_compute': sum(row['incremental_screening_compute'] for row in rows),
            'native_revisit_cases': sum(bool(row['new_revisits']) for row in rows),
            'selected_offer_sources': dict(Counter(identifier for row in rows for identifier in row['selected_offer_source_ids'])),
            'fresh_confirmation_rows': [receipt['observation_rows'] for row in rows for receipt in row['fresh_confirmation_receipts']],
            'training_cpu_seconds': sum(row['training_cpu_seconds'] for row in rows)}
    random, intervals = Random(plan['bootstrap_seed']), {}
    for metric in (*COST_KEYS, 'confirmed_and_solved', 'incoming_mse', 'mean_mse'):
        differences = [(case['paired_retain_minus_remove']['costs'][metric] if metric in COST_KEYS
            else case['paired_retain_minus_remove'][metric]) for case in cases]
        draws = sorted(mean(random.choice(differences) for _ in differences)
            for _ in range(plan['bootstrap_samples'])) if cases else []
        intervals[metric] = {'mean_difference': mean(differences) if cases else None,
            'interval_95': [draws[int(.025*(len(draws)-1))], draws[int(.975*(len(draws)-1))]] if draws else None}
    return {'protocol': plan, 'completed_cases': len(cases), 'execution_failures': failures,
        'prefix_confirmed_and_solved': sum(phase['confirmed_and_solved'] for case in cases for phase in case['prefix'].values()),
        'totals': totals, 'paired_retain_minus_remove': intervals,
        'identical_program_cases': sum(all(case['programs_identical']) for case in cases),
        'identical_fresh_confirmation_cases': sum(case['fresh_confirmation_identical'] for case in cases),
        'cases': [{'seed': case['protocol']['seed'], 'path': case['case_path'],
            'products': case['producer_added_concepts'], 'paired': case['paired_retain_minus_remove']} for case in cases],
        'external_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--source-case', type=Path)
    parser.add_argument('--worker', choices=ARMS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.runtime, args.source_case, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    registration = args.output / 'protocol.json'
    plan = _read(registration) if registration.exists() else cohort_protocol(args.source)
    if not registration.exists():
        _write(registration, plan)
    if (args.output / 'summary.json').exists():
        raise FileExistsError('preserve completed results and use a new output directory')
    cases, failures = [], []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(run_study, case, args.output / f"case_{case['seed']}",
            runtime=args.runtime or plan['runtime']): case for case in plan['cases']}
        for job in as_completed(jobs):
            case = jobs[job]
            try:
                result = job.result()
                cases.append(result)
                print(json.dumps({'seed': case['seed'], 'status': 'completed',
                    'joint': {arm: result['results'][arm]['confirmed_and_solved'] for arm in ARMS},
                    'paired': result['paired_retain_minus_remove']}), flush=True)
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

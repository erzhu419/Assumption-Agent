#!/usr/bin/env python3
"""Frozen-cohort experiments on online subquestions and structural transfer.

The lab parameters are evaluator-only. Researchers receive a lab's public API,
not this protocol's intervention labels. No language-model calls are made here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from random import Random
import sys
from time import process_time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / 'src'))
ARMS = ('recursive', 'flat', 'frozen_method')
MECHANISMS = ('linear', 'quadratic', 'absolute', 'threshold')
CONDITIONS = (
    'same_structure_same_surface',
    'same_structure_changed_surface',
    'changed_structure_same_surface',
    'changed_structure_changed_surface',
)
PROFILE_SEEDS = {
    'development': tuple(78101003 + 1009 * i for i in range(6)),
    'main': tuple(79101013 + 1009 * i for i in range(30)),
}
BINARY_METRICS = (
    'hypothesis_executable', 'correct_law', 'fresh_confirmation_passed',
    'independent_prediction_passed', 'confirmed_and_correct',
    'false_confirmation', 'successful_direct_call', 'method_revised',
    'successful_method_revision', 'online_subresearch_completed',
    'old_method_step_outperformed',
)


def write_json(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=True, allow_nan=False)


def task_spec(seed, position, *, changed_structure=False, changed_surface=False):
    """Construct a factorial intervention, not a fresh random-seed-only task."""
    family = position % len(MECHANISMS)
    sign = 1 if (position // 2) % 2 == 0 else -1
    offset = position % 3 - 1
    return {
        'seed': seed,
        'surface': int(changed_surface),
        'mechanism': MECHANISMS[(family + int(changed_structure)) % len(MECHANISMS)],
        'driver': position % 2,
        'orientation': 1 if (position // 4) % 2 == 0 else -1,
        'calibration_sign': -sign if changed_surface else sign,
        'calibration_offset': (-1, 0, 1)[(position + 1) % 3] if changed_surface else offset,
    }


def cohort_protocol(profile='development'):
    """The 6 development and 30 main source chains are disjoint and predetermined."""
    seeds = PROFILE_SEEDS[profile]
    cases = []
    for position, seed in enumerate(seeds):
        consumers = []
        for changed_structure, changed_surface in ((False, False), (False, True),
                                                    (True, False), (True, True)):
            condition = CONDITIONS[2 * int(changed_structure) + int(changed_surface)]
            consumers.append({
                'task_id': f'{seed}:{condition}', 'condition': condition,
                'same_structure': not changed_structure,
                'same_surface': not changed_surface,
                'lab': task_spec(seed, position, changed_structure=changed_structure,
                                 changed_surface=changed_surface),
                'evaluation_seed': seed + 10000019 + len(consumers) * 100003,
            })
        cases.append({'chain_id': str(seed), 'seed': seed,
            'source': {'task_id': f'{seed}:source', 'lab': task_spec(seed, position),
                       'evaluation_seed': seed + 10000019 + 4 * 100003},
            'consumers': consumers})
    return {
        'experiment': 'online_inquiry_and_factorial_transfer',
        'profile': profile, 'cases': cases, 'arms': list(ARMS),
        'registered_chain_count': len(cases),
        'cohorts': {name: list(values) for name, values in PROFILE_SEEDS.items()},
        'observation_budget': 24,
        'compute_budget': 1000000,
        'evaluation': {'length': 256, 'mse_threshold': 1e-10},
        'primary_outcomes': list(BINARY_METRICS),
        'primary_comparison': ['recursive', 'flat'],
        'diagnostic_intervention': 'frozen_method uses the same recursive dependency tree '
            'and source procedure but cannot rewrite a stored experiment step before '
            'sampling. Both recursive modes can use fresh search after stored steps are '
            'exhausted. This tests online step rewriting, not later use of newly saved methods.',
        'source_freeze': 'One recursive source acquisition per chain; all modes and four '
            'consumers inherit independent copies of exactly this paid source state. This '
            'is a paired continuation intervention, not an end-to-end from-scratch ranking.',
        'transfer_interventions': 'Structure change replaces the response family by the next '
            'member of linear, quadratic, absolute, threshold. Surface change renames the '
            'interface, permutes its control positions, and reverses instrument sign while '
            'changing its offset. The physical '
            'driver and orientation remain fixed. Source acquisition never sees the new '
            'surface or replacement family, and consumer knowledge never crosses forks. '
            'These are held-out mechanism-by-binding combinations within the public grammar, '
            'not structures outside its four response families.',
        'fairness': 'Identical public controls, measurements, calibrations, finite hypothesis '
            'language, source knowledge, observation cap and confirmation threshold. Flat '
            'and recursive differ in research organization. Costs constrain research and '
            'describe behavior; savings are not a success criterion.',
        'terminal_evaluation': 'Complete and freeze all consumers in a chain before any '
            'blind terminal evaluation. Terminal outcomes never enter research state. '
            'Correctness includes fresh confirmation and independent prediction; symbolic '
            'equivalence is reported separately and may be unknown for arbitrary programs. '
            'correct_law means predictive equivalence within the finite grammar, not exact '
            'AST or parameter recovery: opposite orientations of a quadratic are equivalent.',
        'process_outcomes': 'Resolved child questions and consumed answers establish online '
            'dependency execution. Their counts are not evidence of an accuracy advantage. '
            'Causal effects require paired action or candidate interventions.',
        'failure_policy': 'Retain every registered chain and every unsuccessful or errored '
            'task. No replacement seeds, post-result filters, or threshold changes.',
        'resources': 'Report actual task observations and research CPU. The shared source '
            'prefix is charged once per four-consumer arm when describing a whole chain.',
        'external_model_calls': 0,
    }


def make_lab(spec):
    from hegel_machine.cognition.online_inquiry import InquiryLab
    return InquiryLab(**spec)


def evaluate_record(report, lab, *, evaluation_seed, evaluation_length=256,
                    mse_threshold=1e-10):
    """Common terminal result schema; usable with an external-LLM research report.

    `report` is an already completed record containing the serialized executable
    Program and confirmation outcome. This function never invokes a researcher.
    """
    from hegel_machine.cognition.language import Program

    confirmed = bool(report.get('confirmed', False))
    evaluation, evaluation_error = None, None
    executable = False
    if report.get('program') is not None:
        try:
            program = Program.from_dict(report['program'])
            evaluation = lab.blind_evaluate(program, length=evaluation_length,
                                            seed=evaluation_seed)
            mse = evaluation.get('independent_mse', evaluation.get('mse'))
            executable = mse is not None and isfinite(float(mse))
        except Exception:
            evaluation_error = traceback.format_exc()
    mse = None if evaluation is None else evaluation.get('independent_mse', evaluation.get('mse'))
    prediction_passed = executable and float(mse) <= mse_threshold
    structural = None if evaluation is None else evaluation.get('structural_equivalence')
    nodes = report.get('nodes', [])
    if isinstance(nodes, dict):
        nodes = list(nodes.values())
    resolved_children = [node for node in nodes
        if node.get('parent_id') is not None and node.get('opened_at') is not None
        and node.get('resolved_at') is not None
        and node.get('status') == 'resolved'
        and node['opened_at'] < node['resolved_at']]
    events = report.get('events', [])
    resolved_ids = {node.get('question_id') for node in resolved_children}
    consumed = [event for event in events if event.get('event') == 'parent_resumed'
                and event.get('consumes_answer') in resolved_ids]
    replays = [event for event in events if event.get('event') == 'method_replay_finished']
    method_steps = [event for event in events if event.get('event') in
                    {'method_rewritten', 'method_step_reused'}]
    outperformed_steps = [event for event in method_steps
                          if event['new_information_gain'] > event['old_information_gain'] + 1e-10]
    direct_call = bool(report.get('direct_call', False))
    revised = bool(report.get('method_revisions', []))
    return {
        'evaluation': evaluation, 'evaluation_error': evaluation_error,
        'metrics': {
            'hypothesis_executable': executable,
            'correct_law': None if structural is None else bool(structural),
            'fresh_confirmation_passed': confirmed,
            'independent_prediction_passed': bool(prediction_passed),
            'confirmed_and_correct': bool(confirmed and prediction_passed),
            'false_confirmation': bool(confirmed and not prediction_passed),
            'direct_call': direct_call,
            'successful_direct_call': bool(direct_call and confirmed and prediction_passed),
            'method_revised': revised,
            'successful_method_revision': bool(revised and confirmed and prediction_passed),
            'online_subresearch_completed': bool(resolved_children),
            'resolved_subquestions': len(resolved_children),
            'consumed_child_answers': len(consumed),
            'frozen_replay_resolved': bool(replays[-1]['resolved']) if replays else None,
            'old_method_step_outperformed': bool(outperformed_steps),
            'old_method_steps_outperformed': len(outperformed_steps),
        },
    }


def _train(spec, *, mode, state, observation_budget, compute_budget):
    from hegel_machine.cognition.online_inquiry import OnlineInquiryResearcher

    lab = make_lab(spec)
    researcher = OnlineInquiryResearcher(mode=mode, state=deepcopy(state),
                                         observation_budget=observation_budget,
                                         compute_budget=compute_budget)
    start = process_time()
    report = researcher.run(lab)
    return {'research': deepcopy(report), 'public_spec': lab.public_spec(),
            'training_cpu_seconds': process_time() - start}, deepcopy(researcher.to_dict()), lab


def run_case(case, plan, output):
    """Freeze one source, run the full factorial continuation, then score."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', case)
    source, state, source_lab = _train(case['source']['lab'], mode='recursive', state=None,
                                     observation_budget=plan['observation_budget'],
                                     compute_budget=plan['compute_budget'])
    write_json(output / 'source_knowledge.json', state)
    pending = []
    # Randomizing execution order changes neither tasks nor available evidence.
    schedule = [(arm, task) for arm in plan['arms'] for task in case['consumers']]
    Random(case['seed'] + 29).shuffle(schedule)
    for arm, task in schedule:
        try:
            record, frozen_state, lab = _train(task['lab'], mode=arm, state=state,
                                              observation_budget=plan['observation_budget'],
                                              compute_budget=plan['compute_budget'])
            record.update({'task_id': task['task_id'], 'arm': arm,
                           'condition': task['condition'], 'status': 'completed',
                           'frozen_state': frozen_state})
        except Exception:
            lab = make_lab(task['lab'])
            record = {'task_id': task['task_id'], 'arm': arm, 'condition': task['condition'],
                      'status': 'error', 'error': traceback.format_exc(), 'research': {}}
        pending.append((record, lab, task['evaluation_seed']))
    # At this point every research run is complete. No score has been disclosed.
    source.update(evaluate_record(source['research'], source_lab,
        evaluation_seed=case['source']['evaluation_seed'],
        evaluation_length=plan['evaluation']['length'],
        mse_threshold=plan['evaluation']['mse_threshold']))
    results = []
    for record, lab, seed in pending:
        record.update(evaluate_record(record['research'], lab, evaluation_seed=seed,
            evaluation_length=plan['evaluation']['length'],
            mse_threshold=plan['evaluation']['mse_threshold']))
        results.append(record)
    results.sort(key=lambda row: (plan['arms'].index(row['arm']), CONDITIONS.index(row['condition'])))
    value = {'chain_id': case['chain_id'], 'seed': case['seed'], 'source': source,
             'results': results, 'external_model_calls': 0}
    write_json(output / 'case.json', value)
    return value


def _aggregate(rows, expected):
    metrics = {}
    for field in BINARY_METRICS:
        known = [row['metrics'].get(field) for row in rows
                 if row.get('metrics', {}).get(field) is not None]
        successes = sum(value is True for value in known)
        metrics[field] = {'count': successes, 'registered_tasks': expected,
                          'assessed_tasks': len(known),
                          'rate_over_registered': successes / expected if expected else None}
    costs = Counter()
    for row in rows:
        for key, value in row.get('research', {}).get('costs', {}).items():
            if isinstance(value, (int, float)):
                costs[key] += value
    return {'registered_tasks': expected, 'recorded_tasks': len(rows),
        'runtime_errors': sum(row.get('status') == 'error' for row in rows),
        'evaluation_errors': sum(row.get('evaluation_error') is not None for row in rows),
        'metrics': metrics, 'costs': dict(costs),
        'training_cpu_seconds': sum(row.get('training_cpu_seconds', 0.0) for row in rows)}


def summarize(cases, failures, plan):
    """Keep registered denominators, including unsuccessful source acquisitions."""
    expected = len(plan['cases'])
    by_id = {case['chain_id']: case for case in cases}
    totals = {}
    for arm in plan['arms']:
        rows = [row for case in cases for row in case['results'] if row['arm'] == arm]
        totals[arm] = _aggregate(rows, expected * len(CONDITIONS))
        totals[arm]['conditions'] = {condition: _aggregate(
            [row for row in rows if row['condition'] == condition], expected)
            for condition in CONDITIONS}
        totals[arm]['complete_chains_correct'] = sum(
            len([row for row in case['results'] if row['arm'] == arm
                 and row['metrics']['confirmed_and_correct']]) == len(CONDITIONS)
            for case in cases)
        source_costs = Counter()
        for case in cases:
            source_costs.update(case['source']['research'].get('costs', {}))
        totals[arm]['chain_costs_including_source_once'] = dict(
            Counter(totals[arm]['costs']) + source_costs)
    pairs = {}
    for comparator in [arm for arm in plan['arms'] if arm != 'recursive']:
        values = {}
        for condition in CONDITIONS:
            counts = Counter()
            for registered in plan['cases']:
                case = by_id.get(registered['chain_id'])
                indexed = {} if case is None else {
                    row['arm']: row for row in case['results'] if row['condition'] == condition}
                left, right = indexed.get('recursive'), indexed.get(comparator)
                if left is None or right is None:
                    counts['missing_pair'] += 1
                    continue
                a, b = (bool(row['metrics']['confirmed_and_correct']) for row in (left, right))
                counts['both_correct' if a and b else 'recursive_only' if a else
                       'comparator_only' if b else 'neither_correct'] += 1
            values[condition] = {key: counts[key] for key in (
                'both_correct', 'recursive_only', 'comparator_only', 'neither_correct', 'missing_pair')}
            values[condition]['paired_success_difference'] = (
                counts['recursive_only'] - counts['comparator_only']) / expected if expected else None
        pairs[f'recursive_vs_{comparator}'] = values
    return {
        'experiment': plan['experiment'], 'profile': plan['profile'],
        'registered_chains': expected, 'completed_chains': len(cases), 'failed_chains': failures,
        'source_confirmed_and_correct': sum(case['source']['metrics']['confirmed_and_correct'] for case in cases),
        'totals': totals, 'paired_continuations': pairs,
        'interpretation': 'Primary outcomes are valid laws, independent predictions, false '
            'confirmations and successful knowledge use. Equal success is a valid result. '
            'Method-revision successes and child-answer counts are descriptive. Recursive '
            'versus frozen_method tests permission to rewrite stored steps online under '
            'the same recursive organization, with fresh fallback available to both. '
            'These forks do not measure later transfer of the newly saved method. Costs '
            'are secondary.',
        'external_model_calls': 0,
    }


def run_cohort(plan, output, *, workers=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', plan)
    cases, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, case, plan, output / f"case_{case['seed']}"): case
                   for case in plan['cases']}
        for future in as_completed(futures):
            registered = futures[future]
            try:
                cases.append(future.result())
            except Exception:
                failure = {'chain_id': registered['chain_id'], 'seed': registered['seed'],
                           'error': traceback.format_exc()}
                failures.append(failure)
                write_json(output / f"failure_{registered['seed']}.json", failure)
            print(f"completed {len(cases) + len(failures)}/{len(plan['cases'])} chains", flush=True)
    cases.sort(key=lambda case: case['seed'])
    summary = summarize(cases, failures, plan)
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILE_SEEDS, default='development')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--protocol-only', action='store_true')
    args = parser.parse_args()
    plan = cohort_protocol(args.profile)
    if args.protocol_only:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / 'protocol.json', plan)
        print(args.output / 'protocol.json')
    else:
        result = run_cohort(plan, args.output, workers=args.workers)
        print(json.dumps({'registered_chains': result['registered_chains'],
                          'completed_chains': result['completed_chains'],
                          'failed_chains': len(result['failed_chains'])}))


if __name__ == '__main__':
    main()

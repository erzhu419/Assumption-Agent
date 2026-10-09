#!/usr/bin/env python3
"""Test measured workflow value on an explicitly paid unfinished consumer."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from random import Random
import sys
from time import process_time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_current_offline_ablation import _compact
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for

COST_KEYS = ('observation_cost', 'measurement_cost', 'compute_cost')
ARMS = ('retained_returns', 'removed_returns')


def protocol(seed=63007001, producer_task_index=10):
    config = ResearchConfig(observation_budget=192, compute_budget=120000,
        minimum_improvement=1e-7, max_rounds=4, max_candidates=16,
        active_probes=False, seed=seed)
    arms = list(ARMS)
    Random(seed + 17011).shuffle(arms)
    return {'experiment': 'paid_demand_workflow_value_development', 'seed': seed,
        'return_tasks': [{'task_index': 8, 'seed': seed}, {'task_index': 8, 'seed': seed + 1}],
        'competition_tasks': [{'task_index': 1, 'seed': seed + 10007}],
        'tasks': [{'task_index': 8, 'seed': seed + 20014},
                  {'task_index': producer_task_index, 'seed': seed + 30021}],
        'research_config': asdict(config), 'measurement_budget': 768,
        'partial_compute_budget': 512, 'partial_probe_seed': seed,
        'demand_collection': 'independent incoming partially researched consumer',
        'evaluation': {'length': 24, 'trajectories': 2, 'seed_offset': 1000003},
        'solved_threshold': 1e-6, 'arms': arms, 'arm_order_seed': seed + 17011,
        'information_contract': 'Learn a real task-8 source and confirm its later return; '
            'then normally research task 1. A separate empty-library researcher buys a new '
            'partial task-8 visit with the same public control-probe seed and an independent '
            'laboratory seed. Hand its real unfinished report to the shared-library session. Clone all state '
            'and remove only workflow knowledge_outcomes from one arm. Both arms may learn '
            'new knowledge and receive new returns. Never supply response programs or returns.',
        'accounting': 'The source prefix, competing-method visit and partial demand are physically '
            'shared. Each counterfactual bill includes them once. Native successor research and '
            'revisits share only their two-task remaining caps; no earlier-prefix budget is borrowed. '
            'All learning finishes before terminal scoring. Record inactive and negative cases.',
        'scope': 'A repeated observed consumer condition tests value consumption, not generalization '
            'of response signatures. Exact condition matches and executable competing workflows '
            'are observed outcomes, not imposed successful activation.'}


def cohort_protocol():
    return {'experiment': 'workflow_demand_12_registered_consumers', 'bootstrap_samples': 10000,
        'bootstrap_seed': 63007019,
        'cases': [protocol(63107001 + 100003 * index) for index in range(12)],
        'case_policy': 'Retain every registered case, including inactive interventions, failed '
            'research, negative benefit and execution exceptions. Selection change is operational '
            'evidence; compare full paid costs and independent terminal quality separately.'}


def _fork_state(state, arm):
    cloned = deepcopy(state)
    if arm == 'removed_returns':
        for method in cloned['knowledge']['research_workflows']['methods']:
            if 'knowledge_outcomes' in method:
                method['knowledge_outcomes'] = []
    return cloned


def _partial_visit(session, plan, factory, *, incoming=False):
    task = session.state['tasks'][0]
    library = KnowledgeLibrary() if incoming else session.library
    available = [concept.concept_id for concept in library.concepts]
    with factory(task) as laboratory:
        report = UnifiedResearchAgent(replace(session.config,
            seed=plan['partial_probe_seed'], compute_budget=plan['partial_compute_budget']),
            measurement_budget=plan['measurement_budget'], library=library).run(laboratory)
    session.state['results'].append({'task_position': 0, 'available_concepts': available,
                                     'research': report})
    session.state['status'] = 'paused'
    return deepcopy(report)


def _summary(session, plan, tasks):
    report = session.evaluate(**plan['evaluation'])
    result = _compact(report, {**plan, 'tasks': tasks})
    for row, entry, score in zip(result['tasks'], report['results'], report['evaluation']):
        research = entry['research']
        confirmed = research['status'] == 'completed' and research.get('confirmation_loss') is not None
        confirmed = confirmed and research['confirmation_loss'] <= plan['research_config']['minimum_improvement']
        row.update(fresh_confirmation_passed=confirmed, confirmed_and_solved=confirmed and row['solved'],
            actions=deepcopy(research['actions']), knowledge_return=deepcopy(research.get('knowledge_return')),
            research_workflow_id=research.get('research_workflow_id'),
            program=research['program'], concepts_added=research.get('concepts_added', []),
            evaluation_costs={key: score[key] for key in COST_KEYS})
    result.update(fresh_confirmed=sum(row['fresh_confirmation_passed'] for row in result['tasks']),
        confirmed_and_solved=sum(row['confirmed_and_solved'] for row in result['tasks']),
        evaluation_costs={key: sum(row['evaluation_costs'][key] for row in result['tasks']) for key in COST_KEYS},
        final_workflows=deepcopy(report['knowledge']['research_workflows']),
        scheduler_events=deepcopy(report['scheduler_events']), revisits=deepcopy(report['revisits']))
    return result


def run_study(plan, lab_factory=None):
    factory = lab_factory or laboratory_for
    config = ResearchConfig(**plan['research_config'])
    start = process_time()
    source = UnifiedResearchSession(config, tasks=plan['return_tasks'],
        measurement_budget=plan['measurement_budget'], library=KnowledgeLibrary(), lab_factory=factory)
    initial_source = _partial_visit(source, plan, factory)
    source.run(max_tasks=1)
    queued_source = deepcopy(source.to_dict()['pending_revisits'])
    source.run()
    return_book = source.library.research_workflows.to_dict()
    source_cpu = process_time() - start
    start = process_time()
    competing = UnifiedResearchSession(config, tasks=plan['competition_tasks'],
        measurement_budget=plan['measurement_budget'],
        library=KnowledgeLibrary.from_dict(source.library.to_dict()), lab_factory=factory)
    competing.run()
    competing_cpu = process_time() - start
    start = process_time()
    demand = UnifiedResearchSession(config, tasks=plan['tasks'],
        measurement_budget=plan['measurement_budget'],
        library=KnowledgeLibrary.from_dict(competing.library.to_dict()), lab_factory=factory)
    partial = _partial_visit(demand, plan, factory, incoming=True)
    demands = demand._workflow_demands()
    checkpoint = demand.to_dict()
    demand_cpu = process_time() - start
    receipts = [receipt for method in checkpoint['knowledge']['research_workflows']['methods']
        for outcome in method.get('knowledge_outcomes', []) for receipt in outcome.get('source_returns', [])]
    confirmed_returns = [receipt for receipt in receipts
        if receipt.get('confirmation_status') == 'confirmed' and receipt.get('retained_normalized_gain', 0.) > 0.]
    sessions, cpus, initial_books = {}, {}, {}
    for arm in plan['arms']:
        state = _fork_state(checkpoint, arm)
        initial_books[arm] = deepcopy(state['knowledge']['research_workflows'])
        session = UnifiedResearchSession(state=state, lab_factory=factory)
        start = process_time()
        session.run()
        sessions[arm], cpus[arm] = session, process_time() - start

    # No terminal observations are available until both counterfactual continuations finish.
    start = process_time()
    prefix = {'source_return': _summary(source, plan, plan['return_tasks']),
              'competing_method': _summary(competing, plan, plan['competition_tasks'])}
    prefix_evaluation_cpu = process_time() - start
    prefix_costs = {key: sum(row['costs'][key] for row in prefix.values()) for key in COST_KEYS}
    prefix_evaluation_costs = {key: sum(row['evaluation_costs'][key] for row in prefix.values()) for key in COST_KEYS}
    results = {}
    for arm, session in sessions.items():
        start = process_time()
        result = _summary(session, plan, plan['tasks'])
        replays = [{**deepcopy(action), 'task_position': row['task_position']}
            for row in result['tasks'] for action in row['actions'][
                len(partial['actions']) if row['task_position'] == 0 else 0:]
            if action['action'] == 'replay_research_workflow']
        result.update(initial_workflows=initial_books[arm], replays=replays,
            task_sequence_costs=deepcopy(result['costs']),
            task_sequence_resource_limits=deepcopy(result['resource_limits']),
            resource_limits={key: result['resource_limits'][key] + sum(
                row['resource_limits'][key] for row in prefix.values()) for key in COST_KEYS},
            costs={key: result['costs'][key] + prefix_costs[key] for key in COST_KEYS},
            continuation_costs={key: result['costs'][key] - partial['costs'][key] for key in COST_KEYS},
            training_cpu_seconds=source_cpu + competing_cpu + demand_cpu + cpus[arm],
            continuation_cpu_seconds=cpus[arm], evaluation_cpu_seconds=process_time() - start,
            full_evaluation_costs={key: result['evaluation_costs'][key] + prefix_evaluation_costs[key]
                                   for key in COST_KEYS},
            action_counts=dict(Counter(action['action'] for row in result['tasks'] for action in row['actions'])),
            nonzero_value_replays=sum(any(action.get('knowledge_priority', (0., 0.))) for action in replays))
        results[arm] = result
    retained, removed = (results[arm] for arm in ARMS)
    choices = {arm: [row['method_id'] for row in result['replays'] if row['task_position'] == 1]
               for arm, result in results.items()}
    return {'protocol': deepcopy(plan), 'prefix': prefix, 'common_prefix_costs': prefix_costs,
        'common_prefix_training_cpu_seconds': source_cpu + competing_cpu,
        'common_prefix_evaluation_cpu_seconds': prefix_evaluation_cpu,
        'source_initial_status': initial_source['status'], 'source_queued_offers': queued_source,
        'source_return_workflows': return_book, 'partial_demand': {
            'status': partial['status'], 'costs': partial['costs'], 'actions': partial['actions'],
            'demands': demands, 'probe_seed': plan['partial_probe_seed'],
            'laboratory_seed': plan['tasks'][0]['seed'], 'training_cpu_seconds': demand_cpu},
        'learned_workflow_count': len(checkpoint['knowledge']['research_workflows']['methods']),
        'confirmed_positive_returns': len(confirmed_returns),
        'demand_matches_confirmed_return': any(d['condition'] == r['condition']
            for d in demands for r in confirmed_returns),
        'producer_workflow_choices': choices, 'producer_choice_changed': choices[ARMS[0]] != choices[ARMS[1]],
        'results': results, 'paired_retained_minus_removed': {
            'costs': {key: retained['costs'][key] - removed['costs'][key] for key in COST_KEYS},
            'confirmed_and_solved': retained['confirmed_and_solved'] - removed['confirmed_and_solved'],
            'mean_mse': retained['mean_mse'] - removed['mean_mse']}, 'external_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=63007001)
    parser.add_argument('--producer-task-index', type=int, choices=(9, 10), default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registration = args.output.with_suffix('.protocol.json')
    if args.output.exists() or registration.exists():
        raise FileExistsError('preserve existing results and choose another output')
    plan = protocol(args.seed, args.producer_task_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registration.write_text(json.dumps(plan, indent=2) + '\n')
    try:
        result = run_study(plan)
    except Exception:
        args.output.write_text(json.dumps({'protocol': plan, 'status': 'execution_failed',
            'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('learned_workflow_count', 'confirmed_positive_returns',
        'demand_matches_confirmed_return', 'producer_workflow_choices', 'producer_choice_changed',
        'paired_retained_minus_removed')}), flush=True)


if __name__ == '__main__':
    main()

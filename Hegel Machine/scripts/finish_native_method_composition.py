#!/usr/bin/env python3
"""After a frozen unit closes, compile and consume its retained method history."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from scripts.return_partition_pair import merge_pair_returns
from scripts.stage_partition_method import prepare_paid
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.common_object_research import CommonObjectDependencyMethods
from hegel_machine.cognition.method_feedback import ensure_python_method_topic
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.staged_method_research import record_staged_method_practice


def run(output):
    output = Path(output)
    cfg = load(output / 'registration.json')
    if (output / 'receipt.json').exists():
        return load(output / 'receipt.json')
    unit = Path(cfg['completed_unit'])
    if not (unit / 'receipt.json').exists():
        return {'status': 'waiting_for_frozen_unit', 'new_compute_cost': 0}
    closed = load(unit / 'receipt.json')
    if (output / 'started.json').exists():
        raise ValueError('Preserve the interrupted method occurrence; do not repeat its analysis')
    costs = read_method_reference(closed['closed_source_costs_ref'], unit)
    available = closed['remaining_compute']
    balance_ref = ref(unit / 'receipt.json', 'remaining_compute')
    if cfg.get('previous_occurrence_ref'):
        prior_account = read_method_reference(cfg['previous_occurrence_ref'], output)
        if prior_account['closed_unit_ref'] != ref(unit / 'receipt.json'):
            raise ValueError('Keep the same completed unit and its actual later account')
        costs = read_method_reference(prior_account['source_costs_ref'], output)
        available = prior_account['remaining_compute']
        balance_ref = {**cfg['previous_occurrence_ref'], 'json_path': ['remaining_compute']}
        prior = Path(cfg['previous_occurrence_ref']['file']).parent / 'method_return'
    limit = min(cfg['maximum_compute'], available)
    if limit <= cfg['path_reserve']:
        return {'status': 'method_formation_unfunded', 'new_compute_cost': 0}
    if not cfg.get('previous_occurrence_ref'):
        prior = Path(closed['latest_method_review'])
    owner = next(item for item in load(prior / 'common/questions.json')
                 if item['id'] == 'task:method_choice')
    pairs = []
    for location in owner['method_use_plan']['native_return_refs']:
        record = read_method_reference(location, prior)
        pair = record.get('method_pair_ref')
        if pair is not None and pair not in pairs:
            pairs.append(pair)
    if not pairs:
        return {'status': 'retained_method_provenance_missing', 'new_compute_cost': 0}
    save(output / 'started.json', {'closed_unit_ref': ref(unit / 'receipt.json'),
        'source_costs_ref': closed['closed_source_costs_ref'], 'maximum_compute': limit,
        'ledger_scope': 'suballocation_of_closed_unit_balance_not_an_extra_budget'})
    formation_allocation = output / 'formation_allocation.json'
    save(formation_allocation, {'allocation_id': output.name + ':formation',
        'maximum_compute': limit - cfg['path_reserve'], 'new_observation_limit': 0,
        'parent_balance_ref': balance_ref})
    spent = 0
    try:
        if cfg.get('completed_formation_ref'):
            formation_path = Path(cfg['completed_formation_ref']['file']).parent
            if formation_path != prior:
                raise ValueError('Resume the actual completed formation in this paid occurrence')
            formed = read_method_reference(cfg['completed_formation_ref'], output)
        else:
            formation_path = output / 'method_return'
            formed = merge_pair_returns(prior, formation_path, pairs, ref(formation_allocation))
            spent += formed['compute_cost']
        action = formed['return'].get('research_result', {}).get('method_action_research', {})
        candidates = []
        for published in action.get('staged_method_proposals', []):
            location = published.get('proposal_ref')
            if not location:
                continue
            proposal = read_method_reference(location, output)
            if (proposal['basis'].get('operation') == 'revise_then_reidentify'
                    and proposal['basis'].get('actual_result_ref') == cfg['source_execution_ref']):
                candidates.extend((location, method) for method in published['research_program_refs'])
        result = {'status': 'composition_not_formed', 'candidate_count': len(candidates)}
        if len(candidates) == 1:
            proposal_ref, method_ref = candidates[0]
            allocation = output / 'path_allocation.json'
            save(allocation, {'allocation_id': output.name + ':paid_path',
                'maximum_compute': min(cfg['path_reserve'], limit - spent),
                'formation_reserve': 0, 'new_observation_limit': 0,
                'source_formation_ref': ref(formation_path / 'receipt.json'),
                'parent_balance_ref': balance_ref})
            path = output / 'paid_path'
            prepare_paid([Path(cfg['source_execution_ref']['file']).parent], method_ref,
                ref(allocation), path, proposal_ref=proposal_ref)
            env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONPATH': os.pathsep.join(str(path / 'runtime' / part) for part in ('src', '.', 'scripts'))}
            with (path / 'process.log').open('a') as log:
                done = subprocess.run([sys.executable, '-B', '-u',
                    str(path / 'runtime/scripts/stage_partition_method.py'), 'run-paid',
                    '--output', str(path)], cwd=path / 'runtime', env=env, stdout=log, stderr=subprocess.STDOUT)
            practice = load(path / ('failure.json' if done.returncode else 'receipt.json'))
            spent += practice['compute_cost']
            complete = practice.get('status') == 'paid_stage_paths_completed'
            practice_return = None
            if not done.returncode:
                directory = output / 'practice_return'
                previous = next(item for item in load(formation_path / 'common/questions.json')
                    if item['id'] == 'task:method_choice')
                owner = deepcopy(previous)
                owner['mission'] = {'purpose': 'method_learning',
                    'mission_id': str(directory), 'allocation': {'compute': limit - spent},
                    'resource_usage': {'compute': 0}}
                save(directory / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
                common = CommonObjectDependencyMethods(directory / 'common', [],
                    budget=SearchBudget(limit - spent), actions=[]).common
                ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
                    topic_ref=ref(directory / 'previous_topic.json'))
                topic = next(item for item in common.questions if item['id'] == 'task:method_choice')
                returned = record_staged_method_practice(common, topic, proposal_ref=proposal_ref,
                    practice_ref=ref(path / 'receipt.json'), compute_limit=limit - spent)
                complete = complete and returned.get('execution_complete', False)
                routed = common._execute('run', {'operator': 'research_route',
                    'research_question': {'id': topic['id']},
                    'allocation': {'compute': limit - spent - common.costs['compute_cost']}}, None)
                spent += common.costs['compute_cost']
                save(directory / 'common/questions.json', common.questions)
                save(directory / 'receipt.json', {'practice_return': returned, 'return': routed,
                    'compute_cost': common.costs['compute_cost'], 'remaining_compute': limit - spent,
                    'new_fits': 0, 'new_observations': 0, 'independent_support': False})
                practice_return = ref(directory / 'receipt.json')
            result = {'status': ('composition_ready_for_successor_registration'
                if complete and closed['status'] == 'confirmed' else
                'composition_path_verified_source_confirmation_unresolved' if complete else
                'composition_path_incomplete'), 'candidate_count': 1,
                'proposal_ref': proposal_ref, 'method_ref': method_ref,
                'practice_return_ref': practice_return,
                'practice_ref': ref(path / ('failure.json' if done.returncode else 'receipt.json'))}
    except Exception as error:
        save(output / 'failure.json', {'error': repr(error), 'known_compute_cost': spent,
            'compute_cost_interval': [spent, limit], 'parent_balance_ref': balance_ref,
            'old_records_preserved': True})
        raise
    updated = {**costs, 'previous_source_costs_ref': (prior_account['source_costs_ref']
            if cfg.get('previous_occurrence_ref') else closed['closed_source_costs_ref']),
        'cumulative_protocol_compute_interval': [v + spent for v in costs['cumulative_protocol_compute_interval']],
        'complete_relation_unit_remaining': available - spent,
        'method_composition_compute': spent, 'method_composition_receipt_ref': ref(output / 'receipt.json')}
    save(output / 'closed_source_costs.json', updated)
    result.update(compute_cost=spent, remaining_compute=available - spent,
        closed_unit_ref=ref(unit / 'receipt.json'), source_costs_ref=ref(output / 'closed_source_costs.json'),
        formation_ref=ref(formation_path / 'receipt.json'),
        independent_successor_started=False, independent_support=False,
        model_calls=0, new_observations=0,
        scope='ordinary_method_formation_and_paid_path_reuse_not_new_scientific_effect')
    save(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(run(parser.parse_args().output)['status'])

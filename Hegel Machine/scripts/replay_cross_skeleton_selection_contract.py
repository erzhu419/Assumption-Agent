#!/usr/bin/env python3
"""Replay the saved PR2 panel with its declared answer and native contracts.

Both arms use their original sixteen admissions, complete old pool, panel,
observation allowance and total compute budget. The old constructors restore
the saved Programs; no new relation application, acquisition or held
continuation label participates in selection.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from audit_cross_skeleton_consumption import reconstruct_inputs, RECEIPT
from hegel_machine.cognition.language import MissingHistory
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_routing import register_theories
from replay_cross_skeleton_relations import (
    PANEL_ACTIONS, POINTER, ROOT, TOTAL_COMPUTE, ReplayLab,
    declaration, new_agent, plan_inputs, source_ref,
)


OUTPUT = ROOT / 'docs/evidence/cross_skeleton_selection_contract_20261021.json'
CONTRACTS = ('original_step1_declared_resolution', 'aligned_step0_native')


def _inputs(target, identifiers, contract, fallback, source):
    question, panel = plan_inputs(target, identifiers)
    if contract == 'aligned_step0_native':
        question['question'] = 'Which retained alternative predicts step 0 of the next public trajectory?'
        for query in [question['query_semantics']['query_expression'], *[plan['query'] for plan in panel]]:
            action = query['binding']['action']
            query.update(step=0)
            query['binding'].update(features=deepcopy(action['sequence'][0]), history=[])
        question['query_semantics']['evaluation'] = {
            'kind': 'native_evaluation', 'fallback': fallback,
            'source_ref': source_ref(source, 'episodes[6].fallback')}
    return question, panel


def replay_selection_contract(receipt_path=RECEIPT, pointer=POINTER, output=OUTPUT):
    data = reconstruct_inputs(receipt_path, pointer)
    receipt, source, baseline = data['receipt'], data['source'], data['baseline']
    target, programs = data['payload']['episodes'][6], data['programs']
    mission = {'purpose': 'theory_learning', 'mission_id': 'cross_skeleton_selection_contract',
        'theory_learning_allocation': {'compute': TOTAL_COMPUTE, 'author_calls': 0, 'observations': 0}}
    result = {'status': 'completed', 'scope': 'same_material_selection_contract_diagnostic',
        'source_receipt': str(receipt_path), 'source_archive': source,
        'protocol': {'contracts_fixed_before_both_arms': list(CONTRACTS),
            'admission': 'original saved rank/frontier admissions; sixteen per arm',
            'rank_candidates_changed': False, 'candidate_programs_changed': False,
            'panel_action_indices': list(PANEL_ACTIONS), 'same_total_compute_budget': TOTAL_COMPUTE,
            'observation_budget': 8, 'answer_atol': 1e-6, 'experiment_tolerance': 1e-6,
            'task_pair_rule': 'absolute answer difference > answer_atol, independently per pair',
            'experiment_separation_rule': 'absolute forecast difference > 2 * experiment_tolerance',
            'exact_signature_grouping_preserved': True,
            'held_continuation_labels_read_for_selection': False,
            'new_learning_or_relation_application_runs': 0, 'new_paid_observations': 0,
            'provider_calls': 0, 'original_receipt_modified': False,
            'auxiliary_raw_readout_diagnostic_compute': 0,
            'actual_old_constructor_reconstruction_compute': sum(baseline['costs'].values())},
        'retained_evidence_ref': {'file': str(receipt_path), 'field': 'counterexample'},
        'contracts': {}}
    for contract in CONTRACTS:
        arms = {}
        for arm in ('remove', 'keep'):
            original = receipt['arms'][arm]
            identifiers = original['admitted_program_ids']
            available = original['prefix_scores']['rank_order']
            if len(identifiers) != 16 or not set(identifiers) <= set(available):
                raise AssertionError('The original finite admissions are unavailable')
            question, panel = _inputs(target, identifiers, contract, baseline['fallback'], source)
            raw_missing = None
            if contract == 'aligned_step0_native':
                binding = question['query_semantics']['query_expression']['binding']
                row = Observation(binding['features'], 0., tuple(binding['history']))
                raw_missing = 0
                for identifier in identifiers:
                    try:
                        programs[identifier].predict(row)
                    except MissingHistory:
                        raw_missing += 1
                result['protocol']['auxiliary_raw_readout_diagnostic_compute'] += len(identifiers)
            with tempfile.TemporaryDirectory(prefix=f'hegel-selection-contract-{arm}-') as temporary:
                agent, laboratory = new_agent(temporary), ReplayLab()
                register_theories(agent, [declaration(identifier, programs[identifier])
                                          for identifier in available])
                # Reuse the original pre-route debit; reconstruction does not
                # rerun ranking or construction as part of this comparison.
                costs = original['compute']
                before = sum(costs[key] for key in ('old_constructors', 'prefix_ranking',
                    'object_construction', 'object_application'))
                agent.costs['compute_cost'] = before
                routed = agent._execute('run', {'operator': 'research_route',
                    'research_question': question, 'experiments': panel, 'mission': mission}, laboratory)
                selection = routed['selection']
                route_cost = agent.costs['compute_cost'] - before
                if laboratory.execute_calls or agent.costs['model_calls'] or agent.costs['observation_cost']:
                    raise AssertionError('The contract diagnostic acquired external data')
                if len(agent.theory_index) != len(available):
                    raise AssertionError('Routing changed the original retained candidate pool')
                if not all(group['known'] for group in selection['groups']):
                    raise AssertionError('A declared execution contract left an admitted prediction unknown')
                arms[arm] = {'candidate_count': len(available), 'admitted_program_ids': identifiers,
                    'readout_step': question['query_semantics']['query_expression']['step'],
                    'evaluation': deepcopy(question['query_semantics'].get('evaluation')),
                    'status': routed['status'], 'signature_count': len(selection['groups']),
                    'groups': selection['groups'], 'pairwise_edge_count': len(selection['pairwise_edges']),
                    'effective_pairs': selection['scores'][0]['effective_pairs'],
                    'scores': selection['scores'], 'tied_experiments': selection['tied_experiments'],
                    'selected_experiment_id': (selection.get('selected_experiment') or {}).get('id'),
                    'pending_query_prepared': any('pending_query' in item for item in agent.questions),
                    'public_row_validation': selection['public_row_validation'],
                    'excluded': selection['excluded'],
                    'compute': {'original_pre_route': before, 'actual_common_route': route_cost,
                        'prediction_compute': selection['compute_cost'],
                        'total_debit': agent.costs['compute_cost'],
                        'remaining': TOTAL_COMPUTE - agent.costs['compute_cost'],
                        'original_other_debits': costs['total'] - before - costs['common_route'],
                        'matched_protocol_total': costs['total'] - costs['common_route'] + route_cost,
                        'other_debits_scope': 'already executed and retained from original receipt; not rerun',
                        'original_full_compute_ledger_ref': {'file': str(receipt_path),
                            'field': f'arms.{arm}.compute'}},
                    'observation_cost': agent.costs['observation_cost'],
                    'support_updated': selection['support_updated'], 'independent_confirmation': False}
                if raw_missing is not None:
                    arms[arm].update(raw_missing_history_count=raw_missing,
                        raw_diagnostic_scope='same sixteen task readouts at step 0 without native contract',
                        raw_diagnostic_compute=len(identifiers))
                if agent.costs['compute_cost'] > TOTAL_COMPUTE:
                    raise AssertionError('The original compute budget was exceeded')
        result['contracts'][contract] = {'arms': arms,
            'same_tied_experiments': arms['remove']['tied_experiments'] == arms['keep']['tied_experiments'],
            'unique_experiment_available': any(row['selected_experiment_id'] for row in arms.values()),
            'new_parent_prediction_or_adoption': False}
    result['conclusion'] = {
        'original_step1_does_not_match_scored_step0': True,
        'aligned_native_contract_still_tied': all(arm['status'] == 'tied'
            for arm in result['contracts']['aligned_step0_native']['arms'].values()),
        'unique_experiment_was_not_lost': not any(item['unique_experiment_available']
            for item in result['contracts'].values()),
        'scientific_or_parent_adoption_gain_established': False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=RECEIPT)
    parser.add_argument('--pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    replay = replay_selection_contract(arguments.receipt, arguments.pointer, arguments.output)
    print(json.dumps({'output': str(arguments.output), 'conclusion': replay['conclusion'],
        'contracts': {name: {arm: {key: row[key] for key in ('status', 'signature_count',
            'effective_pairs', 'tied_experiments', 'compute')} for arm, row in item['arms'].items()}
            for name, item in replay['contracts'].items()}}, ensure_ascii=False, indent=2))

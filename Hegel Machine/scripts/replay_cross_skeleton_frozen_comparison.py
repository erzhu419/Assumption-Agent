#!/usr/bin/env python3
"""Compare frozen candidates on the retained, partial post-freeze evidence.

No tied action is selected. The two real saved rows belong to an already paid
representative trajectory, not to the tied initial actions. Four stored final
evaluation examples remain downstream of the unchanged native adoption rule.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
import tempfile

from hegel_machine.cognition.candidate_comparison import compare_candidate_adoption
from hegel_machine.cognition.evaluation import point_loss, predict
from hegel_machine.cognition.models import Candidate, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.selection import rank_candidates
from audit_cross_skeleton_consumption import reconstruct_inputs, RECEIPT
from replay_cross_skeleton_relations import (
    POINTER, ROOT, TOTAL_COMPUTE, ReplayLab, declaration, new_agent,
)
from replay_cross_skeleton_selection_contract import _inputs
from retained_cross_skeleton_evidence import recover_t07_evidence


OUTPUT = ROOT / 'docs/evidence/cross_skeleton_frozen_comparison_20261021.json'


def _assessment(row):
    return {'program_id': row.candidate.candidate_id, 'score': row.score,
        'prefix_mse': row.evidence_loss, 'reference_mse': row.reference_loss,
        'fresh_mse': row.fresh_loss, 'size': row.candidate.program.size}


def replay_frozen_comparison(receipt_path=RECEIPT, pointer=POINTER, output=OUTPUT):
    data = reconstruct_inputs(receipt_path, pointer)
    original, payload, source = data['receipt'], data['payload'], data['source']
    programs, baseline = data['programs'], data['baseline']
    # Every candidate, fit and admission comes from the pre-existing receipt.
    # No later label may add a candidate, alter parameters or recover a slot.
    admitted = {arm: original['arms'][arm]['admitted_program_ids'] for arm in ('remove', 'keep')}
    union = sorted(set().union(*map(set, admitted.values())))
    incumbent_id = next(identifier for identifier in union
                        if programs[identifier] == baseline['incumbent'])
    if not all(incumbent_id in identifiers for identifiers in admitted.values()):
        raise AssertionError('The original finite shortlist must retain its incumbent')
    config = ResearchConfig(**payload['config'])
    spec = TaskSpec('retained_t07_frozen_comparison', 'Original numeric prediction contract',
                    ('x', 'z'), (), (), risk_quantile=.5)
    evidence = recover_t07_evidence(payload, source)
    if evidence['training_rows'] != baseline['prefix']:
        raise AssertionError('Retained recovery changed the original fitting material')
    training, fresh, later = (evidence[key] for key in
                             ('training_rows', 'comparison_rows', 'later_rows'))
    # Save predictions for every frozen admitted Program before invoking the
    # label-dependent comparison. Rows and bindings are never fitted here.
    forecasts = {identifier: {
        'comparison': [predict(programs[identifier], row, baseline['fallback']) for row in fresh],
        'later': [predict(programs[identifier], row, baseline['fallback']) for row in later]}
        for identifier in union}
    mission = {'purpose': 'theory_learning', 'mission_id': 'retained_frozen_comparison',
        'theory_learning_allocation': {'compute': TOTAL_COMPUTE, 'author_calls': 0, 'observations': 0}}
    arms = {}
    for arm in ('remove', 'keep'):
        saved = original['arms'][arm]
        identifiers = admitted[arm]
        candidates = [Candidate(identifier, programs[identifier], data['scores'][identifier]['operation'])
                      for identifier in identifiers]
        incumbent = next(candidate for candidate in candidates if candidate.candidate_id == incumbent_id)
        with tempfile.TemporaryDirectory(prefix=f'hegel-frozen-comparison-{arm}-') as temporary:
            agent, laboratory = new_agent(temporary), ReplayLab()
            available = saved['prefix_scores']['rank_order']
            register_theories(agent, [declaration(identifier, programs[identifier]) for identifier in available])
            question, panel = _inputs(payload['episodes'][6], identifiers, 'aligned_step0_native',
                                      baseline['fallback'], source)
            # The full prior ledger includes already executed checks. Retain
            # those debits while replacing just the route executed below.
            base = saved['compute']['total'] - saved['compute']['common_route']
            agent.costs['compute_cost'] = base
            routed = agent._execute('run', {'operator': 'research_route',
                'research_question': question, 'experiments': panel, 'mission': mission}, laboratory)
            route_compute = agent.costs['compute_cost'] - base
            if (laboratory.execute_calls or agent.costs['model_calls']
                    or agent.costs['observation_cost'] or routed['status'] != 'tied'):
                raise AssertionError('The retained tie was changed or external data were acquired')
            if any('pending_query' in item for item in agent.questions):
                raise AssertionError('A tied experiment was forced into a pending plan')
            budget = SearchBudget(TOTAL_COMPUTE - agent.costs['compute_cost'])
            assessed = rank_candidates(candidates, training, fresh, fresh=fresh, fresh_weight=.5,
                fallback=baseline['fallback'], q=spec.risk_quantile, budget=budget,
                complexity_weight=config.complexity_weight)
            if len(assessed) != len(candidates):
                raise AssertionError('The original budget cannot compare every frozen admission')
            comparison = compare_candidate_adoption(assessed, incumbent, training, [], fresh,
                fallback=baseline['fallback'], spec=spec, config=config)
            replayed = comparison.best if comparison.adopt else comparison.current
            replayed_id = replayed.candidate.candidate_id
            before_predictions = forecasts[incumbent_id]['later']
            after_predictions = forecasts[replayed_id]['later']
            # Native support uses two predictions per fresh row in this
            # ordinary numeric protocol; account for this actual work as well
            # as the explicitly retained forecast table diagnostics.
            support_predictions = 2 * len(fresh)
            forecast_predictions = len(candidates) * (len(fresh) + len(later))
            total = agent.costs['compute_cost'] + budget.spent + support_predictions + forecast_predictions
            if total > TOTAL_COMPUTE:
                raise AssertionError('The full diagnostic work exceeds the original allocation')
            selection = routed['selection']
            arms[arm] = {
                'candidate_count': len(available), 'frozen_admitted_program_ids': identifiers,
                'selection': {'status': routed['status'], 'tied_experiments': selection['tied_experiments'],
                    'selected_experiment': selection['selected_experiment'],
                    'tie_diagnostics': selection['tie_diagnostics'],
                    'pending_query_prepared': False, 'support_updated': selection['support_updated']},
                'comparison': {'scope': 'partial_retained_postfreeze_rows_only',
                    'complete_original_comparison_replayed': False,
                    'observed_action_index': 65, 'tied_experiment_executed': False,
                    'ranking': [_assessment(row) for row in assessed],
                    'best': _assessment(comparison.best), 'current': _assessment(comparison.current),
                    'native_gate_adopt_on_available_rows': comparison.adopt,
                    'support': comparison.support, 'margin': comparison.margin,
                    'fresh_gain': comparison.fresh_gain, 'evidence_gain': comparison.evidence_gain,
                    'counterexample_gain': comparison.counterexample_gain,
                    'support_unit': comparison.support_unit},
                'frozen_source_parent': {'program_id': incumbent_id, 'changed_by_this_audit': False},
                'partial_replay_parent': {'before_program_id': incumbent_id, 'after_program_id': replayed_id,
                    'changed': replayed_id != incumbent_id, 'actual_deployment_or_source_revision': False},
                'later_excerpt': {'scope': 'four_retained_evaluation_examples_only',
                    'before_forecast_ref': f'forecast_table.{incumbent_id}.later',
                    'after_forecast_ref': f'forecast_table.{replayed_id}.later',
                    'before_mse': mean(point_loss(value, row.target, .5)
                        for value, row in zip(before_predictions, later)),
                    'after_mse': mean(point_loss(value, row.target, .5)
                        for value, row in zip(after_predictions, later)),
                    'predictions_changed': before_predictions != after_predictions,
                    'used_for_candidate_admission_or_adoption': False,
                    'independent_new_confirmation': False},
                'compute': {'retained_original_full_protocol': saved['compute']['total'],
                    'route_replacement': route_compute - saved['compute']['common_route'],
                    'new_native_ranking': budget.spent, 'native_support_predictions': support_predictions,
                    'frozen_forecast_diagnostics': forecast_predictions,
                    'total_including_diagnostics': total, 'remaining': TOTAL_COMPUTE - total},
                'new_paid_observations': 0, 'new_provider_calls': 0}
    receipt = {'status': 'completed_partial_historical_comparison',
        'source_receipt': str(receipt_path), 'source_archive': source,
        'protocol': {'freeze_action_index': 44, 'generation_and_fitting_changed': False,
            'admission_changed': False, 'baseline_and_rules_changed': False,
            'native_gate_ref': 'src/hegel_machine/cognition/candidate_comparison.py:compare_candidate_adoption',
            'controller_ref': 'src/hegel_machine/cognition/controller.py',
            'selection_policy': config.selection_policy, 'complexity_weight': config.complexity_weight,
            'minimum_improvement': config.minimum_improvement, 'fresh_weight': .5,
            'same_compute_allocation': TOTAL_COMPUTE, 'observation_allowance': 8,
            'comparison_split': 'all unique retained post-action44 paid rows, chronological: action65 steps0/1',
            'later_split': 'all four stored final evaluation examples in writer order',
            'comparison_is_new_or_blind': False, 'forced_tie_break': False,
            'candidate_fit_or_refit_calls': 0, 'new_paid_observations': 0, 'new_provider_calls': 0},
        'retained_evidence': {key: deepcopy(evidence[key]) for key in (
            'training_refs', 'comparison_refs', 'later_refs', 'coverage', 'duplicates',
            'historical_costs', 'later_history_order_contract', 'retained_summary_refs', 'scope')},
        'original_run_outcome': {
            'comparison_summary_ref': deepcopy(evidence['retained_summary_refs']['comparison']),
            'comparison_summary': {key: deepcopy(payload['episodes'][6]['actions'][68]['details'][key])
                for key in ('candidates_scored', 'best_loss', 'fresh_gain', 'dispersion_margin',
                            'adopt', 'selection_policy', 'validation_is_adaptive_development')},
            'revision_ref': deepcopy(evidence['retained_summary_refs']['final_revision']),
            'prediction_check_ref': deepcopy(evidence['retained_summary_refs']['prediction_check']),
            'prediction_check_summary': deepcopy(payload['episodes'][6]['actions'][70]['details']),
            'scope': 'original run result only; not a replacement for missing per-row ablation evidence',
            'final_revision_used_as_new_candidate': False},
        'forecast_table': forecasts, 'arms': arms,
        'diagnostic_work': {'shared_constructor_reconstruction_compute': sum(baseline['costs'].values()),
            'unique_frozen_programs': len(union),
            'actual_shared_forecast_row_predictions': len(union) * (len(fresh) + len(later)),
            'forecast_storage': 'one table, referenced by both arms; original sources not restored'},
        'conclusion': {'same_partial_replay_parent': arms['remove']['partial_replay_parent']['after_program_id']
            == arms['keep']['partial_replay_parent']['after_program_id'],
            'same_later_excerpt_predictions': forecasts[arms['remove']['partial_replay_parent']['after_program_id']]['later']
            == forecasts[arms['keep']['partial_replay_parent']['after_program_id']]['later'],
            'complete_same_material_ablation_status': 'unresolved_missing14of16postfreeze_paid_rows',
            'scientific_independent_gain_established': False,
            'new_selected_experiment_executed': False}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=RECEIPT)
    parser.add_argument('--pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = replay_frozen_comparison(arguments.receipt, arguments.pointer, arguments.output)
    print(json.dumps({'output': str(arguments.output), 'conclusion': result['conclusion'],
        'arms': {name: {key: row[key] for key in ('partial_replay_parent', 'later_excerpt', 'compute')}
                 for name, row in result['arms'].items()}}, ensure_ascii=False, indent=2))

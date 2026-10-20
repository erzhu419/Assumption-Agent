#!/usr/bin/env python3
"""Diagnose the retained relation replay without learning or acquiring data.

Reconstruct the old constructors and the saved relation bindings.  The original
candidate pool, budgets, ranking and held-out prediction remain unchanged.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean, median, quantiles

from hegel_machine.cognition.evaluation import point_loss, predict
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation
from replay_cross_skeleton_relations import HISTORICAL_OUTPUT as RECEIPT, POINTER, ROOT, old_baseline, load_source


OUTPUT = ROOT / 'docs/evidence/cross_skeleton_consumption_20261021.json'


def reconstruct_inputs(receipt_path=RECEIPT, pointer=POINTER):
    """Restore Programs from existing sources/bindings, without new proposals."""
    receipt = json.loads(Path(receipt_path).read_text())
    payload, source = load_source(Path(pointer))
    baseline = old_baseline(payload, source)
    programs = {f'program_{i:04d}': entry['candidate'].program
                for i, entry in enumerate(baseline['candidates'])}
    scores = {row[0]: dict(zip(receipt['prefix_score_table']['columns'], row))
              for row in receipt['prefix_score_table']['rows']}
    definition = Definition.from_dict(receipt['frozen_relation']['definition'])
    derivations = {row['program_id']: row for row in receipt['generated_program_derivations']['rows']}
    for identifier, row in derivations.items():
        children = tuple(programs[f'program_{int(row["components"][hole].split(":")[1]):04d}']
                         for hole in definition.arguments)
        programs[identifier] = Program('call', name=definition.definition_id,
            definition=definition, children=children).expanded()
    if set(programs) != set(scores):
        raise AssertionError('Saved scores and reconstructed Programs do not identify the same pool')
    return {'receipt': receipt, 'payload': payload, 'source': source, 'baseline': baseline,
            'programs': programs, 'scores': scores, 'derivations': derivations}


def _distribution(values):
    quartiles = quantiles(values, n=4, method='inclusive')
    return {'minimum': min(values), 'q25': quartiles[0], 'median': median(values),
            'q75': quartiles[2], 'maximum': max(values)}


def _tie_ids(plan):
    # This is exactly the native selector's final max-G, then cheapest rule.
    best = max(row['G'] for row in plan['scores'])
    options = [row for row in plan['scores'] if row['G'] == best]
    cost = min(row['declared_cost'] for row in options)
    return [row['experiment_id'] for row in options if row['declared_cost'] == cost]


def audit_consumption(receipt_path=RECEIPT, pointer=POINTER, output=OUTPUT):
    data = reconstruct_inputs(receipt_path, pointer)
    receipt, payload, baseline = data['receipt'], data['payload'], data['baseline']
    programs, scores, derivations = data['programs'], data['scores'], data['derivations']
    prefix, fallback = baseline['prefix'], baseline['fallback']
    old_ids = receipt['arms']['remove']['prefix_scores']['rank_order']
    all_ids = receipt['arms']['keep']['prefix_scores']['rank_order']
    new_ids = [identifier for identifier in all_ids if identifier not in set(old_ids)]
    all_new_scored = set(derivations) <= set(scores) and set(derivations) <= set(all_ids)
    readout_step = receipt['protocol']['public_panel_readout_step']
    inherited_mature_readout = all(programs[identifier].op == 'if'
        and programs[identifier].children[0].op == 'available'
        and programs[identifier].children[0].lag <= readout_step
        and programs[identifier].children[1] == programs[
            f'program_{int(row["components"]["hole_0"].split(":")[1]):04d}'].expanded()
        for identifier, row in derivations.items())
    # Native predictions are computed once and reused for startup/mature/MSE
    # summaries; no target continuation label participates in their ordering.
    predictions = {identifier: [predict(programs[identifier], row, fallback) for row in prefix]
                   for identifier in all_ids}
    recalculated = {identifier: mean(point_loss(value, row.target, .5)
        for value, row in zip(values, prefix)) for identifier, values in predictions.items()}
    mse_differences = [abs(recalculated[identifier] - scores[identifier]['prefix_mse']) for identifier in all_ids]
    size_differences = [abs(programs[identifier].size - round(
        (scores[identifier]['score'] - scores[identifier]['prefix_mse']) / .001)) for identifier in all_ids]
    if max(mse_differences) > 1e-12 or max(size_differences):
        raise AssertionError('The saved pool cannot be reconstructed with its original native scores')
    by_score = lambda identifier: (scores[identifier]['score'], programs[identifier].size)
    best_old, best_new = min(old_ids, key=by_score), min(new_ids, key=by_score)
    accurate_old = min(old_ids, key=lambda identifier: scores[identifier]['prefix_mse'])
    accurate_new = min(new_ids, key=lambda identifier: scores[identifier]['prefix_mse'])
    incumbent_id = next(identifier for identifier in old_ids if programs[identifier] == baseline['incumbent'])
    minimum_improvement = payload['config']['minimum_improvement']
    incumbent_score_gain = scores[incumbent_id]['score'] - scores[best_old]['score']
    incumbent_evidence_change = scores[best_old]['prefix_mse'] - scores[incumbent_id]['prefix_mse']
    score_margin_passed = scores[best_old]['score'] + minimum_improvement < scores[incumbent_id]['score']
    # The archived binding is chosen by its source components, not its later
    # forecast: frozen incumbent + the pre-existing startup-capable Program.
    inherited_startup = next(identifier for identifier in new_ids
        if derivations[identifier]['components'] == {'hole_0': 'old:0017', 'hole_1': 'old:0016'})
    prefix_positions = {'startup': [i for i, row in enumerate(prefix) if not row.history],
                        'mature': [i for i, row in enumerate(prefix) if row.history]}
    admitted = {arm: receipt['arms'][arm]['admitted_program_ids'] for arm in ('remove', 'keep')}
    new_admitted = [identifier for identifier in admitted['keep'] if identifier in new_ids]
    displaced = [identifier for identifier in admitted['remove'] if identifier not in admitted['keep']]
    highlight_ids = list(dict.fromkeys([best_old, best_new, accurate_old, accurate_new, inherited_startup,
        *new_admitted, *displaced]))
    # Read the already paid continuation target only after fixing every rank,
    # subset and illustration above. This stage is a diagnostic counterfactual.
    target = payload['episodes'][6]
    future = Observation(deepcopy(target['actions'][65]['details']['sequence'][0]), 0., ())
    future_predictions = {identifier: predict(programs[identifier], future, fallback) for identifier in all_ids}
    actual = target['memory_events'][0]['observation']['history'][0]['__target__']
    ranking_prediction = future_predictions[best_old]
    summaries = {}
    for name, identifiers in (('old', old_ids), ('new', new_ids)):
        summaries[name] = {'count': len(identifiers),
            'prefix_mse': _distribution([scores[identifier]['prefix_mse'] for identifier in identifiers]),
            'program_size': _distribution([programs[identifier].size for identifier in identifiers]),
            'penalized_score': _distribution([scores[identifier]['score'] for identifier in identifiers]),
            'prefix_mse_below_ranking_winner': sum(scores[identifier]['prefix_mse']
                < scores[best_old]['prefix_mse'] - 1e-12 for identifier in identifiers),
            'prefix_mse_at_most_1e_minus_12': sum(scores[identifier]['prefix_mse'] <= 1e-12 for identifier in identifiers),
            'both_saved_startup_rows_within_1e_minus_6': sum(all(abs(predictions[identifier][i] - prefix[i].target)
                <= 1e-6 for i in prefix_positions['startup']) for identifier in identifiers),
            'posthoc_continuation_predictions_different_from_ranking_winner': sum(abs(future_predictions[identifier]
                - ranking_prediction) > 1e-12 for identifier in identifiers),
            'posthoc_continuation_within_1e_minus_6': sum(abs(future_predictions[identifier] - actual) <= 1e-6
                                                       for identifier in identifiers)}
    rows = []
    for identifier in highlight_ids:
        program = programs[identifier]
        row = {'program_id': identifier, 'origin': 'new_relation_binding' if identifier in new_ids else 'old_baseline',
            'operation': scores[identifier]['operation'], 'size': program.size,
            'minimum_history_depth': program.minimum_history_depth,
            'prefix_mse': scores[identifier]['prefix_mse'], 'score': scores[identifier]['score'],
            'complexity_penalty': .001 * program.size,
            'remove_rank': old_ids.index(identifier) + 1 if identifier in old_ids else None,
            'keep_rank': all_ids.index(identifier) + 1,
            'admitted_in_remove': identifier in admitted['remove'], 'admitted_in_keep': identifier in admitted['keep'],
            'startup_predictions': [predictions[identifier][i] for i in prefix_positions['startup']],
            'startup_mse': mean(point_loss(predictions[identifier][i], prefix[i].target, .5)
                                for i in prefix_positions['startup']),
            'mature_mse': mean(point_loss(predictions[identifier][i], prefix[i].target, .5)
                               for i in prefix_positions['mature']),
            'posthoc_continuation_prediction': future_predictions[identifier],
            'posthoc_continuation_squared_error': point_loss(future_predictions[identifier], actual, .5)}
        if identifier in derivations:
            row['binding_ref'] = {'file': str(receipt_path), 'field':
                f'generated_program_derivations.rows[program_id={identifier}]'}
            row['components'] = derivations[identifier]['components']
        else:
            index = int(identifier.split('_')[1])
            row['old_constructor_candidate_id'] = baseline['candidates'][index]['candidate'].candidate_id
            if index < len(baseline['archive']):
                row['source_ref'] = baseline['archive'][index]['source_ref']
        if identifier in {best_old, best_new, accurate_old, accurate_new, inherited_startup}:
            row['expression'] = program.describe()
        rows.append(row)
    mature_counts = Counter(row['components']['hole_0'] for row in derivations.values())
    failed_binding = receipt['counterexample']['actual_transfer']['generated_theories'][0]['binding']['components']
    original_counts = mature_counts.copy()
    original_counts[failed_binding['hole_0']['component_id']] += 1
    result = {'status': 'completed', 'source_replay_receipt': str(receipt_path),
        'source_archive': receipt['source'], 'protocol': {
            'original_old_candidates': len(old_ids), 'original_new_candidates': len(new_ids),
            'original_relation_application_allocation': receipt['protocol']['relation_application_allocation'],
            'original_total_compute_budget': receipt['protocol']['same_total_compute_budget'],
            'new_learning_runs': 0, 'new_relation_application_runs': 0, 'new_paid_observations': 0, 'provider_calls': 0,
            'continuation_labels_used_for_reconstruction_ranking_or_highlight_selection': False,
            'original_replay_and_receipt_modified': False},
        'reconstruction': {'program_count': len(programs), 'prefix_row_count': len(prefix),
            'maximum_native_prefix_mse_difference': max(mse_differences),
            'maximum_program_size_difference': max(size_differences),
            'all_new_candidates_scored': all_new_scored},
        'candidate_distributions': summaries,
        'parent_objective': {'scope': 'ranking_only_not_controller_adoption',
            'formula': 'prefix_MSE + 0.001 * expanded_Program_size',
            'ranking_winner_in_both_arms': best_old, 'best_new_by_same_score': best_new,
            'old_best_by_prefix_mse': accurate_old, 'new_best_by_prefix_mse': accurate_new,
            'source_bound_startup_example': inherited_startup,
            'old_accurate_candidate_lambda_crossover': (scores[best_old]['prefix_mse'] - scores[accurate_old]['prefix_mse'])
                / (programs[accurate_old].size - programs[best_old].size),
            'crossover_is_prefix_only_diagnostic_not_a_parameter_change': True},
        'parent_adoption': {'scope': 'existing_controller_gate_diagnostic_no_new_adoption_executed',
            'original_replay_parent_was_only_ranking_winner': True,
            'actual_frozen_incumbent_program_id': incumbent_id,
            'actual_frozen_incumbent_ref': {**{key: receipt['source'][key] for key in ('archive', 'unit', 'member')},
                'field': 'episodes[6].actions[44].details.after_program'},
            'ranking_winner_program_id': best_old,
            'ranking_winner_score_gain_over_incumbent': incumbent_score_gain,
            'ranking_winner_evidence_loss_change_over_incumbent': incumbent_evidence_change,
            'minimum_improvement': minimum_improvement,
            'score_margin_passed': score_margin_passed,
            'fresh_comparison_rows_after_freeze': 0,
            'observed_counterexample_gain_over_incumbent': False,
            'gate_status': 'deferred_no_fresh_comparison',
            'scientific_adoption_established': False,
            'effective_parent_remains_program_id': incumbent_id,
            'posthoc_retained_parent_continuation_prediction': future_predictions[incumbent_id],
            'posthoc_retained_parent_continuation_squared_error': point_loss(future_predictions[incumbent_id], actual, .5),
            'held_label_used_for_gate': False,
            'native_gate_ref': {'file': 'src/hegel_machine/cognition/controller.py',
                'symbols': ['if not fresh', 'counterexample_gain', 'adopt'],
                'rule': 'A funded fresh comparison precedes score-margin or supported-counterexample adoption.'}},
        'admission': {'maximum': 16, 'new_admitted_program_ids': new_admitted,
            'displaced_old_program_ids': displaced, 'old_still_admitted_program_ids':
                [identifier for identifier in admitted['keep'] if identifier in old_ids],
            'all_raw_old_programs_remain_available': receipt['arms']['keep']['all_old_candidates_retained']},
        'highlighted_candidates': rows,
        'retained_prefix': {'startup_positions': prefix_positions['startup'],
            'startup_targets': [prefix[i].target for i in prefix_positions['startup']],
            'mature_positions': prefix_positions['mature'],
            'source_ref': {'file': str(receipt_path), 'field': 'protocol.prefix_sources'}},
        'held_diagnostic': {'action_index': 65, 'step': 0, 'features': future.features, 'actual': actual,
            'label_ref': {**{key: receipt['source'][key] for key in ('archive', 'unit', 'member')},
                'field': 'episodes[6].memory_events[0].observation.history[0].__target__'},
            'used_to_change_candidates_ranking_or_budget': False, 'independent_confirmation': False},
        'bounded_enumeration': {'generated_before_real_countercheck': receipt['mainline_application']['generated_count'],
            'one_numeric_binding_refuted': 1, 'remaining_new_bindings': len(new_ids),
            'generated_mature_component_counts_before_countercheck': dict(original_counts),
            'remaining_mature_component_counts': dict(mature_counts),
            'old_components_with_minimum_history_one': [f'old:{i:04d}' for i, entry in enumerate(baseline['candidates'])
                                                       if entry['candidate'].program.minimum_history_depth == 1],
            'allocation_exhausted': receipt['mainline_application']['budget_exhausted'],
            'actual_application_compute': receipt['mainline_application']['compute'],
            'complete_cross_product_not_claimed': True},
        'experiment_contract': {'readout_step': readout_step,
            'startup_hole_executes_at_original_panel_readouts': not inherited_mature_readout,
            'new_mature_values_are_existing_old_component_values': inherited_mature_readout,
            'remove_signature_groups': receipt['arms']['remove']['plan']['signature_count'],
            'keep_signature_groups': receipt['arms']['keep']['plan']['signature_count'],
            'native_tie_rule': 'max G, then lowest declared cost, retain exact ties',
            'remove_tied_ids_recovered_from_saved_scores': _tie_ids(receipt['arms']['remove']['plan']),
            'keep_tied_ids_recovered_from_saved_scores': _tie_ids(receipt['arms']['keep']['plan']),
            'both_selected_action_ids': [receipt['arms'][arm]['plan']['selected_experiment_id'] for arm in ('remove', 'keep')],
            'plan_only_no_action_outcome_measured': True},
        'diagnosis': {'candidate_generation_has_executable_effect': bool(new_ids and new_admitted),
            'candidate_scoring_or_admission_disconnected': not all_new_scored,
            'saved_native_mse_or_size_arithmetic_error_observed': False,
            'unchanged_ranking_winner_due_to_existing_penalized_ranking': all(receipt['arms'][arm]['parent']['program_id'] == best_old
                for arm in ('remove', 'keep')) and scores[best_old]['score'] < scores[best_new]['score'],
            'old_replay_did_not_execute_the_native_parent_adoption_gate': True,
            'original_step_one_question_does_not_probe_the_added_startup_branch': inherited_mature_readout,
            'accurate_startup_forecasts_already_available_in_old_baseline': summaries['old']['both_saved_startup_rows_within_1e_minus_6'] > 0,
            'new_object_independent_prediction_gain_established': False},
        'diagnostic_compute': {'old_constructor_reconstruction_compute': sum(baseline['costs'].values()),
            'saved_binding_reconstructions': len(derivations),
            'saved_binding_expanded_nodes': sum(programs[identifier].size for identifier in new_ids),
            'native_prefix_row_predictions': len(programs) * len(prefix),
            'posthoc_continuation_row_predictions': len(programs),
            'original_arm_costs_unchanged': True},
        'limitations': ['The stored target startup labels are development data; the one retained continuation label is used only for posthoc diagnostics.',
            'All 384 saved proposals are reconstructed, but the original 20k application exhausted before the full finite product was enumerated.',
            'The original reported parent was a ranking counterfactual. Without a post-freeze fresh comparison the real action44 incumbent remains deployed.',
            'This audit identifies the recorded ranking and query-contract mechanisms; it does not establish a new scientific model or an independent forecasting gain.']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pointer', type=Path, default=POINTER)
    parser.add_argument('--replay-receipt', type=Path, default=RECEIPT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit_consumption(args.replay_receipt, args.source_pointer, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output),
        'reconstruction': result['reconstruction'], 'diagnosis': result['diagnosis']}, ensure_ascii=False))

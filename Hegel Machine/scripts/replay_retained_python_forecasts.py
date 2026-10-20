#!/usr/bin/env python3
"""Check saved Python forecasts on 48 real successor rows without fitting.

Only the original finite forecast-column adapter is executed. Its feature
Programs project saved numbers; the Python scientific models stay Python.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from hegel_machine.cognition.inquiry_revision import complete_revision
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget
from hegel_machine.cognition.probing import choose_probe


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/erzhu419/hegel-active-experiments/luna_window_parameter_transfer_20261017/frozen/target_000')
OUTPUT = ROOT / 'docs/evidence/retained_python_forecasts_20261021.json'


def reference(file, *path):
    return {'file': str(file), 'json_path': list(path)}


def frozen_selection(comparison, allocation):
    """The same saved-column projection and chooser as compare_experiments."""
    eligible = comparison['eligible_models']
    candidates = [Candidate(model, Program('feature', name=f'forecast_{index}'), 'authored_forecast')
                  for index, model in enumerate(eligible)]
    budget, scores = SearchBudget(allocation), []
    for experiment in comparison['comparisons']:
        forecasts = {item['model']: item for item in experiment['forecasts'] if item['model'] in eligible}
        if set(forecasts) != set(eligible) or any(
                item.get('returncode') != 0 or len(item['predictions']) != experiment['row_count']
                for item in forecasts.values()):
            raise ValueError('The original candidate panel forecasts are incomplete')
        probe = Probe(experiment['id'], tuple({f'forecast_{index}': forecasts[model]['predictions'][row]
            for index, model in enumerate(eligible)} for row in range(experiment['row_count'])))
        selected, separation = choose_probe([probe], candidates, 0., budget)
        if selected is None:
            raise ValueError('The fixed original allocation cannot evaluate the complete panel')
        scores.append({'id': experiment['id'], 'predicted_separation': separation,
            'saved_predicted_separation': experiment['predicted_separation'],
            'observation_cost': experiment['observation_cost']})
    best = max(item['predicted_separation'] for item in scores)
    tied = [item['id'] for item in scores if item['predicted_separation'] == best]
    return {'status': 'tied' if len(tied) != 1 else 'selected', 'scores': scores,
        'selected_experiment_id': tied[0] if len(tied) == 1 else None,
        'tied_experiments': tied if len(tied) != 1 else [], 'compute_cost': budget.spent,
        'new_predictor_executions': 0, 'labels_used': False}


def replay(source=SOURCE, output=OUTPUT):
    source = Path(source)
    state_file, observations_file = source / 'research_state.json', source / 'laboratory_observations.json'
    state = json.loads(state_file.read_text())
    comparison, revision = state['experiment_comparisons'][0], state['revisions'][0]
    if comparison['evidence_cutoff'] != revision['evidence_cutoff'] or revision['record_indices'] != [24, 25]:
        raise ValueError('The original comparison and successor cutoffs changed')
    if comparison['evidence_cutoff'] != {'records': 24, 'rows': 72}:
        raise ValueError('This replay requires the retained 24-record, 72-row development prefix')
    selection = frozen_selection(comparison, state['limits']['compute_budget'])
    if any(item['predicted_separation'] != item['saved_predicted_separation'] for item in selection['scores']):
        raise ValueError('The original chooser does not reproduce its saved numerical panel scores')
    # Public candidates/panel and selection are frozen before opening labels.
    observations = json.loads(observations_file.read_text())
    records = [observations[index] for index in revision['record_indices']]
    if [record['action'] for record in records] != revision['actions']:
        raise ValueError('Saved successor actions do not match the frozen experiment')
    if ([len(record['rows']) for record in records] != revision['action_row_counts']
            or revision['action_row_counts'] != [24, 24]):
        raise ValueError('The original two 24-step successor trajectories are incomplete')
    if revision['proposal'].get('readout'):
        raise ValueError('This retained revision has no authored executable readout')
    paired_rows, summaries = [], []
    for forecast_index, forecast in enumerate(revision['forecasts']):
        original = revision['proposal']['original_models'][forecast['model']]
        panel_forecast = next(item for item in comparison['forecasts'] if item['model'] == original)
        if (panel_forecast['predictions'][:48] != forecast['predictions']
                or panel_forecast['prediction_inputs'][:48] != forecast['prediction_inputs']):
            raise ValueError('A revision forecast differs from its preselection frozen values or inputs')
        offset, errors, parts = 0, [], []
        for record_index, record in zip(revision['record_indices'], records):
            segment = []
            for step, row in enumerate(record['rows']):
                frozen_input = forecast['prediction_inputs'][offset + step]
                if frozen_input != {'features': row['features'], 'history': row.get('history', [])}:
                    raise ValueError(f'Frozen input differs from actual record {record_index}, step {step}')
                if forecast_index == 0:
                    paired_rows.append([record_index, step, record['row_indices'][step], offset + step])
                segment.append((forecast['predictions'][offset + step] - row['target']) ** 2)
            errors.extend(segment)
            parts.append({'record_index': record_index, 'prediction_offset': offset,
                'rows': len(segment), 'mse': sum(segment) / len(segment),
                'observation_ref': reference(observations_file, record_index)})
            offset += len(segment)
        if len(errors) != len(forecast['predictions']) or len(errors) != 48:
            raise ValueError('The saved frozen forecast is not complete for all successor rows')
        summaries.append({'model': forecast['model'], 'original_model': original,
            'model_ref': {'file': str(source / forecast['model'])},
            'forecast_ref': reference(state_file, 'revisions', 0, 'forecasts', forecast_index),
            'comparison_forecast_ref': reference(state_file, 'experiment_comparisons', 0, 'forecasts',
                comparison['forecasts'].index(panel_forecast)),
            'rows': len(errors), 'mse': sum(errors) / len(errors),
            'saved_mse': forecast['prospective_mse'], 'per_record': parts})
    # Execute the existing post-acquisition scorer on the original saved inputs.
    # No readout was declared, so this invokes no authored code or interpreter.
    agent = SimpleNamespace(_prepare_program_state=lambda: None)
    laboratory = SimpleNamespace(observation_rows=lambda record: [Observation(
        row['features'], row['target'], tuple(row.get('history', []))) for row in record['rows']])
    completed = complete_revision(agent, deepcopy(revision), records, revision['record_indices'], laboratory)
    for saved, current in zip(summaries, completed['forecasts']):
        if saved['mse'] != current['prospective_mse']:
            raise ValueError('Independent row arithmetic differs from the native completion routine')
        saved['complete_revision_mse'] = current['prospective_mse']
        saved['saved_mse_reproduced'] = saved['mse'] == saved['saved_mse']
        saved['scoped_status'] = 'refuted' if saved['mse'] > comparison['mse_threshold'] else 'supported'
    original_selected = comparison['selected_experiment']['id']
    outcomes = [{'id': item['id'], 'status': 'observed' if item['id'] == original_selected else 'UNKNOWN',
        'record_indices': list(revision['record_indices']) if item['id'] == original_selected else [],
        'reason': 'saved_paid_records' if item['id'] == original_selected else 'panel_not_executed_no_labels'}
        for item in selection['scores']]
    costs = {'forecast_column_selection': selection['compute_cost'],
        'independent_successor_error_terms': sum(item['rows'] for item in summaries),
        'native_complete_revision_error_terms': sum(item['rows'] for item in summaries)}
    receipt = {'status': 'completed', 'scope': 'complete_saved_python_forecast_historical_successor_check',
        'source_refs': {'state': reference(state_file), 'paid_records': reference(observations_file),
            'comparison': reference(state_file, 'experiment_comparisons', 0),
            'revision': reference(state_file, 'revisions', 0)},
        'protocol': {'evidence_cutoff': deepcopy(revision['evidence_cutoff']),
            'original_limits': deepcopy(state['limits']), 'mse_threshold': comparison['mse_threshold'],
            'candidate_and_panel_selection_before_successor_labels': True, 'new_fit_runs': 0,
            'new_predictor_executions': 0, 'new_paid_observations': 0, 'new_model_calls': 0,
            'scientific_predictor_representation': 'retained_arbitrary_Python',
            'adapter_representation': 'feature_projection_of_saved_forecast_columns',
            'PR2_route_executed': False, 'PR3_Python_source_induction_established': False},
        'selection': selection, 'historical_selected_experiment_id': original_selected,
        'same_selected_experiment': selection['selected_experiment_id'] == original_selected,
        'panel_outcomes': outcomes, 'forecasts': summaries,
        'row_pairing': {'columns': ['record_index', 'step', 'paid_row_id', 'prediction_index'],
            'rows': paired_rows, 'feature_history_mismatches': 0, 'actual_successor_rows': len(paired_rows)},
        'native_completion': {'routine': 'hegel_machine.cognition.inquiry_revision.complete_revision',
            'status': completed['status'], 'record_indices': completed['record_indices'],
            'readout_declared': False, 'new_independent_confirmation': False},
        'original_account': {'costs': deepcopy(state['costs']),
            'comparison_compute': comparison['compute_cost'], 'paid_trace_rows': sum(len(record['rows']) for record in observations),
            'paid_trace_observation_cost': sum(record['observation_cost'] for record in observations),
            'accounted_observations_outside_paid_trace': state['costs']['observation_cost']
                - sum(record['observation_cost'] for record in observations),
            'configured_confirmation_rows': state['limits']['confirmation_rows'],
            'separate_confirmation_labels_loaded': False,
            'fees_recharged': False}, 'replay_compute': {**costs, 'total': sum(costs.values()),
            'budget': state['limits']['compute_budget'],
            'within_original_compute_cap_including_original_account': sum(costs.values()) + state['costs']['compute_cost']
                <= state['limits']['compute_budget']},
        'failure_preservation': {'all_offered_models': deepcopy(comparison['models']),
            'eligible_models': deepcopy(comparison['eligible_models']),
            'prior_fit_ref': reference(state_file, 'experiment_comparisons', 0, 'prior_fit'),
            'all_frozen_successor_instances_refuted': all(item['scoped_status'] == 'refuted' for item in summaries)},
        'limitations': ['The chooser consumes frozen finite predictions, not executable scientific Program trees or inferred Python structure.',
            'Only the originally purchased panel has labels; alternative-panel outcomes remain UNKNOWN.',
            'This historical replay is not a new independent confirmation or autonomous method benefit.']}
    Path(output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = replay(args.source, args.output)
    print(json.dumps({'status': result['status'], 'rows': result['row_pairing']['actual_successor_rows'],
        'mse': [item['mse'] for item in result['forecasts']], 'replay_compute': result['replay_compute']['total']}, ensure_ascii=False))

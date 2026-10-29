"""Fixed development readout of already fitted candidates; no new research."""

import argparse
from pathlib import Path

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_json_checkpoint
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.parallel_prediction import predict_rows
from hegel_machine.cognition.registered_source_control import _frozen_parent_metrics
from hegel_machine.cognition.temporal_operator_agent import _closed_prediction_values


def read(reference):
    return read_method_reference(reference, Path.cwd())


def paid_predictions(reference):
    assessment = read(reference)
    if 'closed_prediction_state' in assessment:
        return _closed_prediction_values({**reference, 'json_path': [
            *reference.get('json_path', []), 'closed_prediction_state']}, Path.cwd())['predictions']
    result = []
    for part in assessment['result']['paid_prediction_parts']:
        if 'closed_prediction_state_ref' in part:
            result.extend(_closed_prediction_values(part['closed_prediction_state_ref'],
                                                   Path.cwd())['predictions'])
        else:
            result.extend(read(part))
    return result


def run(refs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    timing = ExecutionTiming(('load', 'predict', 'summarize'))
    with timing.phase('load'):
        raw = read(refs['rows_ref'])
        records = read(refs['records_ref'])
        assert len(raw) == 193 and len(records) == 13
        rows = [Observation(row['features'], row['target'], tuple(row.get('history', []))) for row in raw]
        residual = read(refs['residual_ref'])
        first = read(refs['first_repair_program_ref'])
        models = {'residual_32': Program.from_dict(residual['program']),
                  'first_repair': Program.from_dict(first)}
        predictions = {'incumbent_0165': paid_predictions(refs['parent_assessment_ref']),
                       'exported_20': paid_predictions(refs['best_exported_assessment_ref'])}
        assert all(len(values) == len(rows) for values in predictions.values())
        budget = SearchBudget(386)
        contract = {'scope': 'fixed_exposed_development_readout', 'source_refs': refs,
            'new_prediction_names': list(models), 'compute_cap': budget.limit,
            'selection_objective': 'row_weighted_full_193_development_mse',
            'old_48_role': 'exposed_old_region_diagnostic_not_independent_validation',
            'first_repair_role': 'saved_first_revision_not_refitted_or_selected_on_new_readout',
            'new_fits': 0, 'new_observations': 0, 'model_calls': 0,
            'parent_replacement': False, 'independent_confirmation': False}
        write_json_checkpoint(output / 'contract.json', contract)
    receipts = {}
    for name, program in models.items():
        assert budget.spend(len(rows))
        with timing.phase('predict'):
            values = predict_rows(program, rows)
        predictions[name] = values
        receipts[name] = {'file': str(output / (name + '.json')), 'json_path': ['predictions']}
        write_json_checkpoint(output / (name + '.json'), {
            'program_ref': ({**refs['residual_ref'], 'json_path': [
                *refs['residual_ref']['json_path'], 'program']} if name == 'residual_32'
                else refs['first_repair_program_ref']),
            'rows_ref': refs['rows_ref'], 'predictions': values,
            'new_compute': len(rows), 'new_fits': 0})
    with timing.phase('summarize'):
        validation = [i for r in refs['validation_records'] for i in records[r]['row_indices']]
        training = [i for i in range(len(rows)) if i not in validation]
        # Preserve exact recorded trajectories; duplicates are descriptive only.
        groups = []
        for r, record in enumerate(records):
            actual = [raw[i] for i in record['row_indices']]
            same = next((group for group in groups if actual == [raw[i] for i in
                records[group[0]]['row_indices']]), None)
            if same is None:
                groups.append([r])
            else:
                same.append(r)
        distinct = [i for group in groups for i in records[group[0]]['row_indices']]
        metrics = {}
        for name, values in predictions.items():
            metrics[name] = {label: _frozen_parent_metrics(values, rows, indices)
                for label, indices in [('full_193', list(range(len(rows)))),
                    ('fit_145', training), ('old_48', validation),
                    ('distinct_record_rows_descriptive', distinct)]}
            metrics[name]['records'] = [dict(record=r, **_frozen_parent_metrics(
                values, rows, record['row_indices'])) for r, record in enumerate(records)]
        winner = min(metrics, key=lambda name: metrics[name]['full_193']['mse'])
        result = {'status': 'completed', 'contract_ref': {'file': str(output / 'contract.json'),
            'json_path': []}, 'prediction_refs': receipts, 'reused_assessment_refs': {
                'incumbent_0165': refs['parent_assessment_ref'],
                'exported_20': refs['best_exported_assessment_ref']},
            'metrics': metrics, 'identical_record_groups': groups,
            'working_readout_winner': winner, 'parent_replacement': False,
            'independent_confirmation': False,
            'costs': {'compute_cost': budget.spent, 'new_fits': 0,
                'observation_cost': 0, 'measurement_cost': 0, 'model_calls': 0,
                'reported_tokens': 0, 'external_python_execution_seconds': 0.},
            'old_accounts_ref': {'file': str(Path(refs['source']) / 'summary.json'),
                'json_path': ['costs']},
            'local_known_compute_including_prior': 3206597 + budget.spent,
            'prior_unknown_compute_hold': 773448}
    result['timing'] = timing.result()
    write_json_checkpoint(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(read_json_checkpoint(args.refs), args.output)
    print({name: {scope: value['mse'] for scope, value in metrics.items() if scope != 'records'}
           for name, metrics in result['metrics'].items()})

#!/usr/bin/env python3
"""Revalue saved forecasts, without models, refits, observations or LLM calls."""
import argparse
import json
from pathlib import Path
from statistics import pvariance

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.probing import value_forecast_experiments


def analyze(source):
    summary = json.loads((source / 'summary.json').read_text())
    cases = []
    for target in summary['frozen']['revised']:
        workspace = source / 'revised/frozen' / target['case']
        state = json.loads((workspace / 'research_state.json').read_text())
        comparison = state['experiment_comparisons'][0]
        eligible = comparison['eligible_models']
        forecasts = {f['model']: f['predictions'] for f in comparison['forecasts']}
        panels = []
        for experiment in comparison['comparisons']:
            by_model = {f['model']: f['predictions'] for f in experiment['forecasts']}
            panels.append({'id': experiment['id'], 'observation_cost': experiment['observation_cost'],
                           'predictions': [by_model[model] for model in eligible]})
        paid = json.loads((workspace / 'observations.json').read_text())[:comparison['evidence_cutoff']['rows']]
        noise = max(min(f['mse'] for f in comparison['prior_fit'] if f['model'] in eligible),
                    pvariance(r['target'] for r in paid) * 1e-6, 1e-8)
        budget = SearchBudget(1000000)
        value = value_forecast_experiments([forecasts[m] for m in eligible], panels,
            weights=[1. / len(eligible)] * len(eligible), noise_scale=noise, budget=budget)
        values = [p['value_per_observation'] for p in value['comparisons']]
        revisions = target['experimental_design']['revisions']
        cases.append({'case': target['case'], 'eligible_models': eligible, 'noise_scale': noise,
            'original_variances': {p['id']: p['predicted_separation'] for p in comparison['comparisons']},
            'equal_value_at_float_precision': max(values) == min(values),
            'value': value, 'measured_outcomes': [r['measured_outcome'] for r in revisions]})
    return {'source': str(source), 'method': 'existing_finite_forecast_task_risk',
        'reference_design': 'Equal weight on the saved public inputs of both proposed experiments.',
        'scope': 'Post-hoc finite-channel diagnosis, not a new terminal task, acquired experiment, '
            'method revision or independent performance gain. No task labels enter risk valuation. '
            'The supplied point hypotheses were refuted in the original measured experiments.',
        'new_model_calls': 0, 'new_observations': 0, 'model_executions': 0,
        'diagnostic_compute_units': sum(c['value']['compute_cost'] for c in cases),
        'cases': cases, 'equal_value_cases': sum(c['equal_value_at_float_precision'] for c in cases)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('new_model_calls', 'new_observations',
        'model_executions', 'diagnostic_compute_units', 'equal_value_cases')}))

"""Read only public results and distinguish integration from scientific success."""
from collections import Counter
import json
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from scripts.run_discoveryworld_pilot import write_json

out = Path(__file__).resolve().parent
old = json.loads((out.parent / 'discoveryworld_estimator_research_20261008/author/trials/trial_001/result.json').read_text())
segments = [json.loads((folder / 'result.json').read_text()) for folder in (out, out / 'continued')]
latest = segments[-1]
state = json.loads((out / 'continued/agent/shared/reference_recalibration.json').read_text())
reference = {item['crystal']: item for item in state['research_table']['targets']}
target = Observation({key: item['value'] for key, item in reference[5]['measurements'].items()}, 0.)
fits = []
for qid, fitted in latest['research']['relation_research'].items():
    if not qid.endswith('_5') or 'basis' not in fitted:
        continue
    previous = old['research']['relation_research'].get(qid[:-1] + '4', {})
    before = Program.from_dict(previous['selected_program']).predict(target) if previous.get('selected_program') else None
    fits.append({'question_id': qid, 'fit_rows': len(fitted['fit_rows']),
        'operational_settings': [row for row in fitted['fit_rows'] if row.get('target_kind')],
        'diagnosis': fitted['diagnosis'],
        'best_fit_mse': min((c['fit_mse'] for c in fitted['candidates']), default=None),
        'prediction_without_C4_setting': before,
        'prediction_with_C4_setting': (fitted.get('selected_probe') or {}).get('predictions')})
events = [event for event in latest['events'] if event.get('model_call', 0) > old['costs']['model_calls']]
actions = Counter(event['action'] + (':' + event.get('arguments', {}).get('operator', '')
                   if event.get('arguments', {}).get('operator') else '') for event in events)
cost_keys = ('model_calls', 'reported_tokens', 'input_tokens', 'output_tokens',
             'cached_input_tokens', 'usage_missing_calls', 'python_execution_seconds',
             'research_compute_units', 'wall_time_seconds')
summary = {'study_status': 'finished',
    'status': latest['research']['llm_inquiries']['reference_extension_5']['status'],
    'agent_status': latest['status'], 'score': latest['score'],
    'prior_model_calls': old['costs']['model_calls'],
    'new_costs': {key: sum(segment['new_costs'][key] for segment in segments) for key in cost_keys},
    'new_environment_steps': sum(segment['new_environment_steps'] for segment in segments),
    'replay_environment_steps': [segment['replayed_environment_steps'] for segment in segments],
    'new_model_actions': dict(actions),
    'new_declared_hypotheses': [event['arguments']['hypothesis'] for event in events
        if event.get('arguments', {}).get('operator') == 'reference_hypothesis'],
    'new_physical_tests': [test for test in state.get('tests', []) if test['started'] >= len(old['paid_observations'])],
    'C4_successful_settings': [item for item in reference[4]['response_settings'] if item['activated']],
    'C5_fits': fits,
    'old_cases': json.loads((out / 'old_case_results.json').read_text()),
    'scope': 'Opened development task, host evidence integration and model continuation; '
             'not fresh transfer or a matched agent comparison. Operational settings are '
             'not exact latent resonance. One rejected oversized request remains counted. '
             'Canonical environment/research costs are reported above; legacy observation_cost '
             'and compute_cost aliases in raw restored agent files are not used.'}
write_json(out / 'summary.json', summary)
print(json.dumps({key: summary[key] for key in ('status', 'new_costs', 'new_environment_steps',
    'new_model_actions', 'new_declared_hypotheses', 'new_physical_tests')}, indent=2))

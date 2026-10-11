"""Summarize the completed fixed continuations without invoking a model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cohort = json.loads((ROOT / 'live/summary.json').read_text())
rows = []
for run in cohort['runs']:
    directory = ROOT / 'live' / f"repeat_{run['repetition']:02d}" / run['variant']
    row = {key: run[key] for key in ('variant', 'repetition', 'status', 'success',
                                    'new_model_calls', 'costs', 'cache')}
    # Nested research events are not in provider-call order. Read actual calls
    # instead of slicing three events to exclude recorded prefix decisions.
    cached = []
    logs = sorted((directory / 'agent').glob('code_call_*.jsonl'))
    for log in logs:
        usage = {}
        for line in log.read_text().splitlines():
            event = json.loads(line)
            if event.get('type') == 'turn.completed':
                usage = event.get('usage', {})
        if isinstance(usage.get('cached_input_tokens'), int):
            cached.append(usage['cached_input_tokens'])
    row['stored_summary_cache'] = row['cache']
    row['cache'] = {'reported_calls': len(cached), 'missing_calls': len(logs) - len(cached),
                    'hit_calls': sum(value > 0 for value in cached),
                    'zero_hit_calls': sum(value == 0 for value in cached)}
    row['source'] = str((directory / 'summary.json').relative_to(ROOT))
    row['actions_after_common_prefix'] = run['costs']['environment_steps'] - 171
    row['research_units_after_common_prefix'] = run['costs']['research_compute_units'] - 934942
    row['new_reference_hypothesis_submissions'] = [
        {'model_call': event['model_call'], 'hypothesis': event['declared_hypothesis']}
        for event in run['scientific_trace']['decisions']
        if event.get('declared_hypothesis') and (event.get('model_call') or 0) > 3]
    row['confirmed_c5_points'] = [trial for trial in run['point_confirmations']
                                if trial['number'] == 5 and trial['physically_confirmed']]
    rows.append(row)

totals = {}
for variant in ('before', 'after'):
    selected = [row for row in rows if row['variant'] == variant]
    totals[variant] = {
        'runs': len(selected), 'completed': sum(row['success'] for row in selected),
        'runs_with_new_function_basis_submission': sum(bool(row['new_reference_hypothesis_submissions']) for row in selected),
        'runs_with_c5_requested_point_confirmation': sum(bool(row['confirmed_c5_points']) for row in selected),
        'new_model_calls': sum(row['new_model_calls'] for row in selected),
        'actions_after_common_prefix': sum(row['actions_after_common_prefix'] for row in selected),
        'research_units_after_common_prefix': sum(row['research_units_after_common_prefix'] for row in selected),
        'costs': {key: sum(row['costs'][key] for row in selected)
                  for key in ('environment_steps', 'research_compute_units', 'input_tokens',
                              'output_tokens', 'cached_input_tokens')},
        'cache': {key: sum(row['cache'][key] for row in selected)
                  for key in ('reported_calls', 'missing_calls', 'hit_calls', 'zero_hit_calls')}}

analysis = {'status': cohort['status'], 'finished_runs': cohort['finished_runs'],
    'planned_runs': cohort['planned_runs'], 'common_prefix': {'recorded_decisions': 3,
        'observations': 171, 'research_compute_units': 934942},
    'totals': totals, 'runs': rows,
    'cache_source': 'Actual agent/code_call_*.jsonl completion usage; original run summaries preserved.',
    'scope': 'Three model repetitions per return policy on one previously opened failed state. '
        'Both arms are Hegel with the same historical-view repair. New basis submissions exclude '
        'the replayed interaction proposal; they do not count every supporting premise or question. '
        'The continuation is not a fresh task or a Hegel-versus-Code comparison.'}
(ROOT / 'live_analysis.json').write_text(json.dumps(analysis, indent=2))
print(json.dumps(totals, indent=2))

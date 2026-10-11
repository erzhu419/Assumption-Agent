"""Reuse raw-provider usage accounting; recorded decisions are not new proposals."""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
source = OUT.parent / 'response_research_connections_live_20261010/analyze.py'
spec = importlib.util.spec_from_file_location('prior_usage_analysis', source)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def analyze(cohort, protocol):
    rows = []
    for run in cohort['runs']:
        directory = OUT / 'live' / f"repeat_{run['repetition']:02d}" / run['variant']
        row = {key: run.get(key) for key in
               ('variant', 'repetition', 'status', 'success', 'new_model_calls', 'costs', 'next_decision')}
        row['source'] = str((directory / 'summary.json').relative_to(OUT))
        row['provider_usage'] = prior.actual_call_usage(directory)
        row['confirmed_c5_requested_points'] = [trial for trial in run.get('point_confirmations', [])
            if trial['number'] == 5 and trial['physically_confirmed']]
        row['new_reference_hypothesis_submissions'] = [
            {'model_call': event['model_call'], 'hypothesis': event['declared_hypothesis']}
            for event in run.get('scientific_trace', {}).get('decisions', [])
            if event.get('declared_hypothesis') and
            (event.get('model_call') or 0) > protocol['recorded_prefix_calls']]
        rows.append(row)
    totals = {}
    for variant in ('before', 'after'):
        group = [row for row in rows if row['variant'] == variant]
        totals[variant] = {
            'finished_runs': len(group), 'completed': sum(row['success'] is True for row in group),
            'missing_success_records': sum(row['success'] is None for row in group),
            'c5_confirmed_runs': sum(bool(row['confirmed_c5_requested_points']) for row in group),
            'new_basis_submission_runs': sum(bool(row['new_reference_hypothesis_submissions']) for row in group),
            'new_model_calls': prior.reported_sum([row['new_model_calls'] for row in group]),
            'costs': {key: prior.reported_sum([(row['costs'] or {}).get(key) for row in group])
                      for key in prior.COSTS},
            'cache': {key: sum(row['provider_usage']['cache'][key] for row in group)
                      for key in prior.CACHE}}
    return {**{key: cohort[key] for key in ('status', 'planned_runs', 'finished_runs')},
        'totals': totals, 'runs': rows, 'scope': protocol['scope'],
        'cost_scope': 'Full-run costs; only actual provider logs enter cache-call accounting. '
                      'The seven recorded choices are excluded from new-basis submissions.'}


if __name__ == '__main__':
    result = analyze(json.loads((OUT / 'live/summary.json').read_text()),
                     json.loads((OUT / 'live_protocol.json').read_text()))
    (OUT / 'live_analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('status', 'finished_runs', 'totals')}), flush=True)

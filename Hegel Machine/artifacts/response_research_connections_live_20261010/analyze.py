"""Summarize finished continuations; never invoke or steer an experimental agent."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COSTS = ('environment_steps', 'research_compute_units', 'python_execution_seconds',
         'input_tokens', 'output_tokens', 'cached_input_tokens')
CACHE = ('reported_calls', 'missing_calls', 'hit_calls', 'zero_hit_calls')


def reported_sum(values):
    """Keep missing measurements distinct from a measured zero."""
    known = [value for value in values if isinstance(value, (int, float))]
    return {'reported_total': sum(known) if known else None,
            'reported_runs': len(known), 'missing_runs': len(values) - len(known)}


def actual_call_usage(path):
    """Read only provider calls, independent of nested research-event order."""
    calls = []
    for log in sorted((path / 'agent').glob('code_call_*.jsonl')):
        usage = {}
        for line in log.read_text().splitlines():
            event = json.loads(line)
            if event.get('type') == 'turn.completed':
                # Match CodexCodeProvider: the last completion supplies usage.
                usage = event.get('usage', {})
        calls.append(usage)
    cached = [usage['cached_input_tokens'] for usage in calls
              if isinstance(usage.get('cached_input_tokens'), int)]
    return {
        'calls': len(calls),
        'cache': {'reported_calls': len(cached),
                  'missing_calls': len(calls) - len(cached),
                  'hit_calls': sum(value > 0 for value in cached),
                  'zero_hit_calls': sum(value == 0 for value in cached)},
        'tokens': {key: reported_sum([usage.get(key) for usage in calls])
                   for key in ('input_tokens', 'output_tokens', 'cached_input_tokens')},
    }


def analyze(cohort, protocol, directory):
    rows = []
    for run in cohort['runs']:
        path = directory / 'live' / f"repeat_{run['repetition']:02d}" / run['variant']
        decisions = run.get('scientific_trace', {}).get('decisions')
        trials = run.get('point_confirmations')
        usage = actual_call_usage(path)
        rows.append({
            **{key: run.get(key) for key in
               ('variant', 'repetition', 'status', 'success', 'new_model_calls', 'costs')},
            'cache': usage['cache'],
            'provider_usage': usage,
            'stored_summary_cache': run.get('cache'),
            'cache_summary_disagrees': run.get('cache') != usage['cache'],
            'source': str((path / 'summary.json').relative_to(directory))
                if (path / 'summary.json').exists() else run.get('log'),
            'next_decision': run.get('next_decision'),
            'new_reference_hypothesis_submissions': [
                {'model_call': event['model_call'], 'hypothesis': event['declared_hypothesis']}
                for event in decisions
                if event.get('declared_hypothesis') and (event.get('model_call') or 0) > 3
            ] if decisions is not None else None,
            'confirmed_c5_requested_points': [trial for trial in trials
                if trial['number'] == 5 and trial['physically_confirmed']
            ] if trials is not None else None,
        })

    boundaries = []
    for job in protocol['jobs']:
        path = directory / 'live' / f"repeat_{job['repetition']:02d}" / job['variant'] / 'next_decision.json'
        row = next((row for row in rows if row['variant'] == job['variant']
                    and row['repetition'] == job['repetition']), None)
        boundaries.append({**job, 'finished': row is not None,
            'next_decision': row['next_decision'] if row and row['next_decision'] is not None
                else json.loads(path.read_text()) if path.exists() else None})

    totals = {}
    for variant in ('before', 'after'):
        selected = [row for row in rows if row['variant'] == variant]
        totals[variant] = {
            'planned_runs': sum(job['variant'] == variant for job in protocol['jobs']),
            'finished_runs': len(selected),
            'completed': sum(row['success'] is True for row in selected),
            'missing_success_records': sum(row['success'] is None for row in selected),
            'runs_with_c5_requested_point_confirmation': sum(
                bool(row['confirmed_c5_requested_points']) for row in selected),
            'missing_confirmation_records': sum(
                row['confirmed_c5_requested_points'] is None for row in selected),
            'runs_with_new_reference_hypothesis_submissions': sum(
                bool(row['new_reference_hypothesis_submissions']) for row in selected),
            'missing_scientific_traces': sum(
                row['new_reference_hypothesis_submissions'] is None for row in selected),
            'new_model_calls': reported_sum([row['new_model_calls'] for row in selected]),
            'costs': {key: reported_sum([(row['costs'] or {}).get(key) for row in selected])
                      for key in COSTS},
            'cache': {key: reported_sum([(row['cache'] or {}).get(key) for row in selected])
                      for key in CACHE},
        }
    return {'status': cohort['status'], 'finished_runs': cohort['finished_runs'],
        'planned_runs': cohort['planned_runs'], 'totals': totals, 'runs': rows,
        'fresh_decision_boundaries': boundaries,
        'recorded_decisions_per_run': protocol['recorded_prefix_calls'],
        'cost_scope': 'Full-run world, research-compute and Python costs. No subtraction of '
            '171 or 180 observations. Token/cache counts are supplied by actual model calls; '
            'the three replayed decisions report zero tokens and do not enter the cache denominator. '
            'Cache-call counts are recomputed from agent/code_call_*.jsonl: slicing the first '
            'three nested research events in the original runner can discard live calls and retain '
            'recorded calls. Original run summaries remain unchanged.',
        'submission_scope': 'Recorded declarations after model call 3, not generalization or '
            'discovery judgments. Failed finished runs remain in the finished-run denominator; '
            'missing scientific or cost records are reported explicitly.',
        'protocol_scope': protocol['scope']}


if __name__ == '__main__':
    cohort = json.loads((ROOT / 'live/summary.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    result = analyze(cohort, protocol, ROOT)
    (ROOT / 'live_analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in
                      ('status', 'finished_runs', 'planned_runs', 'totals')}, indent=2))

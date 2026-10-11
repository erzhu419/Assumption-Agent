"""Aggregate frozen offline study reports without rerunning learning or scoring."""
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
protocol = json.loads((output / 'protocol.json').read_text())
def interval_width(intervals):
    # A relation can appear twice when its disconnected admissible cells are
    # copied into alternative candidate records. Count each parameter's union.
    domains = {}
    for interval in intervals:
        key = json.dumps({k: interval.get(k) for k in ('symbol', 'source_concept_id',
                         'source_definition_id', 'relation', 'predicate')}, sort_keys=True)
        domains.setdefault(key, []).extend(interval['radius_intervals'])
    total = 0.0
    for values in domains.values():
        merged = []
        for lo, hi in sorted(values):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        total += sum(hi - lo for lo, hi in merged)
    return total

rows = []
for arm in protocol['arms']:
    for case, declared in protocol['cases'].items():
        folder = output / 'runs' / arm / case
        if not (folder / 'report.json').exists():
            continue
        saved = json.loads((folder / 'report.json').read_text())
        report = saved['research']
        evaluation = json.loads((folder / 'evaluation.json').read_text()) if (folder / 'evaluation.json').exists() else None
        revisions = [a for a in report['actions'] if a['action'] == 'revise_structure_boundary']
        nonshrinking = [a for a in revisions if a.get('before_interval') and a.get('after_interval') and
                       a['after_interval'][1] - a['after_interval'][0] >=
                       a['before_interval'][1] - a['before_interval'][0] - 1e-12]
        previous_width = None
        conditioned = 0
        chain_conditioned = 0
        nonshrinking_conditionings = 0
        repeated_nonshrinking_conditionings = 0
        nonshrinking_conditioning_compute = 0
        for action in report['actions']:
            if action['action'] not in ('inherit_relation_intervals', 'condition_inherited_relation_intervals'):
                continue
            width = interval_width(action['intervals'])
            if action['action'] == 'inherit_relation_intervals':
                chain_conditioned = 0
            if action['action'] == 'condition_inherited_relation_intervals':
                if previous_width is not None and width >= previous_width - 1e-12:
                    nonshrinking_conditionings += 1
                    repeated_nonshrinking_conditionings += chain_conditioned > 0
                    nonshrinking_conditioning_compute += action.get('compute_cost', 0)
                conditioned += 1
                chain_conditioned += 1
            previous_width = width
        completed = report['status'] == 'completed'
        terminal_pass = evaluation['mse'] <= 1e-7 if evaluation else None
        rows.append({'arm': arm, 'case': case, 'knowledge_mode': declared['knowledge_mode'],
                     'observation_budget': declared['observation_budget'], 'status': report['status'],
                     'completed': completed, 'terminal_pass': terminal_pass,
                     'completed_and_terminal_pass': completed and terminal_pass if evaluation else None,
                     'false_completed': completed and not terminal_pass if evaluation else None,
                     'mse': evaluation['mse'] if evaluation else None,
                     'costs': report['costs'], 'cpu_seconds': saved['cpu_seconds'],
                     'elapsed_seconds': saved['elapsed_seconds'], 'boundary_revisions': len(revisions),
                     'nonshrinking_boundary_revisions': len(nonshrinking),
                     'nonshrinking_boundary_compute': sum(a.get('compute_cost', 0) for a in nonshrinking),
                     'relation_conditionings': conditioned,
                     'nonshrinking_relation_conditionings': nonshrinking_conditionings,
                     'repeated_nonshrinking_relation_conditionings': repeated_nonshrinking_conditionings,
                     'nonshrinking_relation_conditioning_compute': nonshrinking_conditioning_compute,
                     'original_runtime': saved['runtime'], 'original_version': saved['version']})
groups = []
for mode, budget in sorted({(r['knowledge_mode'], r['observation_budget']) for r in rows}):
    grouped = {}
    indexed = {}
    for arm in protocol['arms']:
        rs = [r for r in rows if r['arm'] == arm and r['knowledge_mode'] == mode and r['observation_budget'] == budget]
        indexed[arm] = {r['case']: r for r in rs}
        evaluated = [r for r in rs if r['mse'] is not None]
        grouped[arm] = {'learning_jobs': len(rs), 'evaluated_jobs': len(evaluated),
                       'completed': sum(r['completed'] for r in rs),
                       'terminal_pass': sum(r['terminal_pass'] for r in evaluated),
                       'completed_and_terminal_pass': sum(r['completed_and_terminal_pass'] for r in evaluated),
                       'false_completed': sum(r['false_completed'] for r in evaluated),
                       'mean_mse': sum(r['mse'] for r in evaluated) / len(evaluated) if evaluated else None,
                       'cpu_seconds': sum(r['cpu_seconds'] for r in rs),
                       'elapsed_seconds': sum(r['elapsed_seconds'] for r in rs),
                       'costs': {k: sum(r['costs'][k] for r in rs) for k in ('observation_cost', 'measurement_cost', 'compute_cost', 'queries')},
                       'boundary_revisions': sum(r['boundary_revisions'] for r in rs),
                       'nonshrinking_boundary_revisions': sum(r['nonshrinking_boundary_revisions'] for r in rs),
                       'nonshrinking_boundary_compute': sum(r['nonshrinking_boundary_compute'] for r in rs),
                       'relation_conditionings': sum(r['relation_conditionings'] for r in rs),
                       'nonshrinking_relation_conditionings': sum(r['nonshrinking_relation_conditionings'] for r in rs),
                       'repeated_nonshrinking_relation_conditionings': sum(r['repeated_nonshrinking_relation_conditionings'] for r in rs),
                       'nonshrinking_relation_conditioning_compute': sum(r['nonshrinking_relation_conditioning_compute'] for r in rs)}
    common = sorted(set(indexed['baseline']) & set(indexed['revised']))
    common = [c for c in common if indexed['baseline'][c]['mse'] is not None and indexed['revised'][c]['mse'] is not None]
    differences = [indexed['revised'][c]['mse'] - indexed['baseline'][c]['mse'] for c in common]
    comparison = {'knowledge_mode': mode, 'observation_budget': budget, **grouped,
                  'paired_cases': len(common), 'better': sum(d < -1e-7 for d in differences),
                  'worse': sum(d > 1e-7 for d in differences), 'within_1e_minus_7': sum(abs(d) <= 1e-7 for d in differences)}
    if common and len(common) == len(indexed['baseline']) == len(indexed['revised']):
        comparison['relative_change_percent'] = {
            k: ((grouped['revised'][k] / grouped['baseline'][k] - 1) * 100 if grouped['baseline'][k] else None)
            for k in ('mean_mse', 'cpu_seconds', 'elapsed_seconds')}
        comparison['relative_change_percent'].update({k: (grouped['revised']['costs'][k] / grouped['baseline']['costs'][k] - 1) * 100
            for k in grouped['baseline']['costs'] if grouped['baseline']['costs'][k]})
    groups.append(comparison)
result = {'planned_jobs': len(protocol['cases']) * len(protocol['arms']), 'learning_jobs': len(rows),
          'evaluated_jobs': sum(r['mse'] is not None for r in rows),
          'scope': protocol.get('scope'), 'groups': groups, 'rows': rows}
(output / 'aggregate_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))

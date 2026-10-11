"""Pair fixed task outcomes and resource totals after all terminal scoring."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
for group in ('development', 'fresh_replication'):
    output = base / group
    if not (output / 'summary.json').exists():
        continue
    summary = json.loads((output / 'summary.json').read_text())
    by_arm = {}
    for row in summary['rows']:
        by_arm.setdefault(row['arm'], {})[row['root'], row['position']] = row
    comparisons = {}
    pairs = [('new_independent', 'old_independent')]
    if 'new_cumulative' in by_arm:
        pairs.append(('new_cumulative', 'new_independent'))
    for candidate, reference in pairs:
        rows = []
        for key, row in by_arm[candidate].items():
            old = by_arm[reference][key]
            a, b = old['mean_response_mse'], row['mean_response_mse']
            rows.append({'root': row['root'], 'position': row['position'], 'stage': row['stage'],
                'reference_mse': a, 'candidate_mse': b,
                'result': 'win' if b < a else 'loss' if b > a else 'tie',
                'reference_status': old['status'], 'candidate_status': row['status'],
                'reference_kind': old['kind'], 'candidate_kind': row['kind']})
        result = {'paired_tasks': {name: sum(row['result'] == name for row in rows)
                                  for name in ('win', 'tie', 'loss')},
                  'by_stage': {}, 'by_root': {}, 'relative_change': {}, 'rows': rows}
        for field, destination in (('stage', 'by_stage'), ('root', 'by_root')):
            for value in sorted({row[field] for row in rows}):
                selected = [row for row in rows if row[field] == value]
                a = sum(row['reference_mse'] for row in selected) / len(selected)
                b = sum(row['candidate_mse'] for row in selected) / len(selected)
                result[destination][value] = {'reference_mse': a, 'candidate_mse': b,
                    'relative_change': b / a - 1 if a else None,
                    'result': 'win' if b < a else 'loss' if b > a else 'tie'}
        a, b = summary['totals'][reference], summary['totals'][candidate]
        for category in ('learning_costs', 'all_costs'):
            result['relative_change'][category] = {key: b[category][key] / a[category][key] - 1
                for key in ('observation_cost', 'compute_cost')}
        for key in ('mean_mse', 'mean_response_mse'):
            result['relative_change'][key] = b[key] / a[key] - 1 if a[key] else None
        result['paired_roots'] = {name: sum(row['result'] == name for row in result['by_root'].values())
                                  for name in ('win', 'tie', 'loss')}
        result['worst_pair_increase'] = max(rows, key=lambda row: row['candidate_mse'] - row['reference_mse'])
        result['largest_pair_improvement'] = min(rows, key=lambda row: row['candidate_mse'] - row['reference_mse'])
        result['relative_change']['learning_cpu_seconds'] = b['learning_cpu_seconds'] / a['learning_cpu_seconds'] - 1
        comparisons[f'{candidate}_vs_{reference}'] = result
    report = {'comparisons': comparisons, 'learning_cpu_seconds': {arm: total['learning_cpu_seconds']
        for arm, total in summary['totals'].items()}, 'all_cpu_seconds': {arm: sum(total[key] for key in
        ('source_cpu_seconds', 'learning_cpu_seconds', 'terminal_cpu_seconds'))
        for arm, total in summary['totals'].items()}}
    (output / 'paired_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(group, {name: {key: value[key] for key in ('paired_tasks', 'relative_change')}
                  for name, value in comparisons.items()})

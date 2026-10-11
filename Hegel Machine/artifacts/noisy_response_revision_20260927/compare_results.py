"""Summarize fixed paired terminal outcomes; never feeds results back to research."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
for group in ('development_recovery', 'fresh_replication', 'deterministic_control',
              'development_noise_support', 'fresh_replication_noise_support',
              'development_final', 'deterministic_control_noise_support'):
    output = base / group
    if not (output / 'summary.json').exists():
        continue
    summary = json.loads((output / 'summary.json').read_text())
    old = {(row['root'], row['position']): row for row in summary['rows']
           if row['arm'] == 'baseline_empty'}
    metric = 'mse' if group.startswith('deterministic_control') else 'mean_response_mse'
    rows = []
    for row in summary['rows']:
        if row['arm'] != 'empty_start':
            continue
        previous = old[row['root'], row['position']]
        a, b = previous[metric], row[metric]
        rows.append({'root': row['root'], 'position': row['position'], 'stage': row['stage'],
                     'old_mse': a, 'new_mse': b,
                     'result': 'win' if b < a else 'loss' if b > a else 'tie',
                     'old_status': previous['status'], 'new_status': row['status'],
                     'old_kind': previous['kind'], 'new_kind': row['kind']})
    report = {'primary_metric': metric, 'paired_tasks': {
        name: sum(row['result'] == name for row in rows) for name in ('win', 'tie', 'loss')},
        'by_stage': {}, 'by_root': {}, 'relative_change': {}, 'all_cpu_seconds': {}, 'rows': rows}
    for field, destination in (('stage', 'by_stage'), ('root', 'by_root')):
        for value in sorted({row[field] for row in rows}):
            selected = [row for row in rows if row[field] == value]
            a = sum(row['old_mse'] for row in selected) / len(selected)
            b = sum(row['new_mse'] for row in selected) / len(selected)
            report[destination][value] = {'old_mse': a, 'new_mse': b,
                'relative_change': b / a - 1 if a else None,
                'result': 'win' if b < a else 'loss' if b > a else 'tie'}
    a, b = (summary['totals'][arm] for arm in ('baseline_empty', 'empty_start'))
    for category in ('learning_costs', 'all_costs'):
        report['relative_change'][category] = {key: b[category][key] / a[category][key] - 1
            for key in ('observation_cost', 'compute_cost')}
    for key in ('mean_mse', 'mean_response_mse'):
        if a[key] is not None:
            report['relative_change'][key] = b[key] / a[key] - 1
    for arm, total in summary['totals'].items():
        report['all_cpu_seconds'][arm] = sum(total[name] for name in
            ('source_cpu_seconds', 'learning_cpu_seconds', 'terminal_cpu_seconds'))
    (output / 'paired_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(group, report['paired_tasks'], report['relative_change'], report['all_cpu_seconds'])

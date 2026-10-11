"""Read retained reports to distinguish rediscovery from cross-task reuse."""
from collections import Counter
import gzip
import json
from pathlib import Path

base = Path(__file__).resolve().parent
output = base / 'fresh_replication'
summary = json.loads((output / 'summary.json').read_text())
protocol = json.loads((output / 'protocol.json').read_text())
scores = {(row['arm'], row['root'], row['position']): row for row in summary['rows']}

def widths(program):
    values, pending = set(), [program]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if item.get('op') == 'window_sum':
                values.add(item['lag'])
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return sorted(values)

arms = {}
for arm in protocol['arms']:
    rows, screening = [], []
    for root in protocol['roots']:
        with gzip.open(output / 'runs' / arm / root / 'checkpoint.json.gz', 'rt') as stream:
            state = json.load(stream)
        screening.extend(event for event in state.get('scheduler_events', [])
            if event['action'] == 'try_successor_knowledge')
        for position, record in enumerate(state['results']):
            report = record['research']
            program = report.get('research_program') or {}
            steps = program.get('selection_steps', [])
            discoveries = [{key: step.get(key) for key in
                ('operation', 'width', 'history_bound', 'basis_terms', 'selected_terms', 'width_trials')}
                for step in steps if step['operation'] == 'discover_temporal_coordinates']
            scale_proposals = [candidate for event in report['actions']
                if event['action'] == 'propose_mechanism_revisions'
                for candidate in event.get('alternatives', [])
                if (candidate.get('revision') or {}).get('temporal_revisions')]
            score = scores[arm, root, position]
            rows.append({'root': root, 'position': position, 'stage': protocol['stages'][position],
                'status': report['status'], 'kind': report['selected_kind'],
                'mean_response_mse': score['mean_response_mse'],
                'used_sources': report['used_sources'],
                'reused_procedure': bool(program.get('reused_procedure')),
                'coordinate_search_calls': sum(e['action'] == 'propose_coordinate_reconstruction'
                    for e in report['actions']),
                'coordinate_search_compute': sum(e.get('compute_cost', 0) for e in report['actions']
                    if e['action'] == 'propose_coordinate_reconstruction'),
                'retained_discovery_steps': discoveries,
                'research_program_execution_calls': sum(e['action'] == 'execute_research_programs'
                    for e in report['actions']),
                'research_program_execution_compute': sum(e.get('compute_cost', 0) for e in report['actions']
                    if e['action'] == 'execute_research_programs'),
                'research_program_execution_candidates': sum(len(e['alternatives']) for e in report['actions']
                    if e['action'] == 'execute_research_programs'),
                'temporal_revision_candidates': len(scale_proposals),
                'final_stored_widths': widths(report['program'])})
    def aggregate(selected):
        return {'tasks': len(selected),
            'completed': sum(row['status'] == 'completed' for row in selected),
            'mean_response_mse': sum(row['mean_response_mse'] for row in selected) / len(selected),
            'using_sources': sum(bool(row['used_sources']) for row in selected),
            'reused_procedure': sum(row['reused_procedure'] for row in selected),
            'coordinate_search_calls': sum(row['coordinate_search_calls'] for row in selected),
            'coordinate_search_compute': sum(row['coordinate_search_compute'] for row in selected),
            'retained_discovery_steps': sum(len(row['retained_discovery_steps']) for row in selected),
            **{key: sum(row[key] for row in selected) for key in
                ('research_program_execution_calls', 'research_program_execution_compute',
                 'research_program_execution_candidates')},
            'temporal_revision_candidates': sum(row['temporal_revision_candidates'] for row in selected),
            'final_width_sets': dict(Counter(','.join(map(str,row['final_stored_widths'])) or 'none'
                for row in selected))}
    arms[arm] = {'all_tasks': aggregate(rows),
        'successor_screening': {'events': len(screening),
            'compute_cost': sum(e['compute_cost'] for e in screening),
            'scheduled': sum(e['scheduled'] for e in screening)},
        'successor_tasks': aggregate([row for row in rows if row['position']]),
        'by_stage': {stage: aggregate([row for row in rows if row['stage'] == stage])
            for stage in protocol['stages']}, 'rows': rows}
result = {'scope': 'Retained reports only; later refits may replace the discovery program. Stored widths describe syntax, not an effective aggregate horizon. Using sources or replaying procedures is descriptive evidence of reuse, not causal attribution of all gains.', 'arms': arms}
(output / 'discovery_attribution.json').write_text(json.dumps(result, indent=2) + '\n')
print({arm: {key: data[key] for key in ('all_tasks', 'successor_tasks')} for arm, data in arms.items()})

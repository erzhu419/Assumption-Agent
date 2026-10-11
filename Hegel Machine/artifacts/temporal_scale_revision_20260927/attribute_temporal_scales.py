"""Summarize recorded time-scale trials and fresh-trajectory support."""
from collections import Counter
import gzip
import json
from pathlib import Path

base = Path(__file__).resolve().parent

def widths(program):
    result, pending = Counter(), [program]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if node.get('op') == 'window_sum':
                result[str(node['lag'])] += 1
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return dict(result)

for group in ('development', 'fresh_replication'):
    output = base / group
    if not (output / 'summary.json').exists():
        continue
    protocol = json.loads((output / 'protocol.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    scores = {(row['arm'], row['root'], row['position']): row for row in summary['rows']}
    arms = {}
    for arm in protocol['arms']:
        rows = []
        for root in protocol['roots']:
            with gzip.open(output / 'runs' / arm / root / 'checkpoint.json.gz', 'rt') as stream:
                state = json.load(stream)
            for record in state['results']:
                position = record['task_position']
                report = record['research']
                score = scores[arm, root, position]
                proposals = []
                tested, selected_temporal = [], []
                temporal_candidates = {}
                for event in report['actions']:
                    if event['action'] == 'test_mechanism_predictions':
                        ids = {candidate['candidate_id'] for candidate in event['alternatives']
                            if (candidate.get('revision') or {}).get('temporal_revisions')}
                        tested.extend({key: result.get(key) for key in
                            ('candidate_id', 'status', 'measured_loss', 'rows')}
                            for result in event['evidence']['hypotheses']
                            if result['candidate_id'] in ids)
                    if event['action'] == 'select_mechanism_revision' and event['candidate_id'] in temporal_candidates:
                        selected_temporal.append({'candidate_id': event['candidate_id'],
                            'selection_loss': event['selection_loss'],
                            'requires_fresh_confirmation': event['requires_fresh_confirmation'],
                            'revisions': temporal_candidates[event['candidate_id']]})
                    if event['action'] != 'propose_mechanism_revisions':
                        continue
                    for candidate in event['alternatives']:
                        revision = candidate.get('revision') or {}
                        if revision.get('temporal_revisions'):
                            temporal_candidates[candidate['candidate_id']] = revision['temporal_revisions']
                            proposals.append({'candidate_id': candidate['candidate_id'],
                                'kind': candidate['kind'], 'training_loss': candidate['training_loss'],
                                'validation_loss': candidate['validation_loss'],
                                'revisions': revision['temporal_revisions'],
                                'trials': revision.get('temporal_trials', []),
                                'search_compute': revision.get('temporal_search_compute', 0)})
                revision = report.get('mechanism_revision') or {}
                final = {key: revision[key] for key in ('temporal_revisions', 'temporal_trials',
                    'temporal_search_compute', 'status', 'confirmation_loss', 'confirmation_rows')
                    if key in revision}
                if revision.get('temporal_revisions'):
                    final.update({key: revision[key] for key in ('probe_id', 'measured_loss') if key in revision})
                rows.append({'root': root, 'position': position, 'stage': protocol['stages'][position],
                    'status': report['status'], 'kind': report['selected_kind'],
                    'terminal_mean_response_mse': score['mean_response_mse'],
                    'terminal_noisy_mse': score['mse'],
                    'recorded_proposals': proposals,
                    'paid_temporal_tests': tested,
                    'selected_temporal_hypotheses_requiring_confirmation': selected_temporal,
                    'final_temporal_revision': final if revision.get('temporal_revisions') else None,
                    'fresh_supported_temporal_revision': bool(revision.get('temporal_revisions')
                        and revision.get('status') == 'supported_on_fresh_trajectory'
                        and report['status'] == 'completed'),
                    'stored_window_width_counts_including_definitions': widths(report['program']),
                    'used_sources': report['used_sources']})
        stages = {}
        for stage in protocol['stages']:
            selected = [row for row in rows if row['stage'] == stage]
            proposals = [proposal for row in selected for proposal in row['recorded_proposals']]
            stages[stage] = {'tasks': len(selected),
                'recorded_temporal_proposals': len(proposals),
                'recorded_trials': sum(len(p['trials']) for p in proposals),
                'paid_temporal_tests': sum(len(row['paid_temporal_tests']) for row in selected),
                'paid_temporal_counterexample_records': sum(result['status'] == 'counterexample'
                    for row in selected for result in row['paid_temporal_tests']),
                'selected_temporal_hypotheses_requiring_confirmation': sum(
                    len(row['selected_temporal_hypotheses_requiring_confirmation']) for row in selected),
                'reported_search_compute': sum(p['search_compute'] for p in proposals),
                'final_temporal_candidates': sum(row['final_temporal_revision'] is not None for row in selected),
                'fresh_supported_temporal_candidates': sum(row['fresh_supported_temporal_revision'] for row in selected),
                'retained_temporal_counterexamples': sum(
                    (row['final_temporal_revision'] or {}).get('status') == 'counterexample'
                    for row in selected),
                'completed': sum(row['status'] == 'completed' for row in selected),
                'terminal_mean_response_mse': sum(row['terminal_mean_response_mse'] for row in selected) / len(selected),
                'final_stored_width_sets': dict(Counter(','.join(sorted(
                    row['stored_window_width_counts_including_definitions'], key=int)) or 'none'
                    for row in selected)),
                'final_kinds': dict(Counter(row['kind'] for row in selected))}
        arms[arm] = {'by_stage': stages, 'rows': rows}
    result = {'scope': 'Only candidates and trials present in retained research reports; discarded optimizer searches may not be exposed. Final selection is reported separately from fresh-trajectory support. Stored window widths are syntactic operators, not the effective aggregate horizon.', 'arms': arms}
    (output / 'temporal_scale_attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(group, {arm: value['by_stage'] for arm, value in arms.items()})

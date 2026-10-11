#!/usr/bin/env python3
"""Summarize actual selected draft lineage and paid acquisition returns."""
from collections import Counter
from pathlib import Path
import gzip
import json
import sys

output = Path(sys.argv[1])
protocol = json.loads((output / 'protocol.json').read_text())
summary = json.loads((output / 'summary.json').read_text())
consumption = {}
for arm in protocol['arms']:
    drafts, actions, selected, descendants = [], [], [], []
    for root in protocol['roots']:
        with gzip.open(output / 'runs' / arm / root / 'checkpoint.json.gz', 'rt') as handle:
            state = json.load(handle)
        records = {record['draft_id']: record for record in state['knowledge'].get('research_drafts', [])}
        drafts.extend(records.values())
        for record in records.values():
            parent = records.get(record.get('parent_draft_id'))
            if parent and parent['source_task_id'] != record['source_task_id']:
                descendants.append({'root': root, 'draft_id': record['draft_id'],
                    'parent_draft_id': parent['draft_id'], 'source_task_id': parent['source_task_id'],
                    'target_task_id': record['source_task_id']})
        for result in state['results']:
            report = result['research']
            actions.extend(report['actions'])
            for alternative in report.get('continuation_state', {}).get('alternatives', []):
                identifier = alternative.get('research_draft_id')
                parent = records.get(identifier)
                if (parent and parent['source_task_id'] != report['task_id']
                        and alternative['program'] == report['program']):
                    selected.append({'root': root, 'position': result['task_position'],
                        'draft_id': identifier, 'source_task_id': parent['source_task_id'],
                        'target_task_id': report['task_id'], 'status': report['status'],
                        'validation_loss': report['validation_loss'],
                        'reused_procedure': bool((report.get('research_program') or {}).get('reused_procedure'))})
                    break
    replay = [a for a in actions if a['action'] == 'replay_research_draft']
    assessed = [a for a in actions if a['action'] == 'assess_research_draft']
    temporary = [a for a in actions if a['action'] == 'select_partial_response_method']
    purchases = [a for a in actions if a['action'] == 'revise_competing_observed_conditions']
    decisions = [a for a in actions if a['action'] == 'choose_condition_research_investment']
    plans = [a for a in actions if a['action'] == 'plan_competing_observed_conditions']
    returns = [a['research_return'] for a in purchases if a.get('research_return')]
    planning = sum(a.get('planning_compute_cost', 0) for a in decisions) if decisions else sum(a.get('compute_cost', 0) for a in plans)
    consumption[arm] = {
        'numeric_drafts': sum(bool(d.get('condition_uncertainty')) for d in drafts),
        'replay_attempts': len(replay),
        'replay_attempts_producing_candidates': sum(a.get('training_loss') is not None for a in replay),
        'assessed_candidates': len(assessed),
        'assessment_statuses': dict(Counter(a['status'] for a in assessed)),
        'temporary_selections': len(temporary),
        'final_cross_task_selected_programs': len(selected),
        'final_selection_lineage': selected,
        'cross_task_draft_descendants': len(descendants),
        'descendant_lineage': descendants,
        'numeric_drafts_resolved': sum(d.get('status') == 'resolved' for d in drafts if d.get('condition_uncertainty')),
        'competition': {
            'purchases': len(purchases),
            'changed_programs': sum(a['changed_program'] for a in purchases),
            'observations': sum(a['measured_rows'] for a in purchases),
            'planning_compute_cost': planning,
            'revision_compute_cost': sum(a['compute_cost'] for a in purchases),
            'stop_reasons': dict(Counter(a['reason'] for a in decisions if not a['buy'])),
            'return_records': len(returns),
            'informative_batches': sum(a['informative'] for a in returns),
            'interval_narrowed_batches': sum(a['interval_narrowed'] for a in returns),
            'new_counterexample_batches': sum(a['new_counterexample'] for a in returns),
            'uninformative_batches': sum(not a['informative'] for a in returns),
            'uninformative_observations': sum(a['observation_cost'] for a in returns if not a['informative']),
            'uninformative_revision_compute': sum(a['compute_cost'] for a in returns if not a['informative'])}}
    if len(returns) != len(purchases):
        for key in ('informative_batches', 'interval_narrowed_batches', 'new_counterexample_batches',
                    'uninformative_batches', 'uninformative_observations', 'uninformative_revision_compute'):
            consumption[arm]['competition'][key] = None
result = {'basis': 'Final learning checkpoints only. A final lineage matches the exact report program to its serialized alternative and named earlier-task draft; temporary selection is counted separately. Informative acquisition returns come from recorded paid evidence, not terminal performance.',
    'arms': consumption}
(output / 'knowledge_consumption.json').write_text(json.dumps(result, indent=2) + '\n')
chains = {}
for reference in ('old_cumulative', 'new_independent'):
    paired = []
    for root in protocol['roots']:
        candidate = [r['mse'] for r in summary['rows'] if r['arm'] == 'new_cumulative' and r['root'] == root]
        baseline = [r['mse'] for r in summary['rows'] if r['arm'] == reference and r['root'] == root]
        left, right = sum(candidate) / len(candidate), sum(baseline) / len(baseline)
        paired.append({'root': root, 'candidate_mean_mse': left, 'reference_mean_mse': right,
            'difference': left - right})
    chains[reference] = {'chains': len(paired),
        'wins': sum(item['difference'] < -1e-12 for item in paired),
        'ties': sum(abs(item['difference']) <= 1e-12 for item in paired),
        'losses': sum(item['difference'] > 1e-12 for item in paired), 'paired': paired}
(output / 'chain_comparisons.json').write_text(json.dumps(chains, indent=2) + '\n')
print({arm: {key: value[key] for key in ('replay_attempts', 'assessed_candidates', 'temporary_selections', 'final_cross_task_selected_programs', 'cross_task_draft_descendants', 'competition')} for arm, value in consumption.items()})
print({'chains': {arm: {key: value[key] for key in ('chains', 'wins', 'ties', 'losses')} for arm, value in chains.items()}})

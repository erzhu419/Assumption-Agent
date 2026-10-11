"""Read completed task histories; summarize paid decisions and inherited knowledge."""
from collections import Counter
import gzip
import json
from pathlib import Path

base = Path(__file__).resolve().parent
for group in ('development', 'fresh_replication'):
    output = base / group
    if not (output / 'summary.json').exists():
        continue
    protocol = json.loads((output / 'protocol.json').read_text())
    results = {}
    for arm in protocol['arms']:
        rows, decisions, feedback = [], [], []
        for root in protocol['roots']:
            with gzip.open(output / 'runs' / arm / root / 'checkpoint.json.gz', 'rt') as stream:
                state = json.load(stream)
            for result in state['results']:
                position = result['task_position']
                report = result['research']
                actions = report['actions']
                choices = [a for a in actions if a['action'] == 'choose_noisy_response_update']
                confirmations = [a for a in actions if a['action'] == 'confirm_noisy_response_refit']
                row = {'root': root, 'position': position, 'stage': protocol['stages'][position],
                    'status': report['status'], 'kind': report['selected_kind'],
                    'available_concepts': result['available_concepts'], 'used_sources': report['used_sources'],
                    'costs': report['costs'],
                    'research_operations': [step['operation'] for step in
                        (report.get('research_program') or {}).get('selection_steps', [])],
                    'update_decisions': [{k: v for k, v in a.items()
                        if isinstance(v, (str, float, int, bool)) or v is None} for a in choices],
                    'refit_confirmations': confirmations,
                    'refit_proposals': sum(a['action'] == 'propose_noisy_response_update' for a in actions),
                    'revisit_count': sum(r['task_position'] == position for r in state['revisits'])}
                rows.append(row)
                decisions.extend(choices)
                feedback.extend(a for a in actions if a['action'] == 'research_action_feedback'
                                and a.get('research_action') == 'response_update')
        stages = {}
        for stage in protocol['stages']:
            selected = [row for row in rows if row['stage'] == stage]
            stages[stage] = {'tasks': len(selected), 'kinds': dict(Counter(row['kind'] for row in selected)),
                'tasks_using_sources': sum(bool(row['used_sources']) for row in selected),
                'tasks_with_incoming_concepts': sum(bool(row['available_concepts']) for row in selected),
                'update_decision_reasons': dict(Counter(a.get('reason') for row in selected for a in row['update_decisions'])),
                'response_confirmation_accepted': sum(a['accepted'] for row in selected for a in row['refit_confirmations']),
                'response_confirmation_rejected': sum(not a['accepted'] for row in selected for a in row['refit_confirmations']),
                'revisits': sum(row['revisit_count'] for row in selected)}
        results[arm] = {'action_count_scope': 'Final retained reports, not duplicated checkpoint snapshots',
            'by_stage': stages,
            'update_decision_reasons': dict(Counter(a.get('reason') for a in decisions)),
            'reported_response_update_feedback_costs': {key: sum(a.get(key, 0) for a in feedback)
                for key in ('observation_cost', 'compute_cost', 'measurement_cost')}, 'rows': rows}
    (output / 'attribution.json').write_text(json.dumps(results, indent=2) + '\n')
    print(group, {arm: {'choices': value['update_decision_reasons'],
                       'update_costs': value['reported_response_update_feedback_costs']}
                  for arm, value in results.items()})

#!/usr/bin/env python3
"""Summarize executed ambiguity, role revision and calls to earlier knowledge."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    runs, arms = [], {arm: Counter() for arm in protocol['arms']}
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            report = json.loads((output / 'runs' / f'{seed}_{arm}.json').read_text())
            if len(report['evaluation']) != len(report['tasks']):
                raise ValueError('Finish the complete sequence before reporting behavior.')
            rows = []
            for entry, terminal in zip(report['results'], report['evaluation']):
                result = entry['research']
                model = result.get('object_research') or {}
                equivalence = result.get('object_equivalence') or {}
                actions = result['actions']
                unresolved_x = [role for role in model.get('unresolved', []) if role.endswith(':x')]
                adopted = any(event['action'] == 'adopt_equivalent_object_program' for event in actions)
                shared_with_unresolved_x = (adopted and bool(unresolved_x)
                    and not model.get('complete') and result.get('learned_object_schema') is None
                    and equivalence.get('status') == 'algebraically_shared'
                    and len(equivalence.get('schemas', [])) > 1)
                sources = result.get('used_sources', [])
                previous_sources = [source for source in sources if source in entry['available_concepts']]
                revisions = [event.get('reason') for event in actions
                    if event['action'] == 'reopen_object_dependencies']
                row = {'position': entry['task_position'], 'kind': result['selected_kind'],
                    'loss': terminal['loss'], 'commands': not result['selected_channels'],
                    'unresolved_x': unresolved_x, 'shared_program_with_unresolved_x': shared_with_unresolved_x,
                    'equivalence_schema_count': len(equivalence.get('schemas', [])),
                    'reused_object_alternatives': any(event['action'] == 'reuse_object_alternatives' for event in actions),
                    'role_reopening_reasons': revisions, 'used_sources': sources,
                    'previous_task_sources': previous_sources}
                rows.append(row)
                arms[arm].update(tasks=1, shared_program_with_unresolved_x=int(shared_with_unresolved_x),
                    reused_object_alternatives=int(row['reused_object_alternatives']),
                    role_reopenings=len(revisions), tasks_with_previous_calls=int(bool(previous_sources)),
                    previous_call_edges=len(previous_sources),
                    later_composition_tasks_with_previous_calls=int(entry['task_position'] >= 3 and bool(previous_sources)))
            runs.append({'seed': seed, 'arm': arm, 'tasks': rows})
    result = {'experiment': protocol['experiment'], 'arms': arms, 'runs': runs}
    (output / 'behavior_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(arms, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    summarize(parser.parse_args().output)

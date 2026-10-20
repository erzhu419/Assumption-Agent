#!/usr/bin/env python3
"""Replay paid event prefixes without future labels; retain the original diagnosis.

This offline host diagnostic never writes to an investigator's workspace. Grouped
holdout below conditions on host-selected features/thresholds; it is not a fresh
confirmation or an autonomous discovery result.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_conditional_online_failure import aliasing, diagnose


def acquired_row_indices(value):
    """Read actual acquisition records, including nested failed confirmations."""
    if isinstance(value, dict):
        if 'row_indices' in value and 'rows' in value and 'action' in value:
            yield from value['row_indices']
        for item in value.values():
            yield from acquired_row_indices(item)
    elif isinstance(value, list):
        for item in value:
            yield from acquired_row_indices(item)


def records_at_prefix(training, event_count):
    known = set(acquired_row_indices([event.get('feedback', {})
                                     for event in training['events'][:event_count]]))
    return [record for record in training['paid_observations']
            if record['row_indices'] and set(record['row_indices']).issubset(known)]


def duplicate_groups(records):
    """The task resets on every action; equal complete actions share a holdout fold."""
    distinct, groups = [], []
    for record in records:
        action = record['action']
        index = next((index for index, previous in enumerate(distinct)
                      if action == previous), len(distinct))
        if index == len(distinct):
            distinct.append(action)
        groups.append(index)
    return groups


def analyze(study, checkpoints=(34, 35, 82, 83, 91, 102)):
    source = Path(study)
    plan = json.loads((source / 'protocol.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    training = json.loads(Path(summary['online_training']).read_text())
    previous = json.loads(Path(plan.get('resume_result', plan['practice_result'])).read_text())
    inherited_events = len(previous['events'])
    output = []
    for call in checkpoints:
        if call + inherited_events > len(training['events']):
            continue
        records = records_at_prefix(training, inherited_events + call)
        rows = [row for record in records for row in record['rows']]
        record_groups = duplicate_groups(records)
        groups = np.asarray([group for group, record in zip(record_groups, records)
                             for _ in record['rows']])
        event = training['events'][inherited_events + call - 1]
        repeated = sum(len(record['rows']) for index, record in enumerate(records)
                       if record_groups[index] in record_groups[:index])
        item = {'new_call': call, 'available_rows': len(rows), 'paid_trajectories': len(records),
            'distinct_reset_action_trajectories': len(set(record_groups)),
            'repeated_paid_rows_retained_in_costs': repeated,
            'last_row_index': max(index for record in records for index in record['row_indices']),
            'padding_aliasing': aliasing(rows),
            'actor_notes_at_this_call': event.get('arguments', {}).get('notes'),
            'action': event['action']}
        if call >= 91:
            item['grouped_paid_diagnostics'] = [
                diagnose(rows, groups, -.175, 'padded_polynomial'),
                diagnose(rows, groups, -.176125, 'polynomial_with_count')]
        output.append(item)
    return {'source_study': str(source.resolve()), 'inherited_events': inherited_events,
        'scope': 'Each prefix includes only acquisition records actually returned by then. '
            'Identical full reset actions remain in the same fold. Paid duplicates and costs '
            'are retained. Padding map is the authored three-position map studied earlier '
            'and used in call91; selecting it for this replay is a host diagnostic. '
            'The corrected threshold and feature map are host-selected from paid evidence. '
            'Their grouped coefficient holdout is not validation of the full selection process. '
            'No source workspace, model or old result is changed.', 'prefixes': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'prefixes': [{
        key: prefix[key] for key in ('new_call', 'available_rows', 'paid_trajectories',
                                    'distinct_reset_action_trajectories')}
        for prefix in result['prefixes']]}, indent=2))


if __name__ == '__main__':
    main()

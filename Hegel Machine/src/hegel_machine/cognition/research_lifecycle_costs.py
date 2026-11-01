"""Combine retained execution occurrences; unknown bills stay intervals.

This is a readout of existing ledgers, not a second billing system. An original
execution reference identifies one occurrence even when several returns cite it.
"""

from copy import deepcopy


COSTS = ('compute_cost', 'observation_cost', 'measurement_cost',
         'python_execution_seconds', 'model_calls', 'reported_tokens')


def _interval(record, key):
    value = record.get('costs', {}).get(key)
    if value is not None:
        return [value, value]
    declared = record.get('cost_intervals', {}).get(key)
    return list(declared) if declared is not None else [0, None]


def _sum(intervals):
    rows = list(intervals)
    return [sum(row[0] for row in rows),
            sum(row[1] for row in rows) if all(row[1] is not None for row in rows) else None]


def _difference(right, left):
    return [right[0] - left[1] if left[1] is not None else None,
            right[1] - left[0] if right[1] is not None else None]


def summarize_research_lifecycle(occurrences, *, arms, outcomes=()):
    """Deduplicate actual sources, failures and successors, then compare totals.

    Costs common to both counterfactual policies cancel by their occurrence,
    including an unknown common bill. Absolute bills remain unknown. A failed
    or unfinished target is never removed from the outcome denominator.
    """
    unique, repeated = [], []
    for item in occurrences:
        if item.get('scope') != 'incremental_execution':
            raise ValueError('Cumulative accounts require their original incremental occurrences')
        previous = next((row for row in unique if row['occurrence_ref'] == item['occurrence_ref']), None)
        if previous is not None:
            if previous != item:
                raise ValueError('One original execution has inconsistent costs or attribution')
            repeated.append(deepcopy(item['occurrence_ref']))
        else:
            unique.append(deepcopy(item))
    nested = [row for row in unique if row.get('included_in_ref') and
              any(parent['occurrence_ref'] == row['included_in_ref'] for parent in unique)]
    billable = [row for row in unique if row not in nested]
    for child in nested:
        parent = next(row for row in unique if row['occurrence_ref'] == child['included_in_ref'])
        if not set(child['charged_to']) <= set(parent['charged_to']):
            raise ValueError('A nested bill cannot disappear from an arm that lacks its parent bill')
    totals = {arm: {key: _sum(_interval(row, key) for row in billable if arm in row['charged_to'])
                    for key in COSTS} for arm in arms}
    differences = {}
    for left_index, left in enumerate(arms):
        for right in arms[left_index + 1:]:
            common = [row for row in billable if left in row['charged_to'] and right in row['charged_to']]
            only_left = [row for row in billable if left in row['charged_to'] and row not in common]
            only_right = [row for row in billable if right in row['charged_to'] and row not in common]
            differences[f'{right}_minus_{left}'] = {
                'cost_intervals': {key: _difference(_sum(_interval(row, key) for row in only_right),
                    _sum(_interval(row, key) for row in only_left)) for key in COSTS},
                'cancelled_shared_occurrence_refs': [row['occurrence_ref'] for row in common]}
    by_arm = {arm: [row for row in outcomes if row['arm'] == arm] for arm in arms}
    return {'status': 'retained_execution_account_readout',
        'unique_occurrences': len(unique), 'duplicate_references_ignored': repeated,
        'nested_occurrences_not_recharged': [row['occurrence_ref'] for row in nested],
        'source_occurrence_refs': [row['occurrence_ref'] for row in billable if row['phase'] == 'source'],
        'failed_occurrence_refs': [row['occurrence_ref'] for row in billable if row.get('failed')],
        'cost_intervals': totals, 'paired_differences': differences,
        'actual_experiment_cost_intervals': {key: _sum(_interval(row, key) for row in billable) for key in COSTS},
        'outcomes': {arm: {'registered': len(rows),
            'delivered': sum(bool(row.get('predictor_available')) for row in rows),
            'confirmed': sum(row.get('prediction_pass') is True for row in rows),
            'unmeasured': sum(row.get('prediction_measured') is not True for row in rows)}
            for arm, rows in by_arm.items()},
        'net_utility': None,
        'net_utility_reason': 'Costs and prediction outcomes are separate; no arbitrary exchange rate or saving inferred',
        'amortization_count': {arm: len(rows) for arm, rows in by_arm.items()}}

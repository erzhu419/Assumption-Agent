"""Read the single registered action's already-paid closed prediction vectors.

No evaluation, fitting, laboratory action, or mutation of scientific records.
The later 49 rows are an exposed subset of the same paid 145 rows.
"""

from copy import deepcopy
from math import isfinite

from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.temporal_operator_agent import (
    _closed_prediction_values, _same_closed_prediction_scope,
)


PAIRED_HISTORY_REF = {
    'file': '/mnt/hegel-snapshot-cow/pr3_v7_condition_successor_recovery_20261026/'
            'target/case_002/pr3/method_feedback/use_000004.json',
    'json_path': [],
}


def read_paid_prediction_pair(before_ref, after_ref, workspace, *, expected_rows=145,
                              history_ref=PAIRED_HISTORY_REF):
    """Summarize two full assessment refs from the completed strong comparison.

References name entire paid assessments, not the assessment's result child.
The caller supplies them from paid_score_reuse.execution_basis_ref. Missing
closed vectors remain UNKNOWN; this reader never falls back to prediction.
    """
    receipt = {
        'status': 'UNKNOWN',
        'scope': 'single_action_exposed_paid_history_not_independent_confirmation',
        'prediction_conditioning': 'recorded_controls_and_history_not_closed_loop_rollout',
        'before_assessment_ref': deepcopy(before_ref),
        'after_assessment_ref': deepcopy(after_ref),
        'original_paired_history_ref': deepcopy(history_ref),
        'expected_rows': expected_rows,
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0,
        'additional_scientific_compute': 0, 'additional_Python_execution_seconds': 0.,
        'causal_keep_remove_effect': 'UNTESTED_single_action_before_after',
        'original_mse_threshold': 1e-10,
    }
    if not before_ref or not after_ref:
        return {**receipt, 'reason': 'same_scope_paid_assessment_reference_unavailable'}

    def read(reference):
        return load_source_values([reference], workspace)[0]['value']

    before, after = read(before_ref), read(after_ref)
    if any(value.get('result', {}).get('returncode') != 0
           or value.get('result', {}).get('timeout')
           or not value.get('closed_prediction_state') for value in (before, after)):
        return {**receipt, 'reason': 'complete_closed_paid_prediction_state_unavailable'}
    before_scope, after_scope = read(before['scope_ref']), read(after['scope_ref'])
    if not _same_closed_prediction_scope(before_scope, after_scope):
        return {**receipt, 'reason': 'before_after_paid_prediction_scopes_differ'}
    rows = before_scope['rows']
    if len(rows) != expected_rows:
        return {**receipt, 'reason': 'paid_prediction_scope_has_different_row_count',
                'actual_rows': len(rows)}
    states = []
    for reference in (before_ref, after_ref):
        states.append(_closed_prediction_values(
            {**reference, 'json_path': [*reference.get('json_path', []),
                                       'closed_prediction_state']}, workspace)['predictions'])
    if any(len(values) != len(rows) or any(not isinstance(value, (int, float))
           or not isfinite(value) for value in values) for values in states):
        return {**receipt, 'reason': 'paid_prediction_vector_incomplete_or_nonfinite'}
    prior = read(history_ref)['paired_future']
    first, last = prior['response_row_range']
    if (not prior.get('same_rows') or last - first + 1 != prior['reference_rows']
            or first < 0 or last >= len(rows)):
        return {**receipt, 'reason': 'retained_later_paid_window_is_not_a_scope_subset'}

    def panel(indices):
        losses = [sum((values[index] - rows[index]['target']) ** 2
                      for index in indices) / len(indices) for values in states]
        return {'rows': len(indices), 'before_mse': losses[0], 'after_mse': losses[1],
                'after_minus_before_mse': losses[1] - losses[0],
                'predictions_changed': any(states[0][i] != states[1][i] for i in indices),
                'changed_prediction_rows': sum(states[0][i] != states[1][i] for i in indices),
                'before_within_threshold': losses[0] <= 1e-10,
                'after_within_threshold': losses[1] <= 1e-10}

    receipt.update(status='recorded_from_paid_predictions',
        before_model=before['model'], after_model=after['model'],
        before_scope_ref=deepcopy(before['scope_ref']),
        after_scope_ref=deepcopy(after['scope_ref']),
        all_paid=panel(range(len(rows))),
        retained_later_paid_window={**panel(range(first, last + 1)),
            'response_row_range_inclusive': [first, last],
            'data_role': 'already_exposed_development_subset_not_unseen_holdout',
            'original_paid_comparison_refs': deepcopy(prior['source_refs'])},
        reused_execution_costs={'before': deepcopy(before.get('execution_costs')),
                                'after': deepcopy(after.get('execution_costs'))})
    return receipt

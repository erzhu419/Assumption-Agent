#!/usr/bin/env python3
"""Read-only arithmetic and synthetic provenance checks for the 3806d239 audit.

This does NOT import Hegel, execute predictors, refit models, buy observations,
re-run repository tests, or access external workspaces. Values are transcribed
from committed portable receipts. Synthetic traces only illustrate dependency
identity; they are not additional project scientific evidence.
"""
from __future__ import annotations
import json
from pathlib import Path


def dependency_closure(events: list[dict], root: str) -> list[str]:
    """Illustrative exact-reference traversal; not a proposed production module."""
    by_id = {event['id']: event for event in events}
    if len(by_id) != len(events):
        raise ValueError('Duplicate event identities')
    visited: set[str] = set()
    active: set[str] = set()
    ordered: list[str] = []
    def visit(identifier: str) -> None:
        if identifier not in by_id:
            raise ValueError(f'Unknown dependency: {identifier}')
        if identifier in active:
            raise ValueError('Dependency cycle')
        if identifier in visited:
            return
        active.add(identifier)
        for dependency in by_id[identifier].get('depends_on', []):
            visit(dependency)
        active.remove(identifier)
        visited.add(identifier)
        ordered.append(identifier)
    visit(root)
    return ordered


def main() -> dict:
    n, k = 145, 49
    before = 0.03948039024287745
    after = 0.03920693311707223
    before_subset = 0.01469959787808627
    after_subset = 0.012436476702112634
    rest_before = (n * before - k * before_subset) / (n-k)
    rest_after = (n * after - k * after_subset) / (n-k)
    result = {
        'review_commit': '3806d239b5c53b9b78ac23fcf7fa4c74f16c27ed',
        'scope': 'Read-only arithmetic, no scientific execution',
        'mse_decomposition': {
            'all_145': {'before': before, 'after': after,
                        'relative_reduction_percent': 100*(before-after)/before},
            'exposed_rows_72_to_120': {'rows': k, 'before': before_subset,
                'after': after_subset,
                'relative_reduction_percent': 100*(before_subset-after_subset)/before_subset},
            'other_96': {'before_derived': rest_before, 'after_derived': rest_after,
                'relative_increase_percent': 100*(rest_after-rest_before)/rest_before},
            'interpretation': 'Descriptive same-data decomposition; not a learned scope or causal diagnosis.'
        },
        'completed_unit_accounts': {
            'preparatory_compute': 56517,
            'continuation_compute': 204716,
            'combined_new_compute': 56517 + 204716,
            'known_cumulative': 1195711,
            'retained_unknown_reservation': 773448,
            'original_cap': 5357744,
            'remaining_with_unknown_reserved': 5357744 - 1195711 - 773448,
            'search_estimate': 446856,
            'old_pool_comparison_upper': 4762 * 145,
            'original_finish_reserve': 145 + 96,
            'old_complete_quote': 446856 + 4762*145 + 145 + 96,
            'new_example_only_same_search_estimate_plus182_exports':
                446856 + (4762+182)*145 + 145 + 96,
            'new_example_is_grant_or_full_future_quote': False
        },
        'timing': {
            'combined_active_seconds': 1483.1388041900937,
            'combined_restore_seconds': 1124.6095513510518,
            'restore_fraction_of_recorded_combined_active_percent':
                100*1124.6095513510518/1483.1388041900937,
            'authorized_active_seconds': 1800.0,
            'unused_not_authorization_to_spend': 1800.0-1483.1388041900937,
            'warning': 'Subset/total relation is from this receipt only; do not add overlapping clocks.'
        }
    }
    assert result['completed_unit_accounts']['combined_new_compute'] == 261233
    assert result['completed_unit_accounts']['remaining_with_unknown_reserved'] == 3388585
    assert result['completed_unit_accounts']['old_complete_quote'] == 1137587
    assert rest_after > rest_before and after < before
    # Two calls with the same operation name are not the same fitted instance.
    events = [
        {'id': 'fit:A', 'action': 'fit', 'depends_on': []},
        {'id': 'fit:B', 'action': 'fit', 'depends_on': []},
        {'id': 'score:A', 'action': 'compare', 'depends_on': ['fit:A']},
        {'id': 'adopt:A', 'action': 'adopt', 'depends_on': ['score:A']}
    ]
    actual = dependency_closure(events, 'adopt:A')
    nearest_fit_by_action_name = next(e['id'] for e in reversed(events[:2]) if e['action']=='fit')
    assert actual == ['fit:A', 'score:A', 'adopt:A']
    assert nearest_fit_by_action_name == 'fit:B'
    # A callback consumes an old record, without reacquiring it.
    callback = dependency_closure([
        {'id': 'record:7', 'action': 'query', 'depends_on': []},
        {'id': 'boundary:q2:r9:p3', 'action': 'boundary_update', 'depends_on': ['record:7']},
        {'id': 'fit:new', 'action': 'fit', 'depends_on': ['boundary:q2:r9:p3']}
    ], 'fit:new')
    result['synthetic_dependency_checks'] = {
        'exact_ref_closure': actual,
        'nearest_action_name_would_select': nearest_fit_by_action_name,
        'paid_record_callback_closure': callback,
        'scope': 'Illustration only; not a reproduction of a current production bug.'
    }
    output = Path(__file__).with_name('readonly_diagnostic_results.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result

if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))

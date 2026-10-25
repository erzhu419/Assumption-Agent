"""Finite parent-revision admission using retained preparation and real quotes.

This module does not solve a Program, select a winner or write checkpoints.
Prospective frontier positions describe a plan, never completed research.
"""
from copy import deepcopy


def retained_grammar_preparation(source_request, source_state, *, components,
                                 evidence_checks, fitting, source_ref):
    """Reuse only the complete grammar prefix for these exact saved inputs."""
    def evidence(values):
        return [{key: item.get(key) for key in ('source_ref', 'context_ref', 'criterion')}
                for item in values]

    fields = ('id', 'version', 'source_ref')
    roster = [{key: item.get(key, 1 if key == 'version' else None) for key in fields}
              for item in components]
    original = [{key: item.get(key, 1 if key == 'version' else None) for key in fields}
                for item in source_request['components']]
    if (source_request['fitting'] != fitting or roster != original
            or evidence(source_request['evidence_checks']) != evidence(evidence_checks)):
        raise ValueError('retained grammar inputs differ from the new research contract')
    if (len(source_state.get('structures', [])) < fitting['max_structures']
            or not source_state.get('grammar_cursor')):
        raise ValueError('retained complete finite grammar preparation required')
    return {'source_ref': deepcopy(source_ref), 'state': {
        key: deepcopy(source_state[key]) for key in ('structures', 'grammar_cursor')}}


def symmetric_complete_budget(plans, *, unpaid_scoring_compute=0, readout_compute=2):
    """Return equal caps sufficient for both finite, already quoted plans."""
    totals = {arm: sum(unit['compute'] for unit in plan) + unpaid_scoring_compute + readout_compute
              for arm, plan in plans.items()}
    return {'additional_compute_per_arm': max(totals.values(), default=0),
            'quoted_arm_upper_bounds': totals,
            'initial_unpaid_scoring_compute': unpaid_scoring_compute,
            'final_readout_compute': readout_compute,
            'scope': 'finite_same_parent_complete_unit_plan_not_expected_spend'}


def retained_four_round_progress(plan, actions):
    """Keep consumed bindings even when their comparison/checkpoint failed.

    A completed fit advances its own scientific cursor before the outer runner
    increments its validated count. The next run compares that saved work
    first; it must not repeat the fit or borrow another parent's position.
    """
    consumed, compared, counts = 0, 0, {}
    for action in actions:
        if action['action'] == 'frozen_recall_on_registered_parent_duty':
            index = action['unit_index']
            if index != consumed or index >= len(plan) or action['registered_unit'] != plan[index]:
                raise ValueError('saved work differs from the registered fair prefix')
            visits = [item for item in action.get('feedback', {}).get(
                'complete_research_visits', []) if item.get('status') == 'complete_visit_executed']
            if len(visits) != 1:
                break
            visit, unit = visits[0], plan[index]
            if (visit['candidate_id'] != unit['candidate_id']
                    or visit['search_position'] != unit['search_position']):
                raise ValueError('saved binding differs from its independent parent cursor')
            consumed += 1
            owner = unit['candidate_id']
            counts[owner] = counts.get(owner, 0) + 1
        elif action['action'] == 'research_choice' and not action.get('truncated', False):
            compared = consumed
    return {'consumed_units': consumed, 'compared_units': compared,
        'completed_unit_counts_by_group': counts,
        'comparison_of_committed_work_required': compared < consumed,
        'remaining_units': [{**deepcopy(unit), 'original_unit_index': index}
            for index, unit in enumerate(plan) if index >= consumed]}


def four_round_unit_plan(groups, quotations, *, rounds=4):
    """Keep the existing execution-group order and include skipped preparation.

    ``quotations`` comes from actual transfer preparation plus the shared
    revision-frontier quotation. A rejected parent has no frontier; a rejected
    binding remains a real position in an otherwise applicable frontier.
    """
    plan, skipped, prior_fits = [], [], []
    for round_index in range(rounds):
        for group in groups:
            owner = group[0]
            info = quotations[owner]
            frontier = info.get('frontier', [])
            if round_index >= len(frontier):
                skipped.append({'candidate_id': owner,
                    'compute': info['outer_preparation_compute'] + info['preparation_compute']})
                continue
            unit = frontier[round_index]
            quote = unit.get('quote') or {'compute': 0, 'maximum_outputs': 0}
            family = info['fit_contract']
            growth = sum(item['compute'] for item in prior_fits if item['contract'] == family)
            checks = info['paid_rows'] * quote['maximum_outputs']
            own = (info['outer_preparation_compute'] + info['preparation_compute']
                   + unit.get('quotation_compute', 0) + quote['compute'] + checks + growth)
            plan.append({'round': round_index + 1, 'candidate_id': owner,
                'candidate_ids': list(group), 'search_position': deepcopy(unit['position']),
                'compute': own + sum(item['compute'] for item in skipped),
                'source_skips': skipped, 'quote': deepcopy(quote),
                'preparation_upper_compute': info['preparation_compute'] + unit.get('quotation_compute', 0),
                'outer_preparation_compute': info['outer_preparation_compute'],
                'caller_paid_check_compute': checks,
                'future_fit_query_upper_compute': growth,
                'status': unit['status']})
            skipped = []
            if quote['maximum_outputs']:
                prior_fits.append({'contract': family,
                    # Conditional profiles can each retain a complete
                    # initial/ridge/stable batch, not just the winning one.
                    'compute': unit.get('future_fit_query_compute_bound', 0)
                        * ((quote['maximum_outputs'] + 2) // 3)})
    return plan

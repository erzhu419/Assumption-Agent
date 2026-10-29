"""Publish existing paid object-research segments without running a search.

The input comparison is the current ``_revise_paid_structure`` result (or the
same all-paid comparison before that search). These are native alternatives,
not learned methods or evidence of an advantage over the ordinary author.
"""

from copy import deepcopy


def publish_object_response_options(spec, paid_rows, comparison, *,
                                    compute_remaining, tolerance, data_ref,
                                    parent_ref, pool_ref, comparison_ref,
                                    completed_segments=()):
    """Expose bounded N segments at a measured, unresolved parent decision.

    References bind the actual same-scene inputs. The ordinary executor must
    keep the complete pool, reuse old scores, score every new output, and
    return the selected parent. A later independent confirmation is separate.
    """
    result = {'status': 'unavailable', 'options': [], 'excluded': [],
              'new_compute_cost': 0, 'new_observations': 0,
              'default_operation': None,
              'publication_scope': 'existing_native_response_continuation',
              'method_selection_identified': False,
              'comparison_ref': deepcopy(comparison_ref)}
    action = comparison.get('action', {})
    if (not comparison.get('full_comparison')
            or action.get('observation_rows') != len(paid_rows)):
        return {**result, 'reason': 'current_complete_paid_comparison_required'}
    parent = comparison.get('selected')
    if parent is None:
        return {**result, 'reason': 'selected_parent_required'}
    if parent['validation_loss'] <= tolerance:
        return {**result, 'status': 'already_explained',
                'reason': 'retain_current_parent_without_repeating_research'}
    structure = parent.get('structure') or {}
    fits = structure.get('fits', [])
    if not spec.context.get('object_schema') or not fits or not paid_rows:
        return {**result, 'reason': 'current_object_structure_required'}

    binding = {'data_ref': deepcopy(data_ref), 'parent_ref': deepcopy(parent_ref),
               'pool_ref': deepcopy(pool_ref)}
    completed = {item['operation'] for item in completed_segments
                 if item.get('binding') == binding and item.get('execution_complete')}
    # This is an actual completed relation attempt, including its full-pool
    # comparison. A mere parameter fit or an unfunded search is not that result.
    segment_completed = (action.get('action') == 'revise_paid_structure'
        and action.get('search_compute', 0) > 0
        and comparison.get('status') in {'improved', 'no_improvement'})
    if segment_completed:
        completed.add(action.get('research_action', 'structural_research'))

    proposals = []
    if 'structural_research' not in completed:
        proposals.append(('structural_research', 'propose_structures', 90000, 16,
                          'ordinary_paid_structure_revision'))
    # Match the native response_basis_revision precondition: an actual
    # relation attempt left the response unexplained. No source label, hidden
    # family, guessed term, or target evaluation creates this alternative.
    if ('structural_research' in completed and 'response_basis_revision' not in completed
            and fits[0].get('response_terms')):
        proposals.append(('response_basis_revision', 'propose_response_revisions',
                          32000, 8, 'paid_relation_attempt_left_response_unexplained'))

    # The native continuation has an ordered prerequisite, hence this view
    # currently exposes at most one stage at a time. It is not a two-arm
    # method contrast; the wider ordinary action publisher owns that choice.
    result['default_operation'] = proposals[0][0] if proposals else None
    for operation, function, native_cap, exports, basis in proposals[:2]:
        reserve = len(paid_rows) * exports
        search_limit = min(native_cap, max(0, compute_remaining - reserve))
        if search_limit <= 0:
            result['excluded'].append({'operation': operation,
                'reason': 'new_candidate_comparison_not_funded',
                'new_comparison_reserve': reserve})
            continue
        result['options'].append({'id': f'object_response:{operation}', 'family': 'N',
            'operation': operation, 'basis': basis,
            'role': 'ordinary_native_continuation',
            'content_status': 'existing_native_segment_not_learned',
            'binding': deepcopy(binding), 'comparison_ref': deepcopy(comparison_ref),
            'execution': {'module': 'hegel_machine.cognition.unified_research',
                'class': 'UnifiedResearchAgent', 'method': '_revise_paid_structure',
                'research_action': operation, 'native_proposer': function,
                'arguments_from_current_scene': ['spec', 'parent', 'paid_rows',
                    'finalists', 'transitions', 'paid_scores'],
                'budget_limit': search_limit + reserve},
            'complete_quote': {'status': 'available',
                'paid_rows': len(paid_rows), 'retained_pool_models': len(comparison['claims']),
                'search_compute_limit': search_limit, 'max_exported_models': exports,
                'old_comparison_compute': 0, 'new_comparison_reserve': reserve,
                'compute_cost': search_limit + reserve, 'observation_cost': 0,
                'scope': 'bounded_search_and_full_paid_pool_comparison',
                'independent_confirmation_included': False},
            'return_contract': {'retain_complete_old_pool': True,
                'compare_all_new_outputs': True, 'return_selected_parent': True,
                'gain_label': 'unmeasured_method_effect'}})
    result['status'] = 'available' if result['options'] else 'no_new_legal_segment'
    return result

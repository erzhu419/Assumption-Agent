"""Reconstruct a finite research method from shared, already measured sources.

The archive and fixed-organization arms execute the same proposal algorithms
as Common. Source outcomes justify a proposal, never a successor parameter.
"""

from copy import deepcopy

from .method_action_research import (
    quote_native_partition_analysis, quote_partition_followup_analysis,
    read_method_reference,
)
from .transfer_partitions import propose_transfer_partitions


def reconstruct_partition_source(source, workspace, budget):
    before = read_method_reference(source['before_ref'], workspace)
    after = read_method_reference(source['after_ref'], workspace)
    before = before.get('research_program', before)
    after = after.get('research_program', after)
    required = quote_native_partition_analysis(before, after)
    result = {'source_ref': deepcopy(source['source_ref']), 'methods': [],
        'status': 'compute_unresolved', 'compute_cost': 0, 'new_fits': 0,
        'new_observations': 0, 'cost_scope': 'included_in_parent_search_budget'}
    if not budget.spend(required):
        return result
    result['compute_cost'] += required
    proposal = propose_transfer_partitions(before, after, source_ref=source['source_ref'],
        before_ref=source['before_ref'], after_ref=source['after_ref'])
    bodies = [item['research_program'] for item in proposal['candidates']]
    if source.get('study_ref'):
        from .method_action_research import partition_followup_body
        study = read_method_reference(source['study_ref'], workspace)
        # The study indices refer to the original executed proposal. Comparing
        # bodies prevents a later proposal from borrowing an earlier result.
        executed = read_method_reference(source['proposal_ref'], workspace)['proposal']
        if any(not any(item['research_program'] == candidate['research_program']
                       for candidate in proposal['candidates']) for item in executed['candidates']):
            return {**result, 'status': 'source_proposal_changed'}
        for eligible in quote_partition_followup_analysis(executed, study):
            if not budget.spend(eligible['required_compute']):
                break
            result['compute_cost'] += eligible['required_compute']
            body = executed['candidates'][eligible['method_index'] - 1]['research_program']
            bodies.insert(0, partition_followup_body(body))
    bodies.append(proposal['baseline_research_program'])
    result['methods'] = [{'source_concept_id': source['source_concept_id'],
        'source_ref': deepcopy(source['source_ref']), 'research_program': body,
        **({'source_model_ref': deepcopy(source['source_model_ref'])} if source.get('source_model_ref') else {})}
        for body in bodies]
    result['status'] = 'reconstructed' if bodies else proposal['status']
    return result

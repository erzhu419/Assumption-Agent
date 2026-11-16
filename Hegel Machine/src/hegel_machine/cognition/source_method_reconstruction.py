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


def _reconstruct_composition(source, workspace, budget):
    """Invoke the ordinary proposer on the same actual sequential source records."""
    from pathlib import Path
    from .common_object_research import CommonObjectDependencyMethods
    from .method_feedback import ensure_python_method_topic
    from .models import SearchBudget
    from .staged_method_research import propose_native_method_compositions

    result = {'status': 'composition_dependency_unresolved', 'methods': [], 'compute_cost': 0}
    execution_ref, pair_ref = (source.get(name) for name in (
        'composition_execution_return_ref', 'composition_baseline_pair_ref'))
    if not execution_ref or not pair_ref:
        return {**result, 'reason': 'actual_sequential_return_and_prefix_invocation_required'}
    try:
        execution = read_method_reference(execution_ref, workspace)
        pair = read_method_reference(pair_ref, workspace)
        fields = ('executed_method_ref', 'method_execution_ref', 'execution_complete', 'execution_status')
        if any(name not in execution for name in fields):
            return {**result, 'reason': 'complete_native_execution_return_required'}
        group = {'pair_refs': [deepcopy(pair_ref)]}
        for side, key in (('baseline', 'baseline_body'), ('challenger', 'body')):
            if not pair.get(side + '_ref'):
                return {**result, 'reason': 'actual_prefix_invocation_return_missing'}
            returned = read_method_reference(pair[side + '_ref'], workspace)
            if not returned.get('method_artifact_ref'):
                return {**result, 'reason': 'actual_prefix_method_body_required'}
            artifact = read_method_reference(returned['method_artifact_ref'], workspace)
            declaration = artifact.get('execution', {})
            if declaration.get('entry') != 'replay_research_program' or not declaration.get('research_program_ref'):
                return {**result, 'reason': 'actual_prefix_method_body_required'}
            group[key] = {'execution': {'research_program': read_method_reference(
                declaration['research_program_ref'], workspace)}}
    except FileNotFoundError as error:
        return {**result, 'reason': 'missing_declared_source_record', 'missing_file': str(error.filename)}
    directory = Path(workspace) / 'source_composition' / source['source_concept_id']
    common = CommonObjectDependencyMethods(directory, [], budget=SearchBudget(budget.remaining), actions=[]).common
    question = ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'})
    question.setdefault('method_use_plan', {})
    try:
        proposals = propose_native_method_compositions(common, question,
            [{key: deepcopy(execution[key]) for key in fields} | {'return_ref': deepcopy(execution_ref)}],
            [group], budget.remaining)
    finally:
        if not budget.spend(common.costs['compute_cost']):
            raise RuntimeError('Source composition analysis exceeded the parent allowance')
    result.update(compute_cost=common.costs['compute_cost'], proposal_receipts=proposals)
    formed = [item for item in proposals if item.get('status') == 'staged_program_proposed']
    if len(formed) != 1:
        return {**result, 'reason': (proposals[0].get('reason', proposals[0]['status'])
                                   if proposals else 'no_actual_composition_return')}
    body = read_method_reference(formed[0]['research_program_refs'][0], directory)
    result.update(status='reconstructed', proposal_ref=formed[0]['proposal_ref'], methods=[body,
        body['staged_research']['baseline'], body['staged_research']['continuation']])
    return result


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
    if source.get('composition_execution_return_ref') or source.get('composition_baseline_pair_ref'):
        composed = _reconstruct_composition(source, workspace, budget)
        result['composition_reconstruction'] = composed
        result['compute_cost'] += composed['compute_cost']
        if composed['status'] != 'reconstructed':
            # The retained old bank remains usable, but it is not an equal
            # replacement for the unformed sequential method.
            result.update(status=composed['status'], reason=composed['reason'])
        unique = []
        for body in [*composed['methods'], *bodies]:
            if body not in unique:
                unique.append(body)
        bodies = unique
    result['methods'] = [{'source_concept_id': source['source_concept_id'],
        'source_ref': deepcopy(source['source_ref']), 'research_program': body,
        **({'source_model_ref': deepcopy(source['source_model_ref'])} if source.get('source_model_ref') else {})}
        for body in bodies]
    if 'composition_reconstruction' not in result or result['composition_reconstruction']['status'] == 'reconstructed':
        result['status'] = 'reconstructed' if bodies else proposal['status']
    return result

"""Connect addressed method experience to finite ordinary action revisions."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path

from .checkpoint_storage import _JsonSource, read_json_checkpoint


def read_method_reference(reference, workspace):
    """Read one declared value, leaving the surrounding archive file-backed."""
    path = Path(workspace) / reference['file']
    with ExitStack() as opened:
        source, span = _JsonSource(path, opened), None
        for part in reference.get('json_path', []):
            span = source.members(span, keys={part})[part]
        if span is None:
            # Standalone feedback/profile contracts are already one input.
            # Large research archives are addressed by the caller's json_path.
            return read_json_checkpoint(path)
        return json.loads(source.text[span[1]:span[2]])


def _topic_method_candidates(agent, question):
    owners = [question]
    topic = question.get('origin_method_question_ref')
    if topic:
        owners.extend(item for item in agent.questions
            if item['id'] == topic['question_id']
            and item.get('meaning_version', 1) == topic.get('meaning_version', 1))
    library, methods = agent.theory_consolidation.get('candidates', {}), []
    for owner in owners:
        references = list(owner.get('candidate_method_refs', []))
        identifier = owner.get('method_use_plan', {}).get('candidate_id')
        if identifier in library:
            references.append({'id': identifier, 'version': library[identifier]['version']})
        for reference in references:
            candidate = library.get(reference['id'])
            if (candidate is not None and candidate.get('version') == reference.get('version')
                    and candidate.get('selection_plan', {}).get('condition') is not None
                    and all(item['id'] != candidate['id'] for item in methods)):
                methods.append(candidate)
    return methods


def _native_method_pair_groups(agent, records):
    """Keep the measured challenger attached to its actual frozen native body."""
    fields = ('id', 'version', 'kind', 'execution', 'selection_plan',
              'ordinary_source_ref', 'source', 'source_ref')
    def content(artifact):
        body = {key: deepcopy(artifact.get(key)) for key in fields}
        execution = body.get('execution') or {}
        if execution.get('entry') == 'replay_research_program' and execution.get('research_program_ref'):
            execution['research_program'] = read_method_reference(
                execution.pop('research_program_ref'), agent.workspace)
        return body
    groups, excluded = [], []
    for reference, record in records:
        pair_ref = record.get('method_pair_ref')
        if pair_ref is None:
            continue
        method_ref, artifact_ref = record.get('method_ref'), record.get('method_artifact_ref')
        reason = None
        if method_ref is None or artifact_ref is None:
            reason = 'actual_native_method_body_required'
        else:
            artifact = read_method_reference(artifact_ref, agent.workspace)
            pair = read_method_reference(pair_ref, agent.workspace)
            challenger = read_method_reference(pair['challenger_ref'], agent.workspace)
            measured_ref = challenger.get('method_artifact_ref')
            if ({key: artifact.get(key) for key in ('id', 'version')} != method_ref
                    or challenger.get('method_ref') != method_ref):
                reason = 'paired_native_body_identity_mismatch'
            elif measured_ref is None:
                reason = 'actual_paired_native_body_required'
            else:
                measured = read_method_reference(measured_ref, agent.workspace)
                if content(measured) != content(artifact):
                    reason = 'paired_native_body_content_mismatch'
            baseline = read_method_reference(pair['baseline_ref'], agent.workspace)
            baseline_artifact_ref = baseline.get('method_artifact_ref')
            if reason is None:
                if baseline_artifact_ref is None or baseline.get('method_ref') is None:
                    reason = 'actual_baseline_method_body_required'
                else:
                    baseline_artifact = read_method_reference(baseline_artifact_ref, agent.workspace)
                    if {key: baseline_artifact.get(key) for key in ('id', 'version')} != baseline['method_ref']:
                        reason = 'paired_baseline_body_identity_mismatch'
        if reason is not None:
            excluded.append({'feedback_ref': deepcopy(reference),
                'pair_ref': deepcopy(pair_ref), 'reason': reason})
            continue
        previous = next((item for item in groups if item['method_ref'] == method_ref), None)
        body = content(artifact)
        if previous is not None and previous['body'] != body:
            excluded.append({'feedback_ref': deepcopy(reference),
                'pair_ref': deepcopy(pair_ref), 'reason': 'native_method_version_changed_body'})
            continue
        baseline_ref = baseline['method_ref']
        baseline_body = content(baseline_artifact)
        previous_baseline = next((item for item in groups if item['baseline_method_ref'] == baseline_ref), None)
        if previous_baseline is not None and previous_baseline['baseline_body'] != baseline_body:
            excluded.append({'feedback_ref': deepcopy(reference),
                'pair_ref': deepcopy(pair_ref), 'reason': 'native_baseline_version_changed_body'})
            continue
        group = next((item for item in groups if item['method_ref'] == method_ref
                      and item['baseline_method_ref'] == baseline_ref), None)
        if group is None:
            group = {'method_ref': deepcopy(method_ref), 'method_artifact_ref': deepcopy(artifact_ref),
                     'body': body, 'baseline_method_ref': deepcopy(baseline_ref),
                     'baseline_method_artifact_ref': deepcopy(baseline_artifact_ref),
                     'baseline_body': baseline_body, 'pair_refs': []}
            groups.append(group)
        if pair_ref not in group['pair_refs']:
            group['pair_refs'].append(deepcopy(pair_ref))
    return groups, excluded


def quote_native_partition_analysis(before, after):
    """Bound the two contract reconstructions and finite template matching.

    This is a node-visit allowance for proposal analysis, not a model fit or a
    world readout. Complete columns, saved parameter templates and graph output
    are included even when a fitted coefficient is zero.
    """
    def nodes(value):
        return 1 + sum(nodes(child) for child in value.get('children', []))

    size = 0
    for payload in (before, after):
        graph = payload['graph']['nodes']
        size += sum(nodes(graph[key]['answer'])
                    for key in [*payload['term_nodes'], payload['output_node']])
        for step in payload.get('selection_steps', []):
            if step.get('operation') == 'resolve_relation_intervals':
                size += sum(nodes(term) for term in step.get('parameter_terms', []))
                if step.get('parameter_template'):
                    size += nodes(step['parameter_template'])
    # Role-revision compilation enumerates at most 48 typed assignments and
    # reconstructs the complete saved columns for each. Quote that work before
    # the existing native-return route publishes a proposal.
    template, schema = before.get('input_template'), after.get('source_schema') or before.get('source_schema')
    from .transfer_partitions import _observed_feature_rebinding
    from .structural_research_program import research_program_fit_contract
    if (template and schema and set(template.get('object_roles', {})) == set(template['roles'])
            and _observed_feature_rebinding(research_program_fit_contract(before)['fit_terms'],
                                             research_program_fit_contract(after)['fit_terms'])):
        from math import perm
        objects = len(schema['objects'])
        roles = len({value[0] for value in template['object_roles'].values()})
        assignments = min(48, 1 + perm(objects, roles)) if roles <= objects else 1
        size += assignments * size
    return 2 + 3 * size


def quote_existing_interval_return(before, after):
    """Quote checking a fixed-readout execution, without new edits.

    The two retained fit contracts are visited once: all columns, the output
    answer, the readout expression, and the parameterized columns. The two
    addressed execution/method records each cost one unit. This excludes the
    template search and role-assignment passes of new proposal construction.
    The caller first recognizes the actual execution contract; a declaration
    alone does not qualify for this narrower work scope.
    """
    def nodes(value):
        return 1 + sum(nodes(child) for child in value.get('children', []))

    required = 2
    for body in (before, after):
        graph = body['graph']['nodes']
        required += sum(nodes(graph[key]['answer']) for key in body['term_nodes'])
        output = graph[body['output_node']]
        required += nodes(output['answer']) + nodes(output['expression'])
        binding = next((step for step in reversed(body.get('selection_steps', []))
            if step.get('operation') == 'resolve_relation_intervals'), {})
        required += sum(nodes(term) for term in binding.get('parameter_terms', []))
    return required


def record_native_partition_study(agent, question, *, proposal_ref, study_ref,
        comparison_ref=None, cost_refs=(), compute_limit):
    """Return an existing matched study to its method question without refitting."""
    plan = question['method_use_plan']
    if proposal_ref not in [*plan.get('transfer_partition_proposal_refs', []),
                           *plan.get('staged_method_proposal_refs', [])]:
        raise ValueError('The study must address a proposal owned by this method question')
    basis = {'proposal_ref': deepcopy(proposal_ref), 'study_ref': deepcopy(study_ref),
             'comparison_ref': deepcopy(comparison_ref)}
    references = plan.get('transfer_partition_return_refs', [])
    for reference in references:
        retained = read_method_reference(reference, agent.workspace)
        if retained['basis'] == basis:
            return {**retained, 'return_ref': deepcopy(reference),
                    'cache_hit': True, 'new_compute_cost': 0}
    bundle = read_method_reference(proposal_ref, agent.workspace)
    study = read_method_reference(study_ref, agent.workspace)
    if study['proposal'] != bundle['proposal']:
        raise ValueError('The measured study does not use the addressed executable proposal')
    methods, pairs = study.get('methods', []), study.get('pairs', [])
    required = 2 + len(methods) + len(pairs) + int(comparison_ref is not None)
    if compute_limit < required:
        return {'status': 'scoped_unknown', 'reason': 'partition_return_not_funded',
                'required_compute': required, 'new_compute_cost': 0}
    agent.costs['compute_cost'] += required
    threshold = study['basis']['minimum_improvement']
    method_count = 1 + len(bundle['proposal']['candidates'])
    complete = (not study.get('budget_incomplete') and not study.get('time_incomplete')
        and {item['index'] for item in methods} == set(range(method_count))
        and all(item.get('execution_complete') for item in methods)
        and {item['method_index'] for item in pairs} == set(range(1, method_count))
        and all(item.get('execution_complete') for item in pairs))
    outcomes = []
    for position, pair in enumerate(pairs):
        method_index = pair['method_index']
        indexed = next(((index, item) for index, item in enumerate(methods)
                        if item['index'] == method_index), None)
        if indexed is None:
            continue
        method_position, method = indexed
        measured = method.get('validation_loss')
        passed = measured <= threshold if method.get('execution_complete') and measured is not None else None
        outcomes.append({'method_index': method_index,
            'method_ref': {**deepcopy(study_ref), 'json_path': [*study_ref.get('json_path', []),
                'methods', method_position]},
            'pair_ref': {**deepcopy(study_ref), 'json_path': [*study_ref.get('json_path', []), 'pairs', position]},
            'validation_loss': measured, 'validation_loss_change': pair.get('validation_loss_change'),
            'validation_threshold': threshold, 'validation_passed': passed,
            'paired_status': pair['status'], 'execution_complete': pair.get('execution_complete', False)})
    improved = complete and any(item['paired_status'] == 'local_step_improvement' for item in outcomes)
    comparison = read_method_reference(comparison_ref, agent.workspace) if comparison_ref else {}
    result = {'status': ('local_development_improvement' if improved else
                         'complete_no_supported_improvement' if complete else 'incomplete_not_negative'),
        'basis': basis, 'source_task_ref': deepcopy(bundle['source_task_ref']),
        'execution_complete': complete, 'method_outcomes': outcomes,
        'validation_basis_ref': {**deepcopy(study_ref),
            'json_path': [*study_ref.get('json_path', []), 'basis']},
        'remaining_validation_residual': any(item['validation_passed'] is False for item in outcomes),
        'full_pool_comparison': {key: deepcopy(comparison[key]) for key in (
            'full_comparison', 'parent_changed', 'selected_ref', 'confirmed_parent_ref', 'paid_rows')
            if key in comparison},
        'counterexample_refs': deepcopy(study.get('counterexample_refs', [])),
        'cost_refs': [deepcopy(study_ref), *deepcopy(list(cost_refs))],
        'analysis_compute_cost': required, 'data_role': 'exposed_development',
        'independent_support': False, 'source_support_inherited': False,
        'installed_candidate': None, 'new_observations': 0, 'new_fits': 0,
        'next_action': None}
    directory, index = agent.workspace / 'method_feedback', 0
    while (directory / f'transfer_partition_return_{index:06d}.json').exists():
        index += 1
    path = f'method_feedback/transfer_partition_return_{index:06d}.json'
    agent._save(path, json.dumps(result, separators=(',', ':')))
    reference = {'file': str(agent.workspace / path), 'json_path': []}
    plan.setdefault('transfer_partition_return_refs', []).append(reference)
    question.update(status='open', agenda_basis=deepcopy(plan))
    question.pop('completed_agenda_basis', None)
    agent._save('questions.json', json.dumps(agent.questions, separators=(',', ':')))
    return {**result, 'return_ref': reference, 'cache_hit': False, 'new_compute_cost': required}


def quote_partition_followup_analysis(proposal, study):
    """Quote only followups supported by a completed fit and its actual return."""
    if study.get('budget_incomplete') or study.get('time_incomplete'):
        return []
    threshold = study['basis']['minimum_improvement']
    eligible = []
    for index, source in enumerate(proposal['candidates'], 1):
        body = source['research_program']
        partition = body.get('transfer_partition', {})
        if partition.get('kind') != 'Pi1' or partition.get('second_stage'):
            continue
        measured = next((item for item in study.get('methods', []) if item['index'] == index), None)
        if not measured or not measured.get('execution_complete'):
            continue
        training_loss, returned_loss = measured.get('training_loss'), measured.get('validation_loss')
        if (training_loss is None or returned_loss is None
                or training_loss > threshold or returned_loss <= threshold):
            continue
        fitted = measured.get('research_program') or {}
        binding = next((step for step in reversed(fitted.get('selection_steps', []))
                        if step['operation'] == 'resolve_relation_intervals'), None)
        if not binding or not binding.get('parameter_terms'):
            continue
        symbols = {item['symbol'] for item in binding.get('parameters', [])}
        if not set(partition.get('released_groups', [])).issubset(symbols):
            continue
        eligible.append({'method_index': index,
            'required_compute': quote_native_partition_analysis(body, fitted)})
    return eligible


def partition_followup_body(baseline):
    """The same paid-return branch is available to Common and archive authors."""
    revised = deepcopy(baseline)
    revised['transfer_partition']['second_stage'] = {
        'operation': 'resolve_relation_intervals',
        'trigger': 'fitted_training_with_unexplained_return',
        'evidence_role': 'paid_return_after_first_fit', 'compute_limit': 15000}
    return revised


def _propose_paid_partition_followups(agent, question, proposals, retained, returned,
                                     result, compute_limit):
    """Use a real failed post-fit return to propose one second scientific stage."""
    references = question['method_use_plan'].get('transfer_partition_proposal_refs', [])
    for parent in list(proposals):
        parent_ref = parent['proposal_ref']
        parent_return = next(((ref, value) for ref, value in reversed(returned)
                              if value['basis']['proposal_ref'] == parent_ref), None)
        if not parent_return or not parent_return[1].get('execution_complete'):
            continue
        return_ref, observed = parent_return
        if not observed.get('remaining_validation_residual'):
            continue
        parent_bundle = next(value for ref, value in retained if ref == parent_ref)
        if parent_bundle.get('followup_of'):
            continue
        study_ref = observed['basis']['study_ref']
        cached = [value for _ref, value in retained
                  if value.get('followup_of') == parent_ref
                  and value['basis']['study_return_ref'] == return_ref]
        if cached:
            choices = [{'method_index': value['basis']['method_index'], 'required_compute': 0}
                       for value in cached]
        else:
            study = read_method_reference(study_ref, agent.workspace)
            choices = quote_partition_followup_analysis(parent_bundle['proposal'], study)
        for eligible in choices:
            method_index = eligible['method_index']
            basis = {'parent_proposal_ref': deepcopy(parent_ref),
                'study_return_ref': deepcopy(return_ref), 'method_index': method_index}
            saved = next(((ref, value) for ref, value in retained
                          if value.get('followup_of') == parent_ref and value['basis'] == basis), None)
            if saved is None:
                required = eligible['required_compute']
                if compute_limit - result['new_compute_cost'] < required:
                    result.update(status='scoped_unknown', reason='partition_followup_analysis_not_funded',
                        required_compute=result['new_compute_cost'] + required)
                    return
                agent.costs['compute_cost'] += required
                result['new_compute_cost'] += required
                # Preserve the proposed source family. Fitted target numbers
                # remain only in the addressed execution, never this method.
                baseline = deepcopy(parent_bundle['proposal']['candidates'][method_index - 1]['research_program'])
                revised = partition_followup_body(baseline)
                second = revised['transfer_partition']['second_stage']
                candidate = {'research_program': revised, 'status': 'requires_paired_evaluation',
                    'edit': {'kind': 'append_result_branch', 'operation': 'resolve_relation_intervals',
                        'trigger': second['trigger']},
                    'source_refs': [deepcopy(study_ref), deepcopy(return_ref)]}
                bundle = {'status': 'pending_validation', 'basis': basis,
                    'followup_of': deepcopy(parent_ref),
                    'source_task_ref': deepcopy(parent_bundle['source_task_ref']),
                    'source_refs': [deepcopy(parent_ref), deepcopy(study_ref), deepcopy(return_ref)],
                    'source_operation': 'completed_fit_with_actual_return_counterexample',
                    'proposal': {'status': 'proposed', 'baseline_research_program': baseline,
                        'candidates': [candidate]},
                    'analysis_compute_cost': required, 'data_role': 'exposed_development',
                    'independent_support': False, 'source_support_inherited': False,
                    'paired_effect': None, 'installed_candidate': None,
                    'required_work': 'resume_paid_first_segment_then_compare_second_segment'}
                directory, serial = agent.workspace / 'method_feedback', 0
                while (directory / f'transfer_partition_{serial:06d}.json').exists():
                    serial += 1
                path = f'method_feedback/transfer_partition_{serial:06d}.json'
                agent._save(path, json.dumps(bundle, separators=(',', ':')))
                reference = {'file': str(agent.workspace / path), 'json_path': []}
                references.append(reference)
                question['method_use_plan']['transfer_partition_proposal_refs'] = references
                retained.append((reference, bundle))
                agent._save('questions.json', json.dumps(agent.questions, separators=(',', ':')))
                saved = reference, bundle
            reference, bundle = saved
            published = {'proposal_ref': deepcopy(reference), 'status': bundle['status'],
                'followup_of': deepcopy(parent_ref), 'source_task_ref': deepcopy(bundle['source_task_ref']),
                'candidate_refs': [{**deepcopy(reference), 'json_path': ['proposal', 'candidates', 0]}],
                'baseline_ref': {**deepcopy(reference), 'json_path': ['proposal', 'baseline_research_program']}}
            actual = next(((ref, value) for ref, value in reversed(returned)
                           if value['basis']['proposal_ref'] == reference), None)
            if actual:
                ref, value = actual
                published.update(status=value['status'], study_return_ref=deepcopy(ref),
                    execution_complete=value['execution_complete'],
                    remaining_validation_residual=value['remaining_validation_residual'],
                    full_pool_comparison=deepcopy(value['full_pool_comparison']))
            proposals.append(published)


def _existing_parent_parameter_return(evidence, method, execution, parent, after, workspace):
    """Keep the actual rebound method, including an unfinished second stage."""
    from .object_continuation import decode_entry
    from .transfer_partitions import prepare_parent_parameter_revision, transfer_partition_view

    binding = method.get('parent_parameter_revision', {})
    research = execution.get('research', {})
    parent_ref = evidence.get('research_parent_ref')
    source_ref = binding.get('source_method_ref')
    if (parent is None or not source_ref or not parent_ref
            or execution.get('action') != 'revise_paid_structure'
            or execution.get('research_action') != 'relation_parameter_revision'
            or binding.get('parent_ref') != parent_ref
            or research.get('parent_ref') != parent_ref
            or research.get('source_method_ref') != source_ref
            or after.get('parent_parameter_revision') != binding):
        return None
    source = read_method_reference(source_ref, workspace)
    expected = prepare_parent_parameter_revision(decode_entry(parent), source,
        source_method_ref=source_ref, parent_ref=parent_ref)
    if expected.get('research_program') != method:
        return None
    # Validate both saved fit families; a returned column loss must not be
    # mistaken for execution of the preserved complete parent family.
    before_view, after_view = transfer_partition_view(method), transfer_partition_view(after)
    fields = ('kind', 'max_attempts', 'max_candidates', 'second_stage')
    if (before_view['status'] != 'ready' or after_view['status'] != 'ready'
            or len(before_view['term_nodes']) != len(after_view['term_nodes'])
            or any(after.get('transfer_partition', {}).get(key) != method['transfer_partition'].get(key)
                   for key in fields)
            or not research.get('search')
            or research['search'] != after.get('transfer_partition_search')
            or research.get('followup', {}) != after.get('transfer_partition_followup', {})
            or execution.get('search_complete') != research.get('execution_complete')):
        return None
    return {'executed_method_ref': deepcopy(evidence['executed_method_ref']),
        'method_execution_ref': deepcopy(evidence['method_execution_ref']),
        'execution_complete': bool(research.get('execution_complete') and execution.get('full_comparison')),
        'execution_status': research['status'],
        'proposal': {'status': 'existing_method_execution',
            'baseline_research_program': deepcopy(method), 'candidates': []}}


def _existing_method_execution_return(evidence, before, after, workspace):
    """Address actual existing research without reconstructing a narrower edit."""
    method_ref = evidence.get('executed_method_ref')
    execution_ref = evidence.get('method_execution_ref')
    if method_ref is None or execution_ref is None:
        return None
    method = read_method_reference(method_ref, workspace)
    execution = read_method_reference(execution_ref, workspace)
    parent = None
    if evidence.get('research_parent_ref'):
        parent = read_method_reference(evidence['research_parent_ref'], workspace)
        before = parent['research_program']
    if method.get('parent_parameter_revision') and execution.get('action') == 'revise_paid_structure':
        return _existing_parent_parameter_return(evidence, method, execution, parent, after, workspace)
    contract = method.get('transfer_partition', {})
    if (method != before or not execution.get('fixed_readout_only')
            or execution.get('new_fits') != 0
            or contract.get('second_stage', {}).get('operation') != 'resolve_relation_intervals'
            or after.get('transfer_partition') != contract):
        return None
    from .structural_research_program import research_program_fit_contract
    original, returned = research_program_fit_contract(before), research_program_fit_contract(after)
    bindings = [next((step for step in reversed(body.get('selection_steps', []))
        if step.get('operation') == 'resolve_relation_intervals'), {}) for body in (before, after)]
    if (original['coefficients'] != returned['coefficients']
            or not bindings[0].get('parameter_terms')
            or bindings[0]['parameter_terms'] != bindings[1].get('parameter_terms')):
        return None
    return {'executed_method_ref': deepcopy(method_ref),
        'method_execution_ref': deepcopy(execution_ref),
        'execution_complete': execution.get('execution_complete', False),
        'execution_status': execution['status'],
        'review_compute_cost': quote_existing_interval_return(before, after),
        'proposal': {'status': 'existing_method_execution',
            'baseline_research_program': deepcopy(method), 'candidates': []}}


def _propose_native_transfer_partitions(agent, question, records, result, compute_limit):
    """Turn addressed before/after research into a saved, unmeasured method.

    Native returns are not scalar method-use feedback. Keep their actual source
    identity and create no relative-effect label or independent-source support.
    """
    from .transfer_partitions import propose_transfer_partitions

    references = question['method_use_plan'].get('transfer_partition_proposal_refs', [])
    retained = [(reference, read_method_reference(reference, agent.workspace))
                for reference in references]
    additions = question['method_use_plan'].get('native_revision_evidence', [])
    returned = [(reference, read_method_reference(reference, agent.workspace))
                for reference in question['method_use_plan'].get('transfer_partition_return_refs', [])]
    proposals, seen = [], []
    for feedback_ref, record in records:
        evidence_refs = ([record['revision_evidence_ref']]
                         if record.get('revision_evidence_ref') else [])
        evidence_refs.extend(item['revision_evidence_ref'] for item in additions
                             if item['feedback_ref'] == feedback_ref)
        for evidence_ref in evidence_refs:
            basis = {'actual_result_ref': record['actual_result_ref'],
                     'revision_evidence_ref': evidence_ref}
            if basis in seen:
                continue
            seen.append(basis)
            saved = next(((reference, bundle) for reference, bundle in retained
                          if bundle['basis'] == basis), None)
            if saved is None:
                evidence = read_method_reference(evidence_ref, agent.workspace)
                # Resumed returns previously republished equal evidence under a
                # new address. Reuse the paid analysis of the same actual run.
                saved = next(((reference, bundle) for reference, bundle in retained
                    if bundle['basis'].get('actual_result_ref') == record['actual_result_ref']
                    and read_method_reference(bundle['basis']['revision_evidence_ref'],
                        agent.workspace) == evidence), None)
            if saved is None:
                if evidence['source_task_ref'] != record['source_task_ref']:
                    raise ValueError('Revision evidence must retain its actual native source')
                before = read_method_reference(evidence['before_entry_ref'], agent.workspace)['research_program']
                after = read_method_reference(evidence['after_entry_ref'], agent.workspace)['research_program']
                # Read-only contract recognition determines the work to quote,
                # just as the ordinary quote inspects the saved fit families.
                # Do not reconstruct a proposal before discovering that the
                # return is an already-addressed method execution.
                executed = _existing_method_execution_return(evidence, before, after, agent.workspace)
                execution_quote = executed.get('review_compute_cost') if executed else None
                required = (execution_quote if execution_quote is not None else
                    quote_native_partition_analysis(before, after))
                if compute_limit - result['new_compute_cost'] < required:
                    result.update(status='scoped_unknown',
                        reason=('existing_method_return_check_not_funded' if execution_quote is not None
                            else 'native_partition_analysis_not_funded'),
                        required_compute=result['new_compute_cost'] + required)
                    return
                agent.costs['compute_cost'] += required
                result['new_compute_cost'] += required
                proposal = (executed['proposal'] if executed else
                    propose_transfer_partitions(before, after,
                        source_ref=evidence_ref, before_ref=evidence['before_entry_ref'],
                        after_ref=evidence['after_entry_ref']))
                bundle = {'status': ('existing_method_execution_returned' if executed else
                                    'pending_validation' if proposal['candidates'] else
                                    'complete_no_partition_proposal'),
                    'basis': deepcopy(basis), 'feedback_ref': deepcopy(feedback_ref),
                    'source_task_ref': deepcopy(record['source_task_ref']),
                    'source_refs': [deepcopy(evidence_ref), deepcopy(record['actual_result_ref'])],
                    'source_operation': evidence.get('source_operation', 'unspecified'),
                    'comparison_ref': deepcopy(evidence.get('comparison_ref')),
                    'cost_refs': deepcopy(evidence.get('cost_refs', record.get('cost_refs', []))),
                    'data_role': evidence.get('data_role', 'exposed_development'),
                    'proposal': proposal, 'analysis_compute_cost': required,
                    'source_support_inherited': False, 'independent_support': False,
                    'paired_effect': None, 'installed_candidate': None,
                    'required_work': (None if executed else
                                      'same_start_same_pool_matched_partition_evaluation')}
                if executed:
                    bundle.update({key: value for key, value in executed.items() if key != 'proposal'})
                directory = agent.workspace / 'method_feedback'
                index = 0
                while (directory / f'transfer_partition_{index:06d}.json').exists():
                    index += 1
                path = f'method_feedback/transfer_partition_{index:06d}.json'
                agent._save(path, json.dumps(bundle, separators=(',', ':')))
                reference = {'file': str(agent.workspace / path), 'json_path': []}
                references.append(reference)
                question['method_use_plan']['transfer_partition_proposal_refs'] = references
                retained.append((reference, bundle))
                # The small addressed proposal survives interruption before
                # the enclosing route returns; its paid compilation is reused.
                agent._save('questions.json', json.dumps(agent.questions, separators=(',', ':')))
                saved = reference, bundle
            reference, bundle = saved
            if any(item['proposal_ref'] == reference for item in proposals):
                continue
            proposals.append({'proposal_ref': deepcopy(reference), 'status': bundle['status'],
                'candidate_refs': [{**deepcopy(reference), 'json_path': ['proposal', 'candidates', index]}
                    for index in range(len(bundle['proposal']['candidates']))],
                'baseline_ref': {**deepcopy(reference),
                    'json_path': ['proposal', 'baseline_research_program']},
                'source_task_ref': deepcopy(bundle['source_task_ref'])})
            if bundle.get('executed_method_ref'):
                result.setdefault('existing_method_executions', []).append({
                    'return_ref': deepcopy(reference),
                    'source_task_ref': deepcopy(bundle['source_task_ref']),
                    **{key: deepcopy(bundle[key]) for key in ('executed_method_ref',
                        'method_execution_ref', 'execution_complete', 'execution_status')}})
            latest = next(((ref, value) for ref, value in reversed(returned)
                           if value['basis']['proposal_ref'] == reference), None)
            if latest:
                ref, value = latest
                proposals[-1].update(status=value['status'], study_return_ref=deepcopy(ref),
                    execution_complete=value['execution_complete'],
                    remaining_validation_residual=value['remaining_validation_residual'],
                    full_pool_comparison=deepcopy(value['full_pool_comparison']))
    _propose_paid_partition_followups(agent, question, proposals, retained, returned, result, compute_limit)
    if not proposals:
        return
    result['transfer_partition_proposals'] = proposals
    if result['status'] != 'scoped_unknown':
        if any(item['status'] == 'pending_validation' for item in proposals):
            result.update(status='method_content_proposed_pending_validation',
                reason='actual_native_revision_produced_unmeasured_partition_content',
                required_work='same_start_same_pool_matched_partition_evaluation')
        elif any(item['status'] == 'incomplete_not_negative' for item in proposals):
            result.update(status='scoped_unknown', reason='partition_development_incomplete',
                required_work='resolve_retained_incomplete_study_before_new_scientific_work')
        elif any(item.get('study_return_ref') for item in proposals):
            result.update(status='complete_native_partition_development_review',
                reason='actual_paired_partition_return_consumed', required_work=None)


def _native_condition_revision_view(agent, question, group):
    """Use the latest exposed-development view without replacing old results."""
    latest_ref, latest = None, None
    effective = deepcopy(group['pair_refs'])
    for reference in question['method_use_plan'].get('native_condition_revision_refs', []):
        revision = read_method_reference(reference, agent.workspace)
        if (revision['method_ref'] != group['method_ref']
                or revision['baseline_method_ref'] != group['baseline_method_ref']):
            continue
        latest_ref, latest = reference, revision
        if revision.get('budget_incomplete'):
            continue
        originals, views = revision['original_pair_refs'], revision['development_pair_refs']
        effective = [deepcopy(views[originals.index(pair)] if pair in originals else pair)
                     for pair in effective]
        # Reference views keep the original origin. Only newly returned
        # invocations extend this generation; its old validation is now paid
        # development evidence, while the original files retain their verdict.
        unique = []
        for pair in effective:
            if pair not in unique:
                unique.append(pair)
        effective = unique
    views = latest['development_pair_refs'] if latest else []
    cached = bool(latest and not latest.get('budget_incomplete')
        and len(effective) == len(views) and all(pair in views for pair in effective))
    return latest_ref, latest, effective, cached


def review_native_method_returns(agent, question, *, compute_limit):
    """Review actual object-parent returns without inventing Python-method uses.

    Saved decision proposals belong to their original scenes. They are possible
    matched-development inputs, never instructions to repeat a completed task.
    A returned predictor, including a confirmed one, is not a method preference.
    """
    references = question['method_use_plan']['native_return_refs']
    required = 1 + len(references)
    if compute_limit < required:
        return {'status': 'scoped_unknown', 'reason': 'native_return_review_not_funded',
            'required_compute': required, 'new_compute_cost': 0, 'next_action': None}
    returns, sources, decisions, alternatives, choice_sources = [], [], [], [], []
    records = []
    for reference in references:
        record = read_method_reference(reference, agent.workspace)
        records.append((reference, record))
        source = record.get('source_task_ref')
        if source is not None and source not in sources:
            sources.append(deepcopy(source))
        returns.append({'feedback_ref': deepcopy(reference), **{key: deepcopy(record[key])
            for key in ('source_task_ref', 'actual_result_ref', 'before_ref', 'status',
                'comparison_ref', 'confirmation_ref', 'cost_refs', 'execution_complete',
                'method_pair_ref', 'method_ref', 'method_artifact_ref')
            if key in record}})
        proposal_ref = record.get('decision_options_ref')
        if not proposal_ref:
            continue
        published = read_method_reference(proposal_ref, agent.workspace)
        sequence = isinstance(published, list)
        published = published if sequence else [published]
        for index, decision in enumerate(published):
            addressed = {**deepcopy(proposal_ref), 'json_path': [
                *proposal_ref.get('json_path', []), index]} if sequence else deepcopy(proposal_ref)
            if decision.get('action') == 'choose_research_action':
                # These are the ordinary policy's real eligible alternatives,
                # not the later single-stage continuation publisher. Preserve
                # their provenance without treating the chosen action's quote
                # as a complete quote for every unexecuted alternative.
                chosen = decision.get('selected_action')
                eligible = list(dict.fromkeys(decision.get('eligible_actions', [])))
                decisions.append({'decision_ref': addressed,
                    'source_task_ref': deepcopy(source), 'status': 'recorded_native_choice',
                    'default_operation': chosen, 'operations': eligible,
                    'reason': decision.get('reason')})
                other = [name for name in eligible if name != chosen]
                if chosen in eligible and other:
                    choice_sources.append({'decision_ref': deepcopy(addressed),
                        'source_task_ref': deepcopy(source), 'ordinary_operation': chosen,
                        'alternative_operations': other,
                        'status': 'decision_state_and_complete_alternative_quotes_required'})
                continue
            if decision.get('action') not in (None, 'publish_object_method_options'):
                continue
            if 'options' not in decision:
                continue
            options = decision.get('options', [])
            default = decision.get('default_operation')
            decisions.append({'decision_ref': addressed, 'source_task_ref': deepcopy(source),
                'status': decision.get('status'), 'reason': decision.get('reason'),
                'default_operation': default,
                'operations': [item['operation'] for item in options]})
            # One legal next stage is not an alternative to the default. The
            # publisher must explicitly identify the actual ordinary action.
            if default is None or not any(item['operation'] == default for item in options):
                continue
            for position, option in enumerate(options):
                if option['operation'] == default:
                    continue
                alternatives.append({'source_task_ref': deepcopy(source),
                    'decision_ref': deepcopy(addressed), 'proposal_ref': {
                        **deepcopy(addressed), 'json_path': [
                            *addressed.get('json_path', []), 'options', position]},
                    'ordinary_operation': default, 'alternative_operation': option['operation'],
                    'status': 'matched_execution_required'})
    agent.costs['compute_cost'] += required
    result = {'status': 'complete_native_return_review',
        'reason': ('different_native_segments_require_matched_execution' if alternatives else
                   'retained_native_choices_require_same_state_binding' if choice_sources else
                   'no_new_method_content_from_recorded_native_returns'),
        'ordinary_returns': returns, 'distinct_source_refs': sources,
        'recorded_decisions': decisions, 'matched_development_inputs': alternatives,
        'native_choice_sources': choice_sources,
        'relative_labels_created': 0, 'installed_candidate': None,
        'new_compute_cost': required, 'support_updated': False,
        'independent_confirmation': False, 'independent_method_gain': None,
        'next_action': None}
    groups, excluded = _native_method_pair_groups(agent, records)
    result['paired_method_exclusions'] = excluded
    result['paired_condition_research'] = []
    result['native_condition_revisions'] = []
    if groups:
        from .method_feedback import research_paired_method_conditions, revise_paired_method_conditions

        paid_research = [(reference, read_method_reference(reference, agent.workspace))
            for reference in question['method_use_plan'].get('paired_condition_research_refs', [])]
        result['reason'] = 'native_matched_returns_consumed_by_existing_condition_learner'
        for group in groups:
            revision_ref, revision, effective, cached = _native_condition_revision_view(agent, question, group)
            if cached:
                learned = {**deepcopy(revision['research']), 'pair_refs': deepcopy(effective),
                    'compute_cost': 0, 'new_compute_cost': 0, 'reused': True,
                    'condition_revision_ref': deepcopy(revision_ref)}
            else:
                previous_ref = next((reference for reference, prior in reversed(paid_research)
                    if not prior.get('budget_incomplete')
                    and prior.get('method_ref') == group['method_ref']
                    and prior.get('baseline_method_ref') == group['baseline_method_ref']
                    and prior.get('pair_refs') == effective), None)
                learned = research_paired_method_conditions(agent, group['method_ref'],
                    effective, max(0, compute_limit - result['new_compute_cost']), max_conditions=8,
                    previous_ref=previous_ref)
            result['new_compute_cost'] += learned['new_compute_cost']
            result['paired_condition_research'].append({
                'method_ref': deepcopy(group['method_ref']),
                'baseline_method_ref': deepcopy(group['baseline_method_ref']),
                'baseline_method_artifact_ref': deepcopy(group['baseline_method_artifact_ref']),
                'method_artifact_ref': deepcopy(group['method_artifact_ref']), **learned})
            result['relative_labels_created'] += sum(
                item['preference'] in {'baseline', 'challenger'} for item in learned['preferences'])
            if learned.get('budget_incomplete'):
                result.update(status='scoped_unknown',
                    reason='native_pair_condition_allocation_incomplete',
                    required_compute=result['new_compute_cost'] + learned['required_compute'])
                break
            if (learned.get('status') == 'development_condition_without_group_support'
                    and learned.get('condition_validation_mode') == 'frozen_before_method_call'
                    and learned.get('group_validation_refs')
                    and not learned.get('missing_condition_validation_source_refs')):
                revised = revise_paired_method_conditions(agent, group['method_ref'], effective,
                    max(0, compute_limit - result['new_compute_cost']), previous_revision_ref=revision_ref)
                result['new_compute_cost'] += revised['new_compute_cost']
                result['native_condition_revisions'].append(revised)
                if revised.get('revision_ref'):
                    refs = question['method_use_plan'].setdefault('native_condition_revision_refs', [])
                    if revised['revision_ref'] not in refs:
                        refs.append(deepcopy(revised['revision_ref']))
                    agent._save('questions.json', json.dumps(agent.questions, separators=(',', ':')))
                if revised.get('budget_incomplete'):
                    result.update(status='scoped_unknown', reason='native_condition_revision_not_funded')
                    break
    elif excluded:
        result['reason'] = 'native_pair_body_identity_unresolved'
    _propose_native_transfer_partitions(agent, question, records, result, compute_limit)
    from .staged_method_research import (research_staged_method_conditions,
        revise_staged_method_conditions, propose_native_method_compositions,
        research_parent_bound_method_conditions)
    result['staged_method_proposals'] = propose_native_method_compositions(agent, question,
        result.get('existing_method_executions', []), groups,
        max(0, compute_limit - result['new_compute_cost']))
    result['new_compute_cost'] += sum(item['new_compute_cost'] for item in result['staged_method_proposals'])
    pending_compositions = [item['proposal_ref'] for item in result['staged_method_proposals']
        if item['status'] == 'staged_program_proposed' and item.get('candidate_count')]
    verified_compositions = [item for item in result['staged_method_proposals']
        if item['status'] == 'staged_method_path_verified']
    if any(item.get('budget_incomplete') for item in result['staged_method_proposals']):
        result.update(status='scoped_unknown', reason='native_composition_analysis_not_funded')
    for group in groups:
        staged = research_staged_method_conditions(agent, question, group,
            max(0, compute_limit - result['new_compute_cost']))
        result['new_compute_cost'] += staged['new_compute_cost']
        result['staged_method_proposals'].append(staged)
        if staged.get('budget_incomplete'):
            result.update(status='scoped_unknown', reason='staged_method_analysis_not_funded')
            break
    if result['status'] != 'scoped_unknown' and any(
            item['status'] == 'staged_program_proposed' and item['candidate_count']
            for item in result['staged_method_proposals']):
        result.update(status='method_content_proposed_pending_validation',
            reason='paid_baseline_returns_produced_unmeasured_staged_method',
            required_work='execute_staged_method_from_same_start_then_freeze_before_new_source_validation')
    elif any(item['status'] in {'incomplete_not_negative', 'staged_method_path_incomplete'}
             for item in result['staged_method_proposals']):
        result.update(status='scoped_unknown', reason='staged_method_development_incomplete',
            required_work='resolve_retained_incomplete_staged_study')
    if result['status'] != 'scoped_unknown':
        revision = revise_staged_method_conditions(agent, question, groups,
            max(0, compute_limit - result['new_compute_cost']))
        result['staged_method_revision'] = revision
        result['new_compute_cost'] += revision['new_compute_cost']
        if revision.get('budget_incomplete'):
            result.update(status='scoped_unknown', reason='staged_condition_revision_not_funded')
        elif revision.get('revision_ref'):
            revised = read_method_reference(revision['revision_ref'], agent.workspace)
            pending_compositions = [reference for reference in pending_compositions
                if reference != revised['basis']['proposal_ref']]
            if not pending_compositions:
                result.update(status='method_condition_revision_pending_validation',
                    reason=revision['status'], required_work=revision['required_work'])
    if result['status'] != 'scoped_unknown' and not pending_compositions and any(
            item.get('revision_ref') for item in result['native_condition_revisions']):
        candidates = any(item.get('research', {}).get('candidates')
            for item in result['native_condition_revisions'])
        result.update(status=('method_condition_revision_pending_validation' if candidates
                              else 'method_condition_revision_without_separating_condition'),
            reason='completed_frozen_validation_returned_to_new_condition_development',
            required_work=('freeze_revised_condition_before_new_physical_source_validation' if candidates
                           else 'diagnose_public_evidence_and_condition_language_on_paid_sources'))
    if result['status'] != 'scoped_unknown' and verified_compositions and not pending_compositions:
        result.update(status='method_path_verified_pending_independent_validation',
            reason='native_composition_paid_path_returned',
            required_work='assess_new_source_validation_for_verified_composition')
    if result['status'] != 'scoped_unknown':
        # A returned successor is evidence for the next method question. Do not
        # let the old paid-path receipt ask for that same successor again.
        learned = research_parent_bound_method_conditions(agent, question, groups,
            max(0, compute_limit - result['new_compute_cost']))
        result['native_composition_condition_research'] = learned
        result['new_compute_cost'] += sum(item['new_compute_cost'] for item in learned)
        for item in learned:
            if item.get('budget_incomplete'):
                result.update(status='scoped_unknown', reason='parent_bound_condition_analysis_not_funded')
                break
            if not item.get('usable_origins'):
                continue
            pending_compositions = [ref for ref in pending_compositions
                if ref != item['source_proposal_ref']]
            if pending_compositions:
                continue
            result.update(status=('method_condition_revision_pending_validation'
                if item.get('candidate_count') else 'method_condition_evidence_insufficient'
                if item['research_status'] == 'insufficient_paired_preferences' else
                'method_condition_revision_without_separating_condition'),
                reason=item['status'], required_work=item['required_work'])
    return result


def review_python_method_returns(agent, question, *, public_spec, compute_limit):
    """Accumulate ordinary returns into a method-content hypothesis, not a gain label."""
    from .finite_native_author import publish_research_options, research_segment_options
    from .method_conditions import method_delivery_outcome

    references = question['method_use_plan']['ordinary_return_refs']
    saved = question.get('routing_execution', {}).get('result', {}).get('method_action_research', {})
    if (saved.get('status') in {'method_content_proposed_pending_validation', 'scoped_unknown'}
            and saved.get('proposals', {}).get('candidates')
            and [row['feedback_ref'] for row in saved.get('ordinary_returns', [])] == references
            and all(item.get('status') != 'analysis_budget_unresolved'
                    for item in saved['proposals'].get('experience_revisions', []))):
        return resume_python_method_proposals(agent, question, saved, compute_limit=compute_limit)
    required = 1 + len(references)
    if compute_limit < required:
        return {'status': 'scoped_unknown', 'reason': 'method_return_review_not_funded',
            'required_compute': required, 'new_compute_cost': 0, 'next_action': None}
    rows, sources = [], []
    for reference in references:
        record = read_method_reference(reference, agent.workspace)
        rows.append({'feedback_ref': deepcopy(reference),
            'origin_ref': deepcopy(record.get('origin_ref', reference)),
            'method_ref': deepcopy(record['method_ref']),
            'delivery': method_delivery_outcome(record),
            'profile_ref': deepcopy(record.get('profile_ref')),
            'actual_result_ref': deepcopy(record.get('actual_result_ref')),
            'complete_comparison_ref': deepcopy(record.get('complete_comparison_ref')),
            'costs': deepcopy(record.get('costs')),
            'cost_refs': deepcopy(record.get('cost_refs', []))})
        source = record.get('method_content_source')
        if source and source not in sources:
            sources.append(deepcopy(source))
    view = research_segment_options(public_spec, agent.observations,
        {**agent.knowledge, 'program_results': []}, state=agent._program_state(),
        decision={'research_options': publish_research_options(agent, public_spec)})
    agent.costs['compute_cost'] += required
    result = {'status': 'method_actions_unresolved',
        'reason': 'ordinary_returns_require_new_matched_action_evidence',
        'ordinary_returns': rows, 'current_legal_options': view['options'],
        'excluded_options': view['excluded'],
        'selected_research_action': 'ordinary_research',
        'required_work': 'fresh_current_ordinary_proposal_and_matched_alternative_action_return',
        'relative_labels_created': 0, 'installed_candidate': None, 'next_action': None,
        'new_compute_cost': required, 'support_updated': False,
        'independent_confirmation': False}
    result['method_content_sources'] = sources
    # The invoked P0 and the conditional method being revised have different
    # identities. Use the existing shared topic, not P0's absent selection plan
    # or an arbitrary method from the entire library.
    methods = _topic_method_candidates(agent, question)
    result['revisable_method_refs'] = [
        {'id': item['id'], 'version': item['version']} for item in methods]
    distinct_sources = []
    for item in sources:
        if item['source_task_ref'] not in distinct_sources:
            distinct_sources.append(item['source_task_ref'])
    if len(distinct_sources) < 2 or not methods:
        result['reason'] = ('distinct_completed_sources_required' if len(distinct_sources) < 2
                            else 'existing_method_topic_body_required')
        return result
    if compute_limit < required + len(sources):
        result.update(status='scoped_unknown', reason='method_content_projection_not_funded',
                      required_compute=required + len(sources))
        return result
    experiences, excluded = _paid_action_experiences(agent, sources)
    agent.costs['compute_cost'] += len(sources)
    result['new_compute_cost'] += len(sources)
    result['experience_exclusions'] = excluded
    from .research_programs import derive_paid_witness_binding_revision
    proposals, analyses = [], []
    for method in methods:
        analysis = derive_paid_witness_binding_revision(experiences, method['selection_plan'],
            compute_limit=compute_limit - result['new_compute_cost'])
        cost = analysis['analysis_compute_cost']
        agent.costs['compute_cost'] += cost
        result['new_compute_cost'] += cost
        method_ref = {'id': method['id'], 'version': method['version']}
        analyses.append({'method_ref': method_ref, **analysis})
        if analysis.get('candidate'):
            proposals.append({**deepcopy(analysis['candidate']),
                'previous_method_ref': method_ref})
        if analysis['status'] == 'analysis_budget_unresolved':
            result.update(status='scoped_unknown', reason='method_content_analysis_not_completed')
            break
    result['proposals'] = {'status': 'ordinary_returns_consumed', 'candidates': proposals,
                           'experience_revisions': analyses}
    if proposals and result['status'] != 'scoped_unknown':
        result.update(status='method_content_proposed_pending_validation',
            reason='actual_research_relations_produced_unmeasured_method_content',
            required_work='compare_existing_method_body_before_new_matched_execution')
        if _bind_current_method_dependencies(agent, result, compute_limit):
            _resolve_method_proposals(agent, question, result, methods, compute_limit)
    elif result['status'] != 'scoped_unknown':
        result['reason'] = 'no_new_method_content_supported_by_accumulated_returns'
    return result


def resume_python_method_proposals(agent, question, pending, *, compute_limit):
    """Resolve saved proposals without repeating their paid source induction."""
    if pending.get('status') not in {'method_content_proposed_pending_validation', 'scoped_unknown'}:
        raise ValueError('an unresolved saved method proposal is required')
    result = deepcopy(pending)
    result.update(status='method_content_proposed_pending_validation', new_compute_cost=0,
                  proposal_analysis_reused=True, installed_candidate=None, next_action=None)
    if not result.get('proposals', {}).get('candidates'):
        return {**result, 'status': 'scoped_unknown', 'reason': 'saved_content_proposal_required'}
    methods = _topic_method_candidates(agent, question)
    available = [{'id': item['id'], 'version': item['version']} for item in methods]
    if any(item['previous_method_ref'] not in available for item in result['proposals']['candidates']):
        return {**result, 'status': 'scoped_unknown', 'reason': 'original_method_topic_body_required'}
    if _bind_current_method_dependencies(agent, result, compute_limit):
        _resolve_method_proposals(agent, question, result, methods, compute_limit)
    return result


def _bind_current_method_dependencies(agent, result, compute_limit):
    """New finite plans use the registered current finite execution adapter.

    A predecessor's frozen adapter can predate a newly proposed operation. Keep
    that predecessor intact; bind the new proposal to the actual current entry
    and retain both source references before comparing execution identities.
    """
    if not agent.knowledge.get('finite_native_author'):
        return True
    from .finite_native_author import export_finite_native_author

    if result['new_compute_cost'] >= compute_limit:
        result.update(status='scoped_unknown', reason='method_dependency_resolution_not_funded')
        return False
    result['new_compute_cost'] += 1
    agent.costs['compute_cost'] += 1
    current = agent.workspace / 'ordinary.py'
    if not current.is_file() or current.read_text(encoding='utf-8') != export_finite_native_author():
        result.update(status='scoped_unknown', reason='current_registered_finite_ordinary_source_required')
        return False
    for proposal in result['proposals']['candidates']:
        plan = proposal['research_selection_plan']
        previous = agent.workspace / plan['ordinary_method_path']
        proposal.setdefault('ordinary_dependency_revision', {
            'previous_source_ref': {'file': str(previous), 'json_path': []},
            'current_source_ref': {'file': str(current), 'json_path': []},
            'basis': 'registered_finite_native_author_entry'})
        plan['ordinary_method_path'] = str(current)
    return True


def _method_execution_body(plan):
    """Compare finite executable content without treating provenance as code."""
    from .research_programs import export_research_selection_program

    def content(value):
        if isinstance(value, dict):
            return {key: ('ordinary.py' if key == 'ordinary_method_path' else content(item))
                    for key, item in value.items() if key != 'source_refs'}
        if isinstance(value, list):
            return [content(item) for item in value]
        return value

    return export_research_selection_program(content(plan))


def _resolve_method_proposals(agent, question, result, methods, compute_limit):
    """Reuse an actually supported body, or retain an executable paired proposal.

    All work here is finite evidence analysis. A new body is neither installed
    nor given its predecessor's scientific support. The ordinary pairing entry
    owns the current source, resource and evaluation registration.
    """
    def charge():
        if result['new_compute_cost'] >= compute_limit:
            result.update(status='scoped_unknown', reason='method_proposal_resolution_not_funded')
            return False
        result['new_compute_cost'] += 1
        agent.costs['compute_cost'] += 1
        return True

    def anchored(reference):
        return {**deepcopy(reference), 'file': str(agent.workspace / reference['file'])}

    topic_ref = question.get('origin_method_question_ref') or {
        'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)}
    topic = next((item for item in agent.questions if item['id'] == topic_ref['question_id']
        and item.get('meaning_version', 1) == topic_ref.get('meaning_version', 1)), question)
    pending = []
    for proposal in result['proposals']['candidates']:
        previous = next(item for item in methods if
                        {'id': item['id'], 'version': item['version']} == proposal['previous_method_ref'])
        plan = proposal['research_selection_plan']
        source_path = agent.workspace / plan['ordinary_method_path']
        proposed_body = _method_execution_body(plan)
        for candidate in methods:
            if not charge():
                return
            if (_method_execution_body(candidate['selection_plan']) != proposed_body
                    or (agent.workspace / candidate['selection_plan']['ordinary_method_path']).read_text(
                        encoding='utf-8') != source_path.read_text(encoding='utf-8')):
                continue
            method_ref = {'id': candidate['id'], 'version': candidate['version']}
            feedback = agent.theory_consolidation.get('method_feedback', {}).get(candidate['id'], {})
            for reference in feedback.get('outcome_refs', []):
                if not charge():
                    return
                record = read_method_reference(reference, agent.workspace)
                actual_ref = record.get('actual_result_ref')
                if record.get('method_ref') != method_ref or not actual_ref:
                    continue
                actual = read_method_reference(actual_ref, agent.workspace)
                comparison = actual.get('comparison', {})
                quality = actual.get('quality', {})
                if (actual.get('method_ref') != method_ref
                        or actual.get('execution_complete') is not True
                        or actual.get('completed') is not True
                        or not comparison.get('models_requested')
                        or comparison.get('models_completed') != comparison['models_requested']
                        or comparison.get('truncated') or comparison.get('error')
                        or type(quality.get('loss')) not in (int, float)
                        or type(quality.get('threshold')) not in (int, float)
                        or not quality['loss'] <= quality['threshold']):
                    continue
                parent_ref = question.get('parent_question_ref') or {
                    'question_id': question.get('parent_id')}
                parent = next((item for item in agent.questions
                    if item['id'] == parent_ref.get('question_id')
                    and item.get('meaning_version', 1) == parent_ref.get(
                        'meaning_version', item.get('meaning_version', 1))), None)
                result.update(status='complete_supported_method_retained',
                    reason='same_topic_executable_body_has_actual_completed_support',
                    installed_candidate=deepcopy(candidate),
                    existing_method_support_refs=[anchored(actual_ref)],
                    existing_support_scope='actual_completed_invocation',
                    selected_research_action='reuse_existing_method', required_work=None,
                    next_action=({'action': 'run', 'arguments': {'operator': 'procedure',
                        'path': candidate['execution']['path'],
                        'max_steps': candidate['execution']['max_steps'], 'confirm': False,
                        'research_question': {'id': parent['id'],
                            'meaning_version': parent.get('meaning_version', 1)}}}
                        if parent is not None else None))
                if candidate.get('selection_ref'):
                    result['policy_selection_ref'] = anchored(candidate['selection_ref'])
                result.pop('validation_requests', None)
                return
        if not charge():
            return
        pair_ref, bundle = None, None
        for saved in topic.get('method_content_proposal_refs', []):
            if not charge():
                return
            retained = read_method_reference(saved, agent.workspace)
            if (retained['previous_method_ref'] == proposal['previous_method_ref']
                    and _method_execution_body(retained['cells']['10']['selection_plan']) == proposed_body
                    and Path(retained['ordinary_source_ref']['file']).read_text(
                        encoding='utf-8') == source_path.read_text(encoding='utf-8')):
                pair_ref, bundle = anchored(saved), retained
                break
        if bundle is None:
            directory = agent.workspace / 'method_feedback'
            number = 0
            while (directory / f'content_proposal_{number:06d}.json').exists():
                number += 1
            path = f'method_feedback/content_proposal_{number:06d}.json'
            proposed_id = f"{previous['id']}_content_{number:06d}"
            selection = anchored(previous['selection_ref']) if previous.get('selection_ref') else None
            bundle = {'status': 'unmeasured_local_research_intervention',
                'method_question_ref': deepcopy(topic_ref),
                'ordinary_source_ref': {'file': str(source_path), 'json_path': []},
                'previous_method_ref': deepcopy(proposal['previous_method_ref']),
                'source_refs': deepcopy(proposal['source_refs']),
                'cells': {'00': {'id': f"{previous['id']}_comparison_{number:06d}",
                    'previous_method_ref': deepcopy(proposal['previous_method_ref']),
                    'selection_plan': {**deepcopy(previous['selection_plan']),
                                       'ordinary_method_path': str(source_path)},
                    'source_refs': deepcopy(previous.get('source_evidence_refs', [])),
                    'selection_ref': selection},
                    '10': {'id': proposed_id, 'selection_plan': {**deepcopy(plan),
                            'ordinary_method_path': str(source_path)},
                        'source_refs': deepcopy(proposal['source_refs']), 'selection_ref': selection}},
                'source_support_inherited': False, 'execution_authorized': False}
            agent._save(path, json.dumps(bundle, indent=2))
            pair_ref = {'file': str(agent.workspace / path), 'json_path': []}
            topic.setdefault('method_content_proposal_refs', []).append(deepcopy(pair_ref))
        proposal['method_pair_ref'] = deepcopy(pair_ref)
        pending.append({'entry_point': 'scripts.run_state_method_pairing.prepare',
            'method_pair_ref': pair_ref,
            'method_question_ref': deepcopy(topic_ref),
            'previous_method_ref': deepcopy(proposal['previous_method_ref']),
            'candidate_id': bundle['cells']['10']['id'], 'max_steps': previous['execution']['max_steps'],
            'source_outcome_refs': [anchored(item['outcome_ref'])
                                    for item in result['method_content_sources']],
            'execution_authorized': False,
            'required_work': 'quote_on_current_registered_paid_scene_before_paired_execution'})
    result.update(validation_requests=pending,
        required_work='quote_on_current_registered_paid_scene_before_paired_execution')


def _paid_action_experiences(agent, references):
    """Project actual native returns and their pre-use parent witnesses.

    Sequence rewards are not inferred from final model loss. These records
    support a content proposal, while matched returns still govern adoption.
    """
    experiences, excluded = [], []
    for supplied in references:
        reference = supplied['outcome_ref']
        outcome = read_method_reference(reference, agent.workspace)
        comparison = outcome.get('comparison', {})
        if (outcome.get('execution_complete') is not True or comparison.get('truncated')
                or not comparison.get('models_requested')
                or comparison.get('models_completed') != comparison['models_requested']):
            excluded.append({'outcome_ref': deepcopy(reference),
                'reason': 'completed_whole_research_comparison_required'})
            continue
        profile_ref = outcome['profile_ref']
        profile = read_method_reference(profile_ref, agent.workspace)
        for returned in outcome.get('native_representation_returns', []):
            native_ref = returned['native_result_ref']
            native = read_method_reference(native_ref, agent.workspace)
            identifiers = set(native.get('parent_theory_ids', []))
            parents, seen = [], set()
            for witness in profile.get('research', {}).get('candidate_pool', {}).get('witnesses', []):
                transition = witness.get('transition', {})
                if (transition.get('status') != 'observed'
                        or transition.get('guard_depth') is None
                        or transition.get('mature_first_divergences') != 0
                        or transition.get('startup_first_divergences', 0) <= 0):
                    continue
                identifier = next((name.removesuffix('.py') for name in witness.get('models', [])
                    if name.removesuffix('.py') in identifiers), None)
                if identifier is None or identifier in seen:
                    continue
                seen.add(identifier)
                program_ref = {**deepcopy(outcome['start_state_ref']), 'json_path': [
                    *outcome['start_state_ref'].get('json_path', []), 'theory_index', identifier, 'program']}
                parents.append({'id': identifier,
                    'program': read_method_reference(program_ref, agent.workspace),
                    'source_ref': program_ref})
            experiences.append({'source_task_ref': deepcopy(supplied['source_task_ref']),
                'profile': profile, 'parent_programs': parents, 'native_result': native,
                'source_refs': [deepcopy(reference), deepcopy(profile_ref), deepcopy(native_ref),
                    *[deepcopy(parent['source_ref']) for parent in parents]]})
    return experiences, excluded


def research_method_actions(agent, question, request, *, public_spec, compute_limit):
    """Prepare a real method question; only matched paid pairs can install a body.

    This route never runs an alternative to manufacture a preference. Scientific
    paired execution belongs to its separately admitted ordinary study.
    """
    from .finite_native_author import research_segment_options, publish_research_options
    from .research_episodes import extract_method_segment
    from .research_programs import (propose_research_action_revisions,
        export_research_selection_program, rebuild_research_selection_guard)
    from .method_feedback import research_paired_method_conditions
    from .theory_consolidation import install_python_research_method

    roots = request.get('segments', [])
    # Projection/static choices have an explicit finite engineering-work count;
    # neither prediction nor scientific fitting is hidden in this route.
    experience_refs = request.get('action_experience_refs', [])
    required = 1 + len(roots) + len(experience_refs)
    if compute_limit < required:
        return {'status': 'scoped_unknown', 'reason': 'method_projection_not_funded',
            'required_compute': required, 'new_compute_cost': 0, 'installed_candidate': None}
    segments = [extract_method_segment(agent, root_result_ref=item['root_result_ref'],
        decision_start_ref=item['decision_start_ref']) for item in roots]
    agent.costs['compute_cost'] += required
    sequences = []
    from .research_methods import observe_common_method_segment
    for supplied in request.get('segment_return_refs', []):
        segment = segments[supplied['segment_index']]
        observed = observe_common_method_segment(segment,
            read_method_reference(supplied['actual_return_ref'], agent.workspace),
            actual_return_ref=supplied['actual_return_ref'],
            profile=read_method_reference(supplied['profile_ref'], agent.workspace),
            source_task=supplied['source_task_ref'],
            state=question.get('method_sequence_experience'))
        question['method_sequence_experience'] = observed['state']
        sequences.append({key: value for key, value in observed.items() if key != 'state'})
    missing = [item for segment in segments for item in segment['missing_dependencies']]
    if (not roots or missing or not request.get('ordinary_proposal_ref')
            or not request.get('ordinary_method_path')):
        return {'status': 'method_actions_unresolved', 'reason': 'actual_method_inputs_incomplete',
            'segments': segments, 'missing_dependencies': missing,
            'sequence_observations': sequences, 'new_compute_cost': required, 'installed_candidate': None}
    ordinary = read_method_reference(request['ordinary_proposal_ref'], agent.workspace)
    memory = {**agent.knowledge, 'program_results': []}
    state = agent._program_state()  # Existing live view; do not copy the strong index.
    view = research_segment_options(public_spec, agent.observations, memory, state=state,
        decision={'research_options': publish_research_options(agent, public_spec)})
    tested_ref = request.get('tested_candidate_ref')
    tested = read_method_reference(tested_ref, agent.workspace) if tested_ref else None
    experience_exclusions = []
    if tested is not None:
        if (tested.get('kind') != 'python_research_method'
                or tested.get('execution', {}).get('kind') != 'python_select_actions'):
            raise ValueError('paired evidence requires its actual frozen Python method')
        proposals = {'status': 'actual_frozen_body_reused', 'baseline': {'id': 'P0'},
            'candidates': [{'id': tested['id'], 'research_selection_plan': tested['selection_plan'],
                'source_refs': tested.get('source_evidence_refs', []), 'tested_candidate_ref': tested_ref}]}
    else:
        experiences, experience_exclusions = _paid_action_experiences(agent, experience_refs)
        prior_ref = request.get('prior_selection_plan_ref')
        prior_plan = read_method_reference(prior_ref, agent.workspace) if prior_ref else None
        proposals = propose_research_action_revisions(ordinary, view['options'],
            source_refs=[item['root_result_ref'] for item in roots], max_candidates=2,
            selection_states=request.get('selection_states', ()),
            method_experiences=experiences, prior_selection_plan=prior_plan,
            compute_limit=compute_limit - required)
        analysis_cost = proposals.get('analysis_compute_cost', 0)
        agent.costs['compute_cost'] += analysis_cost
        required += analysis_cost
    candidates = proposals.get('candidates', [])
    result = {'status': 'paired_study_required' if candidates else 'complete_no_distinct_legal_alternative',
        'segments': segments, 'proposals': proposals, 'excluded_options': view['excluded'],
        'sequence_observations': sequences,
        'experience_exclusions': experience_exclusions,
        'installed_candidate': None, 'new_compute_cost': required,
        'execution_authorized': False, 'independent_confirmation': False,
        'support_updated': False}
    if (not candidates and proposals.get('experience_revision', {}).get('status')
            == 'analysis_budget_unresolved'):
        result.update(status='method_actions_unresolved', reason='method_content_analysis_not_completed')
    pair_refs = request.get('pair_refs', [])
    if not candidates or not pair_refs:
        return result
    if tested is None:
        result.update(status='method_actions_unresolved', reason='actual_tested_body_reference_required')
        return result
    selected = next((item for item in candidates if item['id'] == request.get('paired_candidate_id')), None)
    if selected is None:
        result.update(status='method_actions_unresolved', reason='actual_paired_candidate_required')
        return result
    # Pair references must name this exact frozen challenger. An alternative's
    # good return cannot be transferred to an untested body with a similar name.
    for reference in pair_refs:
        pair = read_method_reference(reference, agent.workspace)
        outcome = read_method_reference(pair['challenger_ref'], agent.workspace)
        if outcome.get('method_ref') != {'id': tested['id'], 'version': tested['version']}:
            result.update(status='method_actions_unresolved', reason='paired_body_identity_mismatch')
            return result
        measured_ref = outcome.get('method_artifact_ref')
        if measured_ref is None:
            result.update(status='method_actions_unresolved', reason='actual_paired_body_reference_required')
            return result
        measured = read_method_reference(measured_ref, agent.workspace)
        body_fields = ('id', 'version', 'kind', 'execution', 'selection_plan', 'ordinary_source_ref')
        if any(measured.get(key) != tested.get(key) for key in body_fields):
            result.update(status='method_actions_unresolved', reason='paired_body_content_mismatch')
            return result
    choice = research_paired_method_conditions(agent,
        {'id': tested['id'], 'version': tested['version']}, pair_refs, compute_limit - required,
        max_conditions=8, target_profile=getattr(agent, '_method_use_profile', None))
    result['paired_selection'] = choice
    result['identified_effects'] = deepcopy(choice['identified_effects'])
    result['evaluation_use'] = choice['evaluation_use']
    result['new_compute_cost'] += choice['new_compute_cost']
    if not choice.get('local_program_supported'):
        result['status'] = ('complete_no_supported_action_revision'
            if choice.get('status') != 'UNKNOWN' else 'method_actions_unresolved')
        return result
    rebuild = rebuild_research_selection_guard(selected['research_selection_plan'], choice)
    plan = rebuild['plan']
    result['guard_reconstruction'] = {key: value for key, value in rebuild.items() if key != 'plan'}
    # A prior candidate can still use the old selection file for its coverage.
    # Rechecking the same method must not overwrite that live evidence.
    stem = f"method_feedback/action_selection_{selected['id'].replace(':', '_')}"
    number = 0
    selection_path = f'{stem}_{number:04d}.json'
    while (agent.workspace / selection_path).exists():
        number += 1
        selection_path = f'{stem}_{number:04d}.json'
    agent._save(selection_path, json.dumps(choice, indent=2))
    if rebuild['changed']:
        identifier = request.get('revised_candidate_id')
        if not identifier or identifier == selected['id']:
            result.update(status='method_actions_unresolved', reason='new_guard_body_identity_required')
            return result
        installed = install_python_research_method(agent, candidate_id=identifier,
            source=export_research_selection_program(plan), selection_plan=plan,
            source_refs=selected.get('source_refs', []), paired_support_refs=choice['local_support_refs'],
            max_steps=request.get('max_steps', 48), previous_method_ref={
                'id': tested['id'], 'version': tested['version']},
            selection_ref={'file': selection_path, 'json_path': []},
            method_question_ref={'question_id': question['id'],
                                 'meaning_version': question.get('meaning_version', 1)})
    else:
        installed = agent.theory_consolidation.get('candidates', {}).get(tested['id'])
        if installed is None or any(installed.get(key) != tested.get(key) for key in body_fields):
            result.update(status='method_actions_unresolved', reason='restore_actual_tested_body_before_use')
            return result
        # Supporting observations revise experience, not the already tested
        # body's identity or its source-level two-seed claims.
        group = agent.theory_consolidation.setdefault('method_feedback', {}).setdefault(
            tested['id'], {'outcome_refs': []})
        selection_ref = {'file': selection_path, 'json_path': []}
        supports = group.setdefault('paired_selections', [])
        if selection_ref not in supports:
            supports.append(selection_ref)
        group['method_question_ref'] = {'question_id': question['id'],
                                        'meaning_version': question.get('meaning_version', 1)}
        from .theory_consolidation import _save
        _save(agent)
        installed = deepcopy(installed)
    parent_ref = question.get('parent_question_ref') or {'question_id': question.get('parent_id')}
    parent = next((item for item in agent.questions
        if item['id'] == parent_ref.get('question_id')
        and item.get('meaning_version', 1) == parent_ref.get('meaning_version', item.get('meaning_version', 1))), None)
    result.update(status=('complete_supported_method_retained'
        if rebuild['status'] == 'conditioned_method_retained' else 'local_method_action_revision_selected'),
        installed_candidate=installed,
        next_action=({'action': 'run', 'arguments': {'operator': 'procedure',
            'path': installed['execution']['path'], 'max_steps': installed['execution']['max_steps'],
            'confirm': False, 'research_question': {'id': parent['id'],
                'meaning_version': parent.get('meaning_version', 1)}}} if parent is not None else None))
    if installed.get('selection_ref'):
        result['policy_selection_ref'] = {**deepcopy(installed['selection_ref']),
            'file': str(agent.workspace / installed['selection_ref']['file'])}
    if parent is None:
        result['reason'] = 'actual_parent_use_binding_required_before_ordinary_execution'
    return result

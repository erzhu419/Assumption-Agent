"""Read already checked research objects through explicit versioned queries.

ParameterRegion retrieves an authored, numerically checked working region. It
does not solve for a feasible set or turn a fitted representative into a region.
"""

from copy import deepcopy

from .question_semantics import validate_answer_type


def evaluate_parameter_region(query, questions, *, context=None):
    """Read the current checked region at one question/meaning/answer version.

    ``source_ref`` names question_id, meaning_version and answer_version;
    ``scope`` must equal the source answer's retained scope. Optional family_ref
    and parameter_role require explicit corresponding source declarations.
    The caller still checks the source's existing support paths and installs a
    returned answer_record through record_answer before binding a parent.
    """
    base = {'evidence': [], 'compute_cost': 0}

    def unknown(detail, reason='unknown_due_to_missing_evidence'):
        return {**base, 'status': 'scoped_unknown', 'reason': reason, 'detail': detail}

    reference = query.get('source_ref')
    if not isinstance(reference, dict) or not reference.get('question_id') or any(
            type(reference.get(key)) is not int or reference[key] < 1
            for key in ('meaning_version', 'answer_version')):
        return unknown('explicit_question_meaning_and_answer_versions_required',
                       'unknown_due_to_uncertain_mapping')
    if reference.get('answer_field', 'executable_answer') != 'executable_answer':
        return unknown('parameter_region_requires_executable_answer', 'unknown_due_to_uncertain_mapping')
    values = questions.values() if isinstance(questions, dict) else questions
    source = next((item for item in values if item['id'] == reference['question_id']), None)
    if source is None:
        return unknown('source_question_unavailable')
    if source.get('meaning_version', 1) != reference['meaning_version']:
        return unknown('source_question_meaning_changed')
    answer = source.get('executable_answer', {})
    if answer.get('version') != reference['answer_version']:
        return unknown('source_answer_version_changed')
    if (answer.get('status') != 'supported' or answer.get('evidence_status') != 'supported'
            or not answer.get('evidence') or not answer.get('check_results')
            or not all(check.get('status') == 'satisfied' for check in answer['check_results'])):
        return unknown('source_region_lacks_current_satisfied_checks')
    region, scope = answer.get('parameter_region'), answer.get('scope')
    if not isinstance(region, dict) or not validate_answer_type(region, 'interval'):
        return unknown('source_has_no_checked_parameter_region')
    if not scope or not query.get('scope'):
        return unknown('explicit_source_scope_required', 'unknown_due_to_uncertain_mapping')
    if query['scope'] != scope:
        return {**base, 'status': 'invalid_experiment_binding', 'reason': 'parameter_scope_mismatch'}
    for key in ('family_ref', 'parameter_role'):
        if key in query:
            declared = source.get(key)
            if key == 'family_ref' and declared is None:
                declared = source.get('family_id')
            if declared is None:
                return unknown(f'source_{key}_not_declared', 'unknown_due_to_uncertain_mapping')
            if query[key] != declared:
                return {**base, 'status': 'invalid_experiment_binding', 'reason': f'{key}_mismatch'}
    requested_context = query.get('context_ref', context)
    if requested_context is not None:
        if source.get('context_ref') is None:
            return unknown('source_context_not_declared', 'unknown_due_to_uncertain_mapping')
        if requested_context != source['context_ref']:
            return {**base, 'status': 'invalid_experiment_binding', 'reason': 'context_mismatch'}
    retained = {key: deepcopy(answer[key]) for key in
                ('status', 'evidence_status', 'parameter_region', 'scope', 'evidence', 'check_results')}
    return {'status': 'answered', 'value': deepcopy(region), 'scope': deepcopy(scope),
        'evidence': deepcopy(answer['evidence']), 'compute_cost': 0,
        'source_ref': {**deepcopy(reference), 'answer_field': 'executable_answer'},
        'completeness': 'not_established',
        'region_kind': 'checked_working_region', 'independent_confirmation': False,
        'answer_record': retained}


def evaluate_shared_structure(agent, question, query, lab, mission):
    """Answer a finite typed-source construction query using Common induction.

    Sources are ``{kind:'theory', id:..., version:...}`` references into the
    existing theory index. ``mapping_budget`` is the existing consolidation
    allocation with compute/max_groups. Returned objects and inverse mappings
    remain in shared/theory_library.json; no Program or paid rows are copied.
    The consolidation operator already charges ``compute_cost`` to the agent.
    ``object_type='program_revision_relation'`` requests joint before/after
    reconstruction, rather than accepting an after-only construction object.
    """
    from .question_routing import resolved_theories
    from .theory_consolidation import consolidate_research

    base = {'evidence': [], 'compute_cost': 0, 'independent_confirmation': False,
            'scientific_support': False}

    def author(reason, **extra):
        return {**base, 'status': 'author_work_request', 'object_type': 'typed_shared_structure',
                'reason': reason, **extra}

    object_type = query.get('object_type')
    if object_type not in (None, 'program_revision_relation'):
        return author('unsupported_shared_construction_object_type', requested_object_type=object_type)
    references = query.get('typed_program_refs')
    if not isinstance(references, list) or len(references) < 2:
        return author('at_least_two_typed_program_refs_required')
    selected, retained = [], []
    pool = {item['id']: item for item in resolved_theories(agent)}
    for reference in references:
        if (not isinstance(reference, dict) or reference.get('kind') != 'theory'
                or not reference.get('id') or type(reference.get('version')) is not int
                or reference['version'] < 1):
            return author('explicit_theory_id_and_version_required')
        theory = pool.get(reference['id'])
        if theory is None or theory.get('version') != reference['version']:
            return {**base, 'status': 'scoped_unknown', 'reason': 'unknown_due_to_missing_evidence',
                    'detail': 'typed_source_version_unavailable', 'source_ref': deepcopy(reference)}
        if not theory.get('program') or theory.get('source_status') == 'unavailable_retained_version':
            return author('executable_source_Program_required', source_ref=deepcopy(reference))
        signatures = theory.get('role_signature')
        if (not isinstance(signatures, dict) or theory.get('output_unit') is None
                or theory.get('context_ref') is None
                or any(not isinstance(contract, dict) or any(key not in contract for key in ('role', 'unit', 'time'))
                       for contract in signatures.values())):
            return author('source_role_unit_time_and_context_required', source_ref=deepcopy(reference))
        if theory['id'] not in selected:
            selected.append(theory['id'])
            retained.append(deepcopy(reference))
    if len(selected) < 2:
        return author('at_least_two_distinct_program_refs_required')
    allocation = query.get('mapping_budget')
    if (not isinstance(allocation, dict) or type(allocation.get('compute')) is not int
            or allocation['compute'] < 0 or type(allocation.get('max_groups', 16)) is not int
            or allocation.get('max_groups', 16) < 0):
        return author('explicit_bounded_mapping_budget_required')
    arguments = {'source_theory_ids': selected, 'allocation': deepcopy(allocation),
        'theme': f'FindSharedStructure:{question["id"]}@{question.get("meaning_version", 1)}',
        'mission': mission if mission is not None else agent.research_mission}
    for field in ('context_family_ref', 'episode_sources', 'materials', 'counterexamples', 'object_type'):
        if field in query:
            arguments[field] = deepcopy(query[field])
    result = consolidate_research(agent, arguments, lab, question)
    base['compute_cost'] = result['new_compute_cost']
    if result['status'] == 'research_deferred':
        return {**base, 'status': 'research_deferred', 'reason': result['reason']}
    abstractions = []
    for frozen in result['frozen_candidates']:
        if object_type is not None and frozen['kind'] != object_type:
            continue
        identifier = frozen['candidate_id']
        candidate = agent.theory_consolidation['candidates'][identifier]
        reference = {'file': result['library_path'], 'json_path': ['candidates', identifier],
                     'version': candidate['version']}
        executable_field = 'definition' if candidate.get('definition') else 'template'
        inverse_field = 'source_bindings' if candidate.get('source_bindings') else 'instances'
        abstractions.append({'library_ref': reference, 'kind': candidate['kind'],
            'abstract_program_ref': {**reference, 'json_path': [*reference['json_path'], executable_field]},
            'inverse_bindings_ref': {**reference, 'json_path': [*reference['json_path'], inverse_field]},
            'source_reconstruction_ref': {**reference, 'json_path': [*reference['json_path'], 'source_reconstruction']}})
    partial = {'kind': 'shared_structure', 'abstractions': abstractions}
    scope = {'kind': 'finite_typed_program_sources', 'typed_program_refs': retained}
    common = {**base, 'scope': scope, 'cache_hit': result['cache_hit'],
        'source_reconstruction_only': True, 'search_completeness': 'not_established',
        'budget_exhausted': result['budget_exhausted'],
        'excluded': deepcopy(result['excluded']), 'author_work': deepcopy(result['author_work'])}
    if result.get('bounded_status', result['status']) == 'UNKNOWN':
        return {**common, 'status': 'scoped_unknown', 'reason': 'unknown_due_to_incomplete_search',
                'partial_result': partial}
    if not abstractions:
        return {**common, 'status': 'author_work_request', 'object_type': 'typed_shared_structure',
                'reason': 'no_compatible_executable_shared_structure'}
    return {**common, 'status': 'answered', 'value': partial}

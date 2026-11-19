"""Explicit question meaning, separate from a session and its chosen solver.

Compilation describes executable work; it neither calls a provider nor buys
observations. Legacy prose remains an idea until an author supplies its contract.
"""
from copy import deepcopy
from math import isfinite


_CONTRACT_MEANING_FIELDS = (
    'task_view', 'observation_map', 'readout_spec',
    'scale_map', 'scale_spec', 'protocol_spec',
    'claim_contract', 'decision_region',
)
_COSMETIC_CONTRACT_KEYS = {'display_label', 'description', 'notes', 'ui_hint'}
_MEANING_FIELDS = ('mission_id', 'kind', 'subject_refs', 'context_ref',
                   'answer_type', 'answer_resolution_ref',
                   *_CONTRACT_MEANING_FIELDS)
_ANSWER_FIELDS = ('answer', 'executable_answer', 'answer_history', 'structural_answer_history',
                  'authored_status', 'evidence_status', 'check_results', 'analysis_scope',
                  'evidence', 'revision_index', 'reviewed_through_observation',
                  'child_answers', 'reviewed_child_answers', 'research_return',
                  'last_program_result', 'native_condition_research', 'native_structure_planning', 'semantic_answer',
                  'semantic_answer_history', 'query_execution', 'applied_question_use',
                  'support_paths', 'support_path_status', 'representation_diagnostic',
                  'representation_assessments', 'compilation', 'pending_author_work',
                  'pending_query', 'pending_query_source', 'parent_use_history',
                  'routing_execution', 'routing_history', 'completed_agenda_basis',
                  'resolution_experiments', 'conditional_continuation', 'frozen_recall_execution',
                  'revision_candidate_exports', 'revision_export_errors')
_ANSWER_TYPES = {'number', 'vector', 'interval', 'distribution', 'boolean_claim', 'binding',
                 'program', 'relation', 'scoped_unknown', 'scoped_outcome'}
_OUTCOME_PARTITION_STATES = {
    'SUPPORTED_IN_SCOPE', 'NOT_DETECTED_UNDER_PROTOCOL', 'REFUTED_IN_SCOPE',
    'BOUNDARY_DEGENERATE', 'SENSOR_MISMATCH', 'MODEL_MISMATCH',
    'UNKNOWN', 'PROTOCOL_ERROR', 'EXECUTION_ERROR',
}


def _strip_cosmetic(value):
    if isinstance(value, dict):
        return {k: _strip_cosmetic(v) for k, v in value.items() if k not in _COSMETIC_CONTRACT_KEYS}
    if isinstance(value, list):
        return [_strip_cosmetic(item) for item in value]
    return deepcopy(value)


def native_evaluation_contract(semantics):
    """Validate an explicitly retained native predictor's startup semantics."""
    evaluation = semantics.get('evaluation') if isinstance(semantics, dict) else None
    if evaluation is None:
        return None
    if not isinstance(evaluation, dict) or evaluation.get('kind') != 'native_evaluation':
        raise ValueError('evaluation needs kind native_evaluation')
    fallback = evaluation.get('fallback')
    if type(fallback) not in (int, float) or not isfinite(fallback):
        raise ValueError('native evaluation needs a declared finite fallback')
    reference = evaluation.get('source_ref')
    if not isinstance(reference, dict):
        raise ValueError('native evaluation needs a retained source_ref')
    location = (isinstance(reference.get('file'), str) and bool(reference['file'])) or all(
        isinstance(reference.get(key), str) and reference[key] for key in ('archive', 'unit', 'member'))
    field = (isinstance(reference.get('field'), str) and bool(reference['field'])) or (
        isinstance(reference.get('json_path'), list) and bool(reference['json_path']))
    if not location or not field:
        raise ValueError('native evaluation needs a retained source_ref location and field')
    return evaluation


def _meaning(question):
    meaning = {}
    for key in _MEANING_FIELDS:
        if key in question:
            meaning[key] = (_strip_cosmetic(question[key]) if key in _CONTRACT_MEANING_FIELDS
                            else deepcopy(question[key]))
    query = question.get('query_semantics')
    if isinstance(query, dict):
        meaning['query_semantics'] = {key: deepcopy(query[key]) for key in
                                     ('mode', 'query_expression', 'evaluation') if key in query}
    return meaning


def adopt_question_semantics(existing, incoming):
    """Merge one ID without mistaking an algorithm or new evidence for meaning.

    An explicit change of objective, context or answer resolution keeps the old
    answer under its old meaning version. The original assignment remains fixed.
    """
    if not isinstance(incoming, dict) or not incoming.get('id'):
        raise ValueError('research_question needs an id')
    if existing is None:
        native_evaluation_contract(incoming.get('query_semantics'))
        return {**deepcopy(incoming), 'meaning_version': 1}
    if existing['id'] != incoming['id']:
        raise ValueError('a question meaning update must retain its question id')
    merged = {**deepcopy(existing), **deepcopy(incoming)}
    if 'mission' in incoming:
        from .question_agenda import adopt_research_mission
        merged['mission'] = adopt_research_mission(existing.get('mission', {}), incoming['mission'])
    native_evaluation_contract(merged.get('query_semantics'))
    version = existing.get('meaning_version', 1)
    history = deepcopy(existing.get('meaning_history', []))
    if _meaning(existing) != _meaning(merged):
        snapshot = {key: deepcopy(existing[key]) for key in
                    ('id', 'question', 'parent_id', 'depends_on', 'status', *_MEANING_FIELDS,
                     'query_semantics', 'use_sites', 'theory_target', 'native_research_ref',
                     'mission', 'queue', 'frozen_library_path', 'exploration_policy',
                     'research_comparison', 'allowed_mappings', *_ANSWER_FIELDS) if key in existing}
        snapshot['meaning_version'] = version
        history.append(snapshot)
        for key in _ANSWER_FIELDS:
            merged.pop(key, None)
        merged.update(status='open', evidence=[])
        version += 1
    merged['meaning_version'] = version
    if history:
        merged['meaning_history'] = history
    else:
        merged.pop('meaning_history', None)
    if 'assignment' in existing:
        merged['assignment'] = deepcopy(existing['assignment'])
    return merged


def compile_question(question, mission=None):
    """Return a concrete query or the author work still needed to execute it."""
    result = {'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1),
              'answer_type': question.get('answer_type'),
              'use_sites': deepcopy(question.get('use_sites', [])), 'query': None}

    def finish(status, reason, **extra):
        return {**result, 'status': status, 'reason': reason, **extra}

    semantics = question.get('query_semantics')
    if question.get('kind') in {'score_improvement', 'engineering_action'}:
        return finish('idea', 'not_a_scientific_question')
    if not _meaning(question):
        return finish('idea', 'legacy_question_has_no_executable_meaning')
    if not question.get('answer_type'):
        return finish('needs_operationalization', 'missing_answer_type')
    if question['answer_type'] not in _ANSWER_TYPES:
        return finish('needs_operationalization', 'unsupported_answer_type')
    if not question.get('answer_resolution_ref'):
        return finish('needs_operationalization', 'missing_answer_resolution')
    purpose = (mission or {}).get('purpose')
    if not result['use_sites'] and purpose not in {'theory_learning', 'method_learning'}:
        return finish('deferred', 'no_declared_use')
    if not isinstance(semantics, dict) or semantics.get('mode') not in {'model_query', 'solution_relation'}:
        return finish('needs_operationalization', 'missing_query_semantics')
    native_evaluation_contract(semantics)
    query = semantics.get('query_expression')
    if not isinstance(query, dict) or not query.get('op'):
        object_type = question['answer_type'] if semantics['mode'] == 'solution_relation' else 'query_expression'
        return finish('author_work_request', 'missing_executable_query', request={
            'type': 'AuthorWorkRequest', 'object_type': object_type,
            'required_contract': {'answer_type': question['answer_type'],
                                  'answer_resolution_ref': deepcopy(question['answer_resolution_ref']),
                                  'query_mode': semantics['mode']}})
    return finish('ready', 'executable_query_supplied', query=deepcopy(query))


def validate_answer_type(value, answer_type):
    """Check the JSON answer shape used by executable query results."""
    def number(item):
        return isinstance(item, (int, float)) and not isinstance(item, bool) and isfinite(item)

    def interval(item):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return all(number(x) for x in item) and item[0] <= item[1]
        if isinstance(item, dict) and 'lower' in item and 'upper' in item:
            return number(item['lower']) and number(item['upper']) and item['lower'] <= item['upper']
        return False

    if answer_type == 'number':
        return number(value)
    if answer_type == 'vector':
        return isinstance(value, list) and bool(value) and all(number(item) for item in value)
    if answer_type == 'interval':
        return interval(value) or (isinstance(value, dict) and bool(value)
                                   and all(interval(item) for item in value.values()))
    if answer_type == 'boolean_claim':
        return isinstance(value, bool)
    if answer_type == 'distribution':
        return (isinstance(value, dict) and bool(value)
                and all(number(item) and item >= 0 for item in value.values())
                and abs(sum(value.values()) - 1) <= 1e-9)
    if answer_type in {'binding', 'relation'}:
        return isinstance(value, dict)
    if answer_type == 'program':
        return isinstance(value, (str, dict)) and bool(value)
    if answer_type == 'scoped_unknown':
        return isinstance(value, dict) and value.get('status') in {'unknown', 'scoped_unknown'}
    if answer_type == 'scoped_outcome':
        return (isinstance(value, dict)
                and value.get('outcome_status') in _OUTCOME_PARTITION_STATES)
    return False


def evaluate_decision_region(region, surviving_candidates, *, candidate_losses=None):
    """Evaluate HEC-style action-bound decision region stopping without vacuous pass.

    A region must bind a concrete executable ``action_ref`` and ``output_contract``
    (never a bare precision label like 'L3'). Empty surviving candidates always
    route to ``MODEL_MISMATCH`` rather than vacuously satisfying subset inclusion.
    """
    if not isinstance(region, dict):
        raise ValueError('decision_region must be a dict')
    action_ref = region.get('action_ref')
    output_contract = region.get('output_contract')
    if (not isinstance(action_ref, str) or not action_ref
            or action_ref in {'L0', 'L1', 'L2', 'L3', 'L4'}):
        raise ValueError('decision_region requires a concrete executable action_ref, not a layer label')
    if not isinstance(output_contract, dict) or not output_contract:
        raise ValueError('decision_region requires a non-empty output_contract')
    surviving = list(surviving_candidates or ())
    if not surviving:
        return {
            'status': 'MODEL_MISMATCH',
            'stop_discrimination': False,
            'action_ref': None,
            'surviving_count': 0,
            'requires_independent_confirmation': True,
        }
    compatible = set(region.get('compatible_candidate_refs', ()))
    tol = float(region.get('tolerated_loss', 0.0))
    if candidate_losses is not None:
        all_within = all(
            cand in candidate_losses and isfinite(float(candidate_losses[cand]))
            and float(candidate_losses[cand]) <= tol
            for cand in surviving
        )
    else:
        all_within = bool(compatible) and set(surviving) <= compatible
    return {
        'status': 'DECISION_EQUIVALENT_IN_SCOPE' if all_within else 'NEEDS_FURTHER_DISCRIMINATION',
        'stop_discrimination': bool(all_within),
        'action_ref': action_ref if all_within else None,
        'surviving_count': len(surviving),
        'requires_independent_confirmation': True,
        'independent_confirmation_ref': deepcopy(region.get('independent_confirmation_ref')),
    }


_INSTANCE_FIELDS = ('body_ref', 'parameters', 'fitted_state_ref', 'training_cutoff',
                    'binding_ref', 'representation_ref')


def adopt_claim_instance(existing, incoming):
    """Version an explicitly named family instance, without inferring equivalence.

    Changing a fit does not create a new family. A different declared family is
    retained separately rather than being silently merged by file or text match.
    """
    family = incoming.get('family_id', incoming.get('claim_id'))
    if not family:
        raise ValueError('an explicit claim family_id or claim_id is required')
    if existing is None:
        return {**deepcopy(incoming), 'family_id': family, 'instance_version': 1}
    if existing.get('family_id', existing.get('claim_id')) != family:
        previous = deepcopy(existing.get('family_history', []))
        previous.append({key: deepcopy(value) for key, value in existing.items() if key != 'family_history'})
        return {**deepcopy(incoming), 'family_id': family, 'instance_version': 1,
                'family_history': previous}
    merged = {**deepcopy(existing), **deepcopy(incoming), 'family_id': family}
    version = existing.get('instance_version', 1)
    history = deepcopy(existing.get('instance_history', []))
    if any(existing.get(key) != merged.get(key) for key in _INSTANCE_FIELDS):
        history.append({key: deepcopy(value) for key, value in existing.items()
                        if key not in {'instance_history', 'family_history'}})
        version += 1
    merged['instance_version'] = version
    if history:
        merged['instance_history'] = history
    return merged

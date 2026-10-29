"""Deterministic question proposals and explicitly funded task/theory queues.

Proposals reference existing evidence. They neither diagnose a unique mechanism
nor invent its formula. Scheduling is an event order, not a scientific score.
The caller owns execution and the exact completed_agenda_basis cache contract.
"""

from copy import deepcopy
from math import isfinite


_FIELDS = ('id', 'question', 'kind', 'parent_id', 'depends_on', 'context_ref',
           'answer_type', 'answer_resolution_ref', 'query_semantics', 'use_sites',
           'mission', 'mission_id', 'meaning_version', 'agenda_basis')
_THEORY_PURPOSES = {'theory_learning', 'method_learning'}
_SUPPORTED = {'supported', 'supported_in_scope', 'supported_on_later_readings', 'confirmed'}
_QUERY_OPS = {'Readout', 'Difference', 'RelationResidual', 'CheckApplication',
              'AllOnFiniteSet', 'ParameterRegion', 'FindSharedStructure',
              'FindDistinguishingExperiment', 'MethodEffect'}


def question_work_basis(question):
    """Use one basis for scheduling, completed answers, and pending author work."""
    return deepcopy(question.get('agenda_basis') or {
        'meaning_version': question.get('meaning_version', 1),
        'answer_version': question.get('semantic_answer', question.get('executable_answer', {})).get('version')})


def has_executable_query(question):
    expression = (question.get('query_semantics') or {}).get('query_expression')
    return isinstance(expression, dict) and expression.get('op') in _QUERY_OPS


def has_method_use_plan(question):
    plan = question.get('method_use_plan')
    return (question.get('kind') == 'method_use' and isinstance(plan, dict)
            and (bool(plan.get('native_return_refs')) or (
                isinstance(plan.get('candidate_id'), str) and bool(plan['candidate_id'])
                and isinstance(plan.get('binding'), dict))))


def merge_question_proposal(existing, proposal):
    """A generated missing-contract placeholder cannot erase an authored query."""
    if not (proposal.get('author_work_request') and not has_executable_query(proposal)
            and has_executable_query(existing)):
        return {**existing, **deepcopy(proposal)}
    merged = deepcopy(existing)
    for key in ('agenda_basis', 'origin', 'queue', 'mission_id'):
        if key in proposal:
            merged[key] = deepcopy(proposal[key])
    if 'mission' in proposal:
        merged['mission'] = adopt_research_mission(existing.get('mission', {}), proposal['mission'])
    for use in proposal.get('use_sites', []):
        if use not in merged.setdefault('use_sites', []):
            merged['use_sites'].append(deepcopy(use))
    merged.pop('author_work_request', None)
    return merged


def _records(values):
    return list(values.values()) if isinstance(values, dict) else list(values or [])


def _reference(kind, item, identifier=None):
    reference = {'kind': kind, 'id': identifier if identifier is not None else item['id']}
    version = item.get('version', item.get('meaning_version'))
    if version is not None:
        reference['version'] = version
    if item.get('source_ref') is not None:
        reference['source_ref'] = deepcopy(item['source_ref'])
    return reference


def _owner(question, mission):
    return question.get('mission') or mission or {}


def effective_question_mission(question, fallback=None):
    """Return the retained owner's actual dict, so its usage survives a step."""
    return _owner(question or {}, fallback)


def mission_contract(mission):
    """Usage changes remaining allowance, not the meaning of a cached query."""
    return {key: deepcopy(value) for key, value in mission.items() if key != 'resource_usage'}


def adopt_research_mission(previous, incoming):
    """Repeated declarations of one mission retain its actual compute debit."""
    current = deepcopy(incoming)
    same = (previous.get('mission_id') == incoming.get('mission_id')
            if previous.get('mission_id') or incoming.get('mission_id') else
            previous.get('purpose') == incoming.get('purpose'))
    if same and 'resource_usage' in previous:
        current['resource_usage'] = deepcopy(previous['resource_usage'])
    return current


def _remaining_theory_compute(mission):
    if mission.get('purpose') not in _THEORY_PURPOSES:
        return None
    funding = mission.get('allocation', mission.get('theory_learning_allocation', 0))
    funding = funding.get('compute', 0) if isinstance(funding, dict) else funding
    if type(funding) is not int or funding < 0:
        raise ValueError('theory allocation needs nonnegative integer compute')
    return max(0, funding - mission.get('resource_usage', {}).get('compute', 0))


def _mission_usage(agent, mission):
    """Questions bearing one stable mission ID spend one retained allocation."""
    peers = [mission]
    if mission.get('mission_id'):
        peers.extend(owner for owner in [agent.research_mission,
            *(question.get('mission', {}) for question in agent.questions)]
            if owner.get('mission_id') == mission['mission_id'])
    usage = max((owner.get('resource_usage', {}) for owner in peers),
                key=lambda item: item.get('compute', 0))
    usage.setdefault('compute', 0)
    for owner in peers:
        owner['resource_usage'] = usage
    return usage


def mission_compute_remaining(agent, mission):
    available = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
    if mission.get('purpose') in _THEORY_PURPOSES:
        _mission_usage(agent, mission)
    funded = _remaining_theory_compute(mission)
    return available if funded is None else min(available, funded)


def debit_mission_compute(agent, mission, before_total_compute):
    """Debit one directly executed segment; nested operators debit themselves."""
    actual = agent.costs['compute_cost'] - before_total_compute
    if actual and mission.get('purpose') in _THEORY_PURPOSES:
        usage = _mission_usage(agent, mission)
        usage['compute'] = usage.get('compute', 0) + actual
    return actual


def _uses(question, candidates):
    uses = deepcopy(question.get('use_sites', []))
    for model, candidate in candidates.items():
        for slot, bound in candidate.get('inputs', {}).items():
            if isinstance(bound, dict) and bound.get('question_id') == question['id']:
                use = {'kind': ('bind_parameter_region' if bound.get('kind') == 'parameter_region'
                                else 'supply_component'), 'model': model, 'slot': slot}
                if use not in uses:
                    uses.append(use)
    return uses


def _proposal(identifier, text, kind, origin, references, *, source=None, queue='task',
              uses=None, mission=None, basis=None, required_output=None):
    source = source or {}
    proposal = {key: deepcopy(source[key]) for key in _FIELDS if key in source}
    proposal.update(id=identifier, question=text, kind=kind, queue=queue,
                    answer_type=source.get('answer_type', 'relation'),
                    answer_resolution_ref=deepcopy(source.get('answer_resolution_ref') or {
                        'required': 'execute declared check in its bound scope; otherwise retain unknown'}),
                    use_sites=deepcopy(uses if uses is not None else source.get('use_sites', [])),
                    origin={'kind': origin, 'source_refs': references},
                    agenda_basis=deepcopy(basis or {}))
    if mission:
        proposal['mission'] = deepcopy(mission)
        if mission.get('mission_id'):
            proposal['mission_id'] = mission['mission_id']
    # A supplied resolution query is usable; prose or Program structure is not
    # silently compiled into the scientifically correct query.
    if isinstance(source.get('resolution_query'), dict):
        proposal['query_semantics'] = {'mode': 'model_query',
                                       'query_expression': deepcopy(source['resolution_query'])}
    expression = (proposal.get('query_semantics') or {}).get('query_expression')
    if not isinstance(expression, dict) or expression.get('op') not in _QUERY_OPS:
        proposal['query_semantics'] = {'mode': 'solution_relation'}
        proposal['author_work_request'] = {
            'reason': origin, 'required_output': required_output or
                'Author an executable query or candidate and its declared parent use; preserve counterexamples.'}
    return proposal


def generate_questions(questions, research_candidates, theories, *, mission=None, matching=None):
    """Propose work from six real sources, without new observations or induction."""
    questions, theories = _records(questions), _records(theories)
    candidates, matching = research_candidates or {}, matching or {}
    by_id = {question['id']: question for question in questions}
    proposals = []

    # Measured false claims remain useful research inputs even when none of the
    # compared drafts survived. They do not identify which shared premise failed.
    refutations = []
    for question in questions:
        answer = question.get('semantic_answer') or {}
        query = (question.get('query_semantics') or {}).get('query_expression') or {}
        if (question.get('kind') != 'evidence' or answer.get('finite_status') != 'observed_false'
                or answer.get('value') is not False or not answer.get('evidence')
                or query.get('op') != 'AllOnFiniteSet'):
            continue
        group = next((items for items in refutations if all(
            items[0].get(key) == question.get(key) for key in ('parent_id', 'context_ref'))
            and items[0]['semantic_answer'].get('evidence') == answer['evidence']), None)
        if group is None:
            refutations.append([question])
        else:
            group.append(question)
    for group in refutations:
        premises = [item.get('premise_refs', []) for item in group]
        common = [ref for ref in premises[0] if all(ref in refs for refs in premises[1:])]
        distinct = [{**_reference('question', item), 'premise_refs': [
            ref for ref in refs if ref not in common]} for item, refs in zip(group, premises)]
        source = group[0]
        references = [{**_reference('question', item),
            'answer_version': item['semantic_answer'].get('version')}
            for item in group]
        parent = by_id.get(source.get('parent_id'), {})
        # Continue an explicitly bound revision, rather than copying the
        # already answered boolean check as the new research method.
        owner = next((item for item in (parent, *group)
            if item.get('native_representation_plan')), None)
        declared = owner or source
        execution_source = {key: deepcopy(declared[key]) for key in
            ('parent_id', 'context_ref', 'depends_on') if key in declared}
        execution_source['answer_type'] = 'program'
        if owner is not None:
            execution_source['answer_resolution_ref'] = {
                'evaluator': 'native_representation_and_paid_candidate_comparison',
                'source_question_ref': _reference('question', owner)}
        existing_revision = owner is parent and parent.get('kind') == 'program_revision'
        if not existing_revision and source.get('parent_id') is not None:
            execution_source['parent_id'] = source['parent_id']
        identifier = (parent['id'] if existing_revision else
                      f'agenda:counterexamples:{source["id"]}')
        proposal = _proposal(identifier,
            'Which retained premise or executable revision explains these measured prediction failures?',
            'program_revision', 'measured_counterexamples', references,
            source=execution_source, uses=_uses(declared, candidates),
            mission=_owner(declared, mission), basis={'refuted_answers': references,
                'evidence_rows': deepcopy(source['semantic_answer']['evidence']),
                'common_premise_refs': deepcopy(common), 'distinct_premise_refs': distinct,
                'exhaustive_candidate_pool': False},
            required_output='Bind the refuted forecasts to their actual candidate versions and parent use, '
                'then supply an executable premise check or revision and its comparison budget. '
                'The measured failures do not refute every component or select a true winner.')
        if owner is not None:
            for key in ('native_representation_plan', 'representation_model', 'theory_target',
                        'representation_diagnostic', 'representation_research_basis', 'subject_refs'):
                if key in owner:
                    proposal[key] = deepcopy(owner[key])
            proposal.pop('author_work_request', None)
        proposals.append(proposal)

    # Failed predictions supply suspicious declared premises, not a diagnosis.
    for question in questions:
        failure = question.get('failed_confirmation') or {}
        mse = failure.get('mse')
        if (failure.get('passed') is not False or failure.get('returncode') != 0
                or failure.get('timeout') or type(mse) not in (int, float) or not isfinite(mse)):
            continue
        premises = question.get('premise_refs', failure.get('premise_refs', []))
        for premise in premises:
            identifier = premise if isinstance(premise, str) else premise.get('question_id', premise.get('id'))
            support = by_id.get(identifier)
            if support is None:
                continue
            proposals.append(_proposal(support['id'], support.get('question', 'Resolve the declared premise'),
                support.get('kind', 'applicability'), 'failed_prediction_premise',
                [_reference('question', question), _reference('question', support)], source=support,
                uses=_uses(support, candidates), mission=_owner(support, mission),
                basis={'failure_rows': deepcopy(failure.get('observation_indices', [])),
                       'premise_version': support.get('meaning_version', 1)}))

    # Existing bindings expose actual consumers; no dependency is invented.
    for model, candidate in candidates.items():
        if candidate.get('status') in {'needs_rebuild', 'needs_refit', 'new_paid_rows_unscored'}:
            ids = [entry.get('question_id') for entry in candidate.get('inputs', {}).values()
                   if isinstance(entry, dict) and entry.get('question_id')]
            if candidate.get('changed_question'):
                ids.append(candidate['changed_question'])
            for identifier in dict.fromkeys(ids):
                question = by_id.get(identifier)
                if question is None:
                    continue
                proposals.append(_proposal(identifier, question.get('question', 'Refresh the declared parent input'),
                    question.get('kind', 'parameter_region'), 'stale_binding',
                    [_reference('candidate', candidate, model), _reference('question', question)],
                    source=question, uses=_uses(question, candidates), mission=_owner(question, mission),
                    basis={'candidate_status': candidate['status'], 'paid_rows': candidate.get('paid_rows'),
                           'input_versions': {slot: entry.get('version') for slot, entry in
                                candidate.get('inputs', {}).items() if isinstance(entry, dict)}}))
        diagnostic = candidate.get('current_evaluation', {}).get('representation_diagnostic')
        if not diagnostic:
            diagnostic = candidate.get('current_evaluation', {}).get('training_representation_diagnostic')
        if diagnostic and diagnostic.get('conflicting_groups', 0) and diagnostic.get('witnesses'):
            source = next((q for q in questions if q.get('id') == f'representation:{model}'), {})
            proposals.append(_proposal(source.get('id', f'agenda:representation:{model}'),
                source.get('question', 'Which captured distinctions explain the paid representation collision?'),
                'representation', 'representation_witness', [_reference('candidate', candidate, model)],
                source=source, uses=source.get('use_sites') or [{'kind': 'supply_component', 'model': model}],
                mission=_owner(source, mission), basis={'paid_rows': candidate.get('paid_rows'),
                    'empirical_mse_floor': diagnostic.get('empirical_mse_floor'),
                    'witness_rows': [[row.get('row_index') for row in group.get('rows', [])]
                                     for group in diagnostic['witnesses']]},
                required_output='Author a candidate representation or premise check against the referenced witnesses; no hidden window is supplied.'))

        basis = candidate.get('representation_research_basis')
        if basis:
            source = next((q for q in questions if q.get('id') == f'representation:{model}'), {})
            proposal = _proposal(source.get('id', f'agenda:representation:{model}'),
                'Does the proposed observable distinction improve this failed parent prediction?',
                'program_revision', 'representation_alternative', [_reference('candidate', candidate, model)],
                source=source, uses=source.get('use_sites') or [{'kind': 'supply_component', 'model': model}],
                mission=_owner(source, mission), basis={'paid_rows': candidate.get('paid_rows'),
                    'representation_research_basis': deepcopy(basis)},
                required_output='Check the referenced failures and public distinction; provide a bounded '
                    'native plan or an executable candidate with its parent use.')
            proposal['representation_research_basis'] = deepcopy(basis)
            if source.get('native_representation_plan'):
                proposal['native_representation_plan'] = deepcopy(source['native_representation_plan'])
            proposals.append(proposal)

    for question in questions:
        answer = question.get('executable_answer') or {}
        region = answer.get('parameter_region') or question.get('parameter_region')
        uses = _uses(question, candidates)
        if isinstance(region, dict) and region.get('lower') is not None and region.get('upper') is not None \
                and region['lower'] < region['upper'] and uses:
            proposals.append(_proposal(f'agenda:parameters:{question["id"]}',
                'Do the remaining compatible parameter values change the declared parent decision?',
                'distinguish', 'unidentified_parameters', [_reference('question', question)],
                uses=uses, mission=_owner(question, mission), source={
                    'context_ref': question.get('context_ref'), 'answer_type': 'relation'},
                basis={'answer_version': answer.get('version'), 'parameter_region': deepcopy(region)},
                required_output='Provide task-relevant candidate consequences or an executable sufficiency check; an interval representative does not exclude the family.'))

        target = question.get('theory_target')
        if not isinstance(target, dict):
            continue
        result = matching.get(question['id'], {})
        matches = result.get('matches', [])
        if result.get('status') not in {'REBIND_THEN_TEST', 'REFIT_FAMILY', 'SCOPED_REUSE', 'UNKNOWN_MATCH'}:
            continue
        if not any(match.get('source_id') for match in matches):
            continue
        references = [_reference('question', question)]
        references.extend({'kind': 'theory', 'id': match['source_id']}
                          for match in matches if match.get('source_id'))
        application = {'context_ref': target.get('context_ref', question.get('context_ref')),
                       'answer_type': 'binding'}
        expression = (question.get('query_semantics') or {}).get('query_expression')
        if isinstance(question.get('application_query'), dict):
            application.update(resolution_query=question['application_query'],
                               answer_type=question.get('answer_type', 'relation'))
        elif isinstance(expression, dict) and expression.get('op') == 'CheckApplication':
            application.update(query_semantics=question['query_semantics'],
                               answer_type=question.get('answer_type', 'relation'))
        proposals.append(_proposal(f'agenda:application:{question["id"]}',
            'Does the proposed old-theory binding apply in the declared target context?',
            'applicability', 'theory_transfer', references, source=application,
            uses=question.get('use_sites', []), mission=_owner(question, mission),
            basis={'matching_status': result['status'], 'index_version': result.get('index_version'),
                   'target_context': deepcopy(target.get('context_ref'))},
            required_output='Supply and test the declared role/unit/context mapping before transferring support.'))

    # Cross-episode construction is a mission obligation, not PR3 induction.
    supported = [theory for theory in theories if theory.get('program') and theory.get('context_ref') is not None and
                 theory.get('support', {}).get('status') in _SUPPORTED and
                 theory.get('support', {}).get('evidence')]
    owner = mission or {}
    if owner.get('purpose') in _THEORY_PURPOSES and len(supported) >= 2:
        distinct_contexts = []
        for theory in supported:
            if theory.get('context_ref') not in distinct_contexts:
                distinct_contexts.append(theory.get('context_ref'))
        if len(distinct_contexts) >= 2:
            proposals.append(_proposal(f'agenda:shared:{owner.get("mission_id", "theory")}',
                'Can the retained episodes yield a reconstructible shared structure and a new-source test?',
                'structural_relation', 'multi_episode_commonality',
                [_reference('theory', theory) for theory in supported], queue='theory', mission=owner,
                uses=[{'kind': 'supply_theory_library_candidate', 'mission_id': owner.get('mission_id')}],
                basis={'source_versions': [ref for ref in
                    (_reference('theory', theory) for theory in supported)]},
                required_output='Construct an abstraction, source reconstructions, costs, retained counterexample checks and a transfer test; no induction is executed here.'))
    groups = []
    for theory in supported:
        if theory.get('declared_scope_ref') is None:
            continue
        group = next((items for items in groups if
                      items[0]['context_ref'] == theory['context_ref'] and
                      items[0]['declared_scope_ref'] == theory['declared_scope_ref']), None)
        if group is None:
            groups.append([theory])
        else:
            group.append(theory)
    for group in groups:
        if owner.get('purpose') not in _THEORY_PURPOSES or len(group) < 2 or \
                all(item['program'] == group[0]['program'] for item in group):
            continue
        proposals.append(_proposal(f'agenda:contrast:{group[0]["id"]}',
            'Are these retained theories incompatible, special cases, or equivalent for the declared task?',
            'structural_relation', 'theory_conflict_or_redundancy',
            [_reference('theory', theory) for theory in group], queue='theory', mission=owner,
            uses=[{'kind': 'supply_theory_library_candidate', 'mission_id': owner.get('mission_id')}],
            basis={'source_versions': [_reference('theory', theory) for theory in group]},
            required_output='Author a common physically bound contrast; different programs alone do not establish a conflict.'))

    # Multiple real events may request the same existing premise. Keep all
    # provenance without multiplying that question or its evidence weight.
    merged = {}
    for proposal in proposals:
        prior = merged.get(proposal['id'])
        if prior is not None:
            for reference in proposal['origin']['source_refs']:
                if reference not in prior['origin']['source_refs']:
                    prior['origin']['source_refs'].append(reference)
            prior['agenda_basis'].update(proposal['agenda_basis'])
            for use in proposal['use_sites']:
                if use not in prior['use_sites']:
                    prior['use_sites'].append(use)
        else:
            merged[proposal['id']] = proposal
    return list(merged.values())


def _allocated(owner):
    allocation = owner.get('allocation', owner.get('theory_learning_allocation'))
    if isinstance(allocation, dict):
        return any(type(value) in (int, float) and value > 0 for value in allocation.values())
    return type(allocation) in (int, float) and allocation > 0


def build_agenda(questions, research_candidates, theories, *, mission=None, budget=None, matching=None,
                 parent_research=None):
    """Retain both queues; select only declared, funded work in a fixed order."""
    original = _records(questions)
    combined = {question['id']: dict(question) for question in original}
    for proposal in generate_questions(original, research_candidates, theories,
                                       mission=mission, matching=matching):
        combined[proposal['id']] = merge_question_proposal(combined.get(proposal['id'], {}), proposal)
    budget, mission = budget or {}, mission or {}
    queues = {'task_agenda': [], 'theory_agenda': []}
    deferred = []
    def pending_dependency(question):
        origin = question.get('origin')
        origin_kind = origin.get('kind') if isinstance(origin, dict) else origin
        return (question.get('agenda_basis', {}).get('candidate_status') in
                {'needs_rebuild', 'needs_refit', 'new_paid_rows_unscored'} or
                origin_kind == 'representation_witness' or
                (question.get('semantic_answer', {}).get('status') == 'answered'
                 and question.get('use_sites') and not question.get('applied_question_use')))

    ordered = [q for q in combined.values() if pending_dependency(q)]
    ordered.extend(q for q in combined.values() if not pending_dependency(q))
    for question in ordered:
        retained = (parent_research or {}).get(question['id'])
        if retained is not None:
            for work in retained:
                entry = {**deepcopy(work), 'queue': 'task', 'kind': question.get('kind'),
                         'work_kind': 'parent_revision', 'status': 'ready'}
                if work['state'] in {'completed_no_gain', 'refuted'}:
                    entry.update(status='completed', reason=work['state'])
                elif work['state'] not in {'unstarted', 'budget_incomplete'}:
                    entry.update(status='deferred', reason=work['state'])
                elif min(budget.get('compute', 0), work.get('available_compute', budget.get('compute', 0))) \
                        < work['minimum_compute_for_progress']:
                    entry.update(status='deferred', reason='minimum_progress_compute_unfunded')
                if entry['status'] == 'deferred':
                    deferred.append({key: deepcopy(entry[key]) for key in
                                     ('question_id', 'meaning_version', 'queue', 'reason')})
                queues['task_agenda'].append(entry)
            continue
        owner = _owner(question, mission)
        queue = question.get('queue', 'theory' if question.get('mission', {}).get('purpose') in _THEORY_PURPOSES else 'task')
        expression = (question.get('query_semantics') or {}).get('query_expression')
        region = question.get('executable_answer', {})
        checked = (region.get('status') == 'supported' and region.get('evidence_status') == 'supported'
                   and region.get('parameter_region'))
        origin = question.get('origin')
        consolidation = (isinstance(origin, dict) and origin.get('kind') == 'multi_episode_commonality'
                         and not has_executable_query(question))
        basis = question_work_basis(question)
        pending = question.get('pending_author_work')
        cached_request = pending is not None and pending.get('agenda_basis') == basis
        if cached_request and pending.get('bounded_status') == 'UNKNOWN':
            available = budget.get('compute', 0)
            funding = _remaining_theory_compute(owner)
            if funding is not None:
                available = min(available, funding)
            cached_request = available <= pending.get('bounded_allowance', 0)
        native_plan = question.get('native_representation_plan')
        native_representation = (question.get('kind') in {'representation', 'program_revision'}
            and isinstance(native_plan, dict) and bool(native_plan.get('parent_theory_ids'))
            and isinstance(native_plan.get('validation_records'), list)
            and type(native_plan.get('compute_limit')) is int)
        work = ('method_conditions' if has_method_use_plan(question) else
                'native_representation' if native_representation else
                'author_work_request' if cached_request else 'consolidation' if consolidation
                else 'checked_answer' if checked else 'query'
                if isinstance(expression, dict) and expression.get('op') in _QUERY_OPS
                else 'author_work_request')
        entry = {'question_id': question['id'], 'queue': queue, 'kind': question.get('kind'),
                 'work_kind': work, 'agenda_basis': basis, 'status': 'ready'}
        reason = None
        if question.get('completed_agenda_basis') == basis:
            entry['status'] = 'completed'
            entry['reason'] = 'unchanged_completed_work'
        elif queue == 'theory' and owner.get('purpose') not in _THEORY_PURPOSES:
            reason = 'no_theory_mission'
        elif queue == 'theory' and not _allocated(owner):
            reason = 'theory_allocation_missing'
        elif (queue == 'theory' and work in {'query', 'consolidation', 'native_representation', 'method_conditions'}
              and _remaining_theory_compute(owner) == 0):
            reason = 'theory_compute_budget'
        elif not question.get('use_sites') and queue == 'task':
            reason = 'no_declared_use'
        elif work in {'query', 'consolidation', 'native_representation', 'method_conditions'} and budget.get('compute', 0) <= 0:
            reason = 'compute_budget'
        elif native_representation and native_plan['compute_limit'] <= 0:
            reason = 'native_representation_compute_budget'
        elif work == 'author_work_request' and budget.get('author_calls', 0) <= 0:
            reason = 'author_work_unfunded'
        if reason:
            entry.update(status='deferred', reason=reason)
            deferred.append({'question_id': question['id'], 'queue': queue, 'reason': reason})
        queues[f'{queue}_agenda'].append(entry)

    result = {**queues, 'deferred': deferred, 'selected_question_id': None,
              'selected_queue': None, 'selected_action': None, 'status': 'unresolved'}
    if mission.get('task_supported') is True and mission.get('supported_action') is not None \
            and mission.get('phase') != 'theory':
        result.update(status='task_supported', selected_queue='task',
                      selected_action=deepcopy(mission['supported_action']))
        return result
    theory_phase = (mission.get('phase') == 'theory' or
                    ('phase' not in mission and mission.get('purpose') in _THEORY_PURPOSES))
    order = ('theory', 'task') if theory_phase else ('task', 'theory')
    for queue in order:
        selected = next((entry for entry in queues[f'{queue}_agenda'] if entry['status'] == 'ready'), None)
        if selected:
            result.update(status='selected', selected_question_id=selected['question_id'], selected_queue=queue)
            if selected.get('question_ref'):
                result['selected_question_ref'] = deepcopy(selected['question_ref'])
            break
    return result

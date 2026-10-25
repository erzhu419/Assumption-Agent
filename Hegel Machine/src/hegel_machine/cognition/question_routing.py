"""Connect typed Program recall, question agendas and finite plan selection.

The index holds contracts or references into the existing candidate banks. This
route consumes paid evidence and prepares ordinary query arguments; acquisition
and confirmation remain in Common's existing executor.
"""

from copy import deepcopy
from dataclasses import replace

from .language import Program
from .research_episodes import _child_ref, load_source_values
from .research_assessments import research_comparison_basis, resolve_research_comparison
from .question_agenda import (adopt_research_mission, build_agenda, debit_mission_compute,
    effective_question_mission, generate_questions, mission_compute_remaining, mission_contract,
    has_executable_query, merge_question_proposal, question_work_basis)
from .question_continuation import (step_question, _versioned_support, executed_plan_records,
                                    parent_revision_obligation, select_parent_revision_work)
from .question_selection import select_question_plan
from .question_queries import evaluate_on_evidence, numeric_answer_distance
from .theory_matching import match_theories, prepare_theories_for_question


ROUTING_INTERFACE = """
run {operator:'research_route',research_question:{id:...,...},theories:[...],
 experiments:[...],mission:{...}} runs computable PR2 research with no model call.
Theories declare id, Program JSON or source_ref into a retained native/relation
candidate, role_signature, output_unit, context_ref, parameter_paths and scoped
support. Repeated declarations reuse their index version. source_ref is
{kind:'native_candidate',question_id:...,round_index:0,candidate_index:0} or
{kind:'relation_candidate',question_id:...,round_index:0,candidate_index:0}.
An opaque Python candidate can instead declare forecast_ref {file:...,
json_path:['experiment_comparisons',0,'forecasts',0],model:<saved model path>}.
Only those saved public-row predictions are runnable, through the existing
query interpreter. This finite table is not a scientific Program, has no
structural match or inherited confirmation, and is not a PR3 induction source.
For the finite native author's question, its complete predictor_paths catalog
is also forecast automatically on the declared public task and panel inputs.
Each Python fit uses its retained original training view in an independent
writable context; public rows supply no training labels. Columns share frozen
context and input references. Worker time and candidate-row/research-witness
compute consume the current mission's remaining budgets. Failed or unfunded
columns remain UNKNOWN. Repeating unchanged inputs reuses the saved columns.
Explicit candidate_theory_ids and FindDistinguishingExperiment claim rosters
keep their exact membership and do not gain these automatic entries.
experiments contain id, original public actions, an executable query, tolerance
and observation_cost. Selection preserves ties and independent theories; it is
a finite-pool heuristic. A unique selection prepares the parent's pending_query.
An explicit exploration_policy {kind:'seeded_legal',seed:<integer>} also permits
an operational choice among tied legal plans or partially predicted candidates;
ties, missing predictions and finite-pool coverage remain in the result. This
policy is part of the cached execution contract, not evidence of superiority.
library_path fixes the frozen library used by automatic recall; working
declarations remain available and the selected library is never overwritten.
query {pending_question:'parent'} executes that plan through the normal frozen
forecast/purchase path. The research route itself never buys observations.
Without an explicit question, the route generates obligations from actual
dependencies and conflicts and selects the task/theory agenda under mission.
Theory work needs an explicit theory_learning_allocation. Unknowns and author
work remain visible; an unavailable provider never prevents local computation.
candidate_theory_ids explicitly declares this question's finite comparison
pool. Other indexed Programs remain available as components and retained
evidence. An unavailable requested theory leaves the comparison UNKNOWN.
An optional research_comparison={maximum:<integer>,assessment_basis_ref:<JSON ref>}
reuses complete paid fit MSEs on the exact evidence_checks and risk_quantile=.5
of a retained relation-transfer execution.request_basis. It retains a bounded
Program research pool within the ordinary comparison; every explicitly named
candidate and every explicit FindDistinguishingExperiment claim stays mandatory.
Unassessed and opaque entries retain their original place. Its paid assessments
do not supply reference or future evidence, scientific support, or task adoption.
Public task/panel predictions still consume the ordinary selection budget.
An internal completion reserve may pass allocation.nominal_compute_budget to
record the original task cap in the selection's freshness basis. It does not
increase the executable allocation or the current config/mission budget.
With a typed theory_target, frozen library objects are recalled within the
task's remaining compute or an explicit allocation through the existing
optional transfer operator. Theory/method missions keep their funded limits. Explicit finite
candidate_theory_ids still determine which Programs receive comparison votes.
FindDistinguishingExperiment queries declare target_query (the actual quantity
answered by each model), claim_refs [{id,version}] and legal_constructor_ref
{kind:'resolution_experiments'}. They search the question's retained public
panel through the same selector. A unique plan returns executable actions and
query_refs; ties and missing witnesses remain unresolved. No plan is purchased.
"""


def register_theories(agent, declarations):
    for declaration in declarations:
        identifier = declaration['id']
        previous = agent.theory_index.get(identifier)
        body = {key: deepcopy(value) for key, value in declaration.items()
                if key not in {'version', 'history'}}
        if body.get('source_ref'):
            reference = body['source_ref']
            owner = next((q for q in agent.questions if q['id'] == reference['question_id']), None)
            old_ref = (previous or {}).get('source_ref', {})
            same_source = all(old_ref.get(key) == reference.get(key)
                              for key in ('kind', 'question_id', 'round_index', 'candidate_index'))
            if 'meaning_version' not in reference:
                reference['meaning_version'] = (old_ref.get('meaning_version', 1) if same_source
                                                else (owner or {}).get('meaning_version', 1))
            same_source = same_source and reference['meaning_version'] == old_ref.get('meaning_version', 1)
            if reference['kind'] == 'relation_candidate':
                if same_source and 'event_index' in old_ref:
                    reference['event_index'] = old_ref['event_index']
                    if 'after_observation_index' in old_ref:
                        reference['after_observation_index'] = old_ref['after_observation_index']
                    else:
                        reference.pop('after_observation_index', None)
                elif 'event_index' not in reference:
                    for event_index, event in enumerate(getattr(agent, 'events', [])):
                        feedback = event.get('feedback', {})
                        outcomes = [(None, feedback), *enumerate(feedback.get('after_observation_results', []))]
                        found = False
                        for outcome_index, outcome in outcomes:
                            bank = outcome.get('relation_research', {})
                            if (outcome.get('research_question_id') == reference['question_id']
                                    and outcome.get('question_meaning_version', 1) == reference['meaning_version']
                                    and bank.get('round_index') == reference['round_index']
                                    and 0 <= reference['candidate_index'] < len(bank.get('candidates', []))):
                                reference['event_index'] = event_index
                                if outcome_index is not None:
                                    reference['after_observation_index'] = outcome_index
                                found = True
                                break
                        if found:
                            break
        if previous and {key: value for key, value in previous.items()
                         if key not in {'version', 'history'}} == body:
            continue
        history = deepcopy(previous.get('history', [])) if previous else []
        if previous:
            history.append({key: deepcopy(value) for key, value in previous.items() if key != 'history'})
        agent.theory_index[identifier] = {**body, 'version': (previous or {}).get('version', 0) + 1}
        if history:
            agent.theory_index[identifier]['history'] = history


def _resolved_theory_versions(agent, pending_theory=None):
    """Read retained source versions without copying discarded theory payloads.

    These temporary views never mutate indexed/bank values. Public results and
    retained failure references receive independent copies below.
    """
    versions, indexed_versions = [], []
    for identifier, indexed in agent.theory_index.items():
        for position, old in enumerate(indexed.get('history', [])):
            indexed_versions.append((identifier, old, ['theory_index', identifier, 'history', position], False))
        indexed_versions.append((identifier, indexed, ['theory_index', identifier], True))
    if pending_theory is not None:
        indexed_versions.append((pending_theory['id'], pending_theory, [], 'pending'))
    for identifier, indexed, json_path, current in indexed_versions:
        theory = {key: value for key, value in indexed.items() if key != 'history'}
        # This derived result follows actual retained evidence, rather than a
        # declaration's possibly stale copy from another numerical instance.
        theory.pop('scope_refutation', None)
        reference = theory.get('source_ref')
        if reference:
            owner_index = next((index for index, q in enumerate(agent.questions)
                                if q['id'] == reference['question_id']), None)
            owner = agent.questions[owner_index] if owner_index is not None else None
            owner_path = ['questions', owner_index]
            meaning = reference.get('meaning_version', 1)
            if owner and owner.get('meaning_version', 1) != meaning:
                old_index = next((index for index, snapshot in enumerate(owner.get('meaning_history', []))
                                  if snapshot['meaning_version'] == meaning), None)
                owner = owner['meaning_history'][old_index] if old_index is not None else None
                owner_path.extend(['meaning_history', old_index])
            source = None
            if reference['kind'] == 'native_candidate' and owner:
                rounds = owner.get('native_condition_research', [])
                if 0 <= reference['round_index'] < len(rounds):
                    pool = rounds[reference['round_index']].get('candidates', [])
                    if 0 <= reference['candidate_index'] < len(pool):
                        source = pool[reference['candidate_index']]
                        program_ref = theory.get('program_source_ref', {})
                        tail = ['native_condition_research', reference['round_index'],
                                'candidates', reference['candidate_index'], 'program']
                        if (program_ref.get('file') == 'research_state.json'
                                and program_ref.get('json_path', [])[:2] == ['questions', owner_index]
                                and program_ref['json_path'][-len(tail):] == tail):
                            theory['program_source_ref'] = {'file': 'research_state.json',
                                                          'json_path': [*owner_path, *tail]}
            elif reference['kind'] == 'relation_candidate':
                bank = {}
                event_index = reference.get('event_index')
                events = getattr(agent, 'events', [])
                if event_index is not None and 0 <= event_index < len(events):
                    feedback = events[event_index].get('feedback', {})
                    outcome_index = reference.get('after_observation_index')
                    if outcome_index is not None:
                        outcomes = feedback.get('after_observation_results', [])
                        feedback = outcomes[outcome_index] if 0 <= outcome_index < len(outcomes) else {}
                    if (feedback.get('research_question_id') == reference['question_id']
                            and feedback.get('question_meaning_version', 1) == meaning):
                        bank = feedback.get('relation_research', {})
                if bank.get('round_index') != reference['round_index']:
                    # Offline replay can retain the exact current bank without
                    # loading its larger event history. Never substitute a new round.
                    current_owner = next((q for q in agent.questions if q['id'] == reference['question_id']), None)
                    bank_id = reference['question_id']
                    if current_owner and current_owner.get('meaning_version', 1) != meaning:
                        bank_id = f'{bank_id}@meaning{meaning}'
                    if current_owner or meaning == 1:
                        bank = agent.relation_research.get(bank_id, {})
                    else:
                        bank = {}
                if bank.get('round_index') == reference['round_index']:
                    pool = bank.get('candidates', [])
                    if 0 <= reference['candidate_index'] < len(pool):
                        source = pool[reference['candidate_index']]
            if source is None:
                theory.update(source_status='unavailable_retained_version', compatible=False)
            else:
                theory['program'] = source['program']
                theory['source_status'] = source.get('status', 'retained_candidate')
                if source.get('status') == 'refuted':
                    theory['compatible'] = False
        theory['id'] = identifier
        versions.append((theory, {'file': 'research_state.json', 'json_path': json_path,
                                  'version': theory.get('version', 1)}, current))
    return versions


def _retained_scope_refutations(agent, versions, *, target=None):
    transfers = getattr(agent, 'theory_consolidation', {}).get('transfers', [])
    by_version, legacy_by_id = {}, {}
    for index, transfer in enumerate(transfers):
        retained = transfer.get('theory_ref')
        if retained:
            by_version.setdefault((retained.get('id'), retained.get('version')), []).append((index, transfer))
        else:
            legacy_by_id.setdefault(transfer.get('theory_id'), []).append((index, transfer))
    refutations, library_documents, local_fields = {}, {}, {}
    for theory, theory_ref in versions:
        if target is not None and any(theory.get(key) != target.get(key) for key in
                ('role_signature', 'output_unit', 'context_ref')):
            continue
        if not theory.get('program') or not (theory.get('compatible') is False
                or theory.get('support', {}).get('status') in {'refuted', 'contradicted_in_scope'}):
            continue
        references = []
        if theory.get('source_status') == 'refuted' and theory.get('source_ref'):
            references.append(deepcopy(theory['source_ref']))
        related = [*by_version.get((theory['id'], theory.get('version', 1)), ()),
                   *((index, transfer) for index, transfer in legacy_by_id.get(theory['id'], ())
                     if theory.get('consolidation_ref') == transfer.get('binding'))]
        # Exact-version and legacy records retain their original combined order.
        for index, transfer in sorted(related, key=lambda item: item[0]):
            if transfer.get('binding', {}).get('context_ref') != theory.get('context_ref'):
                continue
            references.extend(deepcopy(transfer.get('retained_counterexample_refs', [])))
            transfer_ref = None
            if transfer.get('paid_mse_check', {}).get('status') == 'failed':
                from .theory_consolidation import library_entry_reference
                transfer_ref = library_entry_reference(agent, 'transfers', index,
                    documents=library_documents, local_fields=local_fields)
                references.append(_child_ref(transfer_ref, 'paid_mse_check'))
            for check_index, check in enumerate(transfer.get('checks', [])):
                if check.get('status') != 'failed' or check.get('context_ref') != theory.get('context_ref'):
                    continue
                from .theory_consolidation import library_entry_reference
                if transfer_ref is None:
                    transfer_ref = library_entry_reference(agent, 'transfers', index,
                        documents=library_documents, local_fields=local_fields)
                references.append(_child_ref(transfer_ref, 'checks', check_index))
                if check.get('source_ref') and check['source_ref'] not in references:
                    references.append(deepcopy(check['source_ref']))
        if (references and theory.get('context_ref') is not None
                and theory.get('output_unit') is not None and 'role_signature' in theory):
            program = Program.from_dict(theory['program']).expanded()
            # Program is already an immutable structural dictionary key used by
            # fitting. Its full equality still decides collisions/membership;
            # this avoids comparing every current Program with every failure.
            refutations.setdefault(program, []).append((theory, theory_ref, references))

    return refutations


def _apply_scope_refutations(theory, refutations):
    program = Program.from_dict(theory['program']).expanded() if theory.get('program') else None
    for failed, theory_ref, references in refutations.get(program, ()):
        if not all(theory.get(key) == failed.get(key) for key in
                ('role_signature', 'output_unit', 'context_ref')):
            continue
        theory['compatible'] = False
        scope = theory.setdefault('scope_refutation', {'status': 'refuted',
            'context_ref': deepcopy(theory.get('context_ref')),
            'theory_refs': [], 'counterexample_refs': []})
        if theory_ref not in scope['theory_refs']:
            scope['theory_refs'].append(deepcopy(theory_ref))
        for reference in references:
            if reference not in scope['counterexample_refs']:
                scope['counterexample_refs'].append(deepcopy(reference))


def resolved_theories(agent, *, pending_theory=None):
    """Resolve complete retained theories and scoped failures in original order."""
    versions = _resolved_theory_versions(agent, pending_theory)
    refutations = _retained_scope_refutations(agent, [(theory, reference)
        for theory, reference, current in versions if current != 'pending'])
    result = []
    for view, _, current in versions:
        if not current:
            continue
        theory = {key: deepcopy(value) for key, value in view.items()}
        _apply_scope_refutations(theory, refutations)
        result.append(theory)
    if pending_theory is None and any(theory.get('forecast_ref') for theory in result):
        from .retained_forecasts import resolve_forecast_references
        result = resolve_forecast_references(result, agent.workspace)
    return result


def scoped_numeric_refutation(agent, declaration):
    """Check only this incoming Program against the complete retained failures."""
    versions = _resolved_theory_versions(agent, declaration)
    incoming, _, _ = versions[-1]
    if not incoming.get('program'):
        return None
    refutations = _retained_scope_refutations(agent, [(theory, reference)
        for theory, reference, current in versions if current != 'pending'], target=incoming)
    _apply_scope_refutations(incoming, refutations)
    return incoming.get('scope_refutation')


def _install_proposals(agent, proposals):
    for proposal in proposals:
        old = next((q for q in agent.questions if q['id'] == proposal['id']), None)
        if old is None or isinstance(old.get('origin'), dict):
            if old:
                proposal = merge_question_proposal(old, proposal)
                if 'mission' in proposal:
                    proposal['mission'] = adopt_research_mission(old.get('mission', {}), proposal['mission'])
            agent._open_question(proposal)
        else:
            # An inferred scheduling obligation cannot redefine a user's
            # existing scientific question and thereby erase its checked answer.
            old['agenda_basis'] = deepcopy(proposal['agenda_basis'])
            old['agenda_origin'] = deepcopy(proposal['origin'])
            old['queue'] = proposal['queue']
            for use in proposal.get('use_sites', []):
                uses = old.setdefault('use_sites', [])
                if not any(all(item.get(key) == use.get(key) for key in ('kind', 'model', 'slot', 'parent_id'))
                           for item in uses):
                    uses.append(deepcopy(use))


def _refresh_completion(agent, mission, allowance, *, compute_limit=None):
    for question in agent.questions:
        owner = effective_question_mission(question, mission)
        available = min(allowance, mission_compute_remaining(agent, owner))
        execution = question.get('routing_execution')
        if not execution:
            continue
        unchanged = execution['basis'] == _basis(agent, question, owner, compute_limit=compute_limit)
        incomplete = execution['result']['status'] in {'scoped_unknown', 'UNKNOWN_MATCH'}
        if unchanged and (not incomplete or available <= execution['allowance']):
            question['completed_agenda_basis'] = deepcopy(question.get('agenda_basis') or {
                'meaning_version': question.get('meaning_version', 1),
                'answer_version': (question.get('semantic_answer') or question.get('executable_answer') or {}).get('version')})
        else:
            question.pop('completed_agenda_basis', None)


def _basis(agent, question, mission, *, compute_limit=None):
    from .theory_consolidation import LIBRARY_PATH, theory_library_candidate_versions

    mission = effective_question_mission(question, mission)
    library_path = question.get('frozen_library_path', LIBRARY_PATH)
    use_paths = [use.get('path', use.get('model')) for use in question.get('use_sites', [])]
    basis = {'meaning_version': question.get('meaning_version', 1),
        'query': deepcopy(question.get('query_semantics')), 'context': deepcopy(question.get('context_ref')),
        'answer_type': question.get('answer_type'), 'resolution': deepcopy(question.get('answer_resolution_ref')),
        'candidate_theory_ids': deepcopy(question.get('candidate_theory_ids')),
        'frozen_library_path': library_path,
        'frozen_library': theory_library_candidate_versions(agent, library_path),
        'uses': deepcopy(question.get('use_sites', [])),
        'program_sources': {path: (agent.workspace / path).read_text()
                            for path in use_paths if path and (agent.workspace / path).is_file()},
        'pool': [{key: deepcopy(theory[key]) for key in ('id', 'version', 'source_ref', 'source_status',
            'forecast_ref', 'forecast_source_version', 'compatible') if key in theory}
            for theory in resolved_theories(agent)],
        'parameter_answer_version': question.get('executable_answer', {}).get('version'),
        'target': deepcopy(question.get('theory_target')),
        'experiments': deepcopy(question.get('resolution_experiments', [])),
        'exploration_policy': deepcopy(question.get('exploration_policy')),
        'required_premises': deepcopy(question.get('required_premises', [])),
        'premise_statuses': deepcopy(question.get('premise_statuses', {})),
        'support_available': _versioned_support(agent, question)[0],
        'allowed_mappings': deepcopy(question.get('allowed_mappings', [])),
        'dependencies': [{'id': q['id'], 'meaning_version': q.get('meaning_version', 1),
                         'answer_version': (q.get('semantic_answer') or q.get('executable_answer') or {}).get('version'),
                         'status': q.get('status'), 'evidence_status': q.get('evidence_status')}
                         for q in agent.questions if q['id'] in question.get('depends_on', [])
                         or any(q['id'] == ref['question_id'] for path in question.get('support_paths', []) for ref in path)],
        'paid_cutoff': [len(agent.observations), len(agent.rows)], 'mission': mission_contract(mission),
        'compute_limit': agent.config.compute_budget if compute_limit is None else compute_limit}
    if question.get('research_comparison') is not None:
        basis['research_comparison'] = research_comparison_basis(agent, question)
    return basis


def pending_selection_current(agent, parent):
    source = parent['pending_query_source']
    question = next(q for q in agent.questions if q['id'] == source['question_id'])
    if source.get('stage') == 2:
        continuation = question.get('conditional_continuation', {})
        return (continuation.get('basis') == _basis(agent, question, agent.research_mission)
                and continuation.get('routing_version') == source['routing_version'])
    execution = question.get('routing_execution', {})
    return (source['meaning_version'] == question.get('meaning_version', 1)
            and source['routing_version'] == execution.get('version')
            and execution.get('basis') == _basis(agent, question, agent.research_mission))


def _prepare_selection(agent, question, selection, version):
    use = next((use for use in question.get('use_sites', [])
                if use.get('kind') == 'choose_parent_experiment'), None)
    if use is None:
        return {'status': 'selection_ready_no_parent_use', 'parent_revision_completed': False}
    parent_id = use.get('parent_id', question.get('parent_id'))
    parent = next((q for q in agent.questions if q['id'] == parent_id), None)
    arguments = selection.get('query_arguments')
    if parent is None or not arguments:
        return {'status': 'selection_ready_no_parent_use', 'parent_revision_completed': False}
    pending = deepcopy(arguments)
    experiment = selection['selected_experiment']
    if 'revision' in experiment:
        pending['revision'] = deepcopy(experiment['revision'])
    pending.setdefault('question', {'id': parent['id']})
    old = parent.get('pending_query')
    if old != pending:
        if old is not None:
            parent.setdefault('pending_query_history', []).append(deepcopy(old))
        parent['pending_query'] = pending
    parent['pending_query_source'] = {'kind': 'question_selection', 'question_id': question['id'],
        'meaning_version': question.get('meaning_version', 1), 'routing_version': version,
        'experiment_id': experiment['id']}
    return {'status': 'parent_query_prepared', 'parent_id': parent_id,
            'experiment_id': experiment['id'], 'changed': old != pending,
            'parent_revision_completed': old != pending}


def _distinguishing_projection(question, pool):
    """Project an experiment search onto the quantity the models must answer."""
    query = question['query_semantics']['query_expression']

    def author_work(reason):
        return {'status': 'author_work_request', 'reason': reason,
                'request': {'type': 'AuthorWorkRequest', 'object_type': 'distinguishing_experiment',
                    'required_contract': {'target_query': 'executable model-side quantity',
                        'claim_refs': '[{id,version}] in the current finite pool',
                        'legal_constructor_ref': {'kind': 'resolution_experiments'},
                        'resolution_experiments': 'retained public actions and executable readouts'}}}

    if not isinstance(query.get('target_query'), dict) or not query['target_query'].get('op'):
        return author_work('missing_distinguishing_target_query')
    references = query.get('claim_refs')
    if not isinstance(references, list) or not references or any(
            not isinstance(ref, dict) or not ref.get('id') or type(ref.get('version')) is not int
            for ref in references):
        return author_work('explicit_current_claim_versions_required')
    identifiers = [ref['id'] for ref in references]
    if len(set(identifiers)) != len(identifiers):
        return author_work('distinct_current_claim_references_required')
    indexed = {theory['id']: theory for theory in pool}
    unavailable = [ref for ref in references if ref['id'] not in indexed
                   or indexed[ref['id']].get('version', 1) != ref['version']
                   or not (indexed[ref['id']].get('program')
                           or indexed[ref['id']].get('_retained_forecast'))]
    if unavailable:
        return {'status': 'scoped_unknown', 'reason': 'requested_claim_version_unavailable',
                'unavailable_claim_refs': deepcopy(unavailable)}
    declared = question.get('candidate_theory_ids')
    if declared is not None and set(declared) != set(identifiers):
        return {'status': 'scoped_unknown', 'reason': 'claim_pool_mismatch'}
    if query.get('legal_constructor_ref') != {'kind': 'resolution_experiments'}:
        return author_work('public_experiment_constructor_required')
    experiments = question.get('resolution_experiments', [])
    if not experiments:
        return author_work('retained_public_experiment_panel_required')
    if any(plan.get('direct_task_action') for plan in experiments):
        return author_work('distinguishing_panel_requires_experimental_readouts')
    projected = deepcopy(question)
    projected['candidate_theory_ids'] = identifiers
    projected['query_semantics']['query_expression'] = deepcopy(query['target_query'])
    if 'target_answer_resolution_ref' in query:
        projected['answer_resolution_ref'] = deepcopy(query['target_answer_resolution_ref'])
    return {'status': 'ready', 'question': projected}


def _distinguishing_answer(question, result):
    selection = result.get('selection') or {}
    selected = selection.get('selected_experiment')
    answer = {'status': result['status'], 'scope': 'finite_candidate_heuristic',
        'support_updated': False, 'independent_confirmation': False,
        'candidate_completeness': 'not_established',
        'reason': result.get('reason', selection.get('unknown_reason')),
        'tied_experiments': deepcopy(selection.get('tied_experiments', []))}
    if selection.get('operational_decision') is not None:
        answer['operational_decision'] = deepcopy(selection['operational_decision'])
    if selected is not None and selection['status'] == 'selected':
        answer.update(status='answered', value={'kind': 'finite_experiment_plan',
            'experiment_id': selected['id'], 'actions': deepcopy(selected['actions']),
            'query_refs': [{'question_id': question['id'],
                'meaning_version': question.get('meaning_version', 1),
                'field': 'resolution_experiments', 'experiment_id': selected['id'],
                'member': 'query'}]})
    return answer


def find_distinguishing_experiment(agent, question, laboratory, *, mission=None):
    """Use the normal finite route once; prepare a plan without acquisition.

    The result contains ``query_answer`` for the query dispatcher. This function
    already accounts for compute and prepares a unique parent pending query;
    callers must not reapply parent uses or record predictions as measurements.
    """
    arguments = {} if mission is None else {'mission': mission}
    return route_research(agent, arguments, laboratory, question)


def _retained_revision_inputs(agent, question):
    """Resolve an old search's exact versions and ordered execution rosters."""
    basis = question['frozen_recall_execution']['basis']
    versions = {(item['id'], item.get('version', 1)): item
                for item, _, _ in _resolved_theory_versions(agent)}
    prefix = f'relation_transfer:frozen_recall:{question["id"]}@{question.get("meaning_version", 1)}:'
    retained = [(key[len(prefix):], execution['request_basis'])
                for key, execution in agent.theory_consolidation.get('executions', {}).items()
                if key.startswith(prefix)]
    constructor_fields = (('pattern_ref', '_pattern'), ('holes_ref', '_holes'),
                          ('components_ref', '_bound_holes'))
    references = []
    for original in (basis, *(item for _, item in retained)):
        source = (original.get('parent') or {}).get('contract_source')
        if source:
            for field, _ in constructor_fields:
                reference = source.get(field)
                if reference is not None and reference not in references:
                    references.append(reference)
    # Several old objects can share one large constructor store. Resolve all
    # addressed fields in one batch, rather than reopening it per execution.
    values = load_source_values(references, agent.workspace) if references else []

    def resolve(reference, *, structural=True):
        if not reference:
            return None
        found = versions.get((reference.get('id'), reference.get('version', 1)))
        if found is None or (structural and not found.get('program')):
            raise ValueError('retained_research_Program_version_unavailable')
        if found.get('source_ref') != reference.get('source_ref'):
            raise ValueError('retained_research_component_premise_changed')
        result = deepcopy(found)
        source = reference.get('contract_source')
        if source:
            result['_contract_source'] = deepcopy(source)
            if all(field in source for field, _ in constructor_fields):
                result['_contract_source'] = {'source_ref': deepcopy(source), **{
                    target: values[references.index(source[field])]['value']
                    for field, target in constructor_fields}}
            if source.get('program_ref'):
                result['program_source_ref'] = deepcopy(source['program_ref'])
        return result

    # Opaque entries retain their original outer identity; transfer rosters
    # below still require actual structural Programs.
    components = [resolve(item, structural=False) for item in basis.get('components', [])]
    parent = resolve(basis.get('parent'))
    executions = {}
    for identifier, original in retained:
        executions[identifier] = {
            '_retained_components': [resolve(item) for item in original.get('components', [])],
            '_retained_parent': resolve(original.get('parent')),
            'component_theory_ids': deepcopy(original.get('component_theory_ids'))}
    return {'_retained_components': components, '_retained_parent': parent,
            '_retained_execution_inputs': executions}


def continue_conditional_plan(agent, parent, executed, purchased):
    """Use an actual first outcome to prepare, but never buy, the second step."""
    source = executed['source']
    if source.get('kind') != 'question_selection' or source.get('stage') == 2:
        return None
    question = next(q for q in agent.questions if q['id'] == source['question_id'])
    execution = question['routing_execution']
    plan = execution['result']['selection']['selected_experiment']
    if not plan.get('branches'):
        return None
    mission = effective_question_mission(question, agent.research_mission)
    before_compute = agent.costs['compute_cost']
    outcome = evaluate_on_evidence(plan['query'], agent.observations, context=question.get('context_ref'),
        compute_budget=mission_compute_remaining(agent, mission),
        record_bindings=executed_plan_records(agent, question))
    agent.costs['compute_cost'] += outcome['compute_cost']
    debit_mission_compute(agent, mission, before_compute)
    fresh = {index for record in purchased.get('observations', []) for index in record.get('row_indices', [])}
    if outcome['status'] != 'answered' or not fresh.intersection(outcome['evidence']):
        return {'status': 'scoped_unknown', 'reason': 'first_step_outcome_not_bound', 'readout': outcome}
    branches = [branch for branch in plan['branches']
                if (distance := numeric_answer_distance(outcome['value'], branch['when']['value']))
                is not None and distance <= branch['when'].get('atol', 0.)]
    if len(branches) != 1:
        return {'status': 'scoped_unknown', 'reason': 'first_step_has_no_unique_declared_branch', 'readout': outcome}
    branch = branches[0]
    pending = {'actions': deepcopy(branch['actions']), 'question': {'id': parent['id']}}
    if branch.get('revision'):
        pending['revision'] = deepcopy(branch['revision'])
    parent['pending_query'] = pending
    parent['pending_query_source'] = {**source, 'stage': 2}
    question['conditional_continuation'] = {'basis': _basis(agent, question, agent.research_mission),
        'routing_version': execution['version'], 'first_outcome': outcome,
        'selected_branch_index': plan['branches'].index(branch)}
    return {'status': 'second_step_prepared', 'first_outcome': outcome,
            'parent_id': parent['id'], 'purchased_second_step': False}


def route_research(agent, arguments, laboratory, question=None):
    # A completion reserve narrows execution, not the task's nominal contract.
    # Pending execution checks this basis again after the caller restores its cap.
    compute_limit = arguments.get('allocation', {}).get('nominal_compute_budget')
    register_theories(agent, arguments.get('theories', []))
    if 'mission' in arguments:
        agent.research_mission = adopt_research_mission(agent.research_mission, arguments['mission'])
    mission = agent.research_mission
    pool = resolved_theories(agent)
    if question is not None:
        for source, target in [('experiments', 'resolution_experiments'), ('target', 'theory_target'),
                               ('allowed_mappings', 'allowed_mappings'),
                               ('candidate_theory_ids', 'candidate_theory_ids'),
                               ('research_comparison', 'research_comparison'),
                               ('library_path', 'frozen_library_path')]:
            if source in arguments:
                question[target] = deepcopy(arguments[source])
    proposals = generate_questions(agent.questions, agent.research_candidates, pool, mission=mission)
    _install_proposals(agent, proposals)
    allowance = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
    _refresh_completion(agent, mission, allowance, compute_limit=compute_limit)
    agenda = build_agenda(agent.questions, agent.research_candidates, pool, mission=mission,
                          budget={'compute': allowance, 'author_calls': 0})
    agent.research_agenda = deepcopy(agenda)
    if agenda.get('status') == 'task_supported':
        return {**agenda, 'theory_status': 'unresolved', 'new_compute_cost': 0}
    if question is None:
        selected = agenda.get('selected_question_id')
        question = next((q for q in agent.questions if q['id'] == selected), None)
    if question is None:
        # Returning an author request does not spend an author call. The
        # existing provider/procedure decides how to construct the missing work.
        blocked = next((entry for queue in ('task_agenda', 'theory_agenda')
            for entry in agenda[queue] if entry.get('reason') == 'author_work_unfunded'), None)
        if blocked:
            waiting = next(q for q in agent.questions if q['id'] == blocked['question_id'])
            pending = waiting.get('pending_author_work')
            if pending is not None and pending.get('agenda_basis') == question_work_basis(waiting):
                return {**deepcopy(pending), 'agenda': agenda, 'cache_hit': True, 'new_compute_cost': 0}
            before_request = agent.costs['compute_cost']
            request = step_question(agent, waiting, laboratory,
                mission=effective_question_mission(waiting, mission))
            return {**request, 'agenda': agenda,
                    'new_compute_cost': agent.costs['compute_cost'] - before_request}
        return {'status': 'research_deferred', 'agenda': agenda, 'new_compute_cost': 0}
    track_agenda = None
    owner = question
    selected, track_agenda = select_parent_revision_work(agent, question, allowance)
    if track_agenda is not None:
        agent.research_agenda = deepcopy(track_agenda)
        if selected is None:
            previous_work = owner.get('routing_execution')
            if previous_work and all(entry['status'] == 'completed' for entry in track_agenda['task_agenda']):
                result = deepcopy(previous_work['result'])
                result.update(cache_hit=True, new_compute_cost=0,
                    parent_research_agenda=track_agenda,
                    research_track_question_ref={'question_id': owner['id'],
                                                 'meaning_version': owner.get('meaning_version', 1)})
                if 'frozen_recall' in result:
                    result['frozen_recall'].update(cache_hit=True, new_compute_cost=0,
                                                   reserved_comparison_compute=0)
                return result
            return {'status': 'research_deferred', 'question_id': owner['id'],
                    'agenda': track_agenda, 'new_compute_cost': 0}
        question = selected
        if question is not owner:
            try:
                retained_inputs = _retained_revision_inputs(agent, question)
            except ValueError as error:
                return {'status': 'research_deferred', 'reason': str(error),
                    'question_id': question['id'], 'agenda': track_agenda,
                    'research_track_question_ref': track_agenda['selected_question_ref'],
                    'new_compute_cost': 0}
            arguments = {key: value for key, value in arguments.items() if key not in
                         {'parent_theory_id', 'fitting', 'target', 'evidence_checks', 'library_path',
                          'mission', 'exploration_policy'}}
            arguments.update(retained_inputs, library_path=question['frozen_library_path'])
        # Keep the chosen original meaning visible to exporters and receipts.
        # Neither this reference nor a proposal adopts a task predictor.
        selected_ref = {'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)}
        owner['parent_research_last_selection'] = deepcopy(selected_ref)
    if 'library_path' in arguments:
        question['frozen_library_path'] = arguments['library_path']
    if 'exploration_policy' in arguments:
        question['exploration_policy'] = deepcopy(arguments['exploration_policy'])
    mission = effective_question_mission(question, mission)
    distinguishing = question.get('query_semantics', {}).get('query_expression', {}).get(
        'op') == 'FindDistinguishingExperiment'
    comparison_question = question
    if distinguishing:
        projection = _distinguishing_projection(question, pool)
        if projection['status'] == 'ready' and not _versioned_support(agent, question)[0]:
            projection = {'status': 'scoped_unknown', 'reason': 'premise_unknown'}
        if projection['status'] != 'ready':
            result = {**projection, 'question_id': question['id'], 'selection': None,
                      'new_compute_cost': 0, 'cache_hit': False}
            result['query_answer'] = _distinguishing_answer(question, result)
            return result
        comparison_question = projection['question']
    prior_answer_execution = None
    if not distinguishing and question.get('semantic_answer', {}).get('status') == 'answered':
        before_answer = agent.costs['compute_cost']
        prior_answer_execution = step_question(agent, question, laboratory, mission=mission)
        if prior_answer_execution.get('answer', {}).get('status') == 'answered':
            obligation = parent_revision_obligation(agent, question, prior_answer_execution)
            if obligation is not None:
                routed = route_research(agent, arguments, laboratory, obligation)
                return {**routed, 'answered_child_ref': {'file': 'research_state.json',
                            'json_path': ['questions', agent.questions.index(question), 'semantic_answer']},
                    'parent_revision_question_ref': deepcopy(question['parent_revision_question_ref']),
                    'research_track_question_ref': deepcopy(obligation.get(
                        'parent_research_last_selection', question['parent_revision_question_ref'])),
                    'new_compute_cost': agent.costs['compute_cost'] - before_answer}
            return {'status': prior_answer_execution['status'], 'question_id': question['id'],
                'answer_execution': prior_answer_execution, 'selection': None,
                'cache_hit': prior_answer_execution['status'] == 'reused_answer',
                'frozen_recall': {'status': 'not_needed_answer_available',
                                  'generated_theory_ids': [], 'new_compute_cost': 0},
                'new_compute_cost': agent.costs['compute_cost'] - before_answer}
    origin = question.get('origin')
    if (not distinguishing and isinstance(origin, dict) and origin.get('kind') == 'multi_episode_commonality'
            and not has_executable_query(question)):
        from .theory_consolidation import consolidate_research
        source_ids = [ref['id'] for ref in origin['source_refs'] if ref.get('kind') == 'theory']
        return consolidate_research(agent, {**arguments, 'source_theory_ids': source_ids,
            'theme': question['id'], 'mission': mission}, laboratory, question)
    from .theory_consolidation import recall_frozen_theories
    start = agent.costs['compute_cost']
    recalled = ({'status': 'not_needed_explicit_claim_pool', 'generated_theory_ids': [],
                 'new_compute_cost': 0} if distinguishing else
                recall_frozen_theories(agent, question, {**arguments, 'mission': mission}))
    python_forecasts = {'status': 'not_needed', 'new_compute_cost': 0,
                        'new_python_execution_seconds': 0., 'support_updated': False}
    if not distinguishing and comparison_question.get('candidate_theory_ids') is None:
        from .python_research_routing import ordinary_forecast_models
        from .retained_forecasts import prepare_python_forecasts
        models = ordinary_forecast_models(agent, question)
        if models:
            before_forecasts = agent.costs['compute_cost']
            python_forecasts = prepare_python_forecasts(agent, question, laboratory, models,
                compute_budget=mission_compute_remaining(agent, mission))
            register_theories(agent, python_forecasts.pop('theories'))
            debit_mission_compute(agent, mission, before_forecasts)
    pool = resolved_theories(agent)
    allowance = mission_compute_remaining(agent, mission)
    if comparison_question.get('candidate_theory_ids') is not None:
        requested = comparison_question['candidate_theory_ids']
        missing = [identifier for identifier in requested if not any(
            theory['id'] == identifier and (theory.get('program') or theory.get('_retained_forecast'))
            for theory in pool)]
        if missing:
            return {'status': 'scoped_unknown', 'reason': 'requested_candidate_unavailable',
                    'missing_theory_ids': missing, 'frozen_recall': recalled,
                    'new_compute_cost': agent.costs['compute_cost'] - start}
        pool = [theory for theory in pool if theory['id'] in requested]
    else:
        # Automatically saved columns belong to this question and meaning.
        # Other explicit declarations retain their original matching rules.
        pool = [theory for theory in pool if not theory.get('forecast_scope') or (
            theory['forecast_scope']['question_id'] == question['id']
            and theory['forecast_scope']['meaning_version'] == question.get('meaning_version', 1))]
    basis = _basis(agent, question, mission, compute_limit=compute_limit)
    previous = question.get('routing_execution')
    if previous and previous['basis'] == basis:
        incomplete = previous['result'].get('status') in {'scoped_unknown', 'UNKNOWN_MATCH'}
        if not incomplete or allowance <= previous['allowance']:
            return {**deepcopy(previous['result']), 'cache_hit': True, 'frozen_recall': recalled,
                    'python_forecasts': python_forecasts,
                    **({'research_track_question_ref': selected_ref,
                        'parent_research_agenda': track_agenda} if track_agenda is not None else {}),
                    'new_compute_cost': agent.costs['compute_cost'] - start}
    before_compute = agent.costs['compute_cost']
    matching = match_theories(comparison_question, pool, target=question.get('theory_target'),
        compute_budget=allowance, allowed_mappings=question.get('allowed_mappings'))
    agent.costs['compute_cost'] += matching.get('compute_cost', 0)
    debit_mission_compute(agent, mission, before_compute)
    _install_proposals(agent, generate_questions(agent.questions, agent.research_candidates, pool,
        mission=mission, matching={question['id']: matching}))
    agent.research_agenda = build_agenda(agent.questions, agent.research_candidates, pool,
        mission=mission, budget={'compute': mission_compute_remaining(agent, mission),
                                 'author_calls': 0}, matching={question['id']: matching})
    # The existing PR1 path consumes checked child returns before selecting new
    # measurements. No model prediction becomes an empirical child answer.
    answer = ({'status': 'scoped_unknown', 'reason': 'finite_experiment_search'} if distinguishing else
              prior_answer_execution or step_question(agent, question, laboratory, mission=mission))
    prepared = prepare_theories_for_question(comparison_question, pool, matching)
    unavailable_python_ids = {item['id'] for item in prepared['blocked']
                              if item['reason'] == 'retained_forecast_unavailable'}
    # These automatic columns were declared with this exact typed target. Their
    # failed/unfunded executions stay in the selector as UNKNOWN, rather than
    # turning the known subset into apparent complete catalog coverage.
    selection_theories = [*prepared['theories'], *(theory for theory in pool
        if theory['id'] in unavailable_python_ids
        and theory.get('forecast_scope', {}).get('kind') == 'ordinary_python_catalog')]
    work = []
    if not answer.get('parent_revision_completed'):
        for blocked in prepared['blocked']:
            action = blocked['required_work']
            kind, inputs = action.get('action'), action.get('arguments', {})
            if kind == 'evaluate' or (kind == 'run' and inputs.get('operator') == 'refit'):
                before_compute = agent.costs['compute_cost']
                original_config = agent.config
                if mission.get('purpose') in {'theory_learning', 'method_learning'}:
                    agent.config = replace(original_config,
                        compute_budget=before_compute + mission_compute_remaining(agent, mission))
                try:
                    feedback = agent._execute(kind, inputs, laboratory)
                except ValueError as error:
                    if (mission.get('purpose') not in {'theory_learning', 'method_learning'}
                            or str(error) not in {'candidate-row scoring budget exhausted',
                                                'shared research operator compute budget exhausted'}):
                        raise
                    feedback = {'status': 'scoped_unknown', 'reason': 'unknown_due_to_incomplete_search',
                                'detail': 'compute_budget', 'error': str(error)}
                finally:
                    agent.config = original_config
                    debit_mission_compute(agent, mission, before_compute)
                work.append({'source_id': blocked['id'], 'action': deepcopy(action), 'feedback': feedback})
    experiments = question.get('resolution_experiments', [])
    selection = None
    answered = not distinguishing and question.get('semantic_answer', {}).get('status') == 'answered'
    if (experiments and not answered and answer.get('reason') != 'premise_unknown'
            and answer.get('status') not in {'answer_applied', 'reused_answer'}):
        before_compute = agent.costs['compute_cost']
        research_comparison = resolve_research_comparison(agent, comparison_question, selection_theories)
        selection = select_question_plan(comparison_question, selection_theories, experiments,
            context=question.get('context_ref'),
            compute_budget=mission_compute_remaining(agent, mission),
            observation_budget=max(0, agent.config.observation_budget - agent.costs['observation_cost']),
            public_prediction_rows=getattr(laboratory, 'prediction_rows', None),
            exploration_policy=question.get('exploration_policy'),
            **({'research_comparison': research_comparison} if research_comparison is not None else {}))
        agent.costs['compute_cost'] += selection.get('compute_cost', 0)
        debit_mission_compute(agent, mission, before_compute)
    version = (previous or {}).get('version', 0) + 1
    result = {'status': answer['status'], 'question_id': question['id'], 'matching': matching,
              'answer_execution': answer, 'selection': selection, 'cache_hit': False,
              'frozen_recall': recalled,
              'python_forecasts': python_forecasts,
              'binding_work': prepared['blocked'], 'prepared_bindings': prepared['prepared'],
              'executed_reuse_work': work,
              'new_compute_cost': agent.costs['compute_cost'] - start}
    if track_agenda is not None:
        result.update(research_track_question_ref=selected_ref, parent_research_agenda=track_agenda)
    if selection is not None:
        result['status'] = selection['status']
        if (selection.get('query_arguments') and selection.get('selected_experiment')
                and not selection['selected_experiment'].get('direct_task_action')):
            result['parent_use'] = _prepare_selection(agent, question, selection, version)
    if previous and previous['basis'] != basis:
        question.setdefault('routing_history', []).append({
            'version': previous['version'], 'status': previous['result']['status'],
            'selected_experiment_id': (previous['result'].get('selection') or {}).get('selected_experiment', {}).get('id')
                if (previous['result'].get('selection') or {}).get('selected_experiment') else None})
    if distinguishing:
        result['query_answer'] = _distinguishing_answer(question, result)
    if track_agenda is not None:
        agent.research_agenda = deepcopy(track_agenda)
    question['routing_execution'] = {'basis': basis, 'allowance': allowance, 'version': version, 'result': deepcopy(result)}
    _refresh_completion(agent, mission, max(0, agent.config.compute_budget - agent.costs['compute_cost']),
                        compute_limit=compute_limit)
    return result

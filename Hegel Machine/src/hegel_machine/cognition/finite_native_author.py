"""A finite, freezable Common procedure using the retained research operators.

Install ``export_finite_native_author()`` as an ordinary select_actions file.
``memory['finite_native_author']`` declares its public ``question`` (including
theory_target roles/unit/context), the complete explicit ``predictor_paths`` and
``predictor_catalog_ref`` (including failed sources), and max_compute,
max_candidates, boundary_resolution. The catalog is retained, never filtered by
success. This policy supplies control flow, not a scientific predictor or rule.
An explicitly funded native_representation_plan declares parent_theory_ids,
validation_records, compute_limit and max_candidates. The same completed-choice
collision evidence may trigger this ordinary tool in every registered group.
"""

from copy import deepcopy
import json
from pathlib import Path


def export_finite_native_author():
    """Freeze this exact standalone procedure into Common's ordinary files."""
    return Path(__file__).read_text(encoding='utf-8')


def export_revision_candidates(agent, question):
    """Expose actual joint-revision results to the ordinary Python comparison.

    This host-side hook uses the existing exporter; the standalone finite
    procedure below still only chooses ordinary actions. A fixed executable can
    be compared even when its paid checks failed or remained incomplete. Its
    theory status, scientific exclusion and confirmation requirements survive.
    """
    from time import monotonic
    from .language import Program
    from .research_episodes import _question_versions, _path
    from .temporal_operator_agent import export_temporal_program
    from .theory_consolidation import _anchor_file_references, library_entry_reference

    result = {'status': 'no_new_revision_exports', 'changed': False, 'new_exports': 0,
        'exports': [], 'errors': [], 'new_compute_cost': 0, 'new_observations': 0,
        'support_updated': False, 'task_adoption_evaluated': False, 'python_execution_seconds': 0.}
    execution = question.get('frozen_recall_execution') or {}
    target = question.get('theory_target') or {}
    if (execution.get('basis', {}).get('target') != target
            or target.get('revision_fitting') is None):
        return result
    identifiers = list(dict.fromkeys(identifier
        for work in execution.get('result', {}).get('work', [])
        for identifier in work.get('theory_ids', [])))
    if not identifiers:
        return result
    started = monotonic()
    saved = question.setdefault('revision_candidate_exports', [])
    failures = question.get('revision_export_errors', [])
    prefix = f'frozen_recall:{question["id"]}@{question.get("meaning_version", 1)}:'
    owner, owner_ref = _question_versions(agent)[question['id'], question.get('meaning_version', 1)]
    if owner is not question:
        raise ValueError('revision_export_requires_original_question_meaning')
    directory = (Path('_experiment_models') /
        f'revision_{_path(owner_ref)[1]:04d}_{question.get("meaning_version", 1):04d}' /
        '_evaluated_model')
    documents = {}
    for identifier in identifiers:
        theory = agent.theory_index.get(identifier, {})
        binding = theory.get('consolidation_ref', {})
        if not identifier.startswith(prefix) or not binding.get('parent') or not theory.get('program'):
            continue
        theory_ref = {'id': identifier, 'version': theory.get('version', 1)}
        if any(item['theory_ref'] == theory_ref for item in saved):
            continue
        # A restored index retains local references while the installed library
        # has anchored its bindings to the original workspace. Compare the same
        # complete binding in that physical reference form.
        anchored_binding = _anchor_file_references(binding, agent.workspace)
        transfer = next(((index, record) for index, record in reversed(list(enumerate(
            agent.theory_consolidation.get('transfers', []))))
            if record.get('theory_ref') == theory_ref
            and _anchor_file_references(record.get('binding'), agent.workspace) == anchored_binding), None)
        try:
            if transfer is None:
                raise ValueError('exact_revision_result_reference_required')
            if (theory.get('context_ref') != target.get('context_ref')
                    or theory.get('output_unit') != target.get('output_unit')
                    or any(target.get('role_signature', {}).get(name) != contract
                           for name, contract in theory.get('role_signature', {}).items())):
                raise ValueError('revision_export_target_contract_mismatch')
            if agent.costs['python_execution_seconds'] + monotonic() - started >= agent.config.execution_seconds:
                raise ValueError('revision_export_python_budget_exhausted')
            source = export_temporal_program(Program.from_dict(theory['program']),
                                             feature_names=tuple(target['role_signature']))
            slots = list(dict.fromkeys(item['theory_ref']['id'] for item in saved))
            slot = slots.index(identifier) if identifier in slots else len(slots)
            model = str(directory / f'candidate_{slot:04d}_v{theory_ref["version"]:04d}.py')
            agent._save(model, source)
            entry = {'theory_ref': theory_ref, 'model': model, 'status': transfer[1]['status'],
                'support_status': theory.get('support', {}).get('status'),
                'program_source_ref': {'file': 'research_state.json',
                    'json_path': ['theory_index', identifier, 'program'], 'version': theory_ref['version']},
                'transfer_ref': library_entry_reference(agent, 'transfers', transfer[0], documents=documents),
                'support_updated': False, 'independent_confirmation': False}
            saved.append(entry)
            if model not in agent.candidate_paths:
                agent.candidate_paths.append(model)
            result['exports'].append(deepcopy(entry))
            result['changed'] = True
        except ValueError as error:
            failure = {'theory_ref': theory_ref, 'reason': str(error)}
            if failure not in failures:
                failures.append(failure)
                question['revision_export_errors'] = failures
                result['errors'].append(deepcopy(failure))
                result['changed'] = True
    if result['changed']:
        result['python_execution_seconds'] = monotonic() - started
        agent.costs['python_execution_seconds'] += result['python_execution_seconds']
    result.update(new_exports=len(result['exports']), status=(
        'exported_pending_comparison' if result['exports'] else
        'UNKNOWN' if result['errors'] else 'no_new_revision_exports'))
    return result


def _author_work(reason, required, **details):
    # The existing procedure executor retains this structured stop reason.
    return {'stop': {'status': 'author_work_request', 'request': {
        'type': 'AuthorWorkRequest', 'reason': reason, 'required': required, **details}}}


def _state(memory):
    path = Path('research_state.json')
    return json.loads(path.read_text()) if path.is_file() else memory.get('research_state')


def _action(proposal, question):
    result = deepcopy(proposal)
    result.setdefault('research_question', deepcopy(question))
    return result


def _attempted(proposal, results, paid_rows):
    def content(value):
        return {key: item for key, item in value.items() if key != 'research_question'}
    return any(content(item.get('proposal') or {}) == content(proposal)
        and item.get('evidence_after', {}).get('rows') == paid_rows for item in results)


def _completed_choice(results, models, paid_rows, question, state):
    """Reuse a matching completed comparison, including an earlier author version.

    The ordinary execution feedback retains the actual question and meaning
    after Common has removed the declaration from the operator's arguments.
    Include the same validated exports appended by the ordinary comparison;
    its explicit models argument need not list them a second time.
    """
    current = next((item for item in state.get('questions', [])
                    if item['id'] == question['id']), question)
    meaning = current.get('meaning_version', 1)
    expected = list(models)
    for result in state.get('operator_results', []):
        if result.get('operator') == 'refit' and result.get('status') == 'refit_supported_on_validation':
            expected.append(result['path'])
        elif result.get('operator') == 'response':
            expected.extend(item['path'] for item in result.get('candidates', [])
                if item.get('validation_mse') is not None and item['validation_mse'] <= 1e-10)
    expected = list(dict.fromkeys(expected))
    for entry in reversed(results):
        proposal, feedback = entry.get('proposal') or {}, entry.get('feedback') or {}
        arguments = proposal.get('arguments', {})
        evaluations = feedback.get('evaluations', [])
        if (proposal.get('action') == 'run' and arguments.get('operator') == 'research_choice'
                and arguments.get('models') == models
                and feedback.get('operator') == 'research_choice'
                and feedback.get('research_question_id') == question['id']
                and feedback.get('question_meaning_version') == meaning
                and feedback.get('paid_rows') == paid_rows
                and not feedback.get('error') and not feedback.get('truncated')
                and [item.get('path') for item in evaluations] == expected
                and not any(item.get('truncated') for item in evaluations)):
            return feedback
    return None


def _same_bank_after_unavailable_refit(state, results, question, latest, paid_rows,
                                      observed_records, max_compute):
    """Consume the native researcher's published full-paid continuation action."""
    owner_index = next((index for index, item in enumerate(state.get('questions', []))
                        if item['id'] == question['id']), None)
    if owner_index is None or latest is None:
        return None
    owner = state['questions'][owner_index]
    meaning = owner.get('meaning_version', 1)
    if not any((entry.get('proposal') or {}).get('arguments', {}).get('operator') == 'refit'
            and entry.get('feedback', {}).get('status') == 'refit_unavailable'
            and entry['feedback'].get('research_question_id') == question['id']
            and entry['feedback'].get('question_meaning_version', 1) == meaning
            and entry.get('evidence_after', {}).get('rows') == paid_rows for entry in results):
        return None
    path = Path('method_decision.json')
    if not path.is_file():
        return None
    for option in json.loads(path.read_text()).get('research_options', []):
        if option.get('id') != 'native_condition_research':
            continue
        for entry in reversed(option.get('current_executions', [])):
            source = entry.get('source', {})
            if (entry.get('question_id') != question['id']
                    or entry.get('round_index') != latest.get('round_index')
                    or entry.get('fit_scope') != 'all_paid_rows'
                    or entry.get('paid_rows') != paid_rows
                    or entry.get('current_paid_rows') != paid_rows
                    or entry.get('observed_record_count') != observed_records
                    or source.get('file') != 'research_state.json'
                    or source.get('json_path') != ['questions', owner_index,
                        'native_condition_research', latest.get('round_index')]):
                continue
            proposal = deepcopy(entry.get('resume_action'))
            if not proposal:
                return None
            arguments = proposal.get('arguments', {})
            if (proposal.get('action') != 'run'
                    or arguments.get('operator') != 'native_condition_research'
                    or arguments.get('research_question', {}).get('id') != question['id']):
                return None
            arguments['max_compute'] = max_compute
            return proposal if not _attempted(proposal, results, paid_rows) else None
    return None


def _comparison_models(old, exports, native):
    models = list(old) + list(exports)
    for round_ in native:
        models.extend(item['model'] for item in round_.get('candidates', []) if item.get('model'))
        models.extend(item['model'] for item in round_.get('exported_models', []) if item.get('model'))
    return list(dict.fromkeys(models))


def _json_reference_value(reference):
    # Author files run without repository imports; these declared local JSON
    # records are the existing canonical score/program protocol.
    with Path(reference['file']).open() as stream:
        value = json.load(stream)
    for part in reference.get('json_path', []):
        value = value[part]
    return value


def _retained_representation_parent(state, results, model, target):
    """Resolve a real export address; an opaque draft stays an author question."""
    indexed = state.get('theory_index', {})
    proposals = []
    owners = []
    for qi, owner in enumerate(state.get('questions', [])):
        owners.append((owner, ['questions', qi]))
        owners.extend((old, ['questions', qi, 'meaning_history', hi])
                      for hi, old in enumerate(owner.get('meaning_history', [])))
    for owner, owner_path in owners:
        for ri, round_ in enumerate(owner.get('native_condition_research', [])):
            for ci, candidate in enumerate(round_.get('candidates', [])):
                if candidate.get('model') == model and candidate.get('program'):
                    proposals.append((candidate['program'], {'file': 'research_state.json',
                        'json_path': [*owner_path, 'native_condition_research', ri, 'candidates', ci, 'program']}))
        for export in owner.get('revision_candidate_exports', []):
            theory = indexed.get(export.get('theory_ref', {}).get('id'), {})
            if export.get('model') == model and theory.get('program'):
                proposals.append((theory['program'], export['program_source_ref']))
    feedbacks = [entry.get('feedback', {}) for entry in results]
    feedbacks.extend(owner.get('routing_execution', {}).get('result', {}) for owner, _ in owners)
    referenced_programs = {}
    for feedback in feedbacks:
        for export in feedback.get('exported_models', []):
            if export.get('model') != model:
                continue
            for candidate in feedback.get('native_representation', {}).get('candidates', []):
                if candidate.get('id') == export.get('candidate_id'):
                    reference = deepcopy(export['source_ref'])
                    reference['json_path'].append('program')
                    program = candidate.get('program')
                    if program is None:
                        key = reference['file'], tuple(reference['json_path'])
                        if key not in referenced_programs:
                            referenced_programs[key] = _json_reference_value(reference)
                        program = referenced_programs[key]
                    proposals.append((program, reference))
    if not proposals or any(program != proposals[0][0] for program, _ in proposals):
        return None
    program, reference = proposals[0]
    # The target's roles/units are already explicitly declared by the common
    # public contract. This carries a Program address, never inherited support.
    identifier = f'retained_representation_parent:{model}'
    return {'id': identifier, 'program': deepcopy(program),
        'program_source_ref': deepcopy(reference),
        'role_signature': deepcopy(target['role_signature']), 'output_unit': target['output_unit'],
        'context_ref': deepcopy(target['context_ref']),
        'source_snapshot': {'model': model, 'source_ref': deepcopy(reference)},
        'support': {'status': 'candidate', 'evidence': [], 'independent_confirmation': False}}


def bind_actual_representation_parent(agent, question):
    """Bind the actual retained native-representation export, in every group."""
    from .language import Program
    from .question_routing import register_theories
    from .temporal_operator_agent import export_temporal_program

    model = (agent.evaluated_model or {}).get('source_model')
    results = list(getattr(agent, 'method_results', []))
    for owner in agent.questions:
        for version in [owner, *owner.get('meaning_history', [])]:
            results.append({'feedback': version.get('routing_execution', {}).get('result', {})})
    results = [entry for entry in results if entry.get('feedback', {}).get('native_representation')
        and any(export.get('model') == model
                for export in entry['feedback'].get('exported_models', []))]
    if not results:
        return {'status': 'not_representation_export', 'source_model': model}
    target = question.get('theory_target') or {}
    roles = target.get('role_signature')
    def missing(reason):
        return {'status': 'author_work_request', 'reason': reason, 'source_model': model,
            'required': ['actual retained export and explicitly declared roles/units/context'],
            'new_compute_cost': 0, 'support_updated': False}
    if not isinstance(roles, dict) or target.get('output_unit') is None or target.get('context_ref') is None:
        return missing('actual_representation_parent_contract_required')
    parent = _retained_representation_parent({'questions': [], 'theory_index': agent.theory_index},
                                            results, model, target)
    if parent is None:
        return missing('actual_representation_parent_program_required')
    program = Program.from_dict(parent['program'])
    if any(not isinstance(roles.get(name), dict)
           or any(roles[name].get(key) is None for key in ('role', 'unit', 'time'))
           for name in program.required_features()):
        return missing('actual_representation_parent_contract_required')
    expected = export_temporal_program(program, feature_names=tuple(roles)).encode('utf-8')
    if ((agent.workspace / model).read_bytes() != expected
            or (agent.workspace / agent._evaluated_summary()['model']).read_bytes() != expected):
        return missing('actual_representation_parent_export_changed')
    previous = agent.theory_index.get(parent['id'])
    if previous and all(previous.get(key) == parent[key] for key in
            ('program', 'program_source_ref', 'role_signature', 'output_unit', 'context_ref')):
        parent = {key: deepcopy(value) for key, value in previous.items() if key not in {'version', 'history'}}
    register_theories(agent, [parent])
    return {'status': 'bound', 'theory_id': parent['id'],
        'version': agent.theory_index[parent['id']]['version'],
        'source_ref': deepcopy(parent['program_source_ref']), 'source_model': model,
        'new_compute_cost': 0, 'support_updated': False}


def _representation_action(contract, question, choice, state, memory, paid_rows):
    """The same finite evidence-triggered repair is available to A, B and C."""
    retained = choice.get('evaluated_model') or {}
    model = retained.get('source_model')
    evaluation = next((item for item in choice.get('evaluations', [])
                       if item.get('path') == model), {})
    diagnostic = evaluation.get('representation_diagnostic') or {}
    if not diagnostic:
        review = evaluation.get('representation_review', {}).get('review_ref')
        if review:
            diagnostic = _json_reference_value(review).get('diagnostic', {})
        elif evaluation.get('score_result_ref'):
            diagnostic = _json_reference_value(evaluation['score_result_ref']).get(
                'representation_diagnostic', {})
    threshold = state.get('limits', {}).get('confirmation_mse_threshold', 1e-10)
    if (evaluation.get('returncode') != 0 or diagnostic.get('empirical_mse_floor', 0) <= threshold
            or not diagnostic.get('conflicting_groups')):
        return None
    plan = contract.get('native_representation_plan')
    if (not isinstance(plan, dict) or not isinstance(plan.get('parent_theory_ids'), list)
            or not plan['parent_theory_ids'] or not isinstance(plan.get('validation_records'), list)
            or not plan['validation_records'] or type(plan.get('compute_limit')) is not int
            or plan['compute_limit'] <= 0 or type(plan.get('max_candidates')) is not int
            or plan['max_candidates'] < 2):
        return _author_work('native_representation_plan_required',
            ['native_representation_plan.parent_theory_ids', 'validation_records',
             'compute_limit', 'max_candidates'], source_model=model,
            representation_diagnostic=deepcopy(diagnostic))
    all_results = list(state.get('method_results', [])) + list(memory.get('program_results', []))
    expected_plan = deepcopy(plan)
    expected_plan['parent_theory_ids'] = list(dict.fromkeys([
        *plan['parent_theory_ids'], f'retained_representation_parent:{model}']))
    # Stop unchanged work before rereading an already consumed canonical draft.
    if any((entry.get('proposal') or {}).get('arguments', {}).get('operator') == 'research_route'
            and (entry.get('proposal') or {}).get('arguments', {}).get('research_question', {}).get('id')
                == f'representation:{model}'
            and (entry.get('proposal') or {}).get('arguments', {}).get('native_representation_plan') == expected_plan
            and entry.get('evidence_after', {}).get('rows') == paid_rows for entry in all_results):
        return _author_work('representation_research_return_requires_supported_revision_or_new_evidence',
            ['actual changed draft, new counterexample, or revised explicit research contract'],
            source_model=model, question_ref={'question_id': f'representation:{model}'})
    parent = _retained_representation_parent(state, all_results, model, question['theory_target'])
    if parent is None:
        return _author_work('actual_retained_representation_program_required',
            ['registered actual retained Program with export address'], source_model=model,
            representation_diagnostic=deepcopy(diagnostic))
    child = {'id': f'representation:{model}',
        'question': 'Which public input distinctions were lost by the actual retained predictor?',
        'kind': 'representation', 'origin': 'computed_representation_conflict',
        'parent_id': question['id'], 'depends_on': [], 'answer_type': 'program',
        'subject_refs': [{'type': 'candidate', 'path': model}],
        'context_ref': deepcopy(question.get('context_ref')),
        'answer_resolution_ref': {'evaluator': 'representation_diagnostic_and_paid_comparison',
                                  'parent_model': model},
        'query_semantics': {'mode': 'solution_relation'},
        'representation_diagnostic': deepcopy(diagnostic),
        'use_sites': [{'kind': 'supply_component', 'model': model, 'slot': 'representation'}]}
    native_plan = deepcopy(plan)
    native_plan['parent_theory_ids'] = list(dict.fromkeys([*plan['parent_theory_ids'], parent['id']]))
    proposal = {'action': 'run', 'arguments': {'operator': 'research_route',
        'research_question': child, 'theories': [parent], 'native_representation_plan': native_plan}}
    if _attempted(proposal, all_results, paid_rows):
        return _author_work('representation_research_return_requires_supported_revision_or_new_evidence',
            ['actual changed draft, new counterexample, or revised explicit research contract'],
            source_model=model, question_ref={'question_id': child['id']})
    return proposal


def ordinary_completion_reserve(agent, question):
    """Leave the finite task's complete comparison and original finish funded."""
    from .python_research_routing import validated_candidates
    from .question_agenda import effective_question_mission

    contract = agent.knowledge.get('finite_native_author') or {}
    # Registered routing supplies a task mission after any preceding theory work.
    mission = effective_question_mission(question, {'purpose': 'task'})
    if (not contract or question.get('theory_target', {}).get('revision_fitting') is None
            or mission.get('purpose') in {'theory_learning', 'method_learning'}):
        return None
    native = [item['native_condition_research'] for item in agent.operator_results
        if item.get('operator') == 'native_condition_research'
        and item.get('research_question_id') == question['id']
        and item.get('question_meaning_version', 1) == question.get('meaning_version', 1)
        and item.get('native_condition_research')]
    models = _comparison_models(contract['predictor_paths'], agent.candidate_paths, native)
    models = list(dict.fromkeys([*models, *validated_candidates(agent)]))
    rows = len(agent.rows)
    stale = bool(agent.evaluated_model is not None
                 and agent.evaluated_model['evaluated_rows'] != rows)
    deliveries = len(agent._inquiry_stack)
    return {'compute_cost': rows * (len(models) + int(stale) + deliveries + 1)
                           + agent.config.confirmation_rows,
        'paid_rows': rows, 'models': len(models), 'stale_incumbent': stale,
        'local_deliveries': deliveries, 'parent_finish_passes': 1,
        'confirmation_rows': agent.config.confirmation_rows,
        'comparison_compute_per_new_proposal': rows}


def select_actions(public_spec, observations, memory):
    contract = memory.get('finite_native_author') or {}
    question = contract.get('question') or {}
    target = question.get('theory_target') or {}
    names = [item['name'] if isinstance(item, dict) else item for item in public_spec.get('controls', [])]
    roles = target.get('role_signature')
    if (not question.get('id') or not question.get('question') or not names
            or not isinstance(roles, dict) or set(roles) != set(names)
            or any(not isinstance(role, dict) or any(role.get(key) is None for key in ('role', 'unit', 'time'))
                   for role in roles.values())
            or target.get('output_unit') is None or target.get('context_ref') is None
            or question.get('context_ref') != target['context_ref']):
        return _author_work('public_typed_research_contract_required',
            ['question.id', 'question.question', 'question.context_ref',
             'question.theory_target.role_signature', 'output_unit', 'context_ref'])
    old = contract.get('predictor_paths')
    if (not isinstance(old, list) or any(not isinstance(path, str) or not path for path in old)
            or not contract.get('predictor_catalog_ref')):
        return _author_work('complete_retained_predictor_catalog_required',
            ['predictor_paths', 'predictor_catalog_ref'],
            catalog_ref=deepcopy(contract.get('predictor_catalog_ref')))
    if (type(contract.get('max_compute')) is not int or contract['max_compute'] < 0
            or type(contract.get('max_candidates')) is not int or contract['max_candidates'] < 1
            or not isinstance(contract.get('boundary_resolution'), (int, float))
            or contract['boundary_resolution'] <= 0):
        return _author_work('bounded_native_research_allocation_required',
                            ['max_compute', 'max_candidates', 'boundary_resolution'])
    state = _state(memory)
    if state is None:
        return _author_work('current_research_state_required', ['research_state.json'])
    missing = [path for path in old if not Path(path).is_file()]
    if missing:
        return _author_work('retained_predictor_source_missing', missing,
                            catalog_ref=deepcopy(contract['predictor_catalog_ref']))
    parent_id = question.get('parent_id') or 'task'
    parent = next((item for item in state.get('questions', []) if item['id'] == parent_id), {})
    paid_rows = sum(len(record.get('rows', [])) for record in observations)
    if not paid_rows:
        return _author_work('registered_source_observations_required', ['paid public prefix'])
    operators = state.get('operator_results', [])
    native_results = [item for item in operators
        if item.get('operator') == 'native_condition_research'
        and item.get('research_question_id') == question['id']
        and item.get('question_meaning_version', 1) == question.get('meaning_version', 1)
        and item.get('native_condition_research')]
    native = [item['native_condition_research'] for item in native_results]
    latest = native[-1] if native else None
    # PR2's frozen query need not carry the native method's after_observation.
    # Consume its new paid records with the original retained-bank search before
    # comparing old exports or using experiments proposed on the old prefix.
    if latest and latest.get('observed_record_count', len(observations)) < len(observations):
        first = latest['observed_record_count']
        added = list(range(first, len(observations)))
        covered_records = latest['fit_record_indices']
        covered_rows = set(latest['fit_evidence'])
        if any(index not in covered_records or
               any(row not in covered_rows for row in observations[index]['row_indices'])
               for index in added):
            previous = native_results[-1].get('arguments', {})
            arguments = {key: deepcopy(previous[key]) for key in ('basis', 'matched_pairs') if key in previous}
            if covered_records != list(range(first)):
                arguments['record_indices'] = list(dict.fromkeys([*covered_records, *added]))
            arguments.update(operator='native_condition_research', research_question=deepcopy(question),
                max_compute=contract['max_compute'], max_candidates=contract['max_candidates'])
            return {'action': 'run', 'arguments': arguments}
    models = _comparison_models(old, state.get('exported_candidate_paths', []), native)
    missing = [path for path in models if not Path(path).is_file()]
    if missing:
        return _author_work('retained_predictor_source_missing', missing,
                            catalog_ref=deepcopy(contract['predictor_catalog_ref']))
    meaning = next((item.get('meaning_version', 1) for item in state.get('questions', [])
                    if item['id'] == question['id']), question.get('meaning_version', 1))
    # A source edit resets program_results, while its actual scientific
    # executions retain their original source, question, meaning and paid scope.
    results = [entry for entry in state.get('method_results', [])
        if entry.get('feedback', {}).get('research_question_id') == question['id']
        and entry['feedback'].get('question_meaning_version', 1) == meaning]
    results += memory.get('program_results', [])
    # The existing choice scores every named old/native model and adds all
    # validated refit/response exports itself. Its retention/confirmation gate
    # decides among them before another purchase or refinement can intervene.
    # This procedure never selects the native minimum.
    choice = _completed_choice(results, models, paid_rows, question, state)
    current_choice = choice is not None
    if models and not current_choice:
        return _action({'action': 'run', 'arguments': {'operator': 'research_choice', 'models': models}}, question)
    if current_choice:
        representation = _representation_action(contract, question, choice, state, memory, paid_rows)
        if representation is not None:
            return representation
    if current_choice and choice.get('next_action') and not _attempted(choice['next_action'], results, paid_rows):
        return _action(choice['next_action'], question)
    if current_choice:
        resumed = _same_bank_after_unavailable_refit(state, results, question, latest,
            paid_rows, len(observations), contract['max_compute'])
        if resumed is not None:
            return _action(resumed, question)
    if parent.get('pending_query'):
        return _action({'action': 'query', 'arguments': {'pending_question': parent_id}}, question)
    # Retain the old method's actual boundary/matched-evidence continuation.
    if latest:
        interval = latest.get('remaining_boundary', {}).get('threshold_interval', [])
        refine = len(interval) == 2 and interval[1] - interval[0] > contract['boundary_resolution']
        for experiment in latest.get('proposed_experiments', []):
            proposal = experiment.get('query_action')
            if (proposal and (refine and experiment.get('purpose') == 'conditional_boundary'
                              or latest.get('status') == 'needs_matched_evidence')
                    and not _attempted(proposal, results, paid_rows)):
                return _action(proposal, question)
    if latest is None:
        return {'action': 'run', 'arguments': {'operator': 'native_condition_research',
            'research_question': deepcopy(question), 'max_compute': contract['max_compute'],
            'max_candidates': contract['max_candidates']}}
    for experiment in latest.get('proposed_experiments', []):
        proposal = experiment.get('query_action')
        if proposal and not _attempted(proposal, results, paid_rows):
            return _action(proposal, question)
    return _author_work('executable_revision_or_scope_required',
        ['revised executable candidate, fitting scope, or finite research policy'],
        question_ref={'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)},
        native_status=latest.get('status'), catalog_ref=deepcopy(contract['predictor_catalog_ref']))

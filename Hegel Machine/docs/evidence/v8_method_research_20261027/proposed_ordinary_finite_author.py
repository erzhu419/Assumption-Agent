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
from math import isfinite


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
            or target.get('revision_fitting') is None and target.get('research_program_fitting') is None):
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
        method = binding.get('kind') == 'research_program'
        if (not identifier.startswith(prefix) or not theory.get('program')
                or not binding.get('parent') and not method):
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
            if method and (not binding.get('source_ref') or not binding.get('fit_split')):
                raise ValueError('exact_replayed_method_fit_reference_required')
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


def _bound_research_proposal(proposal, question):
    result = deepcopy(proposal)
    owner = result.get('research_question') or result.get('arguments', {}).get('research_question') \
        or result.get('arguments', {}).get('question') or {}
    if isinstance(owner, str) and owner == question.get('id'):
        owner = {'id': owner}
    if (not owner or isinstance(owner, dict) and owner.get('id') == question.get('id')
            and owner.get('meaning_version', question.get('meaning_version', 1))
                == question.get('meaning_version', 1)):
        fields = ('id', 'question', 'parent_id', 'depends_on', 'meaning_version',
                  'kind', 'answer_type', 'query_semantics', 'context_ref', 'answer_resolution_ref')
        result['research_question'] = {key: deepcopy(owner.get(key, question.get(key)))
            for key in fields if key in owner or key in question}
        result['research_question'].setdefault('meaning_version', 1)
    return result


def research_action_semantics(proposal, binding=None):
    """Compare executable proposals, excluding their locations and commentary.

    A Python path alone is not exposed program content. Callers comparing two
    different bound programs supply the existing binding/replay input view.
    Native terms, evidence selections, callbacks and resource limits remain.
    This finite comparison only groups trials on the declared public scene.
    """
    locations = {'id', 'version', 'method_id', 'theory_id', 'candidate_id',
        'source_id', 'question_id', 'path', 'file', 'json_path', 'source_ref',
        'proposal_ref', 'method_ref', 'source_refs', 'description', 'reason',
        'question', 'lesson', 'status', 'scope_note'}

    def value(item):
        if isinstance(item, dict):
            return {key: value(child) for key, child in item.items()
                if key not in locations and not key.endswith('_ref')}
        if isinstance(item, (list, tuple)):
            return [value(child) for child in item]
        return item

    result = {'action': proposal.get('action'),
        'arguments': value(proposal.get('arguments', {}))}
    owner = proposal.get('research_question') or proposal.get('arguments', {}).get('question')
    if isinstance(owner, dict):
        result['question_semantics'] = {key: value(owner[key]) for key in (
            'meaning_version', 'kind', 'answer_type', 'query_semantics', 'context_ref',
            'answer_resolution_ref') if key in owner}
    if binding is not None:
        result['program_binding'] = value(binding)
    return result


def _complete_research_attempt(proposal, results, paid_rows, binding=None, question=None):
    expected = research_action_semantics(proposal, binding)
    for entry in results:
        feedback = entry.get('feedback') or {}
        if (entry.get('evidence_before', entry.get('evidence_after', {})).get('rows') != paid_rows
                or feedback.get('error') or feedback.get('truncated')):
            continue
        if question is not None and feedback.get('question_meaning_version', 1) != question.get('meaning_version', 1):
            continue
        actual = entry.get('proposal') or {}
        if question is not None:
            actual = _bound_research_proposal(actual, question)
        if research_action_semantics(actual, entry.get('program_binding', actual.get('program_binding'))) != expected:
            continue
        native = feedback.get('native_condition_research') or {}
        if (feedback.get('execution_complete') is True or
                feedback.get('status') in {'completed', 'complete', 'transferred'} or
                native.get('status') in {'candidates_require_evaluation', 'matched_evidence_without_candidate'}):
            return True
    return False


def research_segment_options(public_spec, observations, memory, *, state=None, decision=None):
    """Expose only current, fully quoted ordinary R/N/Q proposals; never run one."""
    state = _state(memory) if state is None else state
    if decision is None:
        decision = memory.get('method_decision')
        if decision is None:
            path = Path('method_decision.json')
            decision = json.loads(path.read_text()) if path.is_file() else {}
    state = state or {}
    contract = memory.get('finite_native_author') or {}
    declaration = contract.get('question') or {}
    question = next((item for item in state.get('questions', [])
        if item.get('id') == declaration.get('id')), declaration)
    rows = sum(len(record.get('rows', [])) for record in observations)
    native = [entry['native_condition_research'] for entry in state.get('operator_results', [])
        if entry.get('operator') == 'native_condition_research'
        and entry.get('research_question_id') == question.get('id')
        and entry.get('question_meaning_version', 1) == question.get('meaning_version', 1)
        and entry.get('native_condition_research')]
    latest = native[-1] if native else {}
    models = _comparison_models(contract.get('predictor_paths', []),
        state.get('exported_candidate_paths', []), native)
    for entry in state.get('operator_results', []):
        if entry.get('operator') == 'refit' and entry.get('status') == 'refit_supported_on_validation':
            models.append(entry['path'])
        elif entry.get('operator') == 'response':
            models.extend(item['path'] for item in entry.get('candidates', [])
                if item.get('validation_mse') is not None and item['validation_mse'] <= 1e-10)
    models = list(dict.fromkeys(models))
    results = [*state.get('method_results', []), *memory.get('program_results', [])]
    limits, costs = state.get('limits', {}), state.get('costs', {})
    remaining = {name: max(0, limits.get(limit, 0) - costs.get(name, 0))
        for name, limit in (('compute_cost', 'compute_budget'),
                            ('observation_cost', 'observation_budget'))}
    raw = []
    for index, option in enumerate(decision.get('research_options', [])):
        reference = {'file': 'method_decision.json', 'json_path': ['research_options', index]}
        if option.get('id') == 'native_condition_research':
            for execution_index, entry in enumerate(option.get('current_executions', [])):
                offer = entry.get('compute_continuation') or {}
                unit = offer.get('research_unit') or {}
                reserve = unit.get('completion_reserve') or {}
                raw.append({'family': 'N', 'proposal': offer.get('action'),
                    'proposal_ref': {**reference, 'json_path': [*reference['json_path'],
                        'current_executions', execution_index, 'compute_continuation', 'action']},
                    'question_ref': {'id': entry.get('question_id'),
                        'meaning_version': question.get('meaning_version', 1)},
                    'round_index': entry.get('round_index'),
                    'complete_quote': {'status': offer.get('status'),
                        'compute_cost': unit.get('minimum_compute'), 'observation_cost': 0,
                        'paid_rows': unit.get('quoted_paid_rows'),
                        'retained_pool_models': reserve.get('retained_pool_models'),
                        'max_exported_models': reserve.get('max_exported_models'),
                        'remaining_execution_seconds': offer.get('remaining_execution_seconds'),
                        'research_unit': unit}, 'work_reason': unit.get('work_reason')})
        elif option.get('proposal'):
            raw.append({**deepcopy(option), 'proposal_ref': option.get('proposal_ref', reference)})
    options, excluded = [], []
    for option in raw:
        proposal = _bound_research_proposal(option.get('proposal') or {}, question)
        quote = option.get('complete_quote') or {}
        family = option.get('family')
        owner = option.get('question_ref') or proposal.get('research_question') or {}
        actual_owner = proposal.get('research_question') or {}
        reason = None
        if (family not in {'R', 'N', 'Q'} or not proposal.get('action')
                or owner.get('id', question.get('id')) != question.get('id')
                or owner.get('meaning_version', 1) != question.get('meaning_version', 1)
                or not isinstance(actual_owner, dict)
                or actual_owner.get('id') != question.get('id')
                or actual_owner.get('meaning_version', 1) != question.get('meaning_version', 1)):
            reason = 'current_question_proposal_required'
        elif (quote.get('status') != 'available' or quote.get('paid_rows') != rows
                or quote.get('retained_pool_models') != len(models)
                or type(quote.get('max_exported_models')) is not int
                or any(type(quote.get(key)) not in (int, float) or not isfinite(quote[key])
                       or quote[key] < 0 for key in ('compute_cost', 'observation_cost'))
                or not isinstance(quote.get('remaining_execution_seconds'), (int, float))
                or quote['remaining_execution_seconds'] <= 0):
            reason = 'fresh_complete_quote_required'
        elif any(quote[key] > remaining[key] for key in ('compute_cost', 'observation_cost')):
            reason = 'complete_unit_unfunded'
        elif family == 'N' and (option.get('round_index') != latest.get('round_index')
                or not (latest.get('status') == 'search_incomplete'
                    or option.get('work_reason') == 'new_paid_evidence'
                    and len(latest.get('fit_evidence', [])) < rows
                    or option.get('work_reason') == 'changed_construction'
                    and option.get('construction_before') is not None
                    and any(research_action_semantics(proposal)[key] != research_action_semantics(
                        option['construction_before'])[key] for key in ('action', 'arguments')))):
            reason = 'no_unfinished_or_changed_native_work'
        elif (family == 'N' and max(len(latest.get('candidates', [])),
                len(latest.get('exported_models', [])))
                + len(proposal.get('arguments', {}).get('programs', []))
                + proposal.get('arguments', {}).get('max_candidates', 0)
                > quote['max_exported_models']):
            reason = 'changed_export_bound_requires_new_quote'
        elif (family == 'R' and type((option.get('program_binding') or {}).get('source_seed_count')) is int
                and option['program_binding']['source_seed_count'] > quote['max_exported_models']):
            reason = 'changed_export_bound_requires_new_quote'
        elif family in {'N', 'R'} and _complete_research_attempt(
                proposal, results, rows, option.get('program_binding'), question):
            reason = 'same_completed_research'
        elif family == 'R' and option.get('program_binding') is None:
            reason = 'executable_program_binding_required'
        elif family == 'Q':
            reason = _query_segment_problem(option, public_spec, observations,
                question, contract, results, rows)
        if reason:
            excluded.append({'proposal_ref': deepcopy(option.get('proposal_ref')),
                'family': family, 'reason': reason})
        else:
            options.append({**deepcopy(option), 'proposal': proposal,
                'question_declaration': deepcopy(proposal['research_question']),
                'semantic_action': research_action_semantics(
                proposal, option.get('program_binding'))})
    return {'options': options, 'excluded': excluded,
        'question_declaration': _bound_research_proposal({'action': 'run'}, question)['research_question'],
        'scope': 'current_quoted_public_options_no_fitting_prediction_or_acquisition'}


def _query_segment_problem(option, public_spec, observations, question, contract, results, rows):
    proposal = option['proposal']
    arguments = proposal.get('arguments', {})
    actions = arguments.get('actions', [])
    names = {item['name'] if isinstance(item, dict) else item for item in public_spec.get('controls', [])}
    bounds = public_spec.get('feature_bounds', {})
    for action in actions:
        sequence = action.get('sequence', [])
        if (action.get('kind') != 'trajectory' or not sequence
                or len(sequence) > public_spec.get('max_probe_length', float('inf'))
                or any(set(point) != names or any(type(value) not in (int, float)
                    or not isfinite(value) or name in bounds
                    and not bounds[name][0] <= value <= bounds[name][1]
                    or name in public_spec.get('control_levels', {})
                    and value not in public_spec['control_levels'][name]
                    for name, value in point.items()) for point in sequence)):
            return 'illegal_public_query'
    pairs = option.get('matched_action_pairs', [])
    difference = option.get('predicted_separation', 0) > 0 or any(
        pair.get('predicted_separation', 0) > 0 for pair in pairs)
    pending_pair = bool(option.get('unfinished_pair_ref') and pairs)
    if (not actions or not (difference or pending_pair)
            or not (arguments.get('after_observation') or option.get('research_continuation'))):
        return 'bound_distinction_and_analysis_required'
    if any(any(type(pair.get(key)) is not int or not 0 <= pair[key] < len(actions)
        for key in ('reference_action_index', 'action_index')) for pair in pairs):
        return 'complete_pair_required'
    records = paid_experiment_record_indices(actions, observations, public_spec, question, contract)
    needed = sum(len(action['sequence']) for action, record in zip(actions, records) if record is None)
    if option['complete_quote']['observation_cost'] != needed:
        return 'complete_remaining_pair_quote_required'
    if all(record is not None for record in records):
        if not any('boundary_update' in callback for callback in arguments.get('after_observation', [])):
            return 'physical_answer_already_available'
        if any(not (entry.get('feedback') or {}).get('error')
                and (entry.get('feedback') or {}).get('record_indices') == records
                and _boundary_callbacks_completed(_action(proposal, question),
                    entry['feedback'], records) for entry in results):
            return 'same_analysis_already_completed'
    return None


def _current_query_action(experiment, public_spec, observations, memory, state, question):
    """Give P0 the same shared Q callback/price repair, preserving its Q order."""
    original = experiment['query_action']
    if not experiment.get('research_continuation'):
        return _action(original, question)
    decision = memory.get('method_decision')
    if decision is None:
        path = Path('method_decision.json')
        if not path.is_file():
            return _action(original, question)
        decision = json.loads(path.read_text())
    for item in decision.get('research_options', []):
        if (item.get('family') == 'Q' and item.get('query_action') == original
                and item.get('matched_action_pairs', []) == experiment.get('matched_action_pairs', [])):
            view = research_segment_options(public_spec, observations, memory, state=state,
                decision={'research_options': [item]})
            if view['options']:
                return deepcopy(view['options'][0]['proposal'])
            return _author_work('complete_query_research_unit_unavailable',
                ['a current payable complete query and original analysis contract'],
                complete_quote=deepcopy(item.get('complete_quote')),
                excluded=deepcopy(view['excluded']))
    return _author_work('complete_query_research_unit_quote_required',
        ['future-data complete quotation for the actual proposed query and continuation'])


def select_research_segment(public_spec, observations, memory, *, plan,
                            ordinary_select_actions, condition=None):
    """Select at most two learned segments, retaining ordinary delivery duties.

    The ordinary program owns comparison, parent retention and finish. A policy
    only chooses among the current public, fully funded proposals. Its saved
    action is a semantic selector, never an old action/quote to replay.
    """
    ordinary = ordinary_select_actions(public_spec, observations, memory)
    arguments = ordinary.get('arguments', {})
    if (ordinary.get('action') in {'finish', 'evaluate'}
            or arguments.get('operator') == 'research_choice'):
        return ordinary
    view = research_segment_options(public_spec, observations, memory)
    options = view['options']
    branch = plan
    if plan.get('condition') is not None:
        # The compiler supplies the existing exported condition evaluator. A
        # missing public feature does not silently choose the false branch.
        if condition is None:
            return ordinary
        try:
            selected = condition(memory)
        except (KeyError, TypeError, ValueError):
            return ordinary
        if selected is None:
            return ordinary
        branch = plan.get('when_true' if selected else 'when_false') or {}
    selectors = [item for item in (branch.get('first_segment'), branch.get('second_segment'))
                 if item is not None]
    current_question = research_action_semantics({'action': 'run',
        'research_question': view['question_declaration']})['question_semantics']
    selectors = [{**item, 'question_semantics': current_question}
        if 'question_semantics' not in item else item for item in selectors]
    if len(selectors) > 2:
        raise ValueError('a short research program permits at most two learned segments')
    history = memory.get('program_results', [])
    completed = []
    for entry in history:
        proposal, feedback = entry.get('proposal') or {}, entry.get('feedback') or {}
        if 'research_segment_index' not in proposal:
            continue
        completed.append(research_action_semantics(proposal,
            entry.get('program_binding', proposal.get('program_binding'))))
    if len(completed) >= len(selectors):
        return ordinary
    requested = selectors[len(completed)]
    # A true first segment return is necessary before a learned second choice;
    # callback/closure actions in between do not consume another learned slot.
    if completed and completed[0] != selectors[0]:
        return ordinary
    selected = next((item for item in options if item['semantic_action'] == requested), None)
    if selected is None:
        return ordinary
    action = deepcopy(selected['proposal'])
    action['research_segment_index'] = len(completed)
    if selected.get('program_binding') is not None:
        action['program_binding'] = deepcopy(selected['program_binding'])
    return action


def paid_experiment_record_indices(actions, observations, public_spec, question, contract,
                                   *, prediction_rows=None):
    """Bind reset-trajectory answers to their actual paid records, without buying.

    This is the finite author's fixed public episode, not a cache for arbitrary
    laboratories. A new parent can use the same fact; a declared distribution,
    instrument or measurement question still asks for its own measurement.
    """
    missing = [None] * len(actions)
    original = contract.get('question') or {}
    if (not original or not public_spec.get('reset') or not question
            or question.get('context_ref') != original.get('context_ref')
            or question.get('answer_type') == 'distribution'
            or question.get('kind') in {'instrument', 'measurement'}):
        return missing
    names = [item['name'] if isinstance(item, dict) else item
             for item in public_spec.get('controls', [])]

    def inputs(action):
        if (action.get('kind') != 'trajectory' or action.get('channels', []) != []
                or set(action) - {'kind', 'channels', 'sequence'}):
            return None
        sequence = action.get('sequence')
        if not isinstance(sequence, list) or not sequence or any(
                not isinstance(point, dict) or set(point) != set(names) for point in sequence):
            return None
        if prediction_rows is not None:
            try:
                rows = prediction_rows(action)
            except (ValueError, KeyError, TypeError):
                return None
            return [{'features': row.features, 'history': list(row.history)} for row in rows]
        return [{'features': point, 'history': sequence[:index]}
                for index, point in enumerate(sequence)]

    selected = []
    for action in actions:
        expected = inputs(action)
        matches = []
        if expected and expected[0]['history'] == []:
            for index, record in enumerate(observations):
                rows = record.get('rows', [])
                if (record.get('action', {}).get('kind') == 'trajectory'
                        and record['action'].get('channels', []) == []
                        and record.get('experiment_context_ref', original.get('context_ref'))
                            == question.get('context_ref')
                        and rows and all('target' in row for row in rows)
                        and [{'features': row['features'], 'history': list(row.get('history', []))}
                             for row in rows] == expected):
                    matches.append(index)
        # Conflicting answers need research; do not turn one into a cached fact.
        selected.append(matches[0] if matches and all(
            observations[index]['rows'] == observations[matches[0]]['rows'] for index in matches) else None)
    return selected


def _boundary_callbacks_completed(proposal, feedback, record_indices):
    """An acquired point does not complete a later native boundary analysis."""
    arguments = proposal.get('arguments', {})
    owner = proposal.get('research_question') or arguments.get('question') or {}
    for callback in arguments.get('after_observation', []):
        selected = callback.get('boundary_update')
        if selected is None:
            continue
        question = callback.get('research_question') or callback.get('question') or owner
        meaning = question.get('meaning_version', owner.get('meaning_version', 1))
        completed = False
        for outcome in feedback.get('after_observation_results', []):
            actual = outcome.get('arguments', {})
            actual_question = actual.get('research_question') or actual.get('question') or {}
            update = actual.get('boundary_update', {})
            result = outcome.get('native_condition_research', {}).get('boundary_update', {})
            if (outcome.get('operator') == callback.get('operator') == 'native_condition_research'
                    and not outcome.get('error')
                    and outcome.get('research_question_id', actual_question.get('id')) == question.get('id')
                    and outcome.get('question_meaning_version', actual_question.get('meaning_version', 1))
                        == meaning
                    and all(update.get(key) == selected.get(key)
                            for key in ('round_index', 'proposal_index'))
                    and ('paid_record_indices' not in update
                         or update['paid_record_indices'] == record_indices)
                    and result.get('observation_index') in record_indices
                    and result.get('status') in {'boundary_refined', 'structure_mismatch'}):
                completed = True
                break
        if not completed:
            return False
    return True


def _attempted(proposal, results, paid_rows, *, paid_record_indices=None):
    def content(value):
        return {key: item for key, item in value.items() if key != 'research_question'}
    def purpose(value):
        arguments = value.get('arguments', {})
        question = value.get('research_question') or arguments.get('question') or {}
        return {key: question[key] for key in ('kind', 'answer_type', 'context_ref',
            'answer_resolution_ref', 'query_semantics') if key in question}
    query_purpose = purpose(proposal)
    if (proposal.get('action') == 'query' and proposal.get('arguments', {}).get('actions')
            and not proposal['arguments'].get('revision')
            and query_purpose.get('answer_type') != 'distribution'
            and query_purpose.get('kind') not in {'instrument', 'measurement'}):
        return any((item.get('proposal') or {}).get('action') == 'query'
            and (item.get('proposal') or {}).get('arguments', {}).get('actions')
                == proposal['arguments']['actions']
            and purpose(item.get('proposal') or {}) == query_purpose
            and not item.get('feedback', {}).get('error')
            and item.get('feedback', {}).get('record_indices')
            and (paid_record_indices is None
                 or item['feedback']['record_indices'] == paid_record_indices)
            and _boundary_callbacks_completed(proposal, item['feedback'],
                item['feedback']['record_indices']) for item in results)
    return any(content(item.get('proposal') or {}) == content(proposal)
        and item.get('evidence_after', {}).get('rows') == paid_rows for item in results)


def _method_use_action(state, memory, question, paid_rows):
    """Consume funded method questions and their ordinary parent continuation."""
    results = list(state.get('method_results', [])) + list(memory.get('program_results', []))
    questions = {item['id']: item for item in state.get('questions', [])}
    for child in reversed(list(questions.values())):
        if child.get('kind') != 'method_use' or child.get('parent_id') != question['id']:
            continue
        basis = child.get('agenda_basis') or {'meaning_version': child.get('meaning_version', 1),
            'answer_version': child.get('semantic_answer', child.get('executable_answer', {})).get('version')}
        if child.get('completed_agenda_basis') != basis:
            continue
        execution = child.get('routing_execution') or {}
        result = execution.get('result') or {}
        proposal = result.get('next_action')
        if not proposal or result.get('status') not in {'method_use_decided', 'method_use_unresolved'}:
            continue
        completed = next((index for index in reversed(range(len(results)))
            if results[index].get('feedback', {}).get('question_id') == child['id']
            and results[index]['feedback'].get('condition_learning') == result.get('condition_learning')), None)
        subsequent = results[completed + 1:] if completed is not None else results
        if not _attempted(proposal, subsequent, paid_rows):
            return _action(proposal, question)
    agenda = state.get('research_agenda') or {}
    selected = agenda.get('selected_question_id')
    entry = next((item for queue in ('task_agenda', 'theory_agenda')
        for item in agenda.get(queue, []) if item.get('question_id') == selected
        and item.get('status') == 'ready' and item.get('work_kind') == 'method_conditions'), None)
    child = questions.get(selected, {})
    if entry is None or child.get('parent_id') != question['id']:
        return None
    basis = child.get('agenda_basis') or {'meaning_version': child.get('meaning_version', 1),
        'answer_version': child.get('semantic_answer', child.get('executable_answer', {})).get('version')}
    if child.get('completed_agenda_basis') == basis:
        return None
    mission = child.get('mission') or state.get('research_mission') or {}
    if mission.get('purpose') not in {'method_learning', 'theory_learning'}:
        return None
    funding = mission.get('allocation', mission.get('theory_learning_allocation', 0))
    funding = funding.get('compute', 0) if isinstance(funding, dict) else funding
    available = min(max(0, funding - mission.get('resource_usage', {}).get('compute', 0)),
        max(0, state.get('limits', {}).get('compute_budget', 0)
            - state.get('costs', {}).get('compute_cost', 0)))
    if available <= 0:
        return None
    proposal = {'action': 'run', 'arguments': {'operator': 'research_route',
        'research_question': {key: deepcopy(child[key]) for key in
            ('id', 'meaning_version', 'kind', 'agenda_basis', 'method_use_plan') if key in child},
        'allocation': {'compute': available}}}
    return _action(proposal, question) if not _attempted(proposal, results, paid_rows) else None


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


def _native_work_action(state, results, question, latest, paid_rows, observed_records):
    """Consume a priced pending visit without turning it into another short run."""
    owner_index = next((index for index, item in enumerate(state.get('questions', []))
                        if item['id'] == question['id']), None)
    if owner_index is None or latest is None:
        return None
    owner = state['questions'][owner_index]
    meaning = owner.get('meaning_version', 1)
    new_evidence = latest.get('observed_record_count', observed_records) < observed_records
    if not (new_evidence or latest.get('status') == 'search_incomplete'):
        return None
    path = Path('method_decision.json')
    if not path.is_file():
        return None
    for option in json.loads(path.read_text()).get('research_options', []):
        if option.get('id') != 'native_condition_research':
            continue
        for entry in reversed(option.get('current_executions', [])):
            source = entry.get('source', {})
            continuation = entry.get('compute_continuation', {})
            if (entry.get('question_id') != question['id']
                    or entry.get('round_index') != latest.get('round_index')
                    or entry.get('fit_scope') != 'all_paid_rows'
                    or entry.get('current_paid_rows') != paid_rows
                    or (continuation.get('research_unit', {}).get('quoted_paid_rows') != paid_rows if continuation
                        else entry.get('paid_rows') != paid_rows
                             or entry.get('observed_record_count') != observed_records)
                    or source.get('file') != 'research_state.json'
                    or source.get('json_path') != ['questions', owner_index,
                        'native_condition_research', latest.get('round_index')]):
                continue
            if continuation and continuation.get('status') != 'available':
                return _author_work('complete_native_research_unit_unfunded',
                    ['funded complete research unit within the original compute and execution limits'],
                    question_ref={'question_id': question['id'], 'meaning_version': meaning},
                    compute_continuation=deepcopy(continuation))
            proposal = deepcopy(entry.get('resume_action'))
            if not proposal:
                return None
            arguments = proposal.get('arguments', {})
            if (proposal.get('action') != 'run'
                    or arguments.get('operator') != 'native_condition_research'
                    or (arguments.get('research_question') or arguments.get('question') or {}).get('id')
                        != question['id']):
                return None
            if not _attempted(proposal, results, paid_rows):
                return proposal
            if continuation:
                return _author_work('complete_native_research_unit_already_attempted',
                    ['new evidence, a changed executable research proposal, or a resolved incomplete return'],
                    question_ref={'question_id': question['id'], 'meaning_version': meaning},
                    compute_continuation=deepcopy(continuation))
            return None
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


def representation_research_warrant(question, parents, public_features, *, questions=(), resolve=None):
    """Recognize a bounded research reason, never evidence of world validity.

    This pure reference consumer is also present in the standalone author. Its
    inputs are existing paid results, question meanings and executable methods;
    it supplies no representation formula or preferred history length.
    """
    basis = question.get('representation_research_basis') or {}
    if not basis:
        return None
    resolve = resolve or _json_reference_value
    atoms = {'feature', 'lag', 'available', 'accumulate', 'event_delta', 'window_sum'}

    def inputs(program, names):
        if not isinstance(program, dict) or not program.get('op'):
            return None
        found = []
        def visit(node):
            if node.get('op') in {'feature', 'lag'} and node.get('name') not in names:
                raise ValueError('program_requires_nonpublic_input')
            for child in node.get('children', []):
                visit(child)
            for definition in node.get('definitions', {}).values():
                visit(definition['body'])
            if node.get('op') in atoms:
                found.append(node)
        try:
            visit(program)
        except (ValueError, KeyError, TypeError):
            return None
        return found

    def source(reference):
        try:
            return resolve(reference)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None

    def question_source(reference):
        return next((item for item in questions
            if item['id'] == reference.get('id', reference.get('question_id'))
            and item.get('meaning_version', 1) == reference.get('version',
                reference.get('meaning_version', 1))), None)

    alternative = basis.get('alternative_ref')
    if alternative:
        compared = source(alternative) or {}
        program = compared.get('program')
        if program is None and compared.get('program_ref'):
            program = source(compared['program_ref'])
        before, after = (compared.get('best_retained_validation_predictions'),
                         compared.get('validation_predictions'))
        if (inputs(program, public_features) is not None
                and compared.get('best_retained_id') in parents and isinstance(before, list)
                and isinstance(after, list) and before and len(before) == len(after)
                and before != after):
            return 'retained_alternative_has_different_task_predictions'

    semantic = basis.get('semantic_question_ref')
    if semantic:
        unresolved = question_source(semantic)
        if (unresolved and unresolved.get('kind') in {'representation', 'instrument', 'measurement', 'binding'}
                and (unresolved.get('semantic_answer') or {}).get('status') != 'answered'
                and unresolved.get('status', 'open') not in {'completed', 'closed'}
                and unresolved.get('answer_resolution_ref')
                and any(use in question.get('use_sites', []) for use in unresolved.get('use_sites', []))):
            return 'unresolved_bound_representation_semantics'

    method_ref = basis.get('learned_method_ref')
    if method_ref:
        method = source(method_ref) or {}
        roles = method.get('role_signature') or {}
        target = question.get('theory_target') or {}
        target_roles = target.get('role_signature') or {}
        remaining, mapping = dict(target_roles), {}
        for name, role in roles.items():
            bound = next((key for key, value in remaining.items() if value == role), None)
            if bound is None:
                break
            mapping[name] = bound
            del remaining[bound]
        else:
            sources = [source(reference) or {} for reference in method.get('research_program_source_refs', [])]
            procedures = [item.get('research_program') or {} for item in sources]
            contract = method.get('research_program_contract') or {}
            def bind(node):
                return {**node, **({'name': mapping.get(node['name'], node['name'])}
                    if node.get('op') in {'feature', 'lag'} else {}),
                    'children': [bind(child) for child in node.get('children', [])]}
            terms = [inputs(bind(term), public_features) for term in contract.get('terms', [])]
            retained = [inputs(program, public_features) for program in parents.values()]
            changed_terms = (terms and all(value is not None for value in (*terms, *retained))
                and any(atom not in [old for group in retained for old in group]
                        for group in terms for atom in group))
            # Refit-only methods need genuinely different selected coordinates.
            # The existing conditional-coordinate method itself declares a finite
            # representation search; it is not a claimed missing world variable.
            proposes_representation = (changed_terms or contract.get('operation') ==
                'discover_conditional_coordinates' and all(inputs(bind(term), public_features) is not None
                    for term in [contract.get('gate_coordinate', {}), *contract.get('branch_basis', [])]))
            if (roles and not remaining and inputs(method.get('template', {}).get('pattern'), roles) is not None
                    and method.get('kind') == 'research_program' and method.get('producer')
                    and proposes_representation and procedures
                    and method.get('output_unit') == target.get('output_unit')
                    and all(procedure.get('graph', {}).get('nodes', {}).get(
                        procedure.get('output_node'), {}).get('answer') for procedure in procedures)):
                return 'retained_learned_method_proposes_representation_research'

    # A repeated check of two claims on the same rows is one witnessed failure,
    # not two prospective failures. Read the existing measured-question answers.
    witnesses = []
    for reference in basis.get('failure_refs', []):
        failed = question_source(reference)
        answer = (failed or {}).get('semantic_answer') or {}
        evidence = answer.get('evidence')
        if (answer.get('finite_status') == 'observed_false' and answer.get('value') is False
                and evidence and evidence not in witnesses
                and answer.get('version') == reference.get('answer_version', answer.get('version'))):
            witnesses.append(evidence)
    proposed = inputs(basis.get('distinction_program'), public_features)
    retained = [inputs(program, public_features) for program in parents.values()]
    if (len(witnesses) >= 2 and proposed is not None and all(value is not None for value in retained)
            and any(atom not in [old for group in retained for old in group] for atom in proposed)):
        return 'prospective_failures_with_public_unrepresented_distinction'
    # Missing evidence is unresolved author work, never scientific refutation.
    return None


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
    current = next((item for item in state.get('questions', []) if item['id'] == question['id']), {})
    question = {**question, **{key: current[key] for key in ('use_sites',) if key in current}}
    basis = (current.get('representation_research_basis') or question.get('representation_research_basis')
        or state.get('research_candidates', {}).get(model, {}).get('representation_research_basis'))
    collision = (diagnostic.get('empirical_mse_floor', 0) > threshold
                 and bool(diagnostic.get('conflicting_groups')))
    if evaluation.get('returncode') != 0 or not collision and not basis:
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
            and ((entry.get('proposal') or {}).get('arguments', {}).get('research_question', {}).get(
                'representation_research_basis') or None) == (basis or None)
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
    if not collision:
        reason = representation_research_warrant({**question,
            'representation_research_basis': basis}, {parent['id']: parent['program'],
                **{identifier: item['program'] for identifier, item in state.get('theory_index', {}).items()
                   if item.get('program') and identifier in plan['parent_theory_ids']}},
            list(question['theory_target']['role_signature']), questions=state.get('questions', []))
        if reason is None:
            return _author_work('representation_research_warrant_unresolved',
                ['actual competing predictions, bound unresolved semantics, or executable learned-method reference'],
                source_model=model, representation_research_basis=deepcopy(basis))
    child = {'id': f'representation:{model}',
        'question': ('Which public input distinctions were lost by the actual retained predictor?' if collision
            else 'Which executable public representation best serves the declared parent use?'),
        'kind': 'representation', 'origin': ('computed_representation_conflict' if collision else reason),
        'parent_id': question['id'], 'depends_on': [], 'answer_type': 'program',
        'subject_refs': [{'type': 'candidate', 'path': model}],
        'context_ref': deepcopy(question.get('context_ref')),
        'answer_resolution_ref': {'evaluator': 'representation_diagnostic_and_paid_comparison',
                                  'parent_model': model},
        'query_semantics': {'mode': 'solution_relation'},
        'representation_diagnostic': deepcopy(diagnostic),
        **({'representation_research_basis': deepcopy(basis)} if basis else {}),
        **({'theory_target': deepcopy(question['theory_target'])} if basis else {}),
        'use_sites': (deepcopy(question['use_sites']) if basis and question.get('use_sites') else
                      [{'kind': 'supply_component', 'model': model, 'slot': 'representation'}])}
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


def publish_research_options(agent, public_spec=None):
    """Price actual ordinary proposals, without fitting or acquiring evidence.

    Native continuation keeps its existing quote. R uses the proposal's own
    bounded allocation, not another method's historical cost. Q is exposed only
    when its existing callback supplies a bounded complete analysis contract.
    Cold comparison ceilings and the original finish are explicit obligations.
    """
    from .structural_inquiry import native_condition_method
    from .python_research_routing import validated_candidates

    native = native_condition_method(agent)
    published = [native] if native is not None else []
    contract = agent.knowledge.get('finite_native_author') or {}
    question = next((item for item in agent.questions
        if item.get('id') == agent._active_question_id), None)
    if question is not None and question.get('kind') in {'method_use', 'method_learning'}:
        parent = (question.get('parent_question_ref') or {}).get('question_id', question.get('parent_id'))
        if parent == (contract.get('question') or {}).get('id'):
            question = next((item for item in agent.questions if item.get('id') == parent), question)
    if question is None or not contract:
        return published
    rows = len(agent.rows)
    rounds = question.get('native_condition_research', [])
    declared_models = _comparison_models(contract.get('predictor_paths', []), agent.candidate_paths, rounds)
    models = list(dict.fromkeys([*declared_models, *validated_candidates(agent)]))
    results = [entry for entry in agent.method_results
        if entry.get('feedback', {}).get('research_question_id') == question['id']
        and entry['feedback'].get('question_meaning_version', 1) == question.get('meaning_version', 1)]
    current = _completed_choice(results, declared_models, rows, question, {'questions': agent.questions,
        'operator_results': agent.operator_results})
    preparation = 0 if current is not None else rows * len(models)
    remaining = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
    seconds = max(0., agent.config.execution_seconds - agent.costs['python_execution_seconds'])
    deliveries = len(agent._inquiry_stack)

    def quote(core, extra_rows, exports, completion=None):
        comparison_rows = rows + extra_rows
        comparison = comparison_rows * (len(models) + exports)
        finish = comparison_rows * (deliveries + 1) + agent.config.confirmation_rows
        if completion is not None:
            comparison = completion['comparison_compute_cost']
            finish = completion['finish_compute_cost']
        total = preparation + core + comparison + finish
        return {'status': ('available' if total <= remaining and seconds > 0
                           and extra_rows <= agent.config.observation_budget - agent.costs['observation_cost']
                           else 'execution_time_exhausted' if seconds <= 0
                           else 'complete_unit_unfunded'),
            'compute_cost': total, 'observation_cost': extra_rows, 'measurement_cost': 0,
            'paid_rows': rows, 'retained_pool_models': len(models), 'max_exported_models': exports,
            'remaining_execution_seconds': seconds, 'preparation_compute': preparation,
            'bounded_research_compute': core, 'comparison_compute': comparison,
            'finish_compute': finish, 'comparison_paid_rows': comparison_rows,
            'estimate_is_completion_guarantee': False,
            'scope': 'bounded_attempt_and_complete_pool_parent_closure_cold_comparison_ceiling'}

    for index, child in enumerate(agent.questions):
        result = (child.get('routing_execution') or {}).get('result') or {}
        proposal = result.get('next_action') or {}
        arguments = proposal.get('arguments') or {}
        candidate = agent.theory_consolidation.get('candidates', {}).get(arguments.get('candidate_id'), {})
        allocation = arguments.get('allocation') or {}
        fitting = arguments.get('research_program_fitting') or {}
        mapping, roles = arguments.get('mapping', {}), arguments.get('role_signature', {})
        if (child.get('parent_id') != question['id'] or proposal.get('action') != 'run'
                or arguments.get('operator') != 'theory_transfer'
                or candidate.get('kind') != 'research_program'
                or type(allocation.get('compute')) is not int or allocation['compute'] <= 0
                or not fitting.get('validation_records')
                or any(type(value) is not int or not 0 <= value < len(agent.observations)
                       for value in fitting['validation_records'])
                or len(set(fitting['validation_records'])) == len(agent.observations)
                or set(mapping) != set(candidate.get('role_signature', {}))
                or len(set(mapping.values())) != len(mapping)
                or any(roles.get(name) != candidate['role_signature'][role]
                       for role, name in mapping.items())
                or arguments.get('output_unit') != candidate.get('output_unit')
                or arguments.get('context_family_ref', arguments.get('context_ref'))
                    != candidate.get('scope', {}).get('context_ref')):
            continue
        binding = {'research_program_contract': deepcopy(candidate['research_program_contract']),
            'mapping': deepcopy(mapping), 'role_signature': deepcopy(roles),
            'output_unit': arguments['output_unit'], 'context_ref': deepcopy(arguments.get('context_ref')),
            'research_program_fitting': deepcopy(fitting), 'seed_policy': candidate.get('seed_policy'),
            'source_seed_count': len(candidate.get('source_seed_inputs', []))}
        complete = any(entry.get('result', {}).get('search_completed') is True
            and entry.get('basis', {}).get('binding', {}).get('paid_cutoff') == {
                'records': len(agent.observations), 'rows': rows}
            and all(entry.get('basis', {}).get('binding', {}).get(key) == value for key, value in (
                ('mapping', mapping), ('source_refs', candidate.get('research_program_source_refs')),
                ('research_program_contract', candidate['research_program_contract'])))
            and entry.get('basis', {}).get('fitting') == fitting
            for entry in agent.theory_consolidation.get('executions', {}).values())
        exports = len(candidate.get('research_program_source_refs', []))
        if complete or not exports:
            continue
        published.append({'family': 'R', 'proposal': deepcopy(proposal),
            'question_ref': {'id': question['id'], 'meaning_version': question.get('meaning_version', 1)},
            'proposal_ref': {'file': 'research_state.json', 'json_path': [
                'questions', index, 'routing_execution', 'result', 'next_action']},
            'program_binding': binding, 'complete_quote': quote(allocation['compute'], 0, exports)})
    if not public_spec:
        return published
    for round_ in rounds[-1:]:
        for index, experiment in enumerate(round_.get('proposed_experiments', [])):
            proposal = deepcopy(experiment.get('query_action') or {})
            callbacks = proposal.get('arguments', {}).get('after_observation', [])
            deferred = experiment.get('research_continuation') or {}
            future_offer = None
            if not callbacks and deferred.get('operator') == 'native_condition_research':
                if not public_spec.get('reset'):
                    continue
                from types import SimpleNamespace
                from .structural_inquiry import _native_compute_continuation, _native_completion_reserve

                actions = proposal.get('arguments', {}).get('actions', [])
                records = paid_experiment_record_indices(actions, agent.observations, public_spec,
                    question, contract)
                # Only shape/control history enters this future-data quote.
                # There is deliberately no target, simulated output or fit.
                planned = [SimpleNamespace(features=point, history=tuple(action['sequence'][:position]))
                    for action, record in zip(actions, records) if record is None
                    for position, point in enumerate(action.get('sequence', []))]
                retained_count = max(len(round_.get('candidates', [])), len(round_.get('exported_models', [])))
                if (not planned or 'search_stages' not in round_
                        or type(deferred.get('max_candidates')) is not int):
                    continue
                projected = vars(agent).copy()
                projected['rows'] = [*agent.rows, *planned]
                future = SimpleNamespace(**projected)
                exports = retained_count + len(deferred.get('programs', [])) + deferred['max_candidates']
                reserve = _native_completion_reserve(future, question, exports)
                if 'max_exported_models' not in reserve:
                    continue
                future_offer = _native_compute_continuation(future, question, round_, deferred, reserve)
                callback = deepcopy(deferred)
                callback['max_compute'] = future_offer['research_unit']['search_compute']
                proposal.setdefault('arguments', {})['after_observation'] = [callback]
                callbacks = [callback]
            if not callbacks or any(callback.get('operator') != 'native_condition_research'
                    or 'boundary_update' not in callback and future_offer is None
                    or type(callback.get('max_compute')) is not int
                    or callback['max_compute'] < 0 for callback in callbacks):
                continue
            actions = proposal.get('arguments', {}).get('actions', [])
            records = paid_experiment_record_indices(actions, agent.observations, public_spec,
                question, contract)
            extra = sum(len(action.get('sequence', [])) for action, record in zip(actions, records)
                        if record is None)
            retained = []
            for previous in rounds:
                for candidate in previous.get('candidates', []):
                    if candidate.get('program') not in retained:
                        retained.append(candidate.get('program'))
            exports = len(retained) + sum(callback.get('max_candidates', 8) for callback in callbacks)
            completion = None
            if future_offer is not None:
                completion = future_offer['research_unit']['completion_reserve']
                exports = completion['max_exported_models']
            published.append({**deepcopy(experiment), 'family': 'Q', 'proposal': deepcopy(proposal),
                'question_ref': {'id': question['id'], 'meaning_version': question.get('meaning_version', 1)},
                'proposal_ref': {'file': 'research_state.json', 'json_path': ['questions',
                    agent.questions.index(question), 'native_condition_research', round_['round_index'],
                    'proposed_experiments', index, 'query_action']},
                'complete_quote': quote(sum(callback['max_compute'] for callback in callbacks),
                    extra, exports, completion=completion),
                **({'unfinished_pair_ref': {'file': 'research_state.json', 'json_path': ['questions',
                    agent.questions.index(question), 'native_condition_research', round_['round_index']]}}
                   if round_.get('status') == 'needs_matched_evidence'
                   and experiment.get('matched_action_pairs') else {}),
                **({'future_compute_continuation': future_offer,
                    'common_closure_quote_repair': True} if future_offer is not None else {})})
    return published


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
    meaning = next((item.get('meaning_version', 1) for item in state.get('questions', [])
                    if item['id'] == question['id']), question.get('meaning_version', 1))
    results = [entry for entry in state.get('method_results', [])
        if entry.get('feedback', {}).get('research_question_id') == question['id']
        and entry['feedback'].get('question_meaning_version', 1) == meaning]
    results += memory.get('program_results', [])
    # PR2's frozen query need not carry the native method's after_observation.
    # Consume its new paid records with the original retained-bank search before
    # comparing old exports or using experiments proposed on the old prefix.
    evidence_extended = False
    if latest and latest.get('observed_record_count', len(observations)) < len(observations):
        first = latest['observed_record_count']
        added = list(range(first, len(observations)))
        covered_records = latest['fit_record_indices']
        covered_rows = set(latest['fit_evidence'])
        if any(index not in covered_records or
               any(row not in covered_rows for row in observations[index]['row_indices'])
               for index in added):
            evidence_extended = True
    models = _comparison_models(old, state.get('exported_candidate_paths', []), native)
    missing = [path for path in models if not Path(path).is_file()]
    if missing:
        return _author_work('retained_predictor_source_missing', missing,
                            catalog_ref=deepcopy(contract['predictor_catalog_ref']))
    # A source edit resets program_results, while its actual scientific
    # executions retain their original source, question, meaning and paid scope.
    # The existing choice scores every named old/native model and adds all
    # validated refit/response exports itself. Its retention/confirmation gate
    # decides among them before another purchase or refinement can intervene.
    # This procedure never selects the native minimum.
    choice = _completed_choice(results, models, paid_rows, question, state)
    current_choice = choice is not None
    if models and not current_choice:
        return _action({'action': 'run', 'arguments': {'operator': 'research_choice', 'models': models}}, question)
    method_use = _method_use_action(state, memory, question, paid_rows)
    if method_use is not None:
        return method_use
    if current_choice:
        representation = _representation_action(contract, question, choice, state, memory, paid_rows)
        if representation is not None:
            return representation
    if current_choice and choice.get('next_action') and not _attempted(choice['next_action'], results, paid_rows):
        return _action(choice['next_action'], question)
    if current_choice:
        resumed = _native_work_action(state, results, question, latest, paid_rows, len(observations))
        if resumed is not None:
            return resumed if 'stop' in resumed else _action(resumed, question)
    if evidence_extended:
        resumed = _native_work_action(state, results, question, latest, paid_rows, len(observations))
        if resumed is not None:
            return resumed if 'stop' in resumed else _action(resumed, question)
        return _author_work('complete_native_research_unit_quote_required',
            ['current paid-scope complete research quote retaining the original evidence selection'],
            question_ref={'question_id': question['id'], 'meaning_version': meaning},
            native_round=latest.get('round_index'), paid_rows=paid_rows)
    if parent.get('pending_query'):
        return _action({'action': 'query', 'arguments': {'pending_question': parent_id}}, question)
    paid_answers = []
    def unanswered(proposal):
        arguments = proposal.get('arguments', {})
        if proposal.get('action') == 'query' and arguments.get('actions') and not arguments.get('revision'):
            records = paid_experiment_record_indices(arguments['actions'], observations,
                public_spec, question, contract)
            if all(index is not None for index in records):
                answer = {'query_action': deepcopy(proposal), 'record_indices': records,
                    'observation_refs': [{'file': 'laboratory_observations.json', 'json_path': [index]}
                                         for index in records]}
                if answer not in paid_answers:
                    paid_answers.append(answer)
                # The same physical answer can support a new boundary step.
                # Query reuse will bind these original indices and execute its
                # existing continuation without purchasing another trajectory.
                if any('boundary_update' in step for step in arguments.get('after_observation', [])):
                    return not _attempted(_action(proposal, question), results, paid_rows,
                                          paid_record_indices=records)
                return False
        return not _attempted(proposal, results, paid_rows)
    # Retain the old method's actual boundary/matched-evidence continuation.
    if latest:
        interval = latest.get('remaining_boundary', {}).get('threshold_interval', [])
        refine = len(interval) == 2 and interval[1] - interval[0] > contract['boundary_resolution']
        for experiment in latest.get('proposed_experiments', []):
            proposal = experiment.get('query_action')
            if (proposal and (refine and experiment.get('purpose') == 'conditional_boundary'
                              or latest.get('status') == 'needs_matched_evidence')
                    and unanswered(proposal)):
                return _current_query_action(experiment, public_spec, observations, memory, state, question)
    if latest is None:
        return {'action': 'run', 'arguments': {'operator': 'native_condition_research',
            'research_question': deepcopy(question), 'max_compute': contract['max_compute'],
            'max_candidates': contract['max_candidates']}}
    for experiment in latest.get('proposed_experiments', []):
        proposal = experiment.get('query_action')
        if proposal and unanswered(proposal):
            return _current_query_action(experiment, public_spec, observations, memory, state, question)
    if paid_answers:
        return _author_work('paid_experiments_already_answered',
            ['resume unfinished computation or propose an experiment with different public inputs'],
            question_ref={'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)},
            paid_answers=paid_answers)
    return _author_work('executable_revision_or_scope_required',
        ['revised executable candidate, fitting scope, or finite research policy'],
        question_ref={'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)},
        native_status=latest.get('status'), catalog_ref=deepcopy(contract['predictor_catalog_ref']))

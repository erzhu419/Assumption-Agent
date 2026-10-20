"""Bind authored, numerically checked child answers to ordinary Python fits.

The author chooses parameter regions or previously measured local relations.
Computed checks retain their cited scope, not uniqueness or global truth.
No apparatus equation or preferred representation is supplied here.
"""

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path


PAID_RELATION_FIELD_MEANINGS = {
    'evidence_status': 'Historical authored hypothesis checked against measured data; '
        'not the current predictor accuracy. historical_check records that original assertion.',
    'residual': 'Current candidate predicted_value minus measured_value for the same contrast. '
        'A historically satisfied check can have a large prediction error, and a historically '
        'violated check can have zero prediction error.',
}


EVIDENCE_INTERFACE = """
Executable evidence uses the existing question/revision records, in both arms.
run {operator:'analyze_evidence',path:'local_readout.py',
 research_question:{id:'local',question:'your local question',parent_id:'task'}}
executes readout(result,public_spec,memory) on already paid records, without a
new purchase or invented forecast. Use the revision readout checks to compute
relations from actual cited labels. Numerical checks and your prose remain
separate; notes cannot overwrite a calculated counterexample.

A checked answer may also return parameter_region:{lower:...,upper:...,
lower_closed:false,upper_closed:false}, scope:{...}. This is an authored working
region under the stated conditions, not proof of a unique global parameter.
run {operator:'bind_research',model:'parent.py',inputs:{chosen_slot:'local'}}
binds the current child answer to that parent. An answer needs satisfied numeric
checks before it can become a bound region. Explicit inputs:{} removes bindings.
The Python parent defines bind_research(inputs) and get_parameters()->{slot:value}.
Each input contains question_id,version,region,scope,evidence. bind_research runs
before fitting; choose actual parameters from that region in your own program.
get_parameters exposes the values used, which must lie in the bound regions.
For measured local relations, the same inputs accepts
{chosen_slot:{revision_index:12,check_ids:['your_check']}}. Omitting check_ids
selects all numerical response checks from that revision's latest readout.
The bound kind:'local_relations' contains checks with their actual measured value,
scope and public input/history witnesses, including checks that refuted an old
expectation. bind_research(inputs) receives them before fit; local-only inputs do
not require get_parameters. Candidate predictions are compared with those measured
values after fitting. This is a local diagnostic, not a general law or an automatic
choice of model. The author chooses how to use the evidence in the model or fit.
research_bindings[].local_relations reports measured_value, predicted_value and
residual for each selected check. A refuted expectation remains refuted; binding
its measured contrast does not assert that old expected relation.
Selecting a revision retains its checks when later questions or notes change.
Reanalyzing that same revision invalidates its declared consuming model.
The runtime records this interface use; arbitrary Python semantics remain authored.
No correct equation, parameter or feature is filled in for you.
Updated or refuted child answers mark only their dependent parents for rebuilding.
Hegel materializes current checked inputs before evaluating the parent again;
Code can explicitly repeat bind_research to do the same. Old frozen candidates
remain callable as old versions, with their own inputs and prior evidence.
research_repairs names actual pending input/representation problems and existing
actions; it does not select a scientific formula. A representation conflict
requires changing the mapping or challenging its assumptions, not only fitting
another deterministic regressor on exactly the same mapped inputs.
"""


def _question(agent, question_id):
    return next(q for q in agent.questions if q['id'] == question_id)


def bind_components(agent, model, bindings, *, question_refs=None):
    """Track the retained components the author actually chose for this parent."""
    candidate = agent.research_candidates.setdefault(model, {})
    previous = candidate.get('components', {})
    components = {}
    for slot, path in bindings.items():
        for kind, pool in (('conditional', agent.conditional_models), ('observable', agent.observable_models)):
            for key, entry in pool.items():
                if entry['model'] == path:
                    components[slot] = {'kind': kind, 'id': key, 'model': path,
                                        'version': entry['version'], 'support': entry['support']}
                    reference = (question_refs or {}).get(slot)
                    old = previous.get(slot, {})
                    if reference is None and all(old.get(field) == components[slot][field]
                                                  for field in ('kind', 'id', 'model', 'version')):
                        reference = old.get('question_ref')
                    if reference is not None:
                        components[slot]['question_ref'] = deepcopy(reference)
    if candidate.get('components') != components:
        candidate.update(components=components, status='needs_refit')
        candidate.pop('current_evaluation', None)


def _current_components(agent, candidate):
    result = {}
    for slot, bound in candidate.get('components', {}).items():
        pool = agent.conditional_models if bound['kind'] == 'conditional' else agent.observable_models
        entry = pool[bound['id']]
        result[slot] = {**bound, **{key: entry[key] for key in ('model', 'version', 'support')}}
    return result


def _component_question_supported(agent, component):
    reference = component.get('question_ref')
    if reference is None:
        return True
    question = _question(agent, reference['question_id'])
    answer = question.get('semantic_answer', {})
    value = answer.get('value', {})
    from .question_continuation import _versioned_support
    return (question.get('meaning_version', 1) == reference['meaning_version']
        and answer.get('version') == reference['answer_version'] and answer.get('status') == 'answered'
        and isinstance(value, dict) and value.get('kind') == 'retained_component'
        and value.get('component_kind') == component['kind']
        and value.get('component_id') == component['id']
        and value.get('component_version') == component['version']
        and value.get('model') == component['model'] and _versioned_support(agent, question)[0])


def refresh_dependencies(agent):
    _restore_evaluation_ranges(agent)
    for candidate in agent.research_candidates.values():
        for bound in candidate.get('inputs', {}).values():
            if bound.get('question_id') and bound.get('kind') == 'parameter_region':
                question = _question(agent, bound['question_id'])
                if question.get('support_paths') is not None:
                    from .question_continuation import _versioned_support
                    if not _versioned_support(agent, question)[0]:
                        candidate.update(status='needs_rebuild', changed_question=question['id'])
                        candidate.pop('current_evaluation', None)
            if (bound.get('kind') == 'local_relations'
                    and len(agent.revisions[bound['revision_index']].get('analysis_attempts', []))
                    != bound['version']):
                candidate.update(status='needs_rebuild', changed_revision=bound['revision_index'])
                candidate.pop('current_evaluation', None)
        current = _current_components(agent, candidate)
        for component in current.values():
            if not _component_question_supported(agent, component):
                candidate.update(status='needs_rebuild', changed_question=component['question_ref']['question_id'])
                candidate.pop('current_evaluation', None)
        if current != candidate.get('components', {}):
            changed = any(entry['model'] != candidate['components'][slot]['model']
                or entry['version'] != candidate['components'][slot]['version']
                or entry['support'] not in {'paid_fit', 'paid_evaluation', 'supported_on_later_readings'}
                for slot, entry in current.items())
            if changed:
                candidate.update(status='needs_rebuild', changed_components=current)
                candidate.pop('current_evaluation', None)
            else:
                candidate['components'] = current
        if (candidate.get('status') == 'evaluated'
                and candidate.get('paid_rows', len(agent.rows)) < len(agent.rows)):
            candidate['status'] = 'new_paid_rows_unscored'


def _restore_evaluation_ranges(agent):
    """Recover old finish ranges from the actual latest evaluation, not its new labels."""
    pending = {model: candidate for model, candidate in agent.research_candidates.items()
               if candidate.get('current_evaluation')
               and 'paid_rows' not in candidate['current_evaluation']}
    if not pending:
        return
    for event in reversed(agent.events):
        action, arguments = event.get('action'), event.get('arguments', {})
        if action not in {'evaluate', 'finish'} and not (
                action == 'run' and arguments.get('operator') == 'refit'):
            continue
        model = arguments.get('model', 'model.py')
        if model not in pending:
            continue
        feedback = event.get('feedback', {})
        if action == 'run':
            feedback = feedback.get('validation', feedback)
        elif action == 'finish':
            feedback = feedback.get('evaluation', feedback)
        if 'returncode' not in feedback:
            continue
        candidate = pending.pop(model)
        execution = candidate['current_evaluation']
        submitted, fresh = feedback.get('submitted_candidate', {}), feedback.get('fresh_confirmation', {})
        # A later ordinary evaluation wins, even if its scalar error happens to
        # equal the old pre-confirmation error. Only this recorded finish fixes
        # a range known to have included the newly appended confirmation rows.
        if (action == 'finish' and submitted.get('source_model') == model and fresh
                and all(execution.get(key) == feedback.get(key) for key in ('returncode', 'mse'))
                and candidate.get('paid_rows') == submitted['paid_rows'] + fresh['rows']):
            candidate['paid_rows'] = submitted['paid_rows']
            execution['paid_rows'] = submitted['paid_rows']
        if not pending:
            break


def _rebuild_action(agent, model, state):
    arguments = {'model': model}
    if agent.arm != 'hegel':
        if model in agent.research_input_bindings:
            arguments['research_inputs'] = deepcopy(agent.research_input_bindings[model])
        if state.get('components'):
            existing = json.loads((agent.workspace / 'model_bindings.json').read_text()).get(model, {})
            arguments['bindings'] = {**existing, **{slot: entry['model'] for slot, entry
                in _current_components(agent, state).items()}}
    return {'action': 'evaluate', 'arguments': arguments}


def record_answer(agent, question, answer):
    """Keep versions and invalidate declared consumers without changing old fits."""
    prior = question.get('executable_answer')
    region = answer.get('parameter_region')
    if region is None and prior is None:
        return
    current = {key: deepcopy(answer[key]) for key in
               ('parameter_region', 'scope', 'evidence', 'evidence_status', 'check_results', 'status')
               if key in answer}
    if region is not None:
        lo, hi = float(region['lower']), float(region['upper'])
        if not isfinite(lo) or not isfinite(hi) or lo > hi:
            raise ValueError('parameter_region needs finite ordered bounds')
        current['parameter_region'] = {'lower': lo, 'upper': hi,
            'lower_closed': bool(region.get('lower_closed', False)),
            'upper_closed': bool(region.get('upper_closed', False))}
        if lo == hi and not (current['parameter_region']['lower_closed']
                             and current['parameter_region']['upper_closed']):
            raise ValueError('an empty parameter region cannot be bound')
        if not answer.get('scope'):
            raise ValueError('a parameter region needs its authored scope and assumptions')
    if prior and {k: v for k, v in prior.items() if k != 'version'} == current:
        return
    current['version'] = 1 if prior is None else prior['version'] + 1
    if prior is not None:
        question.setdefault('answer_history', []).append(deepcopy(prior))
    question['executable_answer'] = current
    for model, bindings in agent.research_input_bindings.items():
        if question['id'] in bindings.values():
            candidate = agent.research_candidates.setdefault(model, {})
            candidate.update(status='needs_rebuild', changed_question=question['id'])
            candidate.pop('current_evaluation', None)


def _latest_executed_checks(record):
    attempts = record.get('analysis_attempts', [])
    latest = attempts[-1] if attempts else {}
    if latest.get('execution', {}).get('returncode') != 0 or latest.get('error'):
        return []
    return latest.get('check_results', [])


def compare_paid_relations(revisions, rows, predictions, *, compute_budget=None):
    """Compare an existing paid-prefix prediction with authored empirical contrasts.

    This consumes the same predictions as the ordinary score. It neither binds
    research inputs nor executes a model, and does not validate a check's prose.
    """
    if len(rows) != len(predictions):
        raise ValueError('paid relation predictions must cover the scored rows')
    if compute_budget is not None and (type(compute_budget) is not int or compute_budget < 0):
        raise ValueError('relation compute_budget must be a nonnegative integer')
    result = {'scope': 'paid_local_relations', 'evaluated_rows': len(rows),
              'field_meanings': dict(PAID_RELATION_FIELD_MEANINGS),
              'revision_count': 0, 'check_count': 0, 'term_count': 0, 'revisions': [],
              'truncated': False}
    for index, record in enumerate(revisions):
        compared = []
        for check in _latest_executed_checks(record):
            if (check.get('status') not in {'satisfied', 'violated'}
                    or check.get('target') != 'response'
                    or not isinstance(check.get('value'), (int, float))
                    or not isfinite(check['value']) or not check.get('witnesses')):
                continue
            terms = []
            for witness in check['witnesses']:
                position = witness['row']
                if not 0 <= position < len(rows):
                    break
                row = rows[position]
                current = (row if isinstance(row, dict) else
                           {'features': row.features, 'history': row.history, 'target': row.target})
                if (current['features'] != witness['features']
                        or list(current.get('history', ())) != list(witness.get('history', ()))
                        or current['target'] != witness['target']):
                    break
                if compute_budget is not None and result['term_count'] >= compute_budget:
                    result.update(truncated=True, reason='relation_compute_budget_exhausted',
                        pending_check={'revision_index': index, 'id': check['id'],
                                       'completed_terms': len(terms),
                                       'total_terms': len(check['witnesses']), 'terms': terms})
                    break
                terms.append({'row_index': position, 'coefficient': witness['coefficient'],
                              'measured_value': witness['target'],
                              'predicted_value': float(predictions[position])})
                result['term_count'] += 1
            else:
                predicted = sum(term['coefficient'] * term['predicted_value'] for term in terms)
                compared.append({'id': check['id'], 'scope': deepcopy(check.get('scope')),
                                 'measured_value': check['value'], 'predicted_value': predicted,
                                 'residual': predicted - check['value'],
                                 'evidence_status': check['status'],
                                 'historical_check': {key: deepcopy(check[key]) for key in
                                     ('status', 'relation', 'expected', 'atol', 'rtol') if key in check},
                                 'terms': terms})
            if result['truncated']:
                break
        if compared:
            compared.sort(key=lambda item: abs(item['residual']), reverse=True)
            result['revisions'].append({
                'revision_index': index, 'version': len(record.get('analysis_attempts', [])),
                'check_count': len(compared), 'max_abs_residual': abs(compared[0]['residual']),
                'examples': compared[:3]})
            result['check_count'] += len(compared)
        if result['truncated']:
            break
    result['revision_count'] = len(result['revisions'])
    return result


def _local_relations(agent, selection):
    """Reuse the author's executed readout, including its measured counterexamples."""
    index = selection.get('revision_index')
    if type(index) is not int or not 0 <= index < len(agent.revisions):
        raise ValueError('local relations require an existing revision_index')
    record = agent.revisions[index]
    attempts = record.get('analysis_attempts', [])
    latest = attempts[-1] if attempts else {}
    checks = _latest_executed_checks(record)
    if not checks:
        raise ValueError('local relations require executed numerical response checks')
    if 'check_ids' in selection:
        ids = selection['check_ids']
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('check_ids must name distinct executed checks')
        by_id = {check.get('id'): check for check in checks}
        if any(identifier not in by_id for identifier in ids):
            raise ValueError('check_ids must name checks in the selected revision')
        checks = [by_id[identifier] for identifier in ids]
    if any(check.get('status') not in {'satisfied', 'violated'}
           or check.get('target') != 'response'
           or not isinstance(check.get('value'), (int, float))
           or not isfinite(check['value']) or not check.get('witnesses') for check in checks):
        raise ValueError('local relations require executed numerical response checks')
    return {'kind': 'local_relations', 'revision_index': index, 'version': len(attempts),
            'question_id': record.get('question_id'), 'checks': deepcopy(checks),
            'analysis_scope': deepcopy(latest.get('analysis_scope', {})),
            'evidence': sorted({witness['row'] for check in checks for witness in check['witnesses']})}


def materialize(agent, model, inputs=None):
    """Freeze declared regions or measured relations in the predictor context."""
    if '_evaluated_model' in Path(model).parts:
        raise ValueError('bind a mutable candidate; retained old candidates keep their original inputs')
    bindings = (deepcopy(inputs) if inputs is not None
                else deepcopy(agent.research_input_bindings.get(model, {})))
    payload = {}
    for slot, question_id in bindings.items():
        if isinstance(question_id, dict):
            payload[slot] = _local_relations(agent, question_id)
            continue
        question = _question(agent, question_id)
        if question.get('support_paths') is not None:
            from .question_continuation import _versioned_support
            if not _versioned_support(agent, question)[0]:
                raise ValueError(f'{question_id}: current parameter answer has unresolved premises')
        answer = question.get('executable_answer', {})
        if (answer.get('evidence_status') != 'supported' or answer.get('status') != 'supported'
                or not answer.get('parameter_region')):
            raise ValueError(f'{question_id}: current parameter answer lacks satisfied executable checks')
        payload[slot] = {'kind': 'parameter_region', 'question_id': question_id,
                        'meaning_version': question.get('meaning_version', 1),
                        'version': answer['version'], 'region': deepcopy(answer['parameter_region']),
                        'scope': deepcopy(answer['scope']), 'evidence': deepcopy(answer.get('evidence', []))}
        if 'context_ref' in question:
            payload[slot]['context_ref'] = deepcopy(question['context_ref'])
    path = agent.workspace / 'model_research_inputs.json'
    all_inputs = json.loads(path.read_text()) if path.exists() else {}
    all_inputs[model] = payload
    agent._save('model_research_inputs.json', json.dumps(all_inputs, indent=2))
    agent.research_input_bindings[model] = bindings
    candidate = agent.research_candidates.setdefault(model, {})
    if candidate.get('inputs') != payload:
        candidate.update(inputs=deepcopy(payload), status='needs_refit')
        candidate.pop('current_evaluation', None)
    return {'model': model, 'inputs': payload, 'status': candidate.get('status', 'needs_refit'),
            'next_action': {'action': 'evaluate', 'arguments': {'model': model}}}


def before_candidate(agent, model):
    """Rebuild Hegel's declared inputs; flat Code retains its explicit binding."""
    refresh_dependencies(agent)
    if model not in agent.research_candidates:
        return None
    state = agent.research_candidates.get(model, {})
    if state.get('status') == 'needs_rebuild':
        if agent.arm == 'hegel':
            current = _current_components(agent, state)
            for component in current.values():
                if not _component_question_supported(agent, component):
                    raise ValueError(f"{component['question_ref']['question_id']}: dependent component question needs rebuilding")
                if component['support'] not in {'paid_fit', 'paid_evaluation', 'supported_on_later_readings'}:
                    raise ValueError(f"{component['id']}: dependent component needs repair: {component['support']}")
            update = (materialize(agent, model) if model in agent.research_input_bindings else
                      {'model': model, 'status': 'needs_refit'})
            if current:
                path = agent.workspace / 'model_bindings.json'
                bindings = json.loads(path.read_text())
                bindings[model].update({slot: entry['model'] for slot, entry in current.items()})
                agent._save('model_bindings.json', json.dumps(bindings, indent=2))
                state.update(components=current, status='needs_refit')
                state.pop('changed_components', None)
                update['components'] = deepcopy(current)
            return update
        return {'status': 'old_dependency_version', 'model': model,
                'next_action': _rebuild_action(agent, model, state)}
    return None


def record_evaluation(agent, model, result, *, claim=None):
    """Persist actual execution and expose the next unresolved research operation."""
    if not model or '_evaluated_model' in Path(model).parts:
        return
    candidate = agent.research_candidates.setdefault(model, {})
    if claim is not None or candidate.get('claim_instance'):
        from .question_semantics import adopt_claim_instance
        previous = candidate.get('claim_instance')
        declaration = claim if claim is not None else previous
        instance = {**declaration, 'binding_ref': deepcopy(candidate.get('inputs', {}))}
        if result.get('fitting_view'):
            instance['fitted_state_ref'] = deepcopy(result['fitting_view'])
        if result.get('research_bindings'):
            instance['parameters'] = {entry.get('model', str(index)): deepcopy(entry['actual_parameters'])
                for index, entry in enumerate(result['research_bindings']) if 'actual_parameters' in entry}
        # Failed execution remains an attempt; it does not install an instance
        # or inherit the confirmation of an earlier fitted model.
        if result.get('returncode') == 0 and not result.get('truncated'):
            candidate['claim_instance'] = adopt_claim_instance(previous, instance)
    execution = {key: deepcopy(result[key]) for key in
                 ('returncode', 'mse', 'evaluation_scope', 'paid_residuals', 'paid_relation_diagnostic',
                  'research_bindings', 'representation_diagnostic',
                  'training_representation_diagnostic', 'stderr', 'truncated', 'status', 'reason',
                  'research_prediction_count')
                 if key in result}
    # finish appends fresh confirmation rows after computing its top-level paid
    # score. Those new labels were not evaluated by that score or representation.
    paid_rows = result.get('submitted_candidate', {}).get('paid_rows', len(agent.rows))
    execution['paid_rows'] = paid_rows
    candidate['current_evaluation'] = execution
    candidate['paid_rows'] = paid_rows
    if candidate.get('status') != 'needs_rebuild':
        candidate['status'] = ('evaluation_inconclusive_budget' if result.get('truncated') else
                              'evaluated' if result.get('returncode') == 0 else 'execution_failed')
    diagnostics = [result[key] for key in ('representation_diagnostic', 'training_representation_diagnostic')
                   if isinstance(result.get(key), dict)
                   and result[key].get('empirical_mse_floor') is not None]
    diagnostic = max(diagnostics, key=lambda item: item['empirical_mse_floor'], default=None)
    if result.get('returncode') != 0 or diagnostic is None or result.get('loss_kind', 'mse') != 'mse':
        return
    question_id = f'representation:{model}'
    question = next((q for q in agent.questions if q['id'] == question_id), None)
    if diagnostic['empirical_mse_floor'] > agent.config.confirmation_mse_threshold:
        candidate['status'] = 'representation_conflict'
        if agent.arm == 'hegel':
            question = agent._open_question({'id': question_id,
                'question': 'Which publicly observable distinctions were lost by this candidate representation?',
                'parent_id': 'task', 'depends_on': [], 'kind': 'representation',
                'subject_refs': [{'type': 'candidate', 'path': model}],
                'context_ref': {'input_contract': 'actual_downstream_inputs', 'loss_kind': 'mse'},
                'answer_type': 'program',
                'answer_resolution_ref': {'evaluator': 'representation_diagnostic_and_paid_comparison',
                                          'parent_model': model},
                'query_semantics': {'mode': 'solution_relation'},
                'use_sites': [{'kind': 'supply_component', 'model': model, 'slot': 'representation'}]})
            question['origin'] = 'computed_representation_conflict'
            question['status'] = 'open'
            question['representation_diagnostic'] = deepcopy(diagnostic)
            agent._active_question_id = question['id']
    elif question is not None and question.get('origin') == 'computed_representation_conflict':
        question['status'] = 'resolved_on_paid_rows'
        question['representation_diagnostic'] = deepcopy(diagnostic)
        parent = next((q for q in agent.questions if q['id'] == question.get('parent_id')), None)
        if parent is not None:
            parent['status'] = 'reopened'
            if agent._active_question_id == question['id']:
                agent._active_question_id = parent['id']
    if question is not None:
        assessments = question.setdefault('representation_assessments', [])
        if not assessments or assessments[-1] != diagnostic:
            assessments.append(deepcopy(diagnostic))


def research_repairs(agent):
    refresh_dependencies(agent)
    items = []
    for model, state in agent.research_candidates.items():
        if state.get('status') not in {'needs_rebuild', 'needs_refit', 'representation_conflict', 'execution_failed',
                                      'new_paid_rows_unscored'}:
            continue
        item = {'model': model, **deepcopy(state), 'current_paid_rows': len(agent.rows)}
        if state.get('current_evaluation') is not None:
            item['evaluation_provenance'] = {
                'status': 'last_recorded_evaluation',
                'paid_rows': state.get('paid_rows'), 'current_paid_rows': len(agent.rows),
                'scope': 'current_evaluation is the recorded score of this candidate on '
                    'paid_rows. Attaching that score to this repair suggestion performs '
                    'no new evaluation or research step.'}
        if state['status'] in {'needs_rebuild', 'needs_refit', 'new_paid_rows_unscored'}:
            item['next_action'] = _rebuild_action(agent, model, state)
            item['next_action_status'] = 'proposed_not_executed'
        elif state['status'] == 'representation_conflict':
            item['next_research'] = 'Revise the actual representation using its raw witnesses, or explicitly challenge its assumptions.'
        items.append(item)
    return items

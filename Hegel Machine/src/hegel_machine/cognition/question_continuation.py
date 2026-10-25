"""Execute a declared question against paid evidence and its existing parent uses.

Fitting and acquisition still belong to the common executor. A step can prepare
a parent query; buying it requires the ordinary query action.
"""

from copy import deepcopy
from dataclasses import replace
from math import isfinite
import json
import sys

from .executable_evidence import bind_components, materialize, record_answer, refresh_dependencies
from .online_inquiry_llm import _PLAN_RUNNER
from .question_semantics import compile_question, validate_answer_type
from .question_queries import evaluate_on_evidence
from .question_object_queries import evaluate_parameter_region
from .question_agenda import (
    effective_question_mission, mission_compute_remaining, debit_mission_compute, question_work_basis,
)


QUESTION_INTERFACE = """
run {operator:'research_step', research_question:{id:...,...}} compiles a declared
query_semantics.query_expression against already paid records. Question fields
include kind, context_ref, answer_type, answer_resolution_ref and use_sites.
Readout, Difference, RelationResidual, CheckApplication and AllOnFiniteSet are
executable queries; AllOnFiniteSet declares finite_cases of predicate queries,
and its answer keeps that finite scope rather than claiming a universal law;
ParameterRegion reads a checked region at an explicit question, meaning and
executable answer version, preserving the source scope and support dependency;
FindDistinguishingExperiment compares claim_refs at target_query using the
declared resolution_experiments panel, preserving ties and missing evidence.
FindSharedStructure consumes typed_program_refs through the existing budgeted
theory consolidation and returns shared-library reconstruction references.
MethodEffect reads an executed paired comparison_ref for the declared
effect_kind, left_arm and right_arm; retained prior interventions remain typed
as prior configuration effects, with their original prediction and cost scope.
unoperationalized ideas return explicit author_work_request without a model call.
use_sites [{kind:'bind_parameter_region',model:'parent.py',slot:'parameter',
refit:{validation_records:[...],path:'shared/refit'}}] consume the checked child
region through existing bind_research/refit. The optional refit uses paid rows.
use_sites [{kind:'choose_parent_experiment',parent_id:'parent',path:'plan.py'}]
execute the existing select_actions program to prepare pending_query only.
query {pending_question:'parent'} purchases that saved plan through the normal
forecast-before-acquisition path. Changed plans keep their preceding version.
Readout.record_index={experiment_id:'panel_id',action_index:0} refers to the
actual pending purchase by this question and meaning, preserving strict public
binding checks. A plan not yet executed has no measured answer.
use_sites [{kind:'supply_component',model:'parent.py',slot:'chosen_slot'}] consume
this question's current registered observable/conditional model_answer after its
paid support and meaning have been checked. The existing parent bind_models
interface receives the frozen predictor; its input coordinates remain authored.
The parent is evaluated on paid rows, or refitted when use.refit is supplied.
No parent use means an answer is available, not a completed parent revision.
"""


def _question(agent, identifier):
    return next(q for q in agent.questions if q['id'] == identifier)


def executed_plan_records(agent, question):
    """Actual first-step purchases made for this question's retained meaning."""
    bindings = {}
    for parent in agent.questions:
        for executed in parent.get('executed_pending_queries', []):
            source = executed['source']
            if (source.get('kind') == 'question_selection' and source.get('stage') != 2
                    and source.get('question_id') == question['id']
                    and source.get('meaning_version') == question.get('meaning_version', 1)
                    and source.get('experiment_id')):
                bindings[source['experiment_id']] = list(executed['record_indices'])
    return bindings


def _versioned_support(agent, question, visiting=()):
    """Explicit OR paths, each containing necessary AND premises.

    These are justifications for this answer, not arbitrary Python call edges.
    No path obtains support merely by participating in a cycle.
    """
    identifier = question['id']
    if identifier in visiting:
        return False, []
    paths = question.get('support_paths')
    if paths is None:
        return True, []
    assessed = []
    for path in paths:
        complete = bool(path)
        refs = []
        for reference in path:
            parent = next((q for q in agent.questions if q['id'] == reference['question_id']), None)
            answer = {}
            if parent:
                answer = (parent.get(reference['answer_field'], {}) if reference.get('answer_field') else
                          parent.get('semantic_answer') or parent.get('executable_answer') or {})
            valid = bool(parent and answer and (
                answer.get('status') == 'answered' or (
                    answer.get('status') == 'supported' and answer.get('evidence_status') == 'supported')))
            if parent:
                valid = valid and reference.get('meaning_version', parent.get('meaning_version', 1)) == parent.get('meaning_version', 1)
                valid = valid and reference.get('answer_version', answer.get('version')) == answer.get('version')
                valid = valid and _versioned_support(agent, parent, (*visiting, identifier))[0]
                if valid:
                    # Resolve an unversioned justification once, when it is
                    # actually usable. A new meaning cannot revive an old edge
                    # merely by restarting its answer counter at one.
                    reference.setdefault('meaning_version', parent.get('meaning_version', 1))
                    reference.setdefault('answer_version', answer.get('version'))
            refs.append({**deepcopy(reference), 'valid': bool(valid),
                         'current_version': answer.get('version')})
            complete = complete and valid
        assessed.append({'complete': bool(complete), 'premises': refs})
    return any(path['complete'] for path in assessed), assessed


def _author_work(question, reason, object_type='operationalized_question'):
    request = {'status': 'author_work_request', 'question_id': question['id'],
               'meaning_version': question.get('meaning_version', 1),
               'object_type': object_type, 'reason': reason}
    question['pending_author_work'] = deepcopy(request)
    return request


def _discard_answer(question):
    previous = question.pop('semantic_answer', None)
    if previous:
        question.setdefault('semantic_answer_history', []).append(previous)
    question.pop('applied_question_use', None)


def _parameter_region_answer(agent, question, query):
    """Read the current checked source before considering this query's cache."""
    answer = evaluate_parameter_region(query, agent.questions, context=question.get('context_ref'))
    if answer['status'] == 'answered':
        reference = {**answer['source_ref'], 'answer_field': 'executable_answer'}
        source = _question(agent, reference['question_id'])
        if not _versioned_support(agent, source)[0]:
            answer = {'status': 'scoped_unknown', 'reason': 'premise_unknown', 'compute_cost': 0}
        elif source['id'] != question['id']:
            # The retrieved source is necessary on every independently authored
            # support path. Its executable answer has its own version counter.
            paths = question.get('support_paths', [[]])
            question['support_paths'] = [path if reference in path else [*path, deepcopy(reference)]
                                         for path in paths]
        if answer['status'] == 'answered':
            answer['source_ref'] = reference
            record_answer(agent, question, answer['answer_record'])
            return answer
    record_answer(agent, question, {'status': 'inconclusive', 'evidence_status': 'inconclusive',
                                   'evidence': [], 'check_results': []})
    _discard_answer(question)
    refresh_dependencies(agent)
    return answer


def _retained_component_answer(agent, question):
    """Resolve an actually returned component, never a prose or numeric answer."""
    for field, kind, pool in (('conditional_component', 'conditional', agent.conditional_models),
                              ('observable_target', 'observable', agent.observable_models)):
        if field not in question or 'model_answer' not in question:
            continue
        identifier = question[field]
        entry = pool.get(identifier)
        saved = question['model_answer']
        if (entry is None or question.get('status') != entry.get('support')
                or entry.get('support') not in {'paid_fit', 'paid_evaluation', 'supported_on_later_readings'}
                or any(saved.get(key) != entry.get(key) for key in ('model', 'version', 'view'))):
            return {'status': 'scoped_unknown', 'reason': 'current_paid_component_answer_required',
                    'component_id': identifier}
        return {'status': 'answered', 'compute_cost': 0,
            'value': {'kind': 'retained_component', 'component_kind': kind, 'component_id': identifier,
                'component_version': entry['version'], 'model': entry['model'], 'support': entry['support']},
            'evidence': [{'file': 'research_state.json', 'json_path': [
                'conditional_models' if kind == 'conditional' else 'observable_models', identifier],
                'version': entry['version']}],
            'scope': {'kind': 'paid_component_evaluation', 'context_ref': deepcopy(question.get('context_ref')),
                'training_observation_indices': deepcopy(entry.get('training_observation_indices')),
                'evaluated_observation_indices': deepcopy(entry.get('evaluated_observation_indices'))}}
    return None


def _component_use_current(agent, question, uses):
    path = agent.workspace / 'model_bindings.json'
    bindings = json.loads(path.read_text()) if path.exists() else {}
    answer = question.get('semantic_answer', {})
    value = answer.get('value', {})
    reference = {'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1),
                 'answer_version': answer.get('version')}
    return all(bindings.get(use['model'], {}).get(use['slot']) == value.get('model')
        and agent.research_candidates.get(use['model'], {}).get('components', {}).get(
            use['slot'], {}).get('question_ref') == reference
        for use in uses if use['kind'] == 'supply_component')


def _apply_component(agent, question, use, laboratory, attempt, index):
    value = question['semantic_answer']['value']
    if not isinstance(value, dict) or value.get('kind') != 'retained_component':
        return _author_work(question, 'Supply a current paid registered component answer', 'supply_component')
    path = agent.workspace / 'model_bindings.json'
    bindings = json.loads(path.read_text()) if path.exists() else {}
    parent_bindings = bindings.setdefault(use['model'], {})
    parent_bindings[use['slot']] = value['model']
    agent._save('model_bindings.json', json.dumps(bindings, indent=2))
    reference = {'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1),
                 'answer_version': question['semantic_answer']['version']}
    bind_components(agent, use['model'], parent_bindings, question_refs={use['slot']: reference})
    if use.get('refit') is not None:
        refit = deepcopy(use['refit'])
        if refit.get('path'):
            refit['path'] = (f"{refit['path']}/continuation_m{question.get('meaning_version', 1):04d}"
                             f"_{attempt:04d}_{index:02d}")
        feedback = agent._execute('run', {'operator': 'refit', 'model': use['model'], **refit}, laboratory)
    else:
        feedback = agent._execute('evaluate', {'model': use['model']}, laboratory)
    evaluated = feedback.get('validation', feedback)
    if feedback.get('truncated') or evaluated.get('truncated'):
        return {'kind': use['kind'], 'status': 'parent_use_incomplete',
                'reason': 'unknown_due_to_incomplete_search', 'detail': 'compute_budget',
                'feedback': feedback}
    consumed = evaluated.get('returncode') == 0 and any(
        item.get('slot') == use['slot'] and item.get('calls', 0) > 0
        for item in evaluated.get('model_dependencies', []))
    if not consumed:
        return {'kind': use['kind'], **_author_work(question,
            'The parent must execute its authored component binding before this use is complete',
            'component_parent_use'), 'feedback': feedback}
    return {'kind': use['kind'], 'status': 'parent_component_evaluated',
            'binding': {**deepcopy(value), 'slot': use['slot'], 'question_ref': reference}, 'feedback': feedback}


def _prepare_parent_query(agent, question, use, laboratory):
    parent = _question(agent, use.get('parent_id', question.get('parent_id')))
    path = use['path']
    agent._prepare_program_state()
    code, stdout, stderr, timed_out = agent._python(
        [sys.executable, '-c', _PLAN_RUNNER, str(agent.workspace / path)],
        json.dumps({'spec': laboratory.public_spec(), 'observations': agent.observations,
                    'memory': agent._program_memory(path, {'question_id': question['id']})}))
    if code:
        return {'status': 'parent_plan_failed', 'returncode': code,
                'stderr': stderr, 'timeout': timed_out}
    proposed = json.loads(stdout)
    if not isinstance(proposed, dict) or not proposed.get('actions'):
        changed = 'pending_query' in parent
        if 'pending_query' in parent:
            parent.setdefault('pending_query_history', []).append(parent.pop('pending_query'))
            parent.pop('pending_query_source', None)
        return {'status': 'parent_plan_stopped', 'proposal': proposed, 'changed': changed}
    pending = {key: deepcopy(proposed[key]) for key in
               ('actions', 'revision', 'question', 'after_observation', 'knowledge_sources') if key in proposed}
    pending.setdefault('question', {'id': parent['id']})
    previous = parent.get('pending_query')
    if pending != previous:
        if previous is not None:
            parent.setdefault('pending_query_history', []).append(deepcopy(previous))
        parent['pending_query'] = pending
    parent['pending_query_source'] = {'question_id': question['id'],
        'meaning_version': question.get('meaning_version', 1),
        'answer_version': question.get('semantic_answer', question.get('executable_answer', {})).get('version'),
        'path': path, 'paid_records': len(agent.observations),
        'use_version': len(question.get('parent_use_history', [])) + 1}
    return {'status': 'parent_query_prepared', 'parent_id': parent['id'],
            'changed': pending != previous, 'query': deepcopy(pending)}


def pending_query_arguments(agent, parent_id):
    parent = _question(agent, parent_id)
    source = parent['pending_query_source']
    if source.get('kind') == 'question_selection':
        from .question_routing import pending_selection_current
        if not pending_selection_current(agent, parent):
            raise ValueError('pending parent query needs rebuilding from the current research selection')
        return deepcopy(parent['pending_query'])
    child = _question(agent, source['question_id'])
    answer = child.get('semantic_answer') or child.get('executable_answer') or {}
    applied = child.get('applied_question_use', {})
    if (child.get('meaning_version', 1) != source['meaning_version']
            or answer.get('version') != source['answer_version']
            or not _versioned_support(agent, child)[0]
            or len(agent.observations) != source['paid_records']
            or source.get('use_version') != len(child.get('parent_use_history', []))
            or applied.get('uses') != child.get('use_sites', [])
            or any(not (agent.workspace / path).is_file()
                   or (agent.workspace / path).read_text() != code
                   for path, code in applied.get('program_sources', {}).items())):
        raise ValueError('pending parent query needs rebuilding from the current child answer')
    return deepcopy(parent['pending_query'])


def _apply_use(agent, question, use, laboratory, attempt, index):
    if use['kind'] == 'bind_parameter_region':
        model = use['model']
        inputs = deepcopy(agent.research_input_bindings.get(model, {}))
        inputs[use['slot']] = question['id']
        bound = materialize(agent, model, inputs)
        if use.get('refit') is not None:
            refit = deepcopy(use['refit'])
            if refit.get('path'):
                refit['path'] = (f"{refit['path']}/continuation_m{question.get('meaning_version', 1):04d}"
                                 f"_{attempt:04d}_{index:02d}")
            feedback = agent._execute('run', {'operator': 'refit', 'model': model, **refit}, laboratory)
            evaluated = feedback.get('validation', feedback)
            if feedback.get('truncated') or evaluated.get('truncated'):
                return {'kind': use['kind'], 'status': 'parent_use_incomplete', 'binding': bound,
                        'reason': 'unknown_due_to_incomplete_search', 'detail': 'compute_budget',
                        'feedback': feedback}
            return {'kind': use['kind'], 'binding': bound, 'feedback': feedback,
                    'status': 'parent_refitted' if evaluated.get('returncode') == 0
                        and evaluated.get('mse') is not None else 'parent_refit_failed'}
        else:
            return {'kind': use['kind'], 'status': 'parent_bound_needs_refit', 'binding': bound}
    if use['kind'] == 'choose_parent_experiment':
        if use.get('path'):
            return {'kind': use['kind'], **_prepare_parent_query(agent, question, use, laboratory)}
        return _author_work(question, 'A measured answer is ready; supply its parent experiment use',
                            'experiment_program')
    if use['kind'] == 'supply_component':
        return _apply_component(agent, question, use, laboratory, attempt, index)
    return _author_work(question, f"An executable parent use is required for {use['kind']}", use['kind'])


def _apply_uses(agent, question, uses, laboratory):
    results = []
    attempt = len(question.get('parent_use_history', []))
    for index, use in enumerate(uses):
        try:
            results.append(_apply_use(agent, question, use, laboratory, attempt, index))
        except ValueError as error:
            if str(error) not in {'shared research operator compute budget exhausted',
                                  'candidate-row scoring budget exhausted'}:
                raise
            results.append({'kind': use['kind'], 'status': 'parent_use_incomplete',
                            'reason': 'unknown_due_to_incomplete_search', 'detail': 'compute_budget'})
    question.setdefault('parent_use_history', []).append({'answer_version':
        question.get('semantic_answer', {}).get('version'), 'results': [{
            key: deepcopy(result[key]) for key in ('kind', 'status', 'parent_id', 'changed') if key in result
        } | {key: deepcopy(result.get('feedback', {})[key]) for key in
             ('tested_model', 'source_model') if key in result.get('feedback', {})}
        for result in results]})
    return results




def _parent_use_completed(results):
    return any(result['status'] in {'parent_refitted', 'parent_component_evaluated'} or (
        result['status'] in {'parent_query_prepared', 'parent_plan_stopped'} and result.get('changed'))
        for result in results)

def parent_revision_obligation(agent, question, answer_execution):
    """Keep a measured answer and route its unfinished, failed parent separately."""
    target = question.get('theory_target') or {}
    retained = agent.evaluated_model or {}
    mse = retained.get('current_mse')
    parent_id = question.get('parent_id')
    parent_theory = agent.theory_index.get(target.get('parent_theory_id'), {})
    if (answer_execution.get('parent_revision_completed') is not False
            or target.get('revision_fitting') is None or not parent_theory
            or not parent_id or not question.get('use_sites') or not target.get('evidence_checks')
            or type(mse) not in (int, float) or not isfinite(mse)
            or mse <= agent.config.confirmation_mse_threshold
            or not retained.get('evaluated_rows') or not retained.get('source_model')):
        return None
    source = {'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)}
    subject = {'kind': 'theory', 'id': parent_theory['id'], 'version': parent_theory.get('version', 1)}
    paid = {'kind': 'paid_evidence', 'records': len(agent.observations), 'rows': len(agent.rows)}
    obligation = agent._open_question({
        'id': f'parent_revision:{parent_id}:{question["id"]}',
        'question': 'Revise the retained parent against its remaining paid counterexamples for the declared parent use.',
        'parent_id': parent_id, 'depends_on': [], 'kind': 'program_revision',
        'answer_type': 'program', 'context_ref': deepcopy(question.get('context_ref')),
        'answer_resolution_ref': {'relation': 'le', 'atol': agent.config.confirmation_mse_threshold,
            'scope': 'paid_candidate_comparison_not_independent_confirmation'},
        'subject_refs': [subject, paid, {'kind': 'native_research', **source}], 'query_semantics': {'mode': 'solution_relation'},
        'use_sites': deepcopy(question.get('use_sites', [])),
        'origin': {'kind': 'unfinished_parent_revision', 'source_question_ref': source},
        'native_research_ref': source, 'theory_target': deepcopy(target)})
    for key in ('frozen_library_path', 'mission', 'exploration_policy', 'research_comparison'):
        if key in question:
            obligation[key] = deepcopy(question[key])
    reference = {'question_id': obligation['id'], 'meaning_version': obligation.get('meaning_version', 1)}
    question['parent_revision_question_ref'] = reference
    parent = next((owner for owner in agent.questions if owner['id'] == parent_id), None)
    if parent is not None:
        parent['parent_revision_question_ref'] = deepcopy(reference)
        parent['status'] = 'reopened'
    return obligation


def parent_revision_work(agent, question):
    """Expose unfinished meanings of the same parent use, without rebinding them.

    A failed before predictor is a reason to revise it. Only a completed search
    or an unavailable premise closes that research opportunity.
    """
    from .research_episodes import _question_versions

    origin = question.get('origin') or {}
    if (question.get('kind') != 'program_revision' or not isinstance(origin, dict)
            or origin.get('kind') != 'unfinished_parent_revision'):
        return []
    current_target = question.get('theory_target') or {}
    result = []
    for (identifier, meaning), (work, reference) in _question_versions(agent).items():
        if identifier != question['id']:
            continue
        target = work.get('theory_target') or {}
        if (work.get('parent_id') != question.get('parent_id')
                or work.get('context_ref') != question.get('context_ref')
                or work.get('use_sites') != question.get('use_sites')
                or work.get('native_research_ref') != question.get('native_research_ref')
                or any(target.get(key) != current_target.get(key) for key in
                       ('role_signature', 'output_unit', 'context_ref', 'context_family_ref'))):
            continue
        recall = work.get('frozen_recall_execution') or {}
        basis = recall.get('basis') or {}
        routing = (work.get('routing_execution') or {}).get('basis') or {}
        # Earlier snapshots retained these exact execution bases rather than
        # the operative fields. Restore their own contracts, never today's parent.
        for field, value in (
                ('frozen_library_path', basis.get('library_path')),
                ('mission', routing.get('mission')),
                ('exploration_policy', routing.get('exploration_policy')),
                ('allowed_mappings', routing.get('allowed_mappings'))):
            if field not in work and value is not None:
                work[field] = deepcopy(value)
        comparison = routing.get('research_comparison') or {}
        if 'research_comparison' not in work and comparison.get('contract'):
            work['research_comparison'] = deepcopy(comparison['contract'])
        prefix = f'frozen_recall:{identifier}@{meaning}:'
        retained_executions = agent.theory_consolidation.get('executions', {})
        groups = recall.get('result', {}).get('execution_work_groups')
        executions = ([retained_executions[group['execution_key']] for group in groups
                       if group.get('execution_key') in retained_executions] if groups is not None else
                      [execution for key, execution in retained_executions.items()
                       if key.startswith(f'relation_transfer:{prefix}')])
        pending = [execution for execution in executions
                   if execution.get('search_completed', True) is False]
        state = 'unstarted' if not recall else 'completed_no_gain'
        if pending or recall.get('result', {}).get('search_completed') is False:
            state = 'budget_incomplete'
        elif executions and all(execution.get('result', {}).get('status') == 'refuted'
                                for execution in executions):
            state = 'refuted'
        elif recall.get('result', {}).get('status') == 'author_work_request':
            state = 'author_work_required'
        if meaning != question.get('meaning_version', 1) and not recall:
            continue
        valid, _ = _versioned_support(agent, work)
        if not valid:
            state = 'premise_unknown'
        cutoff = target.get('paid_evidence_scope') or next((ref for ref in work.get('subject_refs', [])
                                                           if ref.get('kind') == 'paid_evidence'), {})
        if (cutoff.get('records', 0) > len(agent.observations)
                or cutoff.get('rows', 0) > len(agent.rows)):
            state = 'evidence_unavailable'
        requirements = [execution.get('result', {}).get('minimum_compute_for_progress', 1)
                        for execution in pending]
        requirements.extend(item.get('required_compute', 1)
                            for item in recall.get('result', {}).get('unfunded_objects', []))
        minimum = min(requirements, default=1)
        result.append({'question_id': identifier, 'meaning_version': meaning,
            'question_ref': {'question_id': identifier, 'meaning_version': meaning},
            'state': state, 'minimum_compute_for_progress': minimum,
            'source_ref': deepcopy(reference), 'parent_theory_id': target.get('parent_theory_id'),
            'paid_scope': deepcopy(cutoff),
            'requires_current_evidence_comparison': cutoff.get('rows') != len(agent.rows)})
    return result


def select_parent_revision_work(agent, question, allowance):
    """Offer one funded meaning in the existing agenda's persistent event order."""
    from .question_agenda import build_agenda
    from .research_episodes import _question_versions

    work = parent_revision_work(agent, question)
    if not work:
        return question, None
    versions = _question_versions(agent)
    for item in work:
        snapshot = versions[item['question_id'], item['meaning_version']][0]
        mission = effective_question_mission(snapshot, agent.research_mission)
        item['available_compute'] = (min(allowance, mission_compute_remaining(agent, mission))
            if mission.get('purpose') in {'theory_learning', 'method_learning'} else allowance)
    order = [item['meaning_version'] for item in work]
    next_meaning = question.get('parent_research_cursor', {}).get(
        'next_meaning_version', question.get('meaning_version', 1))
    first = order.index(next_meaning) if next_meaning in order else order.index(question.get('meaning_version', 1))
    ordered = [*work[first:], *work[:first]]
    agenda = build_agenda([question], {}, [], mission=effective_question_mission(question, agent.research_mission),
        budget={'compute': allowance, 'author_calls': 0},
        parent_research={question['id']: ordered})
    question['parent_research_agenda'] = deepcopy(agenda)
    selected = next((entry for entry in agenda['task_agenda'] if entry['status'] == 'ready'), None)
    if selected is None:
        return None, agenda
    meaning = selected['meaning_version']
    question['parent_research_cursor'] = {'next_meaning_version': order[(order.index(meaning) + 1) % len(order)]}
    question['parent_research_last_selection'] = deepcopy(selected['question_ref'])
    return _question_versions(agent)[question['id'], meaning][0], agenda

def step_question(agent, question, laboratory, *, mission=None):
    """The same computable route is available with or without an LLM provider."""
    mission = effective_question_mission(question, mission if mission is not None else agent.research_mission)
    compiled = compile_question(question, mission=mission)
    question['compilation'] = deepcopy(compiled)
    uses = compiled.get('use_sites', question.get('use_sites', []))
    parameter_answer = None
    if compiled['status'] == 'ready' and compiled['query']['op'] == 'ParameterRegion':
        parameter_answer = _parameter_region_answer(agent, question, compiled['query'])
        if parameter_answer['status'] != 'answered':
            return parameter_answer
    # Checked parameter answers predate the query language. Consume them through
    # the existing numeric-check contract instead of inventing a new inference.
    region = question.get('executable_answer', {})
    checked_region = (region.get('status') == 'supported' and region.get('evidence_status') == 'supported'
                      and region.get('parameter_region'))
    component_answer = (_retained_component_answer(agent, question)
        if question.get('answer_type') == 'program' and any(use['kind'] == 'supply_component' for use in uses)
        else None)
    checked_component = (component_answer is not None and component_answer['status'] == 'answered'
        and (question.get('query_semantics') or {}).get('mode') == 'solution_relation'
        and compiled.get('reason') == 'missing_executable_query')
    if component_answer is not None and component_answer['status'] != 'answered':
        _discard_answer(question)
        refresh_dependencies(agent)
        return component_answer
    if compiled['status'] != 'ready' and not checked_region and not checked_component:
        if compiled['status'] == 'deferred':
            return compiled
        required = compiled.get('request', {})
        request = _author_work(question, compiled.get('reason', compiled['status']),
                               required.get('object_type', 'operationalized_question'))
        if required.get('required_contract'):
            request['required_contract'] = deepcopy(required['required_contract'])
            question['pending_author_work'] = deepcopy(request)
        return request
    valid, paths = _versioned_support(agent, question)
    question['support_path_status'] = paths
    if not valid:
        # With another complete path the answer stays usable; only genuine
        # dependents are stale when all its supplied justifications fail.
        for model, bindings in agent.research_input_bindings.items():
            if question['id'] in bindings.values():
                state = agent.research_candidates.setdefault(model, {})
                state['status'] = 'needs_rebuild'
                state.pop('current_evaluation', None)
        _discard_answer(question)
        refresh_dependencies(agent)
        return {'status': 'scoped_unknown', 'reason': 'premise_unknown', 'support_paths': paths}
    operation = (compiled.get('query') or {}).get('op')
    if operation == 'FindDistinguishingExperiment':
        from .question_routing import find_distinguishing_experiment
        return find_distinguishing_experiment(agent, question, laboratory, mission=mission)
    record_bindings = executed_plan_records(agent, question)
    basis = {'meaning_version': question.get('meaning_version', 1),
             'paid_records': len(agent.observations), 'paid_rows': len(agent.rows),
             'query': deepcopy(compiled.get('query')), 'context': deepcopy(question.get('context_ref')),
             'region_version': region.get('version'), 'record_bindings': record_bindings}
    cached = question.get('query_execution')
    if operation == 'FindSharedStructure':
        from .question_object_queries import evaluate_shared_structure
        answer = evaluate_shared_structure(agent, question, compiled['query'], laboratory, mission)
        question['query_execution'] = {'basis': basis, 'answer': deepcopy(answer)}
    elif operation == 'MethodEffect':
        from .method_effect_query import evaluate_method_effect
        answer = evaluate_method_effect(compiled['query'], agent.workspace)
        question['query_execution'] = {'basis': basis, 'answer': deepcopy(answer)}
    elif parameter_answer is not None:
        answer = parameter_answer
        question['query_execution'] = {'basis': basis, 'answer': deepcopy(answer)}
    elif checked_component:
        answer = component_answer
    elif checked_region:
        answer = {'status': 'answered', 'value': deepcopy(region['parameter_region']),
                  'evidence': deepcopy(region.get('evidence', [])), 'compute_cost': 0}
    elif cached and cached['basis'] == basis and cached['answer']['status'] == 'answered':
        answer = deepcopy(cached['answer'])
    else:
        before_compute = agent.costs['compute_cost']
        answer = evaluate_on_evidence(compiled['query'], agent.observations,
            context=question.get('context_ref'),
            compute_budget=mission_compute_remaining(agent, mission), record_bindings=record_bindings)
        agent.costs['compute_cost'] += answer.get('compute_cost', 0)
        debit_mission_compute(agent, mission, before_compute)
        question['query_execution'] = {'basis': basis, 'answer': deepcopy(answer)}
    if answer['status'] != 'answered':
        _discard_answer(question)
        if answer.get('author_work') and question.get('pending_author_work'):
            question['pending_author_work']['agenda_basis'] = question_work_basis(question)
        return answer
    if not validate_answer_type(answer['value'], compiled.get('answer_type') or 'interval'):
        _discard_answer(question)
        return _author_work(question, 'Computed answer does not have the declared answer_type')
    prior = question.get('semantic_answer')
    effective = {key: deepcopy(answer[key]) for key in
                 ('status', 'value', 'units', 'support_units', 'statistic_ref',
                  'evidence', 'scope', 'finite_status', 'case_results',
                  'counterexample_case_indices', 'unresolved_cases', 'source_ref', 'completeness',
                  'region_kind', 'independent_confirmation', 'scientific_support',
                  'source_reconstruction_only', 'search_completeness') if key in answer}
    effective['meaning_version'] = question.get('meaning_version', 1)
    if prior and {k: v for k, v in prior.items() if k != 'version'} == effective:
        current = prior
    else:
        if prior:
            question.setdefault('semantic_answer_history', []).append(deepcopy(prior))
        previous_versions = [item.get('version', 0) for item in question.get('semantic_answer_history', [])]
        current = {**effective, 'version': max([(prior or {}).get('version', 0), *previous_versions]) + 1}
        question['semantic_answer'] = current
    if answer.get('author_work') and question.get('pending_author_work'):
        question['pending_author_work']['agenda_basis'] = question_work_basis(question)
    else:
        question.pop('pending_author_work', None)
    if not uses:
        return {'status': 'answer_ready_no_parent_use', 'answer': deepcopy(current),
                'parent_revision_completed': False}
    use_basis = {'answer': deepcopy(current), 'uses': deepcopy(uses),
                 'paid_cutoff': {'records': len(agent.observations), 'rows': len(agent.rows)},
                 'program_sources': {use['path']: (agent.workspace / use['path']).read_text()
                    for use in uses if use.get('path')},
                 'model_sources': {use['model']: (agent.workspace / use['model']).read_text()
                    for use in uses if use.get('model')}}
    if question.get('applied_question_use') == use_basis and _component_use_current(agent, question, uses):
        return {'status': 'reused_answer', 'answer': deepcopy(current), 'new_parent_work': False,
                'parent_revision_completed': _parent_use_completed(
                    question.get('parent_use_history', [{}])[-1].get('results', []))}
    before_compute = agent.costs['compute_cost']
    original_config = agent.config
    if mission.get('purpose') in {'theory_learning', 'method_learning'}:
        agent.config = replace(original_config, compute_budget=before_compute + mission_compute_remaining(agent, mission))
    try:
        results = _apply_uses(agent, question, uses, laboratory)
    finally:
        agent.config = original_config
        debit_mission_compute(agent, mission, before_compute)
    failed_plan = next((result for result in results if result['status'] == 'parent_plan_failed'), None)
    if failed_plan:
        results.append(_author_work(question, failed_plan.get('stderr') or 'Parent experiment program failed',
                                    'experiment_program'))
    if all(result['status'] in {'parent_query_prepared', 'parent_refitted', 'parent_bound_needs_refit',
                               'parent_plan_stopped', 'parent_component_evaluated'} for result in results):
        question['applied_question_use'] = use_basis
    incomplete = next((result for result in results if result['status'] == 'parent_use_incomplete'), None)
    return {'status': ('scoped_unknown' if incomplete else 'author_work_request' if any(
                result['status'] == 'author_work_request' for result in results) else 'answer_applied'),
            **({key: incomplete[key] for key in ('reason', 'detail')} if incomplete else {}),
            'answer': deepcopy(current), 'uses': results,
            'parent_revision_completed': _parent_use_completed(results)}

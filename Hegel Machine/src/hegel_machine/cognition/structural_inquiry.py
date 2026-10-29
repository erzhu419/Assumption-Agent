"""Connect authored laboratory coordinates to the existing relation researcher."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from time import monotonic

from .code_agent import _copy_model_context, python_model_context
from .conditional_research import _BIND_RUNNER
from .conditional_coordinates import conditional_search_allowance
from .conditional_experiments import assess_frozen_structure_predictions
from .language import Program
from .models import SearchBudget
from .native_condition_research import (
    FINITE_PRIOR, research_native_conditions, update_native_boundary,
    native_structure_planning_contract, plan_native_structure_experiments,
    accumulate_matched_pairs, matched_assessment_compute_upper, matched_assessment_shape_upper,
    retained_structure_refit_quote, _structural_hypotheses,
)
from .temporal_operator_agent import export_temporal_program
from .world_relation_research import research_relation


NATIVE_CONDITION_INTERFACE = """
run {operator:'native_condition_research',question:{id:'your_composition_question',
question:'your competing explanation',parent_id:'task',depends_on:[]}}
reuses the historical finite condition/window researcher on purchased trajectories.
Its explicit prior is linear/quadratic public-control terms, measured thresholds,
finite window sums, condition placement, and two-condition AND/OR/XOR combinations.
This is algorithm-assisted hypothesis search, not unrestricted scientific invention.
You may select record_indices (complete laboratory trajectories), matched_pairs
[{reference_record_index:...,record_index:...}], Program basis terms, competing
Program programs, max_candidates, and max_compute. With no selections, it reuses
measured constant-background pairs for each public control and fits all paid rows.
max_compute bounds this invocation's search, including its query continuation;
omitting it permits the remaining shared compute account less completion_reserve.
max_candidates controls search breadth, not total work or retained bank size. Choose max_compute
when allocating effort across alternative investigations. Actual search cost,
remaining_compute and completion_reserve are returned after execution.
search_incomplete and research_decision.computation_incomplete mean requested
work was not funded, not that an untested model family failed. search_stages
reports actual completed fits, costs and blocked work. A same-question
compute_continuation, when available, offers a larger allocation for your choice;
its research_unit prices preparation, a bounded joint attempt, exports, comparison
and finish. The existing full retained pool and bounded new exports stay funded.
An unaffordable complete unit has no action; an exhausted original execution
window remains exhausted. The restart retains old Programs and recipes and
charges repeated local assessment. Quoted joint work is an estimate; returned
search stages record what actually completed.
It preserves reset history and compares candidate compositions with real evidence.
Selected matched pairs for the same control and constant background are assessed
together, keeping each trajectory's reset history. Different backgrounds remain
separate. Evidence lists every contributing record and response row; a local fit
on one pair does not discard contradictory measurements from the other pairs.
Results retain local evidence, measured candidate losses, frozen executable model
paths and proposed experiments. Choose a returned evaluate_action to test a model
on all paid data, then finish for independent confirmation if warranted. You may
revise the selected evidence/basis or challenge this finite prior using ordinary
Python. Old rounds and refuted candidates remain in the question's
native_condition_research. Repeating the action rescores frozen earlier programs
before fitting new ones. Nothing is purchased, adopted or confirmed by this action.
Suggested boundary query actions return their new evidence to a low-cost update
of the measured interval and retained candidates, without repeating structural
search. Their after_observation.boundary_update identifies the exact previous
round and proposal. research_decision and remaining_boundary describe whether
parameter ambiguity remains or the observed response refutes the tested
structure. You choose another proposed experiment, evaluate a retained candidate,
or explicitly run this operator again with revised evidence/basis for structural
search; no update automatically buys evidence or replaces your selected model.
Each direct run and query continuation records one chronological entry in
research_state.json['operator_results'], using the same flat feedback contract:
{index, arguments, operator:'native_condition_research', research_question_id,
 native_condition_research:{round_index, status, candidates, exported_models,
                            proposed_experiments, fit_evidence}, returned_answer}.
Failed calls instead have error at the entry's top level. research_not_executed means
the selected inputs were incompatible, not that a hypothesis was refuted.
invalid_pairs shows actual changed controls and their trajectory steps;
available_pairs and automatic_matching_action reuse compatible paid inputs.
The retry is optional and keeps your compute allocation and other search choices.
Read the most recent entry for your question, including results produced by query.after_observation;
method_results also records the surrounding method invocation. Each candidates
or exported_models item contains model, evaluate_action, fit_mse and source.
There is no top-level model_path. proposed_experiments items contain query_action.
Registered finite authors receive a research_continuation for joint searches
after acquisition; their next action selection prices it on the new paid rows.
The existing boundary_update continues inline at its original bounded allowance.
Candidate programs and selection decisions remain inline; complete research
recipes are retained once in the question's native_condition_research rounds.
Their full_content_file/json_path references identify the saved recipe.
An explicit question.theory_target {role_signature:{coordinate:{role,unit,time}},
output_unit,context_ref} also registers each typed candidate for the existing
theory routes. native_theory_registration returns its exact source reference.
Missing contracts return AuthorWorkRequest there; the executable native bank and
ordinary evaluate_action remain available. A local fit does not confer support.
This state records outcomes; your program still chooses its next action.
"""


STRUCTURAL_RESEARCH_INTERFACE = """
run {operator:'research_update',path:'your_binding.py',question:{id:'your_relation',
question:'your scientific question',parent_id:'task',depends_on:[]}}
executes your bind(public_spec,observations,memory). observations contains the
full purchased laboratory records; memory contains question_id and previous_research
for this same question. Return rows:[{features:{your_coordinate:number,...},
target:number,history:[earlier_coordinate_maps,...],evidence:[PAID_RESPONSE_ROW_IDS]}]
and your hypotheses, basis, candidate_additions, operations or probes as applicable.
You select the input coordinates, scientific expressions and applicable observations.
Evidence IDs are response row_indices, not laboratory record indices. The previous
candidate bank and rejected candidates persist under this question ID.
Program expressions use {op:'feature',name:'your_coordinate'}, {op:'const',value:number},
{op:'add'|'sub'|'mul'|'abs'|'min'|'max'|'ge'|'lt'|'and'|'or'|'not',children:[...]},
{op:'if',children:[condition,true_expression,false_expression]}, and history operations
{op:'lag',name:'your_coordinate',lag:positive_integer}, {op:'available',lag:integer},
or {op:'window_sum',lag:positive_integer,children:[expression]}. Ordinary Python
bind can construct any coordinates from the paid evidence. Choose the operators,
terms, parameters and conditions yourself. hypotheses supplies executable programs;
basis fits coefficients of your declared terms using the existing fitter. Explicit
candidate_additions adds revised programs without discarding old tested candidates.
When every retained candidate is refuted, an explicitly expanded basis is fitted
once on the paid rows, preserving the earlier tested programs and their evidence.
Each newly fitted candidate reports paid_fit_residuals with up to three actual
bound inputs, cited response IDs, observed targets and predictions. These errors
describe the tested candidate family; the separate representation diagnostic
checks whether identical represented inputs have incompatible observed targets.
Without hypotheses or basis, operations selects the existing finite synthesis
operations; max_history explicitly controls their historical input depth.
Optional probes:[{features:{...},history:[...],actions:[public_action,...]}] names
your proposed inputs and actual public experiment. The existing chooser compares
the surviving programs' predictions and returns an ordinary query action; it buys
nothing automatically. Repeat research_update under the same question after the
purchase to check the frozen candidates against the newly paid observations.
For exported predictors in derived coordinates, define represent(row) in this same
binding file. It receives only public features and history and returns either
{features:{...},history:[...]} or a flat feature map with no history. This must
produce the same coordinates used by bind. Omit represent for identity inputs.
Exported models keep the exact binding source and coefficients in a frozen context.
Evaluate a returned model with the ordinary evaluator before adopting or composing
it. Research update does not install, submit or independently confirm a model.
The full bank, measured losses, old failures and exported paths remain in
research_state.json.relation_research; its compact summary reports their scope.
"""


def structural_summary(entry):
    """Present actual candidate outcomes without repeating their shared paid rows."""
    summary = {key: deepcopy(entry[key]) for key in
        ('status', 'diagnosis', 'binding_path', 'round_index', 'compute_cost',
         'selected_program', 'selected_probe', 'fit_evidence', 'validation_evidence',
         'selected_evidence', 'confirmation_evidence', 'evidence', 'available_evidence',
         'exported_models', 'export_errors', 'round_history') if key in entry}
    for field in ('candidates', 'rejected_candidates'):
        summary[field] = [{key: deepcopy(item[key]) for key in
            ('program', 'status', 'fit_mse', 'validation_mse', 'confirmation_mse',
             'screening_mse', 'paid_fit_residuals', 'fit_evidence', 'validation_evidence', 'confirmation_evidence',
             'screening_evidence', 'proposed_after') if key in item}
            for item in entry.get(field, [])]
    summary['bound_row_count'] = len(entry.get('bound_rows', []))
    summary['full_state_path'] = 'research_state.json.relation_research'
    return summary


def _binding(agent, path, laboratory, question, previous):
    context, inner = python_model_context(agent.workspace, path)
    code, stdout, stderr, timed_out = agent._python(
        [sys.executable, '-c', _BIND_RUNNER, str(context / inner), str(context)],
        json.dumps({'spec': laboratory.public_spec(), 'observations': agent.observations,
                    'memory': {'question_id': question['id'],
                               'previous_research': previous}}))
    if code or timed_out:
        raise ValueError(f'Structural binding {path} failed: {stderr[-4000:]}')
    binding = json.loads(stdout)
    if not isinstance(binding, dict):
        raise ValueError('structural bind must return a binding object with rows')
    return binding


def _export_models(agent, path, result, bank_index, round_index):
    survivors = [(index, item) for index, item in enumerate(result['candidates'])
                 if item['status'] == 'plausible']
    if not survivors:
        return [], []
    destination = (Path('_experiment_models') /
                   f'relation_{bank_index:04d}_{round_index:04d}' / '_evaluated_model')
    inner = _copy_model_context(agent.workspace, path, agent.workspace / destination)
    names = set(result['bound_rows'][0]['features'])
    exported, errors = [], []
    for index, item in survivors:
        try:
            source = export_temporal_program(Program.from_dict(item['program']), feature_names=names)
            source += '\n_structural_predict = predict\n'
            source += ("import runpy as _structural_runpy\nfrom pathlib import Path as _StructuralPath\n"
                       "_structural_view = _structural_runpy.run_path(str(_StructuralPath(__file__).parent / "
                       + repr(inner) + "))\n"
                       "def predict(row):\n"
                       "    transform = _structural_view.get('represent')\n"
                       "    if not callable(transform):\n"
                       "        return _structural_predict(row)\n"
                       "    represented = transform(row)\n"
                       "    bound = (represented if 'features' in represented else\n"
                       "             {'features': represented, 'history': []})\n"
                       "    return _structural_predict({'features': bound['features'],\n"
                       "                                'history': bound.get('history', [])})\n")
            model = destination / f'structural_candidate_{index:04d}.py'
            (agent.workspace / model).write_text(source, encoding='utf-8')
            exported.append({'candidate_index': index, 'model': str(model),
                'status': item['status'], 'fit_mse': item['fit_mse'],
                'evaluate_action': {'action': 'evaluate', 'arguments': {'model': str(model)}},
                'scope': 'Frozen authored bound relation; parent prediction not confirmed.'})
        except (ValueError, KeyError) as error:
            errors.append({'candidate_index': index, 'error': str(error)})
    return exported, errors


def execute_structural_research(agent, arguments, laboratory, question):
    """Research author-selected structures using paid rows and the existing engine."""
    if question is None:
        raise ValueError('research_update requires a named scientific question')
    path = arguments['path']
    previous = agent.relation_research.get(question['id'])
    binding = _binding(agent, path, laboratory, question, previous)
    available = [{'index': index} for record in agent.observations
                 for index in record.get('row_indices', [])]
    budget = SearchBudget(min(50000, max(0, agent.config.compute_budget - agent.costs['compute_cost'])))
    started = monotonic()
    try:
        result = research_relation(binding, available, previous, compute_budget=budget)
    finally:
        agent.costs['compute_cost'] += budget.spent
        agent.costs['python_execution_seconds'] += monotonic() - started
    round_index = 0 if previous is None else previous['round_index'] + 1
    bank_index = list(agent.relation_research).index(question['id']) if previous is not None else len(agent.relation_research)
    result.update(binding_path=path, round_index=round_index,
                  round_history=deepcopy(previous.get('round_history', [])) if previous else [])
    exported, errors = _export_models(agent, path, result, bank_index, round_index)
    result.update(exported_models=exported, export_errors=errors)
    result['round_history'].append({key: deepcopy(result[key]) for key in
        ('round_index', 'binding_path', 'status', 'diagnosis', 'compute_cost', 'exported_models')})
    agent.relation_research[question['id']] = result
    candidates = result['candidates']
    survivor = next((item for item in candidates if item['status'] == 'plausible'), None)
    answer = {'operator': 'research_update', 'status': result['status'],
              'answer': {'program': result.get('selected_program') or (survivor['program'] if survivor else None),
                         'candidates': structural_summary(result)['candidates'],
                         'diagnosis': result['diagnosis'], 'models': deepcopy(exported)},
              'evidence': deepcopy(result['evidence']),
              'scope': 'Author bound relation; not independent parent confirmation.'}
    agent._return_structural_answer(question, answer)
    feedback = {'operator': 'research_update', 'research_question_id': question['id'],
                'binding_path': path, 'relation_research': structural_summary(result),
                'returned_answer': answer}
    probe = result.get('selected_probe')
    if probe:
        feedback['next_action'] = {'action': 'query', 'arguments': {
            'question': {'id': question['id']}, 'actions': deepcopy(probe['actions']),
            'after_observation': [{'operator': 'research_update', 'path': path,
                                   'research_question': {'id': question['id']}}]}}
    agent._prepare_program_state()
    return feedback


def native_global_misfit_review(agent, path):
    """Return unused measured-search options when a live method stops after rejection.

    The finite researcher already produced these experiments and their refitting
    continuations. This only reconnects its actual paid-data failure to the author;
    it neither selects an experiment nor refutes the entire model family.
    """
    rows = len(agent.rows)
    threshold = agent.config.confirmation_mse_threshold
    retained = agent.evaluated_model
    retained_mse = retained.get('current_mse') if retained else None
    if (retained and retained.get('evaluated_rows') == rows
            and isinstance(retained_mse, (int, float)) and retained_mse <= threshold):
        return None
    evaluated = next((entry for entry in reversed(agent.method_results)
        if entry.get('path') == path
        and (entry.get('proposal') or {}).get('action') == 'evaluate'), None)
    if evaluated is None:
        return None
    feedback = evaluated['feedback']
    mse = feedback.get('mse')
    if feedback.get('returncode') != 0 or not isinstance(mse, (int, float)) or mse <= threshold:
        return None
    if feedback.get('paid_residuals', {}).get('evaluated_rows', rows) != rows:
        return None
    model = evaluated['proposal'].get('arguments', {}).get('model', 'model.py')
    rounds = [(question, result) for question in getattr(agent, 'questions', [])
              for result in question.get('native_condition_research', [])]
    selected = next(((question, result) for question, result in reversed(rounds)
        if any(candidate.get('model') == model for candidate in result['candidates'])), None)
    if selected is None:
        return None
    question, result = selected
    # The stored query continuations bind the next record indices. Never offer
    # those bindings after intervening measurements or a subset-only fit.
    if (result['observed_record_count'] != len(agent.observations)
            or sorted(result['fit_evidence']) != list(range(rows))):
        return None
    paid_actions = [record['action'] for record in agent.observations]
    remaining = agent.config.observation_budget - agent.costs['observation_cost']
    experiments = []
    for index, proposal in enumerate(result['proposed_experiments']):
        query = proposal['query_action']
        actions = query['arguments'].get('actions', [])
        already_paid = any(paid_actions[start:start + len(actions)] == actions
                           for start in range(len(paid_actions)))
        if (not actions or already_paid or proposal['observation_cost'] > remaining
                or not query['arguments'].get('after_observation')):
            continue
        experiments.append({'proposal_index': index,
            'observation_cost': proposal['observation_cost'],
            'query_action': deepcopy(query),
            'matched_action_pairs': deepcopy(proposal['matched_action_pairs'])})
    if not experiments:
        return None
    return {'kind': 'global_candidate_misfit', 'path': path,
        'question_id': question['id'], 'round_index': result['round_index'],
        'model': model, 'mse': mse, 'mse_threshold': threshold,
        'current_paid_rows': rows, 'evaluated_method_step': evaluated['step'],
        'local_evidence': [{key: deepcopy(item[key]) for key in
            ('control', 'status', 'scope', 'reference_record_index', 'record_index',
             'matched_pairs', 'reference_record_indices', 'record_indices',
             'reference_evidence', 'evidence') if key in item} for item in result['evidence']],
        'available_experiments': experiments,
        'reason': 'This evaluated candidate fails on the complete current paid evidence. '
                  'The local matched measurements do not establish its applicability to all rows. '
                  'The same research round returned these unpurchased experiments with refitting '
                  'continuations. Choose whether to use one, revise the model or method, or leave '
                  'the question unresolved; this result does not reject the entire model family.'}


def _native_continuation(agent, question, result):
    """Expose a real same-bank search without replaying old experiment bindings."""
    previous = next((entry for entry in reversed(agent.operator_results)
        if entry.get('operator') == 'native_condition_research'
        and entry.get('research_question_id') == question['id']
        and entry.get('native_condition_research', {}).get('round_index', -1) <= result['round_index']
        and 'native_condition_research' in entry
        and 'boundary_update' not in entry['arguments']), None)
    if previous is None:
        return {}
    arguments = previous['arguments']
    observed = result['observed_record_count']
    paid_rows = sum(len(record['row_indices']) for record in agent.observations[:observed])
    full_scope = ('record_indices' not in arguments
                  and sorted(result['fit_evidence']) == list(range(paid_rows)))
    view = {'fit_scope': 'all_paid_rows' if full_scope else 'selected_records',
        'original_evidence_selection': {key: deepcopy(arguments[key]) for key in
            ('record_indices', 'matched_pairs') if key in arguments},
        'current_paid_rows': len(agent.rows), 'observed_record_count': observed}
    if full_scope:
        view['resume_action'] = {'action': 'run', 'arguments': {
            'operator': 'native_condition_research', 'research_question': {'id': question['id']},
            **{key: deepcopy(arguments[key]) for key in
               ('basis', 'programs', 'max_candidates', 'max_compute') if key in arguments}}}
        if (result['status'] == 'search_incomplete' or
                agent.knowledge.get('finite_native_author') and paid_rows < len(agent.rows)):
            retained = {Program.from_dict(item['program']) for old in
                question.get('native_condition_research', []) for item in old['candidates']}
            reserve = _native_completion_reserve(agent, question,
                len(retained) + len(arguments.get('programs', []))
                + int(arguments.get('max_candidates', 8)))
            # On new evidence use the current all-paid bank, rather than replay
            # the old matched-pair indices from the earlier query continuation.
            retry = (view['resume_action']['arguments']
                     if paid_rows < len(agent.rows) else arguments)
            unit = _native_compute_continuation(agent, question, result, retry, reserve)
            view['compute_continuation'] = unit
            view['complete_research_unit'] = deepcopy(unit['research_unit'])
            view.pop('resume_action')
            if 'action' in unit:
                view['resume_action'] = deepcopy(unit['action'])
    return view


def native_evidence_extension_review(agent, path):
    """Hand a stale stopped method back with its original bank's current-data entry."""
    rows = len(agent.rows)
    threshold = agent.config.confirmation_mse_threshold
    retained = agent.evaluated_model
    if (retained and retained.get('evaluated_rows') == rows
            and isinstance(retained.get('current_mse'), (int, float))
            and retained['current_mse'] <= threshold):
        return None
    evaluated = next((entry for entry in reversed(agent.method_results)
        if entry.get('path') == path
        and (entry.get('proposal') or {}).get('action') == 'evaluate'), None)
    if evaluated is None:
        return None
    feedback = evaluated['feedback']
    old_rows = feedback.get('paid_residuals', {}).get('evaluated_rows')
    mse = feedback.get('mse')
    if (feedback.get('returncode') != 0 or not isinstance(mse, (int, float))
            or mse <= threshold or not isinstance(old_rows, int) or old_rows >= rows):
        return None
    model = evaluated['proposal'].get('arguments', {}).get('model', 'model.py')
    selected = next(((question, result) for question in reversed(agent.questions)
        for result in reversed(question.get('native_condition_research', []))
        if any(candidate.get('model') == model for candidate in result['candidates'])), None)
    if selected is None:
        return None
    question, result = selected
    continuation = _native_continuation(agent, question, result)
    if (sorted(result['fit_evidence']) != list(range(old_rows))
            or 'resume_action' not in continuation):
        return None
    candidate = next(item for item in result['candidates'] if item.get('model') == model)
    if any(item['program'] == candidate['program']
           and item.get('fit_evidence') == list(range(rows))
           and isinstance(item.get('fit_mse'), (int, float))
           for later in question['native_condition_research'][result['round_index'] + 1:]
           for item in later['candidates']):
        return None  # The old bank has actually been reassessed on these rows.
    return {'kind': 'research_evidence_extended', 'path': path,
        'question_id': question['id'], 'round_index': result['round_index'],
        'model': model, 'previous_mse': mse, 'previous_paid_rows': old_rows,
        'current_paid_rows': rows, 'new_paid_rows': rows - old_rows,
        'evaluated_method_step': evaluated['step'],
        'resume_action': continuation['resume_action'],
        'original_evidence_selection': continuation['original_evidence_selection'],
        'reason': 'The stopped method evaluated this candidate before the newer paid evidence. '
            'Its old result does not assess the current rows. This action revisits the same '
            'retained candidate bank with the original search options and all current paid rows; '
            'old matched-query indices are not replayed. Choose whether to use it, revise the '
            'method or explanation, or leave the question unresolved.'}


def native_condition_method(agent):
    """Present the retained researcher through the existing method decision view."""
    question = next((q for q in agent.questions if q['id'] == agent._active_question_id), None)
    if question is None:
        return None  # A scientific action needs an existing author/task question.
    executions = []
    for previous_question in agent.questions:
        for result in previous_question.get('native_condition_research', []):
            executions.append({
                'question_id': previous_question['id'], 'round_index': result['round_index'],
                'status': result['status'], 'paid_rows': len(result['fit_evidence']),
                'models': deepcopy(result['exported_models']),
                **({'completion_reserve': deepcopy(result['completion_reserve'])}
                   if 'completion_reserve' in result else {}),
                **{key: deepcopy(result[key]) for key in
                   ('research_decision', 'remaining_boundary', 'search_stages',
                    'compute_continuation', 'search_allowance_estimate') if key in result},
                **_native_continuation(agent, previous_question, result),
                'source': {'file': 'research_state.json', 'json_path': [
                    'questions', agent.questions.index(previous_question),
                    'native_condition_research', result['round_index']]}})
    return {
        'id': 'native_condition_research', 'kind': 'retained_finite_research_operator',
        'finite_prior': FINITE_PRIOR,
        'requires': 'Public scalar controls and complete purchased reset trajectories.',
        'question_id': question['id'], 'current_paid_rows': len(agent.rows),
        'execute_action': {'action': 'run', 'arguments': {
            'operator': 'native_condition_research', 'research_question': {'id': question['id']}}},
        'current_executions': executions,
        'next_step': 'Choose this paid-evidence search, revise its scope or basis, or use ordinary '
            'Python. Its returned evaluate_action tests a frozen predictor; normal finish requests '
            'independent confirmation. This method has no current-task support before execution.'}


def native_condition_summary(entry, question_index):
    """Keep callable decisions inline without copying every search recipe."""
    path = ['questions', question_index, 'native_condition_research', entry['round_index']]
    summary = {key: deepcopy(value) for key, value in entry.items() if key != 'candidates'}
    summary.update(full_content_file='research_state.json', json_path=path)
    candidates = []
    for index, candidate in enumerate(entry['candidates']):
        item = {key: deepcopy(value) for key, value in candidate.items()
                if key != 'research_program'}
        if 'research_program' in candidate:
            recipe = candidate['research_program']
            item['research_program'] = {
                'selection_steps': [{key: deepcopy(step[key]) for key in (
                    'operation', 'gate_coordinate', 'gate_coordinates', 'threshold',
                    'thresholds', 'threshold_interval', 'threshold_intervals',
                    'width', 'placement', 'logic', 'history_bound', 'loss') if key in step}
                    for step in recipe.get('selection_steps', [])],
                'full_content_file': 'research_state.json',
                'json_path': [*path, 'candidates', index, 'research_program'],
            }
        candidates.append(item)
    summary['candidates'] = candidates
    return summary


def _register_native_theories(agent, question, question_index, round_index):
    """Index actual bank cells under the author's typed contract, without copying them."""
    from .question_routing import register_theories

    entry = question['native_condition_research'][round_index]
    base = ['questions', question_index, 'native_condition_research', round_index]
    target = question.get('theory_target') or {}
    roles = target.get('role_signature')
    roles = roles if isinstance(roles, dict) else {}
    registered, missing = [], []
    for index, candidate in enumerate(entry['candidates']):
        program = Program.from_dict(candidate['program'])
        features = sorted({name for name, _ in program.observed_variables()})
        required = [key for key in ('output_unit', 'context_ref') if target.get(key) is None]
        for name in features:
            contract = roles.get(name)
            for key in ('role', 'unit', 'time'):
                if not isinstance(contract, dict) or contract.get(key) is None:
                    required.append(f'role_signature.{name}.{key}')
        source = {'kind': 'native_candidate', 'question_id': question['id'],
            'meaning_version': question.get('meaning_version', 1),
            'round_index': round_index, 'candidate_index': index}
        program_ref = {'file': 'research_state.json',
                       'json_path': [*base, 'candidates', index, 'program']}
        if required:
            missing.append({'source_ref': source, 'required': required})
            continue
        identifier = (f'native:{question["id"]}@meaning{source["meaning_version"]}'
                      f':round{round_index}:candidate{index}')
        declaration = {'id': identifier, 'source_ref': source,
            'program_source_ref': program_ref,
            'role_signature': {name: deepcopy(roles[name]) for name in features},
            'output_unit': deepcopy(target['output_unit']),
            'context_ref': deepcopy(target['context_ref']),
            'support': {'status': 'candidate', 'independent_confirmation': False}}
        for key in ('context_family_ref', 'declared_scope_ref', 'measured_coverage_ref'):
            if key in target:
                declaration[key] = deepcopy(target[key])
        if 'parameter_paths' in candidate:
            declaration['parameter_paths'] = deepcopy(candidate['parameter_paths'])
        for key in ('failure_refs', 'counterexample_refs'):
            if candidate.get(key):
                declaration[key] = deepcopy(candidate[key])
        if candidate.get('status') == 'refuted':
            declaration['support']['status'] = 'refuted'
        register_theories(agent, [declaration])
        registered.append({'theory_id': identifier,
            'version': agent.theory_index[identifier]['version'],
            'source_ref': deepcopy(source), 'program_source_ref': program_ref})
    result = {'status': ('partially_registered' if registered and missing else
                        'author_work_request' if missing else
                        'registered' if registered else 'no_candidates'),
        'registered': registered, 'new_compute_cost': 0, 'support_updated': False}
    if missing:
        result['author_work_request'] = {'type': 'AuthorWorkRequest',
            'status': 'author_work_request', 'object_type': 'native_theory_contract',
            'question_id': question['id'],
            'meaning_version': question.get('meaning_version', 1),
            'reason': 'typed_native_candidate_contract_required',
            'required_contract': {'question_field': 'theory_target', 'candidates': missing},
            'source_ref': {'file': 'research_state.json', 'json_path': base}}
    return result


def _native_completion_reserve(agent, question, max_exported_models):
    """Use the existing full-pool delivery quote for this finite task."""
    from .finite_native_author import ordinary_completion_reserve

    rows = len(agent.rows)
    deliveries = len(agent._inquiry_stack)
    stale = bool(agent.evaluated_model is not None
                 and agent.evaluated_model['evaluated_rows'] != rows)
    reserve = {'compute_cost': rows * (2 + deliveries + int(stale))
               + agent.config.confirmation_rows,
        'paid_rows': rows, 'candidate_evaluations': 1, 'local_deliveries': deliveries,
        'retained_comparisons': int(stale), 'parent_submissions': 1,
        'confirmation_rows': agent.config.confirmation_rows,
        'scope': 'one_candidate_delivery_and_original_finish'}
    complete = ordinary_completion_reserve(agent, question)
    if complete is not None:
        reserve.update(compute_cost=complete['compute_cost'] + rows * max_exported_models,
            candidate_evaluations=complete['models'] + max_exported_models,
            retained_pool_models=complete['models'],
            max_exported_models=max_exported_models,
            comparison_compute_cost=rows * (complete['models'] + max_exported_models + int(stale)),
            finish_compute_cost=rows * (deliveries + 1) + agent.config.confirmation_rows,
            export_compute_cost=0,
            scope='full_retained_pool_and_bounded_exports_then_original_finish',
            export_cost_scope='Native source export has no separate compute charge; '
                'host execution time remains charged by the existing caller.')
    return reserve


def _structure_preparation_quote(agent, question, public_spec, *, source_round_index, control,
                                 background, matched_pairs=(), future_shape=None,
                                 witness_count=2):
    """Price the related old constraints before adding a possible new pair."""
    source = question['native_condition_research'][source_round_index]
    scope = {'control': control, 'background': background,
        'record_indices': list(dict.fromkeys([*source['fit_record_indices'],
            *(pair[key] for pair in matched_pairs
              for key in ('reference_record_index', 'record_index'))]))}
    pairs = accumulate_matched_pairs(public_spec, agent.observations,
                                    matched_pairs, scope)
    upper = matched_assessment_compute_upper(public_spec, agent.observations, pairs)
    matched_rows = sum(pair['matched_rows'] for pair in pairs)
    future_rows = 0
    if future_shape is not None:
        future_rows = future_shape['matched_rows']
        rows = [row for pair in pairs for row in
            agent.observations[pair['record_index']]['rows'][:pair['matched_rows']]]
        history = max((len(row.get('history', [])) for row in rows), default=0)
        levels = {row['features'][control] for row in rows}
        level_count = len(levels | set(future_shape['levels']))
        upper = matched_assessment_shape_upper(matched_rows + future_rows,
            max(history, future_shape['max_history']), level_count)
    retained = {Program.from_dict(candidate['program']) for old in
        question['native_condition_research'] for candidate in old['candidates']}
    scan = sum(len(record['rows']) for record in agent.observations) + future_rows
    rescoring = len(agent.rows) * len(retained)
    checks = 4 * witness_count
    # Planning has at most two contrasted instances per witness. Before the
    # actual forecast exists, price the most expensive distinct saved inputs;
    # this is a structural upper bound, not four new search procedures.
    sources = []
    for evidence in source.get('evidence', []):
        if evidence.get('control') != control or evidence.get('background') != background:
            continue
        for item in _structural_hypotheses(source, evidence):
            index = item['candidate_index']
            sources.append({**source['candidates'][index], 'source_ref': {
                'file': 'research_state.json', 'json_path': ['questions',
                    agent.questions.index(question), 'native_condition_research',
                    source_round_index, 'candidates', index]}})
    prices = retained_structure_refit_quote(sources, len(agent.rows))
    selected_prices = sorted(prices['structures'], key=lambda item: item['compute_upper'],
                             reverse=True)[:2 * witness_count]
    refit_upper = sum(item['compute_upper'] for item in selected_prices)
    return {'compute_cost': scan + rescoring + upper + checks + refit_upper,
        'matched_assessment_compute_upper': upper, 'archive_scan_compute': scan,
        'retained_program_scoring_compute': rescoring, 'frozen_readout_compute': checks,
        'retained_structure_refit_compute_upper': refit_upper,
        'retained_structure_refit_max_count': len(selected_prices),
        'related_paid_pairs': len(pairs), 'related_matched_rows': matched_rows,
        'future_matched_rows': future_rows, 'source_round_index': source_round_index,
        'scope': 'Input-shape bound for matched preparation; joint search remains a bounded estimate.'}


def _native_compute_continuation(agent, question, result, arguments, reserve, *,
                                 structural_preparation=None, public_spec=None):
    """Publish a whole payable restart, never a smaller unfinished fragment."""
    rows = len(agent.rows)
    old_rows = len(result['fit_evidence'])
    measured_preparation = sum(item['compute_cost'] for item in result['search_stages']
                               if not item['stage'].startswith('joint_'))
    extended = rows != old_rows
    preparation = ((measured_preparation * rows + old_rows - 1) // old_rows
                   if old_rows else 0) + 2 * rows
    target = arguments.get('structural_discrimination')
    if structural_preparation is None and target is not None:
        structural_preparation = _structure_preparation_quote(agent, question,
            public_spec if public_spec is not None else agent._method_public_spec,
            source_round_index=target['source_round_index'], control=target['control'],
            background=target['background'], matched_pairs=arguments['matched_pairs'],
            witness_count=len(target['forecast_witnesses']))
    if structural_preparation is not None:
        preparation = max(preparation, structural_preparation['compute_cost'])
    estimate = (conditional_search_allowance(len(agent.rows[0].features), rows,
                    max(len(row.history) for row in agent.rows)) if extended
                else result['search_allowance_estimate']['conditional_joint_attempt'])
    blocked = max((item['blocked_work']['additional_compute']
                   for item in result['search_stages'] if 'blocked_work' in item), default=0)
    minimum = max((0 if extended else result['completion_reserve']['search_allowance'] + blocked),
                  preparation + estimate)
    remaining = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
    available = max(0, remaining - reserve['compute_cost'])
    seconds = max(0., agent.config.execution_seconds
                  - agent.costs.get('python_execution_seconds', 0.))
    payable = available >= minimum and seconds > 0.
    unit = {'status': ('available' if payable else 'execution_time_exhausted'
                      if seconds <= 0. else 'shared_compute_insufficient'),
        'available_compute_after_completion_reserve': available,
        'estimated_search_allowance': minimum,
        'remaining_execution_seconds': seconds,
        'research_unit': {'search_preparation_compute': preparation,
            'bounded_joint_attempt_estimate': estimate,
            'search_compute': minimum, 'completion_reserve': deepcopy(reserve),
            'minimum_compute': minimum + reserve['compute_cost'],
            'quoted_paid_rows': rows, 'previous_paid_rows': old_rows,
            'work_reason': 'new_paid_evidence' if extended else 'search_incomplete',
            'preparation_basis': 'previous_actual_preparation_rescaled_to_current_rows',
            'resumption': 'bounded_restart_with_retained_programs_not_an_internal_fit_cursor',
            'previous_paid_compute': result['compute_cost'],
            'estimate_is_completion_guarantee': False},
        'scope': 'Bounded same-bank restart with retained Programs and recipes. Local assessment '
            'and old-program scoring repeat and remain charged; the quoted joint attempt is '
            'an estimate, not a guarantee. The complete pool comparison and finish stay reserved; '
            'no new total compute or execution window is created.'}
    if structural_preparation is not None:
        unit['research_unit']['structural_preparation'] = deepcopy(structural_preparation)
        unit['research_unit']['preparation_basis'] = 'max_of_measured_preparation_and_accumulated_input_shape'
    if payable:
        retry = {key: deepcopy(arguments[key]) for key in (
            'operator', 'record_indices', 'matched_pairs', 'basis', 'programs',
            'max_candidates', 'structural_discrimination', 'structural_question_id') if key in arguments}
        retry.update(question={'id': question['id']}, max_compute=minimum)
        unit['action'] = {'action': 'run', 'arguments': retry}
    return unit


def _structure_readout(agent, question, plan, proposal, pairs, budget):
    """Keep the paid contrast once, before a later fit can be interrupted."""
    readings = plan.setdefault('readout_results', [])
    existing = next((entry for entry in readings
        if entry['matched_pairs'] == pairs and entry['verdict'].get('complete')), None)
    charged = 0
    if existing is None:
        before = agent.costs['compute_cost']
        verdict = assess_frozen_structure_predictions(agent,
            proposal['structural_discrimination'], pairs,
            planned_actions=proposal['query_action']['arguments']['actions'],
            max_compute=budget.remaining)
        charged = agent.costs['compute_cost'] - before
        budget.spend(charged)
        existing = {'matched_pairs': deepcopy(pairs), 'verdict': verdict}
        readings.append(existing)
    verdict = existing['verdict']
    reference = {'file': 'research_state.json', 'json_path': ['questions',
        agent.questions.index(question), 'native_structure_planning',
        question['native_structure_planning'].index(plan), 'readout_results',
        readings.index(existing), 'verdict']}
    summary = {key: deepcopy(verdict[key]) for key in (
        'status', 'complete', 'required_compute', 'reason', 'evidence',
        'refuted_candidate_refs', 'compatible_candidate_refs',
        'structure_families_refuted', 'next_research') if key in verdict}
    summary.update(result_ref=reference, cache_hit=charged == 0 and verdict.get('complete', False))
    return summary, charged


def _frozen_structure_refit_inputs(question, target):
    """Refit actual forecast witnesses, without expanding the whole old bank."""
    previous = question['native_condition_research'][target['source_round_index']]
    result, seen = [], set()
    for witness in target['forecast_witnesses'].values():
        for index in witness['hypothesis_indices']:
            structure = target['structures'][index]
            key = structure['width'], structure['placement']
            if key in seen:
                continue
            seen.add(key)
            reference = target['hypothesis_refs'][index]
            candidate = previous['candidates'][reference['json_path'][-1]]
            result.append({**candidate, 'source_ref': deepcopy(reference)})
    return result


def execute_native_condition_research(agent, arguments, laboratory, question):
    """Run the existing finite scientific inner loop on this question's paid data."""
    if question is None:
        raise ValueError('native_condition_research requires an authored research question')
    if 'structure_planning' in arguments:
        return _execute_native_structure_planning(agent, arguments, laboratory, question)
    rounds = question.setdefault('native_condition_research', [])
    structure_target = arguments.get('structural_discrimination')
    structure_plan, structure_proposal, structure_child, accumulate_scope = None, None, None, None
    research_contract = {key: deepcopy(arguments[key]) for key in
        ('record_indices', 'basis', 'programs') if key in arguments}
    research_contract['max_candidates'] = int(arguments.get('max_candidates', 8))
    if structure_target is not None:
        research_contract['refit_scope'] = 'retained_coordinates'
        plans = question.get('native_structure_planning', [])
        matches = [(plan, proposed) for plan in plans if plan.get('planning_complete')
            and plan.get('question_meaning_version') == question.get('meaning_version', 1)
            for proposed in plan.get('proposed_experiments', [])
            if proposed.get('structural_discrimination') == structure_target]
        if (not matches or structure_target.get('source_question_ref') != {
                'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)}):
            raise ValueError('structural analysis requires its saved forecast/planning result')
        structure_plan, structure_proposal = matches[-1]
        structure_child = next((item for item in agent.questions
            if item.get('id') == arguments.get('structural_question_id')
                and item.get('structure_target') == structure_target), None)
        original = rounds[structure_target['source_round_index']]
        pairs = arguments.get('matched_pairs', [])
        accumulate_scope = {'control': structure_target['control'],
            'background': deepcopy(structure_target['background']),
            'record_indices': list(dict.fromkeys([*original['fit_record_indices'],
                *(pair[key] for pair in pairs for key in ('reference_record_index', 'record_index'))]))}
        completed = next((entry for entry in reversed(rounds)
            if entry.get('structural_discrimination', {}).get('target') == structure_target
            and entry['structural_discrimination'].get('matched_pairs') == pairs
            and entry['structural_discrimination'].get('research_contract') == research_contract
            and entry.get('observed_record_count') == len(agent.observations)
            and entry.get('status') in {'candidates_require_evaluation', 'matched_evidence_without_candidate',
                                       'frozen_predictions_compared'}), None)
        if completed is not None:
            return {'operator': 'native_condition_research', 'status': completed['status'],
                'research_question_id': question['id'], 'cache_hit': True,
                'native_condition_research': native_condition_summary(completed, agent.questions.index(question))}
    remaining = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
    # As in instrument_action, leave the actual scoring path outside search.
    # An explicit evaluation is followed by each local delivery, parent finish,
    # and fresh confirmation. Comparing a stale retained model costs one more
    # paid-data pass. The ordinary evaluators charge these units when executed.
    rows = len(agent.rows)
    retained = []
    if 'boundary_update' not in arguments:
        for previous in rounds:
            for candidate in previous['candidates']:
                old = next((item for item in retained
                            if item['program'] == candidate['program']), None)
                if old is None:
                    retained.append(deepcopy(candidate))
                else:
                    old.update(deepcopy(candidate))
    reserve = _native_completion_reserve(agent, question,
        len(retained) + len(arguments.get('programs', [])) + int(arguments.get('max_candidates', 8)))
    reserved = reserve['compute_cost']
    search_allowance = max(0, remaining - reserved)
    budget = SearchBudget(min(search_allowance,
                              max(0, int(arguments.get('max_compute', search_allowance)))))
    options = {key: arguments[key] for key in
               ('record_indices', 'matched_pairs', 'basis', 'max_candidates') if key in arguments}
    started, readout_charged = monotonic(), 0
    forecast_verdict = None
    try:
        observation_budget = max(0,
            agent.config.observation_budget - agent.costs['observation_cost'])
        if structure_target is not None:
            forecast_verdict, readout_charged = _structure_readout(
                agent, question, structure_plan, structure_proposal, pairs, budget)
            if structure_child is not None:
                agent._return_structural_answer(structure_child, {
                    'operator': 'native_condition_research',
                    'status': 'answered' if forecast_verdict['complete'] else 'inconclusive',
                    'answer': {'target': deepcopy(structure_target),
                        'matched_pairs': deepcopy(pairs), 'forecast_verdict': deepcopy(forecast_verdict)},
                    'evidence': deepcopy(forecast_verdict.get('evidence', [])),
                    'scope': 'Frozen fitted-predictor contrast answered before refitting; structural families remain unresolved.'})
            accumulated = accumulate_matched_pairs(laboratory.public_spec(),
                agent.observations, pairs, accumulate_scope)
            options['accumulate_matched_scope'] = accumulate_scope
            if forecast_verdict['next_research']['kind'] == 'jointly_refit_frozen_structures':
                options['refit_candidates'] = _frozen_structure_refit_inputs(question, structure_target)
                options['refit_cache'] = structure_plan.setdefault('refit_results', [])
        if (forecast_verdict is not None and forecast_verdict['next_research']['kind']
                != 'jointly_refit_frozen_structures'):
            selected = options.get('record_indices', list(range(len(agent.observations))))
            incomplete = not forecast_verdict['complete']
            unfunded = forecast_verdict.get('reason') == 'complete_frozen_readout_check_unfunded'
            result = {'status': ('search_incomplete' if unfunded else
                'frozen_readout_incomplete' if incomplete else 'frozen_predictions_compared'),
                'finite_prior': FINITE_PRIOR, 'candidates': [], 'proposed_experiments': [],
                'evidence': [], 'available_pairs': accumulated,
                'accumulated_matched_scope': {**deepcopy(accumulate_scope), 'matched_pairs': accumulated},
                'fit_record_indices': selected,
                'fit_evidence': [row for index in selected for row in agent.observations[index]['row_indices']],
                'observed_record_count': len(agent.observations), 'searched_candidates': 0,
                'search_stages': [{'stage': 'frozen_prediction_readout',
                    'status': 'computation_incomplete' if incomplete else 'attempt_finished',
                    'compute_cost': readout_charged, 'completed_fits': 0,
                    **({'blocked_work': {'additional_compute': forecast_verdict['required_compute']}}
                       if unfunded else {})}],
                'search_allowance_estimate': {'conditional_joint_attempt': conditional_search_allowance(
                    len(agent.rows[0].features), rows, max(len(row.history) for row in agent.rows))},
                'compute_cost': budget.spent,
                'research_decision': {**deepcopy(forecast_verdict['next_research']),
                    'computation_incomplete': incomplete, 'frozen_prediction_status': forecast_verdict['status']}}
        elif 'boundary_update' in arguments:
            selected = arguments['boundary_update']
            previous = rounds[selected['round_index']]
            proposal = previous['proposed_experiments'][selected['proposal_index']]
            result = update_native_boundary(laboratory.public_spec(), agent.observations,
                budget, previous=previous, proposal=proposal,
                paid_record_indices=selected.get('paid_record_indices', ()),
                max_candidates=arguments.get('max_candidates', 8),
                observation_budget=observation_budget)
        else:
            result = research_native_conditions(laboratory.public_spec(), agent.observations,
                budget, programs=arguments.get('programs', []), retained_candidates=retained,
                observation_budget=observation_budget, **options)
            if forecast_verdict is not None:
                result['compute_cost'] += readout_charged
                result['search_stages'].insert(0, {'stage': 'frozen_prediction_readout',
                    'status': 'attempt_finished', 'compute_cost': readout_charged,
                    'completed_fits': 0})
    finally:
        agent.costs['compute_cost'] += budget.spent - readout_charged
        agent.costs['python_execution_seconds'] += monotonic() - started
    result['completion_reserve'] = {**reserve,
        'search_allowance': budget.limit,
        'remaining_compute': remaining - budget.spent,
        'completion_funded': remaining - budget.spent >= reserved,
    }
    if structure_target is not None:
        result['structural_discrimination'] = {'target': deepcopy(structure_target),
            'research_contract': research_contract,
            'matched_pairs': deepcopy(arguments.get('matched_pairs', [])),
            'forecast_verdict': deepcopy(forecast_verdict),
            'accumulated_matched_scope': deepcopy(result.get('accumulated_matched_scope')),
            'measured_evidence': [{key: deepcopy(item[key]) for key in
                ('control', 'background', 'status', 'width_candidates', 'placement_candidates',
                 'conditional_loss', 'record_indices', 'record_index', 'response_supported',
                 'condition_identified', 'competing_structures') if key in item}
                for item in result.get('evidence', [])
                if item.get('control') == structure_target['control']
                    and item.get('background') == structure_target['background']],
            'scope': 'Measured alternatives after the planned switch; global validity still requires ordinary comparison.'}
        if forecast_verdict['next_research']['kind'] == 'jointly_refit_frozen_structures':
            loss = min((candidate['fit_mse'] for candidate in result['candidates']), default=None)
            result['structural_discrimination']['refit_return'] = {
                'status': ('computation_incomplete' if result['status'] == 'search_incomplete' else
                    'candidate_supported_on_paid_rows' if loss is not None and loss <= 1e-10 else
                    'no_supported_candidate_in_completed_finite_attempt'),
                'best_fit_mse': loss, 'structure_families_refuted': False,
                'next_research': ('complete_quoted_research' if result['status'] == 'search_incomplete' else
                    'compare_full_retained_pool' if loss is not None and loss <= 1e-10 else
                    'revise_structure_or_representation')}
            if result['status'] != 'search_incomplete' and (loss is None or loss > 1e-10):
                result['research_decision'] = {
                    'kind': 'revise_structure_or_representation',
                    'reason': 'The frozen-coordinate refits and bounded new response search '
                              'still fail the paid evidence; a boundary result is not the only research action.',
                    'boundary_decision': result['research_decision']}
    if result['status'] == 'search_incomplete':
        result['compute_continuation'] = _native_compute_continuation(
            agent, question, result, arguments, reserve, public_spec=laboratory.public_spec())
        result['research_decision']['compute_continuation'] = deepcopy(result['compute_continuation'])
    if result['status'] == 'research_not_executed':
        retry = {key: deepcopy(value) for key, value in arguments.items()
                 if key not in {'matched_pairs', 'question', 'research_question'}}
        retry['question'] = {'id': question['id']}
        return {'operator': 'native_condition_research',
                'research_question_id': question['id'],
                **{key: deepcopy(result[key]) for key in (
                    'status', 'reason', 'error', 'invalid_pairs', 'available_pairs',
                    'compute_cost', 'completion_reserve')},
                'automatic_matching_action': {'action': 'run', 'arguments': retry}}
    index = next(i for i, item in enumerate(agent.questions) if item['id'] == question['id'])
    execution_index = len(rounds)
    execution = Path('_experiment_models') / f'native_conditions_{index:04d}_{execution_index:04d}'
    # Bank chronology is local to this question; inherited exports and reopened
    # banks may already occupy the same file names. Keep those versions intact.
    while (agent.workspace / execution).exists():
        execution_index += 1
        execution = Path('_experiment_models') / f'native_conditions_{index:04d}_{execution_index:04d}'
    directory = execution / '_evaluated_model'
    (agent.workspace / directory).mkdir(parents=True)
    models = []
    for candidate_index, candidate in enumerate(result['candidates']):
        model = directory / f'candidate_{candidate_index:04d}.py'
        source = export_temporal_program(Program.from_dict(candidate['program']),
                                        feature_names=laboratory.public_spec()['controls'])
        (agent.workspace / model).write_text(source, encoding='utf-8')
        candidate.update(model=str(model), evaluate_action={
            'action': 'evaluate', 'arguments': {'model': str(model)}})
        models.append({key: deepcopy(candidate[key]) for key in
                       ('model', 'evaluate_action', 'fit_mse', 'source')})
    continuation = {'operator': 'native_condition_research',
                    'research_question': {'id': question['id']},
                    **{key: deepcopy(arguments[key]) for key in
                       ('basis', 'max_candidates', 'max_compute') if key in arguments}}
    for proposal_index, proposed in enumerate(result['proposed_experiments']):
        query = proposed['query_action']['arguments']
        query['question'] = {'id': question['id']}
        resumed = deepcopy(continuation)
        if proposed.get('purpose') == 'conditional_boundary':
            resumed['boundary_update'] = {
                'round_index': len(rounds), 'proposal_index': proposal_index}
        first = len(agent.observations)
        if proposed['matched_action_pairs']:
            resumed['matched_pairs'] = [
                {'reference_record_index': first + pair['reference_action_index'],
                 'record_index': first + pair['action_index']}
                for pair in proposed['matched_action_pairs']]
        elif 'matched_pairs' in arguments:
            resumed['matched_pairs'] = deepcopy(arguments['matched_pairs'])
        if 'record_indices' in arguments:
            resumed['record_indices'] = list(dict.fromkeys([
                *arguments['record_indices'], *range(first, first + len(query['actions']))]))
        if agent.knowledge.get('finite_native_author') and 'boundary_update' not in resumed:
            # The finite author prices the current complete joint unit after
            # acquisition. Do not silently run its old short search ceiling
            # before that same action-selection step can choose the allocation.
            proposed['research_continuation'] = resumed
            query.pop('after_observation', None)
        else:
            query['after_observation'] = [resumed]
    if result.get('research_decision', {}).get('kind') == 'remaining_parameter_ambiguity':
        boundary = next(item for item in result['proposed_experiments']
                        if item.get('purpose') == 'conditional_boundary')
        result['research_decision']['next_experiment'] = deepcopy(boundary['query_action'])
    result['round_index'] = len(rounds)
    result['exported_models'] = models
    rounds.append(deepcopy(result))
    registration = _register_native_theories(agent, question, index, result['round_index'])
    result['native_theory_registration'] = registration
    rounds[-1]['native_theory_registration'] = deepcopy(registration)
    answer = {'operator': 'native_condition_research', 'status': 'inconclusive',
        'answer': {'status': result['status'], 'finite_prior': result['finite_prior'],
                   'models': deepcopy(models), 'matched_evidence': deepcopy(result['evidence']),
                   'native_theory_registration': deepcopy(registration),
                   **{key: deepcopy(result[key]) for key in
                      ('research_decision', 'remaining_boundary', 'search_stages',
                       'compute_continuation', 'search_allowance_estimate') if key in result}},
        'evidence': list(result['fit_evidence']),
        'scope': 'Finite-prior paid search; parent adoption and independent confirmation remain pending.'}
    agent._return_structural_answer(question, answer)
    if structure_target is not None:
        if structure_child is not None:
            agent._return_structural_answer(structure_child, {'operator': 'native_condition_research',
                'status': 'answered' if forecast_verdict['complete'] else 'inconclusive',
                'answer': deepcopy(result['structural_discrimination']),
                'evidence': list(result['fit_evidence']),
                'scope': 'Return measured structural alternatives to the declared parent revision; no unique world mechanism is asserted.'})
    return {'operator': 'native_condition_research', 'status': result['status'],
            'research_question_id': question['id'],
            'native_theory_registration': registration,
            'native_condition_research': native_condition_summary(result, index),
            'returned_answer': answer}


def _execute_native_structure_planning(agent, arguments, laboratory, question):
    """A bounded design action over a saved bank, without replaying its fits."""
    requested = arguments['structure_planning']
    rounds = question.get('native_condition_research', [])
    index = requested.get('source_round_index')
    if type(index) is not int or not rounds or index != rounds[-1]['round_index']:
        raise ValueError('structure planning requires the current native source round')
    previous = rounds[-1]
    contract = native_structure_planning_contract(laboratory.public_spec(), agent.observations, previous)
    if requested != contract:
        raise ValueError('saved structural alternatives or planning scope changed')
    plans = question.setdefault('native_structure_planning', [])
    source = {'file': 'research_state.json', 'json_path': ['questions',
        agent.questions.index(question), 'native_condition_research', index]}
    existing = next((item for item in plans if item.get('planning_complete')
        and item.get('question_meaning_version') == question.get('meaning_version', 1)
        and item.get('basis') == requested), None)
    if existing is not None:
        return {'operator': 'native_condition_research', 'status': 'structure_planning_reused',
            'research_question_id': question['id'], 'structure_planning': deepcopy(existing),
            'source_ref': {'file': 'research_state.json', 'json_path': ['questions',
                agent.questions.index(question), 'native_structure_planning', plans.index(existing)]},
            'cache_hit': True, 'compute_cost': 0}
    remaining = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
    reserve = _native_completion_reserve(agent, question, 0)
    available = max(0, remaining - reserve['compute_cost'])
    budget = SearchBudget(min(available, max(0, int(arguments.get('max_compute',
        requested['proposal_compute'])))))
    started = monotonic()
    try:
        result = plan_native_structure_experiments(laboratory.public_spec(), agent.observations,
            budget, previous=previous, requested=requested, source_ref=source)
    finally:
        agent.costs['compute_cost'] += budget.spent
        agent.costs['python_execution_seconds'] += monotonic() - started
    result['planning_index'] = len(plans)
    result['question_meaning_version'] = question.get('meaning_version', 1)
    result['completion_reserve'] = reserve
    for proposed in result['proposed_experiments']:
        continuation = proposed['research_continuation']
        continuation.update(research_question={'id': question['id']},
            max_candidates=int(arguments.get('max_candidates', 8)))
        for key in ('basis',):
            if key in arguments:
                continuation[key] = deepcopy(arguments[key])
        target = proposed['structural_discrimination']
        target['source_question_ref'] = {'question_id': question['id'],
            'meaning_version': question.get('meaning_version', 1)}
        continuation['structural_discrimination'] = deepcopy(target)
        child = {'id': f"{question['id']}:history-structure:{index}:{result['planning_index']}",
            'question': 'Which frozen candidate contrasts match the measured switch, and what '
                'remains unresolved about history width and condition placement?',
            'kind': 'structural_discrimination', 'answer_type': 'binding',
            'parent_id': question['id'], 'parent_question_ref': {'question_id': question['id'],
                'meaning_version': question.get('meaning_version', 1)},
            'context_ref': deepcopy(question.get('context_ref')),
            'subject_refs': deepcopy(target['hypothesis_refs']),
            'structure_target': deepcopy(target),
            'answer_resolution_ref': {'kind': 'retained_forecast_contrast', 'atol': 1e-10},
            'use_sites': [{'kind': 'refit_parent_from_matched_evidence', 'parent_id': question['id'],
                'parent_model': (agent.evaluated_model or {}).get('model')}],
            'answer_method': {'operator': 'native_condition_research',
                'matched_pairs': deepcopy(continuation['matched_pairs'])}}
        proposed['query_action'].update(research_question={'id': question['id']},
            structural_discrimination=deepcopy(target))
        proposed['query_action']['arguments']['question'] = child
        continuation['structural_question_id'] = child['id']
        result['research_problem'] = deepcopy(child)
    plans.append(deepcopy(result))
    return {'operator': 'native_condition_research', 'status': result['status'],
        'research_question_id': question['id'], 'structure_planning': result,
        'source_ref': {'file': 'research_state.json', 'json_path': ['questions',
            agent.questions.index(question), 'native_structure_planning', result['planning_index']]},
        'compute_cost': budget.spent,
        'scope': 'Saved-fit planning only; the proposed query, refitting and full-pool comparison remain unexecuted.'}

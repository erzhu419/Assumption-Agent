"""Connect paid method uses, condition research and bounded step revisions.

Feedback projection reads retained results. Explicitly funded step questions
reuse their original split and fits, testing actual replay edits through the
existing fitter. Frozen source methods and their paid records stay available.
"""

from copy import deepcopy
from math import isfinite
from pathlib import Path

from .checkpoint_storage import clone_or_copy, write_json_checkpoint
from .language import Program
from .method_conditions import method_profile, method_use_decision
from .models import Observation, TaskSpec
from .research_episodes import load_source_values
from .research_methods import observe_common_method_return


def _read(agent, reference):
    return load_source_values([reference], agent.workspace)[0]['value']


def _store(agent, name, value, *, base=None):
    path = agent.workspace / 'method_feedback' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_checkpoint(path, value, base=base, ensure_ascii=False, indent=None)
    return {'file': str(path), 'json_path': []}


def _rows(agent):
    return [row if hasattr(row, 'features') else Observation(row['features'], row['target'],
        tuple(row.get('history', []))) for row in agent.rows]


def begin_method_feedback(agent, laboratory):
    """Capture public conditions before a method or comparison can change them."""
    library = agent.theory_consolidation
    if laboratory is None or not any(item.get('kind') == 'research_program'
            for item in library.get('candidates', {}).values()):
        return None
    public = laboratory.public_spec()
    if 'trajectory' not in public.get('action_kinds', []):
        return None
    controls = tuple(item['name'] if isinstance(item, dict) else item for item in public['controls'])
    spec = TaskSpec('public_method_use', 'Already acquired public research conditions',
        controls, (), (), feature_bounds={name: tuple(bounds) for name, bounds in
            public.get('feature_bounds', {}).items()}, max_probe_length=public.get('max_probe_length', 0))
    retained = agent._evaluated_summary() or {}
    owner = next((q for q in agent.questions if q['id'] == agent._active_question_id), {})
    parent = library_parent = (owner.get('theory_target') or {}).get('parent_theory_id')
    parent = agent.theory_index.get(parent, {}) if parent else {}
    program = Program.from_dict(parent['program']) if parent.get('program') else None
    profile = method_profile(spec, _rows(agent), program,
        residual_loss=retained.get('mse') if retained.get('evaluated_rows') == len(agent.rows) else None,
        minimum_improvement=agent.config.confirmation_mse_threshold)
    profile['representation'].pop('program', None)
    profile['data']['controllable'] = True
    if program is None and retained:
        profile['representation']['kind'] = 'not_exposed'
    return {'profile': profile, 'retained_model': deepcopy(retained),
        'parent_theory_id': library_parent,
        'question_ref': {'question_id': owner.get('id'),
            'meaning_version': owner.get('meaning_version', 1)},
        'theory_target': deepcopy(owner.get('theory_target', {})),
        'costs': dict(agent.costs)}


def _feedback_rows(agent, method_id):
    refs = agent.theory_consolidation.get('method_feedback', {}).get(method_id, {}).get('outcome_refs', [])
    rows = [{**_read(agent, ref), 'feedback_ref': deepcopy(ref)} for ref in refs]
    for row in rows:
        row['evidence_basis'] = method_evidence_basis(row['evidence_basis'])
        if row.get('process_comparison_ref'):
            row['process_comparison'] = _read(agent, row['process_comparison_ref'])
    return rows


def _condition_outcomes(agent, reference):
    outcomes = []
    for item in _read(agent, reference):
        outcome = _read(agent, item) if 'file' in item and 'method_ref' not in item else deepcopy(item)
        if 'evidence_basis_ref' in outcome:
            outcome['evidence_basis'] = _read(agent, outcome['evidence_basis_ref'])
        outcome['evidence_basis'] = method_evidence_basis(outcome['evidence_basis'])
        if 'profile' not in outcome:
            outcome['profile'] = _read(agent, outcome['profile_ref'])
        if outcome.get('process_comparison_ref'):
            outcome['process_comparison'] = _read(agent, outcome['process_comparison_ref'])
        outcomes.append(outcome)
    return outcomes


def _condition_state(outcome):
    """Freeze changed decision evidence; keep source bindings and costs once."""
    state = {key: deepcopy(outcome[key]) for key in (
        'method_ref', 'profile_ref', 'feedback_ref', 'status', 'execution_complete',
        'execution_error', 'administrative_censoring', 'censored',
        'working_comparison', 'paired_future', 'process_comparison_ref') if key in outcome}
    if 'process_comparison' in outcome and 'process_comparison_ref' not in outcome:
        state['process_comparison'] = deepcopy(outcome['process_comparison'])
    state['evidence_basis_ref'] = {**deepcopy(outcome['feedback_ref']),
        'json_path': [*outcome['feedback_ref'].get('json_path', []), 'evidence_basis']}
    return state


def _paid_method_evidence(agent, binding, outcomes=()):
    """Retain the actual paid fitting view, sharing an existing equal view."""
    paid = [{"features": dict(row.features if hasattr(row, 'features') else row['features']),
        "target": row.target if hasattr(row, 'target') else row['target'],
        "history": [dict(point) for point in
            (row.history if hasattr(row, 'history') else row.get('history', []))]}
        for row in agent.rows]
    library = agent.theory_consolidation
    retained = list(outcomes)
    if not retained:
        candidate = library.get('candidates', {}).get(binding.get('candidate_id'), {})
        conditions = candidate.get('method_use_conditions')
        if conditions:
            retained.extend(_condition_outcomes(agent, conditions['outcomes_ref']))
        retained.extend(_feedback_rows(agent, binding.get('candidate_id')))
    references = []
    for outcome in retained:
        reference = outcome.get('evidence_basis', {}).get('paid_evidence_ref')
        if reference and reference not in references:
            references.append(reference)
    directory = agent.workspace / 'method_feedback'
    local = sorted(directory.glob('paid_*.json'), reverse=True)
    references.extend({'file': str(path), 'json_path': []} for path in local
                      if {'file': str(path), 'json_path': []} not in references)
    for reference in references:
        if _read(agent, reference) == paid:
            return deepcopy(reference)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'paid_{len(local):06d}.json'
    source = agent.workspace / 'observations.json'
    # The workspace file may still contain a selected or previous fitting view.
    # Only borrow its blocks after checking it against the actual paid rows.
    if source.is_file() and _read(agent, {'file': str(source), 'json_path': []}) == paid:
        clone_or_copy(source, path)
    else:
        base = references[0]['file'] if references else None
        write_json_checkpoint(path, paid, base=base, ensure_ascii=False, indent=None)
    return {'file': str(path), 'json_path': []}


def method_evidence_basis(binding, agent=None):
    # A new version containing only learned usage conditions is not new paid
    # evidence. Preserve the actual mapping, split and executable fit contract.
    basis = {key: deepcopy(value) for key, value in binding.items()
             if key not in {'candidate_id', 'candidate_version', 'library_path', 'source_ref', 'seed_index'}}
    if agent is not None:
        basis['paid_evidence_ref'] = _paid_method_evidence(agent, binding)
    return basis


def procedure_use_decision(agent, candidate, binding):
    conditions = candidate.get('method_use_conditions')
    profile = getattr(agent, '_method_use_profile', None)
    if not conditions or profile is None:
        return None
    outcomes = _condition_outcomes(agent, conditions['outcomes_ref'])
    basis = method_evidence_basis(binding)
    basis['paid_evidence_ref'] = _paid_method_evidence(agent, binding, outcomes)
    # Imported or restored workspaces can reference equal contents at different
    # paths. Align only verified equal views, without editing frozen outcomes.
    paid = _read(agent, basis['paid_evidence_ref'])
    outcomes = deepcopy(outcomes)
    for outcome in outcomes:
        outcome['evidence_basis'] = method_evidence_basis(outcome['evidence_basis'])
        prior = outcome.get('evidence_basis', {}).get('paid_evidence_ref')
        if prior and prior != basis['paid_evidence_ref'] and _read(agent, prior) == paid:
            outcome['evidence_basis']['paid_evidence_ref'] = deepcopy(basis['paid_evidence_ref'])
    research = _read(agent, conditions['research_ref']) if conditions.get('research_ref') else None
    return method_use_decision(conditions['usage_version'], profile, outcomes,
        evidence_basis=basis, conflict_research=research)


def _profile_ref(agent, profile):
    directory = agent.workspace / 'method_feedback'
    for path in sorted(directory.glob('profile_*.json')):
        reference = {'file': str(path), 'json_path': []}
        if _read(agent, reference) == profile:
            return reference
    return _store(agent, f'profile_{len(list(directory.glob("profile_*.json"))):06d}.json', profile)


def _models(agent, identities):
    selected = []
    for question in agent.questions:
        for export in question.get('revision_candidate_exports', []):
            reference = export.get('theory_ref') or {}
            if reference in identities and export.get('model') not in selected:
                selected.append(export['model'])
    return selected


def _observe_return(agent, group, record, reference):
    value = {**record, 'feedback_ref': record.get('origin_ref', reference),
        'profile': record.get('profile') or _read(agent, record['profile_ref'])}
    if record.get('process_comparison_ref'):
        value['process_comparison'] = _read(agent, record['process_comparison_ref'])
    group['returns'] = observe_common_method_return(value, group.get('returns'))


def _replace_feedback_record(agent, group, reference, record):
    """Update a restored use locally while retaining its paid origin identity."""
    source = Path(reference['file'])
    source = source if source.is_absolute() else agent.workspace / source
    directory = agent.workspace / 'method_feedback'
    if source.parent == directory and not reference.get('json_path'):
        return _store(agent, source.name, record)
    record.setdefault('origin_ref', deepcopy(reference))
    current = _store_new_use(agent, record)
    group['outcome_refs'][group['outcome_refs'].index(reference)] = current
    return current


def _store_new_use(agent, record):
    directory = agent.workspace / 'method_feedback'
    number = sum(len(item.get('outcome_refs', []))
                 for item in agent.theory_consolidation.get('method_feedback', {}).values())
    while (directory / f'use_{number:06d}.json').exists():
        number += 1
    return _store(agent, f'use_{number:06d}.json', record)


def publish_method_use_questions(agent, proposals, *, binding, context):
    """Publish unresolved method decisions into the existing question agenda."""
    published = []
    supplied = context.get('question_ref') or {}
    owner = next((q for q in agent.questions if q['id'] == supplied.get('question_id')), {})
    parent = {'question_id': supplied.get('question_id'),
        'meaning_version': supplied.get('meaning_version', owner.get('meaning_version', 1))}
    evidence = (method_evidence_basis(context['evidence_basis']) if context.get('evidence_basis') is not None
                else method_evidence_basis(binding, agent=agent))
    mission = next((value for value in (context.get('mission'), owner.get('mission'))
        if value and value.get('purpose') in {'method_learning', 'theory_learning'}), None)
    for proposal in proposals:
        references = deepcopy(proposal.get('evidence_refs', []))
        returns = []
        for reference in references:
            record = _read(agent, reference)
            retained = {key: deepcopy(record[key]) for key in
                ('method_ref', 'profile_ref', 'status', 'execution_complete', 'execution_error',
                 'administrative_censoring', 'censored') if key in record}
            for field, keys in (
                ('working_comparison', ('complete', 'same_rows', 'before_loss', 'after_loss', 'data_role',
                                        'rows', 'requested_pool_size')),
                ('paired_future', ('before_loss', 'after_loss', 'reference_rows', 'same_rows', 'data_role',
                    'source_refs', 'execution_complete', 'administrative_censoring', 'censored',
                    'response_row_range'))):
                if field in record:
                    retained[field] = {key: deepcopy(record[field][key]) for key in keys if key in record[field]}
            if 'process_comparison' in record:
                retained['process_comparison'] = deepcopy(record['process_comparison'])
            if 'process_comparison_ref' in record:
                retained['process_comparison_ref'] = deepcopy(record['process_comparison_ref'])
            returns.append(retained)
        agenda_basis = {'evidence_basis': deepcopy(evidence), 'method_feedback_refs': references,
                        'method_returns': returns}
        old = next((q for q in agent.questions if q.get('kind') == 'method_use'
            and q.get('parent_question_ref') == parent
            and q.get('method_use_plan', {}).get('candidate_id') == binding['candidate_id']
            and q.get('agenda_basis', {}).get('evidence_basis') == evidence), None)
        if old is not None:
            changed = old.get('agenda_basis') != agenda_basis
            old.update(deepcopy(proposal), agenda_basis=agenda_basis,
                theory_target=deepcopy(context.get('theory_target', owner.get('theory_target', {}))),
                method_use_plan={'candidate_id': binding['candidate_id'], 'binding': deepcopy(binding)})
            if changed:
                old['status'] = 'open'
            if mission is not None:
                old['mission'] = mission
            published.append({'question_id': old['id'],
                'meaning_version': old.get('meaning_version', 1)})
            continue
        question = {**deepcopy(proposal),
            'id': f"method-use:{binding['candidate_id']}:{len(agent.questions)}",
            'kind': 'method_use', 'status': 'open', 'parent_id': parent.get('question_id'),
            'queue': 'theory',
            'parent_question_ref': deepcopy(parent),
            'agenda_basis': agenda_basis,
            'theory_target': deepcopy(context.get('theory_target', owner.get('theory_target', {}))),
            'method_use_plan': {'candidate_id': binding['candidate_id'], 'binding': deepcopy(binding)},
            'use_sites': [{'kind': 'select_research_action', 'question_id': parent.get('question_id')}]}
        current = agent._open_question(question)
        if mission is not None:
            current['mission'] = mission
        published.append({'question_id': current['id'],
            'meaning_version': current.get('meaning_version', 1)})
    if published:
        _refresh_method_agenda(agent)
    return published


def _refresh_method_agenda(agent):
    from .question_agenda import build_agenda
    agent.research_agenda = build_agenda(agent.questions, agent.research_candidates,
        agent.theory_index, mission=agent.research_mission,
        budget={'compute': max(0, agent.config.compute_budget - agent.costs['compute_cost']),
            'author_calls': 0})


def _paired_return(agent, record, evaluations):
    """Read two unchanged frozen predictors on the same newly paid rows."""
    working = record.get('working_comparison') or {}
    cutoff = record['evidence_basis']['paid_cutoff']['rows']
    if len(agent.rows) <= cutoff:
        return None
    paid = _read(agent, record['evidence_basis']['paid_evidence_ref'])
    current = [{'features': row.features, 'target': row.target, 'history': list(row.history)}
               for row in _rows(agent)]
    if current[:cutoff] != paid:
        return None
    from .code_agent import _closed_temporal_export, python_model_context_matches
    from .temporal_operator_agent import _closed_prediction_values
    paired, references = [], []
    for side in ('before', 'after'):
        prior_ref = working.get(f'{side}_score_ref')
        if not prior_ref:
            return None
        prior = _read(agent, prior_ref)
        context = prior['execution_context']
        selected = None
        matching = [item for item in evaluations if item.get('path') == working.get(f'{side}_model')
            or item.get('paid_score_reuse', {}).get('matched_model') == working.get(f'{side}_model')]
        for item in matching:
            reference = item.get('paid_score_reuse', {}).get('execution_basis_ref')
            if not reference or item.get('returncode') != 0 or item.get('timeout'):
                continue
            assessment = _read(agent, reference)
            later = assessment['execution_context']
            # The existing literal export reads only its paid row. New host
            # observations and feedback are not predictor dependencies; actual
            # Program/source changes still reject the old method attribution.
            closed = (_closed_temporal_export(context['workspace'], context['model'])
                and _closed_temporal_export(later['workspace'], later['model']))
            same = ((Path(context['workspace']) / context['model']).read_bytes() ==
                (Path(later['workspace']) / later['model']).read_bytes()) if closed else (
                python_model_context_matches(later['workspace'], later['model'],
                    context['workspace'], context['model']))
            if same:
                selected = (reference, assessment)
                break
        if selected is None or not selected[1].get('closed_prediction_state'):
            return None
        reference, assessment = selected
        values = _closed_prediction_values({**reference, 'json_path': ['closed_prediction_state']},
            agent.workspace)['predictions']
        if len(values) != len(current):
            return None
        paired.append(sum((prediction - row['target']) ** 2 for prediction, row in
            zip(values[cutoff:], current[cutoff:])) / (len(current) - cutoff))
        references.append(reference)
    return {'before_loss': paired[0], 'after_loss': paired[1],
        'reference_rows': len(current) - cutoff, 'same_rows': True,
        'data_role': 'new_paid_readings', 'response_row_range': [cutoff, len(current) - 1],
        'source_refs': references, 'execution_complete': True}


def finish_method_feedback(agent, context, action, arguments, result):
    """Associate actual uses, complete comparisons and later confirmations."""
    if context is None:
        return
    library = agent.theory_consolidation
    feedback = library.setdefault('method_feedback', {})
    changed = False
    for key in dict.fromkeys(getattr(agent, '_method_execution_writes', [])):
        execution = library['executions'][key]
        binding = execution['basis']['binding']
        output = execution['result']
        identifier = binding['candidate_id']
        group = feedback.setdefault(identifier, {'outcome_refs': []})
        basis = method_evidence_basis(binding, agent=agent)
        candidate = library['candidates'][identifier]
        identity = candidate.get('method_use_conditions', {}).get('usage_version',
            {'id': identifier, 'version': binding['candidate_version']})
        previous = next((item for item in _feedback_rows(agent, identifier)
            if item['evidence_basis'] == basis and item['method_ref'] == identity), None)
        if previous is not None and (output.get('cache_hit') or output.get('fit_cache_hit')):
            continue
        profile_ref = _profile_ref(agent, context['profile'])
        complete = output.get('search_completed') is True
        record = {'method_ref': deepcopy(identity),
            'invoked_method_ref': {'id': identifier, 'version': binding['candidate_version']},
            'profile_ref': profile_ref, 'evidence_basis': basis,
            'parent_question_ref': deepcopy(context['question_ref']),
            'theory_target': deepcopy(context['theory_target']),
            'invocation_binding': deepcopy(binding),
            'status': 'pending_comparison' if complete else 'incomplete',
            'execution_complete': complete,
            'execution_ref': {'file': str(agent.workspace / 'shared/theory_library.json'),
                'json_path': ['executions', key]},
            'generated_theories': [deepcopy(item['theory_ref']) for item in
                output.get('generated_theories', []) if item.get('theory_ref')],
            'fit_compute_cost': output.get('new_compute_cost'),
            'seed_replay_work': deepcopy(output.get('seed_replay_work', [])),
            'comparison_cost_scope': 'shared_comparison_ledger_not_attributed_to_each_method',
            'cost_refs': [{'file': str(agent.workspace / 'research_state.json'), 'json_path': ['costs']}],
            'world_support_updated': False, 'independent_confirmation': False}
        if not record['generated_theories']:
            record['generated_theories'] = [{'id': item['theory_id'],
                'version': agent.theory_index[item['theory_id']]['version']}
                for item in output.get('generated_theories', []) if item.get('theory_id')]
        reference = _store_new_use(agent, record)
        group['outcome_refs'].append(reference)
        _observe_return(agent, group, record, reference)
        changed = True
    comparison = action == 'run' and arguments.get('operator') == 'research_choice'
    confirmation = result.get('fresh_confirmation')
    method_questions = []
    for identifier, group in (feedback.items() if comparison or confirmation else ()):
        for reference in group.get('outcome_refs', []):
            record = _read(agent, reference)
            models = _models(agent, record['generated_theories'])
            update = False
            if comparison and models:
                evaluations = result.get('evaluations', [])
                valid = [item for item in evaluations if item.get('returncode') == 0
                    and type(item.get('mse')) in (int, float) and isfinite(item['mse'])]
                method = [item for item in valid if item.get('path') in models]
                baseline = [item for item in valid if item.get('path') not in models]
                complete = (record['execution_complete'] and not result.get('truncated')
                    and len(valid) == len(evaluations) and not any(item.get('timeout') for item in valid)
                    and result.get('models_requested') ==
                    result.get('models_completed') and bool(method and baseline)
                    and len(agent.rows) == record['evidence_basis']['paid_cutoff']['rows'])
                if complete and not record.get('working_comparison'):
                    before = min(baseline, key=lambda item: item['mse'])
                    after = min(method, key=lambda item: item['mse'])
                    record.update(status='completed', working_comparison={
                        'complete': True, 'same_rows': True, 'before_loss': before['mse'],
                        'after_loss': after['mse'], 'data_role': 'exposed_development',
                        'before_model': before['path'], 'after_model': after['path'],
                        'before_score_ref': deepcopy(before.get('paid_score_reuse', {}).get('execution_basis_ref')),
                        'after_score_ref': deepcopy(after.get('paid_score_reuse', {}).get('execution_basis_ref')),
                        'requested_pool_size': result['models_requested'], 'rows': len(agent.rows),
                        'scope': 'complete_requested_pool_on_same_paid_rows'},
                        comparison_ref={'file': str(agent.workspace / 'research_state.json'),
                            'json_path': ['operator_results', len(agent.operator_results) - 1]})
                    update = True
                paired = _paired_return(agent, record, evaluations)
                if paired is not None and paired != record.get('paired_future'):
                    record['paired_future'] = paired
                    update = True
            if confirmation and arguments.get('model') in models:
                record['confirmation'] = {key: deepcopy(confirmation[key]) for key in
                    ('passed', 'returncode', 'timeout', 'mse', 'rows', 'observation_indices') if key in confirmation}
                record['confirmation']['paired_method_gain'] = None
                record['confirmation']['scope'] = 'submitted_predictor_only_no_before_predictor_column'
                update = True
            if update:
                reference = _replace_feedback_record(agent, group, reference, record)
                _observe_return(agent, group, record, reference)
                if record.get('working_comparison') and record.get('parent_question_ref', {}).get('question_id'):
                    method_questions.append((identifier, record, reference))
                changed = True
    for identifier, record, reference in method_questions:
        candidate = library['candidates'][identifier]
        binding = deepcopy(record.get('invocation_binding') or record['evidence_basis'])
        binding.update(candidate_id=identifier, candidate_version=candidate['version'],
            library_path='shared/theory_library.json')
        proposal = {'kind': 'method_use', 'answer_type': 'research_action',
            'question': 'How should these measured method-use results change the next research action?',
            'query_semantics': {'mode': 'solution_relation'},
            'alternatives': ['try_or_continue_method', 'ordinary_research',
                'acquire_distinguishing_evidence', 'revise_method'],
            'evidence_refs': deepcopy(feedback[identifier]['outcome_refs']),
            'parent_use': {'kind': 'select_research_action', 'method_ref': deepcopy(record['method_ref'])},
            'origin': {'kind': 'measured_method_return', 'source_refs': [deepcopy(reference)]}}
        # Publishing work spends no new allowance. The existing agenda defers
        # it unless the actual method-learning mission has remaining funding.
        result.setdefault('method_question_refs', []).extend(publish_method_use_questions(agent,
            [proposal], binding=binding, context={'question_ref': record['parent_question_ref'],
                'theory_target': record.get('theory_target', {}),
                'evidence_basis': record['evidence_basis']}))
    active = getattr(agent, '_active_method_use', None)
    if active and active.get('decision', {}).get('question') and context['question_ref'].get('question_id'):
        result['method_question_refs'] = publish_method_use_questions(agent,
            [active['decision']['question']], binding=active['binding'], context=context)
        _refresh_method_agenda(agent)
        changed = True
    if changed:
        from .theory_consolidation import _save
        _save(agent)


def record_method_execution_error(agent, context, error):
    """Keep an actual interrupted method use without teaching a negative law."""
    active = getattr(agent, '_active_method_use', None)
    if not active or context is None:
        return
    feedback = agent.theory_consolidation.setdefault('method_feedback', {})
    group = feedback.setdefault(active['candidate_id'], {'outcome_refs': []})
    record = {'method_ref': deepcopy(active['method_ref']),
        'invoked_method_ref': deepcopy(active['invoked_method_ref']),
        'profile_ref': _profile_ref(agent, context['profile']),
        'evidence_basis': method_evidence_basis(active['binding'], agent=agent),
        'status': 'execution_error', 'execution_complete': False,
        'execution_error': f'{type(error).__name__}: {error}', 'generated_theories': [],
        'fit_compute_cost': agent.costs['compute_cost'] - active['costs_before']['compute_cost'],
        'uncommitted_compute_cost': 'UNKNOWN',
        'costs_ref': {'file': str(agent.workspace / 'research_state.json'), 'json_path': ['costs']},
        'independent_confirmation': False, 'world_support_updated': False}
    reference = _store_new_use(agent, record)
    group['outcome_refs'].append(reference)
    _observe_return(agent, group, record, reference)
    from .theory_consolidation import _save
    _save(agent)


def research_method_conditions(agent, candidate_id, compute_limit, *, public_experiments=()):
    """Execute finite paid-evidence diagnostics and retain their usable result.

    This is ordinary method research under the existing mission. The original
    executable body, seed fits and strong pool remain unchanged; only an actual
    diagnostic result can produce another condition version.
    """
    from .method_conditions import research_method_conflict
    from .models import SearchBudget
    from .theory_consolidation import _save, library_entry_reference

    candidate = agent.theory_consolidation['candidates'][candidate_id]
    conditions = candidate.get('method_use_conditions')
    if not conditions:
        return {'status': 'actual_method_feedback_required', 'new_compute_cost': 0}
    previous = _read(agent, conditions['research_ref']) if conditions.get('research_ref') else None
    basis = {'method_ref': deepcopy(conditions['usage_version']),
        'outcomes_ref': deepcopy(conditions['outcomes_ref']),
        'public_experiments': deepcopy(list(public_experiments))}
    same = previous is not None and previous.get('execution_basis') == basis
    if same and (not previous.get('budget_incomplete')
            or compute_limit <= previous.get('available_compute', 0)):
        return {**deepcopy(previous), 'cache_hit': True, 'new_compute_cost': 0,
            'research_ref': deepcopy(conditions['research_ref'])}
    outcomes = _condition_outcomes(agent, conditions['outcomes_ref'])
    budget = SearchBudget(max(0, compute_limit))
    result = research_method_conflict(conditions['usage_version'], outcomes, budget,
        target_profile=getattr(agent, '_method_use_profile', None),
        public_experiments=public_experiments, previous=previous if same else None)
    agent.costs['compute_cost'] += budget.spent
    result = {**result, 'execution_basis': basis, 'available_compute': compute_limit,
        'cache_hit': False, 'new_compute_cost': budget.spent}
    if result['status'] in {'no_mixed_returns', 'insufficient_method_feedback'}:
        return result
    files = sorted((agent.workspace / 'method_feedback').glob('condition_research_*.json'))
    saved = _store(agent, f'condition_research_{len(files):06d}.json', result,
        base=files[-1] if files else None)
    if result.get('budget_incomplete'):
        # Preserve a failed/partial attempt without presenting it as learned
        # knowledge. A larger *existing* allowance can resume the same question.
        question = next((item for item in agent.questions
            if item.get('id') == agent._active_question_id), None)
        if question is not None:
            question['method_conflict_attempt_ref'] = saved
        return {**result, 'research_ref': saved}
    old = library_entry_reference(agent, 'candidates', candidate_id)
    path = Path(old['file'])
    if (path if path.is_absolute() else agent.workspace / path) == agent.workspace / 'shared/theory_library.json':
        files = sorted((agent.workspace / 'method_feedback').glob('method_*.json'))
        old = _store(agent, f'method_{len(files):06d}.json', candidate,
            base=files[-1] if files else None)
    revised = deepcopy(candidate)
    revised.update(version=candidate['version'] + 1, previous_version_ref=old)
    revised['method_use_conditions']['research_ref'] = saved
    revised['trigger']['learned_utility_trigger'] = bool(result.get('condition')) or bool(
        revised['trigger'].get('learned_utility_trigger'))
    agent.theory_consolidation['candidates'][candidate_id] = revised
    _save(agent)
    return {**result, 'research_ref': saved,
        'frozen_candidate': {'candidate_id': candidate_id, 'version': revised['version'],
            'previous_version_ref': old}}


def method_step_research_inputs(agent, candidate_id, feedback_ref, *, max_candidates=8, quote_ref=None):
    """Quote actual replay edits on the original paid split, without refitting."""
    from .research_episodes import project_method_paid_step_evidence
    from .research_programs import quote_research_step_revisions, research_program_replay_inputs

    evidence = project_method_paid_step_evidence(feedback_ref, agent.workspace)
    result = {'status': evidence['status'], 'evidence': evidence, 'seeds': [],
        'required_compute': 0, 'coordinate_readout_compute': 0,
        'max_candidates': max_candidates, 'new_fits': 0, 'new_observations': 0}
    if evidence['status'] != 'completed':
        return result
    candidate = agent.theory_consolidation['candidates'][candidate_id]
    if (evidence['method_ref']['id'] != candidate_id or
            evidence['invocation_binding']['source_refs'] != candidate['research_program_source_refs']):
        return {**result, 'status': 'original_method_body_required'}
    paid = _read(agent, evidence['paid_rows_ref'])
    split = {name: [Observation(dict(paid[index]['features']), paid[index]['target'],
                    tuple(paid[index].get('history', []))) for index in value['row_indices']]
             for name, value in evidence['split_rows'].items()}
    mapping = evidence['invocation_binding']['mapping']
    source_basis = [deepcopy(feedback_ref), deepcopy(evidence['paid_rows_ref']),
                   {'fit_split': deepcopy(evidence['fit_split'])}]
    paid_quote = _read(agent, quote_ref) if quote_ref else None
    if paid_quote and (paid_quote['feedback_ref'] != feedback_ref
                       or paid_quote['max_candidates'] != max_candidates):
        return {**result, 'status': 'step_quote_paid_basis_changed'}
    for origin in evidence['source_method_payload_refs']:
        if origin['input_mapping'] is None:
            return {**result, 'status': 'source_role_mapping_required'}
        source = _read(agent, origin['source_ref'])
        features = {name: mapping[role] for role, name in origin['input_mapping'].items()}
        seed_basis = [*source_basis, deepcopy(origin['source_ref'])]
        saved = next((seed for seed in paid_quote['seeds']
            if seed['source_ref'] == origin['source_ref']), None) if paid_quote else None
        if saved:
            if (saved['source_basis'] != seed_basis or saved['feature_mapping'] != features
                or saved['quote']['methods'][0]['replay_inputs'] != research_program_replay_inputs(
                    source['research_program'], feature_mapping=features)):
                return {**result, 'status': 'step_quote_paid_basis_changed'}
            quote = deepcopy(saved['quote'])
        else:
            quote = quote_research_step_revisions(source['research_program'], split['training'],
                split['holdout'], feature_mapping=features, feature_names=tuple(mapping.values()),
                source_ref=origin['source_ref'], max_candidates=max_candidates)
            result['coordinate_readout_compute'] += quote.get('coordinate_readout_compute', 0)
        retained = next((item for item in evidence.get('paid_seed_baselines', [])
                         if item['source_ref'] == origin['source_ref']), None)
        baseline = {}
        if retained:
            baseline = {key: deepcopy(retained[key]) for key in ('execution_complete',
                'program', 'program_ref', 'training_loss', 'validation_loss', 'fit_compute_cost',
                'cost_ref', 'exact_program_deduplication') if key in retained}
            baseline['basis'] = {'replay_inputs': research_program_replay_inputs(
                source['research_program'], feature_mapping=features), 'source_refs': seed_basis,
                'training_rows': len(split['training']), 'validation_rows': len(split['holdout']),
                'risk_quantile': .5}
            baseline['source_plan_ref'] = deepcopy(retained['research_program_ref'])
            baseline['research_program_scope'] = retained['research_program_scope']
        result['seeds'].append({'source_ref': deepcopy(origin['source_ref']),
            'payload': source['research_program'], 'feature_mapping': features, 'quote': quote,
            'source_basis': seed_basis, 'paid_baseline': baseline})
    if any(not seed['paid_baseline'] for seed in result['seeds']):
        return {**result, 'status': 'exact_original_seed_baseline_required'}
    if any(seed['quote'].get('per_method_compute_upper') is None for seed in result['seeds']):
        return {**result, 'status': 'complete_step_budget_unresolved'}
    ceiling = max(seed['quote']['per_method_compute_upper'] for seed in result['seeds'])
    result.update(status='quoted', per_method_compute=ceiling,
        required_compute=sum((len(seed['quote']['methods']) - 1) * ceiling +
            (0 if seed['paid_baseline'].get('validation_loss') is not None else len(split['holdout']))
            for seed in result['seeds']),
        training=split['training'], validation=split['holdout'], paid_quote_ref=deepcopy(quote_ref))
    return result


def research_method_steps(agent, candidate_id, feedback_ref, compute_limit, *,
                          max_candidates=8, previous_ref=None, prepared_inputs=None, quote_ref=None):
    """Research a failed method's consumed inputs through Common's own route.

    Only a completed paired edit can replace its executable method body. The
    old method, strong predictive archive and all unsuccessful interventions
    remain addressable; a local split never becomes independent support.
    """
    from .models import SearchBudget
    from .research_programs import evaluate_research_step_revisions, research_program_replay_inputs
    from .research_drafts import _rebind_step
    from .theory_consolidation import _save

    candidate = agent.theory_consolidation['candidates'][candidate_id]
    if (previous_ref and candidate.get('method_step_revision', {}).get('study_ref') == previous_ref):
        previous = _read(agent, previous_ref)
        return {**previous, 'study_ref': deepcopy(previous_ref), 'cache_hit': True,
            'new_compute_cost': 0, 'frozen_candidate': {'candidate_id': candidate_id,
                'version': candidate['version'], 'previous_version_ref': candidate['previous_version_ref']}}
    context = prepared_inputs if prepared_inputs is not None else method_step_research_inputs(
        agent, candidate_id, feedback_ref, max_candidates=max_candidates, quote_ref=quote_ref)
    if context['status'] != 'quoted':
        return {key: value for key, value in context.items() if key not in {'seeds', 'training', 'validation'}}
    baseline_ref = {'id': candidate_id, 'version': candidate['version']}
    basis = {'baseline_method_ref': baseline_ref, 'feedback_ref': deepcopy(feedback_ref),
        'per_method_compute': context['per_method_compute'], 'max_candidates': max_candidates,
        'seed_inputs': [seed['quote'].get('basis', seed['source_basis']) for seed in context['seeds']]}
    previous = _read(agent, previous_ref) if previous_ref else None
    if previous and previous.get('execution_basis') != basis:
        previous = None
    if previous and not previous.get('budget_incomplete'):
        return {**previous, 'study_ref': deepcopy(previous_ref), 'cache_hit': True, 'new_compute_cost': 0}
    required = context['required_compute']
    if previous:
        for seed, study in zip(context['seeds'], previous['seed_studies']):
            required -= sum(context['per_method_compute'] if method['index'] else
                (0 if seed['paid_baseline'].get('validation_loss') is not None
                 else len(context['validation'])) for method in study['methods']
                if method.get('execution_complete'))
    if compute_limit < required:
        return {'status': 'insufficient_complete_step_budget', 'required_compute': required,
            'available_compute': compute_limit, 'budget_incomplete': True, 'new_compute_cost': 0,
            'execution_basis': basis, 'independent_support': False}
    budget, studies = SearchBudget(compute_limit), []
    files = sorted((agent.workspace / 'method_feedback').glob('step_study_*.json'))
    study_name = f'step_study_{len(files):06d}.json'
    previous_paid = previous.get('total_new_compute_cost', previous.get('new_compute_cost', 0)) if previous else 0
    for index, seed in enumerate(context['seeds']):
        before_seed = budget.spent

        def retain_progress(partial):
            # Commit each completed paid slot through the shared incremental
            # writer. A failed save stops the next fit rather than losing it.
            _store(agent, study_name, {'status': 'incomplete_step_comparison',
                'execution_basis': basis, 'baseline_method_ref': baseline_ref,
                'feedback_ref': deepcopy(feedback_ref), 'seed_studies': [*studies, partial],
                'new_compute_cost': before_seed + partial['compute_cost'],
                'total_new_compute_cost': previous_paid + before_seed + partial['compute_cost'],
                'execution_complete': False, 'budget_incomplete': True,
                'time_incomplete': partial.get('time_incomplete', False),
                'previous_costs_ref': deepcopy(previous_ref), 'data_role': 'exposed_development',
                'independent_support': False, 'support_updated': False, 'new_observations': 0})

        studies.append(evaluate_research_step_revisions(seed['payload'], context['training'],
            context['validation'], budget, per_method_compute=context['per_method_compute'],
            feature_mapping=seed['feature_mapping'],
            feature_names=tuple(context['evidence']['invocation_binding']['mapping'].values()),
            source_refs=seed['source_basis'], source_ref=seed['source_ref'],
            max_candidates=max_candidates,
            previous=(previous['seed_studies'][index]
                if previous and index < len(previous['seed_studies']) else None),
            paid_baseline=seed['paid_baseline'],
            on_progress=retain_progress,
            minimum_improvement=agent.config.confirmation_mse_threshold))
        if studies[-1].get('time_incomplete'):
            break
    agent.costs['compute_cost'] += budget.spent
    # An edit is a shared research method only when every retained seed has
    # actually tested it, without a measured holdout regression on this split.
    def actual_edit(edit):
        # Graph node names and trace positions are provenance, not different
        # executable steps. Both seeds have already been bound to public roles.
        return {key: deepcopy(edit[key]) for key in ('kind', 'term', 'before') if key in edit}

    complete = len(studies) == len(context['seeds']) and all(study.get('methods') and
        {item['index'] for item in study['methods']} == set(range(len(study['proposal']['candidates']) + 1))
        and all(item['execution_complete'] for item in study['methods'])
        and not study.get('time_incomplete') and not study['budget_incomplete'] for study in studies)
    common = []
    eligible_pairs = studies[0]['pairs'] if complete else []
    for pair in eligible_pairs:
        matches = [next((item for item in study['pairs']
                        if actual_edit(item['edit']) == actual_edit(pair['edit'])), None)
                   for study in studies]
        if (all(item and item['execution_complete'] and item['validation_loss_change']
                <= agent.config.confirmation_mse_threshold for item in matches)
                and any(item['status'] == 'local_step_improvement' for item in matches)):
            common.append(matches)
    chosen = min(common, key=lambda pairs: sum(studies[index]['methods'][pair['method_index']][
        'validation_loss'] for index, pair in enumerate(pairs))) if common else None
    result = {'status': ('development_step_revision_selected' if chosen else
        'incomplete_step_comparison' if not complete
        else 'complete_no_supported_step_revision'), 'execution_basis': basis,
        'baseline_method_ref': baseline_ref, 'feedback_ref': deepcopy(feedback_ref),
        'seed_studies': studies, 'new_compute_cost': budget.spent,
        'total_new_compute_cost': previous_paid + budget.spent,
        'execution_complete': complete,
        'budget_incomplete': any(study['budget_incomplete'] for study in studies),
        'time_incomplete': any(study.get('time_incomplete') for study in studies),
        'previous_costs_ref': deepcopy(previous_ref), 'data_role': 'exposed_development',
        'independent_support': False, 'support_updated': False, 'new_observations': 0}
    study_ref = _store(agent, study_name, result,
        base=agent.workspace / 'method_feedback' / study_name)
    if chosen:
        old_files = sorted((agent.workspace / 'method_feedback').glob('method_*.json'))
        old = _store(agent, f'method_{len(old_files):06d}.json', candidate,
            base=old_files[-1] if old_files else None)
        revised = deepcopy(candidate)
        revised.update(version=candidate['version'] + 1, previous_version_ref=old)
        revised.pop('method_use_conditions', None)
        revised['trigger']['learned_utility_trigger'] = False
        revised['research_program_source_refs'], revised['source_seed_inputs'] = [], []
        selected_refs = []
        for index, pair in enumerate(chosen):
            actual = studies[index]['methods'][pair['method_index']]
            reference = _store(agent, f'step_payload_{len(files):06d}_{index:03d}.json', {
                'research_program': actual['research_program'], 'program': actual['program'],
                'parent_source_ref': context['seeds'][index]['source_ref'],
                'paired_replay_ref': {**study_ref, 'json_path': ['seed_studies', index]},
                'data_role': 'exposed_development'})
            selected_refs.append(reference)
            revised['research_program_source_refs'].append(reference)
            revised['source_outcomes'][index]['source_ref'] = deepcopy(reference)
            revised['source_outcomes'][index]['input_mapping'] = deepcopy(
                context['evidence']['invocation_binding']['mapping'])
            payload_ref = {**reference, 'json_path': ['research_program']}
            revised['source_seed_inputs'].append({'source_id': candidate['source_seed_inputs'][index]['source_id'],
                'source_ref': deepcopy(reference), 'retained_term_refs': [
                    {**payload_ref, 'json_path': ['research_program', 'graph', 'nodes', key, 'answer']}
                    for key in actual['research_program']['term_nodes']]})
        canonical = {name: role for role, name in context['evidence']['invocation_binding']['mapping'].items()}
        first_payload = _read(agent, selected_refs[0])['research_program']
        revised['research_program_contract'] = research_program_replay_inputs(
            _rebind_step(first_payload, canonical))
        revised['research_program_source_ref'] = deepcopy(selected_refs[0])
        proof = _store(agent, f'step_revision_{len(files):06d}.json', {'status': 'completed',
            'data_role': 'exposed_development', 'baseline_method_ref': baseline_ref,
            'selected_method_payload_refs': selected_refs, 'paired_replay_refs': [
                {**study_ref, 'json_path': ['seed_studies', index]} for index in range(len(studies))],
            'scope': 'one_paid_target_split_not_new_independent_physical_sources'})
        revised['method_step_revision'] = {'baseline_method_ref': baseline_ref,
            'study_ref': study_ref, 'actual_step_revision_ref': proof}
        agent.theory_consolidation['candidates'][candidate_id] = revised
        result.update(frozen_candidate={'candidate_id': candidate_id, 'version': revised['version'],
            'previous_version_ref': old}, actual_step_revision_ref=proof)
        _save(agent)
    return {**result, 'study_ref': study_ref, 'cache_hit': False}


def consolidate_method_feedback(agent, arguments, allocation):
    """Freeze usage feedback as a new method version, preserving its source."""
    library = agent.theory_consolidation
    references, questions = [], []
    budget = allocation['compute']
    spent = 0
    for supplied in arguments.get('method_process_comparison_refs', []):
        reference = supplied['feedback_ref']
        record = _read(agent, reference)
        identifier = record['method_ref']['id']
        group = library.get('method_feedback', {}).get(identifier, {})
        if reference not in group.get('outcome_refs', []):
            reference = next((item for item in group.get('outcome_refs', [])
                if _read(agent, item).get('origin_ref') == reference), None)
            if reference is None:
                raise ValueError('process results must address an actual retained method use')
            record = _read(agent, reference)
        comparison = _read(agent, supplied['comparison_ref'])
        if comparison['method_ref'] != record['method_ref']:
            raise ValueError('process comparison must retain the actual method version')
        if (record.get('process_comparison_ref') and
                _read(agent, record['process_comparison_ref']) == comparison):
            continue
        if spent >= budget:
            break
        from .research_methods import common_method_process_return
        # This validates the evidence type and preserves UNKNOWN/censoring; it
        # does not turn configuration-level returns into single-method causality.
        common_method_process_return(comparison)
        files = sorted((agent.workspace / 'method_feedback').glob('process_*.json'))
        saved = _store(agent, f'process_{len(files):06d}.json', comparison,
            base=files[-1] if files else None)
        record['process_comparison_ref'] = saved
        reference = _replace_feedback_record(agent, group, reference, record)
        record = _read(agent, reference)
        _observe_return(agent, group, record, reference)
        spent += 1
    for source_ref in arguments.get('method_feedback_source_refs', []):
        from .research_episodes import project_method_use_feedback
        if spent >= budget:
            break
        known = [item for identifier in library.get('method_feedback', {})
            for item in _feedback_rows(agent, identifier)]
        if any(item.get('source_feedback_ref') == source_ref for item in known):
            continue
        record = project_method_use_feedback(source_ref, agent.workspace)
        identifier = record['method_ref']['id']
        if identifier not in library.get('candidates', {}):
            raise ValueError('method feedback must address an installed method')
        record['source_feedback_ref'] = deepcopy(source_ref)
        record['profile_ref'] = _profile_ref(agent, record.pop('profile'))
        record.pop('feedback_ref', None)
        feedback = library.setdefault('method_feedback', {})
        group = feedback.setdefault(identifier, {'outcome_refs': []})
        reference = _store_new_use(agent, record)
        group['outcome_refs'].append(reference)
        _observe_return(agent, group, record, reference)
        spent += 1
    selected = arguments.get('candidate_ids', list(library.get('method_feedback', {})))
    for identifier in selected:
        original = library.get('candidates', {}).get(identifier)
        if not original or original.get('kind') != 'research_program':
            continue
        source = original.get('method_use_conditions', {}).get('usage_version',
            {'id': identifier, 'version': original['version']})
        # A body revision is a new method. Old-body failures remain in the
        # ledger, but cannot become measured outcomes of its untried successor.
        outcomes = [outcome for outcome in _feedback_rows(agent, identifier)
                    if outcome['method_ref'] == source]
        for outcome in outcomes:
            outcome['profile'] = _read(agent, outcome['profile_ref'])
        if not outcomes:
            questions.append({'kind': 'method_use', 'method_ref': {'id': identifier,
                'version': original['version']}, 'reason': 'actual_method_use_feedback_required'})
            continue
        previous = original.get('method_use_conditions')
        prior = _condition_outcomes(agent, previous['outcomes_ref']) if previous else []
        if previous and [_condition_state(item) for item in prior] == [_condition_state(item) for item in outcomes]:
            references.append({'candidate_id': identifier, 'version': original['version'], 'cache_hit': True})
            continue
        if spent + len(outcomes) > budget:
            break
        spent += len(outcomes)
        version = original['version'] + 1
        retained = _read(agent, previous['outcomes_ref']) if previous else []
        states = []
        for outcome in outcomes:
            state = _condition_state(outcome)
            reuse = next((index for index, item in enumerate(prior) if _condition_state(item) == state), None)
            if reuse is not None:
                item = retained[reuse]
                states.append(deepcopy(item) if 'file' in item and 'method_ref' not in item else {
                    **previous['outcomes_ref'], 'json_path': [*previous['outcomes_ref'].get('json_path', []), reuse]})
            else:
                number = len(list((agent.workspace / 'method_feedback').glob('condition_use_*.json')))
                states.append(_store(agent, f'condition_use_{number:06d}.json', state))
        saved = _store(agent, f'conditions_{len(list((agent.workspace / "method_feedback").glob("conditions_*.json"))):06d}.json', states)
        from .theory_consolidation import library_entry_reference
        old = library_entry_reference(agent, 'candidates', identifier)
        if (Path(old['file']) if Path(old['file']).is_absolute() else agent.workspace / old['file']) == agent.workspace / 'shared/theory_library.json':
            files = sorted((agent.workspace / 'method_feedback').glob('method_*.json'))
            old = _store(agent, f'method_{len(files):06d}.json', original,
                base=files[-1] if files else None)
        revised = deepcopy(original)
        deciding = [method_use_decision(source, outcome['profile'], outcomes,
            evidence_basis=outcome['evidence_basis']) for outcome in outcomes]
        revised.update(version=version, previous_version_ref=old,
            method_use_conditions={'usage_version': source, 'outcomes_ref': saved,
                'scope': 'exposed_development_method_use_not_world_application'},
            trigger={'kind': 'observed_method_use_conditions', 'learned_utility_trigger':
                any(item['admission'] in {'admit', 'defer'} for item in deciding)})
        library['candidates'][identifier] = revised
        references.append({'candidate_id': identifier, 'version': version,
            'kind': 'research_program', 'previous_version_ref': old})
        for outcome in outcomes:
            decision = method_use_decision(source, outcome['profile'], outcomes,
                evidence_basis=outcome['evidence_basis'])
            if decision.get('question'):
                questions.append(decision['question'])
                binding = deepcopy(outcome.get('invocation_binding') or outcome['evidence_basis'])
                binding.update(candidate_id=identifier, candidate_version=version,
                    library_path='shared/theory_library.json')
                publish_method_use_questions(agent, [decision['question']], binding=binding,
                    context={'question_ref': outcome.get('parent_question_ref', {
                        'question_id': agent._active_question_id, 'meaning_version': 1}),
                        'evidence_basis': outcome['evidence_basis'],
                        'theory_target': outcome.get('theory_target', {})})
    agent.costs['compute_cost'] += spent
    if questions:
        _refresh_method_agenda(agent)
    from .theory_consolidation import _save
    _save(agent)
    return {'status': 'completed' if references else 'UNKNOWN', 'frozen_candidates': references,
        'method_questions': questions, 'new_compute_cost': spent,
        'support_updated': False, 'new_observations': 0, 'llm_calls': 0}

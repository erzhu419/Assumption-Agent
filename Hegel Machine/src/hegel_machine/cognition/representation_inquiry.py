"""Ask the live researcher to expose its own predictor's complete input contract."""

from copy import deepcopy
from math import isfinite


def native_representation_receipt(result, source_ref):
    """Keep decisions and costs beside references to the one complete study."""
    receipt = {key: deepcopy(value) for key, value in result.items()
               if key not in {'actions', 'candidates', 'selected', 'interrupted_work'}}
    receipt['source_ref'] = deepcopy(source_ref)
    receipt['actions_ref'] = {**deepcopy(source_ref), 'json_path': [
        *source_ref.get('json_path', []), 'actions']}
    receipt['action_count'] = len(result.get('actions', []))
    if result.get('interrupted_work') is not None:
        receipt['interrupted_work_ref'] = {**deepcopy(source_ref), 'json_path': [
            *source_ref.get('json_path', []), 'interrupted_work']}
    receipt['candidates'] = []
    for index, candidate in enumerate(result.get('candidates', [])):
        reference = {**deepcopy(source_ref), 'json_path': [
            *source_ref.get('json_path', []), 'candidates', index]}
        receipt['candidates'].append({
            **{key: deepcopy(value) for key, value in candidate.items()
               if key not in {'program', 'research_program'}},
            'source_ref': reference,
            'program_ref': {**deepcopy(reference), 'json_path': [*reference['json_path'], 'program']},
            'research_program_ref': {**deepcopy(reference),
                                     'json_path': [*reference['json_path'], 'research_program']}})
    receipt['candidate_count'] = len(receipt['candidates'])
    selected = result.get('selected')
    receipt['selected'] = None if selected is None else {
        **{key: deepcopy(value) for key, value in selected.items() if key not in
           {'program', 'research_program', 'validation_predictions', 'best_retained_validation_predictions'}},
        'source_ref': {**deepcopy(source_ref), 'json_path': [
            *source_ref.get('json_path', []), 'selected']}}
    if selected is not None:
        for field in ('program', 'research_program'):
            receipt['selected'][field + '_ref'] = {**deepcopy(source_ref), 'json_path': [
                *source_ref.get('json_path', []), 'selected', field]}
    return receipt


def _retain_native_representation_result(agent, question, result):
    from .checkpoint_storage import write_json_checkpoint

    directory = agent.workspace / 'native_representation_research'
    directory.mkdir(parents=True, exist_ok=True)
    # Every study, including failures, remains in this append-only small folder.
    path = directory / f'record_{len(list(directory.glob("record_*.json"))):04d}.json'
    saved = dict(result)
    selected = result.get('selected')
    if selected is not None:
        saved['selected'] = dict(selected)
        candidate = next(((index, item) for index, item in enumerate(result['candidates'])
                          if item['id'] == selected['id']), None)
        if candidate is not None:
            index, candidate = candidate
            for field in ('program', 'research_program'):
                if selected.get(field) is not None and selected[field] == candidate.get(field):
                    saved['selected'].pop(field)
                    saved['selected'][field + '_ref'] = {'json_path': ['candidates', index, field]}
    write_json_checkpoint(path, saved, indent=None, ensure_ascii=False)
    source_ref = {'file': str(path.resolve()), 'json_path': []}
    receipt = native_representation_receipt(result, source_ref)
    question.setdefault('native_representation_research', []).append(receipt)
    return receipt


def load_native_representation_result(source_ref, workspace):
    """Restore the selected draft from its own canonical candidate reference."""
    from .research_episodes import load_source_values

    result = load_source_values([source_ref], workspace)[0]['value']
    selected = result.get('selected')
    if selected is not None:
        for field in ('program', 'research_program'):
            reference = selected.pop(field + '_ref', None)
            if reference is not None:
                value = result
                for part in reference['json_path']:
                    value = value[part]
                selected[field] = deepcopy(value)
    return result


def _interrupted_native_work(error):
    """Retain paid native values still present in a real exception's frames."""
    from .models import SearchBudget
    from .research_programs import _Fitter

    frames, fitters, budgets = [], [], []
    seen_fitters, seen_budgets = set(), set()
    tracebacks = []
    traceback = error.__traceback__
    while traceback is not None:
        tracebacks.append(traceback)
        traceback = traceback.tb_next
    for traceback in tracebacks:
        frame = traceback.tb_frame
        frames.append({'file': frame.f_code.co_filename,
            'function': frame.f_code.co_name, 'line': traceback.tb_lineno})
        for value in frame.f_locals.values():
            if isinstance(value, SearchBudget) and id(value) not in seen_budgets:
                seen_budgets.add(id(value))
                budgets.append({'limit': value.limit, 'spent': value.spent,
                    'completed_fits': value.completed_fits,
                    'last_denied_request': deepcopy(value.last_denied_request)})
            if not isinstance(value, _Fitter) or id(value) in seen_fitters:
                continue
            seen_fitters.add(id(value))
            pending = [entry.tb_frame.f_locals['term'] for entry in tracebacks
                if entry.tb_frame.f_code.co_name == 'column'
                and entry.tb_frame.f_locals.get('self') is value
                and entry.tb_frame.f_locals.get('term') not in value.columns]
            terms = list(dict.fromkeys([*value.columns,
                *(term for fitted in value.fits.values() if fitted is not None
                  for term in fitted['fit_terms']), *pending]))
            indices = {term: index for index, term in enumerate(terms)}
            fits = [{'fit_terms': [indices[term] for term in fitted['fit_terms']],
                'used_terms': [indices[term] for term in fitted['terms']],
                'coefficients': list(fitted['coefficients']), 'loss': fitted['loss']}
                for fitted in value.fits.values() if fitted is not None]
            fitters.append({'rows': value.count, 'quantile': value.quantile,
                'terms': [term.to_dict() for term in terms],
                'columns': [{'term': indices[term], 'values': deepcopy(column),
                    'status': 'failed_missing_or_nonfinite_input' if column is None else 'complete'}
                            for term, column in value.columns.items()],
                'unfinished_columns': [{'term': indices[term], 'status': 'interrupted_incomplete'}
                                       for term in pending],
                'fits': fits,
                'failed_fit_terms': [[indices[term] for term in terms_]
                    for terms_, fitted in value.fits.items() if fitted is None],
                'budget': {'limit': value.budget.limit, 'spent': value.budget.spent,
                    'completed_fits': value.budget.completed_fits}})
    return {'frames': frames, 'budgets': budgets, 'fitters': fitters,
        'scope': 'Completed paid columns and fitted coefficients at this actual '
            'Python interruption. This retains values, not a resumable search cursor.'}


def load_native_interrupted_fit(source_ref, workspace, *, fitter_index=0, fit_index=0):
    """Reconstruct an already paid fit using its original terms and coefficients."""
    from .continuous_state import _sum_terms
    from .language import Program
    from .research_episodes import load_source_values

    work = load_source_values([source_ref], workspace)[0]['value']
    fitter = work['fitters'][fitter_index]
    fitted = fitter['fits'][fit_index]
    terms = [Program.from_dict(fitter['terms'][index]) for index in fitted['fit_terms']]
    return {**deepcopy(fitted), 'program': _sum_terms(terms,
        fitted['coefficients'][1:], fitted['coefficients'][0]).to_dict(),
        'rows': fitter['rows'], 'source_ref': deepcopy(source_ref)}


def retained_native_research_programs(references, parent_programs, workspace):
    """Resolve a parent's actual procedure without borrowing another parent state."""
    from .language import Program
    from .research_episodes import load_source_values

    if not isinstance(references, list):
        raise ValueError('retained_research_program_refs must be a list of candidate references')
    resolved = []
    for loaded in load_source_values(references, workspace):
        candidate = loaded['value']
        parent = Program.from_dict(candidate['program'])
        parent_ids = [identifier for identifier, program in parent_programs.items()
                      if program == parent]
        if not parent_ids:
            raise ValueError('retained research procedure does not belong to the actual parent')
        procedure = candidate['research_program']
        output = procedure['graph']['nodes'][procedure['output_node']]['answer']
        if output is None or Program.from_dict(output) != parent:
            raise ValueError('retained research procedure output differs from its parent Program')
        resolved.append({'source_ref': loaded['source_ref'], 'parent_theory_ids': parent_ids,
                         'research_program': procedure})
    return resolved


def _root_paid_rejection(agent):
    """A root researcher can encounter an opaque bad fit before any child exists."""
    event = agent.events[-1]
    arguments, evaluation = event.get('arguments', {}), event.get('feedback', {})
    if (event.get('action') not in {'evaluate', 'finish'}
            or 'component' in arguments or arguments.get('target', 'response') != 'response'):
        return None
    model = arguments.get('model', 'model.py')
    candidate = agent.research_candidates.get(model, {})
    mse = evaluation.get('mse')
    if (evaluation.get('returncode') != 0
            or evaluation.get('representation_diagnostic', {}).get('status') != 'not_exposed'
            or not isinstance(mse, (int, float)) or not isfinite(mse)
            or mse <= agent.config.confirmation_mse_threshold
            or candidate.get('paid_rows') != len(agent.rows)
            or candidate.get('status') != 'evaluated'):
        return None
    parent_id = evaluation.get('research_question_id') or 'task'
    owner = next((q for q in agent.questions if q['id'] == parent_id
                  and q.get('parent_id') is None), None)
    if owner is None:
        return None
    return owner, {'parent_id': owner['id'], 'evaluated_model': model,
                   'phase': 'representation_unassessed'}


def open_representation_inquiry(agent):
    """Reuse the ordinary child loop once for each actual authored model source."""
    if (agent.arm != 'hegel' or agent.provider is None or agent.model is not None
            or agent._inquiry_stack or not agent.events
            or any(q.get('flat_review', {}).get('status') == 'running' for q in agent.questions)
            or agent.config.max_model_calls - agent.costs['model_calls'] <= 1
            or agent.costs['python_execution_seconds'] >= agent.config.execution_seconds
            or agent.costs['reported_tokens'] is None
            or agent.costs['reported_tokens'] >= agent.config.token_budget):
        return False
    owner = agent._parent_research()
    root_rejection = owner is None
    if owner is None:
        rejected = _root_paid_rejection(agent)
        if rejected is None:
            return False
        owner, focus = rejected
    else:
        focus = owner['parent_research']
    model = focus.get('evaluated_model')
    evaluation = agent.research_candidates.get(model, {}).get('current_evaluation', {})
    unassessed_method = (focus.get('phase') == 'prediction_method_revision'
        and evaluation.get('returncode') == 0
        and evaluation.get('representation_diagnostic', {}).get('status') == 'not_exposed')
    if focus.get('phase') != 'representation_unassessed' and not unassessed_method:
        return False
    if not model or not (agent.workspace / model).is_file():
        return False
    source = (agent.workspace / model).read_text()
    previous = [q for q in agent.questions if q.get('origin') == 'representation_research']
    if any((q.get('representation_model') == model
            and source in (q.get('source_at_start'), q.get('source_at_return')))
           or (q.get('returned_model') == model and q.get('returned_model_source') == source)
           for q in previous):
        return False
    question = {
        'id': f'representation_research_{len(previous)}',
        'origin': 'representation_research',
        'parent_id': focus['parent_id'], 'depends_on': [],
        'parent_research_question_id': owner['id'],
        'representation_model': model, 'source_at_start': source,
        'question': f'Investigate the actual input representation of your existing predictor {model}. '
            'Inspect its current source and expose the complete inputs and routing it actually uses '
            'through the existing represent(row), predict_features(features), and, when fitting is '
            'needed, fit_features(features, targets) interface. Adapt the fitting path as well as '
            'prediction; the explicit interface does not call the old fit(rows). Initially preserve '
            'your own scientific hypothesis while making its input contract executable. Include '
            'all information your downstream predictor actually needs, rather than an unrelated '
            'diagnostic view. Evaluate the authored model on the existing paid records, inspect '
            'the computed collision witnesses and execution errors, and research any actual '
            'conflict using those records. Inspect relevant fitting and representation code in '
            'the inherited research programs and source_catalog.json before rebuilding it. '
            'Read the complete relevant functions; archived fitted coefficients are not current '
            'evidence. Choose whether to reuse, refit or revise that source using the current paid '
            'counterexamples. Choose any scientific revision yourself. Do not purchase '
            'new observations in this local investigation; return any request for new evidence to '
            'the parent. Return your executable artifact and the measured diagnostic, or state '
            'what remains unresolved. Local finish returns to the parent. No observed collision '
            'and successful execution do not establish independent confirmation.'}
    if root_rejection:
        question['root_paid_rejection'] = {
            'event_index': len(agent.events) - 1, 'model': model,
            'paid_rows': len(agent.rows), 'mse': evaluation['mse']}
    parent_event = len(agent.events) - 1
    result = agent._start_inquiry({'question': question, 'max_model_calls': 8},
                                  parent_event_index=parent_event)
    focus['representation_inquiry_id'] = question['id']
    agent.events[parent_event].setdefault('feedback', {})['representation_research'] = result
    return True


def record_representation_return(agent, frame, result):
    """Retain the actual returned draft so an unresolved edit is not audited twice."""
    if result['question'].get('origin') != 'representation_research':
        return
    question = next(q for q in agent.questions if q['id'] == frame['question_id'])
    model = question['representation_model']
    path = agent.workspace / model
    question['source_at_return'] = path.read_text() if path.is_file() else None
    question['representation_return'] = {key: deepcopy(result[key]) for key in
        ('status', 'model_calls', 'artifacts', 'event_indices') if key in result}
    if question.get('root_paid_rejection'):
        # A real child return creates the ordinary parent synthesis; no failed
        # fresh confirmation or prior child is needed to reach that existing loop.
        question['research_return'] = {
            **deepcopy(question['representation_return']),
            'question_id': question['id'], 'parent_id': question['parent_id'],
            'assignment': deepcopy(question['assignment']), 'revision_indices': []}
        agent._prepare_parent_research()
        owner = next(q for q in agent.questions
                     if q['id'] == question['parent_research_question_id'])
        if owner.get('parent_research'):
            owner['parent_research']['evaluated_model'] = model
    for index in reversed(result.get('event_indices', [])):
        event = agent.events[index]
        args, feedback = event.get('arguments', {}), event.get('feedback', {})
        evaluation = feedback.get('evaluation', {})
        if (event.get('action') != 'finish' or not args.get('model')
                or 'component' in args or args.get('target', 'response') != 'response'
                or evaluation.get('returncode') != 0 or evaluation.get('error')):
            continue
        returned = evaluation.get('executed_path') or args['model']
        path = agent.workspace / returned
        if path.is_file():
            question.update(returned_model=returned, returned_model_source=path.read_text())
            owner = next(q for q in agent.questions
                         if q['id'] == question['parent_research_question_id'])
            if owner.get('parent_research'):
                owner['parent_research']['evaluated_model'] = returned
        break


def execute_native_representation_inquiry(agent, question, *, spec, parent_programs,
        validation_records, compute_limit, max_candidates=8,
        retained_research_program_refs=None):
    """Research a representation or retained-program failure on paid trajectories.

    This is the existing finite native researcher, not a substitute model author.
    It receives public observations and retained executable parents. The caller
    installs its returned draft through the ordinary evaluation/parent loop; this
    function never buys observations, fits on validation, or overwrites a parent.
    """
    from .evaluation import initial_prediction, loss, predict
    from .models import Candidate, Observation, ResearchConfig, SearchBudget
    from .unified_research import UnifiedResearchAgent
    from time import monotonic

    host_started = monotonic()

    if not parent_programs:
        raise ValueError('representation research needs its retained parent pool')
    if type(compute_limit) is not int or compute_limit < 0:
        raise ValueError('native representation compute_limit must be nonnegative')
    if type(max_candidates) is not int or max_candidates < 2:
        raise ValueError('native representation research needs at least two candidate slots')
    if spec.risk_quantile != .5:
        raise ValueError('representation collision research currently uses the MSE contract')
    revision = question.get('kind') == 'program_revision'
    training, validation, split = agent._split({'validation_records': validation_records})
    if not validation:
        raise ValueError('representation research needs distinct validation trajectories')
    # Persist the exact split before any target-dependent proposal or fitting.
    # The ordinary splitter groups equal public reset trajectories together.
    question['native_representation_split'] = deepcopy(split)
    rows = lambda values: [Observation(dict(row['features']), float(row['target']),
        tuple(dict(point) for point in row.get('history', []))) for row in values]
    training, validation = rows(training), rows(validation)
    budget = SearchBudget(min(compute_limit, agent._remaining_compute()))
    threshold = agent.config.confirmation_mse_threshold
    researcher = UnifiedResearchAgent(ResearchConfig(
        compute_budget=max(1, budget.limit), max_candidates=max_candidates,
        minimum_improvement=threshold, active_probes=False))
    fallback = initial_prediction(training, .5)
    result = {'status': 'native_representation_inconclusive_budget',
        'origin': (deepcopy(question.get('origin')) if revision else
                   'computed_representation_conflict'), 'question_id': question['id'],
        'research_kind': 'program_revision' if revision else 'representation',
        'parent_theory_ids': list(parent_programs),
        'split': deepcopy(split), 'mse_threshold': threshold,
        'retained_comparison': [], 'candidates': [], 'actions': [],
        'selected': None, 'fitting_rows': len(training),
        'validation_rows': len(validation), 'provider_calls': 0,
        'new_observations': 0, 'all_parent_candidates_retained': True,
        'evidence_scope': 'Already exposed development material; whole-trajectory '
            'validation selects a working draft, not independent confirmation.'}
    # Fund all retained comparisons, every possible proposed holdout comparison,
    # and the actual selected-vs-retained prediction trace before search begins.
    baseline_cost = len(parent_programs) * (len(training) + len(validation))
    directions = (('retained_composition',) if revision else
                  ('coordinates', 'retained_composition'))
    try:
        procedures = retained_native_research_programs(
            [] if retained_research_program_refs is None else retained_research_program_refs,
            parent_programs, agent.workspace)
        # Saved procedure replays supplement the original grammar's slots; they
        # must not evict its last candidate by occupying the old prefix cap.
        compared_capacity = max(2 * max_candidates,
            len(directions) * max_candidates + len(procedures))
        reserved = (compared_capacity + 2) * len(validation)
        if not budget.can_spend(baseline_cost + reserved):
            result['required_comparison_compute'] = baseline_cost + reserved
            return result
        retained = []
        for identifier, program in parent_programs.items():
            budget.spend(len(training))
            retained.append({'candidate': Candidate(str(identifier), program, 'retained_parent'),
                'training_loss': loss(program, training, fallback, .5), 'retained': True})
        retained = researcher._assess(retained, validation, budget, fallback, .5)
        result['retained_comparison'] = [{'id': entry['candidate'].candidate_id,
            'nodes': entry['candidate'].program.size, 'training_mse': entry['training_loss'],
            'validation_mse': entry['validation_loss']} for entry in retained]
        best_retained = researcher._best(retained)
        allowance = budget.remaining - reserved
        proposals = []
        # A response counterexample with a sufficient view revises its actual
        # retained program; it does not invent another representation conflict.
        for index, direction in enumerate(directions):
            local = SearchBudget(allowance // len(directions)
                + (allowance % len(directions) if index == 0 else 0))
            direction_receipt = {'direction': direction, 'compute_limit': local.limit}
            result.setdefault('search_directions', []).append(direction_receipt)
            try:
                if direction == 'coordinates':
                    claims = researcher._coordinate_claims(spec, training, local, result['actions'])
                    bounded_claims = claims[:max_candidates]
                else:
                    replayed, completed_procedures = [], []
                    for procedure in procedures:
                        source_ref = procedure['source_ref']
                        replay_receipt = {'action': 'replay_retained_parent_research_program',
                            'source_ref': deepcopy(source_ref),
                            'parent_theory_ids': list(procedure['parent_theory_ids']),
                            'fitting_rows': len(training), 'validation_rows_used_for_fit': 0,
                            'compute_cost': 0, 'status': 'running'}
                        result['actions'].append(replay_receipt)
                        duplicate = next((previous for previous in completed_procedures
                            if previous['research_program'] == procedure['research_program']), None)
                        if duplicate is not None:
                            replay_receipt.update(status='duplicate_procedure_reused',
                                reused_source_ref=deepcopy(duplicate['source_ref']))
                            continue
                        completed_procedures.append(procedure)
                        from .research_programs import replay_research_program
                        replay_started = local.spent
                        try:
                            claim = replay_research_program(procedure['research_program'],
                                training, local, risk_quantile=spec.risk_quantile)
                        except Exception as error:
                            replay_receipt.update(status='execution_failed',
                                error=f'{type(error).__name__}: {error}')
                            raise
                        finally:
                            replay_receipt['compute_cost'] = local.spent - replay_started
                        if claim is None:
                            replay_receipt.update(status=('incomplete_budget' if
                                local.last_denied_request else 'no_executable_refit'),
                                last_denied_request=deepcopy(local.last_denied_request))
                            continue
                        claim['research_program']['source_ref'] = deepcopy(source_ref)
                        claim['method_source_refs'] = [deepcopy(source_ref)]
                        replay_receipt.update(status='refitted_requires_validation',
                            training_mse=claim['training_loss'])
                        replayed.append(claim)
                    claims = researcher._program_claims(spec, training, local,
                        list(parent_programs.values()), result['actions'],
                        **({'replayed': replayed} if procedures else {}))
                    reused = [claim for claim in claims
                              if any(claim is item for item in replayed)]
                    generated = [claim for claim in claims
                                 if not any(claim is item for item in replayed)]
                    bounded_claims = [*reused, *generated[:max_candidates]]
            finally:
                budget.spend(local.spent)
                direction_receipt.update(compute_cost=local.spent,
                    completed_fits=local.completed_fits,
                    last_denied_request=deepcopy(local.last_denied_request))
            proposals.extend(bounded_claims)
        unique = {}
        for entry in proposals:
            if entry['candidate'].program not in parent_programs.values():
                unique.setdefault(entry['candidate'].program, entry)
        checked = researcher._assess(list(unique.values()), validation, budget, fallback, .5)
        if len(checked) != len(unique):
            # Incomplete comparisons never select a new draft.
            result['compared_candidates'] = len(checked)
            return result
        result['candidates'] = [{'id': f"native_representation:{index}",
            'program': entry['candidate'].program.to_dict(),
            'nodes': entry['candidate'].program.size,
            'training_mse': entry['training_loss'], 'validation_mse': entry['validation_loss'],
            **({'method_source_refs': deepcopy(entry['method_source_refs'])}
               if entry.get('method_source_refs') else {}),
            'research_program': deepcopy(entry.get('research_program'))}
            for index, entry in enumerate(checked)]
        winner = researcher._best([*retained, *checked])
        selected_program = winner['candidate'].program
        budget.spend(2 * len(validation))
        before = [predict(best_retained['candidate'].program, row, fallback) for row in validation]
        after = [predict(selected_program, row, fallback) for row in validation]
        is_retained = winner.get('retained', False)
        result['selected'] = {'id': winner['candidate'].candidate_id if is_retained else
            f"native_representation:{checked.index(winner)}", 'retained': is_retained,
            'program': selected_program.to_dict(), 'nodes': selected_program.size,
            'training_mse': winner['training_loss'], 'validation_mse': winner['validation_loss'],
            'best_retained_id': best_retained['candidate'].candidate_id,
            'best_retained_validation_mse': best_retained['validation_loss'],
            'validation_predictions': after, 'best_retained_validation_predictions': before,
            'changed_prediction_rows': sum(left != right for left, right in zip(before, after)),
            'max_prediction_change': max(abs(left - right) for left, right in zip(before, after)),
            'research_program': deepcopy(winner.get('research_program'))}
        result.update(status='native_representation_draft_selected' if not is_retained
            else 'native_representation_parent_retained', comparison_completed=True,
            remaining_question=('Compare this actual parent revision against the complete '
                'strong pool; a proposed or unadopted descendant is not completed recursion.'
                if revision else 'Verify the actual exposed representation and frozen draft '
                'against its paid witnesses before resolving the representation child.'))
        return result
    except Exception as error:
        result.update(status='native_representation_execution_failed',
                      error=f'{type(error).__name__}: {error}',
                      interrupted_work=_interrupted_native_work(error))
        raise
    finally:
        agent._charge(budget.spent)
        result['compute_cost'] = budget.spent
        result['compute_limit'] = budget.limit
        result['last_denied_request'] = deepcopy(budget.last_denied_request)
        result['native_host_seconds_before_persistence'] = monotonic() - host_started
        result['host_time_accounting'] = ('Native host duration is separate from the existing '
            'external Python execution ledger; this function does not debit that ledger.')
        _retain_native_representation_result(agent, question, result)

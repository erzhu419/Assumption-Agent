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
               if key not in {'program', 'research_program', 'state_research'}},
            'source_ref': reference,
            'program_ref': {**deepcopy(reference), 'json_path': [*reference['json_path'], 'program']},
            **({'state_research_ref': {**deepcopy(reference),
                'json_path': [*reference['json_path'], 'state_research']}}
                if candidate.get('state_research') else {}),
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
    from .method_action_research import read_method_reference
    from .research_episodes import _canonical, load_source_values

    if not isinstance(references, list):
        raise ValueError('retained_research_program_refs must be a list of candidate references')
    sources, deferred = [], []
    for original in references:
        reference = _canonical(original)
        candidate = None
        if not reference.get('archive'):
            try:
                candidate = read_method_reference(reference, workspace)
            except KeyError:
                # Historical local paths may cross a saved field reference.
                # Their original loader resolves that declaration explicitly.
                pass
        if (isinstance(candidate, dict) and isinstance(candidate.get('program'), dict)
                and isinstance(candidate.get('research_program'), dict)
                and 'graph' in candidate['research_program']):
            sources.append({'source_ref': reference, 'value': candidate})
        else:
            deferred.append((len(sources), reference))
            sources.append(None)
    if deferred:
        for (index, _), loaded in zip(deferred,
                load_source_values([reference for _, reference in deferred], workspace)):
            sources[index] = loaded
    resolved = []
    for loaded in sources:
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


def _representation_search_basis(question, parent_programs, public_features, *, questions=(), resolve=None):
    """Use actual view evidence to open an optional finite alternative path.

    A retained-program revision keeps its existing narrow contract unless the
    question supplies concrete representation evidence. Prediction error alone
    does not establish a missing dependency or prescribe a deeper history.
    """
    from .finite_native_author import representation_research_warrant
    from .representation_diagnostics import representation_evidence_state

    diagnostic = question.get('representation_diagnostic') or {}
    state = representation_evidence_state(diagnostic)
    if state == 'proven_insufficient_on_evidence':
        return state, 'observed_representation_conflict'
    return state, representation_research_warrant(question,
        {identifier: program.expanded().to_dict() for identifier, program in parent_programs.items()},
        public_features, questions=questions, resolve=resolve)


def _training_representation_witnesses(diagnostic, training):
    """Bind lost distinctions to public inputs in the declared fitting split."""
    groups = []
    for witness in diagnostic.get('witnesses', []):
        indices = []
        for example in witness.get('rows', []):
            index = next((index for index, row in enumerate(training)
                if row.features == example['features']
                and tuple(row.history) == tuple(example.get('history', []))), None)
            if index is not None and index not in indices:
                indices.append(index)
        if len(indices) >= 2 and indices not in groups:
            groups.append(indices)
    return groups


def exposed_development_mse(training_mse, validation_mse, training_rows, validation_rows):
    """Combine the two paid row losses on the full exposed comparison view."""
    return (training_rows * training_mse + validation_rows * validation_mse) / (
        training_rows + validation_rows)


def _development_claim_key(entry, training_rows, validation_rows):
    return (exposed_development_mse(entry['training_loss'], entry['validation_loss'],
        training_rows, validation_rows), entry['candidate'].program.size)


def _best_development_claim(claims, training_rows, validation_rows, tolerance):
    ranked = sorted(claims, key=lambda entry: _development_claim_key(
        entry, training_rows, validation_rows))
    lowest = _development_claim_key(ranked[0], training_rows, validation_rows)[0]
    adequate = [entry for entry in ranked if _development_claim_key(
        entry, training_rows, validation_rows)[0] <= lowest + tolerance]
    return min(adequate, key=lambda entry: (entry['candidate'].program.size,
        _development_claim_key(entry, training_rows, validation_rows)[0]))


def _witness_relation_claims(spec, training, parent_programs, coordinates, budget,
                              actions, *, max_candidates):
    """Consume discovered coordinates in the existing response grammars."""
    from .conditional_coordinates import propose_conditional_coordinates
    from .models import SearchBudget
    from .research_programs import propose_research_programs
    from .unified_research import _terms

    current = _terms(spec.feature_names, quadratic=True)
    basis = list(dict.fromkeys([*current, *coordinates]))
    receipt = {'action': 'compose_witness_relations',
        'depends_on_action': 'research_missing_public_distinctions',
        'coordinates': [term.to_dict() for term in coordinates],
        'fitting_rows': len(training), 'validation_rows_used_for_fit': 0,
        'branches': []}
    actions.append(receipt)
    claims, allowance = [], budget.remaining
    for index, direction in enumerate(('residual_composition', 'conditional_response')):
        # Keep room for both kinds of joint fit; unused first-stage work flows
        # into the second stage instead of disappearing in an isolated cap.
        local = SearchBudget((allowance + 1) // 2 if index == 0 else budget.remaining)
        branch = {'direction': direction, 'compute_limit': local.limit, 'progress': {}}
        receipt['branches'].append(branch)
        try:
            if direction == 'residual_composition':
                found = propose_research_programs(spec, training, local,
                    list(parent_programs.values()), coordinates=basis,
                    max_candidates=max_candidates, progress=branch['progress'])
            else:
                found = propose_conditional_coordinates(spec, training, local,
                    basis=basis, gate_coordinates=list(dict.fromkeys([
                        *(term for term in current if term.op == 'feature'), *coordinates])),
                    retained_programs=list(parent_programs.values()),
                    receipt=branch['progress'], max_candidates=max_candidates)
            claims.extend(found)
            branch['proposed_candidates'] = len(found)
        finally:
            budget.spend(local.spent)
            budget.completed_fits += local.completed_fits
            branch.update(compute_cost=local.spent, completed_fits=local.completed_fits,
                last_denied_request=deepcopy(local.last_denied_request),
                completion=('incomplete_compute' if local.last_denied_request
                    else 'bounded_work_delivered'))
    receipt['compute_cost'] = sum(branch['compute_cost'] for branch in receipt['branches'])
    return claims


def _witness_coordinate_claims(spec, training, parent_programs, groups, budget,
                               actions, *, max_candidates, validation, fallback):
    """Fit existing public history terms that distinguish actual counterexamples.

    A distinguishing coordinate is only a proposal. Its whole predictor still
    competes on the declared trajectories and later against the retained pool.
    """
    from .models import ResearchConfig, SearchBudget
    from .research_programs import _entry, _Fitter
    from .unified_research import UnifiedResearchAgent, _terms

    start = budget.spent
    indices = list(dict.fromkeys(index for group in groups for index in group))
    positions = {index: offset for offset, index in enumerate(indices)}
    witnesses = [training[index] for index in indices]
    observed_depth = max(len(row.history) for row in witnesses)
    names = tuple(name for name in spec.feature_names
        if all(name in row.features for row in training))
    trial = {'action': 'research_missing_public_distinctions',
        'training_witness_groups': deepcopy(groups),
        'validation_targets_used_for_fitting': False,
        'validation_targets_used_for_selection': True,
        'observed_history_bound': observed_depth, 'screened_coordinates': 0,
        'distinguishing_coordinates': [], 'attempts': []}
    actions.append(trial)
    # Validate every bounded additive fit and each relation proposal before
    # the export cap. Otherwise one train-best parent can take all slots.
    reserve = max_candidates * (len(parent_programs) + 2) * len(validation)
    if not budget.can_spend(reserve):
        trial.update(status='insufficient_candidate_validation_budget',
                     required_validation_compute=reserve, compute_cost=0)
        return []
    fitting = SearchBudget(budget.remaining - reserve)
    try:
        coordinates = []
        for term in _terms(names, depth=observed_depth, quadratic=True):
            if not fitting.spend(len(witnesses)):
                break
            values = [term.predict(row) for row in witnesses]
            split = sum(len({values[positions[index]] for index in group}) > 1 for group in groups)
            trial['screened_coordinates'] += 1
            if split:
                coordinates.append((split, term))
        coordinates.sort(key=lambda item: (-item[0], item[1].history_depth, item[1].size))
        coordinates = coordinates[:max_candidates]
        trial['distinguishing_coordinates'] = [term.to_dict() for _, term in coordinates]
        fitter, proposed = _Fitter(training, fitting, spec.risk_quantile), []
        current = _terms(names, quadratic=True)
        for _, term in coordinates:
            for parent_id, parent in parent_programs.items():
                before = fitting.spent
                fit = fitter.fit((parent, *current, term))
                attempt = {'coordinate': term.to_dict(), 'parent_theory_id': parent_id,
                    'compute_cost': fitting.spent - before,
                    'status': 'unfitted' if fit is None else 'requires_trajectory_validation'}
                trial['attempts'].append(attempt)
                if fit is None:
                    continue
                separated = None
                if fitting.spend(len(witnesses)):
                    values = [fit['program'].predict(row) for row in witnesses]
                    separated = sum(len({values[positions[index]] for index in group}) > 1 for group in groups)
                else:
                    attempt['status'] = 'witness_check_unfunded'
                attempt.update(training_mse=fit['loss'],
                    predictor_separated_groups=separated, witness_groups=len(groups))
                entry = _entry(fit, [{'operation': 'refit_missing_public_distinction',
                    'question': 'Does this measured coordinate recover a distinction lost by the parent?',
                    'program': fit['program'], 'loss': fit['loss'],
                    'coordinate': term.to_dict(), 'parent_theory_id': parent_id}],
                    len(training), fitting.spent - before, len(proposed))
                entry['representation_witness_check'] = {
                    'training_witness_groups': deepcopy(groups),
                    'predictor_separated_groups': separated, 'witness_groups': len(groups),
                    'status': ('witness_check_unfunded' if separated is None else
                        'distinguishes_inputs_only' if separated == len(groups)
                        else 'remaining_witness_collision'),
                    'prediction_quality_established': False}
                proposed.append(entry)
        proposed.extend(_witness_relation_claims(spec, training, parent_programs,
            [term for _, term in coordinates], fitting, actions,
            max_candidates=max_candidates) if coordinates else [])
    finally:
        budget.spend(fitting.spent)
        budget.completed_fits += fitting.completed_fits
    unique = {}
    for entry in proposed:
        unique.setdefault(entry['candidate'].program, entry)
    researcher = UnifiedResearchAgent(ResearchConfig(max_candidates=max(2, max_candidates)))
    checked = researcher._assess(list(unique.values()), validation, budget, fallback, .5)
    # These are exposed development rows, also used by the outer strong-pool
    # comparison. Rank on that same row-weighted objective before the export
    # cap; retain the old-region loss separately rather than silently imposing
    # a different objective upstream. The fitting split is unchanged.
    ranked = sorted(checked, key=lambda entry: _development_claim_key(
        entry, len(training), len(validation)))
    kept = ranked[:max_candidates]
    trial['candidate_comparisons'] = [{'training_mse': entry['training_loss'],
        'validation_mse': entry['validation_loss'],
        'development_mse': _development_claim_key(entry, len(training), len(validation))[0],
        'exported': any(entry is selected for selected in kept)} for entry in checked]
    # Preserve paid losing fits once in the canonical study; compact receipts
    # retain an actions_ref rather than copying them into parent snapshots.
    trial['unexported_candidates'] = [{'program': entry['candidate'].program.to_dict(),
        'research_program': entry.get('research_program'),
        'training_mse': entry['training_loss'], 'validation_mse': entry['validation_loss'],
        'development_mse': _development_claim_key(entry, len(training), len(validation))[0]}
        for entry in checked if not any(entry is selected for selected in kept)]
    trial.update(compute_cost=budget.spent - start, completed_fits=budget.completed_fits,
        compared_candidates=len(checked), proposed_candidates=len(unique),
        selection='full_exposed_development_mse_before_export_cap',
        selection_rows={'training': len(training), 'validation': len(validation)},
        selection_data_role='exposed_development', independent_confirmation=False,
        last_denied_request=deepcopy(fitting.last_denied_request or budget.last_denied_request))
    return kept


def _reuse_paid_directions(request, agent, spec, parents, split, replay_scope):
    """Reuse declared completed proposals only on their actual original inputs."""
    from .language import Program
    from .method_action_research import read_method_reference
    from .models import Candidate

    if not request:
        return {}, {}
    prior = load_native_representation_result(request['source_ref'], agent.workspace)
    old_observations = read_method_reference(request['paid_observations_ref'], agent.workspace)
    def public_rows(records):
        return [[{'features': dict(row['features']), 'target': float(row['target']),
            'history': [dict(point) for point in row.get('history', [])]}
            for row in record['rows']] for record in records]
    public = {'feature_names': list(spec.feature_names), 'risk_quantile': spec.risk_quantile,
              'max_probe_length': spec.max_probe_length,
              'feature_bounds': {key: list(value) for key, value in spec.feature_bounds.items()},
              'context': spec.context}
    original_public = {'feature_bounds': {}, 'context': {}, **request['public_spec']}
    old_parents = {identifier: Program.from_dict(read_method_reference(reference, agent.workspace))
        for identifier, reference in request['parent_program_refs'].items()}
    if (public_rows(old_observations) != public_rows(agent.observations)
            or prior['split'] != split or old_parents != parents
            or list(old_parents) != list(parents) or original_public != public
            or prior.get('retained_replay_scope', 'procedure') != replay_scope):
        raise ValueError('paid direction reuse requires the same rows, split, parents and public contract')
    if not prior.get('comparison_completed'):
        raise ValueError('paid direction reuse requires completed candidate comparison')
    claims, receipts, seen = {}, {}, set()
    source_directions = {item['direction']: item for item in prior['search_directions']}
    for direction, indices in request['directions'].items():
        if direction not in {'coordinates', 'retained_composition'}:
            raise ValueError('only the unchanged coordinate and retained-composition directions are reused')
        source = source_directions[direction]
        if source.get('last_denied_request'):
            raise ValueError('an incomplete old direction must not be declared completed reuse')
        claims[direction], references = [], []
        for index in indices:
            if index in seen:
                raise ValueError('a paid candidate must belong to one reused direction')
            seen.add(index)
            candidate = prior['candidates'][index]
            reference = {**deepcopy(request['source_ref']), 'json_path': [
                *request['source_ref'].get('json_path', []), 'candidates', index]}
            references.append(reference)
            claims[direction].append({'candidate': Candidate(candidate['id'],
                Program.from_dict(candidate['program']), 'paid_direction_reuse'),
                'training_loss': candidate['training_mse'],
                'validation_loss': candidate['validation_mse'],
                'research_program': deepcopy(candidate.get('research_program')),
                'paid_candidate_ref': reference, 'search_direction': direction,
                **{key: deepcopy(candidate[key]) for key in
                    ('method_source_refs', 'representation_witness_check', 'state_research')
                    if key in candidate}})
        receipts[direction] = {'direction': direction, 'status': 'paid_candidates_reused',
            'compute_cost': 0, 'completed_fits': 0, 'last_denied_request': None,
            'source_compute_cost': source['compute_cost'], 'candidate_refs': references,
            'source_ref': deepcopy(request['source_ref'])}
    return claims, receipts


def _representation_experiment_frontier(agent, question, spec, claims, winner, budget, threshold):
    """Turn equally supported executable laws into one payable public question.

    The historical planner supplies and forecasts actual legal commands. Paid
    fitting scores only choose its competitors; a zero error is not evidence
    that their unobserved predictions agree.
    """
    from .program_disagreement_probes import propose_program_disagreement_probes

    public = question.get('native_representation_public_spec') or {}
    if 'trajectory' not in public.get('action_kinds', []):
        return None
    supported = {}
    for entry in claims:
        if (abs(entry['training_loss'] - winner['training_loss']) <= threshold
                and abs(entry['validation_loss'] - winner['validation_loss']) <= threshold):
            supported.setdefault(entry['candidate'].program, entry['candidate'].candidate_id)
    if len(supported) < 2:
        return None
    start = budget.spent
    proposals = propose_program_disagreement_probes(spec, supported, budget,
        serial=len(agent.observations), length=min(4, spec.max_probe_length),
        control_levels=public.get('control_levels'))
    # A stored trajectory has already answered each of its reset prefixes.
    paid = [tuple(row['features'] for row in record.get('rows', []))
            for record in agent.observations]
    usable = [item for item in proposals if item['predicted_separation'] > threshold
        and not any(tuple(item['probe'].sequence) == sequence[:item['probe'].cost]
                    for sequence in paid)]
    if not usable:
        return None
    selected = max(usable, key=lambda item: (
        item['predicted_separation'] / item['probe'].cost, -item['probe'].cost))
    declaration = {key: deepcopy(question[key]) for key in (
        'id', 'question', 'kind', 'parent_id', 'depends_on', 'answer_type', 'meaning_version',
        'origin', 'context_ref', 'query_semantics', 'answer_resolution_ref', 'use_sites',
        'theory_target', 'representation_diagnostic', 'representation_research_basis') if key in question}
    continuation = {'operator': 'research_route', 'research_question': declaration,
                    'native_representation_plan': deepcopy(question['native_representation_plan'])}
    return {'question': 'Which equally supported explanation predicts the new reset trajectory?',
        'candidate_ids': list(supported.values()), 'candidate_count': len(supported),
        'training_mse': winner['training_loss'], 'validation_mse': winner['validation_loss'],
        'predicted_separation': selected['predicted_separation'],
        'predictions': deepcopy(selected['predictions']),
        'design': {key: deepcopy(value) for key, value in selected.items()
                   if key not in {'probe', 'predictions'}},
        'planning_compute': budget.spent - start,
        'paid_rows': len(agent.rows), 'observed_record_count': len(agent.observations),
        'observation_cost': selected['probe'].cost, 'independent_confirmation': False,
        'query_action': {'action': 'query', 'arguments': {'actions': [{
            'kind': 'trajectory', 'channels': [], 'sequence': list(selected['probe'].sequence)}],
            'question': declaration, 'after_observation': [continuation]}}}


def execute_native_representation_inquiry(agent, question, *, spec, parent_programs,
        validation_records, compute_limit, max_candidates=8,
        retained_research_program_refs=None, retained_replay_scope=None,
        paid_direction_reuse=None):
    """Research a representation or retained-program failure on paid trajectories.

    This is the existing finite native researcher, not a substitute model author.
    It receives public observations and retained executable parents. The caller
    installs its returned draft through the ordinary evaluation/parent loop; this
    function never buys observations, fits on validation, or overwrites a parent.
    """
    from .evaluation import initial_prediction, loss, predict
    from .models import Candidate, Observation, ResearchConfig, SearchBudget
    from .language import Program
    from .research_episodes import load_source_values
    from .unified_research import UnifiedResearchAgent
    from time import monotonic

    host_started = monotonic()

    if not parent_programs:
        raise ValueError('representation research needs its retained parent pool')
    declared_order = (question.get('native_representation_plan') or {}).get('parent_theory_ids')
    if declared_order:
        parent_programs = {identifier: parent_programs[identifier] for identifier in declared_order}
    if type(compute_limit) is not int or compute_limit < 0:
        raise ValueError('native representation compute_limit must be nonnegative')
    if type(max_candidates) is not int or max_candidates < 2:
        raise ValueError('native representation research needs at least two candidate slots')
    if spec.risk_quantile != .5:
        raise ValueError('representation collision research currently uses the MSE contract')
    if retained_replay_scope is None:
        retained_replay_scope = (question.get('native_representation_plan') or {}).get(
            'retained_replay_scope', 'procedure')
    if retained_replay_scope not in {'procedure', 'retained_coordinates'}:
        raise ValueError('retained_replay_scope must be procedure or retained_coordinates')
    revision = question.get('kind') == 'program_revision'
    scope_plan = (question.get('native_representation_plan') or {}).get('scope_plan')
    scope_limit = 0
    if scope_plan is not None:
        if (scope_plan.get('purpose') != 'relative_utility'
                or scope_plan.get('objective') not in {'pointwise_improvement', 'mean_loss'}
                or type(scope_plan.get('max_compute')) is not int or scope_plan['max_compute'] < 0):
            raise ValueError('scope research needs its declared relative-utility objective and compute cap')
        scope_limit = scope_plan['max_compute']
    representation_state, alternative_basis = _representation_search_basis(
        question, parent_programs, spec.feature_names, questions=agent.questions,
        resolve=lambda reference: load_source_values([reference], agent.workspace)[0]['value'])
    training, validation, split = agent._split({'validation_records': validation_records})
    if not validation:
        raise ValueError('representation research needs distinct validation trajectories')
    # Persist the exact split before any target-dependent proposal or fitting.
    # The ordinary splitter groups equal public reset trajectories together.
    question['native_representation_split'] = deepcopy(split)
    reused_directions, reuse_receipts = _reuse_paid_directions(paid_direction_reuse,
        agent, spec, parent_programs, split, retained_replay_scope)
    rows = lambda values: [Observation(dict(row['features']), float(row['target']),
        tuple(dict(point) for point in row.get('history', []))) for row in values]
    training, validation = rows(training), rows(validation)
    witness_groups = (_training_representation_witnesses(
        question.get('representation_diagnostic') or {}, training)
        if representation_state == 'proven_insufficient_on_evidence' else [])
    budget = SearchBudget(min(compute_limit, agent._remaining_compute()))
    threshold = agent.config.confirmation_mse_threshold
    researcher = UnifiedResearchAgent(ResearchConfig(
        compute_budget=max(1, budget.limit), max_candidates=max_candidates,
        minimum_improvement=threshold, active_probes=False))
    # The witness repair exports into the full-row strong-pool comparison.
    # Its local winner must use that same objective; other historical native
    # inquiries retain their declared trajectory-validation selection.
    full_development = bool(witness_groups)

    def best(claims):
        return (_best_development_claim(claims, len(training), len(validation), threshold)
                if full_development else researcher._best(claims))

    def development_loss(entry):
        return ({'development_mse': _development_claim_key(
            entry, len(training), len(validation))[0]} if full_development else {})

    fallback = initial_prediction(training, .5)
    origin = deepcopy(question.get('origin'))
    if origin is None and not revision:
        origin = ('computed_representation_conflict' if representation_state ==
                  'proven_insufficient_on_evidence' else alternative_basis or representation_state)
    result = {'status': 'native_representation_inconclusive_budget',
        'origin': origin, 'question_id': question['id'],
        'research_kind': 'program_revision' if revision else 'representation',
        'representation_evidence_state': representation_state,
        'alternative_representation_basis': alternative_basis,
        'parent_theory_ids': list(parent_programs),
        'retained_replay_scope': retained_replay_scope,
        'split': deepcopy(split), 'mse_threshold': threshold,
        'retained_comparison': [], 'candidates': [], 'actions': [],
        'selected': None, 'fitting_rows': len(training),
        'validation_rows': len(validation), 'provider_calls': 0,
        'new_observations': 0, 'all_parent_candidates_retained': True,
        'evidence_scope': 'Already exposed development material; whole-trajectory '
            'validation selects a working draft, not independent confirmation.'}
    if witness_groups:
        result['evidence_scope'] = ('Already exposed development material; full-row '
            'MSE selects a working draft. The separate trajectory losses are '
            'retention diagnostics, not independent confirmation.')
        result['selection_objective'] = {'kind': 'mse', 'weighting': 'uniform_rows',
            'data_role': 'exposed_development', 'training_rows': len(training),
            'validation_rows': len(validation), 'independent_confirmation': False}
    if paid_direction_reuse:
        result['paid_direction_reuse'] = deepcopy(paid_direction_reuse)
    # Fund all retained comparisons, every possible proposed holdout comparison,
    # and the actual selected-vs-retained prediction trace before search begins.
    baseline_cost = len(parent_programs) * (len(training) + len(validation))
    directions = (('retained_composition',) if revision and alternative_basis is None else
                  ('coordinates', 'retained_composition'))
    if witness_groups:
        directions = (*directions, 'witness_coordinates')
    # The historical discrete learner lived in the older controller. Binary
    # observed responses plus an actual revision/representation question make
    # it an available competitor here; no apparatus family is consulted.
    discrete_state = (len({row.target for row in training}) == 2
        and any(row.history for row in training) and (revision or bool(witness_groups)))
    if discrete_state:
        directions = ('discrete_state', *directions)
    # The old local state repair preserves the learned mature transition. It
    # decides from paid startup errors whether any repair is needed; the
    # presence of a replay node alone does not supply replacement parameters.
    startup_parents = {}
    if len({row.target for row in training}) == 2:
        for identifier, program in parent_programs.items():
            expanded = program.expanded()
            if (expanded.op == 'state_replay' and expanded.children[0].op == 'if'
                    and expanded.children[0].children[0].op == 'available'):
                startup_parents[identifier] = program
    startup_priority = (question.get('native_representation_plan') or {}).get(
        'startup_parent_priority')
    if startup_priority is not None:
        if (not isinstance(startup_priority, list)
                or not all(isinstance(identifier, str) for identifier in startup_priority)
                or len(set(startup_priority)) != len(startup_priority)
                or any(identifier not in startup_parents for identifier in startup_priority)):
            raise ValueError('startup_parent_priority must name distinct retained startup parents')
        # A method may bind a paid witness to the finite generation slots.
        # Preserve all parents and their comparison order; only the local
        # fitter's emission order changes, within the original candidate cap.
        startup_parents = {identifier: startup_parents[identifier] for identifier in
            [*startup_priority, *(identifier for identifier in startup_parents
                                 if identifier not in startup_priority)]}
        result['startup_parent_binding'] = {
            'priority': list(startup_priority),
            'witness_refs': deepcopy((question.get('native_representation_plan') or {}).get(
                'startup_parent_witness_refs', [])),
            'generation_parent_order': list(startup_parents),
            'max_candidates': max_candidates}
    if startup_parents:
        directions = ('startup_revision', *directions)
        # Algorithm selection must not change the working comparison contract.
        # A different direction still has to preserve every paid counterexample
        # that the local startup repair is required to explain.
        full_development = True
        result['evidence_scope'] = ('Already exposed development material; full-row '
            'MSE selects a working draft. Separate trajectory losses remain '
            'diagnostics, not independent confirmation.')
        result['selection_objective'] = {'kind': 'mse', 'weighting': 'uniform_rows',
            'data_role': 'exposed_development', 'training_rows': len(training),
            'validation_rows': len(validation), 'independent_confirmation': False}
    result['available_search_directions'] = list(directions)
    requested_directions = (question.get('native_representation_plan') or {}).get('search_directions')
    if requested_directions is not None:
        if (not isinstance(requested_directions, list) or not requested_directions
                or not all(isinstance(item, str) for item in requested_directions)
                or len(set(requested_directions)) != len(requested_directions)
                or any(item not in directions for item in requested_directions)):
            raise ValueError('search_directions must be a nonempty distinct list of current available directions')
        result['direction_selection'] = {'available': list(directions),
            'selected': list(requested_directions), 'source': 'native_representation_plan'}
        directions = tuple(requested_directions)
    try:
        procedures = retained_native_research_programs(
            [] if retained_research_program_refs is None else retained_research_program_refs,
            parent_programs, agent.workspace)
        # Saved procedure replays supplement the original grammar's slots; they
        # must not evict its last candidate by occupying the old prefix cap.
        compared_capacity = max(2 * max_candidates,
            len(directions) * max_candidates + len(procedures))
        reserved = (compared_capacity + 2) * len(validation)
        if not budget.can_spend(baseline_cost + reserved + scope_limit):
            result['required_comparison_compute'] = baseline_cost + reserved + scope_limit
            return result
        retained = []
        for identifier, program in parent_programs.items():
            budget.spend(len(training))
            retained.append({'candidate': Candidate(str(identifier), program, 'retained_parent'),
                'training_loss': loss(program, training, fallback, .5), 'retained': True})
        retained = researcher._assess(retained, validation, budget, fallback, .5)
        result['retained_comparison'] = [{'id': entry['candidate'].candidate_id,
            'version': (getattr(agent, 'theory_index', {}).get(entry['candidate'].candidate_id) or {}).get('version'),
            'nodes': entry['candidate'].program.size, 'training_mse': entry['training_loss'],
            'validation_mse': entry['validation_loss'], **development_loss(entry)} for entry in retained]
        best_retained = best(retained)
        allowance = budget.remaining - reserved - scope_limit
        proposals = []
        # The existing narrow revision remains a valid local experiment. A
        # concrete missing-distinction warrant opens the ordinary bounded
        # coordinate path without declaring that alternative sufficient.
        for index, direction in enumerate(directions):
            # Cumulative ceilings preserve later work while carrying earlier
            # unused capacity forward. The final direction receives the balance.
            ceiling = allowance * (index + 1) // len(directions)
            spent_search = sum(item['compute_cost'] for item in result.get('search_directions', []))
            local = SearchBudget(max(0, ceiling - spent_search))
            direction_receipt = {'direction': direction, 'compute_limit': local.limit}
            result.setdefault('search_directions', []).append(direction_receipt)

            def complete_state_attempt(run):
                nonlocal local
                claims = run(local)
                denied = local.last_denied_request
                remaining_search = allowance - spent_search
                if (not claims and local.spent == 0 and denied is not None
                        and local.limit < denied['required_compute'] <= remaining_search):
                    # The solver declined before doing work because an equal
                    # share cannot fund its existing complete-entry contract.
                    # Use this unit's unspent search allowance; retain every
                    # comparison reserve and the original total ceiling.
                    direction_receipt['allocation_revision'] = {
                        'reason': 'complete_entry_exceeds_equal_share',
                        'initial_compute_limit': local.limit,
                        'declined_entry': deepcopy(denied),
                        'allocated_compute_limit': remaining_search}
                    local = SearchBudget(remaining_search)
                    direction_receipt['compute_limit'] = local.limit
                    claims = run(local)
                return claims

            try:
                if direction in reused_directions:
                    bounded_claims = reused_directions[direction]
                    capacity = max_candidates + (len(procedures) if direction == 'retained_composition' else 0)
                    if len(bounded_claims) > capacity:
                        raise ValueError('paid reused proposals exceed this direction capacity')
                    direction_receipt.update(reuse_receipts[direction])
                elif direction == 'startup_revision':
                    from .state_search import closed_startup_claims

                    bounded_claims = complete_state_attempt(lambda allocation:
                        closed_startup_claims(list(startup_parents.values()),
                            training, budget=allocation, max_candidates=max_candidates))
                    for claim in bounded_claims:
                        claim['state_research'].update(
                            fit_record_indices=deepcopy(split['training_records']))
                    result['actions'].append({'action': 'revise_paid_startup_condition',
                        'parent_theory_ids': list(startup_parents),
                        'training_records': deepcopy(split['training_records']),
                        'validation_records_used_for_fit': [],
                        'candidates': len(bounded_claims), 'compute_cost': local.spent,
                        'preserved_components': ['history_guard', 'mature_transition']})
                    if 'startup_parent_binding' in result:
                        direction_receipt['parent_binding'] = deepcopy(result['startup_parent_binding'])
                elif direction == 'discrete_state':
                    from .state_search import closed_state_claims

                    bounded_claims = complete_state_attempt(lambda allocation:
                        closed_state_claims(
                            [agent.observations[index] for index in split['training_records']],
                            budget=allocation, max_history=researcher.config.max_history,
                            max_candidates=max_candidates, record_indices=split['training_records']))
                    result['actions'].append({'action': 'learn_discrete_state_from_paid_trajectories',
                        'training_records': deepcopy(split['training_records']),
                        'candidates': len(bounded_claims),
                        'prediction_inputs': 'controls_and_self_predicted_state_from_reset',
                        'validation_records_used_for_fit': [], 'compute_cost': local.spent})
                elif direction == 'coordinates':
                    claims = researcher._coordinate_claims(spec, training, local, result['actions'])
                    bounded_claims = claims[:max_candidates]
                elif direction == 'witness_coordinates':
                    bounded_claims = _witness_coordinate_claims(spec, training,
                        parent_programs, witness_groups, local, result['actions'],
                        max_candidates=max_candidates, validation=validation, fallback=fallback)
                else:
                    replayed, completed_procedures = [], []
                    for procedure in procedures:
                        source_ref = procedure['source_ref']
                        replay_receipt = {'action': 'replay_retained_parent_research_program',
                            'source_ref': deepcopy(source_ref),
                            'execution_scope': retained_replay_scope,
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
                                training, local, risk_quantile=spec.risk_quantile,
                                execution_scope=retained_replay_scope)
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
            for claim in bounded_claims:
                claim.setdefault('search_direction', direction)
            if direction == 'startup_revision' and bounded_claims:
                # A validation-only tie must not retain a model still refuted
                # by a paid startup. Use the existing full-row comparison for
                # actual local repairs, preserving all known counterexamples.
                if not full_development:
                    full_development = True
                    result['evidence_scope'] = ('Already exposed development material; full-row '
                        'MSE selects a working draft. Separate trajectory losses remain '
                        'diagnostics, not independent confirmation.')
                    result['selection_objective'] = {'kind': 'mse', 'weighting': 'uniform_rows',
                        'data_role': 'exposed_development', 'training_rows': len(training),
                        'validation_rows': len(validation), 'independent_confirmation': False}
                    for receipt, entry in zip(result['retained_comparison'], retained):
                        receipt.update(development_loss(entry))
                    best_retained = best(retained)
            # Every direction can satisfy the same paid precision contract.
            # Assess its returned programs before allocating the next search,
            # sharing scores already paid on this exact validation view.
            paid_validation = {entry['candidate'].program: entry['validation_loss']
                for entry in [*retained, *proposals, *bounded_claims] if 'validation_loss' in entry}
            pending = {entry['candidate'].program: entry for entry in bounded_claims
                       if entry['candidate'].program not in paid_validation}
            start_validation = budget.spent
            assessed_direction = researcher._assess(list(pending.values()), validation, budget, fallback, .5)
            direction_receipt['validation_compute'] = budget.spent - start_validation
            paid_validation.update({entry['candidate'].program: entry['validation_loss']
                                    for entry in assessed_direction})
            if any(entry['candidate'].program not in paid_validation for entry in bounded_claims):
                result['compared_candidates'] = sum(entry['candidate'].program in paid_validation
                                                    for entry in bounded_claims)
                return result
            bounded_claims = [{**entry, 'validation_loss': paid_validation[entry['candidate'].program]}
                              for entry in bounded_claims]
            proposals.extend(bounded_claims)
            if any(
                    claim['training_loss'] <= threshold and claim['validation_loss'] <= threshold
                    for claim in bounded_claims):
                result['search_stop'] = {'reason': ('precision_satisfied_after_local_repair'
                    if direction == 'startup_revision' else 'precision_satisfied_after_direction'),
                    'completed_direction': direction, 'mse_threshold': threshold,
                    'unexecuted_directions': list(directions[index + 1:]),
                    'fitting_records': deepcopy(split['training_records']),
                    'validation_records': deepcopy(split['validation_records']),
                    'independent_confirmation': False}
                break
        unique = {}
        for entry in proposals:
            if entry['candidate'].program not in parent_programs.values():
                previous = unique.setdefault(entry['candidate'].program, entry)
                if 'validation_loss' in entry and 'validation_loss' not in previous:
                    unique[entry['candidate'].program] = {
                        **previous, 'validation_loss': entry['validation_loss']}
        # Witness repairs have already paid for this exact trajectory view.
        fresh = researcher._assess([entry for entry in unique.values()
            if 'validation_loss' not in entry], validation, budget, fallback, .5)
        assessed = {entry['candidate'].program: entry for entry in fresh}
        checked = [entry if 'validation_loss' in entry else assessed[program]
            for program, entry in unique.items()
            if 'validation_loss' in entry or program in assessed]
        if len(checked) != len(unique):
            # Incomplete comparisons never select a new draft.
            result['compared_candidates'] = len(checked)
            return result
        winner = best([*retained, *checked])
        if scope_plan is not None:
            from .condition_learning import learn_relative_utility_condition

            local = SearchBudget(scope_limit)
            utility = {'status': 'insufficient_budget', 'scope_purpose': 'relative_utility',
                'application_scope_status': 'not_assessed', 'compute_cost': 0,
                'fitting_records': deepcopy(split['training_records']),
                'selection_records': deepcopy(split['validation_records']),
                'independent_confirmation': False}
            result['relative_utility_research'] = utility
            try:
                if not checked:
                    utility['status'] = 'no_new_draft'
                elif not local.spend(2 * len(training)):
                    utility['last_denied_request'] = deepcopy(local.last_denied_request)
                else:
                    # Fix both branches using training losses before using any
                    # validation result to select a scope-construction input.
                    train_key = lambda item: (item['training_loss'], item['candidate'].program.size)
                    draft = min(checked, key=train_key)
                    retained_scope = min(retained, key=train_key)
                    utility['draft_candidate_ref'] = {'json_path': ['candidates', checked.index(draft)]}
                    utility['baseline_candidate_ref'] = {'json_path': ['retained_comparison', retained.index(retained_scope)]}
                    original = draft['candidate'].program
                    baseline = retained_scope['candidate'].program
                    before = [predict(baseline, row, fallback) for row in training]
                    after = [predict(original, row, fallback) for row in training]
                    utility.update(learn_relative_utility_condition(training, before, after, local,
                        objective=scope_plan['objective'], max_history=scope_plan.get('max_history', 4),
                        world_mse_threshold=threshold))
                    selector = utility.get('selector')
                    if selector is not None:
                        selector = Program.from_dict(selector)
                        if selector.op == 'const':
                            utility['reused_candidate_ref'] = deepcopy(utility[
                                'draft_candidate_ref' if selector.value else 'baseline_candidate_ref'])
                            scoped = original if selector.value else baseline
                        else:
                            scoped = Program('if', children=(selector, original, baseline))
                        if (scoped not in parent_programs.values()
                                and not any(item['candidate'].program == scoped for item in checked)):
                            if local.spend(len(training) + len(validation)):
                                entry = {'candidate': Candidate('relative_utility_scope', scoped,
                                    'learned_relative_utility_selector'),
                                    'training_loss': loss(scoped, training, fallback, .5),
                                    'validation_loss': loss(scoped, validation, fallback, .5),
                                    'relative_utility_scope': {
                                        key: deepcopy(utility[key]) for key in
                                        ('draft_candidate_ref', 'baseline_candidate_ref')}}
                                checked.append(entry)
                                utility['candidate_id'] = f'native_representation:{len(checked) - 1}'
                            else:
                                utility['candidate_status'] = 'comparison_unfunded'
                                utility['last_denied_request'] = deepcopy(local.last_denied_request)
            finally:
                budget.spend(local.spent)
                utility.update(compute_cost=local.spent, compute_limit=local.limit)
            winner = best([*retained, *checked])
        result['candidates'] = [{'id': f"native_representation:{index}",
            'program': entry['candidate'].program.to_dict(),
            'nodes': entry['candidate'].program.size,
            'training_mse': entry['training_loss'], 'validation_mse': entry['validation_loss'],
            **development_loss(entry),
            **({'method_source_refs': deepcopy(entry['method_source_refs'])}
               if entry.get('method_source_refs') else {}),
            **({'representation_witness_check': deepcopy(entry['representation_witness_check'])}
               if entry.get('representation_witness_check') else {}),
            **({'state_research': deepcopy(entry['state_research'])}
               if entry.get('state_research') else {}),
            **{key: deepcopy(entry[key]) for key in ('search_direction', 'paid_candidate_ref')
               if key in entry},
            'research_program': deepcopy(entry.get('research_program')),
            **({'relative_utility_scope': deepcopy(entry['relative_utility_scope'])}
               if entry.get('relative_utility_scope') else {})}
            for index, entry in enumerate(checked)]
        selected_program = winner['candidate'].program
        budget.spend(2 * len(validation))
        before = [predict(best_retained['candidate'].program, row, fallback) for row in validation]
        after = [predict(selected_program, row, fallback) for row in validation]
        is_retained = winner.get('retained', False)
        result['selected'] = {'id': winner['candidate'].candidate_id if is_retained else
            f"native_representation:{checked.index(winner)}", 'retained': is_retained,
            'program': selected_program.to_dict(), 'nodes': selected_program.size,
            'training_mse': winner['training_loss'], 'validation_mse': winner['validation_loss'],
            **development_loss(winner),
            'best_retained_id': best_retained['candidate'].candidate_id,
            'best_retained_validation_mse': best_retained['validation_loss'],
            **({'best_retained_development_mse': development_loss(best_retained)['development_mse']}
               if full_development else {}),
            'validation_predictions': after, 'best_retained_validation_predictions': before,
            'changed_prediction_rows': sum(left != right for left, right in zip(before, after)),
            'max_prediction_change': max(abs(left - right) for left, right in zip(before, after)),
            'research_program': deepcopy(winner.get('research_program'))}
        result['selected']['selection'] = {
            'status': 'working_draft', 'data_role': ('exposed_development' if full_development
                                                  else 'exposed_development_validation'),
            'candidate_ref': {'json_path': (['retained_comparison', retained.index(winner)]
                if is_retained else ['candidates', checked.index(winner)])},
            'fitting_records': None if is_retained else deepcopy(split['training_records']),
            'fitting_provenance': ('retained_parameters_original_fit_not_established'
                if is_retained else 'new_fits_use_current_training_records'),
            'inherited_parameter_provenance': 'original_component_fitting_views_not_established',
            'selection_records': (sorted(set(split['training_records'] + split['validation_records']))
                                  if full_development else deepcopy(split['validation_records'])),
            **({'diagnostic_validation_records': deepcopy(split['validation_records'])}
               if full_development else {}),
            'competitor_refs': [{'json_path': ['retained_comparison']}, {'json_path': ['candidates']}],
            'comparison_scope': 'declared_parents_and_current_native_candidates',
            'loss': {'kind': 'mse', 'weighting': 'uniform_rows', 'risk_quantile': .5},
            'independent_confirmation': False,
            'uncompared_archive_validation': 'not_established'}
        result.update(status='native_representation_draft_selected' if not is_retained
            else 'native_representation_parent_retained', comparison_completed=True,
            remaining_question=('Compare this actual parent revision against the complete '
                'strong pool; a proposed or unadopted descendant is not completed recursion.'
                if revision else 'Verify the actual exposed representation and frozen draft '
                'against its paid witnesses before resolving the representation child.'))
        frontier = _representation_experiment_frontier(
            agent, question, spec, [*retained, *checked], winner, budget, threshold)
        if frontier is not None:
            result['experiment_frontier'] = frontier
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
        unfinished = []
        for direction in result.get('search_directions', []):
            if direction.get('last_denied_request'):
                unfinished.append({'stage': direction['direction'],
                    'last_denied_request': deepcopy(direction['last_denied_request'])})
        for action in result['actions']:
            for branch in action.get('branches', []):
                if branch.get('last_denied_request'):
                    unfinished.append({'stage': branch['direction'],
                        'last_denied_request': deepcopy(branch['last_denied_request']),
                        'progress': deepcopy(branch.get('progress', {}))})
        result['incomplete_search_stages'] = unfinished
        result['research_completion'] = ('bounded_search_incomplete' if unfinished
            else 'bounded_search_completed' if result.get('comparison_completed')
            else 'comparison_incomplete')
        result['native_host_seconds_before_persistence'] = monotonic() - host_started
        result['host_time_accounting'] = ('Native host duration is separate from the existing '
            'external Python execution ledger; this function does not debit that ledger.')
        _retain_native_representation_result(agent, question, result)

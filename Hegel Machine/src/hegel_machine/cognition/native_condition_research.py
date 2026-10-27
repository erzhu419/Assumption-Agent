"""Adapt paid public trajectories to the retained condition research inner loop.

This is the finite arithmetic/threshold/window prior of the native researcher,
not a second apparatus or a replacement for independent confirmation. It only
reads purchased records; every proposed purchase is returned to its caller.
"""

from copy import deepcopy
from itertools import combinations_with_replacement

from .conditional_coordinates import (
    conditional_search_allowance, propose_experimental_conditional_coordinates,
    revise_conditional_threshold,
)
from .conditional_experiments import (
    _separation, assess_conditional_experiment, propose_conditional_experiments,
)
from .composed_conditions import (
    propose_composed_conditions, propose_measured_window_response,
)
from .evaluation import initial_prediction, loss
from .language import Program
from .models import Observation, TaskSpec


FINITE_PRIOR = (
    "Retained native finite search: public scalar controls, linear/quadratic "
    "response terms, measured threshold intervals, finite window sums, "
    "inside/outside condition placement and two-condition AND/OR/XOR. "
    "An explicit author basis replaces the default response terms. "
    "This action fits purchased evidence; it does not independently confirm."
)


def _spec(public_spec):
    controls = tuple(public_spec['controls'])
    if 'trajectory' not in public_spec['action_kinds']:
        raise ValueError('native condition research requires public trajectory actions')
    return TaskSpec(
        task_id='public_paid_condition_research', description=FINITE_PRIOR,
        feature_names=controls, initial_probes=(), available_probes=(),
        feature_bounds={name: tuple(public_spec['feature_bounds'][name])
                        for name in controls},
        max_probe_length=int(public_spec['max_probe_length']),
    )


def _rows(record):
    return [Observation(dict(row['features']), float(row['target']),
                        tuple(dict(point) for point in row.get('history', [])))
            for row in record['rows']]


def _pair(spec, observations, reference_index, record_index):
    """Recognize the actual contract required by the old matched-delta solver."""
    reference, intervention = observations[reference_index], observations[record_index]
    count = min(len(reference['rows']), len(intervention['rows']))
    original, changed = reference['rows'][:count], intervention['rows'][:count]
    if count < 8:
        raise ValueError('matched reset prefixes must contain at least eight rows')
    if original[0].get('history') or changed[0].get('history'):
        raise ValueError('matched trajectories must start at the public reset, not a suffix')
    names = spec.feature_names
    background = {name: original[0]['features'][name] for name in names}
    if any(any(row['features'][name] != background[name] for name in names)
           for row in original):
        raise ValueError('matched reference must retain one constant public background')
    varying = [name for name in names
               if any(row['features'][name] != background[name] for row in changed)]
    if len(varying) != 1:
        raise ValueError('matched intervention must vary exactly one public control')
    # The fitter reads every earlier control, so a matched present row cannot
    # conceal a changed background in its recorded history.
    for trajectory in (original, changed):
        prefix = []
        for row in trajectory:
            if list(row.get('history', [])) != prefix:
                raise ValueError('matched rows must preserve their complete reset history')
            prefix.append(dict(row['features']))
    control = varying[0]
    return {'reference_record_index': reference_index, 'record_index': record_index,
            'control': control, 'background': background, 'mode': 'paid_matched',
            'levels': sorted({row['features'][control] for row in changed}),
            'matched_rows': len(changed),
            'scope': 'Equal-length measured prefixes from the same public reset.'}


def _available_pairs(spec, observations):
    references = {}
    for index, record in enumerate(observations):
        rows = record.get('rows', [])
        if len(rows) < 8 or rows[0].get('history'):
            continue
        background = rows[0]['features']
        if all(all(row['features'].get(name) == background.get(name)
                   for name in spec.feature_names) for row in rows):
            key = tuple(background[name] for name in spec.feature_names)
            previous = references.get(key)
            if previous is None or len(rows) > len(observations[previous]['rows']):
                references[key] = index
    pairs = []
    for index, record in enumerate(observations):
        if len(record.get('rows', [])) < 8:
            continue
        for reference in references.values():
            if reference == index:
                continue
            try:
                pairs.append(_pair(spec, observations, reference, index))
            except ValueError:
                continue
    return pairs


def _pair_input_differences(spec, observations, reference_index, record_index):
    """Describe selected experimental inputs without interpreting their responses."""
    reference, changed = observations[reference_index], observations[record_index]
    count = min(len(reference['rows']), len(changed['rows']))
    changes = {}
    for name in spec.feature_names:
        steps = [{'step': step, 'reference': left['features'][name],
                  'intervention': right['features'][name]}
                 for step, (left, right) in enumerate(zip(reference['rows'], changed['rows']))
                 if left['features'][name] != right['features'][name]]
        if steps:
            changes[name] = steps
    return {'reference_record_index': reference_index, 'record_index': record_index,
            'compared_rows': count, 'changed_controls': list(changes),
            'control_changes': changes,
            'reference_evidence': reference['row_indices'][:count],
            'evidence': changed['row_indices'][:count]}


def _matched_groups(spec, pairs):
    """Pool paid levels without mixing controls, backgrounds or reset histories."""
    groups = {}
    for pair in pairs:
        key = (pair['control'], tuple(pair['background'][name] for name in spec.feature_names))
        groups.setdefault(key, []).append(pair)
    return list(groups.values())


def _assess_matched_group(spec, observations, pairs, budget):
    reference_rows, measured_rows = [], []
    reference_evidence, evidence = [], []
    for pair in pairs:
        reference, changed = pair['reference_record_index'], pair['record_index']
        count = pair['matched_rows']
        reference_rows.extend(_rows(observations[reference])[:count])
        measured_rows.extend(_rows(observations[changed])[:count])
        reference_evidence.extend(observations[reference]['row_indices'][:count])
        evidence.extend(observations[changed]['row_indices'][:count])
    # Observation histories remain independent reset prefixes; concatenating
    # the samples must not turn successive interventions into one trajectory.
    outcome = assess_conditional_experiment(
        spec, pairs[0], reference_rows, measured_rows, budget)
    outcome.update(reference_evidence=reference_evidence, evidence=evidence)
    if len(pairs) == 1:
        outcome.update(reference_record_index=pairs[0]['reference_record_index'],
                       record_index=pairs[0]['record_index'], scope=pairs[0]['scope'])
    else:
        outcome.update(
            matched_pairs=[{key: pair[key] for key in
                            ('reference_record_index', 'record_index', 'matched_rows')}
                           for pair in pairs],
            reference_record_indices=list(dict.fromkeys(
                pair['reference_record_index'] for pair in pairs)),
            record_indices=list(dict.fromkeys(pair['record_index'] for pair in pairs)),
            scope='Pooled matched reset prefixes with the same control and constant background; '
                  'each row retains its own complete reset history.')
    return outcome


def _basis(names):
    features = [Program('feature', name=name) for name in names]
    return [*features, *(Program('mul', children=(left, right))
                        for left, right in combinations_with_replacement(features, 2))]


def _program(value):
    return value if isinstance(value, Program) else Program.from_dict(value)


def _proposals(public_spec, spec, rows, programs, budget, allowance):
    packs = propose_conditional_experiments(
        spec, rows, programs, budget, observation_budget=allowance)
    proposals = []
    for pack in packs:
        probes = [pack['reference'], *(item['probe'] for item in pack['interventions'])]
        sequences = [[dict(point) for point in probe.sequence] for probe in probes]
        levels = public_spec.get('control_levels', {})
        if levels:
            # The historical proposer uses continuous bounds. Re-express its
            # actual proposed controls on the public instrument's legal grid.
            for sequence in sequences:
                for point in sequence:
                    for name, legal in levels.items():
                        point[name] = min(legal, key=lambda value: abs(value - point[name]))
        pairs = []
        for index, plan in enumerate(pack['interventions'], 1):
            control = plan['control']
            if sequences[index] == sequences[0]:
                continue
            pairs.append({'reference_action_index': 0, 'action_index': index,
                          'control': control,
                          'background': dict(sequences[0][0]),
                          'levels': sorted({point[control] for point in sequences[index]}),
                          'predicted_separation': _separation(
                              programs, sequences[0], sequences[index], budget)})
        if pairs:
            proposals.append({
                'query_action': {'action': 'query', 'arguments': {'actions': [
                    {'kind': 'trajectory', 'channels': [], 'sequence': sequence}
                    for sequence in sequences]}},
                'matched_action_pairs': pairs,
                'observation_cost': sum(map(len, sequences)),
                'status': 'proposed_not_purchased',
            })
    return proposals


def _boundary_proposals(public_spec, candidates, evidence, budget, allowance):
    """Offer the historical joint-fit boundary experiment without buying it.

    The two endpoint thresholds agree on the paid bracket but disagree at its
    midpoint. A later ordinary fit includes the purchased reset row among all
    paid controls, so its fitted bracket reflects the measured outcome.
    """
    if allowance < 1:
        return []
    for candidate in sorted(candidates, key=lambda item: item['fit_mse']):
        if candidate['fit_mse'] > 1e-10:
            break
        steps = candidate.get('research_program', {}).get('selection_steps', [])
        step = next((item for item in reversed(steps)
                     if item.get('operation') == 'discover_conditional_coordinates'), None)
        if step is None:
            continue
        control = step['gate_coordinate']['name']
        source = next((item for item in evidence if item.get('control') == control
                       and item.get('gain', 0.) > 1e-10), None)
        if source is None:
            continue
        left, right = step['threshold_interval']
        middle = (left + right) / 2.
        if control in public_spec.get('control_levels', {}):
            legal = [value for value in public_spec['control_levels'][control]
                     if left < value < right]
            if not legal:
                continue
            middle = min(legal, key=lambda value: abs(value - middle))
        point = {**source['background'], control: middle}
        response = _program(candidate['program'])
        if not budget.spend(2 * response.size + 2):
            return []
        alternatives = [revise_conditional_threshold(response, control, value)
                        for value in (left, right)]
        predictions = [program.predict(Observation(point, 0.)) for program in alternatives]
        separation = (predictions[0] - predictions[1]) ** 2
        if separation <= 1e-10:
            continue
        return [{'query_action': {'action': 'query', 'arguments': {'actions': [
                    {'kind': 'trajectory', 'channels': [], 'sequence': [point]}]}},
                 'matched_action_pairs': [], 'observation_cost': 1,
                 'purpose': 'conditional_boundary', 'control': control,
                 'threshold_interval': [left, right],
                 'interval_width': right - left,
                 'control_range': list(public_spec['feature_bounds'][control]),
                 'branch_predictions': predictions, 'predicted_separation': separation,
                 'candidate_source': candidate['source'],
                 'candidate_program': deepcopy(candidate['program']),
                 'status': 'proposed_not_purchased'}]
    return []


def _research_decision(result):
    incomplete = [stage for stage in result.get('search_stages', [])
                  if stage['status'] == 'computation_incomplete']
    if incomplete:
        result['research_decision'] = {
            'kind': 'computation_incomplete',
            'reason': 'The selected compute allocation left requested search work unexecuted. '
                      'Returned old-program scores and completed fits remain valid on their paid rows; '
                      'they do not test the unexecuted candidates or refute this research family.',
            'stages': [stage['stage'] for stage in incomplete],
            'completed_joint_fits': sum(stage['completed_fits'] for stage in result['search_stages']
                                       if stage['stage'].startswith('joint_')),
        }
        return
    boundary = next((item for item in result['proposed_experiments']
                     if item.get('purpose') == 'conditional_boundary'), None)
    if boundary:
        result['remaining_boundary'] = {key: deepcopy(boundary[key]) for key in
            ('control', 'threshold_interval', 'interval_width', 'control_range',
             'branch_predictions', 'predicted_separation')}
        result['research_decision'] = {
            'kind': 'remaining_parameter_ambiguity',
            'reason': 'A paid-fit candidate still permits different predictions inside its measured interval.',
            'next_experiment': deepcopy(boundary['query_action']),
        }
    else:
        result['research_decision'] = {
            'kind': 'no_discriminating_boundary',
            'reason': 'No boundary query returned within the current evidence and allowances; '
                      'this is not independent confirmation or proof of identification.',
        }


def update_native_boundary(public_spec, observations, budget, *, previous, proposal,
                           max_candidates=8, observation_budget=0, paid_record_indices=()):
    """Consume a purchased boundary using the retained two-endpoint experiment.

    This is the inexpensive interval-update stage of unified research: keep the
    response and history structure fixed, compare actual data to both endpoint
    predictions, then revise only the threshold. A conflicting observation is
    returned to the author for reselection/refitting, never hidden by an automatic
    whole-structure search. Old candidates remain scored on the same paid rows.
    """
    spec, before = _spec(public_spec), budget.spent
    start = previous.get('observed_record_count',
                         max(previous['fit_record_indices'], default=-1) + 1)
    paid = list(dict.fromkeys(paid_record_indices))
    if any(type(index) is not int or not 0 <= index < len(observations) for index in paid):
        raise ValueError('paid boundary records must select existing laboratory records')
    selected = list(dict.fromkeys([*previous['fit_record_indices'],
                                   *paid, *range(start, len(observations))]))
    rows = [row for position in selected for row in _rows(observations[position])]
    result = {key: deepcopy(previous[key]) for key in ('finite_prior', 'scope',
                                                       'evidence', 'available_pairs')}
    result.update(status='compute_budget_exhausted', candidates=[], searched_candidates=0,
        proposed_experiments=[], fit_record_indices=selected,
        observed_record_count=len(observations),
        fit_evidence=[index for position in selected for index in observations[position]['row_indices']])
    original = next(candidate for candidate in previous['candidates']
                    if candidate['program'] == proposal['candidate_program'])
    left, right = proposal['threshold_interval']
    update = {'status': 'compute_budget_exhausted', 'before_interval': [left, right],
              'after_interval': [left, right]}
    result['boundary_update'] = update
    result['research_decision'] = {'kind': 'compute_budget_exhausted',
        'reason': 'Retain the scored candidate; boundary evidence has not yet been resolved.'}
    fallback = initial_prediction(rows, spec.risk_quantile)

    def score(candidate):
        if not budget.spend(len(rows)):
            return False
        kept = {key: deepcopy(candidate[key]) for key in
                ('program', 'source', 'research_program') if key in candidate}
        kept.update(fit_mse=loss(_program(kept['program']), rows, fallback, spec.risk_quantile),
                    fit_evidence=list(result['fit_evidence']))
        result['candidates'].append(kept)
        return True

    def finish():
        result['candidates'].sort(key=lambda item: (item['fit_mse'], _program(item['program']).size))
        ranked = result['candidates']
        result['candidates'] = ranked[:max_candidates]
        # A query must name a candidate that is returned with this round. Keep
        # its updated interval when near-zero old/new fits tie at the list cap.
        if result['proposed_experiments'] and result['candidates']:
            active_program = result['proposed_experiments'][0]['candidate_program']
            if not any(item['program'] == active_program for item in result['candidates']):
                result['candidates'][-1] = next(item for item in ranked
                                              if item['program'] == active_program)
        if result['candidates']:
            result['status'] = 'candidates_require_evaluation'
        result['compute_cost'] = budget.spent - before
        return result

    # Score the incumbent before spending on research, so a constrained update
    # cannot silently replace it with an untested or worse parameter revision.
    if not score(original):
        return finish()
    action, = proposal['query_action']['arguments']['actions']
    acquired = next((position for position in [*paid, *range(start, len(observations))]
                     if observations[position]['action']['sequence'] == action['sequence']), None)
    if acquired is None:
        raise ValueError('boundary update requires its actual purchased trajectory')
    measured, = _rows(observations[acquired])
    response = _program(original['program'])
    if not budget.spend(2 * response.size + 2):
        return finish()
    control = proposal['control']
    predictions = [revise_conditional_threshold(response, control, threshold).predict(measured)
                   for threshold in (left, right)]
    errors = [(prediction - measured.target) ** 2 for prediction in predictions]
    side = min(range(2), key=errors.__getitem__)
    supported = errors[side] <= 1e-10
    update.update(observation_index=acquired, branch_predictions=predictions,
                  observed_target=measured.target, branch_errors=errors,
                  matched_branch=side if supported else None)
    if supported:
        middle = measured.features[control]
        left, right = (left, middle) if side == 0 else (middle, right)
        update.update(status='boundary_refined', after_interval=[left, right])
        revised = deepcopy(original)
        revised['program'] = revise_conditional_threshold(response, control, (left + right) / 2.).to_dict()
        revised['source'] = 'boundary_parameter_update'
        step = next(item for item in reversed(revised['research_program']['selection_steps'])
                    if item.get('operation') == 'discover_conditional_coordinates')
        step['threshold_interval'] = [left, right]
        step['threshold'] = (left + right) / 2.
        if not score(revised):
            return finish()
    else:
        update['status'] = 'structure_mismatch'
        result['research_decision'] = {'kind': 'structural_mismatch',
            'reason': 'Neither endpoint of the probed candidate predicts the new observation. '
                      'Reselect a supported candidate or explicitly refit/revise its response or structure.',
            'branch_errors': errors, 'observation_index': acquired}

    seen = {_program(item['program']) for item in result['candidates']}
    for candidate in sorted(previous['candidates'], key=lambda item: item['fit_mse']):
        program = _program(candidate['program'])
        if program not in seen and len(result['candidates']) < max_candidates:
            if not score(candidate):
                break
            seen.add(program)
    if supported:
        # Updated metadata follows the revised parameter, not an equally-fitting
        # old model whose broader bracket would repeat the same experiment.
        result['proposed_experiments'] = _boundary_proposals(
            public_spec, [result['candidates'][1]], result['evidence'], budget, observation_budget)
        _research_decision(result)
        if result['candidates'][1]['fit_mse'] > 1e-10:
            result['research_decision'] = {'kind': 'structural_mismatch',
                'reason': 'The endpoint update conflicts with other paid trajectories; '
                          'compare retained candidates and explicitly revise/refit.',
                'revised_fit_mse': result['candidates'][1]['fit_mse']}
    return finish()


def research_native_conditions(public_spec, observations, budget, *, record_indices=None,
                               matched_pairs=None, basis=None, programs=(),
                               retained_candidates=(), max_candidates=8, observation_budget=0):
    """Research author-selected paid evidence through the old scientific solver.

    ``record_indices`` selects complete trajectories for fitting. A matched pair
    names ``reference_record_index`` and ``record_index`` in the complete paid
    archive. Compatible pairs pool their measured levels for each control and
    fixed background. When omitted, all available pairs are used; an explicit
    list restricts the evidence to those pairs. Reset histories remain separate.
    ``basis`` and ``programs`` are Program objects or their public JSON form.
    Retained programs are scored on the same selected paid evidence before new
    candidates are proposed. The caller accounts ``budget.spent`` and exports
    the returned programs through the ordinary model evaluation path.
    """
    spec, before = _spec(public_spec), budget.spent
    selected = (list(range(len(observations))) if record_indices is None
                else list(dict.fromkeys(record_indices)))
    if any(index < 0 or index >= len(observations) for index in selected):
        raise ValueError('record_indices must select existing paid laboratory records')
    result = {'status': 'needs_matched_evidence', 'finite_prior': FINITE_PRIOR,
              'scope': 'Paid-data model search; no independent confirmation or automatic adoption.',
              'evidence': [], 'available_pairs': [], 'candidates': [],
              'searched_candidates': 0, 'observed_record_count': len(observations),
              'search_stages': [],
              'proposed_experiments': [], 'fit_record_indices': selected,
              'fit_evidence': [index for position in selected
                               for index in observations[position]['row_indices']]}

    def finish():
        result['compute_cost'] = budget.spent - before
        return result

    def stage(name, operation):
        start, denied, fits = budget.spent, budget.denied_requests, budget.completed_fits
        value = operation()
        limited = budget.denied_requests > denied
        result['search_stages'].append({
            'stage': name, 'status': 'computation_incomplete' if limited else 'attempt_finished',
            'compute_cost': budget.spent - start,
            'completed_fits': budget.completed_fits - fits,
            **({'blocked_work': deepcopy(budget.last_denied_request)} if limited else {}),
        })
        return value

    # Finding existing experiments is actual work over the paid archive too.
    if not budget.spend(sum(len(record['rows']) for record in observations)):
        result['status'] = 'compute_budget_exhausted'
        return finish()
    retained_info = {_program(item['program']): item for item in retained_candidates}
    retained = list(dict.fromkeys([*retained_info, *(_program(value) for value in programs)]))
    if any(not program.required_features() <= set(spec.feature_names) for program in retained):
        raise ValueError('retained programs must use public control coordinates')
    rows = [row for position in selected for row in _rows(observations[position])]
    if not rows:
        result['proposed_experiments'] = _proposals(
            public_spec, spec, rows, retained, budget, observation_budget)
        return finish()
    available = _available_pairs(spec, observations)
    result['available_pairs'] = available
    if matched_pairs is None:
        chosen = [item for name in spec.feature_names for item in available
                  if item['control'] == name]
    else:
        chosen = []
        invalid = []
        for item in matched_pairs:
            reference, changed = item['reference_record_index'], item['record_index']
            if not (0 <= reference < len(observations) and 0 <= changed < len(observations)):
                raise ValueError('matched_pairs must select existing paid laboratory records')
            try:
                pair = _pair(spec, observations, reference, changed)
                if item.get('control', pair['control']) != pair['control']:
                    raise ValueError('the supplied control differs from the actual matched intervention')
                chosen.append(pair)
            except ValueError as error:
                invalid.append({**_pair_input_differences(spec, observations, reference, changed),
                                'selection': deepcopy(item), 'error': str(error)})
        if invalid:
            result.update(status='research_not_executed', reason='invalid_matched_evidence',
                          error='Selected matched evidence is incompatible; no hypothesis was fitted '
                                'or refuted. Inspect the actual input differences, correct the selection '
                                'or omit matched_pairs to use available compatible paid trajectories.',
                          invalid_pairs=invalid)
            return finish()

    for program in retained:
        if not budget.spend(len(rows)):
            break
        result['candidates'].append({
            **{key: deepcopy(retained_info[program][key]) for key in
               ('research_program',) if program in retained_info and key in retained_info[program]},
            'program': program.to_dict(), 'source': 'retained_program',
            'fit_mse': loss(program, rows, initial_prediction(rows, spec.risk_quantile),
                            spec.risk_quantile),
            'fit_evidence': list(result['fit_evidence']),
        })

    for pairs in _matched_groups(spec, chosen):
        outcome = stage('matched_evidence',
                        lambda: _assess_matched_group(spec, observations, pairs, budget))
        result['evidence'].append(outcome)
        if outcome['status'] == 'budget_exhausted':
            break

    terms = _basis(spec.feature_names) if basis is None else [_program(value) for value in basis]
    if any(not term.required_features() <= set(spec.feature_names) for term in terms):
        raise ValueError('basis terms must use public control coordinates')
    claims = []
    evidence = result['evidence']
    if evidence:
        for name, proposer in (
                ('joint_window_response', propose_measured_window_response),
                ('joint_conditional_response', propose_experimental_conditional_coordinates),
                ('joint_composed_response', propose_composed_conditions)):
            claims.extend(stage(name, lambda: proposer(spec, rows, budget, basis=terms,
                evidence=evidence, max_candidates=max_candidates)))
    result['searched_candidates'] = len(claims)
    unique = {}
    for claim in claims:
        unique.setdefault(claim['candidate'].program, claim)
    claims = sorted(unique.values(), key=lambda claim:
                    (claim['training_loss'], claim['candidate'].program.size))[:max_candidates]
    for claim in claims:
        result['candidates'].append({
            'program': claim['candidate'].program.to_dict(),
            'source': claim['candidate'].candidate_id,
            'fit_mse': claim['training_loss'],
            'fit_evidence': list(result['fit_evidence']),
            'research_program': deepcopy(claim['research_program']),
        })
    if result['candidates']:
        result['status'] = 'candidates_require_evaluation'
    elif evidence:
        result['status'] = 'matched_evidence_without_candidate'
    if any(stage['status'] == 'computation_incomplete' for stage in result['search_stages']):
        result['status'] = 'search_incomplete'
        result['search_allowance_estimate'] = {
            'conditional_joint_attempt': conditional_search_allowance(
                len(spec.feature_names), len(rows), max(len(row.history) for row in rows)),
            'scope': 'Historical bounded joint-search estimate, not a guaranteed completion cost. '
                     'An explicit rerun repeats local assessment and rescoring on the same bank.',
        }
    boundaries = _boundary_proposals(
        public_spec, result['candidates'], evidence, budget, observation_budget)
    result['proposed_experiments'] = [*boundaries, *_proposals(
        public_spec, spec, rows, retained, budget, observation_budget - len(boundaries))]
    _research_decision(result)
    return finish()

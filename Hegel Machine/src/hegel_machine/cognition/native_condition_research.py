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
    propose_conditional_boundary_probe, propose_conditional_structure_experiment,
)
from .composed_conditions import (
    propose_composed_conditions, propose_measured_window_response,
)
from .evaluation import initial_prediction, loss
from .language import Program
from .models import Observation, TaskSpec
from .research_programs import replay_research_program


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


def _available_pairs(spec, observations, *, record_indices=None):
    indices = range(len(observations)) if record_indices is None else record_indices
    references = {}
    for index in indices:
        record = observations[index]
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
    for index in indices:
        record = observations[index]
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


def accumulate_matched_pairs(public_spec, observations, matched_pairs, scope):
    """Join new matched evidence to related paid reset experiments in one meaning.

    The caller binds ``record_indices`` to the source question's retained fit
    scope and the actual new callback records. This function reads controls and
    reset histories only; interpreting targets remains the original joint
    matched-delta solver's work.
    """
    spec = _spec(public_spec)
    indices = list(dict.fromkeys(scope['record_indices']))
    if (not public_spec.get('reset') or not indices
            or any(type(index) is not int or not 0 <= index < len(observations)
                   for index in indices)):
        raise ValueError('accumulated matched evidence requires an existing reset record scope')
    control, background = scope['control'], scope['background']
    if control not in spec.feature_names or set(background) != set(spec.feature_names):
        raise ValueError('accumulated matched evidence requires the actual control and complete background')
    selected, seen = [], set()
    for item in matched_pairs:
        reference, changed = item['reference_record_index'], item['record_index']
        if reference not in indices or changed not in indices:
            raise ValueError('the new matched pair lies outside the source meaning evidence scope')
        pair = _pair(spec, observations, reference, changed)
        if pair['control'] != control or pair['background'] != background:
            raise ValueError('the new matched pair differs from the retained structural experiment scope')
        key = reference, changed
        if key not in seen:
            selected.append(pair)
            seen.add(key)
    for pair in _available_pairs(spec, observations, record_indices=indices):
        key = pair['reference_record_index'], pair['record_index']
        if (pair['control'] == control and pair['background'] == background and key not in seen):
            selected.append(pair)
            seen.add(key)
    return selected


def matched_assessment_shape_upper(matched_rows, max_history, distinct_levels):
    """Price a matched group shape before its still-unpurchased labels exist."""
    widths = max(1, max_history)
    intervals = max(0, distinct_levels - 1)
    return matched_rows * (1 + 7 * widths + 22 * widths * intervals)


def matched_assessment_compute_upper(public_spec, observations, matched_pairs):
    """Bound only the existing matched-delta preparation, using public inputs.

    Each ``_Fitter.fit`` of k columns spends at most (k+2)*rows. Include every
    width/threshold/placement and the optional six-column varying-response
    branch, even when caches or earlier support would skip them. Subsequent
    joint candidate search, retained scoring and exports are priced separately.
    """
    spec = _spec(public_spec)
    pairs = [_pair(spec, observations, item['reference_record_index'], item['record_index'])
             for item in matched_pairs]
    total = 0
    for group in _matched_groups(spec, pairs):
        count = sum(pair['matched_rows'] for pair in group)
        rows = [row for pair in group
                for row in observations[pair['record_index']]['rows'][:pair['matched_rows']]]
        history = max(len(row.get('history', [])) for row in rows)
        levels = len({row['features'][group[0]['control']] for row in rows})
        total += matched_assessment_shape_upper(count, history, levels)
    return total


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


def propose_local_conditional_scans(public_spec, observations, result, *, continuation,
                                    source_ref, minimum_improvement=1e-10):
    """Use a measured local interval to propose a scan, without forecasting.

    A locally supported gate can remain a research question when the global
    model fails. Its existing matched-delta consumer, not the single-point
    global-model boundary updater, must assess the new reset trajectory.
    """
    if not result.get('evidence') or not public_spec.get('reset'):
        return []
    spec = _spec(public_spec)
    if spec.max_probe_length < 8:
        return []
    proposals = []
    for index, evidence in enumerate(result.get('evidence', [])):
        if (evidence.get('status') != 'conditional'
                or evidence.get('conditional_gain', 0) <= minimum_improvement
                or evidence.get('conditional_loss', float('inf')) > minimum_improvement):
            continue
        control, background = evidence['control'], evidence['background']
        left, right = evidence['threshold_interval']
        if right <= left:
            continue
        references = evidence.get('reference_record_indices',
                                  [evidence.get('reference_record_index')])
        for reference_index in references:
            if reference_index is None:
                continue
            reference = observations[reference_index]
            probe = propose_conditional_boundary_probe(spec, evidence,
                serial=result['round_index'], length=min(12, len(reference['rows'])),
                control_levels=public_spec.get('control_levels'))
            if (probe is None or len(probe.sequence) < 8
                    or not any(left < point[control] < right for point in probe.sequence[1:])):
                continue
            # Only planned controls and reset history enter this check. There
            # is no target value, prediction, fit, or laboratory execution.
            planned = {'rows': [{'features': dict(point),
                'history': [dict(old) for old in probe.sequence[:position]]}
                for position, point in enumerate(probe.sequence)]}
            try:
                pair = _pair(spec, [reference, planned], 0, 1)
            except ValueError:
                continue
            if pair['control'] != control or pair['background'] != background:
                continue
            callback = deepcopy(continuation)
            callback.pop('max_compute', None)  # Price the new evidence shape.
            first = len(observations)
            callback['matched_pairs'] = [
                {'reference_record_index': first, 'record_index': first + 1}]
            local = {'round_index': result['round_index'], 'evidence_index': index,
                'control': control, 'background': deepcopy(background),
                'threshold_interval': [left, right]}
            evidence_ref = {**source_ref, 'json_path': [*source_ref['json_path'], 'evidence', index]}
            proposals.append({'query_action': {'action': 'query', 'arguments': {
                    'actions': [deepcopy(reference['action']), {'kind': 'trajectory', 'channels': [],
                        'sequence': [dict(point) for point in probe.sequence]}],
                    'question': deepcopy(callback['research_question'])}},
                'matched_action_pairs': [{'reference_action_index': 0, 'action_index': 1,
                    'control': control, 'background': deepcopy(background), 'levels': pair['levels']}],
                'purpose': 'local_conditional_interval', 'local_conditional_interval': local,
                'local_evidence_ref': evidence_ref, 'research_continuation': callback,
                'status': 'proposed_not_purchased',
                'scope': 'Test the retained local gate hypothesis; global validity remains unresolved.'})
            break
    return proposals


def _structural_hypotheses(result, evidence):
    """Read actual fitted alternatives, never expand a representative into fits."""
    by_structure = {}
    for index, candidate in enumerate(result.get('candidates', [])):
        steps = candidate.get('research_program', {}).get('selection_steps', [])
        step = next((item for item in reversed(steps)
            if item.get('operation') == 'discover_conditional_coordinates'
                or item.get('execution_scope') == 'retained_coordinates'
                    and item.get('retained_structure')), None)
        if step is not None and step.get('retained_structure'):
            step = {**step['retained_structure'], 'program': step['program']}
        if (step is None or step.get('program') != candidate.get('program')
                or step.get('gate_coordinate', {}).get('op') != 'feature'
                or step['gate_coordinate'].get('name') != evidence['control']
                or step.get('threshold_interval') != evidence.get('threshold_interval')
                or step.get('width') not in evidence.get('width_candidates', [])
                or step.get('placement') not in evidence.get('placement_candidates', [])):
            continue
        key = step['width'], step['placement']
        item = {'candidate_index': index, 'width': key[0], 'placement': key[1],
            'program': _program(candidate['program']), 'fit_mse': candidate['fit_mse']}
        if key not in by_structure or item['fit_mse'] < by_structure[key]['fit_mse']:
            by_structure[key] = item
    return list(by_structure.values())


def native_structure_planning_contract(public_spec, observations, previous):
    """Quote an unexecuted pulse/step design over existing fitted explanations."""
    if (not public_spec.get('reset') or not observations
            or 'trajectory' not in public_spec.get('action_kinds', [])):
        return None
    if previous.get('observed_record_count') != len(observations):
        # New records belong to the ordinary evidence-update decision. They
        # do not turn an unchanged saved-bank forecast into new design work.
        return None
    spec = _spec(public_spec)
    for index, evidence in enumerate(previous.get('evidence', [])):
        if (evidence.get('status') != 'conditional'
                or evidence.get('conditional_gain', 0.) <= 1e-10
                or len(evidence.get('width_candidates', [])) < 2
                    and len(evidence.get('placement_candidates', [])) < 2):
            continue
        hypotheses = _structural_hypotheses(previous, evidence)
        if (evidence['control'] not in spec.feature_names
                or set(evidence['background']) != set(spec.feature_names)
                or any(not h['program'].required_features() <= set(spec.feature_names)
                       for h in hypotheses)):
            continue
        structures = {(h['width'], h['placement']) for h in hypotheses}
        if len(structures) < 2:
            continue
        references = evidence.get('reference_record_indices',
            [evidence.get('reference_record_index')])
        for reference in references:
            if reference is None or not 0 <= reference < len(observations):
                continue
            record = observations[reference]
            length = min(12, spec.max_probe_length, len(record.get('rows', [])))
            if (length < 8 or record['rows'][0].get('history')
                    or any(row['features'] != evidence['background'] for row in record['rows'])):
                continue
            # Two modes, two full reset forecasts per fitted hypothesis, plus
            # the existing public-grid projection charge. No fitting is priced.
            compute = 2 * length * (2 * len(hypotheses) +
                (len(spec.feature_names) if public_spec.get('control_levels') else 0))
            return {'source_round_index': previous['round_index'], 'evidence_index': index,
                'reference_record_index': reference, 'length': length,
                'candidate_indices': [item['candidate_index'] for item in hypotheses],
                'structures': [{'width': h['width'], 'placement': h['placement']}
                               for h in hypotheses],
                'observed_record_count': len(observations),
                'fit_record_indices': list(previous.get('fit_record_indices', [])),
                'proposal_compute': compute, 'max_new_observations': length,
                'fits': 0, 'scope': 'One bounded design over saved fitted alternatives; no purchase or refit.'}
    return None


def plan_native_structure_experiments(public_spec, observations, budget, *, previous,
                                     requested, source_ref):
    """Plan from a completed native bank without repeating its paid fitting."""
    contract = native_structure_planning_contract(public_spec, observations, previous)
    result = {'status': 'structure_planning_unresolved', 'proposed_experiments': [],
        'planning_complete': False, 'compute_cost': 0, 'fits': 0,
        'source_ref': deepcopy(source_ref), 'source_round_index': previous['round_index'],
        'scope': 'Planning only; old fits, paid scores and the retained parent are unchanged.'}
    if contract is None:
        result['reason'] = ('source_fit_scope_changed_requires_native_evidence_update'
            if previous.get('observed_record_count') != len(observations) else
            'saved_structural_alternatives_and_matched_reset_reference_required')
        return result
    if requested != contract:
        raise ValueError('the current saved structure-planning contract changed')
    result['basis'] = deepcopy(contract)
    if budget.remaining < contract['proposal_compute']:
        result.update(reason='complete_structure_planning_unfunded',
            required_compute=contract['proposal_compute'])
        return result
    evidence = previous['evidence'][contract['evidence_index']]
    hypotheses = _structural_hypotheses(previous, evidence)
    before = budget.spent
    planned = propose_conditional_structure_experiment(_spec(public_spec), evidence,
        hypotheses, budget, serial=previous['round_index'], length=contract['length'],
        control_levels=public_spec.get('control_levels'))
    result.update(planning_complete=True, compute_cost=budget.spent - before)
    if planned is None:
        result.update(status='no_discriminating_structure_experiment',
            reason='saved_structures_have_no_distinct_forecast_in_the_bounded_pulse_step_design')
        return result
    references = [{**deepcopy(source_ref), 'json_path': [*source_ref['json_path'],
        'candidates', h['candidate_index']]} for h in hypotheses]
    target = {'source_round_index': previous['round_index'],
        'evidence_index': contract['evidence_index'], 'control': evidence['control'],
        'background': deepcopy(evidence['background']),
        'threshold_interval': list(evidence['threshold_interval']),
        'hypothesis_refs': references, 'structures': deepcopy(contract['structures']),
        'forecast_witnesses': deepcopy(planned['witnesses']), 'mode': planned['mode']}
    reference = observations[contract['reference_record_index']]
    first = len(observations)
    result.update(status='structure_experiments_planned', proposed_experiments=[{
        'query_action': {'action': 'query', 'arguments': {'actions': [
            deepcopy(reference['action']), {'kind': 'trajectory', 'channels': [],
                'sequence': [dict(point) for point in planned['intervention'].sequence]}]}},
        'purpose': 'conditional_history_structure', 'structural_discrimination': target,
        'predicted_separation': planned['predicted_separation'],
        'matched_action_pairs': [{'reference_action_index': 0, 'action_index': 1,
            'control': evidence['control'], 'background': deepcopy(evidence['background']),
            'levels': sorted({p[evidence['control']] for p in planned['intervention'].sequence})}],
        'research_continuation': {'operator': 'native_condition_research',
            'matched_pairs': [{'reference_record_index': first, 'record_index': first + 1}]},
        'status': 'proposed_not_purchased', 'scope': planned['scope']}])
    return result


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


def retained_structure_refit_quote(candidates, rows):
    """Price a saved-coordinate fit, without executing or optimizing a model."""
    prices = []
    for item in candidates:
        payload = item['research_program']
        terms = tuple(dict.fromkeys(_program(payload['graph']['nodes'][key]['answer'])
                                    for key in payload['term_nodes']))
        prices.append({'source_ref': deepcopy(item['source_ref']),
            'retained_terms': [term.to_dict() for term in terms],
            'compute_upper': rows * (len(terms) + 2)})
    return {'compute_upper': sum(item['compute_upper'] for item in prices),
        'rows': rows, 'structures': prices,
        'execution_scope': 'retained_coordinates',
        'scope': 'One complete saved-coordinate fit per selected frozen witness; '
                 'no gate, threshold, width or placement search.'}


def _refit_saved_structures(candidates, rows, budget, *, fit_evidence,
                            observed_record_count, risk_quantile, cache):
    quote = retained_structure_refit_quote(candidates, len(rows))
    selected = []
    for source, price in zip(candidates, quote['structures']):
        scope = {key: deepcopy(price[key]) for key in ('source_ref', 'retained_terms')}
        scope.update(fit_evidence=list(fit_evidence),
            observed_record_count=observed_record_count, risk_quantile=risk_quantile)
        previous = next((item for item in cache if item['input_scope'] == scope), None)
        selected.append((source, price, scope, previous))
    required = sum(price['compute_upper'] for _, price, _, previous in selected if previous is None)
    summary = {'complete': False, 'execution_scope': 'retained_coordinates',
        'quote': quote, 'new_compute_upper': required, 'attempts': []}
    # A partial set cannot answer the planned comparison. Retain paid fits from
    # an interrupted call, but do not start a smaller arbitrary prefix now.
    if not budget.can_spend(required):
        summary.update(status='complete_saved_structure_refit_unfunded',
            blocked_work=deepcopy(budget.last_denied_request))
        return [], summary
    outputs = []
    for source, _, scope, previous in selected:
        cached = previous is not None
        if previous is None:
            before = budget.spent
            claim = replay_research_program(source['research_program'], rows, budget,
                risk_quantile=risk_quantile, execution_scope='retained_coordinates')
            candidate = (None if claim is None else {
                'program': claim['candidate'].program.to_dict(),
                'source': 'retained_structure_refit',
                'replay_source_ref': deepcopy(source['source_ref']),
                'fit_mse': claim['training_loss'], 'fit_evidence': list(fit_evidence),
                'research_program': claim['research_program']})
            previous = {'input_scope': scope, 'candidate': candidate,
                'compute_cost': budget.spent - before,
                'status': 'fitted' if candidate is not None else 'no_refittable_response'}
            cache.append(previous)
        summary['attempts'].append({'source_ref': deepcopy(source['source_ref']),
            'status': previous['status'], 'cache_hit': cached,
            'compute_cost': 0 if cached else previous['compute_cost'],
            'prior_paid_compute': previous['compute_cost'] if cached else 0})
        if previous['candidate'] is not None:
            outputs.append(deepcopy(previous['candidate']))
    summary.update(complete=True, status='retained_structures_refitted')
    return outputs, summary


def research_native_conditions(public_spec, observations, budget, *, record_indices=None,
                               matched_pairs=None, basis=None, programs=(),
                               retained_candidates=(), max_candidates=8, observation_budget=0,
                               accumulate_matched_scope=None, refit_candidates=(), refit_cache=None):
    """Research author-selected paid evidence through the old scientific solver.

    ``record_indices`` selects complete trajectories for fitting. A matched pair
    names ``reference_record_index`` and ``record_index`` in the complete paid
    archive. Compatible pairs pool their measured levels for each control and
    fixed background. When omitted, all available pairs are used; an explicit
    list restricts the evidence to those pairs. Reset histories remain separate.
    A structural callback can explicitly supply ``accumulate_matched_scope`` to
    add related older pairs from its source meaning's declared record scope.
    This does not change the fitting scope or the default explicit-pair contract.
    ``basis`` and ``programs`` are Program objects or their public JSON form.
    Retained programs are scored on the same selected paid evidence before new
    candidates are proposed. ``refit_candidates`` names the actual frozen
    witness recipes whose coordinates must be refitted, rather than rescored.
    The caller accounts ``budget.spent`` and exports
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

    if accumulate_matched_scope is not None:
        if matched_pairs is None:
            raise ValueError('a structural accumulation requires its actual new matched pair')
        chosen = accumulate_matched_pairs(public_spec, observations, matched_pairs,
            accumulate_matched_scope)
        result['accumulated_matched_scope'] = {
            **deepcopy(accumulate_matched_scope),
            'matched_pairs': [{key: pair[key] for key in
                ('reference_record_index', 'record_index', 'matched_rows')} for pair in chosen],
            'matched_rows': sum(pair['matched_rows'] for pair in chosen),
            'scope': 'Related paid reset pairs from one question meaning, control and background; '
                     'other backgrounds retain their separate historical evidence.'}

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

    refitted = []
    if refit_candidates:
        refitted, result['retained_structure_refit'] = stage('retained_structure_refit',
            lambda: _refit_saved_structures(refit_candidates, rows, budget,
                fit_evidence=result['fit_evidence'], observed_record_count=len(observations),
                risk_quantile=spec.risk_quantile,
                cache=[] if refit_cache is None else refit_cache))
    refit_complete = result.get('retained_structure_refit', {}).get('complete', True)
    for pairs in _matched_groups(spec, chosen) if refit_complete else ():
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
    if evidence and refit_complete:
        for name, proposer in (
                ('joint_window_response', propose_measured_window_response),
                ('joint_conditional_response', propose_experimental_conditional_coordinates),
                ('joint_composed_response', propose_composed_conditions)):
            claims.extend(stage(name, lambda: proposer(spec, rows, budget, basis=terms,
                evidence=evidence, max_candidates=max_candidates)))
    result['searched_candidates'] = len(claims) + len(refitted)
    unique = {_program(candidate['program']): candidate for candidate in refitted}
    for claim in claims:
        unique.setdefault(claim['candidate'].program, {
            'program': claim['candidate'].program.to_dict(),
            'source': claim['candidate'].candidate_id,
            'fit_mse': claim['training_loss'],
            'fit_evidence': list(result['fit_evidence']),
            'research_program': deepcopy(claim['research_program']),
        })
    result['candidates'].extend(sorted(unique.values(), key=lambda candidate:
        (candidate['fit_mse'], _program(candidate['program']).size))[:max_candidates])
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

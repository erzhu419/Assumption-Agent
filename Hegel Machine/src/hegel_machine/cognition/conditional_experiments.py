"""Matched physical controls for cold-start condition and duration discovery."""

from copy import deepcopy

from .language import Program
from .models import Observation, Probe
from .program_disagreement_probes import _forecast, _legal_point, _legal_sequence, _public_control_levels
from .research_programs import _Fitter, _rank


def _sequence(background, control, interval, length, mode):
    low, high = interval
    points = [dict(background) for _ in range(length)]
    if mode == "sweep":
        levels = [low + (high - low) * fraction for fraction in (0., .25, .5, .75, 1.)]
        prefix = 2
        block = max(1, (length - prefix) // len(levels))
        for index in range(prefix, length):
            points[index][control] = levels[min(len(levels) - 1, (index - prefix) // block)]
    else:
        levels, prefix = [background[control], low, high], 1
        if mode == "pulse":
            # The same switch before and after buildup distinguishes an outer
            # condition's unknown amplitude from its unknown window duration.
            for start in (prefix, length - 3):
                points[start][control] = low
                points[start + 1][control] = high
        else:
            for index in range(prefix, length):
                points[index][control] = low if index < length // 2 else high
    return tuple(points), levels, prefix


def _separation(programs, reference, intervention, budget):
    predictions = []
    for program in programs:
        original = _forecast(program, reference, budget.spend)
        changed = _forecast(program, intervention, budget.spend)
        if original is None or changed is None:
            break
        predictions.append([None if left is None or right is None else right - left
                            for left, right in zip(original, changed)])
    common = [row for row in zip(*predictions) if all(value is not None for value in row)]
    return sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.


def propose_conditional_experiments(spec, paid_rows, programs, budget, *, serial=0,
                                    observation_budget=96, control_levels=None):
    """Cover every public control in two retained, nonzero backgrounds.

    One reference is shared by the interventions in each background. The first
    context scans levels; the second tests transient switches, choosing pulse
    or step by the current explanations' forecast disagreement. Acquisition is
    performed by the caller, which pays the entire quoted pack before fitting.
    """
    bounds = spec.feature_bounds
    legal_levels = _public_control_levels(bounds, control_levels)
    controls = tuple(name for name in spec.feature_names if name in bounds)
    length = min(12, spec.max_probe_length)
    pack_cost = (len(controls) + 1) * length
    if not controls or length < 8 or observation_budget < pack_cost:
        return []
    programs = [program for program in programs
                if {name for name, _ in program.observed_variables()} <= set(controls)][:4]
    # Off-center and unequal coordinates retain both linear and quadratic
    # branch differences when a zero or sign-symmetric background hides them.
    backgrounds = [{name: low + (high - low) * fraction
        for index, name in enumerate(controls)
        for low, high in [bounds[name]]
        for fraction in [( .31 + .05 * (index % 3)) if context == 0
                         else (.72 - .04 * (index % 3))]}
        for context in range(min(2, observation_budget // pack_cost))]
    if legal_levels is not None:
        backgrounds = [_legal_point(background, legal_levels) for background in backgrounds]
    packs = []
    for context, background in enumerate(backgrounds):
        before = budget.spent
        reference_sequence = tuple(dict(background) for _ in range(length))
        interventions = []
        for control in controls:
            modes = ("sweep",) if context == 0 else ("pulse", "step")
            trials = []
            for mode in modes:
                if not budget.spend(length):
                    break
                sequence, levels, prefix = _sequence(background, control, bounds[control], length, mode)
                sequence = _legal_sequence(sequence, legal_levels, budget.spend)
                if sequence is None:
                    break
                if legal_levels is not None:
                    levels = list(dict.fromkeys(point[control] for point in sequence[prefix:]))
                score = _separation(programs, reference_sequence, sequence, budget)
                trials.append((score, mode, sequence, levels, prefix))
            if not trials:
                break
            score, mode, sequence, levels, prefix = max(trials, key=lambda item: item[0])
            interventions.append({"probe": Probe(f"conditional:{serial}:{context}:{control}:{mode}", sequence),
                "control": control, "mode": mode, "levels": levels, "prefix_length": prefix,
                "background": dict(background), "predicted_separation": score})
        if len(interventions) != len(controls):
            break
        packs.append({"reference": Probe(f"conditional:{serial}:{context}:reference", reference_sequence),
            "interventions": interventions, "background": background,
            "observation_cost": pack_cost, "measurement_cost": 0,
            "compute_cost": budget.spent - before,
            "predicted_separation": sum(plan["predicted_separation"] for plan in interventions)})
    return packs


def propose_conditional_structure_experiment(spec, evidence, hypotheses, budget, *,
                                             serial=0, length=12, control_levels=None):
    """Discriminate actually fitted widths/placements with a matched switch.

    The existing pulse/step generator and forecast meter are used unchanged.
    Each witness compares two saved fitted explanations differing on one
    structural coordinate; coefficients are never supplied or refitted here.
    """
    if len(hypotheses) < 2:
        return None
    control, background = evidence['control'], evidence['background']
    legal = _public_control_levels(spec.feature_bounds, control_levels)
    if legal is not None:
        background = _legal_point(background, legal)
    length = min(length, spec.max_probe_length)
    if length < 8:
        return None
    reference = tuple(dict(background) for _ in range(length))
    trials, start = [], budget.spent
    for mode in ('pulse', 'step'):
        sequence, levels, prefix = _sequence(background, control,
            evidence['threshold_interval'], length, mode)
        sequence = _legal_sequence(sequence, legal, budget.spend)
        if sequence is None:
            break
        predictions = []
        for hypothesis in hypotheses:
            program = hypothesis['program']
            before = _forecast(program, reference, budget.spend)
            after = _forecast(program, sequence, budget.spend)
            if before is None or after is None:
                return None  # A partial forecast does not establish separation.
            predictions.append((before, after))
        witnesses = {}
        for i, left in enumerate(hypotheses):
            for j, right in enumerate(hypotheses[i + 1:], i + 1):
                dimensions = []
                if left['placement'] == right['placement'] and left['width'] != right['width']:
                    dimensions.append('history_width')
                if left['width'] == right['width'] and left['placement'] != right['placement']:
                    dimensions.append('condition_position')
                if not dimensions:
                    continue
                for step, values in enumerate(zip(*predictions[i], *predictions[j])):
                    if any(value is None for value in values):
                        continue
                    a0, a1, b0, b1 = values
                    difference = ((a1 - a0) - (b1 - b0)) ** 2
                    for dimension in dimensions:
                        if difference > 1e-10 and difference > witnesses.get(
                                dimension, {}).get('squared_delta_difference', 0.):
                            witnesses[dimension] = {'hypothesis_indices': [i, j],
                                'readout_step': step, 'delta_predictions': [a1 - a0, b1 - b0],
                                'squared_delta_difference': difference}
        if witnesses:
            trials.append((len(witnesses), sum(w['squared_delta_difference']
                for w in witnesses.values()), mode, sequence, witnesses))
    if not trials:
        return None
    _, separation, mode, sequence, witnesses = max(trials, key=lambda item: item[:2])
    return {'reference': Probe(f'conditional_structure:{serial}:reference', reference),
        'intervention': Probe(f'conditional_structure:{serial}:{control}:{mode}', sequence),
        'mode': mode, 'witnesses': witnesses, 'predicted_separation': separation,
        'compute_cost': budget.spent - start,
        'scope': 'Predictions of retained fitted alternatives; not a guarantee of unique structural identification.'}


def assess_frozen_structure_predictions(agent, target, matched_pairs, *, planned_actions,
                                        max_compute=None, atol=1e-10):
    """Check the purchased experiment's saved predictions before any new fit.

    This is the existing paid linear-relation checker applied to intervention
    minus reference readings. It does not execute a predictor or exclude an
    entire structure family when a frozen fitted instance fails.
    """
    from .inquiry_revision import _check_local_relations

    witnesses = target.get('forecast_witnesses', {})
    required = 4 * len(witnesses)  # Two two-term observed deltas per witness.
    result = {'status': 'inconclusive', 'complete': False, 'compute_cost': 0,
        'required_compute': required, 'witnesses': [], 'evidence': [], 'refuted_candidate_refs': [],
        'compatible_candidate_refs': [], 'structure_families_refuted': False,
        'scope': 'Frozen fitted predictors at declared readouts, not whole structure families.',
        'next_research': {'kind': 'complete_frozen_readout_check'}}
    if not witnesses:
        result['reason'] = 'no_frozen_prediction_witnesses'
        return result
    if len(matched_pairs) != 1 or len(planned_actions) != 2:
        result['reason'] = 'one_planned_reference_intervention_pair_required'
        return result
    pair = matched_pairs[0]
    indices = [pair['reference_record_index'], pair['record_index']]
    if any(type(i) is not int or not 0 <= i < len(agent.observations) for i in indices):
        result['reason'] = 'mapped_paid_record_missing'
        return result
    records = [agent.observations[i] for i in indices]
    for record, action in zip(records, planned_actions):
        sequence = action.get('sequence', [])
        rows = record.get('rows', [])
        if (not sequence or record.get('action', {}).get('sequence') != sequence
                or len(rows) != len(sequence) or len(record.get('row_indices', [])) != len(rows)
                or any(row['features'] != point or list(row.get('history', [])) != sequence[:step]
                       for step, (row, point) in enumerate(zip(rows, sequence)))):
            result['reason'] = 'paid_controls_or_reset_history_differ_from_frozen_experiment'
            return result
    checks, selections, evidence = [], [], []
    for dimension, witness in witnesses.items():
        step = witness['readout_step']
        if (type(step) is not int or step < 0
                or any(step >= len(record['rows']) for record in records)):
            result['reason'] = 'frozen_readout_not_measured'
            return result
        paid_rows = [record['row_indices'][step] for record in records]
        if any(type(i) is not int or not 0 <= i < len(agent.rows) for i in paid_rows):
            result['reason'] = 'frozen_readout_row_binding_missing'
            return result
        for record, row_index in zip(records, paid_rows):
            actual, saved = agent.rows[row_index], record['rows'][step]
            if (actual.features != saved['features'] or actual.target != saved['target']
                    or list(actual.history) != list(saved.get('history', []))):
                result['reason'] = 'frozen_readout_row_binding_changed'
                return result
        evidence.extend(paid_rows)
        refs = [target['hypothesis_refs'][i] for i in witness['hypothesis_indices']]
        selections.append((dimension, witness, paid_rows, refs))
        for position, (reference, prediction) in enumerate(zip(refs, witness['delta_predictions'])):
            checks.append({'id': f'{dimension}:frozen:{position}',
                'scope': {'candidate_ref': deepcopy(reference), 'readout_step': step},
                'terms': [{'row': paid_rows[1], 'coefficient': 1.},
                          {'row': paid_rows[0], 'coefficient': -1.}],
                'expected': prediction, 'relation': 'eq', 'atol': atol})
    result['evidence'] = list(dict.fromkeys(evidence))
    available = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
    if max_compute is not None:
        available = min(available, max_compute)
    if available < required:
        result.update(reason='complete_frozen_readout_check_unfunded', available_compute=available)
        return result
    before = agent.costs['compute_cost']
    checked, _ = _check_local_relations(agent,
        {'checks': checks, 'evidence': result['evidence']}, 'response')
    result['compute_cost'] = agent.costs['compute_cost'] - before
    candidate_results = []
    for offset, (dimension, witness, paid_rows, refs) in enumerate(selections):
        readings = checked[2 * offset:2 * offset + 2]
        statuses = [reading['status'] for reading in readings]
        complete = all(status in {'satisfied', 'violated'} for status in statuses)
        status = ('not_identified' if not complete else
                  'both_predictions_refuted' if statuses == ['violated', 'violated'] else
                  'not_distinguished' if statuses == ['satisfied', 'satisfied'] else
                  'predictions_distinguished')
        result['witnesses'].append({'dimension': dimension, 'status': status,
            'readout_step': witness['readout_step'], 'record_indices': list(indices),
            'evidence': paid_rows, 'candidate_refs': deepcopy(refs),
            'delta_predictions': list(witness['delta_predictions']),
            'observed_delta': readings[0].get('value'), 'checks': readings})
        for reference, reading in zip(refs, readings):
            candidate = next((item for item in candidate_results if item['candidate_ref'] == reference), None)
            if candidate is None:
                candidate = {'candidate_ref': deepcopy(reference), 'statuses': []}
                candidate_results.append(candidate)
            candidate['statuses'].append(reading['status'])
    for candidate in candidate_results:
        key = ('refuted_candidate_refs' if 'violated' in candidate['statuses'] else
               'compatible_candidate_refs' if all(s == 'satisfied' for s in candidate['statuses']) else None)
        if key is not None:
            result[key].append(candidate['candidate_ref'])
    categories = {item['status'] for item in result['witnesses']}
    result['complete'] = 'not_identified' not in categories
    if result['complete']:
        result['status'] = (next(iter(categories)) if len(categories) == 1 else 'partially_distinguished')
        result['next_research'] = {'kind': (
            'jointly_refit_frozen_structures' if 'both_predictions_refuted' in categories else
            'compare_surviving_fixed_candidates' if 'predictions_distinguished' in categories else
            'preserve_ambiguity')}
    return result


def assess_conditional_experiment(spec, plan, reference_rows, measured_rows, budget,
                                  *, minimum_improvement=1e-10):
    """Compare smooth and switched explanations of an actually measured delta.

    Frozen backgrounds reduce the response to one controlled coordinate. A
    smooth linear/quadratic temporal response competes with a switched response
    before a control can become a proposed gate. Equivalent widths and nesting
    orders remain explicit when this experiment has not distinguished them.
    A relative winner that still misses the measured response only nominates
    further research; the other actually fitted structures remain available.
    """
    reference_rows, measured_rows = list(reference_rows), list(measured_rows)
    count, start = len(measured_rows), budget.spent
    control, background = plan["control"], plan["background"]
    result = {"status": "budget_exhausted", "control": control,
        "background": dict(background), "mode": plan["mode"],
        "gain": 0., "conditional_gain": 0., "compute_cost": 0,
        "matched_rows": count}
    if not budget.spend(2 * count):
        return result
    rows = [Observation(dict(changed.features), changed.target - original.target,
                        tuple(dict(point) for point in changed.history))
            for original, changed in zip(reference_rows, measured_rows)]
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    reference_fitter = _Fitter(reference_rows, budget, spec.risk_quantile)
    feature = Program("feature", name=control)
    constant = Program("const", value=float(background[control]))
    difference = Program("sub", children=(feature, constant))
    square_difference = Program("sub", children=(
        Program("mul", children=(feature, feature)),
        Program("const", value=background[control] ** 2)))
    upper = max(len(row.history) for row in rows)
    reference_widths = []
    if budget.spend(2 * count):
        mean = sum(row.target for row in reference_rows) / count
        reference_widths.append((sum((row.target - mean) ** 2 for row in reference_rows) / count, 1))
    for width in range(2, max(1, upper) + 1):
        fit = reference_fitter.fit((Program("window_sum", lag=width,
            children=(Program("const", value=1.),)),))
        if fit is not None:
            reference_widths.append((fit["loss"], width))
    reference_supported = []
    if reference_widths:
        reference_loss = min(item[0] for item in reference_widths)
        result["reference_loss"] = reference_loss
        if reference_loss <= minimum_improvement:
            reference_supported = [width for measured_loss, width in reference_widths
                                   if measured_loss <= reference_loss + minimum_improvement]
            result["reference_width_candidates"] = reference_supported
    smooth = []
    for width in range(1, max(1, upper) + 1):
        terms = [Program("window_sum", lag=width, children=(term,)) if width > 1 else term
                 for term in (difference, square_difference)]
        fit = fitter.fit(terms)
        if fit is not None:
            smooth.append((fit, width))
    if not smooth:
        result["compute_cost"] = budget.spent - start
        return result
    smooth_fit, smooth_width = min(smooth, key=lambda item: _rank(item[0]))
    result.update(smooth_loss=smooth_fit["loss"], smooth_width=smooth_width)
    if smooth_fit["loss"] <= minimum_improvement:
        result.update(status="smooth", compute_cost=budget.spent - start)
        return result

    levels = sorted(set(row.features[control] for row in rows))
    trials = []
    # The paid reference startup already excluded other durations. Search the
    # same supported candidates directly, preserving funds for a local scan
    # of the unresolved gate interval and its joint response refit.
    trial_widths = reference_supported or range(1, max(1, upper) + 1)
    for width in trial_widths:
        window_count = Program("window_sum", lag=width, children=(Program("const", value=1.),))
        for left, right in zip(levels, levels[1:]):
            threshold = (left + right) / 2.
            condition = Program("ge", children=(feature, Program("const", value=threshold)))
            changed_condition = Program("sub", children=(condition,
                Program("const", value=float(background[control] >= threshold))))
            for placement in ("inside_window", "outside_window"):
                term = (Program("window_sum", lag=width, children=(changed_condition,))
                        if placement == "inside_window" else
                        Program("mul", children=(changed_condition, window_count)))
                fit = fitter.fit((term,))
                if fit is not None:
                    trials.append((fit, width, placement, [left, right]))
    if not trials:
        result["compute_cost"] = budget.spent - start
        return result
    usable = [trial for trial in trials if not reference_supported or trial[1] in reference_supported]
    if not usable:
        usable = trials
    best, width, placement, interval = min(usable, key=lambda item: _rank(item[0]))
    response_model = "constant_branches"
    if best["loss"] > minimum_improvement:
        # A coordinate can vary smoothly within either branch while also
        # deciding which branch is active. A constant jump alone does not
        # exclude such a measured condition.
        from .composed_condition_experiments import varying_condition_delta_terms
        varying = []
        for proposed_width in trial_widths:
            for left, right in zip(levels, levels[1:]):
                threshold = (left + right) / 2.
                for proposed_placement in ("inside_window", "outside_window"):
                    terms = varying_condition_delta_terms(control, background[control],
                        threshold, proposed_width, proposed_placement)
                    fit = fitter.fit(terms)
                    if fit is not None:
                        varying.append((fit, proposed_width, proposed_placement, [left, right]))
        if varying:
            revised = min(varying, key=lambda item: _rank(item[0]))
            if revised[0]["loss"] + minimum_improvement < best["loss"]:
                best, width, placement, interval = revised
                response_model = "varying_response"
                usable = [*usable, *varying]
    gain = smooth_fit["loss"] - best["loss"]
    equivalent = [trial for trial in usable
                  if trial[0]["loss"] <= best["loss"] + minimum_improvement]
    if response_model == "varying_response":
        # Five purchased levels can support several piecewise quadratics.
        # Keep every observationally equivalent cut for the denser experiment
        # instead of turning one polynomial interpolation into a physical cut.
        interval = [min(trial[3][0] for trial in equivalent),
                    max(trial[3][1] for trial in equivalent)]
    identified = gain > minimum_improvement and best["loss"] <= minimum_improvement
    # Relative improvement nominates a useful direction, but a misfitting local
    # winner cannot exclude the other structures already fitted on these rows.
    retained = equivalent if identified else usable
    competitors = {}
    for fit, proposed_width, proposed_placement, proposed_interval in retained:
        key = proposed_width, proposed_placement
        if key not in competitors or _rank(fit) < _rank(competitors[key][0]):
            competitors[key] = fit, proposed_interval
    competing = [{"width": proposed_width, "placement": proposed_placement,
        "threshold_interval": list(proposed_interval), "conditional_loss": fit["loss"],
        "gain": smooth_fit["loss"] - fit["loss"]}
        for (proposed_width, proposed_placement), (fit, proposed_interval)
        in sorted(competitors.items(), key=lambda item: _rank(item[1][0]))]
    result.update(status="conditional" if gain > minimum_improvement else "smooth",
        conditional_loss=best["loss"], gain=gain, conditional_gain=gain,
        response_supported=best["loss"] <= minimum_improvement,
        condition_identified=identified,
        threshold_interval=interval, width=width, placement=placement,
        width_candidates=sorted({trial[1] for trial in retained}),
        placement_candidates=sorted({trial[2] for trial in retained}),
        threshold_intervals=sorted({tuple(trial[3]) for trial in retained}),
        competing_structures=competing,
        response_model=response_model,
        compute_cost=budget.spent - start)
    return result


def propose_conditional_boundary_probe(spec, evidence, *, serial=0, length=24, control_levels=None):
    """Propose a local gate scan inside the currently measured interval."""
    control, background = evidence["control"], evidence["background"]
    left, right = evidence["threshold_interval"]
    low, high = spec.feature_bounds[control]
    legal_levels = _public_control_levels(spec.feature_bounds, control_levels)
    left, right = max(left, low), min(right, high)
    length = min(length, spec.max_probe_length)
    if length < 3 or right <= left:
        return None
    if legal_levels is not None:
        scan = [value for value in legal_levels[control] if left <= value <= right]
        if not scan:
            return None
        background = _legal_point(background, legal_levels)
        # Preserve the requested reset history on a discrete grid too. Fewer
        # distinct levels do not shorten the matched consumer's input contract.
        points = [dict(background) for _ in range(length)]
        for index, point in enumerate(points[1:]):
            point[control] = scan[index * max(0, len(scan) - 1) // max(1, len(points) - 2)]
        return Probe(f"conditional_boundary:{serial}:{control}", tuple(points))
    points = [dict(background) for _ in range(length)]
    for index in range(1, length):
        points[index][control] = left + (right - left) * (index - 1) / (length - 2)
    return Probe(f"conditional_boundary:{serial}:{control}", tuple(points))

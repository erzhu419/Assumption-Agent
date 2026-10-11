"""Finite Boolean condition research on paid activation outcomes.

The old condition language searches the actual yes/no target. Residuals from
retained numerical proposals are computed coordinates, not exact labels or
accepted physical laws. Every proposed rule remains an unconfirmed candidate.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from math import ceil, floor

from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget
from .probing import _affine_controls, choose_probe
from .program_disagreement_probes import predicate_boundary_groups
from .synthesis import synthesize


def _features(measurements, bound_features, frequency, derivations):
    measured = {key: value['value'] if isinstance(value, dict) else value
                for key, value in measurements.items()}
    values = {**measured, **{'bound_' + key: value for key, value in bound_features.items()},
              'frequency': frequency}
    for name, source in derivations.items():
        values[name] = frequency - Program.from_dict(source).predict(Observation(bound_features, 0.))
    return values


def predict_response_condition(candidate, measurements, bound_features, frequency):
    """Evaluate a frozen Boolean rule with its original computed coordinates."""
    values = _features(measurements, bound_features, frequency, candidate['feature_derivations'])
    return bool(Program.from_dict(candidate['program']).predict(Observation(values, 0.)))


def _candidate_bound_features(candidate, inputs, number):
    return candidate.get('bound_rows', inputs['bound_rows']).get(str(number), {})


def _inputs(packet):
    measured = {str(row['crystal']): {name: value['value'] for name, value in row['measurements'].items()}
                for row in packet.get('measurements', [])}
    bound = {str(row['crystal']): row['features'] for row in packet.get('bound_rows', [])}
    responses = [{key: deepcopy(row[key]) for key in ('crystal', 'frequency', 'activated', 'evidence')}
                 for row in packet.get('responses', [])]
    return {'measurements': measured, 'bound_rows': bound, 'responses': responses,
            'control_grids': deepcopy(packet.get('control_grids', {})),
            'targets': list(packet.get('targets', [])),
            'residual_sources': _residuals(packet),
            'candidate_additions': deepcopy(packet.get('candidate_additions', [])),
            'controller_path': packet['controller_path']}


def _boolean(program):
    return program.op in {'ge', 'lt', 'and', 'or', 'not'}


def _normalize_condition(program):
    """Remove the redundant Boolean threshold emitted by the old synthesizer."""
    program = replace(program, children=tuple(_normalize_condition(child) for child in program.children))
    if program.op == 'ge':
        left, right = program.children
        if _boolean(left) and right.op == 'const' and right.value == .5:
            return left
    return program


def _residuals(packet):
    sources = [item['program'] for item in packet.get('response_candidates', [])
               if item.get('status') == 'compatible' and 'program' in item]
    sources.extend(item['witness']['program'] for item in packet.get('paid_family_checks', {}).values()
                   if item.get('status') == 'SAT_WITNESS' and item.get('witness'))
    programs = dict.fromkeys(Program.from_dict(source).simplified() for source in sources)
    names = [set(row['features']) for row in packet.get('bound_rows', [])]
    available = set.intersection(*names) if names else set()
    return {'residual_' + str(index): program.to_dict() for index, program in enumerate(programs)
            if not program.history_depth and program.required_features() <= available}


def _transition_seeds(rows, names, budget):
    """Seed observed label boundaries that the old seven-quantile pilot misses."""
    seeds, intervals = [], []
    for name in names:
        if not budget.spend(len(rows)):
            break
        values = defaultdict(set)
        for row in rows:
            values[row.features[name]].add(row.target)
        levels = sorted(values)
        positive = [value for value in levels if 1. in values[value]]
        negative = [value for value in levels if 0. in values[value]]
        if positive and not any(min(positive) <= value <= max(positive) for value in negative):
            # Reuse the old response hypothesis endpoint/midpoint policy. Give
            # its complete executable interval to the Boolean composition beam;
            # tied individual predicates need not survive that beam together.
            low, high = min(positive), max(positive)
            before = max((value for value in negative if value < low), default=None)
            after = min((value for value in negative if value > high), default=None)
            lowers = [low] if before is None else [low, (before + low) / 2.]
            uppers = [high] if after is None else [high, (after + high) / 2.]
            coordinate = Program('feature', name=name)
            for lower in lowers:
                for upper in uppers:
                    intervals.append(Program('and', children=(
                        Program('ge', children=(coordinate, Program('const', value=lower))),
                        Program('ge', children=(Program('const', value=upper), coordinate)))))
        for left, right in zip(levels, levels[1:]):
            if values[left] | values[right] == {0., 1.}:
                for threshold in (left, (left + right) / 2., right):
                    for operation in ('ge', 'lt'):
                        seeds.append(Program(operation, children=(Program('feature', name=name),
                            Program('const', value=threshold))))
    return list(dict.fromkeys([*intervals, *seeds]))


def _screen(candidate, inputs, budget):
    responses = inputs['responses']
    if not budget.spend(len(responses) * (1 + len(candidate['feature_derivations']))):
        return None
    conflicts = []
    for row in responses:
        number = row['crystal']
        prediction = predict_response_condition(candidate, inputs['measurements'][str(number)],
            _candidate_bound_features(candidate, inputs, number), row['frequency'])
        if prediction != row['activated']:
            conflicts.append({**deepcopy(row), 'predicted_activation': prediction})
    return {**deepcopy(candidate), 'status': 'refuted' if conflicts else 'compatible',
            'checked_response_count': len(responses), 'conflict_count': len(conflicts),
            'conflicting_responses': conflicts[:8],
            'evidence': sorted({index for row in responses for index in row['evidence']})}


def _authored_candidate(supplied, inputs, identifier):
    """Use the relation researcher's explicit-addition contract for Boolean rules."""
    program = Program.from_dict(supplied.get('program', supplied)).expanded()

    def boolean(node):
        if node.op in {'ge', 'lt'}:
            return True
        return node.op in {'and', 'or', 'not'} and all(boolean(child) for child in node.children)

    def uses_unbound_or_history(node):
        return node.op in {'arg', 'available'} or any(uses_unbound_or_history(child)
                                                    for child in node.children)

    if not boolean(program) or program.history_depth or uses_unbound_or_history(program):
        raise ValueError('authored response conditions need a Boolean program without history')
    program = _normalize_condition(program.simplified())
    derivations = inputs['residual_sources']
    names = [set(measured) | {'bound_' + name for name in inputs['bound_rows'].get(number, {})}
             | {'frequency'} | set(derivations)
             for number, measured in inputs['measurements'].items()]
    available = set.intersection(*names) if names else set()
    required = program.required_features()
    if not required <= available:
        raise ValueError('authored response condition uses unavailable features: '
                         + ', '.join(sorted(required - available)))
    used = {name: source for name, source in derivations.items() if name in required}
    return {'id': identifier, 'program': program.to_dict(), 'origin': 'authored',
            **({'authored_id': supplied['id']} if 'id' in supplied else {}),
            'claim': supplied.get('claim', 'Activation iff ' + program.describe()),
            'feature_derivations': deepcopy(used), 'bound_rows': deepcopy(inputs['bound_rows']),
            'proposed_evidence': sorted({i for row in inputs['responses'] for i in row['evidence']})}


def _same_rule(left, right):
    return all(left[key] == right[key] for key in ('program', 'feature_derivations', 'bound_rows'))


def _frequency_affine(expression, at_zero, derivations):
    """Bind measured object features before reusing the old affine solver."""
    dynamic = {'frequency', *derivations}

    def bind(node):
        if node.op == 'feature' and node.name not in dynamic:
            return Program('const', value=at_zero[node.name])
        return replace(node, children=tuple(bind(child) for child in node.children))

    affine = _affine_controls(bind(expression.expanded()).simplified())
    if affine is None:
        return None
    weights, intercept = affine
    slope, offset = Fraction(0), Fraction(float(intercept))
    for name, weight in weights.items():
        coefficient = Fraction(float(weight))
        offset += coefficient * Fraction(float(at_zero[name]))
        slope += coefficient
    return slope, offset


def _experiments(result, inputs, budget):
    candidates = [item for item in result['candidates'] if item['status'] == 'compatible']
    result.update(experiment_search_complete=True, selected_experiment=None,
                  experiment_action=None, experiment_count=0, predicted_separation=0.,
                  status=('compatible_unconfirmed_conditions' if candidates
                          else 'no_compatible_rule_proposed'))

    def spend(units):
        paid = budget.spend(units)
        if not paid:
            result['experiment_search_complete'] = False
        return paid

    if len(candidates) < 2:
        return
    boundaries = [(item, predicate_boundary_groups([Program.from_dict(item['program'])], spend=spend))
                  for item in candidates]
    experiments = []
    for raw_number, grid in inputs['control_grids'].items():
        number = int(raw_number)
        if str(number) not in inputs['measurements']:
            continue
        seen = {row['frequency'] for row in inputs['responses'] if row['crystal'] == number}
        settings = set()
        for item, groups in boundaries:
            if not spend(len(item['feature_derivations'])):
                break
            at_zero = _features(inputs['measurements'][str(number)],
                _candidate_bound_features(item, inputs, number), 0., item['feature_derivations'])
            for group in groups:
                expression = group['expression']
                if _boolean(expression) or not spend(expression.size):
                    continue
                affine = _frequency_affine(expression, at_zero, item['feature_derivations'])
                if affine is None or not affine[0]:
                    continue
                slope, offset = affine
                for threshold in group['thresholds']:
                    setting = (Fraction(float(threshold)) - offset) / slope
                    position = (setting - Fraction(float(grid['origin']))) / Fraction(float(grid['step']))
                    for index in (floor(position) - 1, floor(position), ceil(position), ceil(position) + 1):
                        setting = grid['origin'] + index * grid['step']
                        if grid['lower'] <= setting <= grid['upper'] and setting not in seen:
                            settings.add(setting)
        for setting in sorted(settings):
            price = sum(1 + len(item['feature_derivations']) for item in candidates)
            # Keep the remaining allowance for the existing probe chooser.
            if budget.remaining < price + (len(experiments) + 1) * len(candidates):
                result['experiment_search_complete'] = False
                break
            budget.spend(price)
            predictions = {item['id']: predict_response_condition(item,
                inputs['measurements'][str(number)], _candidate_bound_features(item, inputs, number), setting)
                for item in candidates}
            experiments.append({'id': f'condition_{number}_{setting}', 'number': number,
                'prediction': setting, 'outcome': f'Crystal {number} activates at setting {setting}',
                'predictions': predictions})
    if not experiments:
        return
    columns = {item['id']: 'forecast_' + str(index) for index, item in enumerate(candidates)}
    selectors = [Candidate(item['id'], Program('feature', name=columns[item['id']]), 'condition_forecast')
                 for item in candidates]
    probes = [Probe(item['id'], ({columns[key]: float(value)
        for key, value in item['predictions'].items()},)) for item in experiments]
    if budget.remaining < len(probes) * len(candidates):
        result['experiment_search_complete'] = False
    selected, separation = choose_probe(probes, selectors, 0., budget)
    result['experiment_count'] = len(experiments)
    result['selected_experiment'] = next((item for item in experiments
        if selected is not None and item['id'] == selected.probe_id and separation > 0.), None)
    result['predicted_separation'] = separation
    if result['selected_experiment'] is not None:
        experiment = result['selected_experiment']
        result['experiment_action'] = {'action': 'run', 'arguments': {
            'operator': 'reference_response', 'path': inputs['controller_path'],
            'experiment': {'number': experiment['number'], 'setting': experiment['prediction']}}}
        result['status'] = 'discriminating_experiment_available'
    else:
        result['status'] = 'no_disagreement_on_proposed_grid_points'


def _merge_intervals(intervals):
    merged = []
    for left, right in intervals:
        if merged and left <= merged[-1][1] + 1:
            merged[-1][1] = max(right, merged[-1][1])
        else:
            merged.append([left, right])
    return {'intervals': merged, 'grid_count': sum(right - left + 1 for left, right in merged)}


def _target_projection(number, candidates, inputs, experiment, budget):
    """Project finite Boolean rules by predicate cells, not a grid sweep."""
    key = str(number)
    grid = inputs['control_grids'].get(key)
    answer = {'number': number, 'status': 'unresolved', 'grid': deepcopy(grid)}
    if grid is None or key not in inputs['measurements']:
        return {**answer, 'reason': 'public_target_measurements_or_grid_missing'}
    complete = True

    def spend(units):
        nonlocal complete
        paid = budget.spend(units)
        complete &= paid
        return paid

    origin, step = Fraction(float(grid['origin'])), Fraction(float(grid['step']))
    first = ceil((Fraction(float(grid['lower'])) - origin) / step)
    last = floor((Fraction(float(grid['upper'])) - origin) / step)
    cuts = {first, last + 1}
    for candidate in candidates:
        if not spend(len(candidate['feature_derivations'])):
            return {**answer, 'reason': 'projection_compute_limit'}
        at_zero = _features(inputs['measurements'][key], _candidate_bound_features(candidate, inputs, number),
                            0., candidate['feature_derivations'])
        groups = predicate_boundary_groups([Program.from_dict(candidate['program'])], spend=spend)
        if not complete:
            return {**answer, 'reason': 'projection_compute_limit'}
        for group in groups:
            expression = group['expression']
            # Boolean predicates change only at their inner numerical boundaries,
            # which the existing recursive extractor has also returned.
            if _boolean(expression):
                continue
            if not spend(expression.size):
                return {**answer, 'reason': 'projection_compute_limit'}
            affine = _frequency_affine(expression, at_zero, candidate['feature_derivations'])
            if affine is None:
                return {**answer, 'reason': 'nonaffine_control_boundary'}
            slope, offset = affine
            if not slope:
                continue
            for threshold in group['thresholds']:
                position = ((Fraction(float(threshold)) - offset) / slope - origin) / step
                # Isolate grid points on either side and exactly on a boundary;
                # the executable rule decides >= versus <, including endpoints.
                for index in (floor(position), floor(position) + 1, ceil(position), ceil(position) + 1):
                    if first < index <= last:
                        cuts.add(index)
    cuts = sorted(cuts)
    pools = ({outcome: {candidate['id'] for candidate in candidates
              if experiment['predictions'][candidate['id']] == outcome}
              for outcome in (False, True)} if experiment is not None else {})
    intervals, after = [], {outcome: [] for outcome in pools}
    for left, stop in zip(cuts, cuts[1:]):
        if not spend(sum(1 + len(candidate['feature_derivations']) for candidate in candidates)):
            return {**answer, 'reason': 'projection_compute_limit'}
        setting = grid['origin'] + grid['step'] * left
        active = {candidate['id'] for candidate in candidates if predict_response_condition(
            candidate, inputs['measurements'][key], _candidate_bound_features(candidate, inputs, number), setting)}
        if active:
            intervals.append([left, stop - 1])
        for outcome in pools:
            if active & pools[outcome]:
                after[outcome].append([left, stop - 1])
    support = _merge_intervals(intervals)
    projected = {**answer, 'status': 'complete', 'partition_cells': len(cuts) - 1,
                 'support': support}
    if experiment is None:
        return projected
    outcomes = [{'activated': outcome, 'surviving_candidates': len(pools[outcome]),
                 'support': _merge_intervals(after[outcome])} for outcome in pools]
    for outcome in outcomes:
        outcome['removed_settings'] = support['grid_count'] - outcome['support']['grid_count']
    return {**projected, 'by_hypothetical_outcome': outcomes,
            'guaranteed_search_reduction': min(outcome['removed_settings'] for outcome in outcomes)}


def _target_consequences(result, inputs, budget, previous=None):
    """Project target support even when the current rules agree on all probes."""
    experiment = result.get('selected_experiment')
    if not inputs.get('targets'):
        return
    before = budget.spent
    candidates = [item for item in result['candidates'] if item['status'] == 'compatible']
    if not candidates:
        return
    selected = ({'number': experiment['number'], 'prediction': experiment['prediction']}
                if experiment is not None else None)
    # The cached bank has the same rules and evidence. Only unfinished
    # computations need another allowance; unsupported boundaries need new
    # scientific inputs, not repeated identical calculation. A changed
    # experiment invalidates the old hypothetical-outcome projections.
    retained = ({item['number']: item for item in previous['targets']}
                if previous is not None else {})
    same_experiment = previous is not None and previous.get('experiment') == selected
    targets = [deepcopy(retained[number]) if number in retained
               and retained[number].get('reason') != 'projection_compute_limit'
               and (same_experiment or retained[number]['status'] != 'complete')
               else _target_projection(number, candidates, inputs, experiment, budget)
               for number in inputs['targets']]
    result['target_consequences'] = {
        'status': 'complete' if all(item['status'] == 'complete' for item in targets) else 'unresolved',
        'scope': 'possible_activation_under_retained_unconfirmed_finite_rules',
        'experiment': selected,
        'targets': targets, 'compute_cost': budget.spent - before,
        'interpretation': 'Grid intervals index origin + step * index. Hypothetical outcomes are not '
                          'evidence; reduced finite-rule support does not confirm a law or target activation.'}


def synthesize_response_conditions(packet, previous=None, compute_limit=100000):
    """Reuse frozen rules, or propose finite Boolean conditions on actual rows.

    No compatible rule is called supported. Counterexamples precede generation;
    old failed rules remain in the result. Candidate and grid searches are finite
    and cannot refute a broader scientific family.
    """
    inputs = _inputs(packet)
    if (previous and previous.get('inputs') == inputs
            and previous.get('compute_limit') == compute_limit
            and previous.get('status') != 'search_budget_exhausted'):
        result = {**deepcopy(previous), 'cached': True, 'compute_cost': 0}
        projection = result.get('target_consequences')
        projection_pending = projection is not None and any(
            item.get('reason') == 'projection_compute_limit' for item in projection['targets'])
        experiments_pending = (result.get('experiment_search_complete') is False
            or ('experiment_search_complete' not in result and projection_pending))
        if projection is None or projection_pending or experiments_pending:
            # Screening/generation can exhaust a call before choosing a useful
            # experiment. Continue these downstream stages on the frozen bank,
            # rather than paying again to screen or generate the same rules.
            budget = SearchBudget(compute_limit)
            if experiments_pending:
                _experiments(result, inputs, budget)
            _target_consequences(result, inputs, budget, previous=projection)
            result['compute_cost'] = budget.spent
        return result
    budget = SearchBudget(compute_limit)
    result = {'status': 'condition_evidence_needed', 'inputs': inputs, 'cached': False,
        'candidates': [], 'compute_cost': 0, 'selected_experiment': None, 'experiment_action': None,
        'experiment_count': 0, 'regenerated': False, 'scope_is_exhaustive': False,
        'search_scope': 'finite Boolean compositions of paid threshold and residual coordinates',
        'confirmation_status': 'unconfirmed', 'generation_compute_cost': 0}
    result['compute_limit'] = compute_limit
    responses = inputs['responses']
    if not responses or any(str(row['crystal']) not in inputs['measurements'] for row in responses):
        return result
    if any(type(row['activated']) is not bool for row in responses):
        raise ValueError('response condition targets must be actual Boolean outcomes')
    old = deepcopy((previous or {}).get('candidates', []))
    old_inputs = (previous or {}).get('inputs', {})
    for candidate in old:
        candidate.setdefault('bound_rows', deepcopy(old_inputs.get('bound_rows', inputs['bound_rows'])))
    for item in old:
        checked = _screen(item, inputs, budget)
        if checked is None:
            result.update(status='search_budget_exhausted', compute_cost=budget.spent)
            result['candidates'].extend({**deepcopy(pending), 'status': 'unchecked_on_current_responses'}
                for pending in old[len(result['candidates']):])
            result['pending_candidate_additions'] = deepcopy(inputs['candidate_additions'])
            return result
        result['candidates'].append(checked)
    # An explicit new hypothesis competes with the retained bank even if that
    # bank still fits. Editing/removing the submission never erases an old rule.
    for index, supplied in enumerate(inputs['candidate_additions']):
        item = _authored_candidate(supplied, inputs, 'condition_' + str(len(result['candidates'])))
        if any(_same_rule(item, old) for old in result['candidates']):
            continue
        checked = _screen(item, inputs, budget)
        if checked is None:
            result.update(status='search_budget_exhausted', compute_cost=budget.spent,
                pending_candidate_additions=deepcopy(inputs['candidate_additions'][index:]))
            return result
        result['candidates'].append(checked)
    representation_changed = bool(previous) and any(old_inputs.get(key) != inputs[key]
        for key in ('measurements', 'bound_rows', 'residual_sources'))
    if representation_changed or not any(item['status'] == 'compatible' for item in result['candidates']):
        derivations = _residuals(packet)
        if not budget.spend(len(responses) * len(derivations)):
            result.update(status='search_budget_exhausted', compute_cost=budget.spent)
            return result
        rows = [Observation(_features(inputs['measurements'][str(row['crystal'])],
            inputs['bound_rows'].get(str(row['crystal']), {}), row['frequency'], derivations),
            float(row['activated'])) for row in responses]
        seed_names = list(derivations) or list(rows[0].features)
        seeds = _transition_seeds(rows, seed_names, budget)
        allowance = SearchBudget(min(int(budget.remaining * .7), len(rows) * 512))
        generated = synthesize(rows, operations=('relation', 'composition'), budget=allowance,
                               max_history=0, seed_programs=seeds)
        budget.spend(allowance.spent)
        result.update(regenerated=True, generation_compute_cost=allowance.spent,
                      generated_candidate_count=len(generated))
        known = {(str(item['program']), str(item['feature_derivations']), str(item['bound_rows']))
                 for item in result['candidates']}
        screen_end = budget.spent + int(budget.remaining * .6)
        proposals = [*(seed for seed in seeds if seed.op == 'and'),
                     *(proposal.program for proposal in generated),
                     *(seed for seed in seeds if seed.op != 'and')]
        for proposed in proposals:
            program = _normalize_condition(proposed.simplified())
            if not _boolean(program):
                continue
            required = program.required_features()
            used = {key: value for key, value in derivations.items() if key in required}
            key = str(program.to_dict()), str(used), str(inputs['bound_rows'])
            if key in known:
                continue
            known.add(key)
            if budget.spent + len(rows) * (1 + len(used)) > screen_end:
                break
            item = {'id': 'condition_' + str(len(result['candidates'])), 'program': program.to_dict(),
                'feature_derivations': deepcopy(used), 'bound_rows': deepcopy(inputs['bound_rows']),
                'claim': 'Activation iff ' + program.describe(),
                'proposed_evidence': sorted({i for row in responses for i in row['evidence']})}
            checked = _screen(item, inputs, budget)
            if checked is not None:
                result['candidates'].append(checked)
    result['status'] = ('compatible_unconfirmed_conditions' if any(
        item['status'] == 'compatible' for item in result['candidates']) else 'no_compatible_rule_proposed')
    _experiments(result, inputs, budget)
    _target_consequences(result, inputs, budget)
    result['compute_cost'] = budget.spent
    return result

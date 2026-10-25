"""Learn optional scalar edit maps from retained before/after substitutions.

Targets here are saved Program constants, never physical response labels. The
ordinary grammar and strict slot fitter propose empirical maps; held revisions
can reject them. Nothing in this module changes a world-model success gate.
"""

from collections import deque
from copy import deepcopy
from itertools import product
from math import isfinite

from .abstractions import AbstractTemplate, fit_parameter_slots
from .language import Program
from .models import Observation
from .parameter_roles import describe_bound_parameter_roles, role_compatible
from .synthesis import OPERATIONS, iter_synthesis_structures


class _BudgetExhausted(Exception):
    pass


def target_conflicts(features, targets, indices, names, tolerance, budget, *, coordinate_scales=None):
    """Find a necessary single-valued-target failure in one declared view.

    Only exact input identity is grouped. Source-record indices reference the
    caller's existing source table; no labels are added and no Program executes.
    Targets may be canonical coefficients. ``coordinate_scales`` then converts
    each saved raw coefficient to that coordinate; tolerance stays in the raw
    coefficient's original units. A missing conflict is not sufficiency.
    """
    started = budget.spent
    names, indices = list(names), list(indices)
    result = {'status': 'UNKNOWN', 'input_names': names, 'checked_indices': [],
        'witnesses': [], 'budget_exhausted': False,
        'reason': 'no_identical_input_conflict_found_not_sufficiency',
        'scope': 'single_valued_after_parameter_on_declared_source_view',
        'tolerance_coordinate': 'saved_raw_parameter',
        'source_record_index_table': 'source_records', 'actual_compute_cost': 0}
    groups = {}
    for index in indices:
        if not budget.spend(1):
            result.update(budget_exhausted=True, reason='target_conflict_diagnosis_budget_exhausted')
            break
        values = tuple(features[index][name] for name in names)
        groups.setdefault(values, []).append(index)
        result['checked_indices'].append(index)
    for values, records in groups.items():
        expected = [targets[index] for index in records]
        scales = [1. if coordinate_scales is None else coordinate_scales[index] for index in records]
        if any(type(scale) not in (int, float) or not isfinite(scale) or scale == 0 for scale in scales):
            result['reason'] = 'explicit_finite_nonzero_parameter_coordinate_scale_required'
            continue
        span = max(expected) - min(expected)
        lower = max(value - tolerance * abs(scale) for value, scale in zip(expected, scales))
        upper = min(value + tolerance * abs(scale) for value, scale in zip(expected, scales))
        if lower > upper:
            bound = max(abs(left - right) / (abs(left_scale) + abs(right_scale))
                for left, left_scale in zip(expected, scales)
                for right, right_scale in zip(expected, scales))
            result['witnesses'].append({'source_record_indices': records,
                'input_values': list(values), 'target_values': expected,
                'coordinate_scales': scales, 'target_span': span,
                'common_canonical_interval': {'lower': lower, 'upper': upper},
                'necessary_max_abs_error_lower_bound': bound,
                'error_bound_coordinate': 'saved_raw_parameter'})
    if result['witnesses']:
        result.update(status='conflict', reason='identical_inputs_have_incompatible_parameter_targets')
    result['actual_compute_cost'] = budget.spent - started
    return result


def diagnose_parameter_targets(candidate, records, budget):
    """Resolve saved constant coordinates without predicting or fitting.

    A shared hole position is not a shared coefficient role. Actual Programs
    take precedence over fixture declarations. Missing fitter state requests
    source work, while finite empirical maps remain possible on known roles.
    """
    started = budget.spent
    result = {'status': 'UNKNOWN', 'constants': {}, 'raw_constants': {},
        'input_holes': [], 'output_holes': [], 'parameter_roles': {},
        'excluded': [], 'next_actions': [], 'budget_exhausted': False,
        'actual_compute_cost': 0}
    descriptions = [describe_bound_parameter_roles(candidate, record, budget)['roles']
                    for record in records]
    for hole in candidate['holes']:
        name, values = hole['name'], []
        for record in records:
            if not budget.spend(1):
                result['budget_exhausted'] = True
                break
            value = record.get('bindings', {}).get(name)
            value = Program.from_dict(value) if isinstance(value, dict) else value
            if (not isinstance(value, Program) or value.op != 'const'
                    or type(value.value) not in (int, float) or not isfinite(value.value)):
                break
            values.append(float(value.value))
        if len(values) != len(records) or not records:
            continue
        result['raw_constants'][name] = values
        roles = [description.get(name, {}) for description in descriptions]
        bad = [index for index, role in enumerate(roles) if not role_compatible(roles[0], role)
               or role.get('coefficient_unit') != roles[0].get('coefficient_unit')]
        if hole.get('output_unit') is None:
            reason = 'explicit_parameter_unit_required'
        elif bad:
            reason = 'common_computational_parameter_role_required'
        else:
            reason = None
        if reason:
            result['excluded'].append({'hole': name, 'reason': reason,
                'source_record_indices': bad, 'source_reasons': [
                    {'source_record_index': index, 'reason': roles[index].get('reason',
                        'incompatible_computational_parameter_role')} for index in bad]})
            result['next_actions'].append({'reason': reason, 'output_hole': name,
                'required_inputs': ['actual_Program_parameter_role_and_unique_binding'],
                'source_record_indices': bad})
            continue
        result['constants'][name] = [role['canonical_parameter'] for role in roles]
        result['parameter_roles'][name] = {key: deepcopy(roles[0][key]) for key in (
            'status', 'role_key', 'coordinate_scale', 'coefficient_unit')}
        result['parameter_roles'][name]['source_coordinate_scales'] = [
            role['coordinate_scale'] for role in roles]
        result['input_holes' if name in candidate['parent_holes'] else 'output_holes'].append(hole)
    for index, description in enumerate(descriptions):
        if any(role.get('reason') == 'parameter_role_budget_exhausted' for role in description.values()):
            result['budget_exhausted'] = True
    fitter = [index for index, record in enumerate(records) if record.get('fit_state_refs')]
    direct = [index for index, record in enumerate(records)
              if record.get('generating_process') == 'direct_program_edit']
    missing = [index for index in range(len(records)) if index not in fitter and index not in direct]
    generating_process = ('historical_fitter_output' if len(fitter) == len(records) and records
        else 'direct_program_edit' if len(direct) == len(records) and records else 'UNKNOWN')
    result['target_contract'] = {'purpose': 'exact_edit_map', 'target': 'saved_after_parameter',
        'generating_process': generating_process, 'target_identifiability': 'UNKNOWN',
        'input_sufficiency': 'UNKNOWN', 'scope': 'retained_revision_parameters_only',
        'fitter_source_record_indices': fitter, 'unknown_source_record_indices': missing}
    if missing:
        result['next_actions'].append({'reason': 'saved_parameter_generating_process_not_established',
            'required_inputs': ['saved_fitter_state_or_explicit_direct_edit_provenance'],
            'source_record_indices': missing})
    result['status'] = ('known_roles' if result['input_holes'] and result['output_holes']
                        and not result['budget_exhausted'] else 'UNKNOWN')
    result['actual_compute_cost'] = budget.spent - started
    return result


def learn_parameter_relations(candidate, source_bindings_records, budget, *, search_contract):
    """Search the original complete-input metadata frontier without reordering it."""
    return _learn_parameter_relations(candidate, source_bindings_records, budget,
                                     search_contract=search_contract)


def learn_joint_parameter_relations(candidate, source_bindings_records, budget, *, search_contract):
    """Search a bounded sparse view after the original metadata frontiers.

    ``max_structures`` bounds joint attempts per output across all parent ×
    evidence pairs. Each pair reuses the original six-family grammar on its two
    actual coordinates. Layers cover its first affine, interaction and
    conditional joint structures before consuming the remaining stream.
    """
    return _learn_parameter_relations(candidate, source_bindings_records, budget,
                                     search_contract=search_contract, joint=True)


def _joint_structures(rows, operations, budget, parents, evidence, stats):
    """Reorder existing factories' outputs; never author an edit expression."""
    groups = []
    skipped = {}
    for parent, condition in product(sorted(parents), sorted(evidence)):
        names = (parent, condition)
        projected = [Observation({name: row.features[name] for name in names}, row.target)
                     for row in rows]
        cursor = {}
        groups.append({'names': names, 'cursor': cursor, 'pending': [], 'finished': False,
            'stream': iter_synthesis_structures(projected, operations=operations, budget=budget,
                                                max_history=0, cursor=cursor)})
    stats.update(pair_count=len(groups), generated_structure_count=0, joint_attempt_count=0,
                 skipped_non_joint_count=0, skipped=[], complete=False, limit_reached=False,
                 scheduling_phases={})

    def conditional(node):
        return node.op == 'if' or any(conditional(child) for child in node.children)

    def matches(item, phase):
        if phase == 'conditional':
            return conditional(item['program'])
        return phase == 'remaining' or item['operation'] == phase

    def next_joint(group, phase):
        for index, item in enumerate(group['pending']):
            if matches(item, phase):
                return group['pending'].pop(index)
        while not group['finished']:
            try:
                item = next(group['stream'])
            except StopIteration:
                if group['cursor'].get('budget_exhausted'):
                    raise _BudgetExhausted
                group['finished'] = True
                break
            stats['generated_structure_count'] += 1
            if not set(group['names']).issubset(item['program'].required_features()):
                stats['skipped_non_joint_count'] += 1
                key = (group['names'], item['operation'])
                if key not in skipped:
                    skipped[key] = {'input_group': list(group['names']), 'operation': item['operation'],
                        'reason': 'structure_does_not_depend_on_both_parent_and_evidence', 'count': 0}
                    stats['skipped'].append(skipped[key])
                skipped[key]['count'] += 1
                continue
            if matches(item, phase):
                return item
            group['pending'].append(item)
        return None

    def selected(item, group, phase):
        stats['joint_attempt_count'] += 1
        stats['scheduling_phases'][phase] = stats['scheduling_phases'].get(phase, 0) + 1
        return {**item, 'input_group': list(group['names']), 'scheduling_phase': phase}

    # One layer across every pair precedes the next layer. A large vocabulary
    # cannot put every sparse affine behind dense or single-coordinate fits.
    phases = []
    if 'refit' in operations:
        phases.append('refit')
    if 'composition' in operations:
        phases.append('composition')
    if set(operations) & {'relation', 'robustification'}:
        phases.append('conditional')
    for phase in phases:
        for group in groups:
            item = next_joint(group, phase)
            if item is not None:
                yield selected(item, group, phase)
    queue = deque(groups)
    while queue:
        group = queue.popleft()
        item = next_joint(group, 'remaining')
        if item is not None:
            queue.append(group)
            yield selected(item, group, 'remaining')
    stats['complete'] = True


def _learn_parameter_relations(candidate, source_bindings_records, budget, *, search_contract, joint=False):
    """Return bounded, by-reference empirical maps for open constant holes.

    Each record supplies source_id, episode_id, original Program references and
    a transient ``bindings: hole -> Program`` from exact joint reconstruction.
    An optional evidence_context supplies common numeric features, explicit
    role_signature and their measured selection/source references. These are
    inputs; only the saved after-Program constant is a target.
    The last unique record of each physical source is held out. Grammar streams
    alternate output holes; max_structures bounds each stream. All streams,
    fitting and numerical checks debit the supplied shared SearchBudget.
    ``tolerance`` is absolute parameter error, independently of world MSE.
    """
    from .revision_relations import (_Incomplete, _MissingContract,
                                    _check_generated_units, _source_references)

    started = budget.spent
    receipt = {'status': 'UNKNOWN', 'accepted': [], 'attempts': [], 'excluded': [],
        'source_records': [], 'next_actions': [], 'target_conflicts': {},
        'actual_compute_cost': 0, 'budget_exhausted': False,
        'scope': 'retained_revision_parameter_metadata_only', 'support_updated': False,
        'scientific_support': False, 'independent_confirmation': False}

    def spend(units=1):
        if not budget.spend(units):
            raise _BudgetExhausted

    def source_record(record):
        return {**_source_references(record), **({'evidence_context': deepcopy(record['evidence_context'])}
                                               if 'evidence_context' in record else {})}

    def refs(index):
        record = records[index]
        return {'source_record_index': index, **{key: deepcopy(record[key]) for key in (
            'source_id', 'episode_id', 'source_ref', 'after_ref') if key in record}}

    try:
        operations = search_contract.get('operations')
        maximum, quantile = search_contract.get('max_structures'), search_contract.get('risk_quantile')
        tolerance = search_contract.get('tolerance')
        if (not isinstance(operations, list) or not operations or set(operations) - set(OPERATIONS)
                or type(maximum) is not int or maximum < 1 or quantile != .5
                or type(tolerance) not in (int, float) or not isfinite(tolerance) or tolerance < 0):
            receipt['reason'] = 'finite_parameter_relation_search_contract_required'
            return receipt
        receipt['contract'] = deepcopy(search_contract)
        records, identities = [], []
        for record in source_bindings_records:
            spend()
            if not all(record.get(key) for key in ('source_id', 'episode_id', 'before_ref', 'after_ref')):
                receipt['reason'] = 'actual_revision_identity_and_references_required'
                return receipt
            identity = (record['source_id'], record['episode_id'], record['before_ref'], record['after_ref'])
            if identity in identities:
                continue
            identities.append(identity)
            records.append(record)
            receipt['source_records'].append(source_record(record))
        by_source = {}
        for index, record in enumerate(records):
            by_source.setdefault(record['source_id'], []).append(index)
        if len(by_source) < 2 or any(len(indices) < 2 for indices in by_source.values()):
            receipt['reason'] = 'two_physical_sources_with_training_and_held_revisions_required'
            return receipt
        held_indices = {indices[-1] for indices in by_source.values()}
        training = [index for index in range(len(records)) if index not in held_indices]
        held = [index for index in range(len(records)) if index in held_indices]
        receipt['split'] = {'rule': 'last_actual_revision_per_physical_source',
            'training': [refs(index) for index in training],
            'held': [refs(index) for index in held], 'physical_source_count': len(by_source)}
        holes = candidate['holes']
        diagnosis = diagnose_parameter_targets(candidate, records, budget)
        receipt['parameter_target_diagnosis'] = {key: deepcopy(value) for key, value in diagnosis.items()
            if key not in ('constants', 'raw_constants', 'input_holes', 'output_holes', 'next_actions')}
        receipt['next_actions'].extend(diagnosis['next_actions'])
        receipt['excluded'].extend(diagnosis['excluded'])
        receipt['target_contract'] = diagnosis['target_contract']
        if diagnosis['budget_exhausted']:
            raise _BudgetExhausted
        constants, inputs, outputs = (diagnosis['constants'], diagnosis['input_holes'], diagnosis['output_holes'])
        if not inputs or not outputs:
            receipt['reason'] = 'common_computational_parent_and_after_parameter_roles_required'
            return receipt
        parameter_roles = diagnosis['parameter_roles']
        roles = {hole['name']: {'role': 'before_program_parameter',
                               'unit': parameter_roles[hole['name']]['coefficient_unit'],
                               'time': 'before_revision'} for hole in inputs}
        features = [{hole['name']: constants[hole['name']][index] for hole in inputs}
                    for index in range(len(records))]
        evidence_inputs = set()
        contexts = [record.get('evidence_context', {}) for record in records]
        context_names = sorted({name for context in contexts if isinstance(context, dict)
                                for name in context.get('features', {})})
        for name in context_names:
            contract, selection, values, reason = None, None, [], None
            if name in {hole['name'] for hole in holes}:
                reason = 'evidence_feature_conflicts_with_program_hole'
            for context in contexts if reason is None else ():
                spend()
                if not isinstance(context, dict) or name not in context.get('features', {}):
                    reason = 'evidence_feature_not_shared_by_all_revisions'
                    break
                value = context['features'][name]
                declared = context.get('role_signature', {}).get(name, {})
                current_selection = context.get('selection')
                if (type(value) not in (int, float) or not isfinite(value)
                        or not isinstance(declared, dict)
                        or any(declared.get(key) is None for key in ('role', 'unit', 'time'))
                        or not isinstance(current_selection, str) or not current_selection
                        or not context.get('source_refs')):
                    reason = 'explicit_measured_evidence_feature_contract_required'
                    break
                declared = {key: deepcopy(declared[key]) for key in ('role', 'unit', 'time')}
                if contract is not None and (declared != contract or current_selection != selection):
                    reason = 'incompatible_evidence_feature_contract'
                    break
                contract, selection = declared, current_selection
                values.append(float(value))
            if reason is not None:
                receipt['excluded'].append({'feature': name, 'reason': reason})
                continue
            roles[name] = {**contract, 'selection': selection}
            evidence_inputs.add(name)
            for index, value in enumerate(values):
                features[index][name] = value
        receipt['input_scope'] = ('before_parameters_and_measured_evidence' if evidence_inputs
                                  else 'before_parameters_only')
        available_outputs = []
        for hole in outputs:
            conflict = target_conflicts(features, constants[hole['name']], training,
                sorted(roles), tolerance, budget,
                coordinate_scales=parameter_roles[hole['name']]['source_coordinate_scales'])
            receipt['target_conflicts'][hole['name']] = conflict
            if conflict['budget_exhausted']:
                raise _BudgetExhausted
            if conflict['status'] == 'conflict':
                receipt['next_actions'].append({'reason': 'exact_edit_map_input_view_conflicts',
                    'output_hole': hole['name'], 'input_names': sorted(roles),
                    'required_inputs': ['distinct_computational_role_or_declared_additional_source_state'],
                    'conflict_ref': ['target_conflicts', hole['name']]})
            else:
                available_outputs.append(hole)
        outputs = available_outputs
        if not outputs:
            receipt['reason'] = 'all_exact_parameter_target_views_conflict'
            return receipt
        if len(training) < 3:
            receipt['reason'] = 'insufficient_parameter_training_points'
            return receipt
        if joint:
            receipt['input_scope'] = 'sparse_parent_evidence_pairs'
            receipt['frontier'] = {'per_output_limit': maximum, 'outputs': {}}
            receipt['frontier_truncated'] = False
            if not evidence_inputs:
                receipt['reason'] = 'shared_typed_measured_evidence_inputs_required'
                return receipt
        queue, conflicting_views = deque(), set()
        for hole in outputs:
            rows = [Observation(features[index], constants[hole['name']][index]) for index in training]
            cursor = {}
            if joint:
                stats = receipt['frontier']['outputs'][hole['name']] = {}
                stream = _joint_structures(rows, operations, budget, roles.keys() - evidence_inputs,
                                           evidence_inputs, stats)
            else:
                stream = iter_synthesis_structures(rows, operations=operations, budget=budget,
                                                   max_history=0, cursor=cursor)
            queue.append((hole, rows, stream, cursor, 0))
        while queue:
            hole, rows, stream, cursor, count = queue.popleft()
            if count >= maximum:
                if joint:
                    receipt['frontier']['outputs'][hole['name']]['limit_reached'] = True
                    receipt['frontier_truncated'] = True
                continue
            try:
                structure = next(stream)
            except StopIteration:
                if cursor.get('budget_exhausted'):
                    raise _BudgetExhausted
                continue
            queue.append((hole, rows, stream, cursor, count + 1))
            program, paths = structure['program'], structure['parameter_paths']
            attempt = {'output_hole': hole['name'], 'operation': structure['operation'],
                'proposal_index': structure['proposal_index'], 'free_parameter_count': len(paths),
                'status': 'UNKNOWN', 'checks': []}
            if joint:
                attempt.update(input_group=structure['input_group'],
                               scheduling_phase=structure['scheduling_phase'])
            receipt['attempts'].append(attempt)
            names = sorted(program.required_features())
            attempt['input_holes'] = [name for name in names if name not in evidence_inputs]
            attempt['evidence_inputs'] = [name for name in names if name in evidence_inputs]
            conflict = target_conflicts(features, constants[hole['name']], training,
                names, tolerance, budget,
                coordinate_scales=parameter_roles[hole['name']]['source_coordinate_scales'])
            attempt['target_conflict'] = conflict
            if conflict['budget_exhausted']:
                raise _BudgetExhausted
            if conflict['status'] == 'conflict':
                attempt.update(status='refuted', reason='exact_edit_map_input_view_conflicts')
                view = (hole['name'], tuple(names))
                if view not in conflicting_views:
                    conflicting_views.add(view)
                    receipt['next_actions'].append({'reason': 'exact_edit_map_input_view_conflicts',
                        'output_hole': hole['name'], 'input_names': names,
                        'required_inputs': ['distinct_computational_role_or_declared_additional_source_state'],
                        'conflict_ref': ['attempts', len(receipt['attempts']) - 1, 'target_conflict']})
                continue
            distinct = []
            for index in training:
                values = tuple(features[index][name] for name in names)
                if values not in distinct:
                    distinct.append(values)
            attempt['distinct_training_inputs'] = len(distinct)
            if not names:
                attempt.update(status='not_applicable', reason='no_parent_parameter_dependency')
                continue
            if len(distinct) < max(3, len(paths) + 1):
                attempt['reason'] = 'insufficient_parameter_variation_or_points_for_free_parameters'
                continue
            try:
                _check_generated_units(program, parameter_roles[hole['name']]['coefficient_unit'], roles, {}, budget)
            except _MissingContract as error:
                attempt.update(reason=str(error))
                continue
            template = AbstractTemplate(f'{candidate["id"]}:parameter:{hole["name"]}', candidate['id'],
                program, (), (), tuple(paths))
            report = {}
            fitted = fit_parameter_slots(template, {}, rows, budget, quantile, report=report)
            attempt['fitting'] = {key: value for key, value in report.items() if key != 'losses'}
            solver_checks = []
            for fit_index, hypothesis in enumerate(fitted):
                method = report['candidate_methods'][fit_index]
                checks = {'program': hypothesis.to_dict(), 'fitting_method': method,
                          'training': [], 'held': [], 'status': 'UNKNOWN'}
                attempt['checks'].append(checks)
                for label, indices in (('training', training), ('held', held)):
                    for index in indices:
                        spend()
                        canonical_observed = constants[hole['name']][index]
                        canonical_predicted = hypothesis.predict(Observation(features[index], canonical_observed))
                        scale = parameter_roles[hole['name']]['source_coordinate_scales'][index]
                        observed = diagnosis['raw_constants'][hole['name']][index]
                        predicted = canonical_predicted / scale
                        residual = predicted - observed
                        checks[label].append({**refs(index), 'expected_parameter': observed,
                            'predicted_parameter': predicted, 'residual': residual,
                            'coordinate_scale': scale,
                            'canonical_expected_parameter': canonical_observed,
                            'canonical_predicted_parameter': canonical_predicted,
                            'canonical_residual': canonical_predicted - canonical_observed,
                            'tolerance_coordinate': 'saved_raw_parameter',
                            'status': 'passed' if isfinite(predicted) and abs(residual) <= tolerance else 'failed'})
                passed = all(check['status'] == 'passed'
                             for label in ('training', 'held') for check in checks[label])
                checks['status'] = 'passed' if passed else 'failed'
                solver_checks.append({'method': method, **{
                    label: {'passed': all(check['status'] == 'passed' for check in checks[label]),
                            'max_abs_residual': max(abs(check['residual']) for check in checks[label])}
                    for label in ('training', 'held')}})
                if not passed:
                    continue
                existing = next((index for index, item in enumerate(receipt['accepted'])
                    if item['output_hole'] == hole['name'] and item['program'] == hypothesis.to_dict()), None)
                if existing is not None:
                    checks.clear()
                    checks.update(status='passed', accepted_index=existing, fitting_method=method)
                    continue
                receipt['accepted'].append({'output_hole': hole['name'], 'program': hypothesis.to_dict(),
                    'purpose': 'exact_edit_map', 'parameter_coordinate': 'canonical_basis',
                    'input_roles': {name: deepcopy(parameter_roles[name]) for name in names
                                    if name not in evidence_inputs},
                    'output_role': deepcopy(parameter_roles[hole['name']]),
                    'target_contract': deepcopy(diagnosis['target_contract']),
                    'input_holes': [name for name in names if name not in evidence_inputs],
                    'evidence_inputs': [name for name in names if name in evidence_inputs],
                    'input_contracts': {name: deepcopy(roles[name]) for name in names},
                    'output_unit': hole['output_unit'], 'source_refs': [refs(index) for index in range(len(records))],
                    'training_checks': deepcopy(checks['training']), 'held_checks': deepcopy(checks['held']),
                    'fitting_method': method,
                    'parameter_identifiability': report['parameter_identifiability'],
                    'status': 'empirical_candidate', 'scope': 'retained_revision_parameters_only',
                    'support_updated': False, 'scientific_support': False, 'independent_confirmation': False})
                # Accepted evidence lives once in the executable map. Attempts
                # retain its index; failed/truncated evidence remains inline.
                checks.clear()
                checks.update(status='passed', accepted_index=len(receipt['accepted']) - 1,
                              fitting_method=method)
            stable = next((item for item in solver_checks if item['method'] == 'stable_lstsq'), None)
            ridge = next((item for item in solver_checks if item['method'] == 'ridge_lstsq'), None)
            outcome = 'UNKNOWN'
            if stable is not None:
                outcome = ('stable_training_mismatch' if not stable['training']['passed'] else
                    'held_out_failure' if not stable['held']['passed'] else
                    'numerical_rejection_corrected' if ridge is not None and not all(
                        ridge[label]['passed'] for label in ('training', 'held')) else 'stable_passed')
            attempt['numerical_diagnosis'] = {
                'observed_outcome': outcome, 'solver_checks': solver_checks,
                'parameter_identifiability': report['parameter_identifiability'],
                'scope': 'returned_Programs_on_same_saved_training_and_held_rows',
                'structure_impossibility_established': False}
            attempt['status'] = ('accepted' if any(check['status'] == 'passed' for check in attempt['checks'])
                                 else 'UNKNOWN' if report['status'] == 'UNKNOWN' else 'failed')
        receipt['status'] = ('UNKNOWN' if receipt.get('frontier_truncated') or any(
            item['status'] == 'UNKNOWN' or item.get('numerical_diagnosis', {}).get(
                'parameter_identifiability') == 'UNKNOWN' for item in receipt['attempts'])
                             else 'completed')
        if receipt.get('frontier_truncated'):
            receipt['reason'] = 'joint_parameter_frontier_limit_reached'
    except (_BudgetExhausted, _Incomplete):
        receipt.update(status='UNKNOWN', budget_exhausted=True, reason='parameter_relation_compute_budget_exhausted')
    finally:
        receipt['actual_compute_cost'] = budget.spent - started
    return receipt

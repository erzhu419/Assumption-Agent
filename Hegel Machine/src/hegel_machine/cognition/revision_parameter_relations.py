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
from .synthesis import OPERATIONS, iter_synthesis_structures


class _BudgetExhausted(Exception):
    pass


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
        'source_records': [],
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
        if len(training) < 3:
            receipt['reason'] = 'insufficient_parameter_training_points'
            return receipt
        holes = candidate['holes']
        constants = {}
        for hole in holes:
            name = hole['name']
            if hole.get('output_unit') is None:
                receipt['reason'] = 'explicit_parameter_unit_required'
                return receipt
            values = []
            for record in records:
                spend()
                value = record.get('bindings', {}).get(name)
                value = Program.from_dict(value) if isinstance(value, dict) else value
                if (not isinstance(value, Program) or value.op != 'const'
                        or type(value.value) not in (int, float) or not isfinite(value.value)):
                    break
                values.append(float(value.value))
            if len(values) == len(records):
                constants[name] = values
        inputs = [hole for hole in holes if hole['name'] in candidate['parent_holes']
                  and hole['name'] in constants]
        outputs = [hole for hole in holes if hole['name'] not in candidate['parent_holes']
                   and hole['name'] in constants]
        if not inputs or not outputs:
            receipt['reason'] = 'common_parent_and_open_after_constant_holes_required'
            return receipt
        roles = {hole['name']: {'role': 'before_program_parameter', 'unit': hole['output_unit'],
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
        if joint:
            receipt['input_scope'] = 'sparse_parent_evidence_pairs'
            receipt['frontier'] = {'per_output_limit': maximum, 'outputs': {}}
            receipt['frontier_truncated'] = False
            if not evidence_inputs:
                receipt['reason'] = 'shared_typed_measured_evidence_inputs_required'
                return receipt
        queue = deque()
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
                _check_generated_units(program, hole['output_unit'], roles, {}, budget)
            except _MissingContract as error:
                attempt.update(reason=str(error))
                continue
            template = AbstractTemplate(f'{candidate["id"]}:parameter:{hole["name"]}', candidate['id'],
                program, (), (), tuple(paths))
            report = {}
            fitted = fit_parameter_slots(template, {}, rows, budget, quantile, report=report)
            attempt['fitting'] = {key: value for key, value in report.items() if key != 'losses'}
            for hypothesis in fitted:
                checks = {'program': hypothesis.to_dict(), 'training': [], 'held': [], 'status': 'UNKNOWN'}
                attempt['checks'].append(checks)
                for label, indices in (('training', training), ('held', held)):
                    for index in indices:
                        spend()
                        observed = constants[hole['name']][index]
                        predicted = hypothesis.predict(Observation(features[index], observed))
                        residual = predicted - observed
                        checks[label].append({**refs(index), 'expected_parameter': observed,
                            'predicted_parameter': predicted, 'residual': residual,
                            'status': 'passed' if isfinite(predicted) and abs(residual) <= tolerance else 'failed'})
                passed = all(check['status'] == 'passed'
                             for label in ('training', 'held') for check in checks[label])
                checks['status'] = 'passed' if passed else 'failed'
                if not passed:
                    continue
                existing = next((index for index, item in enumerate(receipt['accepted'])
                    if item['output_hole'] == hole['name'] and item['program'] == hypothesis.to_dict()), None)
                if existing is not None:
                    checks.clear()
                    checks.update(status='passed', accepted_index=existing)
                    continue
                receipt['accepted'].append({'output_hole': hole['name'], 'program': hypothesis.to_dict(),
                    'input_holes': [name for name in names if name not in evidence_inputs],
                    'evidence_inputs': [name for name in names if name in evidence_inputs],
                    'input_contracts': {name: deepcopy(roles[name]) for name in names},
                    'output_unit': hole['output_unit'], 'source_refs': [refs(index) for index in range(len(records))],
                    'training_checks': deepcopy(checks['training']), 'held_checks': deepcopy(checks['held']),
                    'status': 'empirical_candidate', 'scope': 'retained_revision_parameters_only',
                    'support_updated': False, 'scientific_support': False, 'independent_confirmation': False})
                # Accepted evidence lives once in the executable map. Attempts
                # retain its index; failed/truncated evidence remains inline.
                checks.clear()
                checks.update(status='passed', accepted_index=len(receipt['accepted']) - 1)
            attempt['status'] = ('accepted' if any(check['status'] == 'passed' for check in attempt['checks'])
                                 else 'UNKNOWN' if not report['search_completed'] else 'failed')
        receipt['status'] = ('UNKNOWN' if receipt.get('frontier_truncated') or any(
            item['status'] == 'UNKNOWN' for item in receipt['attempts'])
                             else 'completed')
        if receipt.get('frontier_truncated'):
            receipt['reason'] = 'joint_parameter_frontier_limit_reached'
    except (_BudgetExhausted, _Incomplete):
        receipt.update(status='UNKNOWN', budget_exhausted=True, reason='parameter_relation_compute_budget_exhausted')
    finally:
        receipt['actual_compute_cost'] = budget.spent - started
    return receipt

"""Learn optional construction contexts and parent-anchored finite revisions.

The body is a native Definition with lexical subtree arguments. Its operations
come entirely from the sources. Joint revisions share typed variables across
before and after, while their measured triggers and property interpretations
remain scoped empirical candidates. Learned equalities concern native Program
properties; they restrict this finite proposer, never the old candidate bank.
"""

from copy import deepcopy
from collections import Counter
from itertools import chain, combinations, islice
import json

from .language import Definition, Program
from .models import Observation, SearchBudget


class _Incomplete(Exception):
    pass


class _MissingContract(Exception):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _unique(values):
    result, seen = [], set()
    for value in values:
        key = _json(value)
        if key not in seen:
            result.append(deepcopy(value))
            seen.add(key)
    return result


def _budget(allocation):
    allocation = allocation or {}
    compute = allocation.get('compute', 10000)
    if type(compute) is not int or compute < 0:
        raise ValueError('compute must be a nonnegative integer')
    return SearchBudget(compute)


def _spend(budget, units=1):
    if not budget.spend(units):
        raise _Incomplete


def _at(program, path):
    for index in path:
        program = program.children[index]
    return program


def _program(value):
    return value if isinstance(value, Program) else Program.from_dict(value)


# These are unit propagation rules for the existing language, not learned
# scientific corrections. A missing internal unit remains unknown.
_SAME_UNIT = {'add', 'sub', 'min', 'max', 'abs', 'window_sum', 'event_delta', 'object_sum'}
_PREDICATES = {'ge', 'lt', 'and', 'or', 'not', 'available'}


def _occurrence_contracts(pattern, program, holes, budget):
    """Recover all lexical occurrences, including repeated arguments, exactly."""
    _match_parent(pattern, program, holes, budget)
    by_name = {hole['name']: hole for hole in holes}
    contracts, origins = {}, {}

    def visit(node, path=()):
        _spend(budget)
        if node.op == 'arg':
            hole = by_name.get(node.name)
            if hole is None or hole.get('output_unit') is None:
                raise _MissingContract('original_hole_output_contract_required')
            key = '.'.join(map(str, path))
            contracts[key] = {field: deepcopy(hole[field]) for field in
                              ('output_unit', 'time_contract') if field in hole}
            origins[key] = {'hole_name': node.name, 'hole_index': holes.index(hole)}
        for index, child in enumerate(node.children):
            visit(child, (*path, index))

    visit(pattern)
    return contracts, origins


def _declared_contract_header(record):
    signatures = record.get('role_signature')
    if not isinstance(signatures, dict) or record.get('output_unit') is None or 'context_ref' not in record:
        raise _MissingContract('explicit_roles_output_unit_and_context_required')
    return signatures


def _prepare(record, program, budget, *, side=None):
    signatures = _declared_contract_header(record)
    variables = {name for name, _ in program.observed_variables()}
    for name in variables:
        if not isinstance(signatures.get(name), dict) or any(
                key not in signatures[name] or signatures[name][key] is None for key in ('role', 'unit', 'time')):
            raise _MissingContract('missing_role_unit_or_time_contract')
    times = _unique(signatures[name]['time'] for name in sorted(variables))
    if not times:
        times = deepcopy(record.get('time_contract', []))
    declarations = deepcopy(record.get(f'{side}_subtree_contracts',
        record.get('subtree_contracts', {}) if side == 'after' else {}) if side else
        record.get('subtree_contracts', {}))
    endpoint_ref = record.get(f'{side}_ref') if side else record.get('program_source_ref')
    recovery = {'side': side, 'program_ref': deepcopy(endpoint_ref),
                'status': 'not_supplied', 'occurrence_count': 0, 'sources': []}
    source = record.get(f'_{side}_contract_source') if side else record.get('_contract_source')
    if source:
        if source.get('reason'):
            raise _MissingContract(source['reason'])
        if source['source_ref']['program_ref'] != endpoint_ref:
            raise _MissingContract('retained_subtree_contract_endpoint_mismatch')
        recovered, origins = _occurrence_contracts(
            _program(source['_pattern']).expanded(), program, source['_holes'], budget)
        if any(origin['hole_name'] not in source['_bound_holes'] for origin in origins.values()):
            raise _MissingContract('retained_typed_hole_binding_required')
        for path, contract in recovered.items():
            previous = declarations.get(path, {})
            if any(previous.get(key) is not None and previous[key] != value
                   for key, value in contract.items()):
                raise _MissingContract(f'conflicting_retained_subtree_contract:{path}')
            declarations[path] = {**previous, **contract}
        recovery.update(status='recovered', occurrence_count=len(recovered),
            sources=[deepcopy(source['source_ref'])])
    elif declarations:
        recovery.update(status='declared', occurrence_count=len(declarations),
                        sources=deepcopy(record.get('subtree_contract_sources', [])))
    contracts, count = {}, 0

    def visit(node, path=(), expected_unit=None):
        nonlocal count
        _spend(budget)
        count += 1
        declared = declarations.get('.'.join(map(str, path)), {})
        if (expected_unit is not None and declared.get('output_unit') is not None
                and declared['output_unit'] != expected_unit):
            raise _MissingContract(f'conflicting_subtree_output_unit:{list(path)}:'
                                   f'{declared["output_unit"]}:{expected_unit}')
        own_unit = declared.get('output_unit', expected_unit)
        if node.op in {'feature', 'lag'}:
            if declared.get('output_unit') is not None and declared['output_unit'] != signatures[node.name]['unit']:
                raise _MissingContract(f'conflicting_feature_output_unit:{list(path)}')
            own_unit = signatures[node.name]['unit']
        elif node.op in _PREDICATES:
            own_unit = own_unit if own_unit is not None else 'dimensionless'
        child_units = [None] * len(node.children)
        if node.op in _SAME_UNIT:
            child_units = [own_unit] * len(node.children)
        elif node.op == 'if':
            child_units = ['dimensionless', own_unit, own_unit]
        elif node.op in {'ge', 'lt'}:
            units = []
            for index, child in enumerate(node.children):
                supplied = declarations.get(
                    '.'.join(map(str, (*path, index))), {}).get('output_unit')
                if child.op in {'feature', 'lag', 'const'} or supplied is not None:
                    known = {child: supplied} if supplied is not None and child.op not in {'feature', 'lag'} else {}
                    unit = _check_generated_units(child, None, signatures, known, budget)
                    if supplied is not None and unit != supplied:
                        raise _MissingContract('incompatible_comparison_operand_units')
                else:
                    _spend(budget)
                    unit = None  # No new inference for an undeclared compound operand.
                units.append(unit)
            concrete = [unit for unit in units if unit is not None]
            if concrete and any(unit != concrete[0] for unit in concrete):
                raise _MissingContract('incompatible_comparison_operand_units')
            child_units = [concrete[0] if concrete and (child.op == 'const' or unit is not None) else None
                           for child, unit in zip(node.children, units)]
        elif node.op == 'mul':
            if any(child.op == 'const' for child in node.children):
                child_units = ['dimensionless' if child.op == 'const' else own_unit for child in node.children]
            elif own_unit is not None:
                # The existing dimensionless-factor rule also applies to a
                # compound lexical argument with its original explicit unit.
                # Without that declaration, do not infer a factor's dimension.
                _spend(budget, len(node.children))
                units = [declarations.get('.'.join(map(str, (*path, index))), {}).get('output_unit')
                         for index in range(len(node.children))]
                for index, unit in enumerate(units):
                    if unit == 'dimensionless':
                        child_units[index] = unit
                        child_units[1 - index] = own_unit
                        break
        elif node.op == 'accumulate':
            child_units = [own_unit, own_unit]
        for index, child in enumerate(node.children):
            visit(child, (*path, index), child_units[index])
        if node.op == 'if' and own_unit is None:
            branches = [contracts[(*path, index)]['output_unit'] for index in (1, 2)]
            if all(unit is not None for unit in branches):
                if branches[0] != branches[1]:
                    raise _MissingContract(f'conflicting_if_branch_units:{list(path)}')
                own_unit = branches[0]
        declared_times = declared.get('time_contract')
        if declared_times and times and declared_times != times:
            raise _MissingContract(f'conflicting_subtree_time_contract:{list(path)}')
        contracts[path] = {'value_type': 'scalar', 'output_unit': own_unit,
                           'time_contract': deepcopy(times),
                           'used_role_signature': {name: deepcopy(signatures[name])
                               for name, _ in sorted(node.observed_variables())}}
        if node.op in {'feature', 'lag'}:
            contracts[path]['feature_contract'] = deepcopy(signatures[node.name])

    visit(program, expected_unit=record['output_unit'])
    return {'record': record, 'program': program, 'contracts': contracts, 'nodes': count,
            'contract_recovery': recovery}


def _anti_unify(sources, budget, before_sources=None):
    holes, replacements, paths, memo = [], [dict() for _ in sources], [dict() for _ in sources], {}

    def visit(nodes, path=(), active=None, side='after'):
        _spend(budget, len(nodes))
        active = sources if active is None else active
        contracts = [source['contracts'][path] for source in active]
        clocks = [contract['time_contract'] for contract in contracts if contract['time_contract']]
        common_time = clocks[0] if clocks else []
        # An unclocked constant can be evaluated in either source's clock.
        # Empty dependencies of an actual history operation remain unknown.
        timeless = [not node.observed_variables() and node.history_depth == 0 for node in nodes]
        comparable = all(contract['output_unit'] == contracts[0]['output_unit']
                         and (contract['time_contract'] == common_time or (
                             not contract['time_contract'] and independent))
                         for contract, independent in zip(contracts, timeless))
        if not comparable:
            error = _MissingContract('incompatible_subtree_unit_or_time')
            error.contract_conflict = {'side': side, 'path': list(path),
                'contracts': [{key: deepcopy(contract[key]) for key in ('output_unit', 'time_contract')}
                              for contract in contracts]}
            raise error
        contracts = [{**contract, 'time_contract': common_time} for contract in contracts]
        first = nodes[0]
        identical = all(node == first for node in nodes)
        if identical and all(contract == contracts[0] for contract in contracts):
            return first
        same_operator = all((node.op, node.value, node.name, node.lag, len(node.children)) ==
                            (first.op, first.value, first.name, first.lag, len(first.children)) for node in nodes)
        if same_operator and first.children:
            return Program(first.op, first.value, first.name, first.lag,
                           tuple(visit(tuple(node.children[index] for node in nodes), (*path, index), active, side)
                                 for index in range(len(first.children))))
        if contracts[0]['output_unit'] is None or (not common_time and not all(timeless)):
            error = _MissingContract('missing_subtree_output_unit_or_time')
            error.contract_conflict = {'side': side, 'path': list(path),
                'contracts': [{key: deepcopy(contract[key]) for key in ('output_unit', 'time_contract')}
                              for contract in contracts]}
            raise error
        key = (tuple(nodes), _json({key: contracts[0][key]
                                  for key in ('value_type', 'output_unit', 'time_contract')}))
        if key not in memo:
            name = f'hole_{len(holes)}'
            memo[key] = name
            holes.append({'name': name, **{key: deepcopy(contracts[0][key])
                           for key in ('value_type', 'output_unit', 'time_contract')}})
        name = memo[key]
        for index, node in enumerate(nodes):
            replacements[index][name] = node
            paths[index].setdefault(name, {'side': side, 'path': list(path)} if before_sources else list(path))
        return Program('arg', name=name)

    before = (visit(tuple(source['program'] for source in before_sources), active=before_sources, side='before')
              if before_sources else None)
    body = visit(tuple(source['program'] for source in sources))
    return (body, holes, replacements, paths, before) if before_sources else (body, holes, replacements, paths)


def _property_references(body, holes):
    result = [{'object': 'program', 'property': 'minimum_history_depth'}]
    result.extend({'object': 'hole', 'name': hole['name'], 'property': 'minimum_history_depth'} for hole in holes)

    def visit(node, path=()):
        if node.lag:
            result.append({'object': 'pattern', 'path': list(path), 'property': 'lag'})
        for index, child in enumerate(node.children):
            visit(child, (*path, index))

    visit(body)
    return result


def _property(reference, body, bindings, program, budget):
    _spend(budget)
    if reference['object'] == 'program':
        target = program
    elif reference['object'] == 'hole':
        target = bindings[reference['name']]
    else:
        target = _at(body, reference['path'])
    return getattr(target, reference['property'])


def _learn_constraints(body, holes, sources, bindings, budget):
    groups = []
    for reference in _property_references(body, holes):
        values = [_property(reference, body, substitution, source['program'], budget)
                  for source, substitution in zip(sources, bindings)]
        group = next((group for group in groups if group['values'] == values), None)
        if group is None:
            groups.append({'values': values, 'references': [reference]})
        else:
            group['references'].append(reference)
    constraints = []
    for group in groups:
        representative, *remaining = group['references']
        if len(set(group['values'])) == 1:
            constraints.append({'op': 'fixed', 'left': representative, 'value': group['values'][0]})
        constraints.extend({'op': 'equal', 'left': reference, 'right': representative} for reference in remaining)
    return constraints


def _constraint_checks(candidate, bindings, program, budget):
    body = Definition.from_dict(candidate['definition']).body
    checks = []
    for constraint in candidate['learned_constraints']:
        left = _property(constraint['left'], body, bindings, program, budget)
        right = constraint['value'] if constraint['op'] == 'fixed' else _property(
            constraint['right'], body, bindings, program, budget)
        checks.append({'constraint': deepcopy(constraint), 'left_value': left,
                       'right_value': right, 'status': 'passed' if left == right else 'violated'})
    return checks


def _expanded_size(body, bindings):
    return bindings[body.name].size if body.op == 'arg' else 1 + sum(
        _expanded_size(child, bindings) for child in body.children)


def _instantiate(candidate, bindings, budget):
    definition = Definition.from_dict(candidate['definition'])
    _spend(budget, _expanded_size(definition.body, bindings))
    return Program('call', name=definition.definition_id, definition=definition,
                   children=tuple(bindings[name] for name in definition.arguments)).expanded()


def _source_references(record):
    result = {key: deepcopy(record[key]) for key in ('episode_id', 'source_id', 'source_ref', 'before_ref',
        'after_ref', 'action_ref', 'evidence_refs', 'cost_refs', 'failure_refs', 'source_cutoff', 'source_outcomes',
        'source_comparison', 'trigger_obligations', 'fit_state_refs')
            if key in record}
    for side in ('before', 'after'):
        source = record.get(f'_{side}_contract_source', {})
        if source.get('source_ref'):
            result[f'{side}_subtree_contract_source'] = deepcopy(source['source_ref'])
    return result


def _ordered_skeleton(program, budget, *, canonical_roles=True):
    names = {}

    def visit(node):
        _spend(budget)
        name = node.name
        if canonical_roles and node.op in {'feature', 'lag'}:
            name = names.setdefault(name, len(names))
        return (node.op, None if node.op == 'const' else node.value,
                name, node.lag, tuple(visit(child) for child in node.children))

    return visit(program)


def _compile(sources, identifier, budget):
    skeletons = [_ordered_skeleton(source['program'], budget) for source in sources]
    if all(skeleton == skeletons[0] for skeleton in skeletons[1:]):
        return None
    body, holes, bindings, paths = _anti_unify(sources, budget)
    if body.op == 'arg' or not holes:
        raise _MissingContract('no_nontrivial_shared_construction_context')
    definition = Definition(identifier, tuple(hole['name'] for hole in holes), body)
    records = [source['record'] for source in sources]
    candidate = {'id': identifier, 'kind': 'program_context_relation', 'definition': definition.to_dict(),
        'holes': holes, 'source_ids': _unique(record['source_id'] for record in records),
        'source_contexts': _unique(record['context_ref'] for record in records),
        'source_outcomes': [_source_references(record) for record in records],
        'source_bindings': [{**_source_references(record), 'bindings_refs': {
            name: {'source_ref': deepcopy(record.get('after_ref', record.get('source_ref'))),
                   'program_path': paths[index][name]} for name in definition.arguments}}
            for index, record in enumerate(records)],
        'source_reconstruction': [], 'scope': {
            'context_family_ref': deepcopy(records[0].get('context_family_ref', records[0]['context_ref']))},
        'output_unit': records[0]['output_unit'],
        'fixed_role_signature': {name: deepcopy(records[0]['role_signature'][name])
            for name, _ in body.observed_variables()},
        'learned_constraints': _learn_constraints(body, holes, sources, bindings, budget),
        'constraint_scope': 'optional_proposer_only', 'construction_only': True,
        'status': 'EXECUTABLE_CANDIDATE', 'frozen': True, 'active_library': False,
        'transfer_status': 'not_established', 'support_updated': False, 'global_support': False,
        'independent_confirmation': False, 'scientific_support_added': False,
        'counterexample_refs': _unique(reference for record in records for reference in record.get('failure_refs', []))}
    for source, substitution in zip(sources, bindings):
        rebuilt = _instantiate(candidate, substitution, budget)
        if rebuilt != source['program']:
            raise ValueError('relation source reconstruction failed')
        checks = _constraint_checks(candidate, substitution, rebuilt, budget)
        candidate['source_reconstruction'].append({**_source_references(source['record']),
            'exact': True, 'target': 'after_program', 'comparison': 'ordered_expanded_Program',
            'before_preserved_by_reference': True, 'constraint_checks': checks})
    payload = {'definition': candidate['definition'], 'holes': holes,
               'constraints': candidate['learned_constraints'], 'selector': candidate['scope'],
               'bindings': candidate['source_bindings'], 'exceptions': candidate['counterexample_refs']}
    original_bytes = len(_json(_unique(source['program'].to_dict() for source in sources)).encode('utf-8'))
    new_bytes = len(_json(payload).encode('utf-8'))
    candidate['encoding_cost'] = {'contract': 'canonical_json_utf8_bytes_v1',
        'scope': 'structural_encoding_comparison_original_sources_remain',
        'original_bytes': original_bytes, 'candidate_bytes': new_bytes,
        'new_definition_bytes': len(_json(candidate['definition']).encode('utf-8')),
        'gain_bytes': original_bytes - new_bytes, 'formal_mdl': False}
    return candidate


def _arguments(program):
    return {program.name} if program.op == 'arg' else set().union(
        *(_arguments(child) for child in program.children))


def _substitute(program, bindings):
    if program.op == 'arg':
        return bindings[program.name]
    return Program(program.op, program.value, program.name, program.lag,
        tuple(_substitute(child, bindings) for child in program.children), program.definition)


def _discrepancy(program, example, budget):
    _spend(budget)
    row = example['observation']
    try:
        value = program.predict(Observation(row['features'], row['target'], tuple(row.get('history', []))))
    except ValueError:
        return {'status': 'prediction_unavailable', 'failed': True}
    error = abs(value - row['target'])
    return {'status': 'computed', 'absolute_error': error, 'prediction': value,
            'target': row['target'], 'squared_error': error ** 2,
            'failed': error > example.get('tolerance', 1e-8)}


def _revision_hypotheses(body, holes, after, bindings, parent_holes, budget):
    constraints = _learn_constraints(body, holes, after, bindings, budget)
    result = [{'id': 'observed_fixed', 'constraints': constraints, 'assignments': [],
               'status': 'empirical_candidate', 'scope': 'retained_sources_only'}]
    for constraint in constraints:
        if constraint['op'] != 'equal':
            continue
        left, right = constraint['left'], constraint['right']
        if right['object'] == 'pattern':
            left, right = right, left
        if (left['object'] != 'pattern' or left['property'] != 'lag' or right['object'] != 'hole'
                or right['name'] not in parent_holes):
            continue
        alternative = [item for item in constraints if not (
            item['op'] == 'fixed' and item['left'] == right)]
        result.append({'id': f'observed_relation_{len(result)}', 'constraints': alternative,
            'assignments': [{'left': deepcopy(left), 'right': deepcopy(right)}],
            'status': 'empirical_candidate', 'scope': 'retained_sources_only'})
    return result


def _revision_change_kind(before, after, record, budget):
    """Separate observed structure edits from numerical family refitting."""
    result = {'status': 'known', 'before_ref': deepcopy(record.get('before_ref')),
              'after_ref': deepcopy(record.get('after_ref'))}
    if _ordered_skeleton(before, budget, canonical_roles=False) == _ordered_skeleton(
            after, budget, canonical_roles=False):
        return {**result, 'kind': 'parameter_refit'}

    def operations(node):
        _spend(budget)
        token = (node.op, node.name, node.lag, None if node.op == 'const' else node.value)
        return Counter([token]) + sum((operations(child) for child in node.children), Counter())

    if operations(before) != operations(after):
        return {**result, 'kind': 'structural_revision'}
    from .parameter_roles import parameter_family_correspondence

    correspondence = parameter_family_correspondence(before, after, record['role_signature'],
                                                      record['output_unit'], budget)
    if correspondence.get('reason') == 'parameter_role_budget_exhausted':
        raise _Incomplete
    return {**result, 'status': correspondence['status'],
            'kind': 'parameter_rebinding' if correspondence['status'] == 'known' else 'UNKNOWN',
            'parameter_role_correspondence': correspondence}


def _compile_revision(sources, identifier, budget):
    records = [source['record'] for source in sources]
    before = [_prepare(record, _program(record['before_program']).expanded(), budget, side='before')
              for record in records]
    body, holes, bindings, paths, pattern = _anti_unify(sources, budget, before_sources=before)
    parent_holes = _arguments(pattern)
    changes = [_revision_change_kind(old['program'], source['program'], source['record'], budget)
               for old, source in zip(before, sources)]
    structural = (None if any(item['status'] == 'UNKNOWN' for item in changes)
                  else any(item['kind'] == 'structural_revision' for item in changes))
    definition = Definition(identifier, tuple(hole['name'] for hole in holes), body)
    triggers = []
    for source, old in zip(sources, before):
        for example in source['record'].get('counterexamples', []):
            if not example.get('source_ref') or 'observation' not in example:
                continue
            prior = _discrepancy(old['program'], example, budget)
            revised = _discrepancy(source['program'], example, budget)
            triggers.append({'source_ref': deepcopy(example['source_ref']),
                'source_id': source['record']['source_id'], 'before': prior, 'after': revised,
                'tolerance': example.get('tolerance', 1e-8),
                **({'criterion': deepcopy(example['criterion'])} if 'criterion' in example else {})})
    candidate = {'id': identifier, 'kind': 'program_revision_relation',
        'purpose': 'structural_revision' if structural else 'parameter_proposal',
        'structural_change': structural,
        'source_change_kind': ('UNKNOWN' if structural is None else 'structural_revision' if structural else
            'parameter_rebinding' if any(item['kind'] == 'parameter_rebinding' for item in changes)
            else 'parameter_refit'),
        'source_change_diagnosis': changes,
        'before_pattern': pattern.to_dict(), 'definition': definition.to_dict(), 'holes': holes,
        'parent_holes': sorted(parent_holes), 'learned_constraints': [],
        'constraint_hypotheses': _revision_hypotheses(body, holes, sources, bindings, parent_holes, budget),
        'trigger_checks': triggers, 'trigger_kind': 'measured_parent_discrepancy',
        'source_ids': _unique(record['source_id'] for record in records),
        'source_contexts': _unique(record['context_ref'] for record in records),
        'source_outcomes': [_source_references(record) for record in records],
        'source_bindings': [], 'source_reconstruction': [],
        'scope': {'context_family_ref': deepcopy(records[0].get('context_family_ref', records[0]['context_ref']))},
        'output_unit': records[0]['output_unit'],
        'fixed_role_signature': {name: deepcopy(records[0]['role_signature'][name])
                                for name, _ in body.observed_variables()},
        'status': 'EXECUTABLE_CANDIDATE', 'frozen': True, 'active_library': False,
        'construction_only': True, 'constraint_scope': 'optional_proposer_only',
        'transfer_status': 'not_established', 'support_updated': False, 'global_support': False,
        'independent_confirmation': False, 'scientific_support_added': False,
        'counterexample_refs': _unique([reference for record in records for reference in record.get('failure_refs', [])])}
    for index, (source, old, substitution) in enumerate(zip(sources, before, bindings)):
        _spend(budget, old['program'].size + source['program'].size)
        if _substitute(pattern, substitution) != old['program'] or _substitute(body, substitution) != source['program']:
            raise ValueError('joint revision source reconstruction failed')
        candidate['source_bindings'].append({**_source_references(source['record']), 'bindings_refs': {
            name: {'source_ref': deepcopy(records[index][paths[index][name]['side'] + '_ref']),
                   'program_path': paths[index][name]['path']} for name in definition.arguments}})
        candidate['source_reconstruction'].append({**_source_references(source['record']),
            'exact': True, 'before_exact': True, 'after_exact': True,
            'comparison': 'ordered_expanded_Program'})
    from .parent_reference_relations import learn_parent_references

    try:
        relations = learn_parent_references(sources, before, paths, parent_holes, budget, identifier)
        if relations:
            candidate['parent_reference_relations'] = relations
    except _Incomplete:
        # The already reconstructed positional object remains usable even when
        # the optional cross-position induction could not finish.
        candidate['parent_reference_search'] = {'status': 'UNKNOWN', 'reason': 'budget_exhausted'}
    except _MissingContract as error:
        candidate['parent_reference_search'] = {'status': 'UNKNOWN', 'reason': str(error)}
    return candidate


def _revision_contract(candidate):
    from .parent_reference_relations import parent_reference_contract

    return {key: candidate.get(key) for key in ('before_pattern', 'holes', 'parent_holes',
        'constraint_hypotheses', 'learned_constraints', 'scope', 'output_unit', 'fixed_role_signature')} | {
        'definition': {key: candidate['definition'].get(key) for key in ('body', 'arguments')},
        'parent_reference_relations': parent_reference_contract(candidate)}


def _parameter_sources(candidate, sources, budget):
    pattern = _program(candidate['before_pattern'])
    body = Definition.from_dict(candidate['definition']).body
    records = []
    for source in sources:
        record = source['record']
        if (record['output_unit'] != candidate['output_unit'] or record.get(
                'context_family_ref', record['context_ref']) != candidate['scope']['context_family_ref']):
            continue
        try:
            bindings, before_paths = _match_parent(pattern, _program(record['before_program']).expanded(),
                                                   candidate['holes'], budget)
            _, after_paths = _match_parent(body, source['program'], candidate['holes'], budget,
                                           bindings=bindings)
        except _MissingContract:
            continue
        records.append({**_source_references(record), 'bindings': bindings,
            '_before_program': _program(record['before_program']).expanded(),
            '_after_program': source['program'],
            '_before_binding_paths': before_paths, '_after_binding_paths': after_paths,
            '_counterexamples': record.get('counterexamples', []),
            '_roles': record['role_signature'], '_output_unit': record['output_unit']})
    return records


def _source_evidence_contexts(records, candidates, budget, retained):
    from .revision_evidence import evidence_context

    for record in records:
        cached = next((item for item in retained if item['before_ref'] == record['before_ref']
            and item['example_refs'] == [example.get('source_ref')
                                       for example in record['_counterexamples']]), None)
        if cached is None:
            checks = []
            for example in record['_counterexamples']:
                measured = next((check['before'] for candidate in candidates
                    if any(item.get('before_ref') == record['before_ref']
                        and item.get('source_id') == record['source_id']
                        for item in candidate['source_outcomes'])
                    for check in candidate['trigger_checks']
                    if check['source_id'] == record['source_id']
                        and check['source_ref'] == example.get('source_ref')), None)
                checks.append(measured)
            context = evidence_context(record['_before_program'], record['_counterexamples'],
                record['_roles'], record['_output_unit'], checks, budget)
            retained.append({'before_ref': record['before_ref'],
                'example_refs': [item.get('source_ref') for item in record['_counterexamples']],
                'context': context})
        else:
            context = cached['context']
        record['evidence_context'] = deepcopy(context)


def _learn_parameter_updates(candidates, sources, budget, allocation):
    """Search each distinct joint construction once, after retaining old work."""
    from .revision_parameter_relations import (learn_parameter_relations,
                                              learn_joint_parameter_relations)
    from .synthesis import OPERATIONS

    counts = {source['record']['source_id']: 0 for source in sources}
    for source in sources:
        counts[source['record']['source_id']] += 1
    if sum(count >= 2 for count in counts.values()) < 2:
        return [], candidates
    contract = {'operations': list(OPERATIONS), 'max_structures': 16,
                'risk_quantile': .5, 'tolerance': 1e-10,
                **deepcopy(allocation.get('parameter_relations', {}))}
    joint_maximum = contract.pop('joint_max_structures', contract['max_structures'])
    groups = []
    for candidate in candidates:
        if candidate.get('kind') != 'program_revision_relation':
            continue
        key = _revision_contract(candidate)
        group = next((item for item in groups if item['contract'] == key), None)
        if group is None:
            group = {'contract': key, 'candidates': []}
            groups.append(group)
        group['candidates'].append(candidate)
    searches = []
    for index, group in enumerate(groups):
        candidate = group['candidates'][0]
        binding_started = budget.spent
        try:
            records = _parameter_sources(candidate, sources, budget)
        except _Incomplete:
            group['records'] = None
            reason = 'source_binding_budget_exhausted'
            searches.append({'candidate_id': candidate['id'], 'source_record_count': None,
                'input_scope': 'parent_parameters', 'status': 'UNKNOWN', 'reason': reason,
                'accepted': [], 'attempts': [], 'excluded': [], 'source_records': [],
                'next_actions': [], 'target_conflicts': {}, 'actual_compute_cost': 0,
                'source_binding_compute_cost': budget.spent - binding_started,
                'budget_exhausted': True, 'scope': 'retained_revision_parameter_metadata_only',
                'support_updated': False, 'scientific_support': False,
                'independent_confirmation': False})
            for item in group['candidates']:
                item['parameter_relations'] = []
                item['parameter_relation_search'] = {
                    'same_constructive_search_candidate_id': candidate['id'],
                    'status': 'UNKNOWN', 'reason': reason, 'source_record_count': None,
                    'accepted_count': 0, 'actual_compute_cost': 0}
            continue
        group['records'] = records
        local = SearchBudget(budget.remaining // (len(groups) - index))
        result = learn_parameter_relations(candidate, records, local, search_contract=contract)
        _spend(budget, local.spent)
        search = {'candidate_id': candidate['id'], 'source_record_count': len(records),
                  'input_scope': 'parent_parameters', **result}
        searches.append(search)
        for item in group['candidates']:
            item['parameter_relations'] = deepcopy(result['accepted'])
            item['parameter_relation_search'] = {
                'same_constructive_search_candidate_id': candidate['id'],
                'status': result['status'], 'accepted_count': len(result['accepted']),
                'actual_compute_cost': result['actual_compute_cost']}
    # The complete old parameter frontier runs first. Additional measured
    # coordinates cannot displace its hypotheses by reordering the grammar.
    retained_contexts = []
    for index, group in enumerate(groups):
        candidate, records = group['candidates'][0], group['records']
        if records is None:
            continue
        try:
            _source_evidence_contexts(records, candidates, budget, retained_contexts)
        except _Incomplete:
            searches.append({'candidate_id': candidate['id'], 'input_scope': 'counterexample_conditions',
                'source_record_count': len(records), 'status': 'UNKNOWN', 'accepted': [],
                'actual_compute_cost': 0, 'budget_exhausted': True,
                'reason': 'measured_condition_readout_budget_exhausted'})
            break
        if not any(record.get('evidence_context', {}).get('status') == 'ready' for record in records):
            continue
        local = SearchBudget(budget.remaining // (len(groups) - index))
        result = learn_parameter_relations(candidate, records, local, search_contract=contract)
        _spend(budget, local.spent)
        search = {'candidate_id': candidate['id'], 'source_record_count': len(records),
                  'input_scope': 'counterexample_conditions', **result}
        searches.append(search)
        for item in group['candidates']:
            item['parameter_relations'].extend(deepcopy(result['accepted']))
            item['parameter_relation_search']['condition_status'] = result['status']
            item['parameter_relation_search']['condition_accepted_count'] = len(result['accepted'])
    # Sparse views supplement the complete two older frontiers. They use the
    # same measured inputs, fitter and held revisions, with the remaining cap.
    for index, group in enumerate(groups):
        candidate, records = group['candidates'][0], group['records']
        if records is None:
            continue
        if not any(record.get('evidence_context', {}).get('status') == 'ready' for record in records):
            continue
        local = SearchBudget(budget.remaining // (len(groups) - index))
        result = learn_joint_parameter_relations(candidate, records, local,
            search_contract={**contract, 'max_structures': joint_maximum})
        _spend(budget, local.spent)
        searches.append({'candidate_id': candidate['id'], 'source_record_count': len(records), **result})
        for item in group['candidates']:
            item['parameter_relations'].extend(deepcopy(result['accepted']))
            item['parameter_relation_search']['joint_status'] = result['status']
            item['parameter_relation_search']['joint_accepted_count'] = len(result['accepted'])
    for candidate in candidates:
        if candidate.get('purpose') != 'parameter_proposal':
            continue
        open_names = _arguments(Definition.from_dict(candidate['definition']).body) - set(candidate['parent_holes'])
        mapped = {item['output_hole'] for item in candidate.get('parameter_relations', [])
                  if item.get('purpose') == 'exact_edit_map'}
        if open_names and open_names <= mapped:
            candidate['purpose'] = 'exact_edit_map'
    return searches, candidates


def consolidate_revision_relations(episodes, allocation=None, candidate_id='relation', *, object_type=None):
    """Unify bounded actual revisions for the declared construction objective.

    The default retains both after contexts and joint revisions. An explicit
    program_revision_relation request spends its allowance on joint before/after
    construction, without first constructing an unrelated after-only object.
    """
    if object_type not in (None, 'program_revision_relation'):
        raise ValueError('unsupported revision construction object_type')
    allocation = allocation or {}
    budget = _budget(allocation)
    limit = allocation.get('max_groups', 16)
    if type(limit) is not int or limit < 0:
        raise ValueError('max_groups must be a nonnegative integer')
    sources, candidates, excluded, unknown = [], [], [], False
    contract_recovery = []
    try:
        for record in episodes:
            _spend(budget)
            if not record.get('source_id') or not record.get('after_ref'):
                excluded.append({'episode_id': record.get('episode_id'), 'status': 'UNKNOWN',
                                 'reason': 'actual_revision_source_and_after_ref_required'})
                continue
            if object_type == 'program_revision_relation' and not record.get('before_ref'):
                excluded.append({'episode_id': record.get('episode_id'), 'status': 'UNKNOWN',
                                 'reason': 'actual_before_ref_required'})
                continue
            if record.get('before_program') is None or record.get('after_program') is None:
                excluded.append({'episode_id': record.get('episode_id'), 'status': 'UNKNOWN',
                                 'reason': 'actual_before_and_after_Programs_required'})
                continue
            try:
                before, after = _program(record['before_program']).expanded(), _program(record['after_program']).expanded()
                old = _prepare(record, before, budget, side='before')
                prepared = _prepare(record, after, budget, side='after')
                contract_recovery.append({'episode_id': record['episode_id'],
                    'before': old['contract_recovery'], 'after': prepared['contract_recovery']})
                sources.append(prepared)
            except _MissingContract as error:
                contract_recovery.append({'episode_id': record.get('episode_id'),
                    'status': 'UNKNOWN', 'reason': str(error),
                    'before_ref': deepcopy(record.get('before_ref')),
                    'after_ref': deepcopy(record.get('after_ref')),
                    'sources': [deepcopy(record[key]['source_ref']) for key in
                        ('_before_contract_source', '_after_contract_source')
                        if record.get(key, {}).get('source_ref')]})
                excluded.append({'episode_id': record.get('episode_id'), 'status': 'UNKNOWN', 'reason': str(error)})
        pairs = combinations(sources, 2)
        if 'source_pairs' in allocation:
            by_episode = {source['record']['episode_id']: source for source in sources}
            pairs = []
            for pair in allocation['source_pairs']:
                if len(pair) != 2 or not all(isinstance(identifier, str) for identifier in pair):
                    raise ValueError('source_pairs must contain pairs of episode IDs')
                if any(identifier not in by_episode for identifier in pair):
                    excluded.append({'episode_ids': list(pair), 'status': 'UNKNOWN',
                                     'reason': 'requested_pair_source_unavailable'})
                    continue
                pairs.append(tuple(by_episode[identifier] for identifier in pair))
        groups = 0
        for left, right in pairs:
            _spend(budget)
            lrecord, rrecord = left['record'], right['record']
            if lrecord['source_id'] == rrecord['source_id']:
                excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                                 'status': 'UNKNOWN', 'reason': 'distinct_episode_sources_required'})
                continue
            if lrecord.get('output_unit') != rrecord.get('output_unit') or lrecord.get(
                    'context_family_ref', lrecord['context_ref']) != rrecord.get('context_family_ref', rrecord['context_ref']):
                excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                                 'status': 'UNKNOWN', 'reason': 'incompatible_output_unit_or_context'})
                continue
            if groups >= limit:
                raise _Incomplete
            groups += 1
            if object_type is None:
                try:
                    item = _compile([left, right], f'{candidate_id}:{len(candidates)}', budget)
                except _MissingContract as error:
                    excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                        'status': 'UNKNOWN', 'reason': str(error),
                        **({'contract_conflict': error.contract_conflict}
                           if hasattr(error, 'contract_conflict') else {})})
                    continue
                if item is None:
                    excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                        'status': 'existing_family', 'reason': 'existing_ordered_parameter_family'})
                else:
                    equivalent = next((old for old in candidates if old['definition']['body'] == item['definition']['body']
                        and old['holes'] == item['holes'] and old['learned_constraints'] == item['learned_constraints']
                        and old['source_bindings'] == item['source_bindings']), None)
                    if equivalent is None:
                        candidates.append(item)
            if all(source['record'].get('before_ref') for source in (left, right)):
                try:
                    joint = _compile_revision([left, right], f'{candidate_id}:revision:{groups - 1}', budget)
                except _MissingContract as error:
                    excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                        'status': 'UNKNOWN', 'reason': str(error),
                        **({'contract_conflict': error.contract_conflict}
                           if hasattr(error, 'contract_conflict') else {})})
                    continue
                if joint is not None:
                    candidates.append(joint)
                    unknown = unknown or joint.get('parent_reference_search', {}).get('reason') == 'budget_exhausted'
                else:
                    excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                        'status': 'UNKNOWN' if object_type else 'not_applicable',
                        'reason': 'no_shared_parent_subprogram_in_revision'})
    except _Incomplete:
        unknown = True
    parameter_searches, candidates = _learn_parameter_updates(candidates, sources, budget, allocation)
    unknown = unknown or any(search.get('reason') == 'source_binding_budget_exhausted'
                             for search in parameter_searches)
    if len(sources) < 2 and not excluded:
        excluded.append({'status': 'UNKNOWN', 'reason': 'two_actual_revision_sources_required'})
    return {'status': 'UNKNOWN' if unknown or any(item['status'] == 'UNKNOWN' for item in excluded)
            or any(item.get('parent_reference_search', {}).get('status') == 'UNKNOWN' for item in candidates)
            else 'completed', 'candidates': candidates,
        'excluded': excluded, 'author_work': _unique(
            {**action, 'type': 'AuthorWorkRequest', 'candidate_id': search['candidate_id']}
            for search in parameter_searches for action in search.get('next_actions', [])),
        'actual_compute_cost': budget.spent,
        'parameter_relation_searches': parameter_searches,
        'subtree_contract_recovery': contract_recovery,
        'budget_exhausted': unknown, 'support_updated': False, 'new_observations': 0, 'llm_calls': 0}


def _component(component, budget):
    literal = _program(component['program'])
    program = literal.expanded()
    prepared = _prepare(component, program, budget)
    used = {name for name, _ in program.observed_variables()}
    times = _unique(component['role_signature'][name]['time'] for name in sorted(used))
    lexical = component.get('parent_reference', {}).get('parameter_coordinates') == 'definition_arguments'
    return {**component, '_program': literal if lexical else program, '_times': times, '_prepared': prepared}


def _bound_proposal(candidate, components, budget):
    holes = candidate['holes']
    contexts = _unique(component['context_ref'] for component in components.values())
    if len(contexts) != 1:
        raise _MissingContract('component_contexts_must_match')
    bindings = {}
    signatures = deepcopy(candidate.get('fixed_role_signature', {}))
    for name, contract in signatures.items():
        supplied = [component['role_signature'][name] for component in components.values()
                    if name in component['role_signature']]
        if not supplied or any(item != contract for item in supplied):
            raise _MissingContract('fixed_context_feature_binding_required')
    for hole in holes:
        component = components[hole['name']]
        if component['output_unit'] != hole['output_unit']:
            raise _MissingContract('component_output_unit_mismatch')
        if component['_times'] and component['_times'] != hole['time_contract']:
            raise _MissingContract('component_time_contract_mismatch')
        for name, _ in component['_program'].observed_variables():
            contract = component['role_signature'][name]
            if name in signatures and signatures[name] != contract:
                raise _MissingContract('incompatible_shared_feature_contract')
            signatures[name] = deepcopy(contract)
        bindings[hole['name']] = component['_program']
    program = _instantiate(candidate, bindings, budget)
    occurrences, origins = _occurrence_contracts(
        Definition.from_dict(candidate['definition']).body.expanded(), program, holes, budget)
    checks = _constraint_checks(candidate, bindings, program, budget)
    if any(check['status'] == 'violated' for check in checks):
        return None, checks
    return {'program': program.to_dict(), 'bindings': {name: {
            'component_id': component['id'], 'version': component.get('version', 1),
            'source_ref': deepcopy(component.get('source_ref'))} for name, component in components.items()},
        'constraint_checks': checks, 'role_signature': signatures,
        'subtree_contracts': occurrences, 'subtree_contract_sources': origins,
        'output_unit': candidate['output_unit'], 'context_ref': contexts[0],
        'support_updated': False, 'independent_confirmation': False}, checks


def instantiate_relation(candidate, bindings, allocation=None):
    """Check one explicit lexical binding without fitting or acquiring data."""
    budget = _budget(allocation)
    result = {'status': 'completed', 'proposals': [], 'excluded': [], 'budget_exhausted': False,
              'affects_existing_candidates': False, 'support_updated': False}
    try:
        if set(bindings) != {hole['name'] for hole in candidate['holes']}:
            raise _MissingContract('complete_hole_binding_required')
        prepared = {name: _component(component, budget) for name, component in bindings.items()}
        proposal, checks = _bound_proposal(candidate, prepared, budget)
        if proposal is None:
            result.update(status='not_applicable', excluded=[{'reason': 'learned_property_constraint_violated',
                                                           'constraint_checks': checks}])
        else:
            result['proposals'].append(proposal)
    except _MissingContract as error:
        result.update(status='UNKNOWN', excluded=[{'reason': str(error)}])
    except _Incomplete:
        result.update(status='UNKNOWN', budget_exhausted=True)
    result['actual_compute_cost'] = budget.spent
    return result


def _unary_requirements(candidate, budget):
    """Propagate only the object's already learned fixed/equality properties."""
    body = Definition.from_dict(candidate['definition']).body
    known, references = {}, {}
    equalities = []
    for constraint in candidate['learned_constraints']:
        _spend(budget)
        sides = [constraint['left']] + ([constraint['right']] if constraint['op'] == 'equal' else [])
        for reference in sides:
            key = _json(reference)
            references[key] = reference
            if reference['object'] == 'pattern' and key not in known:
                known[key] = _property(reference, body, {}, None, budget)
        if constraint['op'] == 'fixed':
            known[_json(constraint['left'])] = constraint['value']
        else:
            equalities.append((_json(constraint['left']), _json(constraint['right'])))
    for _ in range(len(equalities) + 1):
        changed = False
        for left, right in equalities:
            _spend(budget)
            if left in known and right not in known:
                known[right], changed = known[left], True
            elif right in known and left not in known:
                known[left], changed = known[right], True
        if not changed:
            break
    return [(references[key], value) for key, value in known.items()
            if references[key]['object'] == 'hole']


def _diagonal_indices(lengths, cursor=None):
    cursor = cursor or {}
    if not lengths or any(not length for length in lengths):
        return
    def layer(total, dimension=0):
        if dimension == len(lengths):
            if total == 0:
                yield ()
            return
        for index in range(min(total, lengths[dimension] - 1) + 1):
            for tail in layer(total - index, dimension + 1):
                yield (index, *tail)
    for depth in range(cursor.get('layer', 0), sum(length - 1 for length in lengths) + 1):
        offset = cursor.get('offset', 0) if depth == cursor.get('layer', 0) else 0
        for index, indices in enumerate(layer(depth)):
            if index >= offset:
                yield indices, {'layer': depth, 'offset': index + 1}


def propose_relation(candidate, components, allocation=None, *, cursor=None):
    """Enumerate only the supplied finite, typed existing component Programs."""
    budget = _budget(allocation)
    proposals, excluded, prepared, incomplete = [], [], [], False
    next_cursor = deepcopy(cursor or {'layer': 0, 'offset': 0})
    try:
        for component in components:
            try:
                prepared.append(_component(component, budget))
            except _MissingContract as error:
                excluded.append({'component_id': component.get('id'), 'reason': str(error), 'status': 'UNKNOWN'})
        requirements, pools = _unary_requirements(candidate, budget), []
        for hole in candidate['holes']:
            pool, rejected = [], 0
            checks = [(reference['property'], value) for reference, value in requirements
                      if reference['name'] == hole['name']]
            for component in prepared:
                if component['output_unit'] != hole['output_unit'] or (
                        component['_times'] and component['_times'] != hole['time_contract']):
                    continue
                accepted = True
                for attribute, expected in checks:
                    _spend(budget)
                    if getattr(component['_program'], attribute) != expected:
                        accepted = False
                        break
                if accepted:
                    pool.append(component)
                else:
                    rejected += 1
            pools.append(pool)
            if rejected:
                excluded.append({'hole': hole['name'], 'reason': 'learned_property_constraint_violated',
                                 'rejected_component_count': rejected, 'scope': 'optional_proposer_only'})
        for indices, position in _diagonal_indices([len(pool) for pool in pools], cursor):
            _spend(budget)
            items = [pool[index] for pool, index in zip(pools, indices)]
            bindings = dict(zip((hole['name'] for hole in candidate['holes']), items))
            try:
                proposal, checks = _bound_proposal(candidate, bindings, budget)
            except _MissingContract:
                next_cursor = position
                continue
            if proposal is not None:
                if not any(old['program'] == proposal['program'] for old in proposals):
                    proposal['search_position'] = deepcopy(position)
                    proposals.append(proposal)
            else:
                excluded.append({'bindings': {name: item['id'] for name, item in bindings.items()},
                    'reason': 'learned_property_constraint_violated', 'constraint_checks': checks})
            next_cursor = position
    except _Incomplete:
        incomplete = True
    return {'status': 'UNKNOWN' if incomplete or any(item.get('status') == 'UNKNOWN' for item in excluded)
            else 'completed', 'proposals': proposals, 'excluded': excluded,
        'actual_compute_cost': budget.spent, 'budget_exhausted': incomplete,
        'cursor': next_cursor, 'search_completed': not incomplete,
        **({'minimum_compute_for_progress': budget.limit +
            budget.last_denied_request['additional_compute']}
           if incomplete and budget.last_denied_request else {}),
        'affects_existing_candidates': False, 'support_updated': False, 'new_observations': 0, 'llm_calls': 0}


def _match_parent(pattern, parent, holes, budget, bindings=None, paths=None, path=()):
    bindings = {} if bindings is None else bindings
    paths = {} if paths is None else paths
    _spend(budget)
    if pattern.op == 'arg':
        if pattern.name in bindings and bindings[pattern.name] != parent:
            raise _MissingContract('parent_repeated_subtree_mismatch')
        bindings[pattern.name], paths[pattern.name] = parent, list(path)
    elif (pattern.op, pattern.value, pattern.name, pattern.lag, len(pattern.children)) != (
            parent.op, parent.value, parent.name, parent.lag, len(parent.children)):
        raise _MissingContract('before_pattern_does_not_match_parent')
    else:
        for index, (left, right) in enumerate(zip(pattern.children, parent.children)):
            _match_parent(left, right, holes, budget, bindings, paths, (*path, index))
    return bindings, paths


def _assign_properties(body, assignments, bindings, budget):
    changes = {}
    for assignment in assignments:
        _spend(budget)
        reference = assignment['right']
        changes[tuple(assignment['left']['path'])] = getattr(bindings[reference['name']], reference['property'])
    def visit(node, path=()):
        return Program(node.op, node.value, node.name, changes.get(path, node.lag),
            tuple(visit(child, (*path, index)) for index, child in enumerate(node.children)), node.definition)
    return visit(body)


def _check_generated_units(program, expected, roles, known, budget):
    """Check generated syntax using the existing scalar unit rules and typed seeds."""
    _spend(budget)
    if program in known:
        unit = known[program]
    elif program.op in {'feature', 'lag'}:
        unit = roles[program.name]['unit']
    elif program.op == 'const':
        unit = expected
    elif program.op in _SAME_UNIT or program.op == 'accumulate':
        units = [_check_generated_units(child, expected, roles, known, budget) for child in program.children]
        unit = next((value for value in units if value is not None), expected)
        if any(value is not None and value != unit for value in units):
            raise _MissingContract('incompatible_generated_unit')
    elif program.op == 'mul':
        units = [_check_generated_units(child, 'dimensionless' if child.op == 'const' else None,
                                       roles, known, budget) for child in program.children]
        if units[0] == 'dimensionless':
            unit = units[1]
        elif units[1] == 'dimensionless':
            unit = units[0]
        else:
            raise _MissingContract('generated_product_unit_binding_required')
    elif program.op == 'if':
        _check_generated_units(program.children[0], 'dimensionless', roles, known, budget)
        units = [_check_generated_units(child, expected, roles, known, budget) for child in program.children[1:]]
        unit = units[0]
        if units[0] != units[1]:
            raise _MissingContract('incompatible_generated_unit')
    elif program.op in _PREDICATES:
        if program.op in {'ge', 'lt'}:
            units = [_check_generated_units(child, None, roles, known, budget)
                     if child.op != 'const' else None for child in program.children]
            operand_unit = next((unit for unit in units if unit is not None), 'dimensionless')
            units = [unit if child.op != 'const' else _check_generated_units(
                child, operand_unit, roles, known, budget) for child, unit in zip(program.children, units)]
            concrete = [value for value in units if value is not None]
            if concrete and any(value != concrete[0] for value in concrete):
                raise _MissingContract('incompatible_generated_unit')
        else:
            for child in program.children:
                _check_generated_units(child, 'dimensionless', roles, known, budget)
        unit = 'dimensionless'
    else:
        raise _MissingContract('generated_structure_unit_binding_required')
    if expected is not None and unit != expected:
        raise _MissingContract('incompatible_generated_unit')
    return unit


def _revision_fitting_pools(candidate, parent, components, holes, rows, fitting, budget, state,
                            *, parameter_components=None, parent_components=None, excluded=None):
    """Use the original grammar's unfitted structures, never response labels for holes."""
    from .synthesis import OPERATIONS, iter_synthesis_structures

    operations = fitting.get('operations')
    quantile = fitting.get('risk_quantile')
    history = fitting.get('max_history')
    maximum = fitting.get('max_structures')
    if (not isinstance(operations, list) or not operations or set(operations) - set(OPERATIONS)
            or type(quantile) not in (int, float) or not 0 < quantile < 1
            or type(history) is not int or history < 0
            or type(maximum) is not int or maximum < 1):
        raise _MissingContract('finite_revision_fitting_contract_required')
    if not holes:
        return []
    # A Program can remain an untyped prediction/structural seed. In particular,
    # the retained frozen_keep/remove seeds do not declare units or roles.
    # Only explicitly declared components may supply hole values or known units.
    program_components = [item for item in components if item.get('program')]
    typed_components = []
    for item in program_components:
        try:
            _declared_contract_header(item)
        except _MissingContract as error:
            record = {'component_id': item.get('id'), 'version': item.get('version', 1),
                'reason': str(error), 'status': 'UNKNOWN', 'scope': 'typed_revision_pool_only'}
            if item.get('source_ref'):
                record['source_ref'] = deepcopy(item['source_ref'])
            saved = state.setdefault('excluded_components', [])
            if record not in saved:
                saved.append(record)
            if excluded is not None:
                excluded.append(deepcopy(record))
            continue
        typed_components.append(item)
    structures = state.setdefault('structures', [])
    grammar_cursor = state.setdefault('grammar_cursor', {})
    if len(structures) < maximum and grammar_cursor.get('status') != 'completed':
        stream = iter_synthesis_structures(rows, operations=operations, budget=budget,
            max_history=history, cursor=grammar_cursor,
            seed_programs=tuple(_program(item['program']).expanded() for item in program_components))
        for item in stream:
            structures.append({**{key: deepcopy(item[key]) for key in
                ('operation', 'rationale', 'proposal_index', 'source_ref')},
                'program': item['program'].to_dict(),
                'parameter_paths': [list(path) for path in item['parameter_paths']]})
            if len(structures) == maximum:
                break
        if grammar_cursor.get('budget_exhausted'):
            raise _Incomplete
    pools = []
    known_units = {_program(item['program']).expanded(): item['output_unit'] for item in typed_components}
    known_units[_program(parent['program']).expanded()] = parent['output_unit']
    for hole in holes:
        pool = [item for item in typed_components if item.get('context_ref') == parent['context_ref']
                and item.get('output_unit') == hole['output_unit']]
        for index, item in enumerate(structures):
            program = _program(item['program']).expanded()
            declaration = {key: deepcopy(parent[key]) for key in ('role_signature', 'context_ref')}
            declaration.update(id=f'grammar:{item["proposal_index"]}:{hole["name"]}',
                output_unit=hole['output_unit'], program=program.to_dict(),
                source_ref=deepcopy(item['source_ref']), fitting_parameter_paths=item['parameter_paths'])
            try:
                _check_generated_units(program, hole['output_unit'], parent['role_signature'], known_units, budget)
                prepared = _component(declaration, budget)
            except _MissingContract as error:
                state.setdefault('excluded_structures', []).append({'hole': hole['name'],
                    'proposal_index': item['proposal_index'], 'reason': str(error)})
                continue
            if prepared['_prepared']['contracts'][()]['output_unit'] == hole['output_unit']:
                pool.append(declaration)
        pool.extend((parameter_components or {}).get(hole['name'], []))
        # Preserve the ordinary first choice and every old construction. The
        # learned context gets a bounded early opportunity in its observed
        # role, rather than sitting beyond the whole archive's finite prefix.
        pool[1:1] = (parent_components or {}).get(hole['name'], [])
        if not pool:
            raise _MissingContract('typed_open_hole_components_or_fitting_contract_required')
        pools.append(pool)
    return pools


def _revision_fitting_template(candidate, prepared, fixed_names):
    from .abstractions import AbstractTemplate

    definition = Definition.from_dict(candidate['definition'])
    call = Program('call', name=definition.definition_id, definition=definition,
        children=tuple(prepared[name]['_program'] for name in definition.arguments))
    paths = tuple((index, *path) for index, name in enumerate(definition.arguments)
                  if name not in fixed_names for path in prepared[name].get('fitting_parameter_paths',
                      prepared[name].get('parameter_paths', ())))
    template = AbstractTemplate(f'{candidate["id"]}:open_parameters', candidate['id'], call, (), (), paths)
    return template


def _fit_revision_binding(candidate, prepared, fixed_names, rows, quantile, budget, *, template=None):
    """Free parameters live in call arguments; repeated holes share one argument."""
    from .abstractions import fit_parameter_slots

    template = template or _revision_fitting_template(candidate, prepared, fixed_names)
    report = {}
    programs = fit_parameter_slots(template, {}, rows, budget, quantile, report=report)
    return programs, template.parameter_paths, report


def _fit_scope_key(value, budget):
    _spend(budget)
    if isinstance(value, dict):
        return tuple((key, _fit_scope_key(item, budget)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_fit_scope_key(item, budget) for item in value)
    return value


def _fit_template_key(template, quantile, scope, budget):
    # Program's ordinary in-memory equality retains the lexical Definition and
    # repeated argument sharing. Before-only fixed arguments are not evaluated
    # by the after Definition and cannot change this fitting computation.
    def visit(node, names=None):
        _spend(budget)
        if names is not None and node.op == 'arg':
            names.add(node.name)
        for child in node.children:
            visit(child, names)
        if node.definition is not None:
            # A nested Definition's arguments have their own lexical scope.
            visit(node.definition.body)

    pattern = template.pattern
    if pattern.op == 'call':
        _spend(budget)
        used = set()
        visit(pattern.definition.body, used)
        free_arguments = {path[0] for path in template.parameter_paths if path}
        children = []
        for index, (name, child) in enumerate(zip(pattern.definition.arguments, pattern.children)):
            _spend(budget)
            if name in used or index in free_arguments:
                visit(child)
                children.append((index, child))
        # A newly learned proposer can share the same positional computation
        # with an older object. Its assigned outer name is not a new fit;
        # argument order/body retain the actual lexical sharing contract.
        identity = (pattern.op, pattern.value, pattern.lag, pattern.definition.arguments,
                    pattern.definition.body, tuple(children))
    else:
        visit(pattern)
        identity = pattern
    paths = _fit_scope_key(template.parameter_paths, budget)
    contract = _fit_scope_key({key: scope.get(key) for key in
        ('role_signature', 'output_unit', 'context_ref')}, budget)
    return identity, paths, quantile, contract


def _retained_fit_index(records, budget):
    """Index complete saved linear solves without inheriting their support."""
    indexed = {}
    methods = ['initial', 'ridge_lstsq', 'stable_lstsq']
    for record in records:
        _spend(budget)
        report, outputs = record.get('report', {}), record.get('outputs', [])
        linear = report.get('linear_diagnostics', {})
        if (not record.get('source_ref') or report.get('search_completed') is not True
                or report.get('candidate_methods') != methods
                or [item.get('method') for item in outputs] != methods
                or not all(item.get('source_ref') for item in outputs)
                or report.get('actual_fitted_rows') != len(record['rows'])
                or report.get('eligible_fitting_rows') != len(record['rows'])
                or linear.get('status') != 'computed'
                or linear.get('solver') != 'numpy.linalg.lstsq'
                or linear.get('ridge', {}).get('penalty') != 1e-9
                or linear.get('stable', {}).get('penalty') != 0.):
            continue
        template = record['template']
        count = len(template.parameter_paths)
        if (not all(len(item['parameters']) == count for item in outputs)
                or tuple(outputs[0]['parameters']) != template.parameters):
            continue
        key = _fit_template_key(template, record['risk_quantile'], record, budget)
        indexed.setdefault(key, []).append(record)
    return indexed


def _reuse_revision_fit(template, rows, quantile, scope, retained_fits, budget, *, index=None):
    """Restore exactly the old initial/ridge/stable candidate set, without evaluation.

    The initial lexical computation (including every free-slot initial value)
    must match; only unused fixed call arguments are irrelevant. Equal final
    expanded Programs alone would discard initial candidates or sharing.
    """
    index = _retained_fit_index(retained_fits, budget) if index is None else index
    key = _fit_template_key(template, quantile, scope, budget)
    for record in index.get(key, []):
        _spend(budget)
        if len(record['rows']) != len(rows):
            continue
        same_rows = True
        for original, current in zip(record['rows'], rows):
            # The ordinary fitting account charges candidate × row. Exact
            # Observation equality includes all features, target and history.
            _spend(budget)
            if original != current:
                same_rows = False
                break
        if not same_rows:
            continue
        programs, losses = [], {}
        for item in record['outputs']:
            # Instantiation changes only these same declared constant paths.
            _spend(budget, len(template.parameter_paths))
            program = template.instantiate({}, item['parameters'])
            programs.append(program)
            losses[program] = item['training_loss']
        report = {key: deepcopy(value) for key, value in record['report'].items() if key != 'losses'}
        report.update(losses=losses, actual_compute_cost=0,
            original_compute_cost=record['report'].get('actual_compute_cost'),
            fit_reuse_ref=deepcopy(record['source_ref']),
            fit_reuse_output_refs=[deepcopy(item['source_ref']) for item in record['outputs']])
        return programs, template.parameter_paths, report
    return None


def _parameter_components(candidate, parent, fixed, budget, *, context=None, excluded=None,
                          whole_after_fitting=False):
    """Apply learned metadata functions to actual matched parent constants."""
    from .parameter_roles import describe_bound_parameter_roles, role_compatible

    result = {}
    holes = {hole['name']: hole for hole in candidate['holes']}
    actual_inputs = None
    for index, relation in enumerate(candidate.get('parameter_relations', [])):
        purpose = relation.get('purpose', 'exact_edit_map')
        if purpose not in {'exact_edit_map', 'parameter_proposal'} or (
                purpose == 'parameter_proposal' and not whole_after_fitting):
            if excluded is not None:
                excluded.append({'status': 'UNKNOWN', 'parameter_relation': index,
                    'reason': 'whole_after_fitting_required_for_parameter_proposal' if
                        purpose == 'parameter_proposal' else 'unknown_parameter_relation_purpose'})
            continue
        if (relation.get('parameter_coordinate') != 'canonical_basis'
                or relation.get('output_role', {}).get('status') != 'known'):
            if excluded is not None:
                excluded.append({'status': 'UNKNOWN', 'parameter_relation': index,
                    'reason': 'audited_parameter_role_contract_required'})
            continue
        if actual_inputs is None:
            actual_inputs = describe_bound_parameter_roles(candidate, {
                'bindings': {name: _program(value['program']) for name, value in fixed.items()},
                '_before_program': _program(parent['program']).expanded(),
                '_roles': parent['role_signature'], '_output_unit': parent['output_unit'],
                'before_ref': parent.get('source_ref')}, budget)['roles']
            if any(role.get('reason') == 'parameter_role_budget_exhausted' for role in actual_inputs.values()):
                raise _Incomplete
        program = _program(relation['program'])
        features = {}
        for name in program.required_features():
            if name in relation.get('evidence_inputs', []):
                expected = relation['input_contracts'][name]
                actual = (context or {}).get('role_signature', {}).get(name)
                if ((context or {}).get('status') != 'ready'
                        or (context or {}).get('selection') != expected['selection']
                        or actual != {key: expected[key] for key in ('role', 'unit', 'time')}
                        or name not in context['features']):
                    if excluded is not None:
                        excluded.append({'status': 'UNKNOWN', 'parameter_relation': index,
                            'reason': 'measured_revision_condition_unavailable', 'required_input': name})
                    break
                features[name] = context['features'][name]
            elif name in fixed and _program(fixed[name]['program']).op == 'const':
                actual = actual_inputs.get(name, {})
                if not role_compatible(relation.get('input_roles', {}).get(name, {}), actual):
                    if excluded is not None:
                        excluded.append({'status': 'UNKNOWN', 'parameter_relation': index,
                            'reason': 'current_parent_parameter_role_unaligned', 'required_input': name})
                    break
                features[name] = actual['canonical_parameter']
            else:
                break
        else:
            _spend(budget)
            value = program.predict(Observation(features, 0.))
            name = relation['output_hole']
            component = {**{key: deepcopy(parent[key]) for key in ('role_signature', 'context_ref')},
                'id': f'{candidate["id"]}:parameter_relation:{index}',
                'program': Program('const', value=value).to_dict(),
                'output_unit': holes[name]['output_unit'],
                'fitting_parameter_paths': [[]] if purpose == 'parameter_proposal' else [],
                'source_ref': {'kind': 'parameter_relation', 'candidate_id': candidate['id'], 'index': index},
                'parameter_relation': {'candidate_id': candidate['id'], 'index': index,
                    'purpose': purpose,
                    'parameter_coordinate': 'canonical_basis',
                    'output_role': deepcopy(relation['output_role']),
                    'parent_inputs': {key: value for key, value in features.items()
                                      if key not in relation.get('evidence_inputs', [])},
                    'evidence_inputs': {key: features[key] for key in relation.get('evidence_inputs', [])},
                    'evidence_refs': deepcopy((context or {}).get('source_refs', [])),
                    'predicted_parameter': value,
                    'scope': ('parameter_initialization_requires_world_fit_and_comparison' if
                        purpose == 'parameter_proposal' else 'source_parameter_mapping_not_world_confirmation')}}
            result.setdefault(name, []).append(component)
    return result


def _align_parameter_components(candidate, parent, chosen, prepared, budget, excluded):
    """Convert a canonical prediction only after its actual target basis is bound."""
    from .parameter_roles import describe_bound_parameter_roles, role_compatible

    derived = {name: value for name, value in prepared.items() if value.get('parameter_relation')}
    if not derived:
        return True
    after = _substitute(Definition.from_dict(candidate['definition']).body,
                        {name: value['_program'] for name, value in prepared.items()})
    roles = describe_bound_parameter_roles(candidate, {
        'bindings': {name: value['_program'] for name, value in derived.items()},
        '_after_program': after, '_roles': parent['role_signature'],
        '_output_unit': parent['output_unit']}, budget)['roles']
    if any(role.get('reason') == 'parameter_role_budget_exhausted' for role in roles.values()):
        raise _Incomplete
    for name, component in derived.items():
        relation, actual = component['parameter_relation'], roles.get(name, {})
        if not role_compatible(relation['output_role'], actual):
            excluded.append({'status': 'UNKNOWN', 'parameter_relation': relation['index'],
                'reason': 'current_after_parameter_role_unaligned', 'output_hole': name})
            return False
        value = relation['predicted_parameter'] / actual['coordinate_scale']
        declaration = {**chosen[name], 'program': Program('const', value=value).to_dict(),
            'parameter_relation': {**relation, 'canonical_prediction': relation['predicted_parameter'],
                'target_coordinate_scale': actual['coordinate_scale'],
                'applied_parameter': value}}
        chosen[name], prepared[name] = declaration, _component(declaration, budget)
    return True


def _exact_parameter_roles_preserved(candidate, parent, prepared, budget, excluded):
    """A scored fit cannot change the meaning of a fixed exact prediction."""
    from .parameter_roles import describe_bound_parameter_roles, role_compatible

    exact = {name: value for name, value in prepared.items()
             if value.get('parameter_relation', {}).get('purpose') == 'exact_edit_map'}
    if not exact:
        return True
    after = _substitute(Definition.from_dict(candidate['definition']).body,
                        {name: value['_program'] for name, value in prepared.items()})
    roles = describe_bound_parameter_roles(candidate, {
        'bindings': {name: value['_program'] for name, value in exact.items()},
        '_after_program': after, '_roles': parent['role_signature'],
        '_output_unit': parent['output_unit']}, budget)['roles']
    if any(role.get('reason') == 'parameter_role_budget_exhausted' for role in roles.values()):
        raise _Incomplete
    for name, value in exact.items():
        relation, actual = value['parameter_relation'], roles.get(name, {})
        if (not role_compatible(relation['output_role'], actual)
                or actual['coordinate_scale'] != relation['target_coordinate_scale']):
            excluded.append({'status': 'UNKNOWN', 'parameter_relation': relation['index'],
                'reason': 'exact_parameter_basis_changed_during_world_fit', 'output_hole': name})
            return False
    return True


def _syntax_nodes(program):
    # Program.size deliberately expands calls; role walkers also visit unused
    # lexical arguments, which must remain included in their work bound.
    return 1 + sum(_syntax_nodes(child) for child in program.children)


def _contract_work_bound(component):
    """Bound existing unit-recovery walks while only declared constants change."""
    program = _program(component['program']).expanded()

    def walk(node):
        # _prepare visits each node. A compound product may inspect both child
        # declarations; comparison operands may run the scalar unit walker.
        extra = len(node.children) if node.op == 'mul' else 0
        if node.op in {'ge', 'lt'}:
            extra += sum(child.size + 1 for child in node.children)
        return 1 + extra + sum(walk(child) for child in node.children)

    source = component.get('_contract_source')
    recovery = 2 * _program(source['_pattern']).expanded().size if source else 0
    return walk(program) + recovery


def _revision_visit_quote(candidate, parent, prepared, fixed, template, rows, comparison_rows, reused):
    """Price one cold finite fit and every possible checked output, not a frontier."""
    from .abstractions import quote_parameter_slot_fit

    fit = (quote_parameter_slot_fit(template, rows) if reused is None else
           {'compute': 0, 'maximum_outputs': len(reused[0]), 'scope': 'retained_complete_fit'})
    body = Definition.from_dict(candidate['definition']).body
    lexical = _substitute(body, {name: value['_program'] for name, value in prepared.items()})
    expanded = lexical.expanded()
    component_work = sum(_contract_work_bound(value) for name, value in prepared.items() if name not in fixed)
    # _instantiate charges expanded nodes; occurrence matching and declarations
    # each traverse the Definition body; every constraint uses <=2 properties.
    binding_work = (_expanded_size(body, {name: value['_program'] for name, value in prepared.items()})
                    + 2 * body.expanded().size + 2 * len(candidate['learned_constraints']))
    exact = sum(value.get('parameter_relation', {}).get('purpose') == 'exact_edit_map'
                for value in prepared.values())
    role_work = 0
    if exact:
        # Existing role checks scan both patterns, locate every occurrence,
        # differentiate through their ancestors, then canonicalize the basis.
        # Constants preserve syntax. This covers all full walks, even when a
        # changed coefficient lets a previous early UNKNOWN continue farther.
        before_size = _syntax_nodes(_program(candidate['before_pattern']))
        body_size = _syntax_nodes(body)
        role_work = before_size + body_size + exact * (
            body_size ** 2 + expanded.size ** 2 + 14 * expanded.size + 3 * _syntax_nodes(lexical))
    per_output = component_work + binding_work + role_work
    maximum = fit['maximum_outputs']
    held = comparison_rows * maximum
    return {'compute': fit['compute'] + maximum * per_output + held,
            'fit_compute': fit['compute'], 'assembly_compute': maximum * per_output,
            'assembly_compute_per_output': per_output,
            'reserved_comparison_compute': held, 'maximum_outputs': maximum,
            'fit_quote': fit, 'scope': 'one_binding_complete_finite_stage',
            'does_not_include': ['caller_paid_evidence_checks', 'caller_wall_time']}


def _prepare_revision_binding(candidate, parent, fixed, open_holes, pools,
                              hypothesis, indices, budget, excluded):
    """Use the same typed binding construction for execution and quotation."""
    chosen = {**fixed, **{hole['name']: pool[index]
        for hole, pool, index in zip(open_holes, pools, indices)}}
    prepared = {name: _component(value, budget) for name, value in chosen.items()}
    after = _assign_properties(Definition.from_dict(candidate['definition']).body,
        hypothesis['assignments'], {name: value['_program'] for name, value in prepared.items()}, budget)
    definition = Definition(candidate['id'], tuple(hole['name'] for hole in candidate['holes']), after)
    variant = {**candidate, 'definition': definition.to_dict(),
               'learned_constraints': hypothesis['constraints']}
    aligned = _align_parameter_components(variant, parent, chosen, prepared, budget, excluded)
    return chosen, prepared, variant, aligned


def quote_revision_frontier(preparation, count, budget):
    """Quote a finite fresh prefix without solving or changing its real cursor.

    The first parent/pool/retained-fit preparation is reused. Later bindings use
    the exact execution constructor and fitter quotation; these are prospective
    positions, never completed work or replacement pending-fit records.
    """
    prepared_visit = preparation.get('prepared_visit', preparation)
    context = preparation.get('_quote_context') or prepared_visit['_quote_context']
    original = preparation['_original_cursor']
    if original.get('pending_fits') or original.get('layer', 0) or original.get('offset', 0):
        raise ValueError('fresh frontier quotation requires no inherited pending fits')
    candidate, parent = context['candidate'], context['parent']
    result = []

    def future_query_bound(template):
        # A completed future binding can add at most one initial/ridge/stable
        # batch. Its constants change values, not this template's topology.
        start = budget.spent
        _fit_template_key(template, context['fitting']['risk_quantile'], parent, budget)
        key_work = budget.spent - start
        rows, slots = len(context['rows']), len(template.parameter_paths)
        return (3 + _program(candidate['before_pattern']).expanded().size
                + max((len(h['assignments']) for h in context['hypotheses']), default=0)
                + Definition.from_dict(candidate['definition']).body.expanded().size
                + rows + 3 * slots + 1 + 2 * key_work + 1 + rows + 3 * slots)

    for indices, position in islice(_diagonal_indices(context['lengths'], {}), count):
        if prepared_visit and position == prepared_visit.get('search_position'):
            before = budget.spent
            future = future_query_bound(context['first_template'])
            result.append({'position': deepcopy(position), 'indices': list(indices),
                'status': 'quoted', 'quote': deepcopy(prepared_visit['quote']),
                'quotation_compute': budget.spent - before,
                'future_fit_query_compute_bound': future})
            continue
        before = budget.spent
        _spend(budget)
        excluded = []
        _, prepared, variant, aligned = _prepare_revision_binding(candidate, parent,
            context['fixed'], context['open_holes'], context['pools'],
            context['hypotheses'][indices[-1]], indices, budget, excluded)
        initial, checks = (_bound_proposal(variant, prepared, budget) if aligned else (None, []))
        quote = None
        future = 0
        if initial is not None:
            template = _revision_fitting_template(variant, prepared, set(context['fixed']))
            if context['retained_fits'] and context['retained_index'] is None:
                context['retained_index'] = _retained_fit_index(context['retained_fits'], budget)
            reused = (_reuse_revision_fit(template, context['rows'], context['fitting']['risk_quantile'],
                parent, context['retained_fits'], budget, index=context['retained_index'])
                if context['retained_fits'] else None)
            quote = _revision_visit_quote(variant, parent, prepared, context['fixed'], template,
                context['rows'], context['comparison_rows'], reused)
            future = future_query_bound(template)
        result.append({'position': deepcopy(position), 'indices': list(indices),
            'status': 'quoted' if quote is not None else 'rejected_by_source_contract',
            'quote': quote, 'quotation_compute': budget.spent - before,
            'future_fit_query_compute_bound': future,
            'excluded': excluded, 'constraint_checks': checks})
    return result


def propose_revision(candidate, parent, components, allocation=None, *, evidence_checks=(), cursor=None,
                     fitting=None, retained_fits=None, retained_fit_state=None, grammar_preparation=None):
    """Keep the legacy finite frontier proposer and its original allocation."""
    steps = _propose_revision_steps(candidate, parent, components, allocation,
        evidence_checks=evidence_checks, cursor=cursor, fitting=fitting,
        retained_fits=retained_fits, retained_fit_state=retained_fit_state,
        grammar_preparation=grammar_preparation)
    try:
        next(steps)
    except StopIteration as completed:
        return completed.value
    raise AssertionError('legacy revision proposer unexpectedly suspended')


def prepare_revision_visit(candidate, parent, components, allocation=None, *, evidence_checks=(),
                           cursor=None, fitting=None, retained_fits=None, retained_fit_state=None,
                           grammar_preparation=None):
    """Construct and price one pending or next unvisited binding; never fit it.

    ``prepared_visit`` owns a suspended Python frame and is RAM-only. Persist
    its small quote and the ordinary output/cursor records, never this packet.
    The preparation cost is charged here; execution reports only its increment.
    """
    original_cursor = deepcopy(cursor or {'layer': 0, 'offset': 0})
    planning_context = {}
    if fitting is None:
        result = {'status': 'not_applicable', 'reason': 'finite_revision_fitting_contract_required',
                  'proposals': [], 'excluded': [], 'cursor': original_cursor,
                  'actual_compute_cost': 0, 'reserved_comparison_compute': 0}
        return {'prepared_visit': None, 'result': result, 'quote': None,
                'actual_compute_cost': 0, 'cursor': original_cursor}
    steps = _propose_revision_steps(candidate, parent, components, allocation,
        evidence_checks=evidence_checks, cursor=original_cursor, fitting=fitting,
        retained_fits=retained_fits, retained_fit_state=retained_fit_state,
        grammar_preparation=grammar_preparation, _complete_visit=True,
        _planning_context=planning_context)
    try:
        packet = next(steps)
    except StopIteration as completed:
        result = completed.value
        return {'prepared_visit': None, 'result': result, 'quote': None,
                'actual_compute_cost': result['actual_compute_cost'], 'cursor': result['cursor'],
                '_original_cursor': original_cursor, '_quote_context': planning_context}
    packet.update(_generator=steps, _original_cursor=original_cursor,
                  _preparation_compute=packet['_budget'].spent)
    return {'prepared_visit': packet, 'quote': deepcopy(packet['quote']),
            'actual_compute_cost': packet['_preparation_compute'], 'cursor': original_cursor,
            'search_position': deepcopy(packet['search_position']),
            '_original_cursor': original_cursor, '_quote_context': planning_context}


def execute_revision_visit(prepared_visit, allocation=None):
    """Resume the constructed binding once, or leave its cursor intact if unfit."""
    allowed = _budget(allocation).limit
    quote = prepared_visit['quote']
    if allowed < quote['compute']:
        return {'status': 'UNKNOWN', 'reason': 'complete_research_unit_unaffordable',
                'proposals': [], 'excluded': [], 'cursor': deepcopy(prepared_visit['_original_cursor']),
                'actual_compute_cost': 0, 'reserved_comparison_compute': 0,
                'completed_research_unit': False, 'complete_research_unit_quote': deepcopy(quote)}
    steps = prepared_visit.pop('_generator')
    budget = prepared_visit['_budget']
    prepared_cost = prepared_visit['_preparation_compute']
    budget.limit = prepared_cost + allowed
    try:
        next(steps)
    except StopIteration as completed:
        result = completed.value
    else:
        raise AssertionError('complete revision visit continued beyond its one binding')
    result['actual_compute_cost'] -= prepared_cost
    result['preparation_compute_cost'] = prepared_cost
    result['complete_research_unit_quote'] = deepcopy(quote)
    return result


def _propose_revision_steps(candidate, parent, components, allocation=None, *, evidence_checks=(), cursor=None,
                            fitting=None, retained_fits=None, retained_fit_state=None,
                            grammar_preparation=None, _complete_visit=False, _planning_context=None):
    """Fix a real parent and solve remaining holes with the shared old grammar.

    Triggering uses actual cited rows. Source property hypotheses remain
    alternatives. Explicit fitting evaluates the whole after Program and only
    changes declared open-hole constants; it never uses world labels as internal
    hole labels. Reconstruction and target fitting do not add scientific support.
    """
    budget = _budget(allocation)
    comparison_rows = (allocation or {}).get('comparison_rows', 0)
    if type(comparison_rows) is not int or comparison_rows < 0:
        raise ValueError('comparison_rows must be a nonnegative integer')
    proposals, excluded = [], []
    next_cursor = deepcopy(cursor or {'layer': 0, 'offset': 0})
    result = {'status': 'UNKNOWN', 'proposals': proposals, 'excluded': excluded,
              'budget_exhausted': False, 'support_updated': False, 'independent_confirmation': False,
              'affects_existing_candidates': False, 'new_observations': 0, 'llm_calls': 0,
              'retained_fit_reuse_count': 0, 'retained_fit_index_compute': 0}
    next_allowance = 0

    def require_progress(work_budget, fallback=1):
        nonlocal next_allowance
        denied = work_budget.last_denied_request
        additional = denied['additional_compute'] if denied else max(1, fallback)
        next_allowance = max(next_allowance, budget.limit + additional)

    retained_index = (retained_fit_state or {}).get('index')
    try:
        if candidate.get('purpose') == 'parameter_proposal' and fitting is None:
            raise _MissingContract('whole_after_fitting_required_for_parameter_proposal')
        if not any(check['before']['failed'] for check in candidate.get('trigger_checks', [])):
            raise _MissingContract('source_measured_discrepancy_required')
        if parent is None:
            raise _MissingContract('actual_parent_Program_required')
        current = _component(parent, budget)
        result['parent_contract_recovery'] = deepcopy(current['_prepared']['contract_recovery'])
        pattern = _program(candidate['before_pattern'])
        bound, paths = _match_parent(pattern, current['_program'], candidate['holes'], budget)
        relevant = [example for example in evidence_checks if example.get('source_ref') and 'observation' in example
                    and example.get('context_ref', parent['context_ref']) == parent['context_ref']]
        if not relevant:
            raise _MissingContract('target_measured_discrepancy_required')
        trigger = [{'source_ref': deepcopy(example['source_ref']),
                    **_discrepancy(current['_program'], example, budget)} for example in relevant]
        result['trigger_checks'] = trigger
        if not any(check['failed'] for check in trigger):
            result['status'] = 'not_applicable'
            result['reason'] = 'parent_agrees_with_declared_evidence'
            return result
        fixed = {}
        for name, program in bound.items():
            origin = tuple(paths[name])
            declarations = {}

            def rebase(node, path=()):
                _spend(budget)
                contract = current['_prepared']['contracts'][(*origin, *path)]
                declarations['.'.join(map(str, path))] = {
                    key: deepcopy(contract[key]) for key in ('output_unit', 'time_contract')}
                for index, child in enumerate(node.children):
                    rebase(child, (*path, index))

            rebase(program)
            fixed[name] = {**{key: value for key, value in parent.items()
                if not key.startswith('_') and key not in ('subtree_contracts', 'subtree_contract_sources')},
                'program': program.to_dict(), 'program_path': paths[name],
                'output_unit': current['_prepared']['contracts'][origin]['output_unit'],
                'subtree_contracts': declarations, 'subtree_contract_sources': {
                    'program_ref': deepcopy(parent.get('program_source_ref')),
                    'program_path': deepcopy(paths[name]),
                    'constructor_sources': deepcopy(current['_prepared']['contract_recovery']['sources'])}}
        open_holes = [hole for hole in candidate['holes'] if hole['name'] not in fixed]
        result['parent_binding'] = {'component_id': parent['id'], 'version': parent.get('version', 1),
            'source_ref': deepcopy(parent.get('source_ref')), 'fixed_holes': deepcopy(paths),
            'remaining_holes': [hole['name'] for hole in open_holes]}
        rows = [Observation(item['observation']['features'], item['observation']['target'],
                            tuple(item['observation'].get('history', []))) for item in relevant]
        context = None
        if any(item.get('evidence_inputs') for item in candidate.get('parameter_relations', [])):
            from .revision_evidence import evidence_context

            context = evidence_context(current['_program'], relevant, parent['role_signature'],
                parent['output_unit'], trigger, budget)
            result['parameter_condition_context'] = context
        parameter_components = _parameter_components(candidate, parent, fixed, budget,
            context=context, excluded=excluded, whole_after_fitting=fitting is not None)
        if fitting is not None:
            from .parent_reference_relations import parent_reference_components

            parent_components = parent_reference_components(candidate, parent, budget, excluded=excluded)
            saved_fitting = next_cursor.setdefault('fitting', {})
            if grammar_preparation is None and saved_fitting.get('grammar_preparation_ref'):
                raise _MissingContract('retained_grammar_preparation_required')
            fitting_state = saved_fitting
            if grammar_preparation is not None:
                # Only the already validated, complete grammar prefix is shared.
                # Its thresholds/Programs remain in the retained source, while
                # this parent's exclusions and frontier remain independent.
                fitting_state = {**deepcopy(grammar_preparation['state']),
                    **{key: deepcopy(value) for key, value in saved_fitting.items()
                       if key != 'grammar_preparation_ref'}}
            pools = _revision_fitting_pools(candidate, parent, components, open_holes, rows, fitting,
                budget, fitting_state, parameter_components=parameter_components,
                parent_components=parent_components, excluded=excluded)
            if grammar_preparation is not None:
                next_cursor['fitting'] = {
                    'grammar_preparation_ref': deepcopy(grammar_preparation['source_ref']),
                    **{key: value for key, value in fitting_state.items()
                       if key not in ('structures', 'grammar_cursor')}}
        else:
            pools = []
            for hole in open_holes:
                pool = []
                for component in components:
                    _spend(budget)
                    if component.get('context_ref') == parent['context_ref'] and component.get('output_unit') == hole['output_unit']:
                        pool.append(component)
                if not pool and not parameter_components.get(hole['name']):
                    raise _MissingContract('typed_open_hole_components_or_fitting_contract_required')
                pools.append(pool)
        if fitting is None:
            for hole, pool in zip(open_holes, pools):
                pool.extend(parameter_components.get(hole['name'], []))
        # A final hypothesis axis makes each finite empirical interpretation
        # compete, rather than silently conjoining fixed and relational claims.
        hypotheses = candidate['constraint_hypotheses']
        lengths = [*(len(pool) for pool in pools), len(hypotheses)]
        if _planning_context is not None:
            _planning_context.update(candidate=candidate, parent=parent, fixed=fixed,
                open_holes=open_holes, pools=pools, hypotheses=hypotheses, lengths=lengths,
                rows=rows, fitting=fitting, comparison_rows=comparison_rows,
                retained_fits=retained_fits, retained_index=retained_index)
        tried = 0
        pending = next_cursor.setdefault('pending_fits', []) if fitting is not None else []
        prior_pending = deepcopy(pending)
        # Resume unfinished fits before extending the finite new frontier.
        work = chain(
            ((item['indices'], item['position'], True) for item in prior_pending),
            ((indices, position, False) for indices, position in _diagonal_indices(lengths, cursor)))
        if _complete_visit:
            work = islice(work, 1)
        for indices, position, retry in work:
            _spend(budget)
            if fitting is None and comparison_rows:
                # This binding can emit at most one concrete Program.
                if not budget.can_spend(comparison_rows):
                    raise _Incomplete
                budget.reserve += comparison_rows
            hypothesis = hypotheses[indices[-1]]
            chosen, prepared, variant, aligned = _prepare_revision_binding(candidate, parent,
                fixed, open_holes, pools, hypothesis, indices, budget, excluded)
            if not aligned:
                if _complete_visit:
                    result['completed_research_unit'] = True
                    result['research_unit_position'] = deepcopy(position)
                tried += 1
                budget.reserve = comparison_rows * len(proposals)
                if retry:
                    pending.remove({'indices': list(indices), 'position': position})
                else:
                    next_cursor.update(position)
                continue
            fits = [(prepared, None)]
            if fitting is not None:
                # Reject incompatible source-property interpretations before
                # evaluating their possibly unavailable historical branches.
                initial, initial_checks = _bound_proposal(variant, prepared, budget)
                if initial is None:
                    if _complete_visit:
                        result['completed_research_unit'] = True
                        result['research_unit_position'] = deepcopy(position)
                    excluded.append({'reason': 'learned_property_constraint_violated',
                                     'constraint_checks': initial_checks})
                    tried += 1
                    if retry:
                        pending.remove({'indices': list(indices), 'position': position})
                    else:
                        next_cursor.update(position)
                    continue
                template = None
                reused = None
                if retained_fits:
                    template = _revision_fitting_template(variant, prepared, set(fixed))
                    if retained_index is None:
                        before_index = budget.spent
                        try:
                            retained_index = _retained_fit_index(retained_fits, budget)
                        finally:
                            result['retained_fit_index_compute'] += budget.spent - before_index
                        if retained_fit_state is not None:
                            retained_fit_state['index'] = retained_index
                            retained_fit_state['index_compute'] = budget.spent - before_index
                    reused = _reuse_revision_fit(template, rows, fitting['risk_quantile'], parent,
                        retained_fits, budget, index=retained_index)
                if _complete_visit:
                    template = template or _revision_fitting_template(variant, prepared, set(fixed))
                    visit_quote = _revision_visit_quote(variant, parent, prepared, fixed, template,
                        rows, comparison_rows, reused)
                    if _planning_context is not None:
                        _planning_context['retained_index'] = retained_index
                        _planning_context['first_template'] = template
                        _planning_context['retained_fit_index_compute'] = result['retained_fit_index_compute']
                    # The suspended frame owns exactly this constructed binding.
                    # Resuming it does not repeat parent, pool, or saved-fit work.
                    yield {'quote': visit_quote, '_budget': budget,
                           'search_position': deepcopy(position),
                           '_quote_context': {'candidate': candidate, 'parent': parent,
                               'fixed': fixed, 'open_holes': open_holes, 'pools': pools,
                               'hypotheses': hypotheses, 'lengths': lengths, 'rows': rows,
                               'fitting': fitting, 'comparison_rows': comparison_rows,
                               'retained_fits': retained_fits, 'retained_index': retained_index,
                               'first_template': template}}
                if reused is not None:
                    fitted, parameter_paths, report = reused
                    if comparison_rows and not budget.can_spend(comparison_rows * len(fitted)):
                        raise _Incomplete
                    result['retained_fit_reuse_count'] += 1
                else:
                    quota = (visit_quote['fit_compute'] if _complete_visit else
                        budget.remaining if retry else max(5 * len(rows), budget.remaining // max(1, sum(lengths))))
                    if _complete_visit:
                        budget.reserve = visit_quote['reserved_comparison_compute']
                    if comparison_rows and not _complete_visit:
                        from .abstractions import eligible_fitting_rows

                        template = template or _revision_fitting_template(variant, prepared, set(fixed))
                        eligible = len(eligible_fitting_rows(template.pattern, rows))
                        if eligible:
                            # Initial/linear/strict-improvement results all have
                            # measured whole-Program losses. Bound the batch's
                            # output before spending its local fitting allowance.
                            low, high = 0, min(quota, budget.remaining)
                            available = budget.remaining
                            while low < high:
                                middle = (low + high + 1) // 2
                                held = comparison_rows * (middle // eligible + 1)
                                if middle + held <= available:
                                    low = middle
                                else:
                                    high = middle - 1
                            quota = low
                            held = comparison_rows * (quota // eligible + 1)
                            if not budget.can_spend(held):
                                raise _Incomplete
                            budget.reserve += held
                    local = SearchBudget(min(quota, budget.remaining))
                    fitted, parameter_paths, report = _fit_revision_binding(
                        variant, prepared, set(fixed), rows, fitting['risk_quantile'], local,
                        **({'template': template} if template is not None else {}))
                    _spend(budget, local.spent)
                    if not report['search_completed']:
                        # A repeated smaller slice cannot fund the next denied
                        # atomic operation. Keep the existing pending fit, and
                        # expose its next request instead of paying that prefix
                        # indefinitely. A paid-row pass is the fitter's unit.
                        require_progress(local, len(rows))
                if comparison_rows:
                    budget.reserve = comparison_rows * (len(proposals) + len(fitted))
                fits = []
                for fitted_index, fitted_call in enumerate(fitted):
                    fitted_components = dict(prepared)
                    for index, name in enumerate(fitted_call.definition.arguments):
                        if name not in fixed:
                            component = {**chosen[name], 'program': fitted_call.children[index].to_dict()}
                            if component.get('parameter_relation', {}).get('purpose') == 'parameter_proposal':
                                component['parameter_relation'] = {
                                    **component['parameter_relation'], 'world_refit': True,
                                    'fitted_program': fitted_call.children[index].to_dict(),
                                    'exact_edit_map_confirmed': False,
                                    'scope': 'initialization_then_whole_after_fit_not_exact_map_support'}
                            fitted_components[name] = _component(component, budget)
                    fits.append((fitted_components, {'parameter_paths': [list(path) for path in parameter_paths],
                        'parameters': [_at(fitted_call, path).value for path in parameter_paths],
                        'status': report['status'], 'fitted_rows': report['actual_fitted_rows'],
                        'search_completed': report['search_completed'],
                        'parameter_identifiability': report['parameter_identifiability'],
                        'fitting_method': report['candidate_methods'][fitted_index],
                        **({'linear_diagnostics': deepcopy(report['linear_diagnostics'])}
                           if 'linear_diagnostics' in report else {}),
                        'training_loss': report['losses'][fitted_call],
                        **({'fit_reuse_ref': deepcopy(report['fit_reuse_ref']),
                            'fit_reuse_output_ref': deepcopy(report['fit_reuse_output_refs'][fitted_index])}
                           if report.get('fit_reuse_ref') else {}),
                        'evidence_refs': [deepcopy(item['source_ref']) for item in relevant],
                        'scope': 'whole_after_paid_fit_not_confirmation'}))
            tried += 1
            for fit_index, (fitted_components, fit_info) in enumerate(fits):
                if fit_info is not None and not _exact_parameter_roles_preserved(
                        variant, parent, fitted_components, budget, excluded):
                    continue
                try:
                    proposal, checks = _bound_proposal(variant, fitted_components, budget)
                except _MissingContract as error:
                    excluded.append({'reason': str(error)})
                    proposal = None
                if proposal is not None:
                    proposal.update(parent_binding=deepcopy(result['parent_binding']),
                        constraint_hypothesis=hypothesis['id'], search_position=deepcopy(position),
                        trigger_checks=deepcopy(trigger))
                    if fit_info is not None:
                        proposal.update(fitting=fit_info)
                        proposal['search_position']['fit_index'] = fit_index
                    for name in paths:
                        proposal['bindings'][name]['program_path'] = paths[name]
                    for name, component in fitted_components.items():
                        if component.get('parameter_relation'):
                            proposal['bindings'][name]['parameter_relation'] = deepcopy(component['parameter_relation'])
                        if component.get('parent_reference'):
                            proposal['bindings'][name]['parent_reference'] = deepcopy(component['parent_reference'])
                    proposals.append(proposal)
            budget.reserve = comparison_rows * len(proposals)
            if fitting is not None:
                unfinished = {'indices': list(indices), 'position': deepcopy(position)}
                if not report['search_completed'] and not retry:
                    pending.append(unfinished)
                elif report['search_completed'] and retry:
                    pending.remove(unfinished)
            if not retry:
                next_cursor.update(position)
            if _complete_visit:
                result['completed_research_unit'] = report['search_completed']
                result['research_unit_position'] = deepcopy(position)
        frontier_remaining = (_complete_visit and
            next(_diagonal_indices(lengths, next_cursor), None) is not None)
        result.update(status='completed' if proposals else 'not_applicable',
                      attempted_bindings=tried, search_completed=not pending and not frontier_remaining,
                      incomplete_fit_positions=[deepcopy(item['position']) for item in pending])
        if _complete_visit:
            result['frontier_exhausted'] = not frontier_remaining
        if pending:
            result.update(status='UNKNOWN', reason='bounded_parameter_fitting_incomplete')
        elif frontier_remaining:
            result['reason'] = 'complete_research_unit_finished_unvisited_bindings_remain'
    except _MissingContract as error:
        result['reason'] = str(error)
    except _Incomplete:
        result.update(status='UNKNOWN', budget_exhausted=True, search_completed=False)
        require_progress(budget)
    finally:
        result.update(actual_compute_cost=budget.spent, cursor=next_cursor,
                      reserved_comparison_compute=comparison_rows * len(proposals),
                      incomplete_fit_positions=[deepcopy(item['position']) for item in
                                                next_cursor.get('pending_fits', [])])
        if next_allowance and not result.get('search_completed', True):
            result['minimum_compute_for_progress'] = next_allowance
    return result

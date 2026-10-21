"""Learn optional construction contexts and parent-anchored finite revisions.

The body is a native Definition with lexical subtree arguments. Its operations
come entirely from the sources. Joint revisions share typed variables across
before and after, while their measured triggers and property interpretations
remain scoped empirical candidates. Learned equalities concern native Program
properties; they restrict this finite proposer, never the old candidate bank.
"""

from copy import deepcopy
from itertools import chain, combinations
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


def _prepare(record, program, budget):
    signatures = record.get('role_signature')
    if not isinstance(signatures, dict) or record.get('output_unit') is None or 'context_ref' not in record:
        raise _MissingContract('explicit_roles_output_unit_and_context_required')
    variables = {name for name, _ in program.observed_variables()}
    for name in variables:
        if not isinstance(signatures.get(name), dict) or any(
                key not in signatures[name] or signatures[name][key] is None for key in ('role', 'unit', 'time')):
            raise _MissingContract('missing_role_unit_or_time_contract')
    times = _unique(signatures[name]['time'] for name in sorted(variables))
    if not times:
        times = deepcopy(record.get('time_contract', []))
    contracts, count = {}, 0

    def visit(node, path=(), expected_unit=None):
        nonlocal count
        _spend(budget)
        count += 1
        declared = record.get('subtree_contracts', {}).get('.'.join(map(str, path)), {})
        own_unit = declared.get('output_unit', expected_unit)
        if node.op in {'feature', 'lag'}:
            own_unit = signatures[node.name]['unit']
        elif node.op in _PREDICATES:
            own_unit = own_unit if expected_unit is not None else 'dimensionless'
        child_units = [None] * len(node.children)
        if node.op in _SAME_UNIT:
            child_units = [own_unit] * len(node.children)
        elif node.op == 'if':
            child_units = ['dimensionless', own_unit, own_unit]
        elif node.op in {'ge', 'lt'}:
            units = []
            for index, child in enumerate(node.children):
                supplied = record.get('subtree_contracts', {}).get(
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
        elif node.op == 'mul' and any(child.op == 'const' for child in node.children):
            child_units = ['dimensionless' if child.op == 'const' else own_unit for child in node.children]
        elif node.op == 'accumulate':
            child_units = [own_unit, own_unit]
        for index, child in enumerate(node.children):
            visit(child, (*path, index), child_units[index])
        contracts[path] = {'value_type': 'scalar', 'output_unit': own_unit,
                           'time_contract': deepcopy(times),
                           'used_role_signature': {name: deepcopy(signatures[name])
                               for name, _ in sorted(node.observed_variables())}}
        if node.op in {'feature', 'lag'}:
            contracts[path]['feature_contract'] = deepcopy(signatures[node.name])

    visit(program, expected_unit=record['output_unit'])
    return {'record': record, 'program': program, 'contracts': contracts, 'nodes': count}


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
            raise _MissingContract('incompatible_subtree_unit_or_time')
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
            raise _MissingContract('missing_subtree_output_unit_or_time')
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
    return {key: deepcopy(record[key]) for key in ('episode_id', 'source_id', 'source_ref', 'before_ref',
        'after_ref', 'action_ref', 'evidence_refs', 'cost_refs', 'failure_refs', 'source_cutoff', 'source_outcomes',
        'source_comparison', 'trigger_obligations')
            if key in record}


def _ordered_skeleton(program, budget):
    names = {}

    def visit(node):
        _spend(budget)
        name = node.name
        if node.op in {'feature', 'lag'}:
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


def _compile_revision(sources, identifier, budget):
    records = [source['record'] for source in sources]
    before = [_prepare(record, _program(record['before_program']).expanded(), budget) for record in records]
    body, holes, bindings, paths, pattern = _anti_unify(sources, budget, before_sources=before)
    parent_holes = _arguments(pattern)
    shared_parent = bool(parent_holes.intersection(_arguments(body)))
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
    if not shared_parent:
        # A changed scalar can still depend on the old scalar. Admit this edit
        # only if the later, held-out parameter search establishes such a link.
        candidate['_requires_parameter_link'] = True
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
    return candidate


def _revision_contract(candidate):
    return {key: candidate.get(key) for key in ('before_pattern', 'holes', 'parent_holes',
        'constraint_hypotheses', 'learned_constraints', 'scope', 'output_unit', 'fixed_role_signature')} | {
        'definition': {key: candidate['definition'].get(key) for key in ('body', 'arguments')}}


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
            bindings, _ = _match_parent(pattern, _program(record['before_program']).expanded(),
                                        candidate['holes'], budget)
            _match_parent(body, source['program'], candidate['holes'], budget, bindings=bindings)
        except _MissingContract:
            continue
        records.append({**_source_references(record), 'bindings': bindings,
            '_before_program': _program(record['before_program']).expanded(),
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
        try:
            records = _parameter_sources(candidate, sources, budget)
        except _Incomplete:
            records = []
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
    retained = []
    for candidate in candidates:
        needs_link = candidate.pop('_requires_parameter_link', False)
        if not needs_link or candidate.get('parameter_relations'):
            retained.append(candidate)
    return searches, retained


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
                _prepare(record, before, budget)
                prepared = _prepare(record, after, budget)
                sources.append(prepared)
            except _MissingContract as error:
                excluded.append({'episode_id': record.get('episode_id'), 'status': 'UNKNOWN', 'reason': str(error)})
        groups = 0
        for left, right in combinations(sources, 2):
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
                                     'status': 'UNKNOWN', 'reason': str(error)})
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
                                     'status': 'UNKNOWN', 'reason': str(error)})
                    continue
                if joint is not None:
                    candidates.append(joint)
                else:
                    excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                        'status': 'UNKNOWN' if object_type else 'not_applicable',
                        'reason': 'no_shared_parent_subprogram_in_revision'})
    except _Incomplete:
        unknown = True
    unlinked = {item['id']: [entry['episode_id'] for entry in item['source_outcomes']]
                for item in candidates if item.get('_requires_parameter_link')}
    parameter_searches, candidates = _learn_parameter_updates(candidates, sources, budget, allocation)
    candidates = [item for item in candidates if not item.pop('_requires_parameter_link', False)]
    retained_ids = {item['id'] for item in candidates}
    excluded.extend({'episode_ids': episode_ids,
        'status': 'UNKNOWN' if object_type else 'not_applicable',
        'reason': 'no_shared_parent_subprogram_in_revision'}
        for identifier, episode_ids in unlinked.items() if identifier not in retained_ids
        and not any(item.get('episode_ids') == episode_ids and item['status'] == 'existing_family'
                    for item in excluded))
    if len(sources) < 2 and not excluded:
        excluded.append({'status': 'UNKNOWN', 'reason': 'two_actual_revision_sources_required'})
    return {'status': 'UNKNOWN' if unknown or any(item['status'] == 'UNKNOWN' for item in excluded)
            else 'completed', 'candidates': candidates,
        'excluded': excluded, 'author_work': [], 'actual_compute_cost': budget.spent,
        'parameter_relation_searches': parameter_searches,
        'budget_exhausted': unknown, 'support_updated': False, 'new_observations': 0, 'llm_calls': 0}


def _component(component, budget):
    program = _program(component['program']).expanded()
    prepared = _prepare(component, program, budget)
    used = {name for name, _ in program.observed_variables()}
    times = _unique(component['role_signature'][name]['time'] for name in sorted(used))
    return {**component, '_program': program, '_times': times, '_prepared': prepared}


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
    checks = _constraint_checks(candidate, bindings, program, budget)
    if any(check['status'] == 'violated' for check in checks):
        return None, checks
    return {'program': program.to_dict(), 'bindings': {name: {
            'component_id': component['id'], 'version': component.get('version', 1),
            'source_ref': deepcopy(component.get('source_ref'))} for name, component in components.items()},
        'constraint_checks': checks, 'role_signature': signatures,
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
                            *, parameter_components=None):
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
    structures = state.setdefault('structures', [])
    grammar_cursor = state.setdefault('grammar_cursor', {})
    if len(structures) < maximum and grammar_cursor.get('status') != 'completed':
        stream = iter_synthesis_structures(rows, operations=operations, budget=budget,
            max_history=history, cursor=grammar_cursor,
            seed_programs=tuple(_program(item['program']).expanded() for item in components))
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
    known_units = {_program(item['program']).expanded(): item['output_unit'] for item in components}
    known_units[_program(parent['program']).expanded()] = parent['output_unit']
    for hole in holes:
        pool = [item for item in components if item.get('context_ref') == parent['context_ref']
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


def _parameter_components(candidate, parent, fixed, budget, *, context=None, excluded=None):
    """Apply learned metadata functions to actual matched parent constants."""
    result = {}
    holes = {hole['name']: hole for hole in candidate['holes']}
    for index, relation in enumerate(candidate.get('parameter_relations', [])):
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
                features[name] = _program(fixed[name]['program']).value
            else:
                break
        else:
            _spend(budget)
            value = program.predict(Observation(features, 0.))
            name = relation['output_hole']
            component = {**{key: deepcopy(parent[key]) for key in ('role_signature', 'context_ref')},
                'id': f'{candidate["id"]}:parameter_relation:{index}',
                'program': Program('const', value=value).to_dict(),
                'output_unit': holes[name]['output_unit'], 'fitting_parameter_paths': [],
                'source_ref': {'kind': 'parameter_relation', 'candidate_id': candidate['id'], 'index': index},
                'parameter_relation': {'candidate_id': candidate['id'], 'index': index,
                    'parent_inputs': {key: value for key, value in features.items()
                                      if key not in relation.get('evidence_inputs', [])},
                    'evidence_inputs': {key: features[key] for key in relation.get('evidence_inputs', [])},
                    'evidence_refs': deepcopy((context or {}).get('source_refs', [])),
                    'predicted_parameter': value,
                    'scope': 'source_parameter_mapping_not_world_confirmation'}}
            result.setdefault(name, []).append(component)
    return result


def propose_revision(candidate, parent, components, allocation=None, *, evidence_checks=(), cursor=None,
                     fitting=None):
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
              'affects_existing_candidates': False, 'new_observations': 0, 'llm_calls': 0}
    try:
        if not any(check['before']['failed'] for check in candidate.get('trigger_checks', [])):
            raise _MissingContract('source_measured_discrepancy_required')
        if parent is None:
            raise _MissingContract('actual_parent_Program_required')
        current = _component(parent, budget)
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
            fixed[name] = {**parent, 'program': program.to_dict(), 'program_path': paths[name],
                'output_unit': current['_prepared']['contracts'][tuple(paths[name])]['output_unit']}
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
                                                      context=context, excluded=excluded)
        if fitting is not None:
            pools = _revision_fitting_pools(candidate, parent, components, open_holes, rows, fitting,
                budget, next_cursor.setdefault('fitting', {}), parameter_components=parameter_components)
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
        tried = 0
        pending = next_cursor.setdefault('pending_fits', []) if fitting is not None else []
        prior_pending = deepcopy(pending)
        # Resume unfinished fits before extending the finite new frontier.
        work = chain(((item['indices'], item['position'], True) for item in prior_pending),
                     ((indices, position, False) for indices, position in _diagonal_indices(lengths, cursor)))
        for indices, position, retry in work:
            _spend(budget)
            if fitting is None and comparison_rows:
                # This binding can emit at most one concrete Program.
                if not budget.can_spend(comparison_rows):
                    raise _Incomplete
                budget.reserve += comparison_rows
            hypothesis = hypotheses[indices[-1]]
            chosen = {**fixed, **{hole['name']: pool[index]
                for hole, pool, index in zip(open_holes, pools, indices)}}
            prepared = {name: _component(value, budget) for name, value in chosen.items()}
            after = _assign_properties(Definition.from_dict(candidate['definition']).body,
                hypothesis['assignments'], {name: value['_program'] for name, value in prepared.items()}, budget)
            definition = Definition(candidate['id'], tuple(hole['name'] for hole in candidate['holes']), after)
            variant = {**candidate, 'definition': definition.to_dict(), 'learned_constraints': hypothesis['constraints']}
            fits = [(prepared, None)]
            if fitting is not None:
                # Reject incompatible source-property interpretations before
                # evaluating their possibly unavailable historical branches.
                initial, initial_checks = _bound_proposal(variant, prepared, budget)
                if initial is None:
                    excluded.append({'reason': 'learned_property_constraint_violated',
                                     'constraint_checks': initial_checks})
                    tried += 1
                    if retry:
                        pending.remove({'indices': list(indices), 'position': position})
                    else:
                        next_cursor.update(position)
                    continue
                quota = budget.remaining if retry else max(4 * len(rows), budget.remaining // max(1, sum(lengths)))
                template = None
                if comparison_rows:
                    from .abstractions import eligible_fitting_rows

                    template = _revision_fitting_template(variant, prepared, set(fixed))
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
                if comparison_rows:
                    budget.reserve = comparison_rows * (len(proposals) + len(fitted))
                fits = []
                for fitted_call in fitted:
                    fitted_components = dict(prepared)
                    for index, name in enumerate(fitted_call.definition.arguments):
                        if name not in fixed:
                            component = {**chosen[name], 'program': fitted_call.children[index].to_dict()}
                            fitted_components[name] = _component(component, budget)
                    fits.append((fitted_components, {'parameter_paths': [list(path) for path in parameter_paths],
                        'parameters': [_at(fitted_call, path).value for path in parameter_paths],
                        'status': report['status'], 'fitted_rows': report['actual_fitted_rows'],
                        'training_loss': report['losses'][fitted_call],
                        'evidence_refs': [deepcopy(item['source_ref']) for item in relevant],
                        'scope': 'whole_after_paid_fit_not_confirmation'}))
            tried += 1
            for fit_index, (fitted_components, fit_info) in enumerate(fits):
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
        result.update(status='completed' if proposals else 'not_applicable',
                      attempted_bindings=tried, search_completed=not pending,
                      incomplete_fit_positions=[deepcopy(item['position']) for item in pending])
        if pending:
            result.update(status='UNKNOWN', reason='bounded_parameter_fitting_incomplete')
    except _MissingContract as error:
        result['reason'] = str(error)
    except _Incomplete:
        result.update(status='UNKNOWN', budget_exhausted=True, search_completed=False)
    finally:
        result.update(actual_compute_cost=budget.spent, cursor=next_cursor,
                      reserved_comparison_compute=comparison_rows * len(proposals))
    return result

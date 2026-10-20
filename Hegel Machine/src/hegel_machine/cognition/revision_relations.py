"""Learn optional Program construction contexts from real revised Programs.

The body is a native Definition with lexical subtree arguments. Its operations
come entirely from the sources. Learned equalities concern native Program
properties; they restrict this finite proposer, never the old candidate bank.
"""

from copy import deepcopy
from itertools import combinations, product
import json

from .language import Definition, Program
from .models import SearchBudget


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


def _anti_unify(sources, budget):
    holes, replacements, paths, memo = [], [dict() for _ in sources], [dict() for _ in sources], {}

    def visit(nodes, path=()):
        _spend(budget, len(nodes))
        contracts = [source['contracts'][path] for source in sources]
        comparable = all(contract['output_unit'] == contracts[0]['output_unit']
                         and contract['time_contract'] == contracts[0]['time_contract'] for contract in contracts)
        if not comparable:
            raise _MissingContract('incompatible_subtree_unit_or_time')
        first = nodes[0]
        identical = all(node == first for node in nodes)
        if identical and all(contract == contracts[0] for contract in contracts):
            return first
        same_operator = all((node.op, node.value, node.name, node.lag, len(node.children)) ==
                            (first.op, first.value, first.name, first.lag, len(first.children)) for node in nodes)
        if same_operator and first.children:
            return Program(first.op, first.value, first.name, first.lag,
                           tuple(visit(tuple(node.children[index] for node in nodes), (*path, index))
                                 for index in range(len(first.children))))
        if contracts[0]['output_unit'] is None or not contracts[0]['time_contract']:
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
            paths[index].setdefault(name, list(path))
        return Program('arg', name=name)

    body = visit(tuple(source['program'] for source in sources))
    return body, holes, replacements, paths


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
        'after_ref', 'action_ref', 'evidence_refs', 'cost_refs', 'failure_refs', 'source_cutoff', 'source_outcomes')
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


def consolidate_revision_relations(episodes, allocation=None, candidate_id='relation'):
    """Unify bounded pairs of actual after Programs, retaining their histories."""
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
            try:
                item = _compile([left, right], f'{candidate_id}:{len(candidates)}', budget)
            except _MissingContract as error:
                excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                                 'status': 'UNKNOWN', 'reason': str(error)})
                continue
            if item is None:
                excluded.append({'episode_ids': [lrecord['episode_id'], rrecord['episode_id']],
                    'status': 'existing_family', 'reason': 'existing_ordered_parameter_family'})
                continue
            equivalent = next((old for old in candidates if old['definition']['body'] == item['definition']['body']
                and old['holes'] == item['holes'] and old['learned_constraints'] == item['learned_constraints']
                and old['source_bindings'] == item['source_bindings']), None)
            if equivalent is None:
                candidates.append(item)
    except _Incomplete:
        unknown = True
    if len(sources) < 2 and not excluded:
        excluded.append({'status': 'UNKNOWN', 'reason': 'two_actual_revision_sources_required'})
    return {'status': 'UNKNOWN' if unknown or any(item['status'] == 'UNKNOWN' for item in excluded)
            else 'completed', 'candidates': candidates,
        'excluded': excluded, 'author_work': [], 'actual_compute_cost': budget.spent,
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


def propose_relation(candidate, components, allocation=None):
    """Enumerate only the supplied finite, typed existing component Programs."""
    budget = _budget(allocation)
    proposals, excluded, prepared, incomplete = [], [], [], False
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
        for items in product(*pools):
            _spend(budget)
            bindings = dict(zip((hole['name'] for hole in candidate['holes']), items))
            try:
                proposal, checks = _bound_proposal(candidate, bindings, budget)
            except _MissingContract:
                continue
            if proposal is not None:
                if not any(old['program'] == proposal['program'] for old in proposals):
                    proposals.append(proposal)
            else:
                excluded.append({'bindings': {name: item['id'] for name, item in bindings.items()},
                    'reason': 'learned_property_constraint_violated', 'constraint_checks': checks})
    except _Incomplete:
        incomplete = True
    return {'status': 'UNKNOWN' if incomplete or any(item.get('status') == 'UNKNOWN' for item in excluded)
            else 'completed', 'proposals': proposals, 'excluded': excluded,
        'actual_compute_cost': budget.spent, 'budget_exhausted': incomplete,
        'affects_existing_candidates': False, 'support_updated': False, 'new_observations': 0, 'llm_calls': 0}

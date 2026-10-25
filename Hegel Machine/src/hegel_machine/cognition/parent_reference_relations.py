"""Learn source contexts whose operand denotes the actual before Program.

Positional revision holes need not align across episodes. These contexts keep
the parent role across those positions; their new constants remain fitting
slots, rather than copying a source episode's solution.
"""
from copy import deepcopy

from .language import Definition, Program


def parent_reference_contract(candidate):
    """Semantic cache identity, independent of this object's assigned name."""
    return [{key: deepcopy(relation[key]) for key in (
        'parent_argument', 'parent_output_unit', 'parent_contract', 'holes',
        'output_unit', 'time_contract', 'target_holes', 'subtree_contracts',
        'source_ids', 'source_bindings')}
        | {'definition': {key: deepcopy(relation['definition'][key])
                          for key in ('arguments', 'body')}}
        for relation in candidate.get('parent_reference_relations', [])]


def learn_parent_references(sources, before, paths, parent_holes, budget, identifier):
    from .revision_relations import (
        _anti_unify, _at, _ordered_skeleton, _spend,
        _substitute, _unique, _prepare, _json, _MissingContract,
    )

    groups = {}
    for source_index, (source, old, positions) in enumerate(zip(sources, before, paths)):
        parent = old['program']
        # A matching scalar literal does not identify a predictive parent role.
        if not parent.observed_variables():
            continue
        for outer_hole, position in positions.items():
            if outer_hole in parent_holes or position['side'] != 'after':
                continue
            origin = tuple(position['path'])

            def visit(node, path):
                _spend(budget)
                if node == parent:
                    return
                if any(child == parent for child in node.children):
                    declarations = {}

                    def recover(child, relative=()):
                        _spend(budget)
                        contract = source['contracts'][(*path, *relative)]
                        declarations['.'.join(map(str, relative))] = {
                            key: deepcopy(contract[key]) for key in ('output_unit', 'time_contract')
                            if contract[key] is not None}
                        if child == parent:
                            for suffix, original in old['contracts'].items():
                                _spend(budget)
                                key = '.'.join(map(str, (*relative, *suffix)))
                                existing = declarations.setdefault(key, {})
                                for field in ('output_unit', 'time_contract'):
                                    if original[field] is not None:
                                        if existing.get(field) is not None and existing[field] != original[field]:
                                            raise _MissingContract('conflicting_exact_parent_context_contract')
                                        existing[field] = deepcopy(original[field])
                            return
                        for child_index, part in enumerate(child.children):
                            recover(part, (*relative, child_index))

                    recover(node)
                    # An exact before reference supplies its original compound
                    # unit; existing operator rules can then type new operands.
                    typed = _prepare({**{key: source['record'][key] for key in
                        ('role_signature', 'context_ref')},
                        'output_unit': source['contracts'][path]['output_unit'],
                        'subtree_contracts': declarations}, node, budget)
                    contracts = {}

                    def abstract(child, relative=()):
                        _spend(budget)
                        contract = deepcopy(typed['contracts'][relative])
                        if child == parent:
                            contract['used_role_signature'] = {}
                            contracts[relative] = contract
                            return Program('arg', name='parent_program')
                        contracts[relative] = contract
                        return Program(child.op, child.value, child.name, child.lag,
                            tuple(abstract(part, (*relative, index))
                                  for index, part in enumerate(child.children)))

                    context = abstract(node)
                    parent_contract = {key: deepcopy(old['contracts'][()][key])
                                       for key in ('output_unit', 'time_contract')}
                    key = (_ordered_skeleton(context, budget, canonical_roles=False),
                           _json(parent_contract), _json({key: contracts[()][key]
                               for key in ('output_unit', 'time_contract')}))
                    groups.setdefault(key, []).append({
                        'program': context, 'contracts': contracts, 'record': source['record'],
                        'source_binding_index': source_index,
                        'parent': parent, 'parent_contract': parent_contract,
                        'outer_hole': outer_hole, 'origin': path, 'actual': node})
                for index, child in enumerate(node.children):
                    visit(child, (*path, index))

            visit(_at(source['program'], origin), origin)

    relations = []
    for occurrences in groups.values():
        _spend(budget)
        source_ids = _unique(item['record']['source_id'] for item in occurrences)
        if len(source_ids) < 2:
            continue
        try:
            body, holes, bindings, parameter_paths = _anti_unify(occurrences, budget)
        except _MissingContract:
            continue
        if any(value.op != 'const' for binding in bindings for value in binding.values()):
            continue  # This learner delegates only numeric new slots to the fitter.

        def open_constants(node, path=()):
            _spend(budget)
            if node.op == 'const':
                contract = occurrences[0]['contracts'][path]
                if contract['output_unit'] is None:
                    raise _MissingContract('new_parent_context_parameter_unit_required')
                name = f'hole_{len(holes)}'
                holes.append({'name': name, **{key: deepcopy(contract[key])
                    for key in ('value_type', 'output_unit', 'time_contract')}})
                for occurrence, binding, positions in zip(occurrences, bindings, parameter_paths):
                    binding[name] = _at(occurrence['program'], path)
                    positions[name] = list(path)
                return Program('arg', name=name)
            return Program(node.op, node.value, node.name, node.lag,
                tuple(open_constants(child, (*path, index)) for index, child in enumerate(node.children)))

        # Coinciding source values do not establish a target parameter law.
        try:
            body = open_constants(body)
        except _MissingContract:
            continue
        definition = Definition(f'{identifier}:parent_context:{len(relations)}',
            ('parent_program', *(hole['name'] for hole in holes)), body)
        source_bindings, reconstructions = [], []
        for item, binding, positions in zip(occurrences, bindings, parameter_paths):
            _spend(budget, item['actual'].size)
            if _substitute(body, {'parent_program': item['parent'], **binding}) != item['actual']:
                raise ValueError('parent reference source reconstruction failed')
            # The surrounding revision already retains actions, outcomes,
            # failures and costs. Link it rather than copying that ledger into
            # every learned occurrence and every generated component.
            refs = {'source_id': item['record']['source_id'],
                    'source_binding_index': item['source_binding_index'],
                    'before_ref': deepcopy(item['record']['before_ref']),
                    'after_ref': deepcopy(item['record']['after_ref'])}
            source_bindings.append({**refs, 'outer_hole': item['outer_hole'],
                'program_path': list(item['origin']),
                'parent_contract_recovery': 'exact_before_subtree_contract', 'parameter_refs': {
                    name: {'source_ref': deepcopy(item['record']['after_ref']),
                           'program_path': [*item['origin'], *path]}
                    for name, path in positions.items()}})
            reconstructions.append({**refs, 'program_path': list(item['origin']), 'exact': True})
        contracts = {}

        def retain_contracts(node, path=()):
            _spend(budget)
            contract = occurrences[0]['contracts'][path]
            contracts['.'.join(map(str, path))] = {key: deepcopy(contract[key])
                                                   for key in ('output_unit', 'time_contract')}
            for index, child in enumerate(node.children):
                retain_contracts(child, (*path, index))

        retain_contracts(body)
        relations.append({'id': definition.definition_id, 'definition': definition.to_dict(),
            'parent_argument': 'parent_program', 'parent_contract': occurrences[0]['parent_contract'],
            'parent_output_unit': occurrences[0]['parent_contract']['output_unit'],
            'holes': holes, 'output_unit': contracts['']['output_unit'],
            'time_contract': contracts['']['time_contract'], 'subtree_contracts': contracts,
            'target_holes': _unique(item['outer_hole'] for item in occurrences),
            'source_ids': source_ids, 'source_bindings': source_bindings,
            'source_reconstruction': reconstructions, 'construction_only': True,
            'requires_whole_after_fitting': True, 'scientific_support_added': False,
            'independent_confirmation': False})
    return relations


def parent_reference_components(candidate, parent, budget, excluded=None):
    """Construct typed lexical calls; the shared whole-after fitter solves slots."""
    from .revision_relations import _prepare, _program, _spend, _MissingContract

    relations = candidate.get('parent_reference_relations', [])
    if not relations:
        return {}
    program = _program(parent['program']).expanded()
    prepared = _prepare(parent, program, budget)
    actual_contract = {key: prepared['contracts'][()][key] for key in ('output_unit', 'time_contract')}
    holes = {hole['name']: hole for hole in candidate['holes']}
    result = {}
    for index, relation in enumerate(relations):
        _spend(budget)
        if actual_contract != relation['parent_contract']:
            if excluded is not None:
                excluded.append({'parent_reference': relation['id'], 'reason': 'parent_reference_contract_mismatch'})
            continue
        definition = Definition.from_dict(relation['definition'])
        children = (program, *(Program('const', value=0.) for _ in relation['holes']))
        call = Program('call', name=definition.definition_id, definition=definition, children=children)
        declarations = {}

        def declare(node, path=()):
            _spend(budget)
            key = '.'.join(map(str, path))
            if node.op == 'arg' and node.name == relation['parent_argument']:
                for relative, contract in prepared['contracts'].items():
                    _spend(budget)
                    declarations['.'.join(map(str, (*path, *relative)))] = {
                        field: deepcopy(contract[field]) for field in ('output_unit', 'time_contract')}
                return
            declarations[key] = deepcopy(relation['subtree_contracts'][key])
            for child_index, child in enumerate(node.children):
                declare(child, (*path, child_index))

        declare(definition.body)
        declaration = {key: deepcopy(parent[key]) for key in ('role_signature', 'context_ref')}
        declaration.update(id=f'{candidate["id"]}:parent_context:{index}:{parent["id"]}',
            version=parent.get('version', 1), program=call.to_dict(),
            output_unit=relation['output_unit'], subtree_contracts=declarations,
            source_ref=deepcopy(parent.get('program_source_ref', parent.get('source_ref'))),
            fitting_parameter_paths=[[slot + 1] for slot in range(len(relation['holes']))],
            parent_reference={'relation_id': relation['id'], 'candidate_id': candidate['id'],
                'relation_index': index, 'parent_id': parent['id'],
                'parent_version': parent.get('version', 1), 'parameter_coordinates': 'definition_arguments'})
        # Validate compound operand units using the original declarations, too.
        try:
            _prepare(declaration, call.expanded(), budget)
        except _MissingContract as error:
            if excluded is not None:
                excluded.append({'parent_reference': relation['id'], 'reason': str(error)})
            continue
        for name in relation['target_holes']:
            _spend(budget)
            if (name in holes and holes[name]['output_unit'] == relation['output_unit']
                    and holes[name]['time_contract'] == relation['time_contract']):
                result.setdefault(name, []).append(declaration)
    return result

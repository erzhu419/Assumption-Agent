"""Finite, syntax-derived coordinates for saved Program coefficients.

Paths locate values; the typed computation multiplying a value identifies its
role. These helpers never predict observations or infer a physical equivalence
from equal sample columns. Canonical scaling changes parameter coordinates,
not the stored Program or its floating point evaluation order.
"""

from collections import Counter
from copy import deepcopy
from math import isfinite

from .language import Definition, Program


class _Unknown(Exception):
    pass


def _spend(budget, units=1):
    if not budget.spend(units):
        raise _Unknown('parameter_role_budget_exhausted')


def _program(value):
    return Program.from_dict(value) if isinstance(value, dict) else value


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def role_compatible(expected, actual):
    """Compare computation and units; explicit coordinate scales stay separate."""
    return (expected.get('status') == actual.get('status') == 'known'
            and expected.get('role_key') is not None
            and _freeze(expected['role_key']) == _freeze(actual.get('role_key')))


def _unit(unit):
    if not isinstance(unit, str) or not unit:
        raise _Unknown('explicit_parameter_role_unit_required')
    # A literal "1" remains the source's opaque unit token. No alias to
    # "dimensionless" is inferred from a numerical value or a spelling.
    return () if unit == 'dimensionless' else ((unit, 1),)


def _product(left, right, sign=1):
    powers = Counter(dict(left))
    powers.update({name: sign * exponent for name, exponent in right})
    return tuple(sorted((name, exponent) for name, exponent in powers.items() if exponent))


def _unit_text(unit):
    if not unit:
        return 'dimensionless'
    return '*'.join(name if exponent == 1 else f'{name}^{exponent}' for name, exponent in unit)


def _at(program, path, budget):
    for index in path:
        _spend(budget)
        program = program.children[index]
    return program


def _differentiate(program, selected, budget):
    """A symbolic coefficient for the selected shared scalar, or UNKNOWN."""
    zero, one = Program('const', value=0.), Program('const', value=1.)

    def visit(node, path=()):
        _spend(budget)
        if path in selected:
            if node.op != 'const':
                raise _Unknown('selected_parameter_is_not_a_constant')
            return True, one
        if not any(item[:len(path)] == path for item in selected):
            return False, zero
        parts = [visit(child, (*path, index)) for index, child in enumerate(node.children)]
        dependent = [value[0] for value in parts]
        derivatives = [value[1] for value in parts]
        if not any(dependent):
            return False, zero
        if node.op in {'add', 'sub', 'object_sum'}:
            if node.op == 'add' and not all(dependent):
                return True, derivatives[dependent.index(True)]
            if node.op == 'sub' and dependent == [True, False]:
                return True, derivatives[0]
            return True, Program(node.op, children=tuple(derivatives))
        if node.op == 'mul' and sum(dependent) == 1:
            index = dependent.index(True)
            children = list(node.children)
            children[index] = derivatives[index]
            return True, Program('mul', children=tuple(children))
        if node.op == 'if' and not dependent[0]:
            return True, Program('if', children=(node.children[0], *derivatives[1:]))
        if node.op in {'window_sum', 'event_delta'}:
            return True, Program(node.op, lag=node.lag, children=tuple(derivatives))
        raise _Unknown('nonlinear_or_guard_parameter_role')

    if any(node.op == 'call' for _, node in _paths(program, budget=budget)):
        # Replacing the actual argument before expansion preserves a repeated
        # lexical argument's single scalar coordinate.
        marker = '__parameter_role_argument__'
        if marker in program.required_features():
            raise _Unknown('parameter_role_marker_already_used')
        def replace(node, path=()):
            _spend(budget)
            if path in selected:
                return Program('feature', name=marker)
            return Program(node.op, node.value, node.name, node.lag,
                tuple(replace(child, (*path, index)) for index, child in enumerate(node.children)), node.definition)
        marked = replace(program).expanded()
        selected = {path for path, node in _paths(marked, budget=budget)
                    if node.op == 'feature' and node.name == marker}
        # The selected marker is a scalar, not a measured feature.
        def constants(node, path=()):
            _spend(budget)
            return (one if path in selected else Program(node.op, node.value, node.name, node.lag,
                tuple(constants(child, (*path, index)) for index, child in enumerate(node.children)), node.definition))
        program = constants(marked)
    return visit(program)[1]


def _paths(program, path=(), budget=None):
    if budget is not None:
        _spend(budget)
    yield path, program
    for index, child in enumerate(program.children):
        yield from _paths(child, (*path, index), budget)


def _canonical_basis(program, roles, budget):
    """Return (ordered role key, explicit numeric scale, formal unit)."""
    used = {name for name, _ in program.observed_variables()}
    identities = [_freeze(roles.get(name)) for name in used]
    if len(set(identities)) != len(identities):
        raise _Unknown('ambiguous_observed_variable_roles')

    def variable(node):
        declared = roles.get(node.name, {})
        if any(declared.get(key) is None for key in ('role', 'unit', 'time')):
            raise _Unknown('explicit_variable_role_unit_time_required')
        unit = _unit(declared['unit'])
        return (node.op, _freeze(declared['role']), unit, _freeze(declared['time']), node.lag), unit

    def fixed(node):
        _spend(budget)
        if node.op == 'const':
            return ('const', float(node.value)), None
        if node.op in {'feature', 'lag'}:
            return variable(node)
        if node.op == 'available':
            return ('available', node.lag), ()
        children = [fixed(child) for child in node.children]
        units = [unit for _, unit in children if unit is not None]
        if node.op in {'ge', 'lt', 'add', 'sub', 'min', 'max', 'abs', 'window_sum', 'event_delta'}:
            if units and any(unit != units[0] for unit in units):
                raise _Unknown('incompatible_parameter_basis_units')
            unit = () if node.op in {'ge', 'lt'} else (units[0] if units else None)
        elif node.op in {'and', 'or', 'not'}:
            if any(unit not in (None, ()) for unit in units):
                raise _Unknown('non_boolean_parameter_guard')
            unit = ()
        elif node.op == 'mul':
            unit = _product(children[0][1] or (), children[1][1] or ())
        else:
            raise _Unknown('unsupported_fixed_parameter_basis_operator')
        return (node.op, node.lag, tuple(key for key, _ in children)), unit

    def scaled(key, scale):
        return ('zero',) if scale == 0 else key if scale == 1 else ('scale', scale, key)

    def visit(node):
        _spend(budget)
        if node.op == 'const':
            return ('one',), float(node.value), ()
        if node.op in {'feature', 'lag'}:
            key, unit = variable(node)
            return key, 1., unit
        if node.op == 'if':
            guard, guard_unit = fixed(node.children[0])
            if guard_unit != ():
                raise _Unknown('explicit_boolean_parameter_guard_required')
            parts = [visit(child) for child in node.children[1:]]
        else:
            parts = [visit(child) for child in node.children]
        if node.op == 'mul':
            (left, ls, lu), (right, rs, ru) = parts
            key = right if left == ('one',) else left if right == ('one',) else ('mul', left, right)
            return key, ls * rs, _product(lu, ru)
        if node.op in {'add', 'sub', 'object_sum', 'if'}:
            nonzero = [part for part in parts if part[1] != 0]
            if not nonzero:
                return ('one',), 0., ()
            unit, scale = nonzero[0][2], nonzero[0][1]
            if any(part[2] != unit for part in nonzero):
                raise _Unknown('incompatible_parameter_basis_units')
            if node.op != 'if':
                terms = {}
                for index, (key, value, _) in enumerate(parts):
                    terms[key] = terms.get(key, 0.) + (-value if node.op == 'sub' and index else value)
                terms = [(key, value) for key, value in terms.items() if value]
                if not terms:
                    return ('one',), 0., ()
                scale = terms[0][1]
                if len(terms) == 1:
                    return terms[0][0], scale, unit
                return ('sum', tuple(scaled(key, value / scale) for key, value in terms)), scale, unit
            values = [scaled(key, value / scale) for key, value, _ in parts]
            return (node.op, *((guard,) if node.op == 'if' else ()), tuple(values)), scale, unit
        if node.op in {'window_sum', 'event_delta'}:
            key, scale, unit = parts[0]
            return (node.op, node.lag, key), scale, unit
        raise _Unknown('unsupported_linear_parameter_basis_operator')

    return visit(program)


def _describe(program, paths, roles, output_unit, budget, source_ref):
    result = {'status': 'UNKNOWN', 'parameter_paths': [list(path) for path in paths],
              'source_ref': deepcopy(source_ref)}
    try:
        values = [_at(program, path, budget) for path in paths]
        if not values or any(value.op != 'const' for value in values):
            raise _Unknown('selected_parameter_is_not_a_constant')
        if any(value.value != values[0].value for value in values):
            raise _Unknown('shared_parameter_occurrences_disagree')
        basis = _differentiate(program, set(paths), budget)
        key, scale, unit = _canonical_basis(basis, roles, budget)
        if not isfinite(scale) or scale == 0:
            raise _Unknown('unidentifiable_zero_or_nonfinite_parameter_basis')
        coefficient_unit = _product(_unit(output_unit), unit, -1)
        value = float(values[0].value)
        result.update(status='known', role_key=('linear_coefficient', key, coefficient_unit),
            coordinate_scale=scale, coefficient_unit=_unit_text(coefficient_unit),
            parameter_value=value, canonical_parameter=value * scale,
            scope='syntax_derived_linear_parameter_coordinate')
    except (IndexError, TypeError, KeyError):
        result['reason'] = 'actual_parameter_path_required'
    except _Unknown as error:
        result['reason'] = str(error)
    return result


def describe_parameter_roles(program, parameter_paths, role_signature, output_unit, budget, *, source_ref=None):
    """Describe independently selected actual constant paths without fitting."""
    started = budget.spent
    program, paths = _program(program), [tuple(path) for path in parameter_paths]
    roles = {path: _describe(program, [path], role_signature, output_unit, budget, source_ref) for path in paths}
    return {'roles': roles, 'status': 'known' if all(item['status'] == 'known' for item in roles.values())
            else 'UNKNOWN', 'actual_compute_cost': budget.spent - started}


def describe_bound_parameter_roles(candidate, record, budget):
    """Describe hole coordinates using all actual occurrences of each argument.

    The original patterns supply repeated occurrences; `_match_parent`'s last
    path alone does not identify the basis of a shared scalar. Scalar mechanism
    fixtures without actual Programs must supply explicit role contracts.
    """
    started, results = budget.spent, {}
    parents = set(candidate['parent_holes'])
    patterns = {}
    if candidate.get('before_pattern'):
        patterns['before'] = _program(candidate['before_pattern'])
    if candidate.get('definition'):
        patterns['after'] = Definition.from_dict(candidate['definition']).body
    occurrences, scan_error = {}, None
    try:
        for side, pattern in patterns.items():
            occurrences[side] = {}
            for path, node in _paths(pattern, budget=budget):
                if node.op == 'arg':
                    occurrences[side].setdefault(node.name, []).append(path)
    except _Unknown as error:
        scan_error = str(error)
    for hole, value in record.get('bindings', {}).items():
        value = _program(value)
        if value.op != 'const':
            continue
        side = 'before' if hole in parents else 'after'
        if scan_error:
            results[hole] = {'status': 'UNKNOWN', 'side': side, 'reason': scan_error}
            continue
        program = record.get(f'_{side}_program')
        if program is None:
            declared = deepcopy(record.get('parameter_role_contracts', {}).get(hole, {}))
            try:
                _spend(budget)
                scale = declared.get('coordinate_scale')
                if (declared.get('status') != 'known' or declared.get('role_key') is None
                        or type(scale) not in (int, float) or not isfinite(scale) or scale == 0
                        or not declared.get('coefficient_unit')):
                    raise _Unknown('actual_program_or_explicit_parameter_role_contract_required')
                declared.update(parameter_value=float(value.value), canonical_parameter=float(value.value) * scale,
                                side=side)
                results[hole] = declared
            except _Unknown as error:
                results[hole] = {'status': 'UNKNOWN', 'reason': str(error)}
            continue
        paths = occurrences.get(side, {}).get(hole, [])
        if not paths and hole in record.get(f'_{side}_binding_paths', {}):
            paths = [tuple(record[f'_{side}_binding_paths'][hole])]
        result = _describe(_program(program), paths, record.get('_roles', {}), record.get('_output_unit'),
                           budget, record.get(f'{side}_ref'))
        if result.get('parameter_value') != float(value.value) and result['status'] == 'known':
            result.update(status='UNKNOWN', reason='actual_parameter_binding_mismatch')
        result['side'] = side
        results[hole] = result
    return {'roles': results, 'status': 'known' if all(item['status'] == 'known' for item in results.values())
            else 'UNKNOWN', 'actual_compute_cost': budget.spent - started}


def align_parameter_roles(source_roles, target_roles, budget):
    """Return only uniquely established permutations and coordinate scalings."""
    started, mappings, unknown = budget.spent, [], []
    try:
        for source_path, source in source_roles.items():
            matches = []
            for target_path, target in target_roles.items():
                _spend(budget)
                if role_compatible(source, target):
                    matches.append((target_path, target))
            reverse_count = sum(role_compatible(source, other) for other in source_roles.values())
            if len(matches) != 1 or reverse_count != 1:
                unknown.append({'source_path': source_path,
                    'reason': 'ambiguous_parameter_role' if matches else 'no_matching_parameter_role'})
                continue
            target_path, target = matches[0]
            mappings.append({'source_path': source_path, 'target_path': target_path,
                'parameter_scale': source['coordinate_scale'] / target['coordinate_scale'],
                'role_key': deepcopy(source['role_key'])})
    except _Unknown as error:
        unknown.append({'reason': str(error)})
    return {'status': 'known' if not unknown else 'UNKNOWN', 'mappings': mappings,
            'unknown': unknown, 'actual_compute_cost': budget.spent - started}


def parameter_family_correspondence(before, after, role_signature, output_unit, budget):
    """Recognize only the existing linear-role permutation/scaling fragment.

    A zero remainder ensures that matching coefficient roles cover the whole
    output. Guard constants remain fixed in their basis keys. This is a family
    diagnostic, never a claim that the two fitted instances predict alike.
    """
    from .abstractions import _bind_constants

    started = budget.spent
    result = {'status': 'UNKNOWN', 'mappings': [], 'scope': 'syntax_derived_parameter_family'}
    try:
        descriptions = []
        for program in (_program(before).expanded(), _program(after).expanded()):
            paths = [path for path, node in _paths(program, budget=budget) if node.op == 'const']
            roles = describe_parameter_roles(program, paths, role_signature, output_unit, budget)['roles']
            known = {path: role for path, role in roles.items() if role['status'] == 'known'}
            if any(role.get('reason') == 'parameter_role_budget_exhausted' for role in roles.values()):
                raise _Unknown('parameter_role_budget_exhausted')
            if not known:
                raise _Unknown('complete_linear_parameter_roles_not_established')
            remainder = _bind_constants(program, {path: 0. for path in known})
            _, scale, _ = _canonical_basis(remainder, role_signature, budget)
            if scale != 0.:
                raise _Unknown('unmatched_fixed_output_remainder')
            descriptions.append(known)
        aligned = align_parameter_roles(*descriptions, budget)
        if (aligned['status'] != 'known'
                or len(aligned['mappings']) != len(descriptions[1])):
            result['reason'] = 'unique_complete_parameter_role_correspondence_required'
            result['unknown'] = aligned['unknown']
        else:
            result.update(status='known', mappings=aligned['mappings'])
    except _Unknown as error:
        result['reason'] = str(error)
    finally:
        result['actual_compute_cost'] = budget.spent - started
    return result

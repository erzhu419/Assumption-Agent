from copy import deepcopy

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.parameter_roles import (
    align_parameter_roles, describe_bound_parameter_roles,
    describe_parameter_roles, role_compatible,
)


def c(value):
    return Program('const', value=value)


def op(name, *children, **kwargs):
    return Program(name, children=children, **kwargs)


X = Program('feature', name='x')
ROLES = {'x': {'role': 'controlled_input', 'unit': 'dimensionless', 'time': 'current'}}


def describe(program, paths, *, roles=None, output='dimensionless', budget=None):
    return describe_parameter_roles(program, paths, ROLES if roles is None else roles,
        output, SearchBudget(10000) if budget is None else budget,
        source_ref={'file': 'saved.json', 'json_path': ['program']})


def test_parameter_permutation_preserves_guards_and_history_without_fitting():
    guard = op('ge', X, c(-.5))
    current = op('if', guard, X, c(0))
    history = op('window_sum', X, lag=2)
    before = op('add', op('mul', c(3), current), op('mul', c(7), history))
    after = op('add', op('mul', c(8), history), op('mul', c(4), current))
    source = describe(before, [(0, 0), (1, 0)])
    target = describe(after, [(0, 0), (1, 0)])
    budget = SearchBudget(100)
    aligned = align_parameter_roles(source['roles'], target['roles'], budget)
    assert aligned['status'] == 'known'
    assert [(item['source_path'], item['target_path']) for item in aligned['mappings']] == [
        ((0, 0), (1, 0)), ((1, 0), (0, 0))]
    assert aligned['actual_compute_cost'] == budget.spent == 4
    changed_guard = op('mul', c(3), op('if', op('ge', X, c(-.25)), X, c(0)))
    changed_history = op('mul', c(3), op('window_sum', X, lag=3))
    assert not role_compatible(source['roles'][(0, 0)], describe(changed_guard, [(0,)])['roles'][(0,)])
    assert not role_compatible(source['roles'][(1, 0)], describe(changed_history, [(0,)])['roles'][(0,)])


def test_explicit_basis_scale_transforms_the_parameter_coordinate():
    source = describe(op('mul', c(3), op('mul', c(2), X)), [(0,)])['roles']
    target = describe(op('mul', c(6), X), [(0,)])['roles']
    aligned = align_parameter_roles(source, target, SearchBudget(10))
    assert aligned['status'] == 'known'
    assert aligned['mappings'][0]['parameter_scale'] == 2
    assert source[(0,)]['canonical_parameter'] == target[(0,)]['canonical_parameter'] == 6
    assert source[(0,)]['coordinate_scale'] == 2
    # The source value and ordered Program remain unchanged.
    assert source[(0,)]['parameter_value'] == 3


def test_bound_shared_argument_uses_all_occurrences_not_the_last_path():
    h = Program('arg', name='coefficient')
    pattern = op('add', op('mul', h, X), op('mul', h, X))
    program = op('add', op('mul', c(3), X), op('mul', c(3), X))
    candidate = {'parent_holes': ['coefficient'], 'before_pattern': pattern.to_dict()}
    record = {'bindings': {'coefficient': c(3)}, '_before_program': program,
        '_roles': ROLES, '_output_unit': 'dimensionless',
        '_before_binding_paths': {'coefficient': [1, 0]}, 'before_ref': {'file': 'before.json'}}
    untouched = deepcopy(record)
    role = describe_bound_parameter_roles(candidate, record, SearchBudget(1000))['roles']['coefficient']
    single = describe(op('mul', c(6), X), [(0,)])['roles'][(0,)]
    assert role['status'] == 'known' and role['coordinate_scale'] == 2
    assert role['parameter_paths'] == [[0, 0], [1, 0]]
    assert role['side'] == 'before'
    assert role_compatible(role, single)
    assert role['canonical_parameter'] == single['canonical_parameter'] == 6
    assert record == untouched


def test_lexical_shared_call_argument_and_nonlinear_slots_are_distinct():
    h = Program('arg', name='p')
    definition = Definition('shared', ('p',), op('add', op('mul', h, X), op('mul', h, X)))
    call = Program('call', name='shared', children=(c(3),), definition=definition)
    shared = describe(call, [(0,)])['roles'][(0,)]
    assert shared['status'] == 'known' and shared['coordinate_scale'] == 2
    nonlinear = describe(op('mul', c(3), c(3)), [(0,)])['roles'][(0,)]
    assert nonlinear['status'] == 'known'  # The other literal is actually fixed.
    pattern = op('mul', h, h)
    record = {'bindings': {'p': c(3)}, '_before_program': op('mul', c(3), c(3)),
              '_roles': {}, '_output_unit': 'dimensionless'}
    role = describe_bound_parameter_roles({'parent_holes': ['p'], 'before_pattern': pattern.to_dict()},
                                         record, SearchBudget(100))['roles']['p']
    assert role['status'] == 'UNKNOWN' and role['reason'] == 'nonlinear_or_guard_parameter_role'
    guard_slot = describe(op('if', op('ge', X, c(.5)), X, c(0)), [(0, 1)])['roles'][(0, 1)]
    assert guard_slot['status'] == 'UNKNOWN'


def test_equal_basis_columns_are_ambiguous_not_forced_into_a_permutation():
    program = op('add', op('mul', c(3), X), op('mul', c(7), X))
    roles = describe(program, [(0, 0), (1, 0)])['roles']
    aligned = align_parameter_roles(roles, roles, SearchBudget(100))
    assert aligned['status'] == 'UNKNOWN' and not aligned['mappings']
    assert all(item['reason'] == 'ambiguous_parameter_role' for item in aligned['unknown'])


def test_units_time_and_missing_contracts_do_not_become_sample_equivalence():
    program = op('mul', c(3), X)
    unit_one = {'x': {**ROLES['x'], 'unit': '1'}}
    one = describe(program, [(0,)], roles=unit_one, output='1')['roles'][(0,)]
    dimensionless = describe(program, [(0,)])['roles'][(0,)]
    assert one['status'] == dimensionless['status'] == 'known'
    assert not role_compatible(one, dimensionless)  # No declared alias.
    delayed = {'x': {**ROLES['x'], 'time': 'previous'}}
    assert not role_compatible(dimensionless, describe(program, [(0,)], roles=delayed)['roles'][(0,)])
    missing = describe(program, [(0,)], roles={})['roles'][(0,)]
    assert missing['status'] == 'UNKNOWN'
    assert missing['reason'] == 'explicit_variable_role_unit_time_required'
    budget = SearchBudget(0)
    exhausted = describe(program, [(0,)], budget=budget)
    assert exhausted['status'] == 'UNKNOWN' and exhausted['actual_compute_cost'] == budget.spent == 0


def test_scalar_contract_cannot_override_a_real_program_role():
    declared = {'status': 'known', 'role_key': ['test_coordinate', 'offset'],
                'coordinate_scale': 2., 'coefficient_unit': 'dimensionless'}
    candidate = {'parent_holes': ['p'], 'before_pattern': Program('arg', name='p').to_dict()}
    scalar = {'bindings': {'p': c(3)}, 'parameter_role_contracts': {'p': declared}}
    role = describe_bound_parameter_roles(candidate, scalar, SearchBudget(10))['roles']['p']
    assert role['canonical_parameter'] == 6
    actual = {**scalar, '_before_program': c(3), '_roles': {}, '_output_unit': 'dimensionless'}
    actual_role = describe_bound_parameter_roles(candidate, actual, SearchBudget(10))['roles']['p']
    assert actual_role['status'] == 'known' and actual_role['coordinate_scale'] == 1
    assert not role_compatible(role, actual_role)


def test_unrelated_unbound_after_term_is_ignored_but_an_unbound_basis_is_unknown():
    unresolved = Program('arg', name='unbound')
    after = op('add', op('mul', c(2), X), unresolved)
    role = describe(after, [(0, 0)])['roles'][(0, 0)]
    assert role['status'] == 'known'
    after = op('mul', c(2), unresolved)
    role = describe(after, [(0,)])['roles'][(0,)]
    assert role['status'] == 'UNKNOWN'

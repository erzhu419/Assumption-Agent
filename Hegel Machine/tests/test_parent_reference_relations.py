"""Parent-dependent source contexts survive different open-hole positions.

Small typed Program fixtures exercise the actual positional loss found in the
retained case001/002 source steps. They are integration controls, not scientific
answers or a request to run the historical experiment again.
"""
from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.revision_relations import (
    _Incomplete, consolidate_revision_relations,
)
from hegel_machine.cognition.parent_reference_relations import parent_reference_components


ROLES = {'u': {'role': 'public_control', 'unit': 'dimensionless', 'time': 'current'}}
CONTEXT = {'source_id': 'parent_reference_target', 'reset': 'per_trajectory'}


def const(value):
    return Program('const', value=float(value))


def op(name, *children):
    return Program(name, children=tuple(children))


def feature():
    return Program('feature', name='u')


def at(program, path):
    for index in path:
        program = program.children[index]
    return program


def nodes(program):
    yield program
    for child in program.children:
        yield from nodes(child)


def contracts(program):
    """Fixture dimensions are explicitly declared at every actual occurrence."""
    result = {}

    def visit(node, path=()):
        result['.'.join(map(str, path))] = {
            'output_unit': 'dimensionless', 'time_contract': ['current']}
        for index, child in enumerate(node.children):
            visit(child, (*path, index))

    visit(program)
    return result


def source_records(kind='ge'):
    before = [op('mul', const(2.), feature()), op('add', feature(), const(3.))]
    records = []
    for index, old in enumerate(before):
        context = (op('abs', old) if kind == 'abs' else op(kind, old, const(.25 + .5 * index)))
        # The same research role appears on opposite sides of this product.
        # Positional anti-unification alone turns both contexts into open holes.
        extra = op('mul', context, const(.5)) if index == 0 else op('mul', const(.4), context)
        after = op('add', old, extra)
        reference = {'file': 'retained_parent_reference_fixture.json',
                     'json_path': ['episodes', index]}
        records.append({
            'episode_id': f'parent_source_{index}@meaning1',
            'source_id': f'physical_parent_source_{index}',
            'before_ref': {**reference, 'json_path': [*reference['json_path'], 'before'], 'version': 1},
            'after_ref': {**reference, 'json_path': [*reference['json_path'], 'after'], 'version': 1},
            'source_ref': reference,
            'action_ref': {**reference, 'json_path': [*reference['json_path'], 'action']},
            'before_program': old.to_dict(), 'after_program': after.to_dict(),
            'before_subtree_contracts': contracts(old), 'after_subtree_contracts': contracts(after),
            'role_signature': deepcopy(ROLES), 'output_unit': 'dimensionless',
            'context_ref': {'source_id': f'physical_parent_source_{index}', 'reset': 'per_trajectory'},
            'context_family_ref': 'explicit_parent_reference_fixture',
            'evidence_refs': [], 'cost_refs': [reference], 'failure_refs': [reference],
            'counterexamples': [{
                'source_ref': {**reference, 'json_path': [*reference['json_path'], 'paid_row']},
                'observation': {'features': {'u': 2.}, 'target': 100., 'history': []},
                'tolerance': 0.,
            }],
        })
    return records


def compile_candidate(records=None):
    records = source_records() if records is None else records
    result = consolidate_revision_relations(records, allocation={'compute': 100000},
        candidate_id='parent_reference_fixture', object_type='program_revision_relation')
    candidates = [item for item in result['candidates'] if item['kind'] == 'program_revision_relation']
    assert candidates, result
    return candidates[0], result


def target_parent():
    program = op('sub', op('mul', const(7.), feature()), const(4.))
    return {'id': 'actual_target_parent', 'version': 3, 'program': program.to_dict(),
        'program_source_ref': {'file': 'target_state.json', 'json_path': ['theory_index', 'actual_target_parent', 'program'], 'version': 3},
        'source_ref': {'file': 'target_state.json', 'json_path': ['theory_index', 'actual_target_parent'], 'version': 3},
        'role_signature': deepcopy(ROLES), 'output_unit': 'dimensionless',
        'context_ref': deepcopy(CONTEXT), 'subtree_contracts': contracts(program)}


def references(candidate, kind):
    return [relation for relation in candidate.get('parent_reference_relations', [])
            if Definition.from_dict(relation['definition']).body.op == kind]


@pytest.mark.parametrize('kind', ['ge', 'abs', 'max'])
def test_public_induction_learns_parent_context_across_different_hole_positions(kind):
    records = source_records(kind)
    original = deepcopy(records)
    candidate, result = compile_candidate(records)
    relations = references(candidate, kind)
    assert relations
    relation = relations[0]
    assert relation['parent_argument'] == 'parent_program'
    assert relation['parent_output_unit'] == 'dimensionless'
    assert any(node.op == 'arg' and node.name == relation['parent_argument']
               for node in nodes(Definition.from_dict(relation['definition']).body))
    assert len(set(relation['target_holes'])) >= 2
    assert set(relation['target_holes']).isdisjoint(candidate['parent_holes'])
    assert len(relation['source_bindings']) >= 2
    assert len(relation['source_reconstruction']) >= 2
    assert all(item.get('exact') is True for item in relation['source_reconstruction'])
    assert records == original
    assert not candidate['scientific_support_added'] and not candidate['independent_confirmation']
    assert result['actual_compute_cost'] > 0 and result['new_observations'] == result['llm_calls'] == 0


@pytest.mark.parametrize('kind', ['ge', 'abs', 'max'])
def test_components_use_actual_parent_and_only_new_const_slots_are_fittable(kind):
    candidate, _ = compile_candidate(source_records(kind))
    declaration = target_parent()
    original = deepcopy((candidate, declaration))
    parent = Program.from_dict(declaration['program'])
    work = SearchBudget(100000)
    excluded = []
    mapped = parent_reference_components(candidate, declaration, work, excluded=excluded)
    target_holes = {name for relation in references(candidate, kind) for name in relation['target_holes']}
    assert target_holes <= set(mapped)
    hole_contracts = {item['name']: item for item in candidate['holes']}
    source_parents = [Program.from_dict(row['before_program']) for row in source_records(kind)]
    for name in target_holes:
        assert mapped[name]
        applicable = [component for component in mapped[name]
                      if Program.from_dict(component['program']).expanded().op == kind]
        assert applicable
        for component in applicable:
            program = Program.from_dict(component['program'])
            assert program.op == 'call' and program.children[0] == parent
            assert program.definition.arguments[0] == 'parent_program'
            expanded = program.expanded()
            assert any(node == parent for node in nodes(expanded))
            assert not any(node in source_parents for node in nodes(expanded))
            assert component['parent_reference']['parameter_coordinates'] == 'definition_arguments'
            assert component['parent_reference']['parent_version'] == declaration['version']
            assert component['output_unit'] == hole_contracts[name]['output_unit']
            assert component['context_ref'] == declaration['context_ref']
            paths = [tuple(path) for path in component['fitting_parameter_paths']]
            assert len(paths) == (0 if kind == 'abs' else 1)
            assert all(at(program, path).op == 'const' and at(program, path).value == 0. for path in paths)
            # Every source/target parent constant stays inside a fixed parent
            # operand. Neither its old7 nor old4 may be treated as a free slot.
            assert all(at(program, path).value not in (7., 4.) for path in paths)
    assert (candidate, declaration) == original
    assert work.spent > 0


def test_other_retained_program_is_not_accepted_as_a_reference_to_source_parent():
    records = source_records()
    wrong = op('mul', const(99.), feature())
    old_after = Program.from_dict(records[1]['after_program'])
    altered = op('add', old_after.children[0], op('mul', const(.4), op('ge', wrong, const(.75))))
    records[1]['after_program'] = altered.to_dict()
    records[1]['after_subtree_contracts'] = contracts(altered)
    candidate, _ = compile_candidate(records)
    assert references(candidate, 'ge') == []


def test_single_physical_source_does_not_establish_parent_reference_transfer():
    records = source_records()
    records[1]['source_id'] = records[0]['source_id']
    result = consolidate_revision_relations(records, allocation={'compute': 100000},
        object_type='program_revision_relation')
    assert not any(item.get('parent_reference_relations') for item in result['candidates'])
    assert any(item['reason'] == 'distinct_episode_sources_required' for item in result['excluded'])
    assert not result['support_updated']


def test_missing_source_units_are_unknown_without_installing_defaults():
    records = source_records()
    del records[0]['role_signature']['u']['unit']
    original = deepcopy(records)
    result = consolidate_revision_relations(records, allocation={'compute': 100000},
        object_type='program_revision_relation')
    assert result['status'] == 'UNKNOWN'
    assert not any(item.get('parent_reference_relations') for item in result['candidates'])
    assert any(item['reason'] == 'missing_role_unit_or_time_contract' for item in result['excluded'])
    assert records == original and 'unit' not in records[0]['role_signature']['u']


def test_unfunded_induction_and_component_recovery_preserve_inputs_and_old_candidate():
    records = source_records()
    original_records = deepcopy(records)
    unfunded = consolidate_revision_relations(records, allocation={'compute': 0},
        object_type='program_revision_relation')
    assert unfunded['status'] == 'UNKNOWN' and unfunded['budget_exhausted']
    assert unfunded['actual_compute_cost'] == 0 and not unfunded['support_updated']
    assert records == original_records
    candidate, _ = compile_candidate(records)
    declaration = target_parent()
    original = deepcopy((candidate, declaration))
    work = SearchBudget(0)
    try:
        mapped = parent_reference_components(candidate, declaration, work, excluded=[])
    except _Incomplete:
        mapped = {}
    assert not mapped and work.spent == 0
    assert (candidate, declaration) == original


def test_old_candidate_without_new_reference_field_has_no_new_components_or_cost():
    candidate, _ = compile_candidate()
    candidate.pop('parent_reference_relations', None)
    original = deepcopy(candidate)
    work = SearchBudget(100)
    assert parent_reference_components(candidate, target_parent(), work, excluded=[]) == {}
    assert work.spent == 0 and candidate == original


def test_ordinary_revision_pool_consumes_parent_reference_and_existing_whole_after_fit():
    from hegel_machine.cognition.revision_relations import propose_revision

    candidate, _ = compile_candidate()
    declaration = target_parent()
    parent = Program.from_dict(declaration['program'])
    checks = []
    for index, value in enumerate((-2., -1., 0., 1., 2.)):
        current_value = 7. * value - 4.
        checks.append({
            'source_ref': {'file': 'paid_parent_reference_fixture.json', 'json_path': [index]},
            'context_ref': deepcopy(CONTEXT),
            'observation': {'features': {'u': value}, 'history': [],
                            'target': current_value + float(current_value >= 5.)},
            'tolerance': 0.,
        })
    original = deepcopy((candidate, declaration, checks))
    result = propose_revision(candidate, declaration, [], allocation={'compute': 100000},
        evidence_checks=checks, fitting={'operations': ['refit'], 'risk_quantile': .5,
                                         'max_history': 0, 'max_structures': 1})
    consumed = []
    for proposal in result['proposals']:
        program = Program.from_dict(proposal['program']).expanded()
        if any(node.op == 'ge' and node.children[0] == parent for node in nodes(program)):
            consumed.append(proposal)
            assert program.children[0] == parent
    assert consumed, result
    best = min(consumed, key=lambda item: item['fitting']['training_loss'])
    assert best['fitting']['training_loss'] < 1e-12
    best_program = Program.from_dict(best['program']).expanded()
    assert any(node.op == 'ge' and node.children[0] == parent
               and node.children[1].op == 'const' and node.children[1].value > 0.
               for node in nodes(best_program))
    assert all(item['fitting']['scope'] == 'whole_after_paid_fit_not_confirmation' for item in consumed)
    assert all(not item['support_updated'] and not item['independent_confirmation'] for item in consumed)
    assert (candidate, declaration, checks) == original
    assert 0 < result['actual_compute_cost'] <= 100000


def test_frozen_candidate_and_revision_contract_distinguish_new_parent_relations():
    from hegel_machine.cognition.revision_relations import _revision_contract
    from hegel_machine.cognition.theory_consolidation import _same_frozen_candidate

    revised, _ = compile_candidate()
    positional = deepcopy(revised)
    positional.pop('parent_reference_relations')
    assert not _same_frozen_candidate(positional, revised)
    assert not _same_frozen_candidate(revised, positional)
    assert _revision_contract(positional) != _revision_contract(revised)
    changed = deepcopy(revised)
    changed['parent_reference_relations'][0]['target_holes'] = []
    assert not _same_frozen_candidate(revised, changed)
    assert _revision_contract(revised) != _revision_contract(changed)


def test_assigned_object_and_definition_names_do_not_create_a_new_frozen_relation():
    from hegel_machine.cognition.revision_relations import _revision_contract
    from hegel_machine.cognition.theory_consolidation import _same_frozen_candidate

    original, _ = compile_candidate()
    renamed = deepcopy(original)
    renamed['id'] = 'another_assigned_object_id'
    renamed['definition']['definition_id'] = 'another_assigned_revision_definition'
    for index, relation in enumerate(renamed['parent_reference_relations']):
        relation['id'] = f'another_assigned_parent_context_{index}'
        relation['definition']['definition_id'] = relation['id']
    assert _same_frozen_candidate(original, renamed)
    assert _same_frozen_candidate(renamed, original)
    assert _revision_contract(original) == _revision_contract(renamed)


def test_optional_parent_induction_exhaustion_keeps_reconstructed_positional_baseline(monkeypatch):
    from hegel_machine.cognition import parent_reference_relations
    from hegel_machine.cognition.revision_relations import _spend

    records = source_records()
    original = deepcopy(records)

    def exhaust_optional_search(sources, before, paths, parent_holes, budget, identifier):
        # Exercise the actual bounded caller at the newly added optional stage,
        # after the ordinary positional revision has already been reconstructed.
        _spend(budget, budget.remaining)
        _spend(budget)

    monkeypatch.setattr(parent_reference_relations, 'learn_parent_references', exhaust_optional_search)
    candidate, result = compile_candidate(records)
    assert candidate['parent_reference_search']['status'] == 'UNKNOWN'
    assert candidate['parent_reference_search']['reason'] == 'budget_exhausted'
    assert not candidate.get('parent_reference_relations')
    assert all(item['before_exact'] and item['after_exact'] for item in candidate['source_reconstruction'])
    assert result['actual_compute_cost'] == 100000
    assert records == original and not candidate['scientific_support_added']


def test_equal_new_source_constants_stay_free_and_start_zero_in_current_context():
    records = source_records()
    # Equality in two development episodes is not a target-parameter contract.
    records[1]['after_program']['children'][1]['children'][1]['children'][1]['value'] = .25
    original = deepcopy(records)
    candidate, _ = compile_candidate(records)
    relation, = references(candidate, 'ge')
    assert len(relation['holes']) == 1
    body = Definition.from_dict(relation['definition']).body
    assert body.children[1].op == 'arg'
    mapped = parent_reference_components(candidate, target_parent(), SearchBudget(100000))
    for name in relation['target_holes']:
        components = [item for item in mapped[name]
                      if Program.from_dict(item['program']).expanded().op == 'ge']
        assert components
        for component in components:
            call = Program.from_dict(component['program'])
            assert component['fitting_parameter_paths'] == [[1]]
            assert call.children[1] == const(0.)
    assert records == original


def test_exact_before_contract_recovers_missing_compound_operand_and_threshold_units():
    def typed_scalar_contracts(program):
        declared = {}

        def visit(node, path=(), expected='1'):
            own = 'dimensionless' if node.op == 'ge' else expected
            declared['.'.join(map(str, path))] = {
                'output_unit': own, 'time_contract': ['current']}
            if node.op == 'mul':
                children = ('dimensionless', own)
            elif node.op == 'ge':
                children = ('1', '1')
            else:
                children = (own,) * len(node.children)
            for index, (child, unit) in enumerate(zip(node.children, children)):
                visit(child, (*path, index), unit)

        visit(program)
        return declared

    records = source_records()
    gate_paths = []
    for index, record in enumerate(records):
        before = Program.from_dict(record['before_program'])
        gate = op('ge', before, const(.25 + .5 * index))
        left_coefficient, right_coefficient = (gate, const(.4)) if index == 0 else (const(.5), gate)
        after = op('add', before, op('add', op('mul', left_coefficient, feature()),
                                                  op('mul', right_coefficient, feature())))
        record['after_program'] = after.to_dict()
        record['role_signature']['u']['unit'] = '1'
        record['output_unit'] = '1'
        record['before_subtree_contracts'] = typed_scalar_contracts(before)
        record['after_subtree_contracts'] = typed_scalar_contracts(after)
        gate_path = (1, index, 0)
        gate_paths.append(gate_path)
        for child in (0, 1):
            # This is the real source gap: the gate and its before endpoint
            # are typed, but the compound after operand/threshold are unknown.
            record['after_subtree_contracts'].pop('.'.join(map(str, (*gate_path, child))))
    original = deepcopy(records)
    candidate, _ = compile_candidate(records)
    relation, = references(candidate, 'ge')
    assert relation['parent_output_unit'] == '1'
    assert relation['parent_contract']['output_unit'] == '1'
    assert len(relation['holes']) == 1 and relation['holes'][0]['output_unit'] == '1'
    declaration = target_parent()
    declaration['role_signature']['u']['unit'] = '1'
    declaration['output_unit'] = '1'
    declaration['subtree_contracts'] = typed_scalar_contracts(Program.from_dict(declaration['program']))
    mapped = parent_reference_components(candidate, declaration, SearchBudget(100000))
    assert all(mapped.get(name) for name in relation['target_holes'])
    assert records == original
    for record, path in zip(records, gate_paths):
        assert '.'.join(map(str, (*path, 0))) not in record['after_subtree_contracts']
        assert '.'.join(map(str, (*path, 1))) not in record['after_subtree_contracts']


def test_cached_request_without_parent_revision_objective_recomputes_once_then_reuses(tmp_path):
    from test_retained_revision_triggers import retained_source, build
    from test_theory_consolidation import agent_scene

    sources = [retained_source(tmp_path / f'source_{index}', index) for index in range(2)]
    references = [source[3] for source in sources]
    agent, lab = agent_scene(tmp_path / 'target')
    first = build(agent, lab, references)
    prior = agent.theory_consolidation['executions']['actual_retained_edits']
    assert 'cross_position_parent_reference_context' in prior['request_basis']['revision_objectives']
    # Source file versions, rows, Programs, budget and theory IDs are unchanged;
    # only the already saved older learning objective differs.
    del prior['request_basis']['revision_objectives']
    before = deepcopy(agent.costs)
    second = build(agent, lab, references)
    assert not second['cache_hit'] and second['new_compute_cost'] > 0
    assert agent.costs['compute_cost'] == before['compute_cost'] + second['new_compute_cost']
    assert second['frozen_candidates'] == first['frozen_candidates']
    spent = deepcopy(agent.costs)
    third = build(agent, lab, references)
    assert third['cache_hit'] and third['new_compute_cost'] == 0 and agent.costs == spent
    assert lab.paid == agent.costs['model_calls'] == 0


def test_nested_definition_threshold_slot_remains_shared_across_ge_and_lt_uses():
    from hegel_machine.cognition.abstractions import AbstractTemplate, fit_parameter_slots
    from hegel_machine.cognition.models import Observation

    parent = Program.from_dict(target_parent()['program'])
    parent_arg, theta_arg, current_arg = [Program('arg', name=name)
                                          for name in ('parent', 'theta', 'current')]
    context_definition = Definition('shared_threshold_context', ('parent', 'theta', 'current'),
        op('add', op('ge', parent_arg, theta_arg), op('lt', current_arg, theta_arg)))
    context = Program('call', name=context_definition.definition_id,
        definition=context_definition, children=(parent, const(0.), feature()))
    outer_definition = Definition('nested_context', ('context', 'baseline'),
        op('add', Program('arg', name='baseline'), Program('arg', name='context')))
    pattern = Program('call', name=outer_definition.definition_id,
        definition=outer_definition, children=(context, parent))
    selected = AbstractTemplate('nested_shared_threshold_fixture', 'retained_parent_fixture',
        pattern, (), (), ((0, 1),))
    rows = [Observation({'u': value}, 7. * value - 4.
                        + float(7. * value - 4. >= 5.) + float(value < 5.))
            for value in (-2., -1., 0., 1., 2.)]
    report, budget = {}, SearchBudget(10000)
    fitted = fit_parameter_slots(selected, {}, rows, budget, report=report)
    best = min(fitted, key=report['losses'].__getitem__)
    assert report['losses'][best] < 1e-12
    assert best.children[1] == parent and best.children[0].children[0] == parent
    assert best.children[0].children[2] == feature()
    theta = best.children[0].children[1].value
    assert theta > 0.
    predicates = [node for node in nodes(best.expanded()) if node.op in {'ge', 'lt'}]
    assert len(predicates) == 2
    assert all(node.children[1] == const(theta) for node in predicates)
    assert 'coordinate_trial' in report['candidate_methods']
    assert report['parameter_scope'] == 'declared_const_slots_only'
    assert report['status'] == 'UNKNOWN' and not report['global_optimum_established']
    assert budget.spent == report['actual_compute_cost'] > 0
    assert pattern.children[0].children[1] == const(0.)


def test_completed_fit_reuses_semantic_outer_definition_across_assigned_object_ids(monkeypatch):
    from hegel_machine.cognition import abstractions
    from hegel_machine.cognition import revision_relations as revisions
    from hegel_machine.cognition.models import Observation
    from test_revision_fit_reuse import _linear_template, _retained, _scope

    original = _linear_template()
    rows = [Observation({'u': value}, 2. * value + 7.) for value in (-2., -1., 1., 2.)]
    record = _retained(original, rows, scope=_scope())
    saved = deepcopy(record)
    current = deepcopy(original)
    renamed = Definition('consolidated:20:relation:revision:0',
        original.pattern.definition.arguments, original.pattern.definition.body)
    current.pattern = Program('call', name=renamed.definition_id, definition=renamed,
        children=original.pattern.children)

    def forbidden(*args, **kwargs):
        pytest.fail('renaming an identical outer Definition must reuse the saved fit without executing')

    monkeypatch.setattr(Program, 'predict', forbidden)
    monkeypatch.setattr(abstractions, 'fit_parameter_slots', forbidden)
    monkeypatch.setattr(revisions, '_fit_revision_binding', forbidden)
    work = SearchBudget(10000)
    reused = revisions._reuse_revision_fit(current, rows, .5, _scope(), [record], work)
    assert reused is not None
    fitted, paths, report = reused
    assert paths == current.parameter_paths
    assert report['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
    assert report['actual_compute_cost'] == 0
    assert report['original_compute_cost'] == saved['report']['actual_compute_cost']
    assert report['fit_reuse_ref'] == record['source_ref']
    assert [report['losses'][program] for program in fitted] == [row['training_loss'] for row in record['outputs']]
    for program, output in zip(fitted, record['outputs']):
        assert program.definition == renamed and program.name == renamed.definition_id
        assert program.expanded() == original.instantiate({}, output['parameters']).expanded()
    assert 0 < work.spent < work.limit
    assert record == saved

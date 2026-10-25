"""Inputs for the retained typed-occurrence regression; no fitted world labels.

The historical positions/units below come from existing_subtree_unit_contracts.
The small operator trees exercise the same missing nested-if typing path. They
are unit-contract controls rather than scientific predictions or source data.
"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.revision_relations import (
    _Incomplete, _MissingContract, _occurrence_contracts, _prepare, consolidate_revision_relations,
    instantiate_relation,
)
from hegel_machine.cognition.theory_consolidation import _revision_entries


CONSTRUCTOR_FILE = (
    '/mnt/hegel-snapshot-cow/native_pr3_budgeted_20261021/'
    'construction_budget_diagnosis/revision_query_residuals/shared/theory_library.json')
CONSTRUCTOR_ID = 'consolidated:0:relation:revision:0'


def constant(value):
    return Program('const', value=value)


def feature():
    return Program('feature', name='u')


def binary(op, left, right):
    return Program(op, children=(left, right))


def original_hole_contract(index, unit):
    return {'output_unit': unit, 'time_contract': ['current'],
        'original_hole_contract_ref': {'file': CONSTRUCTOR_FILE,
            'json_path': ['candidates', CONSTRUCTOR_ID, 'holes', index]},
        'actual_component_binding_ref': {'file': 'retained_source.json',
            'json_path': ['consolidation_ref', 'components', f'hole_{index}']}}


@pytest.fixture
def side_scoped_occurrence_inputs():
    # In the observed first conflict the whole conditional is a typed hole_21
    # below a multiplication whose children are both compound expressions.
    predicate = binary('ge', feature(), constant(0.))
    coefficient = Program('if', children=(Program('available', lag=1), predicate, constant(0.)))
    after = binary('mul', coefficient, feature())
    before = feature()
    record = {'source_id': 'unit_fixture_physical_source',
        'role_signature': {'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}},
        'output_unit': '1', 'context_ref': {'source_id': 'unit_fixture_physical_source'},
        'before_ref': {'file': 'retained_source.json', 'json_path': ['before', 'program'], 'version': 1},
        'after_ref': {'file': 'retained_source.json', 'json_path': ['after', 'program'], 'version': 1},
        'before_program': before.to_dict(), 'after_program': after.to_dict(),
        'before_subtree_contracts': {},
        'after_subtree_contracts': {'0': original_hole_contract(21, 'dimensionless')}}
    return {'record': record, 'before': before, 'after': after,
        'expected': {'after_compound_path': (0,), 'after_compound_unit': 'dimensionless',
                     'before_unit': '1'},
        'historical_correspondence': {'expanded_path': [1, 1, 0], 'hole_index': 21}}


@pytest.fixture
def fixed_if_branch_contract_inputs(side_scoped_occurrence_inputs):
    result = deepcopy(side_scoped_occurrence_inputs)
    # The other first conflict is a fixed if whose two argument branches have
    # original hole_8/hole_9 output unit '1'. No constant-unit default is needed.
    left = Program('if', children=(Program('available', lag=1),
        binary('ge', feature(), constant(0.)), constant(0.)))
    right = Program('if', children=(Program('available', lag=1), feature(), feature()))
    result['after'] = binary('mul', left, right)
    result['record']['after_program'] = result['after'].to_dict()
    result['record']['after_subtree_contracts'] = {
        '0': original_hole_contract(19, 'dimensionless'),
        '1.1': original_hole_contract(8, '1'),
        '1.2': original_hole_contract(9, '1')}
    result['expected'] = {'after_compound_path': (1,), 'after_compound_unit': '1',
                          'guard_unit': 'dimensionless', 'before_unit': '1'}
    result['historical_correspondence'] = {'expanded_path': [1, 0, 0, 1],
                                          'hole_indices': [8, 9]}
    return result


def test_side_scoped_declared_contracts_reach_compound_without_changing_before(side_scoped_occurrence_inputs):
    fixture = side_scoped_occurrence_inputs
    record = fixture['record']
    budget = SearchBudget(1000)
    prior = _prepare(record, fixture['before'], budget, side='before')
    after = _prepare(record, fixture['after'], budget, side='after')
    assert prior['contracts'][()]['output_unit'] == '1'
    assert after['contracts'][(0,)]['output_unit'] == 'dimensionless'
    assert 'original_hole_contract_ref' not in after['contracts'][(0,)]
    assert after['contract_recovery']['occurrence_count'] == 1
    assert budget.spent > fixture['before'].size + fixture['after'].size
    # The same explicit declaration cannot override a conflicting whole-output contract.
    bad = deepcopy(record)
    bad['after_subtree_contracts'][''] = {'output_unit': 'dimensionless'}
    with pytest.raises(_MissingContract, match='conflicting_subtree_output_unit'):
        _prepare(bad, fixture['after'], SearchBudget(1000), side='after')


def test_fixed_if_only_inherits_two_explicit_equal_branch_units(fixed_if_branch_contract_inputs):
    fixture = fixed_if_branch_contract_inputs
    result = _prepare(fixture['record'], fixture['after'], SearchBudget(1000), side='after')
    assert result['contracts'][(1,)]['output_unit'] == '1'
    bad = deepcopy(fixture['record'])
    bad['after_subtree_contracts']['1.2']['output_unit'] = 'dimensionless'
    with pytest.raises(_MissingContract, match='conflicting_subtree_output_unit'):
        _prepare(bad, fixture['after'], SearchBudget(1000), side='after')
    # An undeclared constant branch remains unknown; it is not assigned its sibling's unit.
    program = binary('mul', fixture['after'].children[0],
        Program('if', children=(Program('available', lag=1), feature(), constant(0.))))
    missing = deepcopy(fixture['record'])
    missing['after_subtree_contracts'].pop('1.2')
    missing['after_subtree_contracts'].pop('0')  # No independent parent-unit propagation either.
    result = _prepare(missing, program, SearchBudget(1000), side='after')
    assert result['contracts'][(1,)]['output_unit'] is None


def test_occurrence_recovery_keeps_repeated_paths_and_charges_budget():
    pattern = binary('add', Program('arg', name='x'), Program('arg', name='x'))
    program = binary('add', feature(), feature())
    holes = [{'name': 'x', 'output_unit': '1', 'time_contract': ['current']}]
    budget = SearchBudget(100)
    recovered, sources = _occurrence_contracts(pattern, program, holes, budget)
    assert set(recovered) == {'0', '1'} and set(sources) == {'0', '1'}
    assert budget.spent == 2 * pattern.size
    with pytest.raises(_MissingContract, match='parent_repeated_subtree_mismatch'):
        _occurrence_contracts(pattern, binary('add', feature(), constant(0.)), holes, SearchBudget(100))
    with pytest.raises(_Incomplete):
        _occurrence_contracts(pattern, program, holes, SearchBudget(2))


def constructor_fixture(tmp_path, fixture):
    after = fixture['after']
    pattern = binary('mul', Program('arg', name='coefficient'), feature())
    candidate = {'id': 'typed_constructor', 'version': 1, 'kind': 'program_context_relation',
        'definition': Definition('typed_constructor', ('coefficient',), pattern).to_dict(),
        'holes': [{'name': 'coefficient', 'output_unit': 'dimensionless', 'time_contract': ['current']}],
        'output_unit': '1', 'fixed_role_signature': fixture['record']['role_signature'],
        'learned_constraints': [], 'before_pattern': feature().to_dict()}
    reference = {'file': str(tmp_path / 'retained_state.json'),
                 'json_path': ['theory_index', 'actual_after', 'program'], 'version': 1}
    before_ref = {'file': str(tmp_path / 'retained_state.json'), 'json_path': ['native_program']}
    native = {'kind': 'native_candidate', 'question_id': 'q', 'meaning_version': 1,
              'round_index': 0, 'candidate_index': 0}
    transfer = {'theory_ref': {'id': 'actual_after', 'version': 1},
        'binding': {'library_path': 'constructor.json', 'candidate_id': candidate['id'], 'candidate_version': 1,
                    'components': {'coefficient': {'component_id': 'existing_component', 'version': 1}},
                    'parent': {'source_ref': native}}}
    (tmp_path / 'constructor.json').write_text(json.dumps({'candidates': {candidate['id']: candidate}}))
    (tmp_path / 'transfer.json').write_text(json.dumps({'transfers': [transfer]}))
    (tmp_path / 'retained_state.json').write_text(json.dumps({
        'native_program': feature().to_dict(), 'theory_index': {'actual_after': {'program': after.to_dict()}}}))
    endpoint = {'kind': 'frozen_revision_export', 'theory_ref': transfer['theory_ref'],
        'program_ref': reference, 'transfer_ref': {'file': str(tmp_path / 'transfer.json'),
                                                'json_path': ['transfers', 0]}}
    episode = {**deepcopy(fixture['record']), 'episode_id': 'q@meaning1', 'theory_id': 'after',
        'actual_changes': [{'before_ref': before_ref, 'source_ref': reference,
            'fit_state_refs': {'before': {'source_ref': native, 'program_ref': before_ref}, 'after': endpoint}}]}
    agent = SimpleNamespace(workspace=tmp_path, questions=[], revisions=[], theory_index={})
    return candidate, agent, episode, {'after': {'program_source_ref': reference}}


def test_historical_actual_endpoint_recovers_from_original_constructor_in_mainline(tmp_path,
        side_scoped_occurrence_inputs, monkeypatch):
    fixture = side_scoped_occurrence_inputs
    _, agent, episode, pool = constructor_fixture(tmp_path, fixture)
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('metadata typing cannot predict'))
    entry, = _revision_entries(agent, [episode], pool, retained=[])
    budget = SearchBudget(1000)
    before = _prepare(entry, Program.from_dict(entry['before_program']), budget, side='before')
    after = _prepare(entry, Program.from_dict(entry['after_program']), budget, side='after')
    assert before['contract_recovery']['status'] == 'recovered'
    assert after['contract_recovery']['occurrence_count'] == 1
    assert after['contracts'][(0,)]['output_unit'] == 'dimensionless'
    provenance = after['contract_recovery']['sources'][0]
    assert provenance['pattern_ref']['file'] == str(tmp_path / 'constructor.json')
    assert provenance['holes_ref']['json_path'] == ['candidates', 'typed_constructor', 'holes']
    # A before endpoint cannot consume after-path metadata, even if its syntax happens to fit.
    entry['_before_contract_source'] = entry['_after_contract_source']
    with pytest.raises(_MissingContract, match='retained_subtree_contract_endpoint_mismatch'):
        _prepare(entry, fixture['before'], SearchBudget(1000), side='before')


def test_producer_and_common_transfer_preserve_typed_occurrences(tmp_path, side_scoped_occurrence_inputs):
    from test_theory_consolidation import agent_scene
    from hegel_machine.cognition.theory_consolidation import _save
    from hegel_machine.cognition.question_routing import register_theories

    fixture = side_scoped_occurrence_inputs
    candidate, _, _, _ = constructor_fixture(tmp_path, fixture)
    component = {'id': 'coefficient', 'version': 1,
        'program': fixture['after'].children[0].to_dict(), 'output_unit': 'dimensionless',
        'role_signature': fixture['record']['role_signature'], 'context_ref': {'source_id': 'target'}}
    proposal, = instantiate_relation(candidate, {'coefficient': component}, {'compute': 1000})['proposals']
    assert proposal['subtree_contracts']['0']['output_unit'] == 'dimensionless'
    agent, lab = agent_scene(tmp_path / 'current')
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    _save(agent)
    register_theories(agent, [component])
    result = agent._execute('run', {'operator': 'theory_transfer', 'candidate_id': candidate['id'],
        'theory_id': 'concrete', 'component_bindings': {'coefficient': 'coefficient'},
        'role_signature': fixture['record']['role_signature'], 'output_unit': '1',
        'context_ref': {'source_id': 'target'}, 'allocation': {'compute': 1000}}, lab)
    generated, = result['generated_theories']
    declaration = agent.theory_index[generated['theory_id']]
    assert declaration['subtree_contracts'] == proposal['subtree_contracts']
    assert declaration['subtree_contract_sources']['0']['hole_contract_ref']['json_path'] == [
        'candidates', candidate['id'], 'holes', 0]
    assert declaration['support']['status'] == 'candidate'
    assert agent.costs['compute_cost'] == result['new_compute_cost'] > 0
    assert not agent.observations and agent.costs['model_calls'] == 0


def test_pair_feedback_distinguishes_missing_contract_from_explicit_unit_conflict():
    from test_revision_relations import revision
    # Both inputs are typed, but the opaque product has no declared subtree
    # output unit; the literal unit of the compared feature remains explicit.
    coefficient = Program('if', children=(Program('available', lag=1),
        binary('ge', feature(), constant(0.)), constant(0.)))
    records = [revision('a', binary('mul', coefficient, feature())),
               revision('b', binary('mul', constant(2.), feature()))]
    result = consolidate_revision_relations(records, {'compute': 1000},
        object_type='program_revision_relation')
    failure, = result['excluded']
    assert failure['reason'] == 'incompatible_subtree_unit_or_time'
    assert failure['contract_conflict']['side'] == 'after'
    assert failure['contract_conflict']['path'] == [0]
    assert [item['output_unit'] for item in failure['contract_conflict']['contracts']] == [None, 'dimensionless']


def test_declared_compound_dimensionless_factor_preserves_existing_mul_unit_rule(side_scoped_occurrence_inputs):
    fixture = side_scoped_occurrence_inputs
    branch = Program('if', children=(binary('ge', feature(), constant(-.45)),
        Program('window_sum', lag=4, children=(feature(),)), constant(0.)))
    product = binary('mul', fixture['after'].children[0], branch)
    record = deepcopy(fixture['record'])
    record['after_subtree_contracts'] = {
        '0': original_hole_contract(17, 'dimensionless'),
        '1.1': original_hole_contract(3, '1')}
    budget = SearchBudget(1000)
    prepared = _prepare(record, product, budget, side='after')
    assert prepared['contracts'][(0,)]['output_unit'] == 'dimensionless'
    assert prepared['contracts'][(1,)]['output_unit'] == '1'
    assert prepared['contracts'][(1, 2)]['output_unit'] == '1'
    # Both public declaration inspections are actual additional charged work.
    assert budget.spent >= product.size + 2

    for unavailable in ('missing_factor_unit', 'literal_1_is_not_dimensionless', 'missing_parent_unit'):
        source, actual = deepcopy(record), product
        if unavailable == 'missing_factor_unit':
            source['after_subtree_contracts'].pop('0')
            path = (1,)
        elif unavailable == 'literal_1_is_not_dimensionless':
            source['after_subtree_contracts']['0']['output_unit'] = '1'
            path = (1,)
        else:
            # A nested product has no declared own unit: the outer root unit
            # cannot cross its two untyped compound factors.
            actual = binary('mul', product, feature())
            source['after_subtree_contracts'] = {'0.' + key: value
                for key, value in source['after_subtree_contracts'].items()}
            path = (0, 1)
        unknown = _prepare(source, actual, SearchBudget(1000), side='after')
        assert unknown['contracts'][path]['output_unit'] is None, unavailable

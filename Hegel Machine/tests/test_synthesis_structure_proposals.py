"""The old proposal grammar can supply holes without fitting them to world y."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition import synthesis


def rows(*, binary=False, alternate=False):
    return [Observation({'u': float(index), 'v': float(3 - index)},
        float(index > 1) if binary else float(100 + index * 7 if alternate else 2 + index * 3),
        ({'u': float(index - 1), 'v': float(4 - index)},)) for index in range(4)]


def encoded(proposals):
    return [{**item, 'program': item['program'].to_dict()} for item in proposals]


def node_at(program, path):
    for index in path:
        program = program.children[index]
    return program


def test_all_six_old_families_supply_unfitted_structures_without_hole_label_fits(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('An unfitted structure must not fit or score world labels as hole labels')

    monkeypatch.setattr(synthesis, '_fit_linear', forbidden)
    monkeypatch.setattr(synthesis, '_loss', forbidden)
    seed = Program('call', name='retained_coordinate', children=(Program('feature', name='u'),),
        definition=Definition('retained_coordinate', ('coordinate',),
            Program('abs', children=(Program('arg', name='coordinate'),))))
    incumbent = Program('add', children=(Program('lag', name='u', lag=1), Program('const', value=.7)))
    original_seed, original_parent = deepcopy(seed.to_dict()), deepcopy(incumbent.to_dict())
    first_cursor, second_cursor = {}, {}
    arguments = {'operations': synthesis.OPERATIONS, 'seed_programs': (seed,), 'incumbent': incumbent}
    first = list(synthesis.iter_synthesis_structures(rows(), budget=SearchBudget(2000),
                                                   cursor=first_cursor, **arguments))
    second = list(synthesis.iter_synthesis_structures(rows(alternate=True), budget=SearchBudget(2000),
                                                    cursor=second_cursor, **arguments))
    assert encoded(first) == encoded(second)
    assert {item['operation'] for item in first} == set(synthesis.OPERATIONS)
    assert first_cursor['status'] == second_cursor['status'] == 'completed'
    assert [item['operation'] for item in first[:6]] == list(synthesis.OPERATIONS)
    assert any(item['rationale'] == 'compose a learned coordinate with itself' for item in first)
    assert any(item['rationale'] == 'fit an interaction of observed coordinates' for item in first)
    assert any(item['rationale'] == 'test whether current evidence still requires historical state' for item in first)
    assert all(node_at(item['program'], path).op == 'const'
               for item in first for path in item['parameter_paths'])
    assert all(node_at(item['program'], path).value in (0., 1.)
               for item in first for path in item['parameter_paths'])
    assert seed.to_dict() == original_seed and incumbent.to_dict() == original_parent


def test_observed_thresholds_and_existing_parent_constants_are_fixed_not_parameter_slots():
    incumbent = Program('add', children=(Program('feature', name='u'), Program('const', value=.7)))
    structures = list(synthesis.iter_synthesis_structures(rows(), operations=('relation', 'robustification'),
        incumbent=incumbent, budget=SearchBudget(1000)))
    stump = next(item for item in structures if item['operation'] == 'relation')
    assert stump['program'].op == 'if'
    assert stump['parameter_paths'] == ((1,), (2,))
    assert stump['program'].children[0].children[1].value in (.5, 1.5, 2.5)
    residual = next(item for item in structures if item['operation'] == 'robustification')
    assert len(residual['parameter_paths']) == 1
    offset_path, = residual['parameter_paths']
    assert node_at(residual['program'], offset_path).value == 0.
    assert incumbent in residual['program'].children
    assert residual['source_ref'] == {'kind': 'synthesis_structure',
                                     'operation': 'robustification', 'proposal_index': 0}
    binary = list(synthesis.iter_synthesis_structures(rows(binary=True), operations=('relation',),
                                                     budget=SearchBudget(1000)))
    assert binary and all(not item['parameter_paths'] for item in binary)


def test_budgeted_boolean_composition_resumes_the_same_round_robin_from_json_cursor():
    operations = ('refit', 'relation', 'composition', 'robustification')
    full_cursor, resumed_cursor = {}, {}
    full_budget, first_budget, second_budget = SearchBudget(120), SearchBudget(60), SearchBudget(60)
    full = list(synthesis.iter_synthesis_structures(rows(binary=True), operations=operations,
                                                   budget=full_budget, cursor=full_cursor))
    first = list(synthesis.iter_synthesis_structures(rows(binary=True), operations=operations,
                                                    budget=first_budget, cursor=resumed_cursor))
    assert resumed_cursor['status'] == 'UNKNOWN' and resumed_cursor['budget_exhausted']
    restored = json.loads(json.dumps(resumed_cursor))
    second = list(synthesis.iter_synthesis_structures(rows(binary=True), operations=operations,
                                                     budget=second_budget, cursor=restored))
    assert encoded(first + second) == encoded(full)
    assert full_budget.spent == first_budget.spent + second_budget.spent == 120
    keys = [(item['operation'], item['proposal_index']) for item in first + second]
    assert len(keys) == len(set(keys))
    assert any(item['rationale'] == 'compose observed Boolean relations' for item in second)
    assert restored['positions'] == full_cursor['positions']
    assert 'features' not in restored and 'target' not in restored


def test_public_term_preparation_resumes_without_repaying_or_skipping_it():
    state = {}
    limited = SearchBudget(4)
    assert list(synthesis.iter_synthesis_structures(rows(), operations=('refit',),
                                                   budget=limited, cursor=state)) == []
    assert limited.spent == 4 and len(state['thresholds_by_term']) == 1
    assert state['status'] == 'UNKNOWN'
    restored = json.loads(json.dumps(state))
    resumed = list(synthesis.iter_synthesis_structures(rows(), operations=('refit',),
                                                      budget=SearchBudget(100), cursor=restored))
    whole = list(synthesis.iter_synthesis_structures(rows(), operations=('refit',), budget=SearchBudget(104)))
    assert encoded(resumed) == encoded(whole)
    assert restored['status'] == 'completed' and not restored['budget_exhausted']
    empty_budget = SearchBudget(0)
    assert list(synthesis.iter_synthesis_structures(rows(), operations=('refit',),
        budget=empty_budget, cursor=restored)) == []
    assert empty_budget.spent == 0


def test_unknown_operation_is_rejected_by_the_same_grammar_contract():
    with pytest.raises(ValueError, match='unknown cognitive operations'):
        list(synthesis.iter_synthesis_structures(rows(), operations=('new_scientific_rule',),
                                                budget=SearchBudget(100)))

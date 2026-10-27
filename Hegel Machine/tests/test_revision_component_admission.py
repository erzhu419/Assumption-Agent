"""Bare retained prediction Programs do not declare typed revision values."""
from copy import deepcopy

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition.models import Observation, SearchBudget
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import component, constant, edit_candidate, feature


def bare_predictions():
    return [{'id': 'frozen_remove', 'program': feature().to_dict()},
            {'id': 'frozen_keep', 'program': constant(3.).to_dict()}]


def test_bare_predictions_remain_structural_seeds_but_not_typed_hole_values(monkeypatch):
    from hegel_machine.cognition import synthesis

    components = [*bare_predictions(), component('declared', constant(7.))]
    original = deepcopy(components)
    seeds = []

    def grammar(rows, *, seed_programs, cursor, **kwargs):
        seeds.extend(seed_programs)
        cursor['status'] = 'completed'
        return iter(())

    monkeypatch.setattr(synthesis, 'iter_synthesis_structures', grammar)
    state, excluded = {}, []
    parent = component('parent', feature())
    pools = revisions._revision_fitting_pools({}, parent, components,
        [{'name': 'open', 'output_unit': 'V'}], [Observation({'u': 2.}, 0.)],
        FITTING, SearchBudget(100), state, excluded=excluded)
    assert [value['id'] for value in pools[0]] == ['declared']
    assert seeds == [revisions._program(item['program']).expanded() for item in original]
    assert {item['component_id'] for item in excluded} == {'frozen_remove', 'frozen_keep'}
    assert all(item['status'] == 'UNKNOWN' and item['scope'] == 'typed_revision_pool_only'
               and item['reason'] == 'explicit_roles_output_unit_and_context_required' for item in excluded)
    assert all('program' not in item for item in state['excluded_components'])
    # The old prediction roster and its measured values stay intact. Resuming
    # preparation does not multiply an unchanged missing-contract record.
    revisions._revision_fitting_pools({}, parent, components,
        [{'name': 'open', 'output_unit': 'V'}], [Observation({'u': 2.}, 0.)],
        FITTING, SearchBudget(100), state, excluded=[])
    assert len(state['excluded_components']) == 2 and components == original
    assert [revisions._program(item['program']).predict(Observation({'u': 2.}, 0.))
            for item in components[:2]] == [2., 3.]


def test_an_existing_explicit_component_declaration_supplies_its_own_contract(monkeypatch):
    from hegel_machine.cognition import synthesis

    monkeypatch.setattr(synthesis, 'iter_synthesis_structures', lambda *args, **kwargs: iter(()))
    bare = bare_predictions()[0]
    declared = component('declared_copy', revisions._program(bare['program']))
    state = {}
    pool = revisions._revision_fitting_pools({}, component('parent', feature()), [bare, declared],
        [{'name': 'open', 'output_unit': 'V'}], [Observation({'u': 2.}, 0.)],
        FITTING, SearchBudget(100), state)[0]
    assert pool == [declared]
    assert pool[0]['output_unit'] == 'V' and 'output_unit' not in bare
    assert state['excluded_components'][0]['component_id'] == 'frozen_remove'


def test_real_revision_preparation_quotes_without_fitting_or_losing_bare_predictions(monkeypatch):
    parent = component('parent', feature())
    components = bare_predictions()
    original = deepcopy((parent, components))
    monkeypatch.setattr(revisions, '_fit_revision_binding',
        lambda *args, **kwargs: pytest.fail('preparation must not run or repeat a paid fit'))
    prepared = revisions.prepare_revision_visit(edit_candidate(), parent, components,
        {'compute': 5000, 'comparison_rows': 4}, evidence_checks=examples(feature()),
        fitting={**FITTING, 'operations': ['refit'], 'max_structures': 1})
    assert prepared['prepared_visit'] is not None and prepared['quote']['compute'] > 0
    frame = prepared['prepared_visit']['_generator'].gi_frame.f_locals
    exclusions = frame['result']['excluded']
    assert {item['component_id'] for item in exclusions} == {'frozen_remove', 'frozen_keep'}
    assert all(item['status'] == 'UNKNOWN' for item in exclusions)
    assert (parent, components) == original
    prepared['prepared_visit']['_generator'].close()

"""An already-started revision gets one priced, fully checked finite fit."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import revision_relations as revisions
from test_revision_context_fitting import FITTING, examples
from test_revision_fit_reuse import _retained
from test_revision_relations import binary, component, constant, edit_candidate, feature


def _scene():
    parent = binary('mul', constant(5.), feature())
    declaration = component('parent', parent)
    candidate = edit_candidate()
    kwargs = {'evidence_checks': examples(parent), 'fitting': FITTING}
    finished = revisions.propose_revision(candidate, declaration, [], {'compute': 5000}, **kwargs)
    cursor = deepcopy(finished['cursor'])
    cursor['pending_fits'] = [
        {'indices': [0, 0], 'position': {'layer': 0, 'offset': 1}},
        {'indices': [1, 0], 'position': {'layer': 1, 'offset': 1}}]
    return candidate, declaration, kwargs, cursor, finished


def test_preparation_is_paid_once_and_only_selected_pending_fit_is_completed(monkeypatch):
    candidate, parent, kwargs, cursor, baseline = _scene()
    original = deepcopy(cursor)
    prepared = revisions.prepare_revision_visit(candidate, parent, [],
        {'compute': 5000, 'comparison_rows': 4}, cursor=cursor, **kwargs)
    assert prepared['actual_compute_cost'] > 0
    assert prepared['cursor'] == cursor == original
    quote = prepared['quote']
    assert quote['assembly_compute'] > 0
    assert quote['compute'] == (quote['fit_compute'] + quote['assembly_compute']
                                + quote['reserved_comparison_compute'])

    def repeated(*args, **kw):
        pytest.fail('execution must reuse prepared parent, pools and saved-fit lookup')

    monkeypatch.setattr(revisions, '_revision_fitting_pools', repeated)
    monkeypatch.setattr(revisions, '_parameter_components', repeated)
    monkeypatch.setattr(revisions, '_reuse_revision_fit', repeated)
    result = revisions.execute_revision_visit(prepared['prepared_visit'], {'compute': quote['compute']})
    assert result['completed_research_unit']
    assert result['attempted_bindings'] == 1
    assert result['cursor']['pending_fits'] == original['pending_fits'][1:]
    assert (result['cursor']['layer'], result['cursor']['offset']) == (cursor['layer'], cursor['offset'])
    assert result['actual_compute_cost'] + result['reserved_comparison_compute'] <= quote['compute']
    assert [item['program'] for item in result['proposals']] == [
        item['program'] for item in baseline['proposals'][:3]]
    assert not result['support_updated'] and not result['independent_confirmation']


def test_short_allocation_does_not_run_fit_or_advance_cursor_and_can_use_same_packet(monkeypatch):
    candidate, parent, kwargs, cursor, _ = _scene()
    prepared = revisions.prepare_revision_visit(candidate, parent, [],
        {'compute': 5000, 'comparison_rows': 4}, cursor=cursor, **kwargs)
    original_fit, calls = revisions._fit_revision_binding, []

    def fit(*args, **kw):
        calls.append(True)
        return original_fit(*args, **kw)

    monkeypatch.setattr(revisions, '_fit_revision_binding', fit)
    quote = prepared['quote']['compute']
    denied = revisions.execute_revision_visit(prepared['prepared_visit'], {'compute': quote - 1})
    assert denied['reason'] == 'complete_research_unit_unaffordable'
    assert denied['actual_compute_cost'] == 0 and denied['cursor'] == cursor and not calls
    completed = revisions.execute_revision_visit(prepared['prepared_visit'], {'compute': quote})
    assert completed['completed_research_unit'] and len(calls) == 1


def test_complete_retained_linear_fit_is_resolved_during_prepare_and_never_refit(monkeypatch):
    candidate, parent, kwargs, cursor, _ = _scene()
    first = revisions.prepare_revision_visit(candidate, parent, [], {'compute': 5000}, cursor=cursor, **kwargs)
    template = first['prepared_visit']['_generator'].gi_frame.f_locals['template']
    rows = first['prepared_visit']['_generator'].gi_frame.f_locals['rows']
    record = _retained(template, rows, scope={key: parent[key] for key in
                                           ('role_signature', 'output_unit', 'context_ref')})
    # The first packet was only preparation; this actual saved solve supplies
    # the same initial/ridge/stable group to a separate recovery.
    first['prepared_visit']['_generator'].close()
    state = {}
    prepared = revisions.prepare_revision_visit(candidate, parent, [],
        {'compute': 5000, 'comparison_rows': 4}, cursor=cursor,
        retained_fits=[record], retained_fit_state=state, **kwargs)
    assert prepared['quote']['fit_compute'] == 0 and 'index' in state
    monkeypatch.setattr(revisions, '_fit_revision_binding',
                        lambda *args, **kw: pytest.fail('complete old solve must remain a reference'))
    monkeypatch.setattr(revisions, '_reuse_revision_fit',
                        lambda *args, **kw: pytest.fail('saved-fit lookup must not run twice'))
    result = revisions.execute_revision_visit(prepared['prepared_visit'],
                                               {'compute': prepared['quote']['compute']})
    assert result['completed_research_unit'] and result['retained_fit_reuse_count'] == 1
    assert all(item['fitting']['fit_reuse_ref'] for item in result['proposals'])
    assert result['cursor']['pending_fits'] == cursor['pending_fits'][1:]


def test_exhausted_frontier_without_pending_does_not_fit_or_create_a_binding(monkeypatch):
    candidate, parent, kwargs, cursor, _ = _scene()
    cursor['pending_fits'] = []
    original = deepcopy(cursor)
    monkeypatch.setattr(revisions, '_fit_revision_binding', lambda *args, **kw: pytest.fail('exhausted frontier'))
    result = revisions.prepare_revision_visit(candidate, parent, [], {'compute': 5000}, cursor=cursor, **kwargs)
    assert result['prepared_visit'] is None and result['quote'] is None
    assert cursor == original
    assert {key: result['result']['cursor'][key] for key in ('layer', 'offset', 'pending_fits')} == {
        key: original[key] for key in ('layer', 'offset', 'pending_fits')}
    assert result['result']['search_completed']
    assert result['result']['attempted_bindings'] == 0


@pytest.mark.parametrize('binding_index', range(5))
def test_complete_quote_covers_existing_generated_shapes_and_all_output_checks(binding_index):
    candidate, parent, kwargs, cursor, _ = _scene()
    cursor['pending_fits'] = [{'indices': [binding_index, 0],
                               'position': {'layer': binding_index, 'offset': 1}}]
    prepared = revisions.prepare_revision_visit(candidate, parent, [],
        {'compute': 5000, 'comparison_rows': 4}, cursor=cursor, **kwargs)
    if prepared['prepared_visit'] is None:
        assert prepared['result']['completed_research_unit']
        assert prepared['result']['cursor']['pending_fits'] == []
        return
    quote = prepared['quote']
    result = revisions.execute_revision_visit(prepared['prepared_visit'], {'compute': quote['compute']})
    assert not result['budget_exhausted']
    assert result['completed_research_unit']
    assert result['cursor']['pending_fits'] == []
    assert len(result['proposals']) <= quote['maximum_outputs']
    assert result['actual_compute_cost'] + result['reserved_comparison_compute'] <= quote['compute']


@pytest.mark.parametrize('frontier_exhausted', [False, True])
def test_last_pending_completion_keeps_whole_search_open_until_old_frontier_exhausted(frontier_exhausted):
    candidate, parent, kwargs, cursor, _ = _scene()
    cursor['pending_fits'] = cursor['pending_fits'][:1]
    if not frontier_exhausted:
        cursor.update(layer=0, offset=1)
    prepared = revisions.prepare_revision_visit(candidate, parent, [],
        {'compute': 5000, 'comparison_rows': 4}, cursor=cursor, **kwargs)
    result = revisions.execute_revision_visit(prepared['prepared_visit'],
                                               {'compute': prepared['quote']['compute']})
    assert result['completed_research_unit']
    assert result['cursor']['pending_fits'] == []
    assert result['frontier_exhausted'] is frontier_exhausted
    assert result['search_completed'] is frontier_exhausted
    assert (result['cursor']['layer'], result['cursor']['offset']) == (cursor['layer'], cursor['offset'])


@pytest.mark.parametrize('retained_grammar', [False, True])
def test_retained_mixed_roster_only_supplies_native_programs_to_fitting(monkeypatch, retained_grammar):
    from hegel_machine.cognition.theory_consolidation import (
        _relation_request_basis, _relation_transfer_components,
    )

    candidate = {**edit_candidate(), 'version': 1}
    parent_program = binary('mul', constant(5.), feature())
    parent = component('parent', parent_program)
    typed = [component('old_coordinate', feature(), version=3),
             component('old_constant', constant(1.), version=7)]
    opaque = {'id': 'python_forecast:native_public_response@1:0', 'version': 8,
        'forecast_ref': {'file': 'forecasts_0008.json', 'json_path': ['forecasts', 0],
                         'model': 'retained_old_python.py'},
        'role_signature': deepcopy(parent['role_signature']),
        'output_unit': parent['output_unit'], 'context_ref': deepcopy(parent['context_ref'])}
    roster = [typed[0], opaque, typed[1]]
    original = deepcopy(roster)
    arguments = {'_retained_components': roster, '_retained_parent': parent,
        'component_theory_ids': [item['id'] for item in roster]}
    # Retained execution keeps the original mixed roster, rather than borrowing
    # newer live declarations or deleting its Python comparison entries.
    agent = SimpleNamespace(questions=[], theory_index={
        typed[0]['id']: component(typed[0]['id'], constant(99.), version=4)})
    context = _relation_transfer_components(agent, arguments, candidate)
    basis = _relation_request_basis(context['parent'], context['components'], arguments,
                                    candidate, 'retained_library.json')
    assert context['components'] == roster
    assert [(item['id'], item['version']) for item in basis['components']] == [
        (item['id'], item['version']) for item in roster]

    monkeypatch.setattr(revisions, '_fit_revision_binding',
                        lambda *a, **kw: pytest.fail('preparation must not fit'))
    kwargs = {'evidence_checks': examples(parent_program),
              'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}
    cursor = {'layer': 0, 'offset': 0, 'pending_fits': []}
    baseline = revisions.prepare_revision_visit(candidate, parent, typed,
        {'compute': 5000}, cursor=cursor, **kwargs)
    if retained_grammar:
        state = baseline['prepared_visit']['_generator'].gi_frame.f_locals['next_cursor']['fitting']
        kwargs['grammar_preparation'] = {'source_ref': {
            'file': 'retained_grammar.json', 'json_path': ['fitting']},
            'state': {key: deepcopy(state[key]) for key in ('structures', 'grammar_cursor')}}
        baseline['prepared_visit']['_generator'].close()
        baseline = revisions.prepare_revision_visit(candidate, parent, typed,
            {'compute': 5000}, cursor=cursor, **kwargs)
    actual = revisions.prepare_revision_visit(candidate, context['parent'], context['components'],
        {'compute': 5000}, cursor=cursor, **kwargs)
    try:
        assert actual['prepared_visit'] is not None
        assert actual['quote'] == baseline['quote']
        assert actual['actual_compute_cost'] == baseline['actual_compute_cost']
        assert actual['search_position'] == baseline['search_position']
        assert actual['_quote_context']['pools'] == baseline['_quote_context']['pools']
        assert [item['id'] for item in actual['_quote_context']['pools'][0]][:2] == [
            item['id'] for item in typed]
        assert actual['cursor'] == baseline['cursor'] == cursor
        assert roster == original and 'program' not in opaque
        assert agent.theory_index[typed[0]['id']]['version'] == 4
        assert basis == _relation_request_basis(context['parent'], context['components'], arguments,
                                                candidate, 'retained_library.json')
    finally:
        baseline['prepared_visit']['_generator'].close()
        if actual['prepared_visit'] is not None:
            actual['prepared_visit']['_generator'].close()

"""Fresh parent research prices real frontier work without borrowing a cursor."""
from copy import deepcopy

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.language import Program
from test_complete_revision_visit import _scene
from test_theory_consolidation import agent_scene, assert_no_new_science


def test_fresh_quote_prepares_once_and_admits_only_one_actual_frontier_binding(monkeypatch):
    candidate, parent, kwargs, _, baseline = _scene()
    prepared = revisions.prepare_revision_visit(candidate, parent, [],
        {'compute': 5000, 'comparison_rows': 4}, **kwargs)
    assert prepared['cursor'] == {'layer': 0, 'offset': 0}
    assert prepared['search_position'] == {'layer': 0, 'offset': 1}
    assert prepared['quote']['fit_compute'] > 0
    fit, calls = revisions._fit_revision_binding, []

    def counted(*args, **kw):
        calls.append(True)
        return fit(*args, **kw)

    monkeypatch.setattr(revisions, '_fit_revision_binding', counted)
    monkeypatch.setattr(revisions, '_revision_fitting_pools',
        lambda *args, **kw: pytest.fail('the prepared finite pool must be reused'))
    denied = revisions.execute_revision_visit(prepared['prepared_visit'],
        {'compute': prepared['quote']['compute'] - 1})
    assert not calls and denied['cursor'] == {'layer': 0, 'offset': 0}
    assert denied['reason'] == 'complete_research_unit_unaffordable'
    assert 'pending_fits' not in denied['cursor']
    result = revisions.execute_revision_visit(prepared['prepared_visit'],
        {'compute': prepared['quote']['compute']})
    assert len(calls) == 1 and result['completed_research_unit']
    assert result['attempted_bindings'] == 1
    assert result['cursor']['layer'] == 0 and result['cursor']['offset'] == 1
    assert result['cursor']['pending_fits'] == []
    assert not result['frontier_exhausted'] and not result['search_completed']
    assert [x['program'] for x in result['proposals']] == [
        x['program'] for x in baseline['proposals'][:3]]


def test_fresh_second_visit_advances_original_frontier_without_fabricating_pending():
    candidate, parent, kwargs, _, _ = _scene()
    positions, cursor = [], None
    for _ in range(2):
        prepared = revisions.prepare_revision_visit(candidate, parent, [],
            {'compute': 5000}, cursor=cursor, **kwargs)
        positions.append(prepared['search_position'])
        result = revisions.execute_revision_visit(prepared['prepared_visit'],
            {'compute': prepared['quote']['compute']})
        cursor = result['cursor']
        assert result['attempted_bindings'] == 1
        assert cursor['pending_fits'] == []
        assert result['completed_research_unit']
    assert positions == [{'layer': 0, 'offset': 1}, {'layer': 1, 'offset': 1}]


def fresh_recall_scene(tmp_path, *, first_kind=None):
    candidate, parent, kwargs, _, _ = _scene()
    parent.pop('source_ref')
    candidate = {**candidate, 'version': 1}
    agent, lab = agent_scene(tmp_path)
    register_theories(agent, [parent])
    candidates = {}
    if first_kind:
        first = deepcopy(candidate)
        first['id'] = 'first'
        if first_kind == 'not_applicable':
            first['before_pattern'] = Program('const', value=0.).to_dict()
        else:
            first['definition']['body'] = Program('abs', children=(
                Program.from_dict(first['definition']['body']),)).to_dict()
        candidates[first['id']] = first
    candidates[candidate['id']] = candidate
    agent.theory_consolidation = {'candidates': candidates}
    consolidation._save(agent)
    for check in kwargs['evidence_checks']:
        check['context_ref'] = deepcopy(parent['context_ref'])
    target = {'parent_theory_id': parent['id'], **{k: deepcopy(parent[k]) for k in
        ('role_signature', 'output_unit', 'context_ref')},
        'evidence_checks': deepcopy(kwargs['evidence_checks'])}
    question = agent._open_question({'id': 'fresh_parent', 'question': 'Revise this parent.',
        'context_ref': deepcopy(parent['context_ref']), 'theory_target': target})
    args = {'allocation': {'compute': 5000, 'comparison_rows': 4},
        'fitting': kwargs['fitting'], '_retained_execution_inputs': {identifier: {
            '_retained_parent': deepcopy(parent), '_retained_components': [],
            'component_theory_ids': []} for identifier in candidates}}
    if first_kind == 'cached_complete':
        inputs = {**target, 'candidate_id': 'first', 'component_theory_ids': [],
            'fitting': kwargs['fitting'], 'allocation': {'comparison_rows': 4}}
        key = f'relation_transfer:frozen_recall:{question["id"]}@1:first'
        agent.theory_consolidation['executions'] = {key: {
            'request_basis': consolidation._relation_request_basis(parent, [], inputs,
                candidates['first'], consolidation.LIBRARY_PATH),
            'allowance': 5000, 'search_completed': True,
            'cursor': {'layer': 100, 'offset': 0, 'pending_fits': []},
            'result': {'status': 'completed', 'generated_theories': [],
                'new_compute_cost': 0, 'search_completed': True}}}
    return agent, lab, question, args, candidate


@pytest.mark.parametrize('first_kind', [None, 'cached_complete', 'not_applicable'])
def test_recall_routes_fresh_fitting_to_one_complete_unit_and_skips_non_work(
        tmp_path, monkeypatch, first_kind):
    agent, lab, question, args, candidate = fresh_recall_scene(tmp_path, first_kind=first_kind)
    fit, calls = revisions._fit_revision_binding, []

    def counted(*a, **kw):
        calls.append(True)
        return fit(*a, **kw)

    monkeypatch.setattr(revisions, '_fit_revision_binding', counted)
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert len(calls) == 1
    visits = result['complete_research_visits']
    assert visits[-1]['candidate_id'] == candidate['id']
    assert visits[-1]['status'] == 'complete_visit_executed'
    if first_kind:
        assert visits[0]['status'] == ('cached_complete' if first_kind == 'cached_complete'
                                       else 'no_applicable_unit')
        if first_kind == 'not_applicable':
            assert visits[0]['reason'] == 'before_pattern_does_not_match_parent'
    key = f'relation_transfer:frozen_recall:{question["id"]}@1:{candidate["id"]}'
    cursor = agent.theory_consolidation['executions'][key]['cursor']
    assert cursor['layer'] == 0 and cursor['offset'] == 1 and not cursor['pending_fits']
    assert_no_new_science(agent, lab)


def test_recall_fresh_unaffordable_quote_never_calls_fit_or_saves_fake_pending(tmp_path, monkeypatch):
    agent, lab, question, args, candidate = fresh_recall_scene(tmp_path)
    original_quote = revisions._revision_visit_quote

    def unaffordable(*a, **kw):
        quote = original_quote(*a, **kw)
        return {**quote, 'compute': 10000}

    monkeypatch.setattr(revisions, '_revision_visit_quote', unaffordable)
    monkeypatch.setattr(revisions, '_fit_revision_binding',
        lambda *a, **kw: pytest.fail('fresh partial fitting is not admitted'))
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert result['complete_research_visits'][0]['status'] == 'contract_unaffordable'
    key = f'relation_transfer:frozen_recall:{question["id"]}@1:{candidate["id"]}'
    assert key not in agent.theory_consolidation.get('executions', {})
    assert candidate['id'] in result['pending_object_ids']
    assert_no_new_science(agent, lab)


def test_fresh_incomplete_preparation_retains_paid_construction_cursor_without_fake_fit(
        tmp_path, monkeypatch):
    agent, lab, question, args, candidate = fresh_recall_scene(tmp_path)

    def incomplete(candidate, parent, components, holes, rows, fitting, budget, state, **kw):
        assert budget.spend(1)
        state['prepared_items'] = 3
        raise revisions._Incomplete

    monkeypatch.setattr(revisions, '_revision_fitting_pools', incomplete)
    monkeypatch.setattr(revisions, '_fit_revision_binding',
        lambda *a, **kw: pytest.fail('preparation is not fitting'))
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert result['complete_research_visits'][0]['status'] == 'preparation_incomplete'
    key = f'relation_transfer:frozen_recall:{question["id"]}@1:{candidate["id"]}'
    cursor = agent.theory_consolidation['executions'][key]['cursor']
    assert cursor['fitting']['prepared_items'] == 3
    assert (cursor['layer'], cursor['offset']) == (0, 0)
    assert 'pending_fits' not in cursor
    assert result['new_compute_cost'] == agent.costs['compute_cost']
    assert_no_new_science(agent, lab)

"""A quoted complete visit can exceed a fair slice without claiming a gain."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.research_resources import allocate_complete_research
from hegel_machine.cognition import theory_consolidation as consolidation
from test_retained_recall_budget_preparation import retained_scene


def test_allocation_preserves_full_minimum_and_other_resource_caps():
    remaining = {'compute_cost': 100, 'observation_cost': 8}
    minimum = {'compute_cost': 60, 'observation_cost': 0}
    pending = [{'compute_cost': 60, 'observation_cost': 4},
               {'compute_cost': 120, 'observation_cost': 0}]
    assert allocate_complete_research(minimum, pending, remaining) == {
        'compute_cost': 60, 'observation_cost': 4}
    assert allocate_complete_research({**minimum, 'compute_cost': 101}, pending, remaining) is None
    assert remaining == {'compute_cost': 100, 'observation_cost': 8}


def pending_scene(tmp_path, identifiers=('unquoted_old', 'pending_old', 'pending_new')):
    agent, lab, question, args = retained_scene(tmp_path, identifiers)
    executions = agent.theory_consolidation.setdefault('executions', {})
    key = f'relation_transfer:frozen_recall:{question["id"]}@1:{identifiers[0]}'
    executions[key] = {'search_completed': True, 'result': {'status': 'completed'}}
    for identifier in identifiers[1:]:
        key = f'relation_transfer:frozen_recall:{question["id"]}@1:{identifier}'
        executions[key] = {'cursor': {'layer': 2, 'offset': 2,
            'pending_fits': [{'indices': [1, 5], 'position': {'layer': 1, 'offset': 5}}]},
            'search_completed': False, 'result': {'minimum_compute_for_progress': 100000}}
    consolidation._save(agent)
    return agent, lab, question, args


def install_complete_visits(monkeypatch, calls, *, quotes=None, preparation=3):
    def transfer(agent, inputs, laboratory=None):
        identifier = inputs['candidate_id']
        if inputs.get('_prepare_complete_visit'):
            key = f'relation_transfer:{inputs["theory_id"]}'
            if agent.theory_consolidation['executions'].get(key, {}).get('search_completed'):
                return {'cache_hit': True, 'status': 'completed', 'new_compute_cost': 0}
            calls.append(('prepare', identifier, inputs['allocation']['compute']))
            agent.costs['compute_cost'] += preparation
            packet = {'candidate_id': identifier}
            return {'prepared_visit': packet, 'quote': {
                'compute': (quotes or {}).get(identifier, 40), 'maximum_outputs': 1},
                'new_compute_cost': preparation, 'actual_compute_cost': preparation,
                'search_position': {'layer': 1, 'offset': 5},
                '_prepared_relation_visit': packet}
        assert inputs['_prepared_relation_visit']['candidate_id'] == identifier
        calls.append(('execute', identifier, inputs['allocation']['compute']))
        agent.costs['compute_cost'] += 30
        key = f'relation_transfer:{inputs["theory_id"]}'
        agent.theory_consolidation['executions'][key]['cursor']['pending_fits'] = []
        return {'status': 'UNKNOWN', 'generated_theories': [], 'new_compute_cost': 30,
            'reserved_comparison_compute': 5, 'search_completed': False,
            'completed_research_unit': True}

    monkeypatch.setattr(consolidation, 'transfer_theory', transfer)


def test_pending_binding_gets_complete_quote_after_cached_owner_is_skipped(tmp_path, monkeypatch):
    agent, _, question, args = pending_scene(tmp_path)
    calls = []
    install_complete_visits(monkeypatch, calls)
    result = consolidation.recall_frozen_theories(agent, question, args)
    # The same fair share would be below 40, and the historical atomic minimum
    # is unrelated to this prepared full unit. Neither blocks its execution.
    assert calls == [('prepare', 'pending_old', 70), ('execute', 'pending_old', 40)]
    assert result['new_compute_cost'] == 43
    assert result['reserved_comparison_compute'] == 5
    assert result['object_cursor']['next_candidate_id'] == 'pending_new'
    assert result['complete_research_visits'][0]['status'] == 'cached_complete'
    assert result['complete_research_visits'][1]['status'] == 'complete_visit_executed'
    assert not result['search_completed']
    assert agent.theory_consolidation['executions'][
        f'relation_transfer:frozen_recall:{question["id"]}@1:pending_new']['cursor']['pending_fits']


def test_next_complete_unit_uses_persistent_owner_order_and_prepares_once(tmp_path, monkeypatch):
    agent, _, question, args = pending_scene(tmp_path)
    calls = []
    install_complete_visits(monkeypatch, calls)
    consolidation.recall_frozen_theories(agent, question, args)
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert [item[:2] for item in calls] == [
        ('prepare', 'pending_old'), ('execute', 'pending_old'),
        ('prepare', 'pending_new'), ('execute', 'pending_new')]
    assert result['object_cursor']['next_candidate_id'] == 'unquoted_old'
    assert result['new_compute_cost'] == 38


def test_unaffordable_complete_unit_keeps_cursor_and_does_not_start_partial_fit(tmp_path, monkeypatch):
    agent, _, question, args = pending_scene(tmp_path, ('unquoted_old', 'pending_new'))
    before = deepcopy(agent.theory_consolidation['executions'])
    calls = []
    install_complete_visits(monkeypatch, calls, quotes={'pending_new': 100})
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert calls == [('prepare', 'pending_new', 70)]
    assert result['new_compute_cost'] == 13
    assert agent.theory_consolidation['executions'] == before
    visit = result['complete_research_visits'][-1]
    assert visit['status'] == 'contract_unaffordable'
    assert visit['available_compute'] == 67
    assert 'pending_new' in result['pending_object_ids']


def test_unaffordable_older_unit_yields_to_next_payable_complete_unit(tmp_path, monkeypatch):
    agent, _, question, args = pending_scene(tmp_path)
    calls = []
    install_complete_visits(monkeypatch, calls, quotes={'pending_old': 100})
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert calls == [('prepare', 'pending_old', 70),
                     ('prepare', 'pending_new', 62), ('execute', 'pending_new', 40)]
    assert result['new_compute_cost'] == 51
    assert result['object_cursor']['next_candidate_id'] == 'unquoted_old'
    assert result['complete_research_visits'][1]['status'] == 'contract_unaffordable'
    assert result['complete_research_visits'][2]['status'] == 'complete_visit_executed'


@pytest.mark.parametrize('last_pending', [False, True])
def test_real_transfer_reuses_preparation_and_checks_all_complete_fit_outputs(tmp_path, monkeypatch, last_pending):
    from test_complete_revision_visit import _scene
    from test_theory_consolidation import agent_scene, assert_no_new_science
    from hegel_machine.cognition.question_routing import register_theories
    from hegel_machine.cognition import revision_relations

    candidate, parent, kwargs, cursor, _ = _scene()
    if last_pending:
        cursor.update(layer=0, offset=1, pending_fits=cursor['pending_fits'][:1])
    candidate = {**candidate, 'version': 1}
    parent.pop('source_ref')
    for check in kwargs['evidence_checks']:
        check['context_ref'] = deepcopy(parent['context_ref'])
    agent, lab = agent_scene(tmp_path)
    register_theories(agent, [parent])
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    consolidation._save(agent)
    question = agent._open_question({'id': 'actual_complete_unit', 'question': 'Resume one binding.',
        'context_ref': deepcopy(parent['context_ref']), 'theory_target': {
            'parent_theory_id': parent['id'],
            **{key: deepcopy(parent[key]) for key in ('role_signature', 'output_unit', 'context_ref')},
            'evidence_checks': deepcopy(kwargs['evidence_checks'])}})
    retained = {'_retained_parent': deepcopy(parent), '_retained_components': [],
                'component_theory_ids': []}
    inputs = {'candidate_id': candidate['id'], 'parent_theory_id': parent['id'],
        **{key: deepcopy(parent[key]) for key in ('role_signature', 'output_unit', 'context_ref')},
        'evidence_checks': deepcopy(kwargs['evidence_checks']), 'component_theory_ids': [],
        'fitting': kwargs['fitting'], 'allocation': {'comparison_rows': 4}}
    key = f'relation_transfer:frozen_recall:{question["id"]}@1:{candidate["id"]}'
    agent.theory_consolidation['executions'] = {key: {
        'request_basis': consolidation._relation_request_basis(parent, [], inputs, candidate,
                                                              consolidation.LIBRARY_PATH),
        'cursor': deepcopy(cursor), 'search_completed': False, 'allowance': 50,
        'result': {'status': 'UNKNOWN', 'minimum_compute_for_progress': 100000}}}
    queries, fits = [], []

    def retained_fits(agent, candidate, parent, budget, **kw):
        queries.append(candidate['id'])
        assert budget.spend(2)
        return []

    old_fit = revision_relations._fit_revision_binding

    def fit(*args, **kw):
        fits.append(True)
        return old_fit(*args, **kw)

    monkeypatch.setattr(consolidation, '_retained_revision_fits', retained_fits)
    monkeypatch.setattr(revision_relations, '_fit_revision_binding', fit)
    result = consolidation.recall_frozen_theories(agent, question, {
        'allocation': {'compute': 5000, 'comparison_rows': 4}, 'fitting': kwargs['fitting'],
        '_retained_execution_inputs': {candidate['id']: retained}})
    assert queries == [candidate['id']] and len(fits) == 1
    visit, = result['complete_research_visits']
    assert visit['status'] == 'complete_visit_executed'
    assert visit['preparation_compute'] > 2
    assert result['new_compute_cost'] == agent.costs['compute_cost']
    execution = agent.theory_consolidation['executions'][key]
    assert execution['result']['completed_research_unit']
    assert not execution['search_completed'] and not result['search_completed']
    assert execution['cursor']['pending_fits'] == cursor['pending_fits'][1:]
    generated = execution['result']['generated_theories']
    assert generated and all(len(item['checks']) == 4 for item in generated)
    assert not execution['result']['budget_exhausted']
    assert_no_new_science(agent, lab)

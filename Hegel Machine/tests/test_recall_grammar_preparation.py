"""Shared pure grammar reaches each fresh parent without sharing its progress."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.language import Program
from test_fresh_complete_revision_visit import fresh_recall_scene


def install_proposer(monkeypatch, descriptor, calls):
    def result(cursor):
        next_cursor = deepcopy(cursor or {'layer': 0, 'offset': 0})
        next_cursor['offset'] += 1
        next_cursor['pending_fits'] = []
        next_cursor['fitting'] = {'grammar_preparation_ref': descriptor['source_ref']}
        return {'status': 'UNKNOWN', 'proposals': [], 'excluded': [],
            'cursor': next_cursor, 'actual_compute_cost': 0,
            'reserved_comparison_compute': 0, 'budget_exhausted': False,
            'search_completed': False, 'completed_research_unit': True}

    def prepare(candidate, parent, components, allocation, *, cursor=None,
                grammar_preparation=None, **kwargs):
        assert grammar_preparation is descriptor
        calls.append((candidate['id'], deepcopy(cursor)))
        return {'prepared_visit': {'cursor': deepcopy(cursor)},
            'quote': {'compute': 10, 'maximum_outputs': 0},
            'actual_compute_cost': 0, 'cursor': deepcopy(cursor),
            'search_position': {'layer': 0, 'offset': (cursor or {}).get('offset', 0) + 1}}

    def execute(packet, allocation):
        return result(packet['cursor'])

    def propose(candidate, parent, components, allocation, *, cursor=None,
                grammar_preparation=None, **kwargs):
        assert grammar_preparation is descriptor
        calls.append((candidate['id'], deepcopy(cursor)))
        return result(cursor)

    monkeypatch.setattr(revisions, 'prepare_revision_visit', prepare)
    monkeypatch.setattr(revisions, 'execute_revision_visit', execute)
    monkeypatch.setattr(revisions, 'propose_revision', propose)
    monkeypatch.setattr(consolidation, '_retained_revision_fits', lambda *args, **kwargs: [])
    monkeypatch.setattr(Program, 'predict',
        lambda *args, **kwargs: pytest.fail('pure grammar routing must not predict'))


def grammar_descriptor(tmp_path):
    return {'source_ref': {'file': str(tmp_path / 'retained_library.json'),
        'json_path': ['executions', 'old65', 'cursor', 'fitting']},
        'state': {'structures': [{'shared_preparation_marker': 'pure_pool'}],
                  'grammar_cursor': {'status': 'completed'}}}


@pytest.mark.parametrize('complete', [False, True])
def test_direct_transfer_passes_only_ram_grammar_to_existing_proposer(
        tmp_path, monkeypatch, complete):
    agent, _, question, args, candidate = fresh_recall_scene(tmp_path)
    descriptor, calls = grammar_descriptor(tmp_path), []
    install_proposer(monkeypatch, descriptor, calls)
    target = question['theory_target']
    retained = args['_retained_execution_inputs'][candidate['id']]
    inputs = {**deepcopy(target), **retained, 'candidate_id': candidate['id'],
        'theory_id': f'frozen_recall:{question["id"]}@1:{candidate["id"]}',
        'allocation': args['allocation'], 'fitting': args['fitting'],
        '_grammar_preparation': descriptor}
    if complete:
        inputs['_prepare_complete_visit'] = True
        prepared = consolidation.transfer_theory(agent, inputs)
        inputs.pop('_prepare_complete_visit')
        inputs['_prepared_relation_visit'] = prepared['_prepared_relation_visit']
    consolidation.transfer_theory(agent, inputs)

    assert calls == [(candidate['id'], None)]
    execution = next(iter(agent.theory_consolidation['executions'].values()))
    assert execution['cursor']['offset'] == 1
    assert execution['cursor']['fitting'] == {
        'grammar_preparation_ref': descriptor['source_ref']}
    assert '_grammar_preparation' not in execution['request_basis']
    assert 'shared_preparation_marker' not in json.dumps(execution)


def test_fresh_complete_groups_keep_four_round_order_without_alias_or_old65_progress(
        tmp_path, monkeypatch):
    agent, _, question, args, candidate = fresh_recall_scene(tmp_path, first_kind='other')
    alias = {**deepcopy(candidate), 'id': 'alias_of_later'}
    agent.theory_consolidation['candidates'][alias['id']] = alias
    args['_retained_execution_inputs'][alias['id']] = deepcopy(
        args['_retained_execution_inputs'][candidate['id']])
    old65 = {'cursor': {'layer': 3, 'offset': 8,
        'pending_fits': [{'indices': [1, 5], 'position': {'layer': 1, 'offset': 5}}],
        'fitting': {'structures': ['original65_pool']}},
        'search_completed': False, 'result': {'status': 'UNKNOWN'}}
    agent.theory_consolidation['executions'] = {'old65_scope': deepcopy(old65)}
    descriptor, calls = grammar_descriptor(tmp_path), []
    args['_grammar_preparation'] = descriptor
    install_proposer(monkeypatch, descriptor, calls)
    consolidation._save(agent)

    for _ in range(8):
        recalled = consolidation.recall_frozen_theories(agent, question, args)
        assert [group['candidate_ids'] for group in recalled['execution_work_groups']] == [
            ['first'], [candidate['id'], alias['id']]]
        assert sum(visit['status'] == 'complete_visit_executed'
                   for visit in recalled['complete_research_visits']) == 1

    assert [identifier for identifier, _ in calls] == ['first', candidate['id']] * 4
    assert [None if cursor is None else cursor['offset'] for _, cursor in calls] == [
        None, None, 1, 1, 2, 2, 3, 3]
    assert agent.theory_consolidation['executions']['old65_scope'] == old65
    assert recalled['object_cursor']['next_candidate_id'] == 'first'
    for key, execution in agent.theory_consolidation['executions'].items():
        if key == 'old65_scope':
            continue
        assert execution['cursor']['layer'] == 0
        assert execution['cursor']['offset'] == 4
        assert execution['cursor']['pending_fits'] == []
        assert execution['cursor']['fitting'] == {
            'grammar_preparation_ref': descriptor['source_ref']}
        assert 'shared_preparation_marker' not in json.dumps(execution)
    assert args['_grammar_preparation'] is descriptor


def test_real_complete_preparation_consumes_retained_pool_without_restarting_grammar(
        tmp_path, monkeypatch):
    from hegel_machine.cognition import synthesis

    agent, _, question, args, candidate = fresh_recall_scene(tmp_path)
    descriptor = grammar_descriptor(tmp_path)
    descriptor['state']['structures'] = [{'operation': 'refit', 'rationale': 'fixture',
        'proposal_index': 0, 'source_ref': {'kind': 'fixture'},
        'program': Program('const', value=0.).to_dict(), 'parameter_paths': [[]]}]
    original = deepcopy(descriptor)
    target = question['theory_target']
    retained = args['_retained_execution_inputs'][candidate['id']]
    inputs = {**deepcopy(target), **retained, 'candidate_id': candidate['id'],
        'theory_id': f'frozen_recall:{question["id"]}@1:{candidate["id"]}',
        'allocation': args['allocation'],
        'fitting': {**args['fitting'], 'max_structures': 1},
        '_grammar_preparation': descriptor, '_prepare_complete_visit': True}
    monkeypatch.setattr(consolidation, '_retained_revision_fits', lambda *args, **kwargs: [])
    monkeypatch.setattr(synthesis, 'iter_synthesis_structures',
        lambda *args, **kwargs: pytest.fail('retained pure grammar must not regenerate'))
    monkeypatch.setattr(revisions, '_fit_revision_binding',
        lambda *args, **kwargs: pytest.fail('preparation must not solve'))

    prepared = consolidation.transfer_theory(agent, inputs)

    assert prepared['prepared_visit'] is not None
    assert prepared['search_position'] == {'layer': 0, 'offset': 1}
    assert prepared['quote']['fit_compute'] > 0
    assert prepared['actual_compute_cost'] > 0
    assert agent.theory_consolidation.get('executions', {}) == {}
    assert descriptor == original
    prepared['prepared_visit']['_generator'].close()


@pytest.mark.parametrize('rejection', ['alignment', 'source_constraint'])
def test_completed_rejected_binding_reports_actual_position_but_parent_mismatch_is_not_unit(
        tmp_path, monkeypatch, rejection):
    agent, _, question, args, candidate = fresh_recall_scene(
        tmp_path, first_kind='not_applicable')
    monkeypatch.setattr(consolidation, '_retained_revision_fits', lambda *args, **kwargs: [])
    monkeypatch.setattr(revisions, '_fit_revision_binding',
        lambda *args, **kwargs: pytest.fail('a rejected binding must not fit'))
    if rejection == 'alignment':
        monkeypatch.setattr(revisions, '_align_parameter_components', lambda *args: False)
    else:
        monkeypatch.setattr(revisions, '_bound_proposal', lambda *args: (None, []))

    for position in ({'layer': 0, 'offset': 1}, {'layer': 1, 'offset': 1}):
        recalled = consolidation.recall_frozen_theories(agent, question, args)
        visits = recalled['complete_research_visits']
        completed = [visit for visit in visits if visit['status'] == 'complete_visit_executed']
        assert len(completed) == 1
        assert completed[0]['candidate_id'] == candidate['id']
        assert completed[0]['search_position'] == position
        first = next(visit for visit in visits if visit['candidate_id'] == 'first')
        assert first['status'] != 'complete_visit_executed'
        assert first['search_position'] is None
        key = f'relation_transfer:frozen_recall:{question["id"]}@1:{candidate["id"]}'
        execution = agent.theory_consolidation['executions'][key]
        assert execution['result']['completed_research_unit']
        assert execution['result']['research_unit_position'] == position
        assert {key: execution['cursor'][key] for key in ('layer', 'offset')} == position
        assert not execution['result']['generated_theories']

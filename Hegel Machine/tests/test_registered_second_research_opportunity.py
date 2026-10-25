"""An answered measurement must not consume an unused native research slice."""

from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_control as control


@pytest.mark.parametrize('route_cost,route_status,parent_completed,search_completed,resume_phase,recalls', [
    (0, 'author_work_request', False, False, None, 2),
    (1, 'author_work_request', False, False, None, 1),
    (0, 'author_work_request', True, False, None, 1),
    (0, 'author_work_request', False, True, None, 1),
    (0, 'scoped_unknown', False, False, None, 1),
    (0, 'author_work_request', False, False, 'last_research_allocation', 1),
])
def test_second_slice_consumes_only_actual_unfinished_zero_cost_request(
        tmp_path, monkeypatch, route_cost, route_status, parent_completed,
        search_completed, resume_phase, recalls):
    @dataclass
    class Limits:
        compute_budget: int = 500040
        execution_seconds: float = 300.

    config = Limits()
    question = {'id': 'native_public_response'}
    old_cursor, new_cursor = {'offset': 2}, {'offset': 7}
    costs = {'compute_cost': 40, 'python_execution_seconds': 0.,
             'observation_cost': 97, 'model_calls': 0}
    parent = {'id': 'retained_parent'}
    component = {'id': 'source_component', 'program': {'op': 'const', 'value': 0.}}
    agent = SimpleNamespace(config=config, costs=costs, workspace=tmp_path,
        questions=[question], rows=[None] * 97, candidate_paths=[], theory_index={},
        theory_consolidation={'candidates': {'old19': {}, 'new20': {}},
            'executions': {'old19': {'cursor': old_cursor}, 'new20': {'cursor': new_cursor}}})
    events, exports = [], []

    def recall(restored, owner, arguments):
        call = sum(event == 'recall' for event in events)
        assert restored is agent and owner is question
        assert arguments['allocation']['compute'] == 150000
        assert agent.config.compute_budget - agent.costs['compute_cost'] == 150000
        assert arguments['_retained_parent'] is parent
        assert old_cursor['offset'] == 2 + call
        assert new_cursor['offset'] == 7 + call
        if call:
            saved = json.loads((tmp_path / 'registered_actions.json').read_text())
            assert saved[-1]['action'] == 'research_route'
            assert saved[-1]['feedback']['status'] == 'author_work_request'
            assert saved[-1]['compute_delta'] == 0
        old_cursor['offset'] += 1
        new_cursor['offset'] += 1
        agent.costs['compute_cost'] += 100000
        question['frozen_recall_execution'] = {'result': {
            'search_completed': search_completed, 'pending_object_ids': ['old19', 'new20']}}
        events.append('recall')
        return {'status': 'UNKNOWN', 'search_completed': search_completed}

    def execute(action, arguments, lab):
        assert action == 'run'
        operator = arguments['operator']
        events.append(operator)
        if operator == 'research_choice':
            assert arguments['models'] == ['complete_strong_parent.py']
            agent.costs['compute_cost'] += 97
            return {'status': 'completed'}
        assert operator == 'research_route'
        expected = 0 if resume_phase == 'last_research_allocation' else 150000
        assert arguments['allocation']['compute'] == expected
        agent.costs['compute_cost'] += route_cost
        return {'status': route_status,
            'answer_execution': {'parent_revision_completed': parent_completed},
            'frozen_recall': {'status': 'not_needed_answer_available'}}

    agent._execute = execute
    agent._evaluated_summary = lambda: {'model': 'complete_strong_parent.py'}
    original_arguments = {'candidate_id': 'new20', 'library_path': 'shared/theory_library.json',
        'parent_theory_id': parent['id'], 'fitting': {'checks': [1]}, 'evidence_checks': [2],
        'allocation': {'compute': 150000, 'comparison_rows': 97}}
    scene = {'agent': agent, 'lab': object(), 'models': ['complete_strong_parent.py'],
        'parent': parent, 'retained_components': [component],
        'source_component_theory_ids': [component['id']], 'resume_phase': resume_phase,
        'transfer_arguments': deepcopy(original_arguments)}
    monkeypatch.setattr(control.consolidation, 'recall_frozen_theories', recall)
    monkeypatch.setattr('hegel_machine.cognition.question_routing._retained_revision_inputs',
        lambda *args: {'_retained_components': [component]})
    monkeypatch.setattr('hegel_machine.cognition.finite_native_author.export_revision_candidates',
        lambda *args: exports.append('export') or {'new_exports': 0})
    monkeypatch.setattr(control, '_checkpoint', lambda *args: None)
    monkeypatch.setattr(control, 'quote_joint_paid_scores', lambda restored: {
        'models': 1, 'paid_rows': 97, 'reused_score_count': 1,
        'unpaid_score_count': 0, 'unpaid_scoring_compute': 0})

    result = control._arm_run(scene, {'budget': {'per_consumption_compute': 150000}})
    assert result['status'] == 'research_completed'
    assert events.count('recall') == recalls
    assert events.count('research_route') == 1
    assert events.count('research_choice') == 2
    assert len(exports) == recalls
    assert old_cursor['offset'] == 2 + recalls
    assert new_cursor['offset'] == 7 + recalls
    assert result['continuation_costs']['compute_cost'] == 100000 * recalls + 194 + route_cost
    assert result['continuation_costs']['compute_cost'] <= 500000
    assert scene['transfer_arguments'] == original_arguments
    assert agent.config is config


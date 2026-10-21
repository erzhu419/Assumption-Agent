"""Automatic revision work leaves the full ordinary comparison inside its cap."""
from copy import deepcopy
from dataclasses import replace
import json
import pytest

from hegel_machine.cognition.finite_native_author import ordinary_completion_reserve
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_finite_native_author import POLICY
from test_registered_revision_target import revision_scene, forbid_acquisition


@pytest.mark.parametrize('construction_compute', [None, 150])
def test_registered_revision_funds_old_and_new_comparison_and_restores_original_limits(
        tmp_path, monkeypatch, construction_compute):
    agent, lab, owner = revision_scene(tmp_path)
    bank = owner['native_condition_research'][0]['candidates']
    agent._execute('evaluate', {'model': bank[-1]['model']}, lab)
    agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + 600)
    original = agent.config
    if construction_compute is not None:
        agent._organization_registration['routing_allocation'] = {'compute': construction_compute}
    agent.knowledge['finite_native_author']['question']['theory_target'] = deepcopy(owner['theory_target'])
    expected = ordinary_completion_reserve(agent, owner)
    assert expected['models'] == 1 + len(bank)
    assert expected['compute_cost'] == len(agent.rows) * (expected['models'] + 1) + original.confirmation_rows
    before = deepcopy(agent.costs)
    forbid_acquisition(lab, monkeypatch)
    agent._active_question_id = owner['id']
    agent._before_model_turn(lab)
    routed = next(event for event in reversed(agent.events)
                  if event.get('arguments', {}).get('operator') == 'research_route')
    if construction_compute is not None:
        assert routed['arguments']['allocation']['compute'] == construction_compute
    reserve = owner['ordinary_completion_reserve']
    exports = owner.get('revision_candidate_exports', [])
    assert exports, owner.get('frozen_recall_execution')
    assert agent.config == original
    state = json.loads((agent.workspace / 'research_state.json').read_text())
    assert state['limits']['compute_budget'] == original.compute_budget
    assert reserve['new_candidate_comparison_compute'] >= len(exports) * len(agent.rows)
    remaining = original.compute_budget - agent.costs['compute_cost']
    assert remaining >= reserve['compute_cost'] + len(exports) * len(agent.rows)
    export_costs = deepcopy(agent.costs)
    outcome = execute_procedure(agent, lab, POLICY, max_steps=1, allow_confirmation=False,
        record_events=True, before_step=agent._before_model_turn, after_step=agent._after_model_turn)
    assert outcome['steps'][0]['proposal']['arguments']['operator'] == 'research_choice'
    comparison = agent.operator_results[-1]
    paths = [item['path'] for item in comparison['evaluations']]
    assert 'archive/failed.py' in paths
    assert all(item['model'] in paths for item in exports)
    assert len(paths) == expected['models'] + len(exports)
    assert not comparison.get('truncated') and not any(item.get('truncated') for item in comparison['evaluations'])
    assert agent.costs['compute_cost'] - export_costs['compute_cost'] == len(paths) * len(agent.rows)
    assert original.compute_budget - agent.costs['compute_cost'] >= len(agent.rows) + original.confirmation_rows
    assert agent.config == original and agent.costs['observation_cost'] == before['observation_cost']
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_insufficient_completion_funding_does_not_spend_the_original_total(tmp_path, monkeypatch):
    agent, lab, owner = revision_scene(tmp_path)
    before = deepcopy(agent.costs)
    expected = ordinary_completion_reserve(agent, owner)
    agent.config = replace(agent.config, compute_budget=before['compute_cost'] + expected['compute_cost'] - 1)
    original = agent.config
    forbid_acquisition(lab, monkeypatch)
    agent._active_question_id = owner['id']
    agent._before_model_turn(lab)
    assert agent.config == original and agent.costs == before
    assert not owner.get('revision_candidate_exports')
    assert owner['ordinary_completion_reserve']['route_compute_cost'] == 0


def test_registered_task_reserve_uses_its_route_mission_after_other_theory_work(tmp_path):
    agent, _, owner = revision_scene(tmp_path)
    expected = ordinary_completion_reserve(agent, owner)
    agent.research_mission = {'purpose': 'theory_learning', 'mission_id': 'preceding_theory_work'}
    assert ordinary_completion_reserve(agent, owner) == expected
    owner['mission'] = deepcopy(agent.research_mission)
    assert ordinary_completion_reserve(agent, owner) is None

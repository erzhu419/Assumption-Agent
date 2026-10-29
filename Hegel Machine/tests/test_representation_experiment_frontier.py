"""Paid candidate ties reach experiments, refitting and complete pool comparison."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from hegel_machine.cognition.finite_native_author import select_actions
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from test_inquiry_revision import PublicLab
from test_native_representation_method_selection import current_memory, prepared


class GridLab(PublicLab):
    """Small fixture: both branch ambiguity and its bought answer are known."""
    def public_spec(self):
        return {**super().public_spec(), 'feature_bounds': {'u': [-1., 1.]},
            'control_levels': {'u': [index / 10 for index in range(-10, 11)]},
            'max_probe_length': 12}

    def execute(self, action):
        assert all(point['u'] in self.public_spec()['control_levels']['u']
                   for point in action['sequence'])
        state, rows = 0., []
        for row in self.prediction_rows(action):
            state = float(row.features['u'] >= (.2 if state else .6))
            rows.append(Observation(row.features, state, row.history))
        first = self.paid
        self.paid += len(rows)
        self.actions.append(deepcopy(action))
        return {'action': deepcopy(action), 'rows': [asdict(row) for row in rows],
            'row_indices': list(range(first, self.paid)), 'observation_index': first,
            'observation_cost': len(rows), 'measurement_cost': 0}


def completed(tmp_path, monkeypatch, *, lab=None):
    agent, original_lab, contract, _ = prepared(tmp_path, monkeypatch)
    lab = lab or GridLab()
    lab.paid = len(agent.rows)
    agent.config = replace(agent.config, compute_budget=100000, observation_budget=28,
                           confirmation_rows=24)
    before = dict(agent.costs)
    proposal = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    result = agent._execute(proposal['action'], proposal['arguments'], lab)
    agent.method_results.append({'proposal': proposal, 'feedback': result,
        'evidence_before': {'rows': 20}, 'evidence_after': {'rows': 20}})
    compared = agent._execute('run', {'operator': 'research_choice',
        'models': ['old.py', *agent.candidate_paths],
        'research_question': deepcopy(contract['question'])}, lab)
    assert compared['models_completed'] == len(['old.py', *agent.candidate_paths])
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    assert agent.costs['observation_cost'] == before['observation_cost']
    return agent, lab, contract, native


def test_actual_tie_is_queried_then_refitted_and_full_pool_compared(tmp_path, monkeypatch):
    agent, lab, contract, native = completed(tmp_path, monkeypatch)
    frontier = native['experiment_frontier']
    assert frontier['candidate_count'] > 1 and frontier['predicted_separation'] > 0
    assert frontier['training_mse'] == frontier['validation_mse'] == 0.
    assert native['compute_cost'] <= native['compute_limit']
    old_source = (agent.workspace / 'old.py').read_bytes()
    old_models = set(agent.candidate_paths)
    query = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert query['action'] == 'query'
    assert query['complete_quote']['confirmation_rows_reserved'] == 24
    prior_rows = len(agent.rows)
    feedback = agent._execute('query', query['arguments'], lab)
    assert 'after_observation_error' not in feedback, feedback
    assert len(agent.rows) == prior_rows + frontier['observation_cost']
    assert len(feedback['after_observation_results']) == 1
    revised = feedback['after_observation_results'][0]
    assert revised['native_representation']['comparison_completed']
    assert revised['native_representation']['fitting_rows'] == 15 + frontier['observation_cost']
    assert revised['native_representation']['split']['validation_records'] == [3]
    assert old_models < set(agent.candidate_paths)
    agent.method_results.append({'proposal': query, 'feedback': feedback,
        'evidence_before': {'rows': prior_rows}, 'evidence_after': {'rows': len(agent.rows)}})
    choice = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert choice['arguments']['operator'] == 'research_choice'
    assert old_models | {'old.py'} <= set(choice['arguments']['models'])
    assert set(agent.candidate_paths) <= set(choice['arguments']['models'])
    compared = agent._execute(choice['action'], choice['arguments'], lab)
    assert compared['models_completed'] == compared['models_requested']
    assert compared['evaluated_model']['mse'] == 0.
    returned = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    # No room remains for another experiment plus the original confirmation.
    assert returned['action'] == 'finish'
    assert (agent.workspace / 'old.py').read_bytes() == old_source
    assert agent.costs['model_calls'] == 0


@pytest.mark.parametrize('resource', ['observations', 'compute'])
def test_unfunded_frontier_does_not_block_ordinary_confirmation(tmp_path, monkeypatch, resource):
    agent, lab, _, native = completed(tmp_path, monkeypatch)
    assert native['experiment_frontier']
    if resource == 'observations':
        agent.config = replace(agent.config, observation_budget=agent.config.confirmation_rows)
    else:
        # Current comparison and confirmation are funded, another research unit is not.
        agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + 100)
    returned = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert returned['action'] == 'finish'


def test_equivalent_legal_grid_predictions_leave_finish_ready(tmp_path, monkeypatch):
    class SparseGrid(GridLab):
        def public_spec(self):
            return {**super().public_spec(), 'control_levels': {'u': [-1., 0., 1.]}}

    agent, lab, _, native = completed(tmp_path, monkeypatch, lab=SparseGrid())
    assert 'experiment_frontier' not in native
    returned = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert returned['action'] == 'finish'


def test_remaining_account_can_fund_revision_below_old_per_visit_cap(tmp_path, monkeypatch):
    agent, lab, contract, native = completed(tmp_path, monkeypatch)
    original_cap = contract['native_representation_plan']['compute_limit']
    agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + 5000)
    query = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert query['action'] == 'query'
    quote = query['complete_quote']
    assert quote['compute_cost'] <= 5000
    assert quote['minimum_native_compute'] <= quote['bounded_research_compute'] < original_cap
    assert quote['max_exported_models'] == len(native['available_search_directions']) * 4
    result = agent._execute('query', query['arguments'], lab)
    assert 'after_observation_error' not in result
    revised = result['after_observation_results'][0]['native_representation']
    assert revised['comparison_completed']
    assert revised['selected']['training_mse'] == revised['selected']['validation_mse'] == 0.
    choice = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert choice['arguments']['operator'] == 'research_choice'
    compared = agent._execute(choice['action'], choice['arguments'], lab)
    assert compared['models_completed'] == compared['models_requested']
    assert agent.costs['compute_cost'] <= agent.config.compute_budget

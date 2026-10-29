"""A method selects existing research directions without changing its evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from test_native_startup_revision import scene
from test_native_discrete_state_route import scene as state_scene
from test_native_representation_routing import PublicOnlyLab


def run(agent, arguments):
    returned = agent._execute('run', arguments, PublicOnlyLab())
    return returned, load_native_representation_result(
        returned['native_representation_ref'], agent.workspace)


def test_explicit_state_research_changes_algorithm_not_parent_pool_split_or_objective(tmp_path, monkeypatch):
    ordinary, _, ordinary_arguments = scene(tmp_path / 'ordinary')
    _, before = run(ordinary, ordinary_arguments)
    assert [item['direction'] for item in before['search_directions']] == ['startup_revision']

    agent, _, arguments = scene(tmp_path / 'alternative')
    old_sources = {path.name: path.read_bytes() for path in agent.workspace.glob('*.py')}
    arguments['native_representation_plan']['search_directions'] = ['discrete_state']

    def unexpected_startup(*args, **kwargs):
        pytest.fail('explicit broad state research must not execute startup repair')

    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_startup_claims', unexpected_startup)
    returned, result = run(agent, arguments)
    assert [item['direction'] for item in result['search_directions']] == ['discrete_state']
    assert result['search_directions'][0]['compute_cost'] > 0
    assert result['direction_selection'] == {
        'available': ['startup_revision', 'discrete_state', 'retained_composition'],
        'selected': ['discrete_state'], 'source': 'native_representation_plan'}
    assert result['split'] == before['split']
    assert result['selection_objective'] == before['selection_objective']
    assert result['retained_comparison'] == before['retained_comparison']
    assert result['comparison_completed'] and result['all_parent_candidates_retained']
    assert all(item['search_direction'] == 'discrete_state' for item in result['candidates'])
    assert any(item['state_research']['kind'] == 'discrete_state_search'
               for item in result['candidates'])
    assert result['selected']['selection']['selection_records'] == [0, 1, 2, 3]
    assert agent.costs['compute_cost'] <= arguments['native_representation_plan']['compute_limit']
    assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0
    assert all((agent.workspace / name).read_bytes() == data for name, data in old_sources.items())
    assert not returned['parent_revision_completed']


def test_changed_direction_is_not_cached_but_repeating_its_paid_result_is(tmp_path):
    agent, _, arguments = scene(tmp_path)
    first, _ = run(agent, arguments)
    cost = agent.costs['compute_cost']
    replay, _ = run(agent, arguments)
    assert replay['cache_hit'] and replay['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == cost
    alternative = deepcopy(arguments)
    alternative['native_representation_plan']['search_directions'] = ['discrete_state']
    second, result = run(agent, alternative)
    assert not second['cache_hit'] and second['new_compute_cost'] > 0
    assert second['native_representation_ref'] != first['native_representation_ref']
    assert result['direction_selection']['selected'] == ['discrete_state']
    cost = agent.costs['compute_cost']
    repeated, _ = run(agent, alternative)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == cost


@pytest.mark.parametrize('directions', [[], 'discrete_state', ['discrete_state', 'discrete_state'],
    ['invented_algorithm'], ['coordinates'], [1]])
def test_invalid_or_unavailable_directions_do_not_silently_fall_back(tmp_path, directions):
    agent, _, arguments = scene(tmp_path)
    arguments['native_representation_plan']['search_directions'] = directions
    with pytest.raises(ValueError, match='current available directions'):
        run(agent, arguments)
    assert agent.costs['compute_cost'] == 0


def test_startup_direction_requires_an_actual_closed_startup_parent(tmp_path):
    agent, arguments = state_scene(tmp_path)
    arguments['native_representation_plan']['search_directions'] = ['startup_revision']
    with pytest.raises(ValueError, match='current available directions'):
        run(agent, arguments)
    assert agent.costs['compute_cost'] == 0

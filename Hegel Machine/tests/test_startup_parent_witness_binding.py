"""Paid witness bindings alter finite generation, not the retained pool."""

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from test_native_startup_revision import scene
from test_native_representation_routing import PublicOnlyLab


def crowded_scene(path):
    agent, supported, arguments = scene(path)
    guard, mature, startup = supported.children[0].children
    u = Program('feature', name='u')
    wrong = [Program('const', value=0.), Program('const', value=1.),
             Program('ge', children=(u, Program('const', value=.85))),
             Program('ge', children=(u, Program('const', value=-.15)))]
    parents = [{'id': f'earlier_{i}', 'program': Program('state_replay', children=(
        Program('if', children=(guard, branch, startup)),)).to_dict()}
        for i, branch in enumerate(wrong)]
    parents += [{'id': 'old', 'program': supported.to_dict()},
                {'id': 'constant', 'program': Program('const', value=0.).to_dict()}]
    arguments['theories'] = parents
    arguments['native_representation_plan'].update(
        parent_theory_ids=[parent['id'] for parent in parents],
        search_directions=['startup_revision'])
    return agent, supported, arguments


def test_witness_uses_original_four_slots_and_preserves_full_comparison(tmp_path):
    old_agent, _, old_arguments = crowded_scene(tmp_path / 'old')
    old_result = old_agent._execute('run', old_arguments, PublicOnlyLab())
    old = load_native_representation_result(old_result['native_representation_ref'], old_agent.workspace)
    new_agent, supported, new_arguments = crowded_scene(tmp_path / 'new')
    evidence = [{'parent_theory_id': 'old',
                 'execution_ref': {'file': '/paid/assessment.json', 'json_path': []},
                 'transition_refs': [{'file': '/paid/assessment.json',
                                      'json_path': ['closed_prediction_state']}]}]
    new_arguments['native_representation_plan'].update(
        startup_parent_priority=['old'], startup_parent_witness_refs=evidence)
    result = new_agent._execute('run', new_arguments, PublicOnlyLab())
    new = load_native_representation_result(result['native_representation_ref'], new_agent.workspace)

    assert 'startup_parent_binding' not in old
    assert old['selected']['development_mse'] > 0
    assert new['selected']['training_mse'] == new['selected']['validation_mse'] == 0
    assert new['selected']['development_mse'] == 0
    assert len(old['candidates']) == len(new['candidates']) == 4
    expected = new_arguments['native_representation_plan']['parent_theory_ids']
    assert old['parent_theory_ids'] == new['parent_theory_ids'] == expected
    assert [item['id'] for item in old['retained_comparison']] == expected
    assert [item['id'] for item in new['retained_comparison']] == expected
    assert new['comparison_completed'] and new['all_parent_candidates_retained']
    binding = new['startup_parent_binding']
    assert binding['priority'] == ['old'] and binding['witness_refs'] == evidence
    assert binding['max_candidates'] == 4
    assert binding['generation_parent_order'] == ['old', 'earlier_0', 'earlier_1', 'earlier_2', 'earlier_3']
    assert new['actions'][0]['parent_theory_ids'] == binding['generation_parent_order']
    revised = Program.from_dict(new['selected']['program'])
    assert revised.children[0].children[:2] == supported.children[0].children[:2]
    assert new_agent.costs['observation_cost'] == new_agent.costs['model_calls'] == 0


@pytest.mark.parametrize('priority', [['missing'], ['constant'], ['old', 'old']])
def test_priority_cannot_bind_an_absent_or_ineligible_parent(tmp_path, priority):
    agent, _, arguments = crowded_scene(tmp_path)
    arguments['native_representation_plan']['startup_parent_priority'] = priority
    with pytest.raises(ValueError, match='startup_parent_priority'):
        agent._execute('run', arguments, PublicOnlyLab())
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0

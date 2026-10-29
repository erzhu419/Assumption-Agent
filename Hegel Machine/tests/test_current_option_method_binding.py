"""One saved method selects actual quoted research on different public scenes."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.finite_native_author import (
    research_segment_options, select_actions, select_research_segment,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.research_programs import (
    current_option_selection_plan, export_research_selection_program,
)
from test_native_representation_method_selection import current_memory, prepared


def offered(agent, lab):
    memory = current_memory(agent, lab)
    view = research_segment_options(lab.public_spec(), agent.observations, memory)
    option = next(item for item in view['options'] if item['id'] == 'native_representation:whole_state')
    return memory, option


def test_same_frozen_method_executes_current_parents_and_data_on_two_scenes(tmp_path, monkeypatch):
    first, lab, _, _ = prepared(tmp_path / 'first', monkeypatch)
    memory, source_option = offered(first, lab)
    plan = current_option_selection_plan(source_option)
    assert plan['first_segment'] == {'current_option': {
        'id': 'native_representation:whole_state', 'family': 'N'}}
    compiled_source = export_research_selection_program(plan)
    compiled = {}
    exec(compiled_source, compiled)
    first_action = compiled['select_actions'](lab.public_spec(), first.observations,
        {**memory, 'ordinary_method_path': str(first.workspace / 'ordinary.py')})
    first_result = first._execute(first_action['action'], first_action['arguments'], lab)
    first_native = load_native_representation_result(first_result['native_representation_ref'], first.workspace)
    assert [item['direction'] for item in first_native['search_directions']] == ['discrete_state']

    second, other_lab, _, original = prepared(tmp_path / 'second', monkeypatch)
    second._save('other_parent.py', (second.workspace / 'old.py').read_text())
    register_theories(second, [{'id': 'other_parent', 'program': original.to_dict()},
        {'id': 'other_constant', 'program': Program('const', value=0.).to_dict()}])
    contract = second.knowledge['finite_native_author']
    question = next(q for q in second.questions if q['id'] == second._active_question_id)
    contract['predictor_paths'] = ['other_parent.py']
    contract['native_representation_plan'].update(
        parent_theory_ids=['other_parent', 'other_constant'], validation_records=[2], compute_limit=9000)
    question['native_representation_plan'] = deepcopy(contract['native_representation_plan'])
    question['use_sites'] = [{'kind': 'supply_component', 'model': 'other_parent.py', 'slot': 'representation'}]
    contract['question'] = deepcopy(question)
    # A distinct public control history with the same independently specified
    # fixture law: changing .4 to .45 crosses neither of its transition bounds.
    for record in second.observations:
        for row in record['rows']:
            if row['features']['u'] == .4:
                row['features']['u'] = .45
            for step in row['history']:
                if step['u'] == .4:
                    step['u'] = .45
        for step in record['action']['sequence']:
            if step['u'] == .4:
                step['u'] = .45
    second.rows = [Observation(row['features'], row['target'], tuple(row['history']))
        for record in second.observations for row in record['rows']]
    compared = second._execute('run', {'operator': 'research_choice', 'models': ['other_parent.py'],
        'research_question': deepcopy(question)}, other_lab)
    assert compared['models_completed'] == 1
    memory, current_option = offered(second, other_lab)
    assert source_option['semantic_action'] != current_option['semantic_action']
    before = dict(second.costs)
    action = compiled['select_actions'](other_lab.public_spec(), second.observations,
        {**memory, 'ordinary_method_path': str(second.workspace / 'ordinary.py')})
    assert second.costs == before
    assert action['arguments']['native_representation_plan']['parent_theory_ids'] == ['other_parent', 'other_constant']
    assert action['arguments']['native_representation_plan']['validation_records'] == [2]
    assert action['arguments']['native_representation_plan']['compute_limit'] == 9000
    assert action['research_segment_option'] == plan['first_segment']['current_option']
    result = second._execute(action['action'], action['arguments'], other_lab)
    native = load_native_representation_result(result['native_representation_ref'], second.workspace)
    assert native['parent_theory_ids'] == ['other_parent', 'other_constant']
    assert [item['direction'] for item in native['search_directions']] == ['discrete_state']
    assert native['comparison_completed']
    assert second.costs['model_calls'] == second.costs['observation_cost'] == 0
    assert export_research_selection_program(plan) == compiled_source


@pytest.mark.parametrize('unavailable', ['missing', 'unfunded', 'invalid_parent'])
def test_saved_choice_falls_back_when_no_current_legal_quoted_option(tmp_path, monkeypatch, unavailable):
    agent, lab, _, _ = prepared(tmp_path, monkeypatch)
    memory, option = offered(agent, lab)
    plan = current_option_selection_plan(option)
    options = memory['method_decision']['research_options']
    whole = next(item for item in options if item['id'] == 'native_representation:whole_state')
    if unavailable == 'missing':
        options.remove(whole)
    elif unavailable == 'unfunded':
        whole['complete_quote']['status'] = 'complete_unit_unfunded'
    else:
        whole['proposal']['arguments']['native_representation_plan']['parent_theory_ids'] = ['missing_parent']
    ordinary = select_actions(lab.public_spec(), agent.observations, memory)
    action = select_research_segment(lab.public_spec(), agent.observations, memory,
        plan=plan, ordinary_select_actions=lambda *args: ordinary)
    assert action == ordinary
    assert 'research_segment_index' not in action


def test_current_choice_keeps_closure_and_completed_segment_order(tmp_path, monkeypatch):
    agent, lab, _, _ = prepared(tmp_path, monkeypatch)
    memory, option = offered(agent, lab)
    plan = current_option_selection_plan(option)
    ordinary = {'stop': 'ordinary'}
    select = lambda: select_research_segment(lab.public_spec(), agent.observations, memory,
        plan=plan, ordinary_select_actions=lambda *args: ordinary)
    action = select()
    assert action['research_segment_index'] == 0
    memory['program_results'] = [{'proposal': action, 'feedback': {'execution_complete': True}}]
    assert select() == ordinary
    # A second supported segment can follow the actual first binding, while a
    # different first executed option cannot silently stand in for it.
    plan['second_segment'] = deepcopy(plan['first_segment'])
    assert select()['research_segment_index'] == 1
    memory['program_results'][0]['proposal']['research_segment_option']['id'] = 'native_representation:ordinary'
    assert select() == ordinary
    memory['program_results'] = []
    ordinary = {'action': 'run', 'arguments': {'operator': 'research_choice', 'models': ['old.py']}}
    assert select() == ordinary

"""Existing method selection can invoke an actual alternative state researcher."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.finite_native_author import (
    export_finite_native_author, publish_research_options, research_segment_options,
    select_actions,
)
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.research_programs import (
    export_research_selection_program, propose_research_action_revisions,
)
from test_finite_native_author import declaration
from test_native_startup_revision import scene
from test_native_representation_routing import PublicOnlyLab


def prepared(tmp_path, monkeypatch):
    agent, original, arguments = scene(tmp_path)
    agent.rows = [Observation(row['features'], row['target'], tuple(row['history']))
                  for row in agent.rows]
    contract = declaration()
    question = {**contract['question'], **arguments['research_question']}
    question['native_representation_plan'] = deepcopy(arguments['native_representation_plan'])
    question = agent._open_question(question)
    register_theories(agent, arguments['theories'])
    agent._active_question_id = question['id']
    contract.update(question=deepcopy(question), predictor_paths=['old.py'],
        native_representation_plan=deepcopy(arguments['native_representation_plan']),
        max_compute=10000, max_candidates=4)
    agent.knowledge['finite_native_author'] = contract
    monkeypatch.chdir(agent.workspace)
    lab = PublicOnlyLab()
    comparison = agent._execute('run', {'operator': 'research_choice', 'models': ['old.py'],
        'research_question': deepcopy(question)}, lab)
    assert comparison['models_completed'] == 1
    agent._save('ordinary.py', export_finite_native_author())
    return agent, lab, contract, original


def current_memory(agent, lab):
    options = publish_research_options(agent, lab.public_spec())
    state = agent._program_state()
    agent._save('research_state.json', json.dumps(state))
    return {**agent.knowledge, 'research_state': state,
        'method_decision': {'research_options': options}, 'program_results': []}


def test_publisher_compiler_and_common_execute_the_actual_whole_state_alternative(tmp_path, monkeypatch):
    agent, lab, contract, original = prepared(tmp_path, monkeypatch)
    memory = current_memory(agent, lab)
    before = dict(agent.costs)
    view = research_segment_options(lab.public_spec(), agent.observations, memory)
    assert not view['excluded']
    assert [item['id'] for item in view['options']] == [
        'native_representation:ordinary', 'native_representation:whole_state']
    for item in view['options']:
        quote = item['complete_quote']
        assert quote['compute_cost'] == sum(quote[key] for key in (
            'preparation_compute', 'bounded_research_compute', 'comparison_compute', 'finish_compute'))
        assert quote['observation_cost'] == 0 and quote['paid_rows'] == 20
    ordinary = select_actions(lab.public_spec(), agent.observations, memory)
    assert ordinary['arguments']['operator'] == 'research_route'
    assert 'search_directions' not in ordinary['arguments']['native_representation_plan']
    proposed = propose_research_action_revisions(ordinary, view)
    assert len(proposed['candidates']) == 1
    plan = proposed['candidates'][0]['research_selection_plan']
    compiled = {}
    exec(export_research_selection_program(plan), compiled)
    chosen = compiled['select_actions'](lab.public_spec(), agent.observations,
        {**memory, 'ordinary_method_path': str(agent.workspace / 'ordinary.py')})
    assert chosen['research_segment_index'] == 0
    assert chosen['arguments']['native_representation_plan']['search_directions'] == ['discrete_state']
    assert agent.costs == before  # Publication, selection and compilation perform no fitting.
    result = agent._execute(chosen['action'], chosen['arguments'], lab)
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    assert [item['direction'] for item in native['search_directions']] == ['discrete_state']
    assert native['comparison_completed'] and native['all_parent_candidates_retained']
    assert native['selection_objective']['data_role'] == 'exposed_development'
    assert set(native['parent_theory_ids']) == {'old', 'constant'}
    assert agent.theory_index['old']['program'] == original.to_dict()


def test_completed_representation_is_not_offered_again_as_conditional_continuation(tmp_path, monkeypatch):
    agent, lab, _, _ = prepared(tmp_path, monkeypatch)
    memory = current_memory(agent, lab)
    ordinary = select_actions(lab.public_spec(), agent.observations, memory)
    result = agent._execute(ordinary['action'], ordinary['arguments'], lab)
    assert result['native_representation']['comparison_completed']
    assert result['native_representation']['research_completion'] == 'bounded_search_completed'
    agent.method_results.append({'proposal': ordinary, 'feedback': result,
        'evidence_before': {'rows': 20}, 'evidence_after': {'rows': 20}})
    memory = current_memory(agent, lab)
    view = research_segment_options(lab.public_spec(), agent.observations, memory)
    assert all(item['id'] != 'native_representation:ordinary' for item in view['options'])
    assert any(item['reason'] == 'same_completed_research' for item in view['excluded'])
    assert all(item['reason'] != 'no_unfinished_or_changed_native_work' for item in view['excluded'])
    # A different question or an unfinished search cannot discharge this
    # question's outstanding method decision, even on equal paid rows.
    saved = deepcopy(agent.method_results[-1])
    agent.method_results[-1]['feedback']['research_question_id'] = 'unrelated'
    view = research_segment_options(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert any(item['id'] == 'native_representation:ordinary' for item in view['options'])
    agent.method_results[-1] = deepcopy(saved)
    agent.method_results[-1]['feedback']['native_representation']['research_completion'] = 'bounded_search_incomplete'
    view = research_segment_options(lab.public_spec(), agent.observations, current_memory(agent, lab))
    assert any(item['id'] == 'native_representation:ordinary' for item in view['options'])
    agent.method_results[-1] = saved
    compared = agent._execute('run', {'operator': 'research_choice',
        'models': ['old.py', *agent.candidate_paths],
        'research_question': deepcopy(agent.knowledge['finite_native_author']['question'])}, lab)
    assert compared['models_completed'] == len(['old.py', *agent.candidate_paths])
    returned = select_actions(lab.public_spec(), agent.observations, current_memory(agent, lab))
    # A zero-error local tie now returns its genuinely distinguishing question;
    # it still cannot restart unrelated conditional research or repeat fitting.
    assert returned['action'] == 'query'
    assert returned['arguments']['after_observation'][0]['operator'] == 'research_route'
    assert returned['complete_quote']['confirmation_rows_reserved'] == agent.config.confirmation_rows


def test_changed_parent_or_underquoted_representation_option_is_not_selected(tmp_path, monkeypatch):
    agent, lab, _, _ = prepared(tmp_path, monkeypatch)
    memory = current_memory(agent, lab)
    options = memory['method_decision']['research_options']
    options[1]['proposal']['arguments']['native_representation_plan']['parent_theory_ids'] = ['missing']
    view = research_segment_options(lab.public_spec(), agent.observations, memory)
    assert view['excluded'][0]['reason'] == 'current_paid_representation_plan_required'
    memory = current_memory(agent, lab)
    memory['method_decision']['research_options'][1]['complete_quote']['max_exported_models'] = 0
    view = research_segment_options(lab.public_spec(), agent.observations, memory)
    assert view['excluded'][0]['reason'] == 'complete_representation_unit_quote_required'


def test_native_quote_reuses_paid_pool_without_hiding_changed_source_or_data(tmp_path, monkeypatch):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program

    agent, lab, _, _ = prepared(tmp_path, monkeypatch)
    rows = len(agent.rows)
    original = (agent.workspace / 'old.py').read_text()
    before = deepcopy(agent.costs)
    expected = 10000 + rows * 20 + rows + agent.config.confirmation_rows
    agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + expected)

    def no_prediction(*args, **kwargs):
        raise AssertionError('a paid quote must not evaluate or fit the old pool again')

    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_prediction)
    options = publish_research_options(agent, lab.public_spec())
    quote = next(item['complete_quote'] for item in options if item['id'] == 'native_representation:ordinary')
    assert quote['status'] == 'available'
    assert quote['compute_cost'] == expected
    assert quote['preparation_compute'] == 0
    assert quote['comparison_compute'] == rows * quote['max_exported_models']
    assert quote['finish_compute'] == rows + agent.config.confirmation_rows
    assert quote['reused_retained_models'] == quote['retained_pool_models'] == 1
    assert set(quote['retained_score_refs']) == {'old.py'}
    assert agent.costs == before

    agent._save('old.py', export_temporal_program(Program('const', value=0.)))
    changed = next(item['complete_quote'] for item in publish_research_options(agent, lab.public_spec())
                   if item['id'] == 'native_representation:ordinary')
    assert changed['status'] == 'complete_unit_unfunded'
    assert changed['comparison_compute'] == rows * (changed['max_exported_models'] + 1)
    assert 'retained_score_refs' not in changed
    agent._save('old.py', original)
    agent.rows.append(Observation({'u': .55}, 0., ()))
    extended = next(item['complete_quote'] for item in publish_research_options(agent, lab.public_spec())
                    if item['id'] == 'native_representation:ordinary')
    assert extended['preparation_compute'] == rows + 1
    assert extended['comparison_compute'] == (rows + 1) * (extended['max_exported_models'] + 1)
    assert 'retained_score_refs' not in extended
    assert agent.costs == before

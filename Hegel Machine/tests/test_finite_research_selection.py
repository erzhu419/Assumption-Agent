"""Finite method controls only: no solver, world call or new research budget."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.finite_native_author import (
    export_finite_native_author, publish_research_options, research_action_semantics,
    research_segment_options, select_research_segment,
)
from hegel_machine.cognition.research_programs import (
    export_research_selection_program, propose_research_action_revisions,
)


def scene():
    question = {'id': 'q', 'question': 'Resolve the retained public failure',
        'meaning_version': 1, 'context_ref': {'public': 'lab'}}
    observations = [{'rows': [{'features': {'u': .5}, 'target': 0., 'history': []}],
        'row_indices': [0], 'action': {'kind': 'trajectory', 'channels': [], 'sequence': [{'u': .5}]}}]
    latest = {'round_index': 0, 'status': 'search_incomplete', 'fit_evidence': [0],
        'candidates': [], 'exported_models': [], 'observed_record_count': 1}
    state = {'questions': [question], 'operator_results': [{
        'operator': 'native_condition_research', 'research_question_id': 'q',
        'native_condition_research': latest}], 'method_results': [],
        'limits': {'compute_budget': 1000, 'observation_budget': 20}, 'costs': {}}
    memory = {'finite_native_author': {'question': question, 'predictor_paths': ['old.py']},
        'research_state': state, 'program_results': []}
    spec = {'controls': ['u'], 'reset': True, 'max_probe_length': 4,
        'feature_bounds': {'u': [-1., 1.]}}
    quote = {'status': 'available', 'compute_cost': 40, 'observation_cost': 0,
        'paid_rows': 1, 'retained_pool_models': 1, 'max_exported_models': 2,
        'remaining_execution_seconds': 10.}
    proposal = {'action': 'run', 'arguments': {'operator': 'native_condition_research',
        'question': {'id': 'q'}, 'max_compute': 20, 'max_candidates': 2}}
    option = {'family': 'N', 'proposal': proposal, 'question_ref': {'id': 'q', 'meaning_version': 1},
        'round_index': 0, 'work_reason': 'search_incomplete', 'complete_quote': quote,
        'proposal_ref': {'file': 'method_decision.json', 'json_path': ['research_options', 0]}}
    return spec, observations, memory, option


def available(spec, observations, memory, options):
    decision = {'research_options': options}
    memory['method_decision'] = decision
    return research_segment_options(spec, observations, memory, decision=decision)


def test_p0_two_candidate_bound_and_local_id_path_deduplication():
    spec, observations, memory, option = scene()
    second = deepcopy(option)
    second['proposal']['arguments']['basis'] = [{'op': 'feature', 'name': 'u'}]
    third = deepcopy(option)
    third['proposal']['arguments']['basis'] = [{'op': 'const', 'value': 1.}]
    duplicate = deepcopy(second)
    duplicate.update(id='renamed', description='New wording', proposal_ref={'file': '/another/path.json'})
    duplicate['proposal']['arguments']['candidate_id'] = 'renamed'
    view = available(spec, observations, memory, [option, second, duplicate, third])
    baseline = deepcopy(option['proposal'])
    baseline['research_question'] = memory['finite_native_author']['question']
    result = propose_research_action_revisions(baseline, view, max_candidates=10)
    assert result['baseline']['id'] == 'P0'
    assert len(result['candidates']) == 2
    assert [item['reason'] for item in result['grouped_trials']] == [
        'same_current_executable_action', 'same_current_executable_action']
    assert all(item['source_support_inherited'] is False for item in result['candidates'])


def test_completed_native_cannot_reopen_by_id_or_false_changed_construction():
    spec, observations, memory, option = scene()
    latest = memory['research_state']['operator_results'][0]['native_condition_research']
    latest['status'] = 'candidates_require_evaluation'
    for reason in ('search_incomplete', 'new_paid_evidence', 'changed_construction'):
        option['work_reason'] = reason
        option['id'] = 'new-looking-id'
        option['construction_before'] = deepcopy(option['proposal'])
        option['construction_before']['arguments']['question'] = {'id': 'old-looking-id'}
        result = available(spec, observations, memory, [option])
        assert result['options'] == []
        assert result['excluded'][0]['reason'] == 'no_unfinished_or_changed_native_work'


def test_changed_rows_pool_or_export_bound_requires_new_full_quote():
    spec, observations, memory, option = scene()
    for change in ('rows', 'pool', 'exports', 'programs'):
        current, selected = deepcopy(memory), deepcopy(option)
        data = deepcopy(observations)
        if change == 'rows':
            data.append(deepcopy(observations[0]))
        elif change == 'pool':
            current['research_state']['exported_candidate_paths'] = ['new.py']
        elif change == 'exports':
            selected['proposal']['arguments']['max_candidates'] = 3
        else:
            selected['proposal']['arguments']['programs'] = [{'op': 'feature', 'name': 'u'}]
        result = available(spec, data, current, [selected])
        assert result['options'] == []
        assert result['excluded'][0]['reason'] in {
            'fresh_complete_quote_required', 'changed_export_bound_requires_new_quote'}
        assert selected['proposal']['arguments']['max_compute'] == 20


def boundary_option(option, *, sequence=None):
    query = {'action': 'query', 'arguments': {'actions': [{'kind': 'trajectory',
        'channels': [], 'sequence': sequence or [{'u': .5}]}],
        'after_observation': [{'operator': 'native_condition_research',
            'research_question': {'id': 'q', 'meaning_version': 1},
            'boundary_update': {'round_index': 0, 'proposal_index': 0}}]}}
    return {**deepcopy(option), 'family': 'Q', 'proposal': query,
        'predicted_separation': 1., 'matched_action_pairs': []}


def test_paid_answer_reuses_unfinished_boundary_but_completed_analysis_is_excluded():
    spec, observations, memory, option = scene()
    query = boundary_option(option)
    assert available(spec, observations, memory, [query])['options']
    memory['research_state']['method_results'] = [{'proposal': query['proposal'], 'feedback': {
        'record_indices': [0], 'after_observation_results': [{
            'operator': 'native_condition_research', 'research_question_id': 'q',
            'question_meaning_version': 1, 'arguments': query['proposal']['arguments']['after_observation'][0],
            'native_condition_research': {'boundary_update': {'status': 'boundary_refined',
                'observation_index': 0}}}]}}]
    result = available(spec, observations, memory, [query])
    assert not result['options']
    assert result['excluded'][0]['reason'] == 'same_analysis_already_completed'


def test_pair_quote_includes_all_unpaid_corners_and_requires_targeted_callback():
    spec, observations, memory, option = scene()
    query = boundary_option(option, sequence=[{'u': .25}])
    query['proposal']['arguments']['actions'].append({'kind': 'trajectory', 'channels': [],
        'sequence': [{'u': -.25}]})
    query['matched_action_pairs'] = [{'reference_action_index': 0, 'action_index': 1,
        'predicted_separation': 1.}]
    query['complete_quote']['observation_cost'] = 1
    assert available(spec, observations, memory, [query])['excluded'][0]['reason'] == \
        'complete_remaining_pair_quote_required'
    query['complete_quote']['observation_cost'] = 2
    assert available(spec, observations, memory, [query])['options']
    query['proposal']['arguments'].pop('after_observation')
    assert available(spec, observations, memory, [query])['excluded'][0]['reason'] == \
        'bound_distinction_and_analysis_required'


def test_r_replay_binding_is_required_and_completed_id_change_does_not_resell():
    spec, observations, memory, option = scene()
    option.update(family='R', proposal={'action': 'run', 'arguments': {
        'operator': 'theory_transfer', 'candidate_id': 'old', 'theory_id': 'draft',
        'allocation': {'compute': 20}}})
    assert available(spec, observations, memory, [option])['excluded'][0]['reason'] == \
        'executable_program_binding_required'
    option['program_binding'] = {'terms': [{'op': 'feature', 'name': 'u'}]}
    original = deepcopy(option['proposal'])
    original['arguments']['candidate_id'] = 'renamed'
    original['arguments']['theory_id'] = 'another-id'
    original['program_binding'] = deepcopy(option['program_binding'])
    memory['research_state']['method_results'] = [{'proposal': original,
        'evidence_before': {'rows': 1}, 'feedback': {'execution_complete': True}}]
    assert available(spec, observations, memory, [option])['excluded'][0]['reason'] == 'same_completed_research'


def test_short_program_executes_current_proposal_then_returns_to_ordinary_closure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec, observations, memory, option = scene()
    offered = available(spec, observations, memory, [option])['options'][0]
    plan = {'first_segment': offered['semantic_action'], 'ordinary_method_path': 'ordinary.py'}
    ordinary = export_finite_native_author() + "\ndef select_actions(*args): return {'stop': 'ordinary'}\n"
    (tmp_path / 'ordinary.py').write_text(ordinary)
    memory['ordinary_method_path'] = 'ordinary.py'
    namespace = {}
    exec(export_research_selection_program(plan), namespace)
    proposal = namespace['select_actions'](spec, observations, memory)
    assert proposal['arguments']['operator'] == 'native_condition_research'
    assert proposal['research_segment_index'] == 0
    memory['program_results'] = [{'proposal': proposal, 'feedback': {'status': 'search_incomplete'}}]
    assert namespace['select_actions'](spec, observations, memory) == {'stop': 'ordinary'}
    compulsory = {'action': 'run', 'arguments': {'operator': 'research_choice', 'models': ['old.py']}}
    assert select_research_segment(spec, observations, memory, plan=plan,
        ordinary_select_actions=lambda *args: compulsory) == compulsory


def test_one_guard_preserves_unknown_and_an_unseen_legal_branch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec, observations, memory, option = scene()
    semantic = available(spec, observations, memory, [option])['options'][0]['semantic_action']
    plan = {'condition': {'op': 'ge', 'children': [{'op': 'feature', 'name': 'public_0'},
        {'op': 'const', 'value': .5}]}, 'when_true': {'first_segment': semantic},
        'when_false': {}, 'condition_feature_paths': [{'path': ['research', 'stage'], 'kind': 'number'}]}
    (tmp_path / 'ordinary.py').write_text(export_finite_native_author() +
        "\ndef select_actions(*args): return {'stop': 'ordinary'}\n")
    memory['ordinary_method_path'] = 'ordinary.py'
    namespace = {}
    exec(export_research_selection_program(plan), namespace)
    memory['method_condition_support'] = {'can_apply': True}
    memory['method_condition_features'] = {'public_0': None}
    assert namespace['select_actions'](spec, observations, memory) == {'stop': 'ordinary'}
    memory['method_condition_features'] = {'public_0': 0.}
    assert namespace['select_actions'](spec, observations, memory) == {'stop': 'ordinary'}
    memory['method_condition_features'] = {'public_0': 1.}
    assert namespace['select_actions'](spec, observations, memory)['research_segment_index'] == 0
    memory['method_condition_support'] = {'can_apply': False, 'outside_coverage': ['public_0']}
    assert namespace['select_actions'](spec, observations, memory) == {'stop': 'ordinary'}
    memory.pop('method_condition_support')
    assert namespace['select_actions'](spec, observations, memory) == {'stop': 'ordinary'}
    # The false branch's equality on this panel never deletes the true branch.
    assert plan['when_true']['first_segment'] == semantic


def test_old_ordinary_carrier_is_explicit_work_not_native_replay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'old.py').write_text("def select_actions(*args): return {'stop': 'old'}\n")
    namespace = {}
    exec(export_research_selection_program(), namespace)
    result = namespace['select_actions']({}, [], {'ordinary_method_path': 'old.py'})
    assert result['stop']['request']['reason'] == 'ordinary_method_selection_adapter_required'


def test_real_feedback_can_turn_once_to_another_current_segment_without_counting_closure():
    spec, observations, memory, option = scene()
    replay = deepcopy(option)
    replay.update(family='R', program_binding={'terms': [{'op': 'feature', 'name': 'u'}]},
        proposal={'action': 'run', 'arguments': {'operator': 'theory_transfer',
            'candidate_id': 'old-method', 'allocation': {'compute': 20}}})
    selected = available(spec, observations, memory, [option, replay])['options']
    first, second = [item['semantic_action'] for item in selected]
    ordinary = lambda *args: {'stop': 'ordinary'}
    plan = {'condition': {'op': 'const', 'value': 1.},
        'when_true': {'first_segment': first, 'second_segment': second},
        'when_false': {'first_segment': first}}
    before = select_research_segment(spec, observations, memory, plan=plan,
        ordinary_select_actions=ordinary, condition=lambda memory: True)
    assert before['research_segment_index'] == 0
    memory['program_results'] = [{'proposal': before, 'feedback': {'status': 'search_incomplete'}},
        {'proposal': {'action': 'run', 'arguments': {'operator': 'research_choice'}},
         'feedback': {'models_completed': 1}}]
    after = select_research_segment(spec, observations, memory, plan=plan,
        ordinary_select_actions=ordinary, condition=lambda memory: True)
    assert after['arguments']['operator'] == 'theory_transfer'
    assert after['research_segment_index'] == 1
    assert select_research_segment(spec, observations, memory, plan=plan,
        ordinary_select_actions=ordinary, condition=lambda memory: False) == {'stop': 'ordinary'}
    memory['program_results'].append({'proposal': after, 'feedback': {'status': 'completed'}})
    assert select_research_segment(spec, observations, memory, plan=plan,
        ordinary_select_actions=ordinary, condition=lambda memory: True) == {'stop': 'ordinary'}


def test_finished_negative_generation_retains_p0_without_new_candidate():
    spec, observations, memory, option = scene()
    view = available(spec, observations, memory, [option])
    result = propose_research_action_revisions(option['proposal'], view)
    assert result['status'] == 'no_distinct_legal_alternative'
    assert result['baseline']['status'] == 'unchanged_ordinary_program'
    assert result['candidates'] == []


def test_common_comparison_or_parent_delivery_cannot_be_a_method_replacement_witness():
    spec, observations, memory, option = scene()
    view = available(spec, observations, memory, [option])
    for proposal in ({'action': 'finish', 'arguments': {'model': 'parent.py'}},
                     {'action': 'evaluate', 'arguments': {'model': 'parent.py'}},
                     {'action': 'run', 'arguments': {'operator': 'research_choice', 'models': ['parent.py']}}):
        result = propose_research_action_revisions(proposal, view)
        assert result['status'] == 'common_closure_not_a_research_selection'
        assert result['candidates'] == [] and result['baseline']['id'] == 'P0'


def test_compiler_does_not_allow_nested_branch_or_second_without_first():
    with pytest.raises(ValueError, match='requires a first'):
        export_research_selection_program({'second_segment': {'action': 'run'}})
    with pytest.raises(ValueError, match='one condition'):
        export_research_selection_program({'condition': {'op': 'const', 'value': 1.},
            'when_true': {'condition': {'op': 'const', 'value': 0.}}})


def test_actual_r_publisher_prices_preparation_both_seeds_full_pool_and_parent(monkeypatch):
    from hegel_machine.cognition import structural_inquiry
    monkeypatch.setattr(structural_inquiry, 'native_condition_method', lambda agent: None)
    roles = {'u': {'role': 'control', 'unit': 'V', 'time': 'current'}}
    proposal = {'action': 'run', 'arguments': {'operator': 'theory_transfer',
        'candidate_id': 'method', 'theory_id': 'draft', 'mapping': {'role_0': 'u'},
        'role_signature': roles, 'output_unit': 'V', 'context_ref': {'public': 'lab'},
        'research_program_fitting': {'validation_records': [1]},
        'allocation': {'compute': 20}}}
    question = {'id': 'q', 'meaning_version': 1}
    child = {'id': 'method-use', 'parent_id': 'q',
        'routing_execution': {'result': {'next_action': proposal}}}
    candidate = {'kind': 'research_program', 'role_signature': {'role_0': roles['u']},
        'output_unit': 'V', 'research_program_contract': {'operation': 'refit_selected_terms'},
        'scope': {'context_ref': {'public': 'lab'}},
        'source_seed_inputs': [{}, {}], 'research_program_source_refs': [{'file': 'seedA'}, {'file': 'seedB'}],
        'seed_policy': 'retain_both'}
    agent = SimpleNamespace(knowledge={'finite_native_author': {'predictor_paths': ['a.py', 'b.py']}},
        questions=[question, child], _active_question_id='q', rows=[0, 0, 0],
        observations=[{}, {}], candidate_paths=[], method_results=[], operator_results=[],
        config=SimpleNamespace(compute_budget=100, execution_seconds=30., confirmation_rows=2,
            observation_budget=20), costs={'compute_cost': 0, 'python_execution_seconds': 0.,
            'observation_cost': 0}, _inquiry_stack=[],
        theory_consolidation={'candidates': {'method': candidate}, 'executions': {}})
    option, = publish_research_options(agent)
    quote = option['complete_quote']
    assert quote['preparation_compute'] == 6
    assert quote['bounded_research_compute'] == 20
    assert quote['max_exported_models'] == 2
    assert quote['comparison_compute'] == 12
    assert quote['finish_compute'] == 5
    assert quote['compute_cost'] == 43
    agent.operator_results = [{'operator': 'refit', 'status': 'refit_supported_on_validation',
        'path': 'paid_validated.py'}]
    agent.method_results = [{'proposal': {'action': 'run', 'arguments': {
        'operator': 'research_choice', 'models': ['a.py', 'b.py']}}, 'feedback': {
        'operator': 'research_choice', 'research_question_id': 'q', 'question_meaning_version': 1,
        'paid_rows': 3, 'evaluations': [{'path': path} for path in ('a.py', 'b.py', 'paid_validated.py')]}}]
    reused, = publish_research_options(agent)
    assert reused['complete_quote']['retained_pool_models'] == 3
    assert reused['complete_quote']['preparation_compute'] == 0
    assert reused['complete_quote']['compute_cost'] == 40
    agent.operator_results, agent.method_results = [], []
    before = deepcopy(option)
    agent.candidate_paths.append('exposed.py')
    changed, = publish_research_options(agent)
    assert changed['complete_quote']['compute_cost'] == 49
    assert option == before
    agent.theory_consolidation['executions']['old-name'] = {'basis': {'binding': {
        'paid_cutoff': {'records': 2, 'rows': 3}, 'mapping': {'role_0': 'u'},
        'source_refs': candidate['research_program_source_refs'],
        'research_program_contract': candidate['research_program_contract']},
        'fitting': {'validation_records': [1]}}, 'result': {'search_completed': True}}
    assert publish_research_options(agent) == []


def test_publisher_does_not_sell_old_short_deferred_joint_as_a_future_quote(monkeypatch):
    from hegel_machine.cognition import structural_inquiry
    monkeypatch.setattr(structural_inquiry, 'native_condition_method', lambda agent: None)
    spec, observations, memory, option = scene()
    experiment = boundary_option(option)
    callback = experiment['proposal']['arguments']['after_observation'][0]
    callback.update(max_compute=20, max_candidates=1)
    query = {'query_action': experiment['proposal'], 'predicted_separation': 1.,
        'matched_action_pairs': [], 'purpose': 'conditional_boundary'}
    old_joint = {'query_action': {'action': 'query', 'arguments': {'actions': [
        {'kind': 'trajectory', 'sequence': [{'u': .25}], 'channels': []}]}},
        'research_continuation': {'operator': 'native_condition_research', 'max_compute': 50_000}}
    question = {**memory['finite_native_author']['question'], 'native_condition_research': [{
        'round_index': 0, 'candidates': [], 'proposed_experiments': [query, old_joint]}]}
    agent = SimpleNamespace(knowledge={'finite_native_author': memory['finite_native_author']},
        questions=[question], _active_question_id='q', rows=[0], observations=observations,
        candidate_paths=[], method_results=[], operator_results=[], _inquiry_stack=[],
        theory_consolidation={'candidates': {}, 'executions': {}},
        config=SimpleNamespace(compute_budget=100, execution_seconds=30., confirmation_rows=2,
            observation_budget=20), costs={'compute_cost': 0, 'python_execution_seconds': 0.,
            'observation_cost': 0})
    published = publish_research_options(agent, spec)
    assert len(published) == 1 and published[0]['family'] == 'Q'
    assert published[0]['complete_quote']['observation_cost'] == 0
    assert published[0]['complete_quote']['bounded_research_compute'] == 20
    assert published[0]['proposal']['arguments']['after_observation'][0] == callback


def test_future_query_quote_uses_planned_shape_not_old_allowance_or_future_targets(tmp_path, monkeypatch):
    from hegel_machine.cognition import structural_inquiry
    from hegel_machine.cognition.finite_native_author import _current_query_action
    from hegel_machine.cognition.conditional_coordinates import conditional_search_allowance
    from hegel_machine.cognition.models import Observation
    from test_native_complete_research_unit import paid_scene

    monkeypatch.chdir(tmp_path)
    agent, question, latest, _ = paid_scene(tmp_path)
    question['context_ref'] = {'public': 'lab'}
    agent.knowledge['finite_native_author']['question'] = deepcopy(question)
    agent.config = agent.config.__class__(**{**vars(agent.config), 'observation_budget': 300})
    agent.observations = [{'rows': [{'features': {'u': 0.}, 'history': [], 'target': 0.}
        for _ in range(14 if index < 9 else 19)],
        'row_indices': list(range(index * 14, index * 14 + (14 if index < 9 else 19)))}
        for index in range(10)]
    experiment = {'query_action': {'action': 'query', 'arguments': {'actions': [
        {'kind': 'trajectory', 'channels': [], 'sequence': [{'u': .3}, {'u': .4}]},
        {'kind': 'trajectory', 'channels': [], 'sequence': [{'u': .3}, {'u': .8}]}]}},
        'matched_action_pairs': [{'reference_action_index': 0, 'action_index': 1,
            'predicted_separation': 1.}],
        'research_continuation': {'operator': 'native_condition_research',
            'research_question': {'id': question['id']}, 'max_candidates': 2,
            'max_compute': 7, 'matched_pairs': [{'reference_record_index': 10, 'record_index': 11}]}}
    latest = question['native_condition_research'][0]
    latest.update(status='candidates_require_evaluation', proposed_experiments=[experiment])
    agent.rows = [Observation({'u': 0.}, 0., tuple({'u': 0.} for _ in range(11))) for _ in range(145)]
    spec = {'controls': ['u'], 'reset': True, 'feature_bounds': {'u': [-1., 1.]}, 'max_probe_length': 24}
    before = deepcopy(agent.costs)
    option = next(item for item in publish_research_options(agent, spec) if item.get('family') == 'Q')
    quoted = option['future_compute_continuation']['research_unit']
    assert quoted['quoted_paid_rows'] == 149
    assert quoted['bounded_joint_attempt_estimate'] == conditional_search_allowance(1, 149, 11)
    assert quoted['search_compute'] > 7
    assert option['complete_quote']['observation_cost'] == 4
    assert option['proposal']['arguments']['after_observation'][0]['matched_pairs'] == [
        {'reference_record_index': 10, 'record_index': 11}]
    assert option['proposal']['arguments']['after_observation'][0]['max_compute'] == quoted['search_compute']
    assert agent.costs == before and len(agent.rows) == 145
    state = {'questions': agent.questions, 'operator_results': agent.operator_results,
        'limits': vars(agent.config), 'costs': agent.costs}
    memory = {'finite_native_author': agent.knowledge['finite_native_author'],
        'research_state': state, 'method_decision': {'research_options': [option]}}
    # P0 receives exactly the same public callback repair; it is not PR3 content.
    ordinary = _current_query_action(experiment, spec, agent.observations, memory, state, question)
    fresh = research_segment_options(spec, agent.observations, memory, state=state)['options'][0]
    assert ordinary == fresh['proposal']
    assert propose_research_action_revisions(ordinary, [fresh])['candidates'] == []

"""Question identity and compilation checks, without providers or new labels."""
from copy import deepcopy

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, open_assigned_question
from hegel_machine.cognition.question_semantics import (
    adopt_claim_instance, adopt_question_semantics, compile_question, validate_answer_type,
)


def question():
    return {'id': 'response_difference', 'question': 'What is the paired response difference?',
            'kind': 'value', 'subject_refs': [{'type': 'paid_scene', 'id': 'C5'}],
            'context_ref': {'scene': 'C5', 'output_unit': 'V'}, 'answer_type': 'number',
            'answer_resolution_ref': {'relation': 'eq', 'atol': .01},
            'query_semantics': {'mode': 'model_query', 'query_expression': {
                'op': 'Difference', 'left': {'op': 'Readout', 'record_index': 1, 'target': 'response'},
                'right': {'op': 'Readout', 'record_index': 0, 'target': 'response'}}},
            'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'parent', 'path': 'plan.py'}]}


def test_T01_changed_objective_preserves_old_answer_and_bank_once(tmp_path):
    agent = InquiryLLMAgent(None, tmp_path, arm='hegel')
    current = open_assigned_question(agent, question())
    assignment = deepcopy(current['assignment'])
    current.update(semantic_answer={'value': 2., 'version': 1, 'status': 'answered'},
                   native_condition_research={'rounds': [{'evidence_records': [0, 1]}]},
                   native_structure_planning=[{'planning_complete': True,
                       'question_meaning_version': 1, 'source_round_index': 0}],
                   support_paths=[[{'question_id': 'calibration', 'answer_version': 1}]],
                   pending_query={'actions': [{'kind': 'measure', 'point': {'u': 2.}}]},
                   pending_query_source={'question_id': 'child', 'answer_version': 1},
                   parent_use_history=[{'pending_query': {'actions': []}}])
    changed = deepcopy(question()['query_semantics'])
    changed['query_expression']['op'] = 'RelationResidual'
    adopted = open_assigned_question(agent, {'id': current['id'], 'query_semantics': changed,
                                           'assignment': {'question': 'replacement'}})
    assert adopted is current and len(agent.questions) == 1
    assert current['meaning_version'] == 2 and current['assignment'] == assignment
    old = current['meaning_history'][0]
    assert old['semantic_answer']['value'] == 2.
    assert old['native_condition_research']['rounds'][0]['evidence_records'] == [0, 1]
    assert old['native_structure_planning'][0]['question_meaning_version'] == 1
    assert old['support_paths'][0][0]['answer_version'] == 1
    assert old['pending_query']['actions'][0]['point'] == {'u': 2.}
    assert old['pending_query_source']['answer_version'] == 1 and len(old['parent_use_history']) == 1
    assert all(key not in current for key in ('pending_query', 'pending_query_source', 'parent_use_history'))
    assert 'native_condition_research' not in current and 'semantic_answer' not in current
    assert 'native_structure_planning' not in current
    assert 'support_paths' not in current and 'meaning_history' not in old
    open_assigned_question(agent, {'id': current['id'], 'context_ref': {'scene': 'C5', 'output_unit': 'mV'}})
    assert current['meaning_version'] == 3
    assert 'native_condition_research' not in current['meaning_history'][1]


def test_T02_solver_fit_and_new_evidence_do_not_change_question_meaning():
    original = adopt_question_semantics(None, question())
    original['semantic_answer'] = {'status': 'answered', 'value': 2.}
    updated = adopt_question_semantics(original, {'id': original['id'], 'question': 'Paired difference?',
        'algorithm': 'new optimizer', 'fit_parameters': {'max_iter': 80}, 'evidence': [0, 1, 2],
        'query_semantics': {**original['query_semantics'], 'solver': 'new optimizer'}})
    assert updated['meaning_version'] == 1 and updated['semantic_answer']['value'] == 2.
    assert 'meaning_history' not in updated


def test_native_evaluation_contract_changes_meaning_and_preserves_previous_answer():
    source = question()
    source['query_semantics']['evaluation'] = {'kind': 'native_evaluation', 'fallback': .2,
        'source_ref': {'archive': 'history.sqlite3', 'unit': 'D', 'member': 'sequence.json',
                       'field': 'episodes[6].fallback'}}
    current = adopt_question_semantics(None, source)
    current['semantic_answer'] = {'status': 'answered', 'value': .2}
    changed = deepcopy(current['query_semantics'])
    changed['evaluation']['fallback'] = .3
    revised = adopt_question_semantics(current, {'id': current['id'], 'query_semantics': changed})
    assert revised['meaning_version'] == 2 and 'semantic_answer' not in revised
    assert revised['meaning_history'][0]['semantic_answer']['value'] == .2
    assert revised['meaning_history'][0]['query_semantics']['evaluation']['fallback'] == .2
    changed['evaluation']['source_ref']['field'] = 'episodes[7].fallback'
    relocated = adopt_question_semantics(revised, {'id': current['id'], 'query_semantics': changed})
    assert relocated['meaning_version'] == 3
    assert compile_question(relocated)['status'] == 'ready'


@pytest.mark.parametrize('evaluation', [
    {'kind': 'native_evaluation', 'source_ref': {'file': 'retained.json', 'field': 'fallback'}},
    {'kind': 'native_evaluation', 'fallback': float('nan'),
     'source_ref': {'file': 'retained.json', 'field': 'fallback'}},
    {'kind': 'native_evaluation', 'fallback': .2},
    {'kind': 'native_evaluation', 'fallback': .2, 'source_ref': {'field': 'fallback'}},
])
def test_native_evaluation_requires_real_finite_fallback_and_source_location(evaluation):
    source = question()
    source['query_semantics']['evaluation'] = evaluation
    with pytest.raises(ValueError, match='native evaluation'):
        adopt_question_semantics(None, source)


def test_T03_fit_creates_an_instance_of_same_explicit_family():
    first = adopt_claim_instance(None, {'claim_id': 'linear_response', 'body_ref': 'linear.py',
        'parameters': {'slope': .5}, 'training_cutoff': 4})
    second = adopt_claim_instance(first, {'family_id': 'linear_response',
        'parameters': {'slope': 1.}, 'training_cutoff': 8})
    assert second['family_id'] == first['family_id'] == 'linear_response'
    assert second['instance_version'] == 2
    assert second['instance_history'][0]['parameters'] == {'slope': .5}
    assert adopt_claim_instance(second, {'family_id': 'linear_response', 'evidence_refs': [9]})['instance_version'] == 2


def test_explicit_different_families_remain_separate():
    first = adopt_claim_instance(None, {'family_id': 'linear', 'body_ref': 'same.py'})
    second = adopt_claim_instance(first, {'family_id': 'quadratic', 'body_ref': 'same.py'})
    assert second['family_id'] == 'quadratic' and second['instance_version'] == 1
    assert second['family_history'][0]['family_id'] == 'linear'
    with pytest.raises(ValueError, match='explicit claim'):
        adopt_claim_instance(None, {'body_ref': 'same.py'})


def test_T04_context_change_invalidates_only_its_actual_input_consumers(tmp_path):
    agent = InquiryLLMAgent(None, tmp_path, arm='hegel')
    current = open_assigned_question(agent, question())
    current['executable_answer'] = {'status': 'supported', 'parameter_region': {'lower': 0., 'upper': 1.}}
    agent.research_input_bindings = {'bound.py': {'offset': current['id']},
                                     'unrelated.py': {'offset': 'another_question'}}
    agent.research_candidates = {'bound.py': {'status': 'evaluated', 'current_evaluation': {'mse': 0.}},
                                 'unrelated.py': {'status': 'evaluated', 'current_evaluation': {'mse': 1.}}}
    unrelated = deepcopy(agent.research_candidates['unrelated.py'])
    open_assigned_question(agent, {'id': current['id'], 'context_ref': {'scene': 'C5', 'output_unit': 'mV'}})
    assert agent.research_candidates['bound.py']['status'] == 'needs_rebuild'
    assert 'current_evaluation' not in agent.research_candidates['bound.py']
    assert agent.research_candidates['unrelated.py'] == unrelated
    assert 'executable_answer' not in current
    assert current['meaning_history'][0]['executable_answer']['status'] == 'supported'


def test_legacy_idea_keeps_id_and_original_assignment_without_forcing_readiness(tmp_path):
    agent = InquiryLLMAgent(None, tmp_path, arm='hegel')
    current = open_assigned_question(agent, {'id': 'legacy', 'question': 'Can this pattern help?'})
    open_assigned_question(agent, {'id': 'legacy', 'question': 'Narrow the pattern to this source'})
    assert current['meaning_version'] == 1
    assert current['assignment']['question'] == 'Can this pattern help?'
    assert compile_question(current)['status'] == 'idea'


def test_use_site_addition_does_not_rewrite_the_answer_meaning():
    original = adopt_question_semantics(None, question())
    original['semantic_answer'] = {'status': 'answered', 'value': 2.}
    updated = adopt_question_semantics(original, {'id': original['id'], 'use_sites': [
        *original['use_sites'], {'kind': 'bind_parameter_region', 'model': 'other.py', 'slot': 'gain'}]})
    assert updated['meaning_version'] == 1 and updated['semantic_answer'] == original['semantic_answer']
    assert len(compile_question(updated)['use_sites']) == 2


def test_compile_passes_concrete_query_without_running_or_mutating_it():
    source = question()
    before = deepcopy(source)
    compiled = compile_question(source)
    assert compiled['status'] == 'ready'
    assert compiled['query'] == source['query_semantics']['query_expression']
    assert compiled['question_id'] == source['id'] and compiled['meaning_version'] == 1
    compiled['query']['op'] = 'changed'
    assert source == before


def test_compile_requires_scientific_answer_and_resolution_contracts():
    source = question()
    source.pop('answer_type')
    assert compile_question(source)['reason'] == 'missing_answer_type'
    source['answer_type'] = 'number'
    source.pop('answer_resolution_ref')
    assert compile_question(source)['reason'] == 'missing_answer_resolution'
    source['kind'] = 'score_improvement'
    assert compile_question(source)['status'] == 'idea'
    source.update(kind='value', answer_type='anything', answer_resolution_ref='loss')
    assert compile_question(source)['reason'] == 'unsupported_answer_type'


def test_T12_no_parent_use_is_deferred_but_declared_theory_goal_is_allowed():
    source = question()
    source['use_sites'] = []
    assert compile_question(source)['status'] == 'deferred'
    assert compile_question(source, {'purpose': 'theory_learning'})['status'] == 'ready'
    assert compile_question(source, {'purpose': 'method_learning'})['status'] == 'ready'


def test_T19_missing_authored_relation_returns_explicit_work_request_without_provider():
    source = question()
    source.update(kind='discover_object', answer_type='relation',
                  query_semantics={'mode': 'solution_relation'})
    result = compile_question(source)
    assert result['status'] == 'author_work_request' and result['query'] is None
    assert result['request'] == {'type': 'AuthorWorkRequest', 'object_type': 'relation',
        'required_contract': {'answer_type': 'relation',
            'answer_resolution_ref': source['answer_resolution_ref'], 'query_mode': 'solution_relation'}}


def test_answer_types_match_actual_parameter_region_and_numeric_query_results():
    assert validate_answer_type(2., 'number')
    assert not validate_answer_type(True, 'number')
    assert not validate_answer_type(float('nan'), 'number')
    assert validate_answer_type({'lower': .9, 'upper': 1.1}, 'interval')
    assert validate_answer_type({'slope': {'lower': .9, 'upper': 1.1}}, 'interval')
    assert not validate_answer_type({'lower': 1.1, 'upper': .9}, 'interval')
    assert validate_answer_type(True, 'boolean_claim')
    assert not validate_answer_type({'status': 'supported'}, 'boolean_claim')


def test_partial_updates_cannot_select_their_own_version_or_erase_history():
    first = adopt_question_semantics(None, question())
    second = adopt_question_semantics(first, {'id': first['id'], 'answer_resolution_ref': {'atol': .001}})
    third = adopt_question_semantics(second, {'id': first['id'], 'meaning_version': 99, 'meaning_history': []})
    assert third['meaning_version'] == 2 and len(third['meaning_history']) == 1
    assert first['answer_resolution_ref'] == {'relation': 'eq', 'atol': .01}

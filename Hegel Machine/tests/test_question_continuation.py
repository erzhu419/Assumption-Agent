"""PR1 behavior through the current Common/provider=None execution path."""

from copy import deepcopy

import pytest

from test_evidence_procedure_execution import prepared
from test_executable_evidence_binding import return_region
from test_inquiry_revision import action


def step(agent, lab, identifier='slope_question'):
    return agent._execute('run', {'operator': 'research_step',
        'research_question': {'id': identifier}}, lab)


def region_use(agent):
    question = next(q for q in agent.questions if q['id'] == 'slope_question')
    question['use_sites'] = [{'kind': 'bind_parameter_region', 'model': 'parent.py',
        'slot': 'slope', 'refit': {'validation_records': [2], 'path': 'shared/question_refit'}}]
    return question


def test_T08_T11_checked_region_rebuilds_actual_fit_and_same_answer_does_not_repeat(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, 0., .5)
    question = region_use(agent)
    before = deepcopy(agent.costs)
    first = step(agent, lab)
    fit = first['uses'][0]['feedback']
    assert first['parent_revision_completed']
    assert fit['validation']['mse'] > 0
    assert fit['validation']['research_bindings'][0]['actual_parameters']['slope'] == .25
    old_path = fit['tested_model']
    after_first = deepcopy(agent.costs)
    assert step(agent, lab)['status'] == 'reused_answer'
    assert agent.costs == after_first
    # Neither .25 nor the earlier grid is in the new region. The current parent
    # consumes that region and actually fits, rather than rejecting its family.
    return_region(agent, lab, .9, 1.1)
    second = step(agent, lab)
    updated = second['uses'][0]['feedback']
    assert updated['tested_model'] != old_path
    assert updated['validation']['mse'] == 0
    assert updated['validation']['research_bindings'][0]['actual_parameters']['slope'] == 1.
    assert (tmp_path / old_path).exists()
    assert question['semantic_answer_history'][0]['value']['upper'] == .5
    assert agent.costs['observation_cost'] == before['observation_cost'] == lab.paid
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_T12_checked_answer_without_use_does_not_complete_parent(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1)
    before = deepcopy(agent.research_candidates)
    result = step(agent, lab)
    assert result['status'] == 'answer_ready_no_parent_use'
    assert result['parent_revision_completed'] is False
    assert agent.research_candidates == before


def test_T03_actual_parameter_refit_versions_instance_without_new_family(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, 0., .5)
    agent._execute('evaluate', {'model': 'parent.py',
        'research_inputs': {'slope': 'slope_question'},
        'claim': {'family_id': 'fixture_linear_response', 'body_ref': 'parent.py'}}, lab)
    first = deepcopy(agent.research_candidates['parent.py']['claim_instance'])
    return_region(agent, lab, .9, 1.1)
    agent._execute('evaluate', {'model': 'parent.py'}, lab)
    second = agent.research_candidates['parent.py']['claim_instance']
    assert second['family_id'] == first['family_id']
    assert second['instance_version'] == first['instance_version'] + 1
    assert second['instance_history'][0]['parameters'] != second['parameters']
    assert second['parameters']['parent.py']['slope'] == 1.
    assert not agent.confirmations


def test_T09_T10_alternative_support_path_preserves_fit_until_both_fail(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1, question='calibration_a')
    return_region(agent, lab, .9, 1.1, question='calibration_b')
    return_region(agent, lab, .9, 1.1)
    question = region_use(agent)
    question['support_paths'] = [[{'question_id': 'calibration_a', 'answer_version': 1}],
                                 [{'question_id': 'calibration_b', 'answer_version': 1}]]
    assert step(agent, lab)['parent_revision_completed']
    candidate = deepcopy(agent.research_candidates['parent.py'])
    # A genuinely revised premise invalidates its path. B still independently
    # supplies this declared use; no parent fit is discarded or rerun.
    return_region(agent, lab, 2., 3., question='calibration_a')
    costs = deepcopy(agent.costs)
    assert step(agent, lab)['status'] == 'reused_answer'
    assert agent.research_candidates['parent.py'] == candidate
    assert agent.costs == costs
    return_region(agent, lab, 2., 3., question='calibration_b')
    result = step(agent, lab)
    assert result['reason'] == 'premise_unknown'
    assert agent.research_candidates['parent.py']['status'] == 'needs_rebuild'
    blocked = agent._execute('evaluate', {'model': 'parent.py'}, lab)
    assert blocked['status'] == 'dependency_unresolved'
    assert 'unresolved premises' in blocked['error']


def test_T19_missing_authored_query_returns_to_outer_author_once(tmp_path):
    agent, lab = prepared(tmp_path)
    agent._execute('write', {'path': 'shared/ask.py', 'code': '''def select_actions(spec, observations, memory):
    return {'action':'run','arguments':{'operator':'research_step','research_question':{
        'id':'new_structure','question':'Find a structure explaining these paid errors',
        'parent_id':'task','depends_on':[], 'kind':'discover_object','answer_type':'program',
        'use_sites':[{'kind':'supply_component','model':'parent.py','slot':'state'}]}}}
'''}, lab)
    before = deepcopy(agent.costs)
    result = agent._execute('run', {'operator': 'procedure', 'path': 'shared/ask.py',
                                  'max_steps': 5}, lab)
    assert result['status'] == 'author_work_request'
    assert len(result['steps']) == 1
    assert agent.costs['model_calls'] == 0
    assert agent.costs['observation_cost'] == before['observation_cost']
    assert agent._request(lab)['online_research']['questions'][-1]['pending_author_work']['question_id'] == 'new_structure'


def test_T10_ungrounded_cycle_cannot_support_parameter_use(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1)
    question = region_use(agent)
    question['support_paths'] = [[{'question_id': 'slope_question'}]]
    result = step(agent, lab)
    assert result['status'] == 'scoped_unknown'
    assert 'parent.py' not in agent.research_input_bindings


def paired_question(agent):
    def readout(step_index):
        record = agent.observations[0]
        row = record['rows'][step_index]
        return {'op': 'Readout', 'record_index': 0, 'step': step_index, 'target': 'response',
            'binding': {'action': deepcopy(record['action']), 'features': deepcopy(row['features']),
                        'history': deepcopy(list(row.get('history', [])))}}
    return {'id': 'paired', 'question': 'What is this paid response difference?',
        'parent_id': 'parent', 'depends_on': [], 'kind': 'value',
        'context_ref': {'output_unit': 'V'}, 'answer_type': 'number',
        'answer_resolution_ref': {'atol': 0.},
        'query_semantics': {'mode': 'model_query', 'query_expression': {
            'op': 'Difference', 'left': readout(1), 'right': readout(0)}},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'parent', 'path': 'shared/next.py'}]}


def parent_plan(agent, lab):
    parent = agent._open_question({'id': 'parent', 'question': 'Choose the next response experiment'})
    parent['pending_query'] = {'actions': [action(99)]}
    agent._execute('write', {'path': 'shared/next.py', 'code': '''import json
def select_actions(spec, observations, memory):
    state=json.load(open('research_state.json'))
    child=next(q for q in state['questions'] if q['id']=='paired')
    value=child['semantic_answer']['value']
    return {'actions':[{'kind':'trajectory','sequence':[{'u':value}]}],
        'revision':{'id':'parent_probe','affected_object':'current prediction',
                    'models':['incumbent.py']}}
'''}, lab)
    return parent


def test_finite_answer_keeps_scope_and_stopping_an_existing_parent_plan_is_a_change(tmp_path):
    agent, lab = prepared(tmp_path)
    parent = parent_plan(agent, lab)
    agent._save('shared/next.py', '''import json
def select_actions(spec, observations, memory):
    state=json.load(open('research_state.json'))
    answer=next(q for q in state['questions'] if q['id']=='paired')['semantic_answer']
    assert answer['scope']['kind']=='finite_cases'
    return {'stop': 'The declared finite question is answered.'}
''')
    question = paired_question(agent)
    question['answer_type'] = 'boolean_claim'
    question['query_semantics']['query_expression'] = {'op': 'AllOnFiniteSet', 'finite_cases': [
        {'op': 'RelationResidual', 'query': question['query_semantics']['query_expression'],
         'expected': 2., 'atol': 0.}]}
    original_cost = deepcopy(agent.costs)
    result = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert result['answer']['value'] is True and result['answer']['finite_status'] == 'observed_true'
    assert result['answer']['scope']['case_count'] == 1
    assert result['parent_revision_completed'] and result['uses'][0]['status'] == 'parent_plan_stopped'
    assert 'pending_query' not in parent and parent['pending_query_history'] == [{'actions': [action(99)]}]
    from hegel_machine.cognition.research_episodes import extract_episodes
    episode = next(item for item in extract_episodes(agent)['episodes']
                   if item['episode_id'] == 'paired@meaning1')
    assert not episode['adoption']['occurred']
    assert any(change['kind'] == 'parent_plan_stopped' for change in episode['behavior_changes'])
    after = deepcopy(agent.costs)
    assert step(agent, lab, 'paired')['status'] == 'reused_answer'
    assert agent.costs == after
    assert agent.costs['model_calls'] == 0
    assert agent.costs['observation_cost'] == original_cost['observation_cost']


def test_T13_child_changes_saved_parent_probe_then_normal_query_freezes_before_purchase(tmp_path):
    agent, lab = prepared(tmp_path)
    parent = parent_plan(agent, lab)
    before = lab.paid
    result = agent._execute('run', {'operator': 'research_step',
                                   'research_question': paired_question(agent)}, lab)
    assert result['answer']['value'] == 2.
    assert result['parent_revision_completed']
    assert parent['pending_query']['actions'] == [action(2.)]
    assert parent['pending_query']['question'] == {'id': 'parent'}
    assert parent['pending_query_history'] == [{'actions': [action(99)]}]
    assert lab.paid == before
    costs = deepcopy(agent.costs)
    assert step(agent, lab, 'paired')['status'] == 'reused_answer'
    assert agent.costs == costs
    purchased = agent._execute('query', {'pending_question': 'parent'}, lab)
    assert purchased['observations'][0]['action'] == action(2.)
    assert purchased['revision']['evidence_cutoff']['rows'] == before
    assert purchased['revision']['forecasts'][0]['predictions'] == [2.]
    assert purchased['question']['id'] == 'parent'
    assert lab.paid == before + 1 and 'pending_query' not in parent
    assert not agent.confirmations and agent.costs['model_calls'] == 0


def test_new_paid_evidence_rebuilds_parent_use_even_when_measured_answer_unchanged(tmp_path):
    agent, lab = prepared(tmp_path)
    parent = parent_plan(agent, lab)
    agent._execute('run', {'operator': 'research_step', 'research_question': paired_question(agent)}, lab)
    agent._execute('query', {'actions': [action(4.)]}, lab)
    # The old pending plan depends on the older paid cutoff.
    with pytest.raises(ValueError, match='needs rebuilding'):
        agent._execute('query', {'pending_question': 'parent'}, lab)
    assert step(agent, lab, 'paired')['status'] == 'answer_applied'
    assert parent['pending_query_source']['paid_records'] == len(agent.observations)


def test_answer_unknown_then_new_answer_never_reuses_old_answer_version(tmp_path):
    agent, lab = prepared(tmp_path)
    parent_plan(agent, lab)
    request = paired_question(agent)
    agent._execute('run', {'operator': 'research_step', 'research_question': request}, lab)
    question = next(q for q in agent.questions if q['id'] == 'paired')
    assert question['semantic_answer']['version'] == 1
    # A real compute-budget refusal is UNKNOWN for the same question. Removing
    # its old answer must not allow an old answer_version reference to revive.
    question.pop('query_execution')
    agent.costs['compute_cost'] = agent.config.compute_budget
    assert step(agent, lab, 'paired')['status'] == 'scoped_unknown'
    agent.costs['compute_cost'] -= 2
    returned = step(agent, lab, 'paired')
    assert returned['answer']['version'] == 2
    assert question['semantic_answer_history'][0]['version'] == 1


def test_T04_old_revision_reanalysis_does_not_answer_a_changed_question_meaning(tmp_path):
    agent, lab = prepared(tmp_path)
    returned = return_region(agent, lab, .9, 1.1)
    index = returned['revision']['index']
    agent._open_question({'id': 'slope_question', 'context_ref': {'output_unit': 'mV'}})
    owner = next(q for q in agent.questions if q['id'] == 'slope_question')
    assert owner['meaning_version'] == 2 and 'executable_answer' not in owner
    result = agent._execute('run', {'operator': 'analyze_revision', 'revision_index': index}, lab)
    assert result['revision']['answer_application']['status'] == 'historical_meaning_only'
    assert result['revision']['returned_answer']['parameter_region']['upper'] == 1.1
    assert 'executable_answer' not in owner and owner['status'] == 'open'


def test_T18_actual_collision_becomes_specific_author_work_without_supplying_a_repair(tmp_path):
    agent, lab = prepared(tmp_path)
    source = '''def represent(row): return [0.]
def fit_features(features, targets):
    global mean
    mean=sum(targets)/len(targets)
def predict_features(features): return mean
'''
    result = agent._execute('evaluate', {'model': 'collapsed.py', 'code': source}, lab)
    diagnostic = result['representation_diagnostic']
    assert diagnostic['empirical_mse_floor'] > 0 and diagnostic['witnesses']
    request = step(agent, lab, 'representation:collapsed.py')
    assert request['status'] == 'author_work_request' and request['object_type'] == 'program'
    question = next(q for q in agent.questions if q['id'] == 'representation:collapsed.py')
    assert question['use_sites'][0]['model'] == 'collapsed.py'
    assert question['representation_diagnostic'] == diagnostic
    assert (tmp_path / 'collapsed.py').read_text() == source
    assert not agent.confirmations and agent.costs['model_calls'] == 0


def test_T19_missing_executable_parent_use_returns_computed_answer_to_author(tmp_path):
    agent, lab = prepared(tmp_path)
    question = paired_question(agent)
    question['use_sites'] = [{'kind': 'supply_component', 'model': 'parent.py', 'slot': 'state'}]
    result = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert result['status'] == 'author_work_request'
    assert result['answer']['value'] == 2. and not result['parent_revision_completed']
    assert result['uses'][0]['object_type'] == 'supply_component'

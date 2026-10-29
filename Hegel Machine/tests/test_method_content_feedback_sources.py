"""Ordinary completed returns accumulate content evidence without new research."""

from copy import deepcopy
import json

from hegel_machine.cognition.method_feedback import (
    _read, finalize_python_method_feedback, record_python_method_feedback,
)
from test_method_use_conditions import profile
from test_theory_consolidation import agent_scene


def write_ref(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return {'file': str(path), 'json_path': []}


def scene(tmp_path):
    agent, _ = agent_scene(tmp_path)
    public = agent._open_question({'id': 'public', 'question': 'Research response', 'parent_id': 'task'})
    owner = agent._open_question({'id': 'method-topic', 'kind': 'method_use',
        'question': 'Which existing method should be revised?', 'parent_id': public['id']})
    topic = {'question_id': owner['id'], 'meaning_version': 1}
    candidate = {'id': 'p0', 'version': 1, 'kind': 'python_research_method',
        'method_question_ref': topic}
    agent.theory_consolidation.setdefault('candidates', {})['p0'] = candidate
    context = {'profile': profile(), 'question_ref': {'question_id': public['id'], 'meaning_version': 1}}
    binding = {'candidate_id': 'p0', 'candidate_version': 1,
        'origin_ref': {'file': 'invocation_a.json', 'json_path': []}}
    return agent, owner, candidate, binding, context


def outcome(tmp_path, *, complete=True, source=True):
    value = {'status': 'completed', 'execution_complete': complete,
        'comparison': {'models_requested': 8, 'models_completed': 8, 'truncated': False},
        'profile_ref': write_ref(tmp_path, 'before.json', profile()),
        'start_state_ref': {'file': 'original_large_state.json', 'json_path': []},
        'native_representation_returns': [{'native_result_ref': {'file': 'native_return.json', 'json_path': []}}],
        'method_artifact_ref': write_ref(tmp_path, 'p0.json', {'selection_plan': {'first_segment': {}}}),
        'cost_refs': [{'file': 'original_paid_ledger.json', 'json_path': []}]}
    if source:
        value['source_task_ref'] = {'file': 'source_request.json', 'json_path': ['source_id']}
    return write_ref(tmp_path, 'outcome.json', value), value


def test_whole_return_exposes_original_content_refs_without_archive_copy_or_gain(tmp_path):
    agent, owner, candidate, binding, context = scene(tmp_path)
    original = record_python_method_feedback(agent, candidate, binding,
        {'status': 'unresolved', 'execution_complete': False}, context)
    reference, whole = outcome(tmp_path)
    completed = finalize_python_method_feedback(agent, candidate, binding, context, result_ref=reference)
    assert completed['feedback_ref'] == original['feedback_ref']
    saved = _read(agent, completed['feedback_ref'])
    assert saved['method_content_source'] == {
        'source_task_ref': whole['source_task_ref'], 'outcome_ref': reference}
    assert saved['prior_selection_plan_ref'] == {
        **whole['method_artifact_ref'], 'json_path': ['selection_plan']}
    assert saved['cost_refs'] == whole['cost_refs']
    assert saved['method_ref'] == {'id': 'p0', 'version': 1}
    assert not any(key in saved for key in ('native_representation_returns', 'start_state',
        'working_comparison', 'paired_future', 'gain', 'preference'))
    assert _read(agent, saved['profile_ref']) == context['profile']
    assert len(owner['method_action_returns']) == 1
    assert owner['method_use_plan']['ordinary_return_basis'][0]['method_content_source'] == saved['method_content_source']
    assert agent.costs['compute_cost'] == 0 and not agent.observations
    again = finalize_python_method_feedback(agent, candidate, binding, context, result_ref=reference)
    assert again['feedback_ref'] == completed['feedback_ref']
    assert len(owner['method_action_returns']) == 1
    assert agent.theory_consolidation['method_feedback']['p0']['returns']['delivery_reliability']['total_invocations'] == 1


def test_same_topic_accumulates_different_real_methods_without_relabeling(tmp_path):
    agent, owner, candidate, binding, context = scene(tmp_path)
    reference, _ = outcome(tmp_path)
    first = finalize_python_method_feedback(agent, candidate, binding, context, result_ref=reference)
    second_method = {**deepcopy(candidate), 'id': 'conditional-p1'}
    agent.theory_consolidation['candidates'][second_method['id']] = second_method
    second_binding = {**deepcopy(binding), 'candidate_id': second_method['id'],
        'origin_ref': {'file': 'invocation_b.json', 'json_path': []}}
    second = record_python_method_feedback(agent, second_method, second_binding,
        {'status': 'unresolved', 'execution_complete': False}, context)
    expected = [first['feedback_ref'], second['feedback_ref']]
    assert owner['method_action_returns'] == expected
    assert owner['candidate_method_refs'] == [{'id': 'p0', 'version': 1},
        {'id': 'conditional-p1', 'version': 1}]
    for question_ref in second['method_question_refs']:
        question = next(q for q in agent.questions if q['id'] == question_ref['question_id'])
        assert question['method_use_plan']['ordinary_return_refs'] == expected
        assert [row['method_ref']['id'] for row in question['method_use_plan']['ordinary_return_basis']] == ['p0', 'conditional-p1']
    assert len(agent.theory_consolidation['method_feedback']['conditional-p1']['outcome_refs']) == 1
    assert _read(agent, first['feedback_ref'])['method_ref']['id'] == 'p0'


def test_incomplete_or_unidentified_return_is_not_a_content_formation_source(tmp_path):
    agent, _, candidate, binding, context = scene(tmp_path)
    reference, whole = outcome(tmp_path, complete=False)
    failed = finalize_python_method_feedback(agent, candidate, binding, context, result_ref=reference)
    assert 'method_content_source' not in _read(agent, failed['feedback_ref'])
    whole['execution_complete'] = True
    del whole['source_task_ref']
    reference = write_ref(tmp_path, 'outcome.json', whole)
    unidentified = finalize_python_method_feedback(agent, candidate, binding, context, result_ref=reference)
    assert 'method_content_source' not in _read(agent, unidentified['feedback_ref'])
    context['source_task_ref'] = {'file': 'same_source_request.json', 'json_path': ['source_id']}
    identified = finalize_python_method_feedback(agent, candidate, binding, context, result_ref=reference)
    assert _read(agent, identified['feedback_ref'])['method_content_source']['source_task_ref'] == context['source_task_ref']


def test_public_source_identity_is_stable_across_returns_and_state_snapshots(tmp_path):
    agent, owner, candidate, binding, context = scene(tmp_path)
    context['theory_target'] = {'context_ref': {'source_id': 'independent/development/source_0'}}
    reference, whole = outcome(tmp_path, source=False)
    first = finalize_python_method_feedback(agent, candidate, binding, context, result_ref=reference)
    whole['start_state_ref'] = {'file': 'later_state_of_same_task.json', 'json_path': []}
    later_ref = write_ref(tmp_path, 'later_outcome.json', whole)
    later_binding = {**deepcopy(binding), 'origin_ref': {'file': 'later_invocation.json', 'json_path': []}}
    second = finalize_python_method_feedback(agent, candidate, later_binding, context, result_ref=later_ref)
    first_source = _read(agent, first['feedback_ref'])['method_content_source']
    later_source = _read(agent, second['feedback_ref'])['method_content_source']
    assert first_source['source_task_ref'] == later_source['source_task_ref'] == {
        'source_id': 'independent/development/source_0'}
    assert first_source['outcome_ref'] != later_source['outcome_ref']
    assert owner['method_action_returns'] == [first['feedback_ref'], second['feedback_ref']]


def test_production_ordinary_delivery_reaches_feedback_and_native_source_reader(tmp_path):
    from scripts.prepare_pr3_source_successor import ordinary_method_delivery
    from hegel_machine.cognition.method_action_research import _paid_action_experiences

    agent, _, candidate, binding, context = scene(tmp_path)
    candidate['selection_plan'] = {'first_segment': {}}
    public = next(question for question in agent.questions if question['id'] == 'public')
    public['theory_target'] = {'context_ref': {'source_id': 'live/source/ordinary'}}
    original = record_python_method_feedback(agent, candidate, binding,
        {'status': 'unresolved', 'execution_complete': False}, context)
    before_profile_ref = _read(agent, original['feedback_ref'])['profile_ref']
    source = {'question_ref': {'id': 'public'},
        'profile_ref': {'file': 'post_use_profile_must_not_replace_original.json'},
        'cost_refs': [{'file': 'paid_prior_ledger.json', 'json_path': []}]}
    for field in ('start_state', 'data', 'split', 'quality_contract', 'resource_contract', 'evaluation'):
        source[field + '_ref'] = {'file': field + '.json', 'json_path': []}
    native_ref = write_ref(tmp_path, 'native_paid_return.json', {
        'status': 'native_representation_parent_retained', 'parent_theory_ids': [],
        'candidates': [], 'selected': None})
    invoked = {'status': 'registered_native_attempt_returned', 'method_invocation_binding': binding,
        'native_representation_returns': [{'native_result_ref': native_ref, 'complete': True}],
        'native_research_complete': True, 'actual_action_content': [{'operator': 'native_condition_research'}]}
    comparison = {'models_requested': 8, 'models_completed': 8, 'truncated': False}
    comparison_ref = write_ref(tmp_path, 'complete_comparison.json', comparison)
    before = dict(agent.costs)
    delivery = ordinary_method_delivery(agent, source, invoked, comparison,
        comparison_ref=comparison_ref, model='retained.py',
        frozen_model={'model': 'retained.py'}, costs_before=before)
    assert delivery['profile_ref'] == before_profile_ref
    assert delivery['start_state_ref'] == source['start_state_ref']
    assert delivery['native_representation_returns'] == invoked['native_representation_returns']
    assert delivery['source_id'] == 'live/source/ordinary'
    delivery_ref = write_ref(tmp_path, 'production_delivery.json', delivery)
    final = agent._execute('run', {'operator': 'method_action_feedback',
        'candidate_id': candidate['id'], 'binding': binding, 'result_ref': delivery_ref,
        'comparison_ref': comparison_ref}, None)
    assert final['feedback_ref'] == original['feedback_ref']
    recorded = _read(agent, final['feedback_ref'])
    assert recorded['method_content_source'] == {
        'source_task_ref': {'source_id': 'live/source/ordinary'}, 'outcome_ref': delivery_ref}
    projected, excluded = _paid_action_experiences(agent, [recorded['method_content_source']])
    assert not excluded and len(projected) == 1
    assert projected[0]['profile'] == context['profile']
    assert projected[0]['native_result']['status'] == 'native_representation_parent_retained'
    assert projected[0]['source_task_ref'] == {'source_id': 'live/source/ordinary'}
    assert agent.costs == before and not agent.observations

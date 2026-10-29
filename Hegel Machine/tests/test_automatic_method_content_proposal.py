"""Ordinary source returns drive the real Common method-content route."""

from copy import deepcopy
import json

from hegel_machine.cognition import finite_native_author
from hegel_machine.cognition.method_feedback import (
    _read, finalize_python_method_feedback, record_python_method_feedback,
)
from hegel_machine.cognition.research_programs import export_research_selection_program
from hegel_machine.cognition.theory_consolidation import install_python_research_method
from test_executed_evidence_notebook import AccountedPublicLab
from test_method_witness_induction import experience, old_plan
from test_theory_consolidation import agent_scene


def save(directory, name, value):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(value))
    return {'file': str(path), 'json_path': []}


def setup_scene(path):
    agent, lab = agent_scene(path)
    mission = {'purpose': 'method_learning', 'allocation': {'compute': 10000}}
    parent = agent._open_question({'id': 'public', 'question': 'Continue research.',
        'parent_id': 'task', 'mission': mission})
    owner = agent._open_question({'id': 'method-root', 'kind': 'method_use',
        'question': 'Revise the existing conditional method from its source returns.',
        'parent_id': parent['id'], 'mission': mission,
        'method_use_plan': {'candidate_id': 'conditional', 'binding': {}}})
    topic = {'question_id': owner['id'], 'meaning_version': 1}
    agent._save('ordinary.py', finite_native_author.export_finite_native_author())
    p0_plan = {'ordinary_method_path': 'ordinary.py'}
    p0 = install_python_research_method(agent, candidate_id='P0',
        source=export_research_selection_program(p0_plan), selection_plan=p0_plan,
        source_refs=[], method_question_ref=topic)
    prior = old_plan()
    selection = save(path, 'old_guard_selection.json', {'general_guard_supported': True})
    conditional = install_python_research_method(agent, candidate_id='conditional',
        source=export_research_selection_program(prior), selection_plan=prior,
        source_refs=[], selection_ref=selection, method_question_ref=topic)
    owner['candidate_method_refs'] = [{'id': 'conditional', 'version': 1}]
    agent._active_question_id = parent['id']
    return agent, lab, owner, p0, conditional


def append_source(agent, p0, name, *, missed=False):
    measured = experience(name, missed)
    directory = agent.workspace / 'original_sources' / name
    public_profile = deepcopy(measured['profile'])
    public_profile['research']['candidate_pool']['witnesses'][0]['execution_ref'] = save(
        directory, 'paid_execution.json', {'source_id': name, 'pre_use_witness': 'old.py'})
    profile_ref = save(directory, 'preprofile.json', public_profile)
    native_ref = save(directory, 'native.json', measured['native_result'])
    state_ref = save(directory, 'start.json', {'theory_index': {
        item['id']: {'program': item['program']} for item in measured['parent_programs']}})
    context = {'profile': public_profile,
        'question_ref': {'question_id': 'public', 'meaning_version': 1},
        'theory_target': {'context_ref': {'source_id': name}}}
    origin_ref = save(directory, 'invocation.json', {'method': 'P0'})
    binding = {'candidate_id': p0['id'], 'candidate_version': p0['version'], 'origin_ref': origin_ref}
    first = record_python_method_feedback(agent, p0, binding,
        {'status': 'registered_native_attempt_returned', 'execution_complete': False}, context)
    outcome_ref = save(directory, 'outcome.json', {
        'status': 'completed', 'execution_complete': True, 'source_id': name,
        'profile_ref': profile_ref, 'start_state_ref': state_ref,
        'comparison': {'models_requested': 48, 'models_completed': 48, 'truncated': False},
        'native_representation_returns': [{'native_result_ref': native_ref, 'complete': True}],
        'cost_refs': [save(directory, 'bill.json', {'compute_cost': 120})]})
    final = finalize_python_method_feedback(agent, p0, binding, context, result_ref=outcome_ref)
    assert final['feedback_ref'] == first['feedback_ref']
    return final


def review(agent, lab, question_id):
    return agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question_id}, 'allocation': {'compute': 1000}}, lab)


def test_second_ordinary_source_automatically_proposes_the_existing_methods_local_revision(tmp_path):
    agent, lab, owner, p0, conditional = setup_scene(tmp_path)
    first = append_source(agent, p0, 'success')
    followup = first['method_question_refs'][0]['question_id']
    before = review(agent, lab, followup)
    assert before['method_action_research']['reason'] == 'distinct_completed_sources_required'
    assert before['method_action_research']['relative_labels_created'] == 0
    second = append_source(agent, p0, 'missed', missed=True)
    followup = second['method_question_refs'][0]['question_id']
    prior = deepcopy(conditional['selection_plan'])
    result = review(agent, lab, followup)
    assert result['cache_hit'] is False
    action = result['method_action_research']
    assert result['status'] == 'method_content_proposed_pending_validation'
    assert action['revisable_method_refs'] == [{'id': 'conditional', 'version': 1}]
    candidate, = action['proposals']['candidates']
    proposal = candidate['research_selection_plan']
    assert candidate['previous_method_ref'] == {'id': 'conditional', 'version': 1}
    assert proposal['when_false']['first_segment']['parent_binding'] == 'paid_startup_witnesses'
    for field in ('condition', 'condition_feature_paths', 'when_true'):
        assert proposal[field] == prior[field]
    assert conditional['selection_plan'] == prior
    assert candidate['status'] == 'unmeasured_local_research_intervention'
    assert action['installed_candidate'] is None and result['next_action'] is None
    assert action['relative_labels_created'] == 0 and action['support_updated'] is False
    assert all(row['method_ref'] == {'id': 'P0', 'version': 1}
               for row in action['ordinary_returns'])
    assert _read(agent, first['feedback_ref'])['method_ref']['id'] == 'P0'
    assert len(owner['method_action_returns']) == 2
    cost = agent.costs['compute_cost']
    cached = review(agent, lab, followup)
    assert cached['cache_hit'] is True and cached['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == cost and lab.paid == 0


def test_cached_content_proposal_survives_actual_practice_restore(tmp_path):
    agent, lab, _, p0, _ = setup_scene(tmp_path / 'original')
    append_source(agent, p0, 'success')
    returned = append_source(agent, p0, 'missed', missed=True)
    followup = returned['method_question_refs'][0]['question_id']
    result = review(agent, lab, followup)
    assert result['status'] == 'method_content_proposed_pending_validation'
    agent._prepare_program_state()
    previous = agent.checkpoint_result('researching')
    restored, _ = agent_scene(tmp_path / 'restored')
    fresh = AccountedPublicLab()
    receipt = restored.restore_practice_research(previous, agent.workspace, fresh, confirmation_seed=17)
    before = dict(restored.costs)
    continued = review(restored, fresh, followup)
    assert receipt['new_observations'] == 0
    assert continued['cache_hit'] is True and continued['new_compute_cost'] == 0
    assert continued['method_action_research']['proposals'] == result['method_action_research']['proposals']
    assert restored.costs == before and fresh.actions == []


def test_unrelated_installed_guard_is_not_a_target_without_method_topic_membership(tmp_path):
    agent, lab, owner, p0, _ = setup_scene(tmp_path)
    append_source(agent, p0, 'success')
    returned = append_source(agent, p0, 'missed', missed=True)
    owner['candidate_method_refs'] = [{'id': 'P0', 'version': 1}]
    owner['method_use_plan']['candidate_id'] = 'P0'
    result = review(agent, lab, returned['method_question_refs'][0]['question_id'])
    action = result['method_action_research']
    assert action['reason'] == 'existing_method_topic_body_required'
    assert action['revisable_method_refs'] == []
    assert not action.get('proposals')
    assert action['installed_candidate'] is None and lab.paid == 0

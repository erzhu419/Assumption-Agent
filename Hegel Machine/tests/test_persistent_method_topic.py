"""A real ordinary return continues one funded method topic across episodes."""

from copy import deepcopy
from dataclasses import replace

from hegel_machine.cognition.method_feedback import (
    _read, ensure_python_method_topic, export_python_method_topic)
from hegel_machine.cognition.question_agenda import mission_compute_remaining
from test_automatic_method_content_proposal import setup_scene, append_source
from test_ordinary_method_return_agenda import deliver
from test_theory_consolidation import agent_scene


def test_two_episode_returns_keep_topic_scope_and_debit_without_relabeling(tmp_path):
    first, lab, owner, p0, _ = setup_scene(tmp_path / 'first')
    owner['mission'].update(mission_id='shared-original-budget', resource_usage={'compute': 146})
    first_return = append_source(first, p0, 'success')
    delivered = deliver(first, lab, first_return)
    assert delivered['method_research']['research_result']['method_action_research'][
        'reason'] == 'distinct_completed_sources_required'
    topic = export_python_method_topic(first, owner)
    spent = owner['mission']['resource_usage']['compute']
    assert spent > 146

    second, next_lab = agent_scene(tmp_path / 'second')
    second.config = replace(second.config, compute_budget=10000)
    second._open_question({'id': 'public', 'question': 'Next independent source.', 'parent_id': 'task'})
    resumed = ensure_python_method_topic(second,
        parent_question_ref={'question_id': 'public', 'meaning_version': 1}, topic_ref=topic)
    assert resumed['id'] == owner['id']
    assert resumed['mission']['resource_usage']['compute'] == spent
    for identifier in ('P0', 'conditional'):
        candidate = second.theory_consolidation['candidates'][identifier]
        assert (second.workspace / candidate['execution']['path']).is_file()
        assert (second.workspace / candidate['selection_plan']['ordinary_method_path']).is_file()
    p0 = second.theory_consolidation['candidates']['P0']
    final = append_source(second, p0, 'missed', missed=True)
    result = deliver(second, next_lab, final)
    action = result['method_research']['research_result']['method_action_research']
    assert action['proposals']['candidates']
    records = [_read(second, ref) for ref in resumed['method_action_returns']]
    assert len(records) == 2 and all(record['invoked_method_ref']['id'] == 'P0' for record in records)
    assert resumed['method_use_plan']['candidate_id'] == 'conditional'
    assert second.theory_consolidation['method_feedback']['P0']['returns'][
        'delivery_reliability']['total_invocations'] == 2
    before = deepcopy(second.costs)
    repeated = deliver(second, next_lab, final)
    assert repeated['method_research']['new_compute_cost'] == 0 and second.costs == before
    assert next_lab.paid == 0
    exported = export_python_method_topic(second, resumed)
    assert export_python_method_topic(second, resumed) == exported
    restored = _read(second, exported)
    assert restored['question']['mission']['resource_usage']['compute'] >= spent
    assert 'observations' not in restored and 'theory_consolidation' not in restored


def test_restored_topic_keeps_same_ordinary_completion_and_used_allowance(tmp_path):
    agent, lab, owner, p0, _ = setup_scene(tmp_path / 'original')
    owner['mission'].update(mission_id='fixed', allocation={'compute': 160},
        resource_usage={'compute': 146})
    returned = append_source(agent, p0, 'success')
    deliver(agent, lab, returned)
    topic = export_python_method_topic(agent, owner)
    current, current_lab = agent_scene(tmp_path / 'resumed')
    current.config = replace(current.config, compute_budget=160)
    current._open_question({'id': 'public', 'question': 'Resume public source.', 'parent_id': 'task'})
    restored = ensure_python_method_topic(current,
        parent_question_ref={'question_id': 'public'}, topic_ref=topic)
    before = deepcopy(current.costs)
    result = deliver(current, current_lab, returned)
    assert result['method_research']['new_compute_cost'] == 0 and current.costs == before
    assert len(restored['method_action_returns']) == 1
    assert mission_compute_remaining(current, restored['mission']) < 14
    assert current_lab.paid == 0


def test_ordinary_session_entry_consumes_exported_topic_without_host_research_route(tmp_path, monkeypatch):
    from scripts import run_method_condition_cohort as cohort
    from scripts import run_state_method_pairing as pairing

    original, _, owner, p0, _ = setup_scene(tmp_path / 'original')
    owner['candidate_method_refs'].append({'id': p0['id'], 'version': p0['version']})
    owner['mission'].update(mission_id='paid-remainder', allocation={'compute': 400},
        resource_usage={'compute': 146})
    topic = export_python_method_topic(original, owner)
    artifact = cohort.save(original.workspace / 'method_artifact.json', p0)
    sources = []
    for name, missed in [('success', False), ('missed', True)]:
        record = _read(original, append_source(original, p0, name, missed=missed)['feedback_ref'])
        actual = _read(original, record['actual_result_ref'])
        actual.update(method_ref={'id': p0['id'], 'version': p0['version']},
            method_artifact_ref=artifact, method_invocation_binding=record['invocation_binding'])
        sources.append({'source_task_ref': {'source_id': name},
            'outcome_ref': cohort.save(original.workspace / name / 'delivery.json', actual)})
    seen = []

    def restore(registration, workspace):
        agent, lab = agent_scene(workspace)
        agent._open_question(deepcopy(registration['question']))
        execute = agent._execute
        depth = 0

        def ordinary(action, arguments, laboratory):
            nonlocal depth
            if depth == 0:
                seen.append(arguments['operator'])
                assert arguments['operator'] == 'method_action_feedback'
            depth += 1
            try:
                return execute(action, arguments, laboratory)
            finally:
                depth -= 1

        monkeypatch.setattr(agent, '_execute', ordinary)
        return agent, lab, []

    monkeypatch.setattr(pairing, '_scene', restore)
    registration = cohort.save(tmp_path / 'registration.json', {
        'question': {'id': 'public', 'question': 'Continue same scientific problem.', 'parent_id': 'task'}})
    result = cohort.form_method_from_ordinary_session({'registration_ref': registration,
        'method_topic_ref': topic, 'ordinary_return_sources': sources,
        'analysis_compute_cap': 254}, tmp_path / 'acceptance')
    assert seen == ['method_action_feedback'] * 3
    assert result['duplicate_return_reused']
    assert result['returns'][0]['ordinary_returns'] == 1
    assert result['returns'][1]['ordinary_returns'] == 2
    assert result['latest_method_action']['proposals']['candidates']
    assert result['remaining_analysis_compute'] == 254 - result['new_compute_cost']
    assert result['new_observations'] == result['new_predictions'] == result['new_fits'] == 0
    assert not result['source_costs_recharged']


def test_imported_prior_method_support_does_not_become_a_new_formation_source(tmp_path):
    from hegel_machine.cognition.method_feedback import record_python_method_feedback

    original, _, owner, _, conditional = setup_scene(tmp_path / 'original')
    prior_body = {key: deepcopy(value) for key, value in conditional.items()
                  if key != 'method_question_ref'}
    supported = record_python_method_feedback(original, prior_body,
        {'candidate_id': prior_body['id'], 'candidate_version': prior_body['version'],
         'origin_ref': {'file': 'prior_completed_invocation.json', 'json_path': []}},
        {'status': 'completed', 'execution_complete': True},
        {'profile': {'observation_count': 7},
         'question_ref': {'question_id': 'public', 'meaning_version': 1}})
    assert not owner.get('method_action_returns')
    reference = export_python_method_topic(original, owner)
    restored, _ = agent_scene(tmp_path / 'restored')
    restored._open_question({'id': 'public', 'question': 'Next public task.', 'parent_id': 'task'})
    topic = ensure_python_method_topic(restored,
        parent_question_ref={'question_id': 'public'}, topic_ref=reference)
    assert not topic.get('method_action_returns')
    assert restored.theory_consolidation['method_feedback'][conditional['id']]['outcome_refs'] == [
        supported['feedback_ref']]
    assert restored.theory_consolidation['method_feedback'][conditional['id']]['returns'][
        'delivery_reliability']['delivered'] == 1

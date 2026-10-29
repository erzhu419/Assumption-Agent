"""Native return contracts reach the existing learner through real Common routing.

The small declared outcomes below are control fixtures, not research results.
"""

from copy import deepcopy
import json
import pytest

from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, record_native_parent_research_return,
)
from hegel_machine.cognition.models import SearchBudget


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'file': str(path)}


def common(tmp_path):
    return CommonObjectDependencyMethods(tmp_path / 'common', [],
        budget=SearchBudget(100000), actions=[]).common


def deliver(agent, path, name, *, version=1, stage='unresolved', cheaper=True,
            measured_operation='structural_research', declared_operation=None,
            mismatch_data=False, baseline_operation='program_replay',
            decision_index=None, retained_pair_ref=None, knowledge_ref=None):
    root = path / name
    method = {'id': 'native:structural_research', 'version': version}
    body = {**method, 'kind': 'native_research_segment', 'execution': {
        'method': 'execute_research_decision', 'research_action': measured_operation},
        'source_ref': {'file': 'frozen_native_source.py'}}
    baseline_method = {'id': 'native:' + baseline_operation, 'version': 1}
    if knowledge_ref is not None:
        baseline_method['version'] = version
    baseline_body = {**baseline_method, 'kind': 'native_research_segment', 'execution': {
        'method': 'execute_research_decision', 'research_action': baseline_operation},
        'source_ref': {'file': 'frozen_native_source.py'}}
    decision_ref = {'file': 'source_report.json', 'json_path': ['actions', decision_index]}
    if decision_index is not None:
        from scripts.run_native_decision_pair import bind_native_method_artifacts

        root.mkdir(parents=True, exist_ok=True)
        bindings = bind_native_method_artifacts(root, {'operations': {
            'baseline': baseline_operation, 'challenger': measured_operation},
            'retained_method_pair_ref': retained_pair_ref,
            'method_version': version, 'knowledge_contract_ref': knowledge_ref}, decision_ref)
        artifact = bindings['challenger']['method_artifact_ref']
        body = json.loads(open(artifact['file']).read())
        baseline_artifact = bindings['baseline']['method_artifact_ref']
        baseline_body = json.loads(open(baseline_artifact['file']).read())
    else:
        artifact = save(root / 'method.json', body)
        baseline_artifact = save(root / 'baseline_method.json', baseline_body)
    declared = deepcopy(body)
    if declared_operation is not None:
        declared['execution']['research_action'] = declared_operation
    declared_ref = save(root / 'declared_method.json', declared)
    profile = save(root / 'profile.json', {
        'objective': {'kind': 'squared'}, 'research': {'stage': stage},
        'observation_count': 24})
    contracts = {key + '_ref': save(root / f'{key}.json', {'version': name})
        for key in ('start_state', 'data', 'split', 'evaluation')}
    contracts['quality_contract_ref'] = save(root / 'quality.json', {'threshold': 1e-7})
    contracts['resource_contract_ref'] = save(root / 'resource.json', {'compute_limit': 1000})
    baseline = {**contracts, 'profile_ref': profile, 'status': 'completed',
        'execution_complete': True, 'completed': True, 'coverage': 24, 'loss': 1e-20,
        'method_ref': baseline_method, 'method_artifact_ref': baseline_artifact,
        'costs': {'compute_cost': 100, 'observation_cost': 0, 'measurement_cost': 0}}
    challenger = deepcopy(baseline)
    challenger.update(method_ref=method, method_artifact_ref=artifact)
    challenger['costs']['compute_cost'] = 50 if cheaper else 150
    if mismatch_data:
        challenger['data_ref'] = save(root / 'other_data.json', {'different': True})
    source = save(root / 'source.json', {'source': name})
    pair_ref = save(root / 'pair.json', {
        'baseline_ref': save(root / 'baseline.json', baseline),
        'challenger_ref': save(root / 'challenger.json', challenger),
        'physical_source_ref': source, 'origin_ref': {'file': str(root / 'registration.json')},
        'decision_ref': decision_ref,
        'data_role': 'development', 'identification_target': 'fixed_data_method_effect',
        'baseline_action_content': baseline_body['execution'],
        'challenger_action_content': body['execution']})
    returned = save(root / 'receipt.json', {'status': 'paired_completed',
        'method_pair_ref': pair_ref, 'method_ref': method, 'method_artifact_ref': declared_ref})
    feedback = record_native_parent_research_return(agent, source_task_ref=source,
        result_ref=returned,
        mission={'purpose': 'method_learning', 'allocation': {'compute': 100000}})
    return feedback, pair_ref


def route(agent, allowance=100000):
    return agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': 'task:method_choice'},
        'allocation': {'compute': allowance}}, None)


def test_native_pair_reaches_existing_learner_without_publishing_a_general_condition(tmp_path):
    agent = common(tmp_path)
    feedback, pair_ref = deliver(agent, tmp_path, 'one')
    recorded = json.loads(open(feedback['feedback_ref']['file']).read())
    assert recorded['method_pair_ref'] == pair_ref
    reviewed = route(agent)['method_action_research']
    learned = reviewed['paired_condition_research'][0]
    assert learned['pair_refs'] == [pair_ref]
    assert learned['preferences'][0]['preference'] == 'challenger'
    assert learned['identified_effects'] == ['fixed_data_method_effect']
    assert learned['local_program_supported'] is True
    assert learned['general_guard_supported'] is False
    assert learned['condition'] is None and learned['status'] == 'insufficient_paired_preferences'
    assert reviewed['relative_labels_created'] == 1
    assert reviewed['installed_candidate'] is None
    assert not agent.theory_consolidation.get('candidates')
    paid = agent.costs['compute_cost']
    repeated = route(agent)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == paid


def test_measured_body_mismatch_cannot_lend_its_gain_to_another_body(tmp_path):
    agent = common(tmp_path)
    deliver(agent, tmp_path, 'mismatch', declared_operation='response_basis_revision')
    reviewed = route(agent)['method_action_research']
    assert reviewed['paired_condition_research'] == []
    assert reviewed['paired_method_exclusions'][0]['reason'] == 'paired_native_body_content_mismatch'
    assert reviewed['relative_labels_created'] == 0


def test_ordinary_parent_delivery_keeps_the_pair_contract(tmp_path):
    first = common(tmp_path / 'first')
    feedback, pair_ref = deliver(first, tmp_path, 'ordinary')
    saved_return = json.loads(open(feedback['feedback_ref']['file']).read())
    receipt = json.loads(open(saved_return['actual_result_ref']['file']).read())
    consumer = CommonObjectDependencyMethods(tmp_path / 'ordinary_common', [],
        budget=SearchBudget(100), actions=[], source_task_ref=saved_return['source_task_ref'])
    delivered = consumer.record_parent_response(receipt)
    recorded = json.loads(open(delivered['method_feedback']['feedback_ref']['file']).read())
    assert recorded['method_pair_ref'] == pair_ref
    owner = next(q for q in consumer.common.questions if q['id'] == 'task:method_choice')
    owner['mission']['allocation']['compute'] = 10
    reviewed = route(consumer.common, allowance=10)['method_action_research']
    assert reviewed['paired_condition_research'][0]['local_program_supported']


def test_versions_are_grouped_separately_and_same_version_cannot_change_body(tmp_path):
    agent = common(tmp_path)
    _, first = deliver(agent, tmp_path, 'v1')
    _, second = deliver(agent, tmp_path, 'v2', version=2)
    deliver(agent, tmp_path, 'bad_v1', measured_operation='response_basis_revision')
    reviewed = route(agent)['method_action_research']
    assert [x['pair_refs'] for x in reviewed['paired_condition_research']] == [[first], [second]]
    assert reviewed['paired_method_exclusions'][0]['reason'] == 'native_method_version_changed_body'
    assert all(x['condition'] is None for x in reviewed['paired_condition_research'])


def test_same_body_different_data_preserves_unknown_pair(tmp_path):
    agent = common(tmp_path)
    deliver(agent, tmp_path, 'mismatch_data', mismatch_data=True)
    learned = route(agent)['method_action_research']['paired_condition_research'][0]
    assert learned['preferences'][0]['preference'] == 'UNKNOWN'
    assert learned['preferences'][0]['reason'] == 'start_data_or_evaluation_contract_differs'
    assert not learned['local_program_supported']


def test_condition_search_budget_shortfall_remains_resumable(tmp_path):
    agent = common(tmp_path)
    deliver(agent, tmp_path, 'positive_1')
    deliver(agent, tmp_path, 'positive_2')
    deliver(agent, tmp_path, 'negative', stage='already_explained', cheaper=False)
    limited = route(agent, allowance=4)
    assert limited['status'] == 'scoped_unknown'
    assert limited['method_action_research']['paired_condition_research'][0]['budget_incomplete']
    owner = next(q for q in agent.questions if q['id'] == 'task:method_choice')
    assert 'completed_agenda_basis' not in owner
    resumed = route(agent)
    learned = resumed['method_action_research']['paired_condition_research'][0]
    assert not learned.get('budget_incomplete')
    assert learned['status'] == 'development_condition_without_group_support'
    assert not learned['general_guard_supported']


def test_paid_working_comparison_does_not_claim_a_post_freeze_readout():
    from hegel_machine.cognition.method_conditions import (
        paired_method_preference, research_paired_method_conditions,
    )
    from test_v8_paired_method_conditions import pair

    actual = pair(0)
    old = paired_method_preference(actual)
    assert old['evaluation_use'] == 'exposed_development_label_after_outputs_freeze'
    actual['evaluation_role'] = 'paid_working_comparison'
    working = paired_method_preference(actual)
    assert working['evaluation_use'] == 'exposed_paid_working_comparison'
    assert working['preference'] == old['preference'] == 'challenger'
    learned = research_paired_method_conditions({'id': 'native', 'version': 1},
        [actual], SearchBudget(0))
    assert learned['preferences'][0]['evaluation_use'] == 'exposed_paid_working_comparison'
    assert learned['local_program_supported']
    assert not learned['general_guard_supported'] and not learned['independent_confirmation']


def test_native_decision_provenance_reuses_body_without_changing_original_artifacts(tmp_path):
    agent = common(tmp_path)
    _, first = deliver(agent, tmp_path, 'action58', decision_index=58)
    original = (tmp_path / 'action58/challenger_method.json').read_bytes()
    _, second = deliver(agent, tmp_path, 'action85', decision_index=85, retained_pair_ref=first)
    next_bindings = json.loads((tmp_path / 'action85/method_bindings.json').read_text())
    assert next_bindings['decision_ref']['json_path'] == ['actions', 85]
    assert next_bindings['methods']['challenger']['method_artifact_ref'] == {
        'file': str(tmp_path / 'action58/challenger_method.json')}
    assert (tmp_path / 'action58/challenger_method.json').read_bytes() == original
    assert not (tmp_path / 'action85/challenger_method.json').exists()
    assert json.loads(original)['source_ref']['json_path'] == ['actions', 58]
    reviewed = route(agent)['method_action_research']
    assert not reviewed['paired_method_exclusions']
    assert [item['pair_refs'] for item in reviewed['paired_condition_research']] == [[first, second]]
    assert reviewed['relative_labels_created'] == 2
    assert not reviewed['paired_condition_research'][0]['general_guard_supported']


def test_native_conditions_keep_different_baseline_bodies_in_separate_groups(tmp_path):
    agent = common(tmp_path)
    _, first = deliver(agent, tmp_path, 'positive_1')
    _, second = deliver(agent, tmp_path, 'positive_2')
    _, different = deliver(agent, tmp_path, 'other_comparator', cheaper=False,
        stage='changed', baseline_operation='mechanism_revision')
    reviewed = route(agent)['method_action_research']
    groups = reviewed['paired_condition_research']
    assert [item['pair_refs'] for item in groups] == [[first, second], [different]]
    assert [item['baseline_method_ref']['id'] for item in groups] == [
        'native:program_replay', 'native:mechanism_revision']
    assert all(item['condition'] is None and not item['general_guard_supported'] for item in groups)


def test_native_artifact_reuse_rejects_a_different_executable_body(tmp_path):
    from scripts.run_native_decision_pair import bind_native_method_artifacts

    agent = common(tmp_path)
    _, pair_ref = deliver(agent, tmp_path, 'action58', decision_index=58)
    with pytest.raises(ValueError, match='body differs'):
        bind_native_method_artifacts(tmp_path / 'wrong', {'operations': {
            'baseline': 'program_replay', 'challenger': 'response_basis_revision'},
            'retained_method_pair_ref': pair_ref},
            {'file': 'source_report.json', 'json_path': ['actions', 85]})


def test_frozen_knowledge_changes_native_content_and_condition_group(tmp_path):
    agent = common(tmp_path)
    _, old = deliver(agent, tmp_path, 'old_k0', decision_index=58)
    knowledge_ref = {'file': 'confirmed_source/knowledge_delta.json'}
    _, learned = deliver(agent, tmp_path, 'new_k1', decision_index=0, version=2,
                         knowledge_ref=knowledge_ref)
    artifact = json.loads((tmp_path / 'new_k1/baseline_method.json').read_text())
    assert artifact['execution']['knowledge_ref'] == knowledge_ref
    assert artifact['version'] == 2
    reviewed = route(agent)['method_action_research']
    assert not reviewed['paired_method_exclusions']
    assert [group['pair_refs'] for group in reviewed['paired_condition_research']] == [[old], [learned]]
    assert all(group['condition'] is None for group in reviewed['paired_condition_research'])


def test_old_native_body_cannot_be_reused_with_new_frozen_knowledge(tmp_path):
    from scripts.run_native_decision_pair import bind_native_method_artifacts

    agent = common(tmp_path)
    _, old = deliver(agent, tmp_path, 'old_k0', decision_index=58)
    with pytest.raises(ValueError, match='body differs'):
        bind_native_method_artifacts(tmp_path / 'changed', {'operations': {
            'baseline': 'program_replay', 'challenger': 'structural_research'},
            'retained_method_pair_ref': old,
            'knowledge_contract_ref': {'file': 'new/knowledge_delta.json'}},
            {'file': 'new_decision.json'})


def test_shared_pair_entry_preserves_explained_pool_without_running_arms(tmp_path):
    from scripts.run_native_decision_pair import finish_prepared_pair

    result = finish_prepared_pair(tmp_path, {}, {}, None,
        {'comparison_complete': True, 'terminal': True}, 73, None)
    assert result == {'status': 'prepared_pool_requires_no_paired_research',
                      'arms_started': False, 'compute_cost': 73}
    assert not (tmp_path / 'baseline').exists()

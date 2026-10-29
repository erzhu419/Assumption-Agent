"""The existing common allowance funds a complete method-condition return."""

from scripts.run_native_decision_pair import method_review_allocation
from test_native_paired_method_feedback import common, deliver, route


def test_mixed_pair_conditions_use_unspent_preparation_without_more_common_budget(tmp_path):
    agent = common(tmp_path)
    deliver(agent, tmp_path, 'positive_one')
    deliver(agent, tmp_path, 'positive_two')
    deliver(agent, tmp_path, 'negative', stage='changed', cheaper=False)
    registration = {'preparation_compute_limit': 1000, 'method_review_compute_limit': 4}
    allocation = method_review_allocation(registration, 600)
    assert allocation['method_review_compute_limit'] == 404
    assert allocation['common_compute_limit'] == 1004
    result = route(agent, allowance=allocation['method_review_compute_limit'])
    learned = result['method_action_research']['paired_condition_research'][0]
    assert learned['status'] == 'development_condition_without_group_support'
    assert learned['candidates'] and learned['new_compute_cost'] > 0
    assert not learned['budget_incomplete'] and not learned['general_guard_supported']
    assert 600 + agent.costs['compute_cost'] <= allocation['common_compute_limit']


def test_same_direction_pairs_keep_unused_condition_allowance_unspent(tmp_path):
    agent = common(tmp_path)
    for name in ('one', 'two', 'three'):
        deliver(agent, tmp_path, name)
    allocation = method_review_allocation(
        {'preparation_compute_limit': 1000, 'method_review_compute_limit': 4}, 600)
    result = route(agent, allowance=allocation['method_review_compute_limit'])
    learned = result['method_action_research']['paired_condition_research'][0]
    assert learned['status'] == 'insufficient_paired_preferences'
    assert learned['new_compute_cost'] == 0 and learned['condition'] is None
    assert agent.costs['compute_cost'] == allocation['metadata_review_reserved'] == 4


def test_exhausted_common_preparation_preserves_metadata_and_reports_condition_shortfall(tmp_path):
    agent = common(tmp_path)
    deliver(agent, tmp_path, 'positive_one')
    deliver(agent, tmp_path, 'positive_two')
    deliver(agent, tmp_path, 'negative', stage='changed', cheaper=False)
    allocation = method_review_allocation(
        {'preparation_compute_limit': 1000, 'method_review_compute_limit': 4}, 1000)
    assert allocation['method_review_compute_limit'] == 4
    result = route(agent, allowance=allocation['method_review_compute_limit'])
    assert result['status'] == 'scoped_unknown'
    learned = result['method_action_research']['paired_condition_research'][0]
    assert learned['budget_incomplete'] and learned['new_compute_cost'] == 0
    assert 1000 + agent.costs['compute_cost'] == allocation['common_compute_limit']

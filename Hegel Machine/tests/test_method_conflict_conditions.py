"""Finite public-state research changes method use, not a world's predictor."""

from copy import deepcopy

from hegel_machine.cognition.method_conditions import (
    method_use_decision, research_method_conflict, same_research_condition)
from hegel_machine.cognition.models import SearchBudget
from test_method_use_conditions import BASIS, METHOD, feedback, profile


def outcome(index, levels, helpful, *, covered=None):
    result = feedback(after=.02 if helpful else .2, execution_complete=True)
    result['feedback_ref']['json_path'][-1] = index
    result['evidence_basis']['paid_cutoff']['rows'] += index
    result['profile']['data']['response_levels'] = levels
    if covered is not None:
        result['profile']['data']['conditional_coverage'] = [{'condition': 'startup',
            'coverage': [{'role': 0, 'observed': True, 'bins': list(range(covered))}]}]
    return result


def research(outcomes, **kwargs):
    return research_method_conflict(METHOD, outcomes, SearchBudget(100000), **kwargs)


def decide(outcomes, learned, target, *, basis=None):
    return method_use_decision(METHOD, target, outcomes,
        evidence_basis=basis or {'binding': 'new_paid_target'}, conflict_research=learned)


def test_omitted_public_condition_is_learned_and_changes_actual_admission():
    returns = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 3, False)]
    assert same_research_condition(returns[0]['profile'], returns[2]['profile'], 'representation')
    original = deepcopy(returns)
    learned = research(returns)
    assert learned['status'] == 'development_conditions_found'
    assert learned['compute_cost'] <= learned['required_compute']
    assert any(item['coordinates'] == [{'path': ['data', 'response_levels'], 'kind': 'number'}]
        for item in learned['candidates'])
    good = decide(returns, learned, returns[0]['profile'])
    bad = decide(returns, learned, returns[2]['profile'])
    assert good['admission'] == 'admit'
    assert good['next_action'] == 'try_or_continue_method'
    assert bad['admission'] == 'defer'
    assert bad['next_action'] == 'ordinary_research'
    assert bad['next_diagnostic']['kind'] == 'bounded_method_trial'
    assert bad['next_diagnostic']['execution_status'] == 'not_run'
    assert not good['independent_support'] and not bad['scope']['world_support_updated']
    assert learned['scope'] == 'exposed_development'
    assert returns == original


def test_measured_return_labels_not_a_handwritten_response_level_rule():
    first = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 3, False)]
    reversed_returns = [outcome(0, 3, True), outcome(1, 3, True), outcome(2, 2, False)]
    assert decide(first, research(first), first[0]['profile'])['admission'] == 'admit'
    assert decide(reversed_returns, research(reversed_returns), first[0]['profile'])['admission'] == 'defer'


def test_acquired_conditional_coverage_can_supply_a_distinction_without_changing_old_matching():
    returns = [outcome(0, 2, True, covered=3), outcome(1, 2, True, covered=3),
        outcome(2, 2, False, covered=1)]
    assert same_research_condition(returns[0]['profile'], returns[2]['profile'], 'representation')
    learned = research(returns)
    assert learned['condition']
    assert any('conditional_coverage' in coordinate['path'] for item in learned['candidates']
        for coordinate in item['coordinates'])
    assert decide(returns, learned, returns[2]['profile'])['admission'] == 'defer'


def test_same_public_view_conflict_ignores_private_case_labels_and_requests_step_evidence():
    returns = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 2, False)]
    for index, result in enumerate(returns):
        result['profile']['case_id'] = f'task{index}'
        result['profile']['private_kind'] = 'positive' if index < 2 else 'negative'
        result['future_truth'] = {'correct_mechanism': index}
    learned = research(returns)
    assert learned['condition'] is None
    conflict = learned['diagnostics'][0]
    assert conflict['status'] == 'conflict_observed'
    assert conflict['witnesses']
    assert learned['next_diagnostic']['required_evidence'] == [
        'complete_step_comparisons_at_the_conflicting_public_state']
    result = decide(returns, learned, returns[0]['profile'])
    assert result['admission'] == 'bounded_exploration'
    assert result['question']['diagnostic_plan']['witness_refs']
    assert not any('case_id' in item['path'] or 'private_kind' in item['path']
        for item in learned['coordinate_paths'])


def test_outside_measured_coverage_reopens_bounded_research_instead_of_banning_method():
    returns = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 3, False)]
    learned = research(returns)
    target = deepcopy(returns[2]['profile'])
    target['data']['response_levels'] = 4
    result = decide(returns, learned, target)
    assert result['admission'] == 'bounded_exploration'
    assert result['reason'] == 'outside_measured_public_coverage'
    missing = deepcopy(returns[2]['profile'])
    missing['data'].pop('feature_dimensions')
    assert decide(returns, learned, missing)['admission'] == 'bounded_exploration'


def test_competing_development_conditions_make_different_predictions_on_new_public_state():
    returns = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 3, False)]
    for result in returns:
        result['profile']['data']['max_probe_length'] = 2 if result['working_comparison']['after_loss'] < .1 else 3
    learned = research(returns)
    target = deepcopy(returns[0]['profile'])
    target['data']['max_probe_length'] = 3
    result = decide(returns, learned, target)
    assert result['admission'] == 'bounded_exploration'
    assert result['reason'] == 'competing_development_conditions_disagree'
    assert {item['helpful'] for item in result['question']['candidate_predictions']} == {True, False}


def test_known_paid_negative_stays_deferred_and_new_conflicting_feedback_invalidates_a_fit():
    returns = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 3, False)]
    learned = research(returns)
    exact = decide(returns, learned, returns[2]['profile'], basis=returns[2]['evidence_basis'])
    assert exact['next_action'] == 'defer_until_new_evidence'
    changed = returns + [outcome(3, 3, True)]
    result = decide(changed, learned, changed[-1]['profile'])
    assert result['admission'] == 'bounded_exploration'
    assert result['reason'] == 'method_feedback_changed_requires_new_diagnosis'
    assert research(changed)['condition'] is None


def test_complete_diagnostic_budget_is_checked_before_execution_and_paid_pass_is_reused():
    returns = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 3, False)]
    denied_budget = SearchBudget(1)
    denied = research_method_conflict(METHOD, returns, denied_budget)
    assert denied['budget_incomplete'] and denied['compute_cost'] == denied_budget.spent == 0
    assert denied['denied_request']['required_compute'] == denied['required_compute']
    budget = SearchBudget(denied['required_compute'])
    done = research_method_conflict(METHOD, returns, budget, previous=denied)
    assert not done['budget_incomplete'] and done['condition']
    later = SearchBudget(0)
    reused = research_method_conflict(METHOD, returns, later, previous=done)
    assert reused['reused'] and reused['compute_cost'] == later.spent == 0
    assert reused['paid_compute_cost'] == done['paid_compute_cost']


def test_incomplete_censored_and_other_method_records_never_supply_negative_training_labels():
    returns = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 3, False)]
    returns[-1]['administrative_censoring'] = {'kind': 'pause'}
    unrelated = outcome(3, 3, False)
    unrelated['method_ref']['version'] += 1
    learned = research(returns + [unrelated])
    assert learned['status'] == 'no_mixed_returns'
    assert learned['condition'] is None and learned['compute_cost'] == 0
    assert learned['unresolved_feedback_refs'] == [returns[-1]['feedback_ref']]


def process_comparison(*, cheaper=True, unknown=False):
    side = {'execution_complete': True, 'completed': True,
        'quality': {'loss': 1e-12, 'threshold': 1e-10}, 'coverage': 96,
        'costs': {'observation_cost': 120, 'compute_cost': 10000}}
    other = deepcopy(side)
    other['costs']['observation_cost'] = 96 if cheaper else 144
    other['costs']['compute_cost'] = 9000 if cheaper else 11000
    if unknown:
        other['costs']['compute_cost'] = None
    return {'same_task': True, 'same_evaluation_contract': True,
        'source_refs': [{'file': 'actual_complete_research.json'}],
        'scope': 'same_task_completed_research', 'before': side, 'after': other}


def test_same_completed_quality_with_lower_process_cost_is_a_positive_method_return():
    record = outcome(0, 2, False)
    record['working_comparison']['after_loss'] = record['working_comparison']['before_loss']
    record['process_comparison'] = process_comparison()
    result = method_use_decision(METHOD, record['profile'], [record],
        evidence_basis=record['evidence_basis'])
    assert result['admission'] == 'admit'
    assert result['deciding_return_basis']['kind'] == 'process_comparison'
    assert result['deciding_return_basis']['process_return']['resource_changes']['observation_cost']['direction'] == 'lower'
    assert not result['scope']['world_support_updated']


def test_process_return_labels_are_shared_by_conflict_research_and_calling_policy():
    returns = [outcome(0, 2, False), outcome(1, 2, False), outcome(2, 3, True)]
    for index, record in enumerate(returns):
        record['process_comparison'] = process_comparison(cheaper=index < 2)
    learned = research(returns)
    assert learned['condition']
    assert decide(returns, learned, returns[0]['profile'])['admission'] == 'admit'
    assert decide(returns, learned, returns[2]['profile'])['admission'] == 'defer'
    assert all(item['return_basis']['kind'] == 'process_comparison'
        for item in learned['development_witnesses'])


def test_unknown_resource_costs_stay_unknown_and_do_not_create_process_support():
    record = outcome(0, 2, True)
    record['process_comparison'] = process_comparison(unknown=True)
    result = method_use_decision(METHOD, record['profile'], [record], evidence_basis=BASIS)
    assert result['deciding_return_basis']['kind'] == 'working_comparison'
    assert not result['independent_support']


def test_existing_completed_repair_support_is_used_without_source_id_or_attempt_count():
    returns = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 2, False)]
    for index, result in enumerate(returns):
        result['profile']['source_revision'] = {'source_id': f'private_parent_{index}',
            'failed_attempts': index + 1, 'status': 'parameter_repair_exhausted',
            'support': {'history_depth': 0 if index < 2 else 2}}
    learned = research(returns)
    assert learned['condition']
    assert any(item['path'] == ['source_revision', 'support', 'history_depth']
        for item in learned['coordinate_paths'])
    assert not any(part in ('source_id', 'failed_attempts')
        for item in learned['coordinate_paths'] for part in item['path'])
    assert decide(returns, learned, returns[2]['profile'])['admission'] == 'defer'

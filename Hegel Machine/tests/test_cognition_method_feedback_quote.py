"""Response evidence reprices synthetic quotes separately from route expense."""

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.numeric_method_revisits import (
    METHOD_VISIT_ACTION, METHOD_RESPONSE_ACTION, refresh_numeric_method_quote,
    reprice_numeric_method_offer,
)


CONTEXT = 'squared:numeric:multiple:successor_method_paid_failure:' + METHOD_VISIT_ACTION + ':knowledge:1'
RESPONSE_CONTEXT = CONTEXT.replace(METHOD_VISIT_ACTION, METHOD_RESPONSE_ACTION)
APPLICABILITY = {'method_steps': [{'operation': 'replay_numeric_research_drafts'}],
    'response_basis': [{'op': 'window_sum', 'lag': 5}], 'commands': 3, 'readings': 1}
POOL = dict(observation_cost=1000, measurement_cost=2000, compute_cost=600000)


@pytest.fixture
def quote():
    return {'candidate': {'program': {'op': 'const', 'value': 3.},
        'method_experiment': {'probe': {'probe_id': 'paid_plan',
            'sequence': [{'command': float(i)} for i in range(24)]},
            'expected_disagreement': 0., 'source_draft_ids': ['original_draft']},
        'method_opportunity': {'draft_ids': ['original_draft'],
            'paid_counterexample_loss': .03, 'candidate_paid_loss': 4.,
            'predicted_separation': 0., 'planning_compute_cost': 30000,
            'method_applicability': deepcopy(APPLICABILITY)}},
        'old_evidence_gain': .03 - 4., 'old_evidence_rows': 120,
        'pricing_inputs': {'method_context': CONTEXT, 'compute_quote': 122000,
            'observation_quote': 72, 'task_compute_cap': 320000, 'measurement_rate': 2,
            'batch_length': 24, 'tolerance': 1e-7, 'condition_uncertainty': None,
            'condition_research_returns': None}}


def experience(*, gain=1., observations=144, measurement=288, compute=155000):
    return {CONTEXT: {METHOD_VISIT_ACTION: {'knowledge_applicability': deepcopy(APPLICABILITY),
        'trials': 1, 'expected_gain': 1., 'mean_observation_cost': observations,
        'mean_measurement_cost': measurement, 'mean_compute_cost': compute + 30000,
        'execution_costs': {'visits': 1, 'observation_cost': observations,
            'measurement_cost': measurement, 'compute_cost': compute}}},
        RESPONSE_CONTEXT: {METHOD_RESPONSE_ACTION: {
            'knowledge_applicability': deepcopy(APPLICABILITY),
            'trials': 1, 'expected_gain': gain}}}


def test_failed_quote_reopens_on_actual_applicable_return_with_the_same_paid_plan(quote):
    plan = deepcopy(quote['candidate']['method_experiment'])
    assert refresh_numeric_method_quote(quote, {}, POOL) is None
    assert quote['expected_marginal_value'] == 0.
    assert quote['minimum_resources']['compute_cost'] == 320000
    history = experience()
    before_history = deepcopy(history)
    assert refresh_numeric_method_quote(quote, history, POOL) is quote
    assert quote['old_evidence_gain'] < 0.
    assert quote['expected_marginal_value'] == 1.
    assert quote['minimum_resources'] == dict(observation_cost=144,
        measurement_cost=288, compute_cost=155000)
    assert quote['candidate']['method_experiment'] == plan
    assert history == before_history
    assert quote['method_feedback']['trials'] == 1


def test_unrelated_and_repeated_feedback_do_not_change_the_cached_quote(quote):
    refresh_numeric_method_quote(quote, {}, POOL)
    denied = deepcopy(quote)
    unrelated = experience()
    for actions in unrelated.values():
        for record in actions.values():
            record['knowledge_applicability']['response_basis'][0]['lag'] = 7
    assert refresh_numeric_method_quote(quote, unrelated, POOL) is None
    assert quote == denied
    history = experience()
    refresh_numeric_method_quote(quote, history, POOL)
    restored = json.loads(json.dumps(quote))
    for _ in range(2):
        assert refresh_numeric_method_quote(restored, history, POOL) is restored
        assert restored == quote


def test_negative_or_removed_applicable_returns_withdraw_the_pending_value(quote):
    refresh_numeric_method_quote(quote, experience(), POOL)
    positive = deepcopy(quote['method_feedback'])
    adverse = experience(gain=-.5, compute=210000)
    assert refresh_numeric_method_quote(quote, adverse, POOL) is None
    assert quote['relative_gain'] == 0.
    assert quote['method_feedback'] != positive
    assert quote['method_feedback']['expected_gain'] == -.5
    assert quote['minimum_resources']['compute_cost'] == 210000
    assert refresh_numeric_method_quote(quote, None, POOL) is None
    empty = deepcopy(quote)
    assert quote['method_feedback']['trials'] == 0
    assert quote['minimum_resources']['compute_cost'] == 320000
    assert refresh_numeric_method_quote(quote, {}, POOL) is None
    assert quote == empty


def test_execution_and_confirmation_floors_survive_cheap_historical_visits(quote):
    history = experience(observations=24, measurement=20, compute=40000)
    refresh_numeric_method_quote(quote, history, POOL)
    assert quote['minimum_resources'] == dict(observation_cost=72,
        measurement_cost=144, compute_cost=122000)
    # An attractive quote remains unaffordable if even its complete execution
    # floor cannot fit. Repricing never lowers it to consume the available pool.
    insufficient = {**POOL, 'compute_cost': 121999}
    refresh_numeric_method_quote(quote, history, insufficient)
    assert quote['minimum_resources']['compute_cost'] > insufficient['compute_cost']
    assert quote['candidate']['method_opportunity']['confirmation_rows'] == 24


def test_unconfirmed_attempt_cost_is_not_evidence_of_positive_prediction_return(quote):
    history = {CONTEXT: {METHOD_VISIT_ACTION: {
        'knowledge_applicability': deepcopy(APPLICABILITY), 'trials': 0,
        'unscored_attempts': 2, 'unscored_compute_cost': 800000,
        'unscored_observation_cost': 200, 'unscored_measurement_cost': 400,
        'execution_costs': {'visits': 2, 'observation_cost': 200,
            'measurement_cost': 400, 'compute_cost': 740000}}}}
    assert refresh_numeric_method_quote(quote, history, POOL) is None
    assert quote['minimum_resources'] == dict(observation_cost=100,
        measurement_cost=200, compute_cost=370000)
    assert quote['method_feedback']['unscored_attempts'] == 2
    assert quote['expected_marginal_value'] == 0.


def test_context_numbers_do_not_replace_the_actual_applicability(quote):
    history = experience()
    moved = {context.removesuffix('1') + '2': value for context, value in history.items()}
    moved[CONTEXT] = {METHOD_VISIT_ACTION: {
        'knowledge_applicability': {'different_method': True}, 'trials': 20, 'expected_gain': 50.}}
    moved[RESPONSE_CONTEXT] = {METHOD_RESPONSE_ACTION: {
        'knowledge_applicability': {'different_method': True}, 'trials': 20, 'expected_gain': 50.}}
    assert refresh_numeric_method_quote(quote, moved, POOL) is quote
    assert quote['method_feedback']['trials'] == 1
    assert quote['expected_marginal_value'] == pytest.approx(1.)


def test_saved_pending_route_gain_cannot_be_repriced_as_confirmed_response_evidence():
    path = Path(__file__).parent / 'fixtures/method_route_pending_root11.json.gz'
    paid = json.loads(gzip.decompress(path.read_bytes()))
    opportunity = paid['opportunity']
    assert paid['expected_marginal_value'] == 1.
    assert opportunity['observed_visit_trials'] == 1
    assert opportunity['observed_expected_gain'] == 1.
    assert opportunity['predicted_separation'] == 0.
    assert opportunity['candidate_paid_loss'] > opportunity['paid_counterexample_loss']
    # Only the actual saved pricing projection is needed. This test neither
    # reconstructs an executable probe nor invents the absent source evidence.
    queued = {'candidate': {'method_experiment': {}, 'method_opportunity': opportunity},
        'old_evidence_gain': opportunity['paid_counterexample_loss'] - opportunity['candidate_paid_loss']}
    original = deepcopy(queued)
    restored = json.loads(json.dumps(queued))
    reprice_numeric_method_offer(restored, POOL)
    assert restored['expected_marginal_value'] == 0.
    assert restored['expected_loss_reduction'] == 0.
    assert restored['relative_gain'] == 0.
    assert restored['candidate'] == original['candidate']

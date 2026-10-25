"""Operational exploration preserves finite epistemic uncertainty and evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_selection import select_question_plan, validate_exploration_policy
from test_question_selection import (X, SQUARE, action, experiment, public_experiment,
                                     public_query, question, readout, theory)


POLICY = {'kind': 'seeded_legal', 'seed': 9}


def choose(q, theories, plans, **kwargs):
    return select_question_plan(q, theories, plans, exploration_policy=POLICY, **kwargs)


def test_seeded_choice_prepares_one_action_but_preserves_the_epistemic_tie():
    q, theories = question(), [theory('linear'), theory('square', SQUARE)]
    plans = [experiment('first', 2.), experiment('second', 4.)]
    original = deepcopy((q, theories, plans, POLICY))
    old = select_question_plan(q, theories, plans)
    assert old['status'] == 'tied' and old['selected_experiment'] is None
    selected = choose(q, theories, plans)
    assert selected['status'] == 'tied' and selected['tied_experiments'] == ['first', 'second']
    assert selected['query_arguments']['actions'] == selected['selected_experiment']['actions']
    assert selected['operational_decision']['epistemic_status'] == 'tied'
    assert selected['operational_decision']['all_pool_optimality_established'] is False
    assert selected == choose(q, theories, plans)
    assert selected['evidence'] == [] and selected['support_updated'] is False
    assert selected['theory_status'] == 'unresolved'
    assert (q, theories, plans, POLICY) == original


def test_unknown_on_one_plan_does_not_discard_known_predictions_on_another():
    q = question()
    q['query_semantics']['query_expression'] = readout(3., history=[{'u': 3.}])
    lagged = Program('add', children=(X, Program('lag', name='u', lag=1)))
    theories = [theory('linear'), theory('square', SQUARE), theory('lagged', lagged)]
    cold, warm = experiment('cold', 2.), experiment('warm', 2.)
    warm.update(actions=[action(2., [{'u': 1.}])], query=readout(2., history=[{'u': 1.}]))
    old = select_question_plan(q, theories, [cold, warm])
    assert old['selected_experiment'] is None and old['unknown_reason'] == 'incomplete_candidate_predictions'
    result = choose(q, theories, [cold, warm])
    assert result['selected_experiment']['id'] == 'warm'
    lagged_group = next(group for group in result['groups'] if group['theory_ids'] == ['lagged'])
    assert lagged_group['panel_predictions'] == {'warm': (3.,)}
    assert lagged_group['prediction_status']['cold']['status'] == 'scoped_unknown'
    assert result['prediction_coverage']['pair_denominator'] == 3
    assert result['prediction_coverage']['common_comparable_pair_count'] == 1
    assert [(s['comparable_pairs'], s['unknown_pairs'], s['G']) for s in result['scores']] == [
        (1, 2, pytest.approx(1 / 3)), (3, 0, 1.)]
    assert result['prediction_coverage']['complete'] is False and result['next_work']


def test_different_plan_subpools_share_the_same_denominator_instead_of_comparing_local_G():
    q = question()
    q['query_semantics']['query_expression'] = readout(3., history=[{'u': 3.}])
    theories = [theory('zero', Program('const', value=0.)),
                theory('one', Program('const', value=1.)),
                theory('lagged', Program('lag', name='u', lag=1))]
    cold, warm = experiment('cold'), experiment('warm')
    warm.update(actions=[action(2., [{'u': 0.}])], query=readout(2., history=[{'u': 0.}]))
    result = choose(q, theories, [cold, warm])
    first, second = result['scores']
    assert first['separated_pairs'] == first['comparable_pairs'] == 1
    assert first['G'] == pytest.approx(1 / 3) and first['G_upper'] == 1.
    assert second['separated_pairs'] == 2 and second['comparable_pairs'] == 3
    assert second['G'] == pytest.approx(2 / 3)
    assert first['effective_pairs'] == second['effective_pairs'] == 3
    assert result['selected_experiment']['id'] == 'warm'


def test_unknown_task_answer_is_retained_outside_the_declared_task_known_subpool():
    theories = [theory('linear'), theory('square', SQUARE),
                theory('unavailable_task', Program('lag', name='u', lag=1))]
    result = choose(question(), theories, [experiment()])
    assert result['selected_experiment']['id'] == 'probe'
    coverage = result['prediction_coverage']
    assert coverage['scope'] == 'task_answer_known_subpool' and not coverage['complete']
    assert coverage['task_unknown_theory_ids'] == ['unavailable_task']
    assert coverage['pair_denominator'] == 1
    unknown, = [group for group in result['groups'] if not group['known']]
    assert unknown['theory_ids'] == ['unavailable_task']
    assert result['scores'][0]['partial'] is True and result['evidence'] == []


def test_physical_action_aliases_and_different_readouts_do_not_add_lottery_slots():
    theories = [theory('linear'), theory('square', SQUARE)]
    plans = [experiment('first', 2.), experiment('second', 4.)]
    aliases = []
    for index in range(40):
        alias = experiment(f'alias_{index}', 2.)
        alias['query']['record_index'] = 1  # Same action, another retained-readout contract.
        aliases.append(alias)
    ordinary, duplicated = choose(question(), theories, plans), choose(question(), theories, [*plans, *aliases])
    assert duplicated['selected_experiment']['actions'] == ordinary['selected_experiment']['actions']
    assert duplicated['operational_decision']['physical_option_count'] == 2
    options = duplicated['operational_decision']['physical_options']
    shared, = [option for option in options if len(option['alias_ids']) > 1]
    assert len(shared['alias_ids']) == 41 and shared['same_prediction_contract'] is False
    assert len(duplicated['tied_experiments']) == 42


def test_conditional_action_trees_are_deduplicated_and_only_the_first_action_is_prepared():
    plans = []
    for name, value in [('first', 2.), ('alias', 2.), ('other_branch', 4.)]:
        plan = experiment(name, 0.)
        plan['branches'] = [{'when': {'value': 0., 'atol': 0.},
                             'actions': [action(value)], 'query': readout(value)}]
        plans.append(plan)
    result = choose(question(), [theory('linear'), theory('square', SQUARE)], plans)
    assert result['status'] == 'tied' and result['operational_decision']['physical_option_count'] == 2
    assert result['query_arguments'] == {'actions': [action(0.)]}
    assert result['conditional_plan'] == result['selected_experiment']
    assert result['evidence'] == []


def test_hard_unit_or_public_row_contracts_cannot_be_bypassed_by_partial_exploration():
    theories = [theory('linear'), theory('square', SQUARE)]
    bad, valid = experiment('wrong_units'), experiment('valid', 4.)
    bad['query']['units'] = 'kg'
    result = choose(question(), theories, [bad, valid])
    assert result['selected_experiment']['id'] == 'valid'
    assert result['prediction_coverage']['contract_blocked_plans'] == ['wrong_units']
    alone = choose(question(), theories, [bad])
    assert alone['selected_experiment'] is None
    assert alone['unknown_reason'] == 'no_legal_discriminating_witness'
    lab = AnonymousHistoryLab()
    q = question()
    q['query_semantics']['query_expression'] = public_query([.8])
    bad = public_experiment('wrong_public_row', [.2, .7])
    bad['query']['binding']['features']['u'] = .9
    result = choose(q, theories, [bad], public_prediction_rows=lab.prediction_rows)
    assert result['selected_experiment'] is None
    assert result['excluded'] == [{'id': 'wrong_public_row', 'reason': 'public_row_query_mismatch'}]
    assert lab.costs == {'observation_cost': 0, 'measurement_cost': 0, 'queries': 0}


def test_reordered_conditional_branches_are_the_same_physical_option():
    plans = []
    for name, level in [('first', 2.), ('second', 4.)]:
        plan = experiment(name, 0.)
        plan['branches'] = [{'when': {'value': value, 'atol': 0.},
            'actions': [action(probe)], 'query': readout(probe)}
            for value, probe in [(0., level), (1., 5.)]]
        plans.append(plan)
    alias = deepcopy(plans[0])
    alias['id'] = 'reordered_alias'
    alias['branches'].reverse()
    result = choose(question(), [theory('linear'), theory('square', SQUARE)], [*plans, alias])
    assert result['operational_decision']['physical_option_count'] == 2


def test_zero_information_is_not_randomly_selected_and_keeps_a_constructive_reason():
    result = choose(question(), [theory('linear'), theory('square', SQUARE)], [experiment(level=1.)])
    assert result['selected_experiment'] is None and 'operational_decision' not in result
    assert result['scores'][0]['G'] == 0. and result['unknown_reason'] == 'no_discriminating_witness'
    assert result['next_work'] == {'kind': 'construct_or_validate_candidates',
        'task_unknown_theory_ids': [], 'missing_plan_predictions': {'probe': []},
        'reason': 'no_discriminating_witness'}


def test_clones_and_nontransitive_tolerance_do_not_change_partial_selection_cost_or_votes():
    theories = [theory(name, Program('const', value=value))
                for name, value in [('A', 0.), ('B', .75), ('C', 1.5)]]
    q, plans = question(), [experiment('first', 2., tolerance=.5), experiment('second', 4., tolerance=.5)]
    base = choose(q, theories, plans)
    copies = choose(q, [*theories, *[theory(f'clone_{i}', Program.from_dict(theories[0]['program']))
                                   for i in range(100)]], plans)
    assert base['scores'] == copies['scores'] and base['compute_cost'] == copies['compute_cost']
    assert base['pairwise_edges'] == copies['pairwise_edges']
    assert len(copies['groups']) == 3 and len(copies['groups'][0]['theory_ids']) == 101
    assert copies['scores'][0]['effective_pairs'] == 3 and copies['scores'][0]['G'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('policy', [{}, {'kind': 'other', 'seed': 1},
                                   {'kind': 'seeded_legal', 'seed': True}])
def test_exploration_policy_requires_the_explicit_finite_contract(policy):
    assert validate_exploration_policy(None) is None
    with pytest.raises(ValueError, match='exploration_policy'):
        select_question_plan(question(), [], [], exploration_policy=policy)


def test_common_comparable_pairs_match_legacy_intersection_with_missing_and_irrelevant_cells():
    from hegel_machine.cognition.question_selection import _partial_prediction_scores

    plans = [{'id': 'cold', 'tolerance': .5, 'observation_cost': 1},
             {'id': 'conditional', 'tolerance': .5, 'observation_cost': 2,
              'branches': [{'tolerance': .1}, {'tolerance': .1}]}]
    evaluated = {}
    for identifier, answer, predictions in [
            ('A', 0., {'cold': (0.,), 'conditional': (0., 0, 0.)}),
            ('alias_A', 0., {'cold': (0.,), 'conditional': (0., 0, 0.)}),
            ('B', .1, {'cold': (.2,), 'conditional': (0., 1, 6.)}),
            ('C', 3., {'conditional': (0., 0, 3.)}),
            ('D', 4., {'cold': (1.,)}),
            ('E', 1., {'cold': (2.,), 'conditional': (2., 0, 0.)})]:
        evaluated[identifier] = {'task_known': True, 'task': {'value': answer},
            'predictions': predictions, 'prediction_key': identifier,
            'prediction_status': {plan['id']: {'status': 'predicted'} if plan['id'] in predictions
                else {'status': 'scoped_unknown', 'reason': 'premise_unknown' if identifier == 'D'
                      else 'insufficient_history'} for plan in plans}}
    evaluated['task_unknown'] = {'task_known': False,
        'task': {'status': 'scoped_unknown', 'reason': 'insufficient_history'}}
    result = {'groups': [], 'pairwise_edges': [], 'scores': []}
    universe, blocked, task_blocked = _partial_prediction_scores(result, evaluated, plans, .25)
    coverage = result['prediction_coverage']
    comparable = {row['id']: row['comparable_pairs'] for row in coverage['plans']}
    # The original post-pass is a small independent oracle, including its order.
    legacy = [pair for pair in universe if all(pair in comparable[plan['id']] for plan in plans)]
    assert coverage['common_comparable_pairs'] == legacy == [
        ['group_000', 'group_004'], ['group_001', 'group_004']]
    assert coverage['common_comparable_pair_count'] == 2
    assert len(result['pairwise_edges']) == 10 and len(universe) == 9
    assert result['pairwise_edges'][0]['different_task_answer'] is False
    assert result['groups'][0]['theory_ids'] == ['A', 'alias_A']
    assert coverage['task_unknown_theory_ids'] == ['task_unknown']
    assert blocked == ['conditional'] and task_blocked is False
    assert result['unknown_reason'] == 'incomplete_candidate_predictions'
    assert [(score['separated_pairs'], score['comparable_pairs'], score['unknown_pairs'],
             score['effective_pairs'], score['common_comparable_pairs'], score['G'], score['G_upper'])
            for score in result['scores']] == [(2, 5, 4, 9, 2, 2 / 9, 6 / 9),
                                              (4, 5, 4, 9, 2, 4 / 9, 8 / 9)]


def test_large_partial_group_pool_keeps_all_edges_without_quadratic_pair_membership(monkeypatch):
    import hegel_machine.cognition.question_selection as selection

    count = 256  # A reachable distinct-forecast subset of the retained 1932-model roster.
    pair_count = count * (count - 1) // 2
    comparison_limit = pair_count * 2
    comparisons = 0

    class CountedID(str):
        def __eq__(self, other):
            nonlocal comparisons
            comparisons += 1
            assert comparisons <= comparison_limit, 'repeated pair-ID membership is quadratic'
            return str.__eq__(self, other)

    original_combinations = selection.combinations

    def tracked_pairs(groups, size):
        for group in groups:
            group['id'] = CountedID(group['id'])
        return original_combinations(groups, size)

    monkeypatch.setattr(selection, 'combinations', tracked_pairs)
    plans = [{'id': name, 'tolerance': 0.} for name in ('first', 'second')]
    evaluated = {str(i): {'task_known': True, 'task': {'value': float(i)},
        'predictions': {plan['id']: (float(i),) for plan in plans},
        'prediction_status': {plan['id']: {'status': 'predicted'} for plan in plans}}
        for i in range(count)}
    result = {'groups': [], 'pairwise_edges': [], 'scores': []}
    universe, blocked, task_blocked = selection._partial_prediction_scores(result, evaluated, plans, 0.)
    assert comparisons <= comparison_limit
    assert len(result['groups']) == count
    assert len(result['pairwise_edges']) == len(universe) == pair_count
    assert result['prediction_coverage']['common_comparable_pairs'] == universe
    assert universe[0] == ['group_000', 'group_001']
    assert universe[-1] == ['group_254', 'group_255']
    assert not blocked and not task_blocked
    assert all(score['effective_pairs'] == score['comparable_pairs'] == pair_count
               and score['G'] == score['G_upper'] == 1. for score in result['scores'])

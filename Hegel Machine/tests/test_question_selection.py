"""Finite-pool choices use executable consequences without updating support."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_selection import select_question_plan


CONTEXT = {'scene': 'public_reset_device', 'output_unit': 'V'}
X = Program('feature', name='u')
SQUARE = Program('mul', children=(X, X))


def action(level, history=None):
    return {'reset': True, 'inputs': [{'u': level}], 'history': deepcopy(history or [])}


def readout(level, *, history=None, index=0):
    return {'op': 'Readout', 'record_index': index, 'step': 0,
            'context_ref': deepcopy(CONTEXT), 'units': 'V',
            'binding': {'action': action(level, history), 'features': {'u': level},
                        'history': deepcopy(history or [])}}


def question(level=3.):
    return {'id': 'target_response', 'context_ref': deepcopy(CONTEXT),
            'query_semantics': {'mode': 'model_query', 'query_expression': readout(level)}}


def theory(identifier, program=X, **fields):
    return {'id': identifier, 'program': program.to_dict(),
            'role_signature': {'u': 'public_input', 'response': 'measured_voltage'},
            'output_unit': 'V', 'context_ref': deepcopy(CONTEXT),
            'support': {'status': 'candidate', 'evidence': []}, **fields}


def experiment(identifier='probe', level=2., tolerance=0., **fields):
    return {'id': identifier, 'actions': [action(level)], 'query': readout(level),
            'tolerance': tolerance, 'observation_cost': 1., **fields}


def test_T05_new_probe_splits_current_signature_without_merging_source_mechanisms():
    theories = [theory('linear'), theory('quadratic', SQUARE)]
    q = question(0.)
    old_panel = [experiment('old', 1.)]
    before = deepcopy((q, theories, old_panel))
    old = select_question_plan(q, theories, old_panel)
    assert len(old['groups']) == 1
    assert old['groups'][0]['theory_ids'] == ['linear', 'quadratic']
    assert old['groups'][0]['task_answer'] == 0.
    assert old['status'] == 'scoped_unknown'
    assert (q, theories, old_panel) == before

    expanded = select_question_plan(q, theories, old_panel + [experiment('new', 2.)])
    assert len(expanded['groups']) == 2
    assert [g['theory_ids'] for g in expanded['groups']] == [['linear'], ['quadratic']]
    assert expanded['pairwise_edges'] == [{'left': 'group_000', 'right': 'group_001',
        'different_task_answer': False, 'experiments': ['new']}]
    assert all(score['effective_pairs'] == 0 and score['G'] == 0 for score in expanded['scores'])
    assert expanded['selected_experiment'] is None  # The current task answer already agrees.

    relevant = select_question_plan(question(3.), theories, old_panel + [experiment('new', 2.)])
    assert relevant['status'] == 'selected' and relevant['selected_experiment']['id'] == 'new'
    assert relevant['scores'][1]['G'] == 1.


def test_T06_near_chain_remains_pairwise_and_declared_tolerance_is_fixed():
    theories = [theory(name, Program('const', value=value))
                for name, value in [('A', 0.), ('B', .75), ('C', 1.5)]]
    probe = experiment(tolerance=.5)
    result = select_question_plan(question(), theories, [probe])
    assert len(result['groups']) == 3
    assert [(edge['left'], edge['right'], edge['experiments'])
            for edge in result['pairwise_edges']] == [
                ('group_000', 'group_001', []),
                ('group_000', 'group_002', ['probe']),
                ('group_001', 'group_002', [])]
    assert result['scores'][0]['effective_pairs'] == 3
    assert result['scores'][0]['separated_pairs'] == 1
    assert result['scores'][0]['G'] == pytest.approx(1 / 3)
    assert result['selected_experiment']['tolerance'] == probe['tolerance'] == .5

    boundary = select_question_plan(question(),
        [theory('A', Program('const', value=0.)), theory('B', Program('const', value=1.))], [probe])
    assert boundary['scores'][0]['G'] == 0.
    assert boundary['unknown_reason'] == 'no_discriminating_witness'


def test_task_resolution_is_pairwise_without_merging_signatures_or_changing_probe_tolerance():
    theories = [theory(name, Program('const', value=value))
                for name, value in [('A', 0.), ('B', .75), ('C', 1.5)]]
    q, probe = question(), experiment(tolerance=0.)
    q['answer_resolution_ref'] = {'atol': .75}
    result = select_question_plan(q, theories, [probe])
    assert len(result['groups']) == 3
    assert [edge['different_task_answer'] for edge in result['pairwise_edges']] == [False, True, False]
    assert result['scores'][0]['effective_pairs'] == 1
    assert result['scores'][0]['G'] == 1.
    assert all(edge['experiments'] == ['probe'] for edge in result['pairwise_edges'])
    assert result['selected_experiment']['tolerance'] == 0.
    q.pop('answer_resolution_ref')
    exact = select_question_plan(q, theories, [probe])
    assert exact['scores'][0]['effective_pairs'] == 3


@pytest.mark.parametrize('history, expected', [(None, -4.), ([{'u': 2.}], 2.)])
def test_explicit_native_evaluation_uses_archived_fallback_only_for_missing_history(history, expected):
    q = question()
    q['query_semantics']['query_expression'] = readout(3., history=history)
    theories = [theory('linear'), theory('lagged', Program('lag', name='u', lag=1))]
    raw = select_question_plan(q, theories, [experiment()])
    assert raw['unknown_reason'] == 'incomplete_candidate_predictions'
    q['query_semantics']['evaluation'] = {'kind': 'native_evaluation', 'fallback': -4.,
        'source_ref': {'file': 'retained.json', 'field': 'episodes[6].fallback'}}
    result = select_question_plan(q, theories, [experiment()])
    lagged = next(group for group in result['groups'] if 'lagged' in group['theory_ids'])
    assert lagged['task_answer'] == expected
    assert result['status'] == 'selected' and all(group['known'] for group in result['groups'])
    assert result['evidence'] == [] and result['support_updated'] is False


def test_T07_hundred_clones_add_neither_votes_compute_nor_support():
    base = [theory('linear'), theory('quadratic', SQUARE)]
    panel = [experiment()]
    original = deepcopy(base)
    ordinary = select_question_plan(question(), base, panel, compute_budget=4)
    clones = [theory(f'clone_{index}') for index in range(100)]
    copied = select_question_plan(question(), base + clones, panel, compute_budget=4)
    assert ordinary['status'] == copied['status'] == 'selected'
    assert ordinary['scores'] == copied['scores']
    assert ordinary['compute_cost'] == copied['compute_cost'] == 4
    assert ordinary['pairwise_edges'] == copied['pairwise_edges']
    assert len(copied['groups'][0]['theory_ids']) == 101
    assert copied['groups'][1]['theory_ids'] == ['quadratic']
    assert copied['evidence'] == [] and copied['support_updated'] is False
    assert base == original


def test_budget_exhaustion_is_unknown_and_unknown_signatures_never_merge():
    theories = [theory('linear'), theory('quadratic', SQUARE), theory('linear_clone')]
    result = select_question_plan(question(), theories, [experiment()], compute_budget=1)
    assert result['status'] == 'scoped_unknown' and result['selected_experiment'] is None
    assert result['unknown_reason'] == 'incomplete_candidate_predictions'
    assert result['compute_cost'] == 1
    assert len(result['groups']) == 3
    assert all(group['known'] is False and len(group['theory_ids']) == 1 for group in result['groups'])
    assert all(group['unknown_reason'] == 'unknown_due_to_incomplete_search' for group in result['groups'])
    assert result['pairwise_edges'] == []


def test_same_completed_query_is_executed_once_per_distinct_program():
    result = select_question_plan(question(2.), [theory('linear'), theory('quadratic', SQUARE)],
                                  [experiment('a'), experiment('b')], compute_budget=2)
    assert result['status'] == 'tied' and result['tied_experiments'] == ['a', 'b']
    assert result['compute_cost'] == 2 and result['selected_experiment'] is None


def test_equal_G_uses_declared_cost_order_and_keeps_exact_cost_ties():
    theories = [theory('linear'), theory('quadratic', SQUARE)]
    expensive = experiment('expensive', 2., observation_cost=2.)
    cheap_compute = experiment('cheap_compute', 4., compute_cost=1.)
    cheaper_model = experiment('cheaper_model', 5., compute_cost=0., model_cost=8.)
    selected = select_question_plan(question(), theories, [expensive, cheap_compute, cheaper_model])
    assert [score['G'] for score in selected['scores']] == [1., 1., 1.]
    assert selected['selected_experiment']['id'] == 'cheaper_model'
    assert selected['query_arguments'] == {'actions': cheaper_model['actions']}
    tied = experiment('tied', 6., compute_cost=0., model_cost=8.)
    result = select_question_plan(question(), theories, [tied, cheaper_model])
    assert result['status'] == 'tied' and result['tied_experiments'] == ['tied', 'cheaper_model']
    assert result['selected_experiment'] is None and 'query_arguments' not in result


def test_tied_information_coverage_does_not_imply_equal_numeric_predictions():
    result = select_question_plan(question(), [theory('linear'), theory('quadratic', SQUARE)],
                                  [experiment('a', 2.), experiment('b', 4.)])
    assert result['status'] == 'tied' and result['selected_experiment'] is None
    comparison = result['tie_diagnostics']['comparisons'][0]
    assert comparison['same_G'] and comparison['same_task_relevant_coverage']
    assert comparison['same_pairwise_separation'] and comparison['same_exact_prediction_partition']
    assert comparison['same_candidate_predictions'] is False
    assert comparison['different_prediction_theory_count'] == 2
    assert comparison['prediction_difference_witness']['left'] == (2.,)
    assert comparison['prediction_difference_witness']['right'] == (4.,)


def test_equal_G_can_cover_different_candidate_pairs_and_keeps_the_tie():
    theories = [theory('linear'), theory('quadratic', SQUARE), theory('constant', Program('const', value=2.))]
    result = select_question_plan(question(), theories, [experiment('a', 0.), experiment('b', 2.)])
    assert result['status'] == 'tied' and result['selected_experiment'] is None
    assert [score['G'] for score in result['scores']] == [2 / 3, 2 / 3]
    comparison = result['tie_diagnostics']['comparisons'][0]
    assert comparison['same_G'] is True and comparison['same_task_relevant_coverage'] is False
    assert comparison['same_pairwise_separation'] is False
    assert comparison['same_exact_prediction_partition'] is False
    assert comparison['different_task_pair_count'] == comparison['different_separation_pair_count'] == 2


def test_tie_diagnostics_preserve_exact_partitions_and_nontransitive_threshold_edges():
    theories = [theory(name, Program('const', value=value))
                for name, value in [('A', 0.), ('B', .75), ('C', 1.5)]]
    result = select_question_plan(question(), theories,
        [experiment('a', 2., tolerance=.5), experiment('b', 3., tolerance=.5)])
    assert result['status'] == 'tied'
    diagnostic = result['tie_diagnostics']
    assert all(len(item['exact_prediction_partition']) == 3 for item in diagnostic['experiments'])
    assert all(item['all_separated_pairs'] == 1 for item in diagnostic['experiments'])
    assert diagnostic['comparisons'][0]['same_candidate_predictions'] is True
    assert result['compute_cost'] == 6 and result['support_updated'] is False


def test_unaffordable_and_incompatible_candidates_do_not_change_selected_panel():
    theories = [theory('linear'), theory('quadratic', SQUARE),
                theory('refuted', Program('const', value=-100.), support={'status': 'refuted'}),
                theory('incompatible', Program('const', value=100.), compatible=False)]
    result = select_question_plan(question(), theories,
        [experiment('too_expensive', observation_cost=2.), experiment('affordable')], observation_budget=1.)
    assert result['selected_experiment']['id'] == 'affordable'
    assert result['scores'][0]['effective_pairs'] == 1
    assert result['excluded'] == [{'id': 'too_expensive', 'reason': 'observation_budget'},
        {'id': 'refuted', 'reason': 'incompatible_candidate'},
        {'id': 'incompatible', 'reason': 'incompatible_candidate'}]


def test_T25_local_relation_predictions_do_not_become_global_scientific_support():
    q = question(3.)
    q['query_semantics']['query_expression'] = {'op': 'RelationResidual',
        'query': readout(3.), 'expected': 9., 'relation': 'eq', 'atol': .01,
        'premises': ['local_calibration']}
    theories = [theory('linear'), theory('quadratic', SQUARE)]
    original = deepcopy((q, theories))
    result = select_question_plan(q, theories, [experiment()], context={
        'context_ref': CONTEXT, 'premises': {'local_calibration': True}})
    assert result['selected_experiment']['id'] == 'probe'
    assert [group['task_answer'] for group in result['groups']] == [-6., 0.]
    assert result['context_ref'] == CONTEXT and result['scope'] == 'finite_candidate_heuristic'
    assert result['evidence'] == [] and result['support_updated'] is False
    assert result['theory_status'] == 'unresolved'
    assert (q, theories) == original


def test_T26_posterior_labels_do_not_claim_pool_exhaustiveness_or_weight_G():
    theories = [theory('linear', posterior=.999999), theory('quadratic', SQUARE, posterior=.000001)]
    result = select_question_plan(question(), theories, [experiment()])
    assert result['scores'][0]['G'] == 1. and result['scores'][0]['effective_pairs'] == 1
    assert result['scope'] == 'finite_candidate_heuristic'
    assert 'posterior' not in result and 'confidence' not in result
    assert result['theory_status'] == 'unresolved' and result['support_updated'] is False


def test_T27_supported_task_action_can_finish_without_buying_mechanism_information():
    theories = [theory('linear'), theory('quadratic', SQUARE)]
    finish = {'id': 'finish', 'direct_task_action': True,
              'support': {'status': 'supported', 'evidence': ['paid_record_14']},
              'actions': [{'action': 'APPLY', 'value': 3.}], 'observation_cost': 0.}
    original = deepcopy((theories, finish))
    result = select_question_plan(question(), theories, [experiment(), finish], compute_budget=0)
    assert result['status'] == 'direct_action_available'
    assert result['selected_experiment'] == finish and result['query_arguments']['actions'] == finish['actions']
    assert result['compute_cost'] == 0 and result['groups'] == []
    assert result['theory_status'] == 'unresolved' and result['evidence'] == []
    assert result['support_updated'] is False and (theories, finish) == original
    del finish['support']
    unsupported = select_question_plan(question(), theories, [finish, experiment()])
    assert unsupported['status'] == 'selected' and unsupported['selected_experiment']['id'] == 'probe'


def test_two_step_plan_executes_public_history_and_only_prepares_first_action():
    lagged = Program('add', children=(X, Program('lag', name='u', lag=1)))
    theories = [theory('linear'), theory('lagged', lagged)]
    q = question()
    q['query_semantics']['query_expression'] = readout(3., history=[{'u': 2.}])
    plan = experiment('conditional', 0.)
    plan['query'] = readout(0., history=[{'u': 0.}])
    plan['actions'] = [deepcopy(plan['query']['binding']['action'])]
    second = readout(2., history=[{'u': 4.}], index=1)
    plan['branches'] = [
        {'when': {'value': 0., 'atol': 0.}, 'actions': [second['binding']['action']],
         'query': second},
        {'when': {'value': 10., 'atol': 0.}, 'actions': [action(99.)], 'query': readout(99.)}]
    original = deepcopy((q, theories, plan))
    result = select_question_plan(q, theories, [plan])
    assert result['status'] == 'selected' and result['scores'][0]['G'] == 1.
    assert [group['panel_predictions']['conditional'] for group in result['groups']] == [(0., 0, 2.), (0., 0, 6.)]
    assert result['query_arguments'] == {'actions': [action(0., [{'u': 0.}])]}
    assert result['conditional_plan'] == plan and result['evidence'] == []
    assert (q, theories, plan) == original


def test_different_conditional_second_actions_do_not_fake_comparable_readouts():
    theories = [theory('zero', Program('const', value=0.)),
                theory('one', Program('const', value=1.))]
    plan = experiment('conditional', tolerance=1.)
    plan['branches'] = [
        {'when': {'value': value, 'atol': 0.}, 'actions': [action(level)],
         'query': readout(level), 'tolerance': 0.}
        for value, level in [(0., 2.), (1., 4.)]]
    result = select_question_plan(question(), theories, [plan])
    assert result['scores'][0]['G'] == 0.
    assert result['status'] == 'scoped_unknown'
    assert result['unknown_reason'] == 'no_discriminating_witness'


def test_context_and_unit_checks_remain_PR1_binding_checks():
    theories = [theory('linear'), theory('quadratic', SQUARE)]
    result = select_question_plan(question(), theories, [experiment()], context=deepcopy(CONTEXT))
    assert result['status'] == 'selected'
    invalid = select_question_plan(question(), theories, [experiment()],
                                   context={'context_ref': {'scene': 'other'}, 'output_unit': 'V'})
    assert invalid['status'] == 'scoped_unknown' and invalid['selected_experiment'] is None
    assert invalid['groups'] == [] and invalid['compute_cost'] == 0
    assert invalid['unknown_reason'] == 'no_compatible_candidates'
    assert invalid['excluded'] == [{'id': 'linear', 'reason': 'context_mismatch'},
                                  {'id': 'quadratic', 'reason': 'context_mismatch'}]
    bad_unit = theory('wrong_unit', Program('const', value=100.), output_unit='mV')
    selected = select_question_plan(question(), theories + [bad_unit], [experiment()])
    assert selected['status'] == 'selected' and len(selected['groups']) == 2
    assert selected['excluded'] == [{'id': 'wrong_unit', 'reason': 'unit_mismatch'}]


def test_unbound_panel_query_is_excluded_without_changing_task_query_or_global_indices():
    theories = [theory('linear'), theory('quadratic', SQUARE)]
    mismatch = experiment('mismatch')
    mismatch['query']['binding']['action'] = action(99.)
    valid = experiment('valid')
    valid['query']['record_index'] = 45
    q = question(3.)  # The task input need not itself be an experiment action.
    result = select_question_plan(q, theories, [mismatch, valid])
    assert result['selected_experiment']['id'] == 'valid'
    assert result['excluded'] == [{'id': 'mismatch', 'reason': 'action_query_mismatch'}]
    assert [score['experiment_id'] for score in result['scores']] == ['valid']
    assert result['scores'][0]['G'] == 1.
    assert result['selected_experiment']['query']['record_index'] == 45


def test_wrong_second_step_binding_excludes_the_whole_conditional_plan():
    theories = [theory('linear'), theory('quadratic', SQUARE)]
    plan = experiment('conditional', 0.)
    plan['branches'] = [{'when': {'value': 0., 'atol': 0.},
        'actions': [action(2.)], 'query': {'op': 'RelationResidual', 'query': readout(99.),
            'expected': 0., 'relation': 'eq', 'atol': 0.}}]
    result = select_question_plan(question(), theories, [plan, experiment('valid')])
    assert result['selected_experiment']['id'] == 'valid'
    assert result['excluded'] == [{'id': 'conditional', 'reason': 'action_query_mismatch'}]


def public_query(levels, *, record_index=45):
    request = {'kind': 'trajectory', 'sequence': [{'u': level} for level in levels], 'channels': []}
    query = readout(levels[-1], history=request['sequence'][:-1], index=record_index)
    query['step'] = len(levels) - 1
    query['binding']['action'] = request
    return query


def public_experiment(identifier, levels):
    query = public_query(levels)
    return {'id': identifier, 'actions': [query['binding']['action']], 'query': query,
            'tolerance': 0., 'observation_cost': len(levels)}


@pytest.mark.parametrize('changed', ['features', 'history', 'step', 'step_out_of_range'])
def test_real_public_interface_excludes_wrong_row_bindings_before_scoring(changed):
    lab = AnonymousHistoryLab()
    q = question(.8)
    q['query_semantics']['query_expression'] = public_query([.8], record_index=129)
    bad = public_experiment('bad', [-.4, .7])
    if changed == 'features':
        bad['query']['binding']['features']['u'] = .6
    elif changed == 'history':
        bad['query']['binding']['history'] = []
    else:
        bad['query']['step'] = 0 if changed == 'step' else 2
    valid = public_experiment('valid', [-.2, .9])
    original = deepcopy((q, bad, valid))
    costs = lab.costs
    result = select_question_plan(q, [theory('linear'), theory('quadratic', SQUARE)],
        [bad, valid], public_prediction_rows=lab.prediction_rows)
    assert result['selected_experiment']['id'] == 'valid'
    assert result['excluded'] == [{'id': 'bad', 'reason': 'public_row_query_mismatch'}]
    assert result['public_row_validation'] == {'available': True, 'validated_experiments': ['valid']}
    assert result['scores'][0]['G'] == 1. and result['scores'][0]['effective_pairs'] == 1
    assert len(result['scores']) == 1 and lab.costs == costs
    assert (q, bad, valid) == original


def test_missing_public_interface_does_not_claim_row_validation():
    result = select_question_plan(question(), [theory('linear'), theory('quadratic', SQUARE)],
                                  [experiment()])
    assert result['status'] == 'selected'
    assert result['public_row_validation'] == {'available': False, 'validated_experiments': []}


def test_real_public_interface_checks_legal_conditional_rows_and_branch_mismatch():
    lab = AnonymousHistoryLab()
    lagged = Program('add', children=(X, Program('lag', name='u', lag=1)))
    theories = [theory('linear'), theory('lagged', lagged)]
    q = question()
    q['query_semantics']['query_expression'] = public_query([.2, .8], record_index=129)
    conditional = public_experiment('conditional', [0., 0.])
    branch = public_experiment('second', [.4, .6])
    conditional['branches'] = [{'when': {'value': 0., 'atol': 0.},
        'actions': branch['actions'], 'query': branch['query']}]
    conditional['observation_cost'] = 4
    before = deepcopy((q, theories, conditional))
    result = select_question_plan(q, theories, [conditional], public_prediction_rows=lab.prediction_rows)
    assert result['status'] == 'selected' and result['scores'][0]['G'] == 1.
    assert [group['panel_predictions']['conditional'] for group in result['groups']] == [(0., 0, .6), (0., 0, 1.)]
    assert result['query_arguments'] == {'actions': conditional['actions']}
    assert result['public_row_validation'] == {'available': True, 'validated_experiments': ['conditional']}
    assert (q, theories, conditional) == before

    conditional['branches'][0]['query']['binding']['history'] = []
    valid = public_experiment('valid', [-.2, .9])
    bad = select_question_plan(q, theories, [conditional, valid], public_prediction_rows=lab.prediction_rows)
    assert bad['selected_experiment']['id'] == 'valid'
    assert bad['excluded'] == [{'id': 'conditional', 'reason': 'public_row_query_mismatch'}]
    assert bad['public_row_validation']['validated_experiments'] == ['valid']
    assert lab.costs == {'observation_cost': 0, 'measurement_cost': 0, 'queries': 0}

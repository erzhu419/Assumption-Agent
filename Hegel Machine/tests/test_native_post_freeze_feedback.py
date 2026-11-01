"""Delayed independent outcomes revise the same method-use evidence."""

import json

import pytest

from test_native_paired_method_feedback import common, deliver, route, save
from hegel_machine.cognition.common_object_research import record_native_parent_research_return
from scripts.run_learned_structure_successor import consume_frozen_pair


def retained_pair(tmp_path):
    agent = common(tmp_path)
    feedback, pair_ref = deliver(agent, tmp_path, 'source')
    pair = json.loads(open(pair_ref['file']).read())
    frozen = {}
    for side in ('baseline', 'challenger'):
        path = pair[side + '_ref']['file']
        value = json.loads(open(path).read())
        value.update(full_comparison=True,
                     selected_ref={'file': side + '_actual_predictor.json'})
        save(tmp_path / path, value)
        frozen[side] = {'selected_ref': value['selected_ref']}
    source = tmp_path / 'source'
    save(source / 'confirmation/frozen_predictors.json', frozen)
    save(source / 'confirmation/receipt.json', {
        'status': 'frozen_pair_readout_completed', 'rows': 2,
        'arms': {'baseline': {'mse': .1, 'predictions': [0, 0]},
                 'challenger': {'mse': .2, 'predictions': [1, 1]}},
        'shared_actual_costs': {'compute_cost': 4, 'observation_cost': 2}})
    return agent, pair_ref, pair, source


def test_later_pair_reverses_local_support_without_adding_an_invocation(tmp_path):
    agent, pair_ref, pair, source = retained_pair(tmp_path)
    first = route(agent)['method_action_research']['paired_condition_research'][0]
    assert first['local_program_supported']
    pair.update(paired_future_ref={'file': str(source / 'confirmation/receipt.json')},
                frozen_predictors_ref={'file': str(source / 'confirmation/frozen_predictors.json')})
    later = save(source / 'later_pair.json', pair)
    original = json.loads(open((source / 'receipt.json')).read())
    original['method_pair_ref'] = later
    record_native_parent_research_return(agent, source_task_ref=pair['physical_source_ref'],
        result_ref=save(source / 'later_return.json', original))
    learned = route(agent)['method_action_research']['paired_condition_research'][0]
    assert learned['paired_invocations'] == 1
    assert learned['preferences'][0]['preference'] == 'baseline'
    assert not learned['local_program_supported'] and not learned['general_guard_supported']
    assert learned['preferences'][0]['evaluation_use'] == 'exposed_development_label_after_outputs_freeze'
    assert json.loads(open(pair['challenger_ref']['file']).read())['loss'] == 1e-20


def test_frozen_scores_cannot_be_attributed_to_a_replaced_predictor(tmp_path):
    agent, pair_ref, pair, source = retained_pair(tmp_path)
    pair.update(paired_future_ref={'file': str(source / 'confirmation/receipt.json')},
                frozen_predictors_ref={'file': str(source / 'confirmation/frozen_predictors.json')})
    frozen = json.loads((source / 'confirmation/frozen_predictors.json').read_text())
    frozen['challenger']['selected_ref'] = {'file': 'another_predictor.json'}
    save(source / 'confirmation/frozen_predictors.json', frozen)
    from hegel_machine.cognition.method_feedback import research_paired_method_conditions
    with pytest.raises(ValueError, match='actual returned predictor'):
        research_paired_method_conditions(agent, {'id': 'native:structural_research', 'version': 1},
            [save(source / 'later_pair.json', pair)], 100)


def test_saved_readout_reaches_common_without_science_or_repeat_charge(tmp_path):
    agent, pair_ref, pair, source = retained_pair(tmp_path)
    route(agent)
    prior = tmp_path
    # Common's existing topic export is the continuation input.
    output = tmp_path / 'post_freeze'
    result = consume_frozen_pair(output, source, pair_ref=pair_ref,
                                prior_review=prior, compute_limit=100)
    groups = result['method_return']['research_result']['method_action_research']['paired_condition_research']
    assert groups[0]['preferences'][0]['preference'] == 'baseline'
    assert groups[0]['paired_invocations'] == 1
    assert result['new_predictions'] == result['new_fits'] == result['new_observations'] == 0
    repeated = consume_frozen_pair(output, source, compute_limit=100)
    assert repeated['reused'] and repeated['new_compute_cost'] == 0


def test_explicit_native_arm_names_reach_common_without_rewriting_paid_scores(tmp_path):
    agent, _, pair, source = retained_pair(tmp_path)
    scores_path, frozen_path = source / 'confirmation/receipt.json', source / 'confirmation/frozen_predictors.json'
    scores, frozen = json.loads(scores_path.read_text()), json.loads(frozen_path.read_text())
    scores['arms'] = {'Pi1': scores['arms']['challenger'], 'P0': scores['arms']['baseline']}
    frozen = {'Pi1': frozen['challenger'], 'P0': frozen['baseline']}
    # The named scientific result is frozen before its condition-learning use.
    scores['arms']['Pi1']['mse'] = 1e-20
    save(scores_path, scores)
    save(frozen_path, frozen)
    paid_bytes = scores_path.read_bytes(), frozen_path.read_bytes()
    pair.update(paired_future_ref={'file': str(scores_path)},
        frozen_predictors_ref={'file': str(frozen_path)},
        evaluation_arm_names={'baseline': 'P0', 'challenger': 'Pi1'})
    actual = json.loads((source / 'receipt.json').read_text())
    actual['method_pair_ref'] = save(source / 'named_pair.json', pair)
    record_native_parent_research_return(agent, source_task_ref=pair['physical_source_ref'],
        result_ref=save(source / 'named_return.json', actual))
    group = route(agent)['method_action_research']['paired_condition_research'][0]
    assert group['paired_invocations'] == 1
    assert group['preferences'][0]['preference'] == 'challenger'
    assert group['preferences'][0]['losses'] == [.1, 1e-20]
    assert group['local_program_supported'] and not group['general_guard_supported']
    assert group['status'] == 'insufficient_paired_preferences'
    assert (scores_path.read_bytes(), frozen_path.read_bytes()) == paid_bytes
    previous_cost = agent.costs['compute_cost']
    assert route(agent)['cache_hit']
    assert agent.costs['compute_cost'] == previous_cost


def test_named_pair_keeps_actual_predictor_attribution(tmp_path):
    agent, _, pair, source = retained_pair(tmp_path)
    scores_path, frozen_path = source / 'confirmation/receipt.json', source / 'confirmation/frozen_predictors.json'
    scores, frozen = json.loads(scores_path.read_text()), json.loads(frozen_path.read_text())
    scores['arms'] = {'Pi1': scores['arms']['challenger'], 'P0': scores['arms']['baseline']}
    frozen = {'Pi1': frozen['challenger'], 'P0': frozen['baseline']}
    frozen['Pi1']['selected_ref'] = {'file': 'later_unconfirmed_predictor.json'}
    save(scores_path, scores)
    save(frozen_path, frozen)
    pair.update(paired_future_ref={'file': str(scores_path)},
        frozen_predictors_ref={'file': str(frozen_path)},
        evaluation_arm_names={'baseline': 'P0', 'challenger': 'Pi1'})
    from hegel_machine.cognition.method_feedback import research_paired_method_conditions
    with pytest.raises(ValueError, match='actual returned predictor'):
        research_paired_method_conditions(agent, {'id': 'native:structural_research', 'version': 1},
            [save(source / 'named_pair.json', pair)], 100)

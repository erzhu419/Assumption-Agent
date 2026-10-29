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

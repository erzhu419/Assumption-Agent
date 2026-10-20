"""Historical connection checks keep the autonomous and statistical scope clear."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def replay_module():
    specification = importlib.util.spec_from_file_location('question_contract_replay',
        ROOT / 'scripts/replay_question_contract_extensions.py')
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    if not module.POINTER.is_file() or not Path(json.loads(module.POINTER.read_text())['archive']).is_file():
        pytest.skip('The retained shared archive is not mounted')
    return module


@pytest.fixture(scope='module')
def replay(replay_module, tmp_path_factory):
    return replay_module.replay(output=tmp_path_factory.mktemp('question_contract') / 'receipt.json')


def test_real_new_question_recalls_an_object_without_its_id_and_selector_consumes_it(replay):
    result = replay['automatic_frozen_recall']
    assert result['status'] == 'completed'
    protocol = result['protocol']
    assert protocol['source_episode_indices'] == [0, 5] and protocol['target_episode_index'] == 14
    assert not protocol['candidate_id_supplied_to_successor']
    assert protocol['explicit_theory_transfer_calls'] == 0
    assert not protocol['target_final_program_or_parameters_used']
    assert not protocol['target_labels_used_for_recall_or_selection']
    assert protocol['mission']['purpose'] == 'task' and protocol['recall_uses_existing_task_budget']
    assert 'theory_learning_allocation' not in protocol['mission']
    assert result['automatic_recall']['generated_theory_ids']
    assert result['generated_candidates_consumed_by_selector']
    assert result['original_pool_preserved']
    assert result['selection']['status'] == 'tied'
    assert result['selection']['selected_experiment'] is None
    excerpt = result['saved_evaluation_excerpt']
    assert excerpt['row_count'] == 4
    assert max(excerpt['recalled_candidate_mse'].values()) < 1e-12 < excerpt['initial_parent_mse']
    assert not excerpt['predictor_adoption_executed'] and not excerpt['selected_experiment_executed']
    assert not result['independent_confirmation']


def test_repeat_has_no_new_compute_or_library_write_and_retains_the_real_counterexample(replay):
    result = replay['automatic_frozen_recall']
    assert result['repeat'] == {'cache_hit': True, 'recall_cache_hit': True,
        'new_compute_cost': 0, 'costs_unchanged': True, 'library_write_calls': 0}
    assert result['source_construction']['source_reconstructions_preserved']
    assert result['source_construction']['retained_failed_binding_count'] > 0
    for cost in (result['costs']['source_construction'], result['costs']['successor']):
        assert cost['observation_cost'] == cost['measurement_cost'] == cost['model_calls'] == 0
    assert result['automatic_recall']['new_compute_cost'] > 0


def test_target_final_program_and_later_labels_do_not_choose_a_binding_or_experiment(replay_module, replay):
    payload, source = replay_module.load_source(replay_module.POINTER)
    changed = deepcopy(payload)
    changed['episodes'][14]['program'] = {'op': 'const', 'value': 999.}
    for row in changed['episodes'][14]['evaluation']['prediction_examples']:
        row['target'] += 100.
    actual = replay_module.automatic_frozen_recall(changed, source)
    original = replay['automatic_frozen_recall']
    for key in ('automatic_recall', 'selection', 'costs'):
        assert actual[key] == original[key]
    assert actual['saved_evaluation_excerpt']['predictions'] == original['saved_evaluation_excerpt']['predictions']
    assert actual['saved_evaluation_excerpt']['recalled_candidate_mse'] != original['saved_evaluation_excerpt']['recalled_candidate_mse']


def test_the_existing_finite_application_still_consumes_the_actual_failure(replay):
    result = replay['finite_application']
    assert result['finite_answer']['value'] is False
    assert result['finite_answer']['scope']['case_count'] == 2
    assert result['parent_revision_completed'] and result['pending_plan_removed']
    assert result['repeat_compute'] == 0 and result['repeat_status'] == 'reused_answer'
    assert result['new_paid_observations'] == result['new_provider_calls'] == 0

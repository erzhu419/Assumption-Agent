"""Matched historical A/B/C keeps the strong archive and unresolved bindings."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, TaskSpec


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def replay_module():
    path = ROOT / 'scripts/replay_three_arm_historical_routing.py'
    specification = importlib.util.spec_from_file_location('three_arm_historical_routing', path)
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def historical_replay(replay_module, tmp_path_factory):
    if not replay_module.POINTER.exists():
        pytest.skip('The retained historical source pointer is absent')
    if not Path(json.loads(replay_module.POINTER.read_text())['archive']).exists():
        pytest.skip('The retained historical archive is not mounted')
    output = tmp_path_factory.mktemp('three_arm_routing') / 'receipt.json'
    return replay_module.replay_three_arms(output=output)


def test_all_three_arms_keep_complete_old_bindings_fitting_order_initial_parent_and_real_counterexample(historical_replay):
    receipt = historical_replay
    assert receipt['status'] == 'completed'
    assert receipt['comparison']['same_old_candidate_programs']
    assert receipt['comparison']['same_native_development_scores']
    for arm in receipt['arms'].values():
        assert arm['retained_source_program_count'] == arm['old_template_count'] == 2
        assert arm['old_template_binding_count'] == 4 and arm['initial_parent_retained']
        assert arm['raw_proposal_count'] == 8 and len(arm['candidate_program_ids']) == 7
        assert arm['counterexample']['status'] == 'refuted'
        assert arm['counterexample']['checks'][0]['status'] == 'failed'
        assert arm['counterexample']['ordinary_alias_compatible'] is False
        assert arm['counterexample']['source_instance_preserved'] is True
        assert arm['new_model_calls'] == arm['new_observations'] == 0
        assert arm['parent']['independent_confirmation'] is False


def test_original_task_value_and_PR2_consume_the_same_full_panel_without_executing_it(historical_replay):
    receipt = historical_replay
    assert receipt['protocol']['public_panel_action_indices'] == [0, 1, 2, 9]
    assert receipt['protocol']['same_total_compute_budget'] == 50000
    a, b, c = (receipt['arms'][name] for name in ('A', 'B', 'C'))
    assert a['selection']['policy'] == 'original_task_value'
    assert len(a['selection']['details']['evaluated_probes']) == 4
    assert {item['probe_id'] for item in a['selection']['details']['evaluated_probes']} == {
        row['experiment_id'] for row in b['selection']['scores']}
    assert a['selection']['selected_experiment_id'] == 'initial_02'
    assert b['selection']['scores'] == c['selection']['scores']
    assert b['selection']['selected_experiment_id'] == c['selection']['selected_experiment_id'] == 'representative_0'
    for arm in (a, b, c):
        assert arm['selection']['selected_action_executed'] is False
        assert arm['selection']['action_outcome_known'] is False
        assert arm['action_benefit'].startswith('UNKNOWN')


def test_induction_recalls_all_legal_bindings_without_adding_distinct_old_pool_computations(historical_replay):
    receipt = historical_replay
    assert set(receipt['protocol']['full_typed_target']['role_signature']) == {'u', 'v'}
    assert receipt['protocol']['candidate_id_supplied_to_successor_recall'] is False
    assert receipt['source_construction']['source_reconstruction_preserved']
    assert [receipt['arms'][name]['frozen_object_count'] for name in ('A', 'B', 'C')] == [0, 0, 1]
    recalled = receipt['arms']['C']['automatic_recall']
    assert recalled['status'] == 'completed' and len(recalled['generated_theory_ids']) == 2
    assert not recalled['author_work']
    assert [item['mapping'] for item in recalled['work']] == [{'role_0': 'u'}, {'role_0': 'v'}]
    assert historical_replay['arms']['C']['new_recalled_candidate_program_ids'] == []
    assert not recalled['support_updated'] and not recalled['independent_confirmation']
    for arm in receipt['arms'].values():
        costs = arm['costs']
        assert costs['total'] == sum(value for key, value in costs.items() if key not in {'total', 'remaining'})
        assert costs['total'] + costs['remaining'] == 50000
        assert costs['old_native_propose'] == 18 and costs['native_prefix_comparison'] == 14
    routing_delta = receipt['arms']['C']['costs']['routing_including_recall'] - receipt['arms']['B']['costs']['routing_including_recall']
    assert routing_delta == recalled['new_compute_cost'] + 8  # Two extra sources receive four public readouts each.
    assert receipt['comparison']['C_minus_B_compute'] == receipt['source_construction']['compute_cost'] + routing_delta
    assert receipt['comparison']['same_parent_program'] and receipt['comparison']['same_successor_predictions']
    assert all(arm['parent']['suffix_mse'] == pytest.approx(3.8466553728192677e-19)
               for arm in receipt['arms'].values())


def test_deleting_only_parent_use_keeps_selection_but_does_not_prepare_a_parent_query(historical_replay):
    ablation = historical_replay['parent_use_ablation']
    keep, remove = (ablation['arms'][name] for name in ('keep', 'remove'))
    assert ablation['protocol']['intervention'] == {'field': 'question.use_sites', 'operation': 'delete_all'}
    assert ablation['protocol']['cold_start_without_completed_routing_or_recall']
    assert keep['question_parent_id'] == remove['question_parent_id'] == 'task'
    assert keep['use_sites'] == [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]
    assert remove['use_sites'] == []
    assert keep['parent_use']['status'] == 'parent_query_prepared'
    assert keep['parent_use']['parent_revision_completed'] is True
    assert remove['parent_use']['status'] == 'selection_ready_no_parent_use'
    assert remove['parent_use']['parent_revision_completed'] is False
    assert keep['pending_before'] is remove['pending_before'] is None
    assert keep['pending_after']['actions'] and keep['pending_source_after']['kind'] == 'question_selection'
    assert remove['pending_after'] is remove['pending_source_after'] is None
    assert all(ablation['comparison'].values())
    assert keep['selection_ref'] == remove['selection_ref'] == {
        'receipt_field': ['parent_use_ablation', 'shared_results', 'selection']}
    assert ablation['shared_results']['selection']['selected_experiment']['id'] == 'representative_0'


def test_parent_use_ablation_preserves_complete_old_capability_counterexamples_and_real_costs(historical_replay):
    ablation = historical_replay['parent_use_ablation']
    for arm in ablation['arms'].values():
        assert arm['old_pool_preserved'] and arm['retained_parent_program_unchanged']
        assert arm['ordinary_counterexample_alias_blocked']
        assert arm['same_total_compute_budget'] == 50000
        assert arm['new_observations'] == arm['new_model_calls'] == 0
        assert arm['selected_action_executed'] is arm['action_outcome_known'] is False
        assert arm['subsequent_prediction_effect'] == 'UNKNOWN_selected_panel_not_executed'
        assert arm['automatic_recall_ref'] == {'receipt_field': [
            'parent_use_ablation', 'shared_results', 'automatic_recall']}
        assert [item['mapping'] for item in ablation['shared_results']['automatic_recall']['work']] == [
            {'role_0': 'u'}, {'role_0': 'v'}]
        assert arm['compute_cost'] == 51
        assert arm['accounted_total_compute'] == (
            ablation['protocol']['shared_setup_accounted_compute'] + arm['compute_cost'])
    assert ablation['actual_harness_compute'] == 102


def test_saved_successor_labels_cannot_change_frozen_candidates_development_scores_or_public_plans(
        replay_module, historical_replay, monkeypatch, tmp_path):
    payload, source = replay_module.load_source(replay_module.POINTER)
    changed = deepcopy(payload)
    for index in replay_module.SUFFIX_INDICES:
        changed['episodes'][14]['evaluation']['prediction_examples'][index]['target'] += 10.
    monkeypatch.setattr(replay_module, 'load_source', lambda _: (changed, source))
    replay = replay_module.replay_three_arms(output=tmp_path / 'different_successor_labels.json')
    assert replay['parent_use_ablation'] == historical_replay['parent_use_ablation']
    for name, arm in replay['arms'].items():
        old = historical_replay['arms'][name]
        assert arm['candidate_program_ids'] == old['candidate_program_ids']
        assert arm['candidate_scores'] == old['candidate_scores']
        assert arm['selection'] == old['selection']
        assert arm['parent']['program_id'] == old['parent']['program_id']
        assert arm['parent']['suffix_predictions'] == old['parent']['suffix_predictions']
        assert arm['parent']['suffix_mse'] > 99


def test_generic_policy_exact_rank_tie_is_preserved_without_arbitrary_action(replay_module):
    data = {'entries': [{'candidate_id': str(value), 'program': Program('const', value=float(value)),
                        'operation': 'retained'} for value in (0, 1)],
        'fallback': .5, 'prefix': [Observation({'u': 0., 'v': 0.}, .5)],
        'spec': TaskSpec('tie_fixture', 'Symmetric public controls', ('u', 'v'), (), ())}
    panel = [{'id': str(value), 'actions': [{'sequence': [{'u': float(value), 'v': 0.}]}],
              'observation_cost': 1} for value in (-1, 1)]
    result = replay_module.generic_plan(data, panel, 50000)
    assert result['status'] == 'tied' and result['selected_experiment_id'] is None
    assert set(result['tied_experiments']) == {'-1', '1'}
    assert result['selected_action_executed'] is False

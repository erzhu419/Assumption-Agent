"""Ablations retain old executable transfer and report actual, possibly null effects."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.abstractions import extract_templates
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.theory_induction import consolidate_theories, instantiate_induction


ROOT = Path(__file__).resolve().parents[1]


def affine(name, intercept, slope):
    return Program('add', children=(Program('const', value=intercept),
        Program('mul', children=(Program('const', value=slope), Program('feature', name=name)))))


def source_episode(identifier, program):
    return {'episode_id': identifier, 'source_id': identifier,
        'source_ref': {'file': 'retained_sources.json', 'json_path': [identifier]},
        'program': program.to_dict(),
        'role_signature': {'x': {'role': 'public_numeric_input', 'unit': 'input_coordinate', 'time': 'current'}},
        'output_unit': 'response_coordinate', 'context_ref': {'source_id': identifier},
        'context_family_ref': 'declared_affine_topic'}


def test_removing_the_new_frozen_object_retains_existing_role_and_parameter_transfer():
    sources = [source_episode('t01', affine('x', .3, 1.4)),
               source_episode('t06', affine('x', -.1, 1.7))]
    before = deepcopy(sources)
    frozen, = consolidate_theories(sources)['candidates']
    original_bindings, new_bindings = [], []
    for source, instance in zip(sources, frozen['instances']):
        old_template = extract_templates(Program.from_dict(source['program']), source['episode_id'])[0]
        for mapping in old_template.mappings(('u', 'v')):
            original_bindings.append(old_template.instantiate(mapping))
            new_bindings.append(instantiate_induction(frozen, {**instance, 'mapping': mapping}))
    assert len(original_bindings) == len(set(original_bindings)) == 4
    assert new_bindings == original_bindings
    assert sources == before
    assert frozen['transfer_status'] == 'not_established' and not frozen['global_support']


@pytest.fixture(scope='module')
def ablation_module():
    path = ROOT / 'scripts/replay_frozen_theory_ablation.py'
    specification = importlib.util.spec_from_file_location('frozen_theory_ablation', path)
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def historical_ablation(tmp_path_factory, ablation_module):
    module = ablation_module
    pointer = module.POINTER
    if not pointer.is_file():
        pytest.skip('The retained historical source pointer is not present in this checkout')
    source = json.loads(pointer.read_text())
    if not Path(source['archive']).is_file():
        pytest.skip('The retained historical archive is not mounted')
    output = tmp_path_factory.mktemp('frozen_ablation') / 'receipt.json'
    return module, module.replay_ablation(pointer=pointer, output=output)


def _arm_result():
    return {'candidate_program_ids': ['p0'],
        'selection': {'signature_predictions': [{'task_answer': 0., 'panel_predictions': [0.]}],
                      'scores': [{'G': 0.}], 'selected_experiment_id': None, 'actual_common_compute': 1},
        'evaluation': {'prediction_matrix': {'p0': [0., 0.]}},
        'native_parent_choice': {'selected_program_id': 'p0', 'suffix_predictions': [0., 0.]}}


@pytest.mark.parametrize('changed', ['scores', 'prediction', 'parent_program', 'suffix'])
def test_comparison_reports_a_real_non_null_difference_without_forcing_the_historical_answer(ablation_module, changed):
    module = ablation_module
    remove, keep = _arm_result(), _arm_result()
    assert module.compare_arm_results(keep, remove)['incremental_effect_observed'] is False
    if changed == 'scores':
        keep['selection']['scores'][0]['G'] = 1.
        field = 'same_scores'
    elif changed == 'prediction':
        keep['evaluation']['prediction_matrix']['p0'][0] = 3.
        field = 'same_predictions'
    elif changed == 'parent_program':
        keep['native_parent_choice']['selected_program_id'] = 'p1'
        field = 'same_parent_selected_program'
    else:
        keep['native_parent_choice']['suffix_predictions'][0] = 3.
        field = 'same_suffix_predictions'
    compared = module.compare_arm_results(keep, remove)
    assert compared[field] is False and compared['incremental_effect_observed'] is True


def test_extra_compute_is_reported_separately_from_changed_scientific_behavior(ablation_module):
    remove, keep = _arm_result(), _arm_result()
    keep['selection']['actual_common_compute'] = 2
    compared = ablation_module.compare_arm_results(keep, remove)
    assert compared['same_common_compute'] is False
    assert compared['incremental_effect_observed'] is False


def test_historical_remove_retains_all_existing_program_bindings_and_the_initial_parent(historical_ablation):
    _, receipt = historical_ablation
    keep, remove = receipt['arms']['keep'], receipt['arms']['remove']
    for arm in (keep, remove):
        assert arm['retained_source_program_count'] == arm['old_template_count'] == 2
        assert arm['old_template_binding_count'] == 4
        assert arm['program_count'] == 5
    assert remove['new_frozen_object_retained'] is False and remove['new_frozen_binding_count'] == 0
    assert keep['new_frozen_object_retained'] is True and keep['new_frozen_binding_count'] == 4
    assert remove['raw_candidate_count'] == 5 and keep['raw_candidate_count'] == 9
    assert set(keep['candidate_program_ids']) == set(remove['candidate_program_ids'])
    catalog = {identifier: Program.from_dict(program) for identifier, program in receipt['program_catalog'].items()}
    retained = [catalog[identifier] for identifier in remove['candidate_program_ids']]
    assert any(program.op == 'const' for program in retained)


def test_historical_clones_do_not_change_actual_signatures_cost_or_experiment_choice(historical_ablation):
    _, receipt = historical_ablation
    keep, remove = receipt['arms']['keep'], receipt['arms']['remove']
    assert keep['signature_count'] == remove['signature_count'] == 5
    assert keep['selection']['scores'] == remove['selection']['scores']
    assert keep['selection']['signature_predictions'] == remove['selection']['signature_predictions']
    assert keep['selection']['selected_experiment_id'] == remove['selection']['selected_experiment_id']
    assert keep['selection']['actual_common_compute'] == remove['selection']['actual_common_compute']
    assert keep['evaluation']['prediction_matrix'] == remove['evaluation']['prediction_matrix']
    for arm in (keep, remove):
        assert arm['selection']['selected_action_executed'] is False
        assert arm['selection']['action_outcome_known'] is False
        assert arm['evaluation']['labels_used_for_plan_selection'] is False


def test_actual_parent_candidate_is_selected_on_prefix_and_then_predicts_saved_suffix(historical_ablation):
    _, receipt = historical_ablation
    for arm in receipt['arms'].values():
        selected = arm['native_parent_choice']
        assert selected['fit_row_indices'] == selected['selection_row_indices'] == [0, 1]
        assert selected['suffix_row_indices'] == [2, 3]
        assert selected['old_propose_raw_candidates'] == 8
        assert selected['unique_candidate_count'] == 7
        assert selected['old_propose_compute'] == 18
        assert selected['prefix_comparison_compute'] == 14
        scores = selected['candidate_scores']
        chosen = next(entry for entry in scores if entry['program_id'] == selected['selected_program_id'])
        assert chosen['prefix_mse'] == min(entry['prefix_mse'] for entry in scores)
        assert selected['selected_prefix_mse'] == chosen['prefix_mse']
        assert len(selected['suffix_predictions']) == selected['suffix_evaluation_rows'] == 2
        assert selected['independent_confirmation'] is False
        assert selected['common_paid_observations_created'] == 0
    keep = receipt['arms']['keep']['native_parent_choice']
    remove = receipt['arms']['remove']['native_parent_choice']
    assert keep['selected_program_id'] == remove['selected_program_id']
    assert keep['suffix_predictions'] == remove['suffix_predictions']
    assert keep['suffix_mse'] == remove['suffix_mse']


def test_both_arms_retain_the_actual_counterexample_without_new_paid_records(historical_ablation):
    _, receipt = historical_ablation
    keep, remove = receipt['arms']['keep'], receipt['arms']['remove']
    assert keep['retained_counterexample'] == remove['retained_counterexample']
    assert keep['retained_counterexample']['history_preserved'] is True
    retained_failure = keep['retained_counterexample']
    assert retained_failure['matched_step'] == 1 and retained_failure['history_has_observed_target'] is True
    assert retained_failure['transfer_recall_status'] == 'refuted'
    assert retained_failure['transfer_recall_compute'] == 0
    assert any(check['status'] == 'failed' for check in retained_failure['executed_checks'])
    assert retained_failure['ordinary_alias_compatible'] is False
    assert retained_failure['ordinary_alias_scope_refutation']
    assert retained_failure['source_instance_not_withdrawn'] is True
    assert retained_failure['other_parameters_not_withdrawn'] is True
    checks = keep['retained_counterexample']['source_counterexample_checks']
    assert any(check['status'] == 'failed' and check['domain'] == 'transfer' for check in checks)
    for arm in (keep, remove):
        assert not any(arm['no_acquisition'].values())
    assert receipt['comparison']['incremental_effect_observed'] is False
    for field in ('same_candidate_programs', 'same_signature_predictions', 'same_scores',
                  'same_selected_experiment', 'same_predictions',
                  'same_parent_selected_program', 'same_suffix_predictions'):
        assert receipt['comparison'][field] is True


def test_diagnostic_predictions_and_frozen_construction_keep_their_actual_cost(historical_ablation):
    _, receipt = historical_ablation
    keep, remove = receipt['arms']['keep'], receipt['arms']['remove']
    for arm in (keep, remove):
        native = arm['native_parent_choice']
        assert arm['selection']['actual_common_compute'] == 20
        assert arm['matrix_diagnostics']['compute'] == arm['matrix_diagnostics']['row_predictions'] == 20
        assert native['suffix_evaluation_compute'] == 2
        assert native['comparison_and_suffix_compute'] == 16
        assert native['comparison_and_suffix_compute'] <= native['comparison_and_suffix_budget']
        costs = arm['accounted_compute']
        assert costs['total_replay_compute_excluding_saved_source_learning'] == sum(
            value for key, value in costs.items() if key != 'total_replay_compute_excluding_saved_source_learning')
    assert remove['accounted_compute']['total_replay_compute_excluding_saved_source_learning'] == 75
    preparation = keep['frozen_preparation_compute']
    # The current mainline also scans actual revisions for cross-skeleton
    # contexts. Keep that real cost; the earlier 28-unit receipt stays historical.
    assert remove['frozen_preparation_compute'] == 0 and preparation > 28
    assert keep['accounted_compute']['total_replay_compute_excluding_saved_source_learning'] == 75 + preparation
    assert receipt['incremental_cost']['total_keep_minus_remove_compute'] == preparation
    assert receipt['incremental_cost']['new_object_construction_compute'] == preparation
    assert receipt['comparison']['given_object_route_compute_difference'] == 0
    assert receipt['comparison']['incremental_behavior_effect_observed'] is False

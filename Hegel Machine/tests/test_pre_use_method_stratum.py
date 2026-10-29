"""Public method-branch sampling keeps nonmatching and unknown sources intact."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from scripts import run_independent_state_method_source as entry
from scripts import run_state_method_pairing as study
from scripts.run_method_condition_cohort import source_receipt
from test_native_representation_routing import PublicOnlyLab
from test_state_method_cohort import frozen_pair
from test_state_method_pairing import paid_fixture


@pytest.mark.parametrize('threshold,coverage,expected,admitted,charge', [
    (5., [0., 100.], False, True, 1),
    (3., [0., 100.], True, False, 1),
    (5., [10., 100.], None, False, 0),
])
def test_registered_false_branch_uses_paid_profile_without_changing_shared_start(
        tmp_path, monkeypatch, threshold, coverage, expected, admitted, charge):
    paid, case, values = paid_fixture(tmp_path / 'paid')
    source = study.read(paid)
    bundle_ref = frozen_pair(tmp_path / 'methods', paid, case)
    bundle = study.read(bundle_ref)
    challenger = bundle['cells']['10']
    original_body = deepcopy(challenger['selection_plan'])
    coordinates = [{'path': ['data', 'max_history'], 'kind': 'number'}]
    selected_ref = study.save(tmp_path / 'frozen_selection.json', {
        'general_guard_supported': True, 'coordinate_paths': coordinates,
        'measured_coverage': {'public_0': coverage}})
    challenger.update(id='frozen_guard', selection_ref=selected_ref,
        selection_plan={'ordinary_method_path': 'ordinary.py',
            'condition': Program('ge', children=(Program('feature', name='public_0'),
                Program('const', value=threshold))).to_dict(),
            'condition_feature_paths': coordinates,
            'when_true': {k: v for k, v in original_body.items() if k != 'ordinary_method_path'},
            'when_false': {}})
    guarded_ref = study.save(tmp_path / 'guarded_bundle.json', bundle)
    prefix_ref = study.save(tmp_path / 'prefix_result.json', {
        'paid_rows': source['expected_rows'], 'records_ref': source['paid_records_ref'],
        'costs': {'compute_cost': 0, 'observation_cost': source['expected_rows']}})
    study.save(tmp_path / 'public_spec.json', values['public'])
    request = {'case': values['case'], 'paid_prefix_ref': prefix_ref,
        'parents': source['parents'], 'diagnostic_parent_id': 'constant',
        'prefix_rows': source['expected_rows'], 'evaluation_seed': 1002, 'evaluation_rows': 20,
        'validation_records': source['validation_records'], 'research_compute_cap': 10000,
        'per_cell_compute_cap': 30000, 'per_cell_activity_seconds': 60,
        'per_cell_python_seconds': 30, 'source_activity_seconds': 120,
        'source_python_seconds': 30, 'source_comparison_compute_upper': 40,
        'mse_threshold': 1e-10, 'comparison_scope': 'complete_two_parent_pool',
        'method_pair_ref': guarded_ref, 'data_role': 'independent_successor',
        'pre_use_condition_scope': {'method_cell': '10', 'branch': 'when_false'}}

    class RestoreOnlyLab(PublicOnlyLab):
        def __init__(self, **_):
            self._past_costs = {'observation_cost': 0, 'measurement_cost': 0, 'queries': 0}

        @property
        def costs(self):
            return self._past_costs

        def observation_rows(self, record):
            return [Observation(row['features'], row['target'], tuple(row['history']))
                    for row in record['rows']]

        def evaluation_rows(self, **_):
            raise AssertionError('screening cannot acquire frozen evaluation labels')

    monkeypatch.setattr(study, 'AnonymousHistoryLab', RestoreOnlyLab)
    comparisons = []
    original_execute = CommonArchiveInquiryAgent._execute

    def recorded_execute(self, action, arguments, lab):
        assert action == 'run' and arguments['operator'] == 'research_choice'
        comparisons.append(self.workspace)
        return original_execute(self, action, arguments, lab)

    monkeypatch.setattr(CommonArchiveInquiryAgent, '_execute', recorded_execute)
    pairs = []

    def run_pair(registration, destination):
        assert admitted
        original = study.read(registration['initial_scene_ref'])
        assert 'theory_consolidation' not in original
        assert registration['quote']['status'] == 'quoted'
        pairs.append(registration)
        return {'status': 'selected_stratum_reached_research'}

    monkeypatch.setattr(entry, 'run_pair', run_pair)
    output = tmp_path / 'source'
    result = entry.run(request, output)
    assert len(comparisons) == 1  # The sole already-required shared pool comparison.
    screen = study.read(result['pre_use_condition_ref'])
    assert screen['condition_value'] is expected
    assert screen['matched'] is admitted
    assert screen['new_compute_cost'] == charge
    assert screen['new_world_predictions'] == screen['new_fits'] == screen['new_observations'] == 0
    assert screen['condition_features'] == {'public_0': 4.}
    assert bool(pairs) is admitted
    start = study.read(result['initial_scene_ref'])
    assert 'theory_consolidation' not in start
    assert start['evaluated_model']['source_model'] == 'old.py'
    assert result['shared_comparison_costs']['compute_cost'] == 40
    if not admitted:
        assert result['status'] == ('method_condition_stratum_unknown' if expected is None
                                   else 'method_condition_other_branch')
        receipt = source_receipt(result, output)
        assert not receipt['arms'] and not receipt['pair_refs']
        assert receipt['costs']['observation_cost'] == 0
        assert receipt['costs']['compute_cost'] == 40 + result['static_proposal_costs']['compute_cost']
    original_static = study.read({'file': str(output / 'static_selection_after_comparison.json')})
    assert result['static_proposal_costs']['compute_cost'] == (
        original_static['previous_preparation_costs']['compute_cost']
        + original_static['new_compute_cost'] + charge)

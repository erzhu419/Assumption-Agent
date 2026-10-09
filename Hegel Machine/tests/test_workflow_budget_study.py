"""Check identical paid starting states and isolated, delayed scoring."""

import importlib.util
import json
from pathlib import Path


def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_workflow_budget_study.py'
    spec = importlib.util.spec_from_file_location('workflow_budget_study', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_roster_keeps_old_caps_and_all_feedback_in_both_arms():
    study = runner()
    roster = study.cohort_protocol()
    old = json.loads((study.BASELINE.parent / 'protocol.json').read_text())['cases']
    assert len(roster['cases']) == 12
    assert not {case['seed'] for case in old} & {case['seed'] for case in roster['cases']}
    for case in roster['cases']:
        assert case['arms'] == ['baseline', 'repair']
        for key in ('observation_budget', 'compute_budget', 'max_rounds', 'max_candidates'):
            assert case['research_config'][key] == old[0]['research_config'][key]
        assert case['measurement_budget'] == old[0]['measurement_budget']
        assert case['partial_compute_budget'] == old[0]['partial_compute_budget']
        assert 'neither arm deletes feedback' in case['information_contract']


def test_real_frozen_runtime_roundtrip_preserves_prefix_and_full_costs(tmp_path):
    study = runner()
    plan = study.protocol(65009001)
    plan['research_config'].update(observation_budget=96, compute_budget=12000, max_rounds=1)
    for task in [*plan['return_tasks'], *plan['competition_tasks'], *plan['tasks']]:
        task['task_index'] = 0
    plan['tasks'][1]['task_index'] = 1
    plan['evaluation'] = {'length': 8, 'trajectories': 1, 'seed_offset': 1000003}
    directory = tmp_path / 'case'
    study.prepare_case(plan, directory)
    common = json.loads((directory / 'common.json').read_text())
    before = json.loads((directory / 'baseline_training.json').read_text())
    assert before['state']['evaluation'] == []
    assert all(state['evaluation'] == [] for state in common['prefix_states'].values())
    assert not list(directory.glob('*scored.json'))
    # The frozen implementation in both positions isolates runner mechanics from an unfinished repair.
    result = study.complete_case(directory, repair_root=study.BASELINE)
    repair = json.loads((directory / 'repair_training.json').read_text())
    assert repair['state']['evaluation'] == []
    for key in ('initial_costs', 'initial_concept_ids', 'initial_workflows', 'initial_tasks', 'initial_config'):
        assert before[key] == repair[key]
    assert before['runtime_module'] == repair['runtime_module'] == str(
        study.BASELINE / 'scripts/run_workflow_demand_study.py')
    assert result['paired_repair_minus_baseline'] == {
        'costs': dict.fromkeys(study.COST_KEYS, 0), 'confirmed_and_solved': 0, 'mean_mse': 0.}
    for arm in study.ARMS:
        report = result['results'][arm]
        assert [task['role'] for task in report['tasks']] == ['incoming_demand', 'producer']
        for key in study.COST_KEYS:
            paid = sum(task['costs'][key] for task in report['tasks'])
            paid += report['screening_compute_cost'] if key == 'compute_cost' else 0
            assert report['task_sequence_costs'][key] == paid
            assert report['costs'][key] == paid + result['common_prefix_costs'][key]
            assert report['continuation_costs'][key] + result['partial_demand']['costs'][key] == paid
            assert report['costs'][key] <= report['resource_limits'][key]
            assert report['full_evaluation_costs'][key] == report['evaluation_costs'][key] + result['common_prefix_evaluation_costs'][key]
        assert 'first_remeasurement' in report['producer_diagnostics']
    summary = study.summarize([result], [], {'bootstrap_seed': 1, 'bootstrap_samples': 20})
    assert summary['paired_repair_minus_baseline']['compute_cost']['interval_95'] == [0, 0]
    assert summary['totals']['baseline']['stages']['incoming_demand']['tasks'] == 1
    assert 'actions' not in summary['totals']['baseline']['producer_diagnostics'][0]

"""Natural shared production, native resumes, and full-cost reporting."""

import importlib.util
import json
from pathlib import Path

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.memory import _called_definitions


def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_paid_revisit_study.py'
    spec = importlib.util.spec_from_file_location('paid_revisit_study', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_roster_preserves_budgets_and_distinguishes_paid_fork():
    study = runner()
    old = json.loads((study.BASELINE.parent / 'protocol.json').read_text())['cases']
    cases = study.cohort_protocol()['cases']
    assert len(cases) == 12
    assert not {case['seed'] for case in old} & {case['seed'] for case in cases}
    assert [case['seed'] for case in cases] == [67107001 + 100003*i for i in range(12)]
    for case in cases:
        assert case['arms'] == ['baseline', 'repair']
        for key in ('observation_budget', 'compute_budget', 'max_rounds', 'max_candidates'):
            assert case['research_config'][key] == old[0]['research_config'][key]
        assert case['measurement_budget'] == old[0]['measurement_budget']
        assert 'no candidate or source is selected by this runner' in case['intervention']


def test_called_sources_follow_nested_definitions_not_unused_library_entries():
    x = Program('feature', name='x')
    first = Definition('first', ('x',), Program('arg', name='x'), 'old_source')
    nested = Program('call', name='first', children=(Program('arg', name='y'),), definition=first)
    second = Definition('second', ('y',), Program('mul', children=(nested, nested)), 'derived_source')
    program = Program('call', name='second', children=(x,), definition=second)
    serialized = program.to_dict()
    serialized['definitions']['unused'] = Definition('unused', (), Program('const', value=0), 'unused_source').to_dict()
    expected = sorted({definition.source_concept_id for definition in _called_definitions(program)})
    assert runner()._called_sources(serialized) == expected == ['derived_source', 'old_source']


def test_real_identical_native_resumes_have_no_injected_candidate_or_extra_bill(tmp_path):
    study = runner()
    plan = study.protocol(67009001)
    plan['research_config'].update(observation_budget=96, compute_budget=12000, max_rounds=1)
    for task in [*plan['return_tasks'], *plan['competition_tasks'], *plan['tasks']]:
        task['task_index'] = 0
    plan['tasks'][1]['task_index'] = 1
    plan['evaluation'] = {'length': 8, 'trajectories': 1, 'seed_offset': 1000003}
    directory = tmp_path / 'case'
    study.prepare_case(plan, directory)
    common = json.loads((directory / 'common.json').read_text())
    producer = json.loads((directory / 'producer_training.json').read_text())
    fork = json.loads((directory / 'fork.json').read_text())
    baseline = json.loads((directory / 'baseline_training.json').read_text())
    assert fork['checkpoint'] == producer['state']
    assert all(state['evaluation'] == [] for state in common['prefix_states'].values())
    assert producer['state']['evaluation'] == baseline['state']['evaluation'] == []
    assert not list(directory.glob('*scored.json'))
    result = study.complete_case(directory, repair_root=study.BASELINE)
    repair = json.loads((directory / 'repair_training.json').read_text())
    assert repair['state']['evaluation'] == []
    for key in ('initial_costs', 'initial_concept_ids', 'initial_workflows', 'initial_tasks', 'initial_config'):
        assert baseline[key] == repair[key]
    assert result['paired_repair_minus_baseline'] == {
        'costs': dict.fromkeys(study.COST_KEYS, 0), 'confirmed_and_solved': 0,
        'incoming_mse': 0., 'mean_mse': 0.}
    for arm in study.ARMS:
        report = result['results'][arm]
        for key in study.COST_KEYS:
            paid = sum(task['costs'][key] for task in report['tasks'])
            paid += report['screening_compute_cost'] if key == 'compute_cost' else 0
            assert report['task_sequence_costs'][key] == paid
            assert report['costs'][key] == paid + result['common_prefix_costs'][key]
            assert report['incremental_costs'][key] + result['fork_costs'][key] == paid
            assert result['fork_remaining'][key] + result['fork_costs'][key] == report['task_sequence_resource_limits'][key]
            assert report['costs'][key] <= report['resource_limits'][key]
            assert report['full_evaluation_costs'][key] == report['evaluation_costs'][key] + result['common_prefix_evaluation_costs'][key]
    summary = study.summarize([result], [], {'bootstrap_seed': 1, 'bootstrap_samples': 20})
    assert summary['paired_repair_minus_baseline']['compute_cost']['interval_95'] == [0, 0]
    assert summary['totals']['baseline']['stages']['incoming_demand']['tasks'] == 1
